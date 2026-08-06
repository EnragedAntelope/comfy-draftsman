# Architecture

comfy-draftsman is a thin MCP wiring layer (`server.py`) over tested modules.
Ground truth is always the live ComfyUI instance's `/object_info`; the server
holds one lazily created client/session per process.

## Module map

```
src/comfy_draftsman/
├── server.py          # MCP tools/prompts - thin wiring only, no logic
├── config.py          # env-driven config (COMFYUI_URL, DRAFTSMAN_SESSION_DIR, ...)
├── session.py         # workflow_id -> Workflow store, persisted under ~/.comfy-draftsman
├── imaging.py         # preview downscaling / JPEG re-encode for inline images
├── graph/
│   ├── model.py       # Workflow/Node/Link graph; from_ui/to_ui (schema 0.4) + to_api
│   ├── widgets.py     # positional widgets_values <-> named values; dynamic combos
│   ├── subgraph.py    # schema-1.0 subgraph flattening (see below)
│   ├── validate.py    # live-instance validation + write-time value checks
│   ├── lint.py        # readability/wiring lint (advisory only)
│   ├── annotate.py    # organize_workflow: titles, groups, notes, knob highlights
│   ├── layout.py      # staged auto-layout
│   └── port.py        # cross-family model ports
├── comfy/
│   ├── client.py      # httpx client for ComfyUI REST endpoints
│   ├── catalog.py     # object_info search/summaries; safetensors metadata digest
│   ├── progress.py    # websocket ProgressTracker for non-blocking runs
│   └── registry.py    # Comfy Registry lookups (missing node packs)
└── knowledge/         # per-model-family tuning floor (YAML) + learned overlay
```

## Data flow

```
UI JSON (schema 0.4/1.0)
  └─ Workflow.from_ui ──► graph model ──► edit ops / organize / validate
                                            └─ to_ui  ──► save_workflow (userdata)
                                            └─ to_api ──► POST /prompt (run_workflow)
                                                 └─ subgraph.flatten() first when
                                                    instances are present
```

- **Validation gates:** `run_workflow` and `save_workflow` refuse on
  `validate()` errors unless `allow_invalid=True`; both refresh `object_info`
  first so combo checks see the current model files. `lint()` never blocks.
- **Write-time value checks:** `edit_workflow`'s `set_widget`/`add_node` ops
  reject invalid widget values (combo membership, ranges, types) immediately
  via `validate.check_widget_value`, with closest-match suggestions; a per-op
  `"force": true` skips the check.

## Subgraphs (schema 1.0 `definitions.subgraphs`)

A subgraph instance is a node whose `type` is the definition's uuid. The
frontend expands instances client-side at queue time; the backend never sees
them, so draftsman mirrors that expansion in `graph/subgraph.py`:

- Definition `inputs`/`outputs` are boundary slots; inner links use pseudo node
  ids **-10** (input boundary) / **-20** (output boundary), with the
  boundary-side slot index pointing into those lists.
- The instance node exposes only *some* boundary inputs as sockets — match
  instance input slots to definition inputs **by name**, never by position.
- Widget promotion: `instance.properties.proxyWidgets` is a list of
  `[innerNodeId, widgetName]`; a non-empty instance `widgets_values` zips
  positionally over it and overrides the inner nodes' own values. Bundled
  templates ship it empty (inner defaults hold).
- Flattened node ids follow the frontend's `instanceId:innerId` convention in
  provenance/reporting; nesting recurses (depth-capped).
- `validate()` flattens first, so inner nodes get full checks with subgraph
  provenance on each finding. `edit_workflow` ops deliberately do **not**
  reach inside definitions — rebuild flat to modify internals.

## Gotchas (hard-won; do not relearn)

- **Dynamic nodes** (text concatenators, switches...) declare dozens of
  optional widgets in `object_info` but the frontend serializes only the ones
  in use — a widgets_values shortfall there is normal. Never pad with `None`.
- **The frontend runs `.replace()` over every string widget at queue time**, so
  a `null` widget value crashes the editor even on connected/optional slots.
- **Seed control widgets are a name heuristic:** the frontend appends a
  `control_after_generate` widget after any INT literally named `seed`/
  `noise_seed`, even when the schema has no flag (`widgets.has_control_slot`
  mirrors this).
- **Never default any path off `Path.cwd()`** — MCP hosts launch servers from
  arbitrary/system directories. Session state lives under
  `~/.comfy-draftsman` (`DRAFTSMAN_SESSION_DIR` overrides).
- **object_info is multi-megabyte** — never return it (or a full combo list, or
  a raw safetensors header) to the model. Everything recurring must be capped
  or digested; full detail belongs to explicitly-requested tools
  (inspect/export/view).
- **Token discipline:** `edit_workflow` returns a compact delta by default
  (`summary=true` opts into the full graph); summaries clip long widget
  strings; guidance sentences are stated once per result, not per item.
- **Every list a tool returns must be bounded, and per-item repetition is a
  bug.** This is the rule the 0.9.0 audit found broken in five places, so it is
  written down with its enforcement points:
  - *Findings* go through `server._cap_findings` (severity-sorted, errors never
    dropped) or `server._cap_lint` (advisory, straight cap). **Every** tool that
    returns findings uses one — `validate_workflow`, `diagnose_workflow`,
    `organize_workflow`, `port_workflow`, `run_workflow`, `save_workflow`.
  - *File and choice lists* are capped with a true `count` plus a hint naming
    the narrowing parameter: `catalog._apply_choices` (combos, 24),
    `list_models` (`_MODEL_FILES_CAP` 60, hint `search=`) and `list_templates`
    (`_TEMPLATES_CAP` 40, `_TEMPLATE_DESC_CAP` 110, hint `search=`). They are the
    same data — an instance with 400 LoRAs must not return 400 names just because
    the caller asked a different tool. **A cap without a count is a correctness
    bug, not just a token one:** `list_templates` returned a bare `list[:60]` out
    of ~450 templates, so a caller who found no match reasonably concluded the
    catalog had none. Search runs over the *full* record, not the clipped entry —
    otherwise the clip makes real matches unfindable.
  - *Folded schemas* are capped: `search_nodes(detail=True)` fills only the top
    `_DETAIL_SCHEMA_CAP` hits, since a schema is ~300-700 tokens each.
  - *A condition affecting N nodes is ONE finding* naming a few ids and counting
    the rest, with the full list in a `node_ids` field — never N findings. See
    `lint._overlap_findings` and validate's `node-disabled`. A pairwise report is
    quadratic: 20 co-located nodes once produced 190 findings identical in
    substance, and 52 produced 1,326 (~30k tokens in one response).
  - `tests/test_round18_tokens.py` asserts ceilings on all of the above, so a
    regression fails in CI instead of in a user's context window.
- **`ok` means "no errors", never "no findings".** `validate_workflow` and
  `diagnose_workflow` both gate on `level == "error"`. Using `not findings` (as
  diagnose once did) makes an informational note — a disabled node, a subgraph
  instance — report a healthy workflow as broken.
- **Subgraph fixtures must be realistic:** minimal hand-built defs without
  boundary links or inner `inputs` arrays behave differently from real
  exports — `tests/fixtures/subgraph_real_template.json` is the reference.
- **Subgraph edit ops** — editing subgraph definitions parses them into
  Workflow objects internally; nested definitions (depth > 1) raise
  NotImplementedError.
- **proxyWidgets** — removing an inner node may invalidate instance
  proxyWidgets overrides — a warning is returned in the op result.
- **FastMCP image returns must be a bare `Image` or a list element** — a dict
  that *contains* an `Image` (e.g. `{"image": Image(...)}`) is repr'd into text
  and never renders. `view_output` and `run_workflow`'s inline preview both
  return the **list** form `[{"meta": {...}}, Image(...)]` (a sibling meta dict
  carries dimensions/filename for text-only models). Never wrap an `Image` in a
  dict.
- **Partial-accept is not success.** ComfyUI can return **HTTP 200 with
  `node_errors`** (not 400): it queues the prompt, executes the still-valid
  subgraph, and drops the rejected nodes' branches. `queue_prompt` only *raises*
  on 400, so those node_errors ride back inside the 200 body;
  `run_and_wait` threads them onto the result and `run_workflow` downgrades
  `status` to `"partial"` with a loud `warning` (and `wait=False` surfaces them
  on the `queued` response). A run that touched only text-utility nodes in 51 ms
  otherwise looks like a clean success with mysteriously-empty outputs.
- **Big-int seeds survive save, not the host transport.** Python `json` keeps
  arbitrary-precision ints, so `save_workflow`/`export` write seeds like
  `17190566679778241971` exactly. If a seed reads back rounded
  (`...43000`) in a tool *response*, that is the MCP host's JS-side
  `JSON.parse` coercing >2^53 to a double — display-only, not in the saved
  file. Draftsman does not (and cannot) fix the host's number handling; never
  "correct" a seed to match a rounded readback.
- **Frontend-only behaviors are mirrored at submit (`to_api`/`run_workflow`), not
  in the saved graph.** The raw `/prompt` backend never runs the JS the browser
  does, so draftsman replays it for headless parity:
  - *Custom JS-widget inputs:* a pack can declare an input whose type is a bespoke
    string (`AUTOCOMPLETE_TEXT_LORAS`, `ZIPN_STYLE_GALLERY_BUTTON`) that its own
    frontend renders as a widget, not a socket. Schema alone can't tell it from a
    connection type, so it's recognized **per-instance**: an input the node did
    *not* serialize in its `inputs` socket array can only be a widget
    (`widgets._is_custom_widget`, gated on `socket_names`). Schema-only paths
    (fresh-node defaults, `add_node`) stay conservative — never infer a custom
    widget without instance context. When such an input is instead exposed as a
    widget-backed slot (carries a `widget` marker) and is unconnected, its value
    is treated as pack-specific JS state the raw API can't replay — `validate`
    blocks it with a `js-widget-input` error (with the remediation) rather than
    silently no-opping the branch (validate.py:388). This block is currently
    value-agnostic (it fires whether the stored value is a dict/list or a plain
    scalar); making it scalar-aware was deliberately not done because the live
    server rejected hand-serialized scalar values anyway — see the OPEN TODO. A
    generic tool cannot replay a pack's client-side JS; the honest outcome is a
    loud, actionable stop.
  - *`%date:FORMAT%` filename tokens:* substituted in `to_api` only (the saved UI
    doc keeps the literal token for the browser). `.NET`-style tokens, longest
    first (`yyyy` before `yy`). See `model._substitute_filename_tokens`.
  - *Seed `control_after_generate`:* `run_workflow(roll_seeds=True)` re-rolls
    randomize/increment/decrement seeds before submit and **persists** the new
    value (so `inspect` reflects the run and increment advances). The API itself
    never re-rolls — a fixed seed repeats forever otherwise.
- **Combo-membership severity is confidence-gated.** A value absent from the
  `/object_info` snapshot blocks (error) only for on-disk file listings or core
  nodes (`python_module` not under `custom_nodes`); third-party nodes that
  repopulate combos client-side (wildcard/LoRA/style pickers) get a non-blocking
  warning. Keeps the "is this model installed" check strict without flooding on
  client-populated pickers. `validate_workflow`/`diagnose_workflow` also cap
  returned findings (errors always kept) for token discipline.
- **Display nodes are layout companions, not outputs.** `organize_workflow`
  treats Show Text-style and `PreviewImage`-style nodes as *companions*: they
  inherit the stage of the node they display and are glued directly beneath it
  (`annotate._companion_sources` + `apply_staged_layout(companion_of=...)`).
  Grouping them into a distant Output band made readers trace wires across the
  canvas to pair previews with samplers — the original layout complaint.
  SaveImage-style disk writers are NOT companions. An unwired display node
  falls back to its classified stage. Empty-latent canvas nodes classify as
  `inputs` (they're the resolution knob), so all user-tweakable things sit on
  the left edge.
- **Front-of-queue is additive, never destructive.** `POST /prompt` accepts
  `"front": true` — the prompt runs next after the current job; pending jobs
  stay queued. `run_workflow(front=None)` (default) refuses to queue when ≥2
  prompts are pending and returns `queue_busy` so the USER decides; it never
  clears/interrupts anything. The check is best-effort (an unreachable
  `/queue` never blocks a run) and happens before seeds are rolled, so a
  gated run doesn't advance increment seeds.
- **Muted/bypassed nodes are not validated, by design.** `to_api` drops mode-2
  (mute) and mode-4 (bypass) nodes, so their own widget values and unconnected
  inputs cannot break a run — and muting a branch is *the* standard way to
  disable it. `validate` reports each disabled node once as `info`
  (`node-disabled`) and checks nothing else about it; validating them anyway
  produced blocking errors that refused `run_workflow`/`save_workflow` for a
  graph whose prompt document didn't contain those nodes at all. What a disabled
  node does to its *consumers* is still checked, on the active consumer:
  `muted-input-source` (a mute never runs, leaving a dangling reference) and
  `dead-input-source` (bypass is a passthrough, so a bypassed node with its own
  input unconnected forwards a hole and `to_api` silently drops the consumer's
  input).
- **The muted/dead-source check must cover OPTIONAL inputs too, not just
  required ones.** Round 21 added `muted-input-source`/`dead-input-source` but
  only walked `schema["input"]["required"]` — a node's required/optional split
  governs whether an input must be wired, not whether a *wired* source is real,
  and `to_api` drops a muted node's own entry from the API document regardless
  of which section declared the consumer's input. A live session muted a node
  feeding RES4LYF's `ClownsharKSampler_Beta.options_group` (`required: false`
  on a live instance) and got a clean `validate()` result, then a raw
  ComfyUI-side `KeyError` at `/prompt` time instead of a normal draftsman
  rejection. `_connected_source_finding` (validate.py) is now shared by a
  parallel walk over `optional` specs — skipping the required-only checks
  (`unconnected-input`, `js-widget-input`) that don't apply when a slot is
  legitimately allowed to be empty. Autogrow markers needed a second fix on top:
  the marker name is never a real socket (see the autogrow gotcha below), so
  neither the required nor optional by-name lookup ever reaches the actual
  synthesized sockets — `_autogrow_source_findings` walks them directly and is
  called from both the required and optional autogrow branches.
- **`socket_names` is required on every path that REBUILDS an existing node's
  `widgets_values`.** A pack's custom JS-widget input is only recognizable
  per-instance (`widgets._is_custom_widget`), so a slot walk without the node's
  declared sockets misses that widget entirely — and `named_to_widgets` then
  writes a *shorter* array, destroying the custom value and shifting every later
  one up a slot. `set_widget`, `get_widget`, `check_widget_value`, `to_api`,
  `apply_seed_control`, `_validate_nodes` and subgraph widget promotion all pass
  it. Schema-only contexts (fresh-node defaults, `add_node`) deliberately do
  **not** — never infer a custom widget without instance context.
- **Relocation covers every output kind; the inline preview does not.**
  `save_output` / `run_workflow(save_dir=...)` relocate images, gifs, videos and
  audio alike (a video render is exactly as stuck inside ComfyUI's output tree),
  but the inline thumbnail stays image-only because `downscale_image` needs a
  decodable still. Relocation needs *finished* files, so `save_dir` does not
  apply to `wait=False`; that returns `save_dir_ignored` naming the follow-up
  `save_output(prompt_id=...)` rather than silently dropping the request.
- **Display/output nodes overflow `widgets_values` on purpose.** ShowText,
  rgthree "Display Any", and preview nodes stash the text/data they display into
  `widgets_values` beyond their declared schema widgets. A count *overflow* on a
  node whose schema sets `output_node` is expected and suppressed (not
  `widget-count-drift`); a shortfall, or any mismatch on a non-output node, still
  reports.
- **ComfyUI's V3 io system declares five META types, and they are not values.**
  `comfy_api/latest/_io.py` defines `COMFY_MATCHTYPE_V3`, `COMFY_AUTOGROW_V3`,
  `COMFY_DYNAMICSLOT_V3`, `COMFY_DYNAMICCOMBO_V3` and `COMFY_MULTITYPED_V3`.
  Each is a schema marker the frontend expands into something else, so **each
  breaks a different draftsman heuristic that assumes a type string describes a
  value**. Two were found broken at once in round 20, so treat a new `COMFY_*_V3`
  as guilty until checked:
  - *MatchType is a wildcard.* `comfy_execution/validation.py` short-circuits
    `validate_node_input` to True whenever either end is a MatchType
    ("validation for this is handled by the frontend"), because such a node
    adopts whatever it is wired to. Core nodes use it — `ComfySwitchNode` is
    MatchType in *and* out. Treating it as concrete made `link-type-mismatch`
    fire on every wire touching a switch, and since that check gates
    `run_workflow`/`save_workflow`, it made any workflow containing one
    unrunnable and unsavable — including every bundled Krea-2 template.
    `model.MATCH_TYPE` + `types_compatible`.
  - *None of them is a widget.* A node routinely does NOT serialize a meta input
    into its `inputs` socket array (an autogrow node emits `value0..valueN`
    instead of its `values` marker; an unconnected MatchType slot is just
    absent) — which is exactly the shape `widgets._is_custom_widget` reads as "a
    pack's JS-rendered widget". Counting one invents a slot and shifts every
    later `widgets_values` entry. `widgets.is_v3_meta_type` matches the
    `COMFY_*_V3` shape rather than a fixed set, so a new core meta type can't
    silently reintroduce it.
  - `COMFY_MULTITYPED_V3` never reaches a type comparison — a MultiType input
    serializes as its comma-joined member types, which the union branch of
    `types_compatible` already handles. `COMFY_DYNAMICCOMBO_V3` **is** a widget
    and is handled separately (`widgets.is_dynamic_combo`); it is excluded from
    `_is_custom_widget` by the earlier `is_widget_input` check, not by the meta
    carve-out.
- **ComfyUI declares which inputs are widgets; believe it over the type name.**
  V3's `WidgetInput.as_dict` serializes `socketless` (the frontend draws a widget
  and never a socket) and `widgetType` (which widget renders a bespoke or union
  io type). `widgets.is_widget_input` honours both, ahead of any type-name
  reasoning — but **after** `forceInput`, which is the node author's explicit
  "draw a socket" and appears together with `socketless` on 7 inputs (a type
  whose class defaults to socketless, overridden per node). Note `socketless`
  is serialized as a real `false` too, so test truthiness, not key presence.
  Before round 21 these were ignored and 26 classes on a stock instance had a
  declared widget treated as a required connection socket: `ColorToRGBInt`
  (whose only parameter is a socketless COLOR) got a phantom socket, no widget
  value, and a *blocking* `unconnected-input` for a graph that was fine, and
  `TextOverlay` dropped `color` and shifted every later widget up a slot.
  `widgets.widget_kind` uses `widgetType` for value checks, since a union like
  `"FLOAT,INT"` names no single checkable kind.
- **An autogrow input is a growing socket LIST, and its API key is dotted.**
  `COMFY_AUTOGROW_V3` (66 inputs on a stock instance) declares only a *marker*
  plus a `template`; the real sockets are synthesized from `prefix`+index
  (`image0`…`image49`) or an explicit `names` list, with the first `min`
  mandatory. Three separate traps, all hit in round 21:
  - *The marker is not a socket.* Reported as an unconnected required input, it
    blocked 56 required markers on a stock instance. `validate` now checks the
    real requirement instead (`autogrow-underfilled`), and `lint` mirrors the
    exemption — lint contradicting validate is noise, and `save_workflow` nags
    about an unclean lint.
  - *The prefix is not the marker name* (`images` → `image0`). It must be read
    from the template, never derived.
  - **The prompt key is `{marker}.{slot}` (`images.image0`), not the bare canvas
    name.** Confirmed twice — from `parse_class_inputs`/`finalize_prefix`, which
    prefixes the marker id onto every expanded name, and independently from the
    frontend bundle, which builds each slot as
    ``{name: `${marker}.${slot}`, display_name: slot}``. Emitting the bare name
    does **not** error: the backend simply never matches it and the node runs
    with that input silently missing. `to_api` normalizes
    (`widgets.autogrow_api_key`), `connect` accepts either spelling, and an
    imported socket is never renamed — canonicalizing one would silently rewrite
    the user's file. **Gaps are legal**: the backend collects whichever names the
    prompt carries, so `image0`+`image2` runs as written and nothing needs
    renumbering.
  - `/object_info` names only the marker, so `get_node_info` expands the slot
    names (capped, `catalog._AUTOGROW_NAMES_SHOWN`) — without that they are
    undiscoverable and `connect` has nothing to aim at.
- **A remediation string must name an op that exists.** Every finding on a node
  inside a subgraph used to end "edit_workflow can't reach inside; rebuild flat
  to change it" — written before the `*_in_definition` ops landed and never
  revisited. `inspect_workflow`'s `subgraph_note` said the same. A live session
  read it, believed it, and hand-rebuilt a 14-node graph to change one wrong
  model path that `set_widget_in_definition` fixes in one call. `flatten`'s
  provenance now carries `definition` + `inner_id` (the exact arguments those
  ops take) and an `editable` flag that is honest about the limit —
  `subgraph_as_workflow` refuses both a nested node *and* any definition that
  itself contains an instance, so `editable` requires depth 1 **and** no nested
  instance. Only a genuinely unreachable node is told to rebuild flat. The ids
  ride on the finding as structured fields; the how-to sentence is stated once
  per result (`server._subgraph_edit_hint`), never per finding.
- **A subgraph definition's boundary inputs are not all connectable.** The
  instance node exposes only some as real sockets and `connect` addresses
  instance sockets, so `inspect_workflow` marks the rest `name (internal)`
  (`server._subgraph_summary(sg, wf)`; omit `wf` and nothing is claimed). The
  bundled Z-Image template declares six and exposes one — listing all six
  unqualified sent a session chasing a `value` socket that does not exist,
  while `edit_workflow`'s own error message listed the real ones. Two tools
  disagreeing about what a node has is worse than either being terse.
- **A lint that fires on correct work is worse than no lint.**
  `no-prompt-preview` walked only the encoder's upstream chain, so it missed a
  Show Text *tapped off* the generator's output (generator → display alongside
  generator → encoder). That is the more common hand-wired shape and shows the
  identical string — arguably better, since the display isn't in the path it
  reports on. `lint._has_text_display` accepts both. Callers who see a rule
  fire on work they know is right stop reading the rule.
- **Socket types are checked; COMBO is not a wildcard.** `model.types_compatible`
  is the single source of truth (case-insensitive, `*`/`ANY`/empty wildcards,
  comma-joined unions intersect) and is used by both `connect` (refuses a
  mismatch unless `"force": true`) and `validate`'s `link-type-mismatch` (error).
  Before round 19, `connect` treated `COMBO` as a second wildcard alongside `*` -
  a STRING wired into a converted combo widget passed local validation and was
  then rejected by ComfyUI's own executor at queue time
  (`return_type_mismatch`), which queues the prompt anyway and runs only the
  rest of the graph. `validate` did not check link types at all, so the same
  mistake arriving via `import_workflow` was invisible until the failed run.
- **`PrimitiveNode` and `Reroute` are real, authorable virtual nodes**, not
  merely tolerated ones. Both were already in `VIRTUAL_TYPES` (so `to_api`
  already inlined a primitive's value and traced through a reroute), but
  `add_node`'s installed-class gate only special-cased `NOTE_TYPES`, so
  `add_node(class_type="PrimitiveNode")` failed with "not installed on this
  instance" even though the class is a standard ComfyUI frontend concept. This
  blocked the correct idiom for "a value that adapts to and can cycle whatever
  it's wired to" (a dropdown, a checkpoint, a LoRA), forcing a same-type-only
  workaround.
  - **A primitive is typeless until connected** (`outputs[0].type == "*"`) and
    adopts its target's type on `connect` (`Workflow._mirror_primitive`,
    mirroring the frontend's `widgetInputs.ts #onFirstConnection`/`setType`,
    which is browser JS a headless author has no other way to replay). Verified
    against a real export: `outputs[0]` carries `{"widget": {"name": "steps"}}`,
    `title` becomes the mirrored widget's name, `widgets_values` is
    `[value, "fixed"]` for number/combo or `[value]` for STRING (the frontend's
    `addValueControlWidget` only fires for number and combo types -
    `widgets.primitive_takes_control`). Only an *unresolved* primitive (still
    `type == "*"`) adopts - one already bound keeps its type, so a second
    connect never silently retypes an in-use value.
  - **`set_widget` addresses a primitive by the mirrored widget's real name or
    the alias `"value"`**, plus `"control_after_generate"` - its slot names come
    from the GRAPH (what it mirrors), not from `object_info`, since a primitive
    has no schema of its own.
  - **Nothing else validates a primitive's value** - its consumer's widget slot
    is *connected*, so the consumer's own widget check is skipped (same as any
    other wired input). `validate._primitive_findings` is the one place that
    checks it, against every widget it drives (`Workflow.primitive_targets`,
    which sees forward through Reroutes) via the same `check_widget_value` used
    everywhere else - so the confidence gate (core enum vs. client-populated
    third-party combo) still applies. An unbound primitive (drives nothing) is a
    non-blocking `primitive-unbound` warning, not an error: `to_api` drops
    virtual nodes, so it cannot break a run, only waste a value.
  - **`apply_seed_control` rolls a primitive's control mode too**
    (`Workflow.roll_primitive_control`) - this is the only way a headless
    caller can cycle a COMBO across runs, since `control_after_generate` is
    applied by browser JS and the raw `/prompt` API never sees it. Faithful to
    the frontend including the parts that feel wrong: an index walk **clamps**
    at the ends of the option list (does not wrap), and randomize is uniform
    over the *option index*, not a value-weighted pick.
  - **Layout/annotate treat a primitive as a tweakable**: `classify` puts it in
    the `inputs` band (same reasoning as the empty-latent canvas node - it
    exists to be hand-set) and `_paint_knobs` colors it green. A `Reroute` is a
    display companion of its source (like Show Text/PreviewImage), not left to
    land wherever its rank happens to fall - it exists purely to shorten a wire,
    so stranding it elsewhere lengthens the very thing it's for.
  - **`_summary` only hides `NOTE_TYPES` links now**, not all of
    `VIRTUAL_TYPES`. Hiding every link touching a primitive or reroute made an
    authored primitive look unconnected in the very view used to verify it built
    correctly; a Note genuinely has no sockets, so it contributes nothing to
    hide.
  - `layout.estimate_size` has no schema for a virtual class and would flatten
    a 75×26 Reroute or a 300×160 multiline-STRING primitive into the generic
    "unknown class" box - `apply_layout`/`apply_staged_layout` skip it for
    anything in `VIRTUAL_TYPES` and keep the size construction already gave it.
- **`OutputSlot` round-trips a primitive's widget marker.** A real
  `PrimitiveNode` output serializes as
  `{"name": "INT", "type": "INT", "widget": {"name": "steps"}, "links": [...]}`
  (confirmed in the bundled `tests/fixtures/sdxl_simple_example.json`, which
  itself uses primitives to drive `steps`/`end_at_step`/both prompts). Before
  round 19 `OutputSlot` had no field for it, so `from_ui` silently dropped the
  marker on every workflow using primitives - including that bundled fixture -
  and the round-tripped `to_ui` primitive no longer named what it mirrored.
- **A custom output node's non-file return values were invisible.**
  `client._collect_outputs` (feeding both `run_workflow`'s `outputs` and
  `save_output`'s relocation) only ever harvested the four FILE keys
  (`images`/`gifs`/`videos`/`audio`). A node like `NH_SaveImagePath` that also
  returns `filenames`/`path`/`saved_count`, or a ShowText-style node returning
  `text`, had those values silently dropped - the caller had to go find the
  real save directory by reading the running ComfyUI process's command line.
  `client._collect_data_outputs` now harvests every OTHER key per output node
  into `run_workflow`/`get_run_status`'s `data_outputs` (omitted entirely when
  empty), value-clipped and total-budgeted the same way findings are capped
  elsewhere in this server - `FILE_OUTPUT_KEYS` stays the file-only source for
  relocation, untouched.
- **Family detection is anchored to the diffusion-model loader, and NEVER a
  LoRA.** `knowledge._model_refs` used to return every string widget matching
  a model file extension with no notion of which one names the diffusion
  model - a placeholder LoRA named `LTX23_The_Cook_….safetensors` on a
  MiniMax H3 graph outscored everything, and `organize_workflow` stamped LTX
  Video's CFG guidance onto a graph with no CFG node at all (a prior session
  hit the identical bug with an SDXL LoRA, explaining an old "the notes talk
  about SDXL checkpoints" complaint). `_model_refs` now tags each reference
  `primary` (`ckpt_name`/`unet_name`/…), `aux` (any widget name matching
  `lora|vae|clip|control_net|…` - can never name the diffusion model), or
  `other`. `detect_family_detail` scores `primary` refs first, only falling
  back to `other` if none match, and **never** considers `aux`. Variant
  matching (turbo/lightning/…) uses `primary_model_filenames` for the same
  reason - a LoRA named `…turbo…` must not select an unrelated base model's
  turbo notes. `model_filenames` (all roles) is kept for search/matching,
  where a LoRA name is a legitimate signal - only *detection* needs the
  narrower view.
- **`apply_staged_layout` does not position Note/MarkdownNote nodes - if
  `annotate` doesn't handle that itself, they land on top of the new
  layout.** A real session's 7 hand-written `MarkdownNote` nodes stayed at
  their old position while everything around them moved, producing 49
  overlapping pairs across 11 nodes - which `organize_workflow`'s own
  returned `lint` block then reported, meaning the tool shipped a layout it
  had already diagnosed as broken. `annotate()` is now four explicit phases:
  (1) lay out via `apply_staged_layout`, (2) `_park_foreign_notes` relocates
  ONLY human-authored notes that actually collide with something (one sitting
  in clear space was placed there on purpose and must not move) into a column
  left of the graph, (3) place generated notes above each band with `_wrap`
  columns sized to the note's REAL width (not a fixed 58 - the previous
  mismatch between the wrap width and the height *estimate* was a secondary
  contributor to the same overlap), (4) `layout.resolve_overlaps` sweeps
  anything still colliding (nodes only ever move down, never sideways, so it
  terminates and never undoes a band's x-position), and group bounds are
  computed LAST, from these final positions, via the shared
  `Workflow.group_from_nodes`/`group_bounding_for` helper `edit_workflow`'s
  `add_group`/`set_group` ops also call - a hand-made group and a generated
  one end up geometrically identical. If an overlap still survives all four
  phases, `organize_workflow` says so in a top-level `warning` instead of
  silently shipping it.
- **A note's `Safe ranges:` line must check the graph has the knob before
  claiming a range for it.** The line used to render unconditionally from
  whatever a family's guidance carried; a learned overlay whose `cfg` block
  is prose-only (H3: `{"note": "No CFG - guidance-distilled"}`, no numeric
  `min`/`max`) rendered the literal string `Safe ranges: CFG None-None, steps
  None-None.` - and the same code would assert a CFG range on any
  guidance-distilled chain (`BasicGuider`, `SamplerCustomAdvanced`) that has
  no `cfg` widget at all. `annotate._graph_knobs` collects the widget/input
  names actually present on a stage's nodes; each range clause now requires
  BOTH a real numeric `min`/`max` AND membership in that set, and a prose-only
  `cfg.note` is emitted as prose instead of a fabricated range. The output
  note is read from the output nodes' own declared input types
  (`_output_medium`: `IMAGE`→images, `VIDEO`→video, `AUDIO`→audio, mixed→the
  honest generic "files") instead of the old hardcoded "Finished images land
  here" on a video/audio workflow.
- **`force: true` on `connect` can create a socket `/object_info` never
  declared - a DIFFERENT case from the `js-widget-input` TODO below, and the
  two must not be conflated.** Some packs (rgthree's "Any Switch", dynamic
  collectors) build their inputs in frontend JS, so the schema declares none
  at all; `Workflow.connect` used to raise "has no input" BEFORE the `force`
  check ever ran, so there was no override. The `in_slot is None and force`
  branch (`model.py`) now creates the slot typed as the origin output's own
  type (there is no schema to check against) and links it - `to_api` emits it
  as an ordinary link and ComfyUI accepts it normally. This differs from
  `js-widget-input`: that block is a *declared widget-backed slot* whose
  VALUE is unreplayable pack-specific frontend state (LoraManager's active-tag
  resolution); this is a *missing socket* carrying an ordinary link. Confusing
  the two would either wrongly loosen the honest `js-widget-input` stop or
  wrongly re-block a working rgthree wire.
- **A curated `sources:`/`multiple_of:` value is a claim about a specific
  model, and a wrong one is worse than none.** Two new family-YAML fields,
  both opt-in and both scoped so a caller who threads no state gets the old
  behavior unchanged: `sources` (list of `{match, what, url}`, surfaced by
  `knowledge.matching_sources` into the Models note's "Where to get these
  files" list - checked reachable by hand before being committed, never
  synthesized; `record_learning`'s docstring documents the key so an agent
  can add one after verifying); `multiple_of` (the VAE/patchify structural
  alignment requirement, checked by `lint._resolution_alignment_findings`
  against a canvas node's width/height, naming the nearest legal value -
  never auto-rewritten or forced via an injected math node, since an odd size
  may be deliberate). `lint()`'s new `learned_dir` parameter **defaults to
  `None`, which skips the resolution check entirely** - this is a deliberate
  opt-out, not "no learned overlay so use the floor" - so no existing caller
  that doesn't thread `_config().learned_dir` through starts seeing the new
  finding just because a family YAML gains `multiple_of`. Only the three
  server.py call sites (`organize_workflow`, `lint_workflow`, `save_workflow`)
  pass the real configured path. Seeded `multiple_of` only where confirmed
  against a live schema step or a vendor's own docs (see the round-23
  changelog entry for which families and how); left unset for families where
  it could not be confirmed rather than guessed.

## Remaining TODOs

Open:

- **[OPEN] `COMFY_DYNAMICSLOT_V3` is classified but never exercised.** It is
  excluded from widget inference like the other meta types
  (`widgets.is_v3_meta_type`), but **zero** inputs on a stock ComfyUI 0.29
  instance with ~3,700 classes declare one, so nothing here has been tested
  against reality — unlike `COMFY_AUTOGROW_V3` (66 inputs) and
  `COMFY_MATCHTYPE_V3` (36), which round 20/21 could verify. Its shape
  (`slotType` + a nested `inputs` dict, expanded under a dotted prefix) is close
  enough to autogrow that `graph/widgets.py`'s autogrow helpers are the model to
  copy. Do not implement it speculatively — wait for a pack that uses it, then
  verify against a real serialized node the way round 21 did.
- **[OPEN] Widget-backed custom-JS inputs stay a loud stop, by design.** Packs
  like LoraManager (`text` / `AUTOCOMPLETE_TEXT_LORAS`) and StyleStringInjector2
  (`gallery` / `ZIPN_STYLE_GALLERY_BUTTON`) expose an input as a widget-backed
  slot whose value is *pack-specific frontend JS state* (e.g. LoraManager's
  effective lora text is resolved client-side from the `active:true` entries at
  queue time). `validate` blocks these with `js-widget-input` (see Gotchas) and
  `to_api` can't emit them. A generic "just emit the scalar" fix was
  **considered and rejected**: on the live instance the server *rejected* a
  hand-serialized `text` until it was rebuilt from the active entries, so
  emitting a raw value would trade a loud, honest error for a silently-wrong
  render. A real fix would need per-pack resolution logic (out of scope for a
  generic tool); until then the honest stop stands. Workaround for a caller that
  must run such a graph headlessly: connect the input to a plain-STRING source,
  swap in the pack's plain-STRING node variant, or run it from the ComfyUI
  frontend. (Code note: `validate` blocks on any widget-backed unconnected
  custom input regardless of value type; the surrounding docs' hint at
  value-awareness is aspirational, not implemented — see above for why.) Note
  this is the *widget-backed slot* case only; a custom input the node did not
  serialize as a socket at all is handled and now survives the write path too —
  see the `socket_names` gotcha above.
  - **Round 21 narrowed this precisely, and the boundary is now evidence-backed
    rather than assumed.** The blocked types carry **no schema flags at all** on
    a live instance — `AUTOCOMPLETE_TEXT_LORAS` (4 nodes), `RANDOMIZER_CONFIG`,
    `LORA_POOL_CONFIG`, `LORAS` are all bare `["TYPE", {}]`. Meanwhile the
    LoraManager nodes that *are* ordinary widgets declare
    `widgetType: AUTOCOMPLETE_TEXT_PROMPT` on a `AUTOCOMPLETE_TEXT_PROMPT,STRING`
    union, and round 21's `socketless`/`widgetType` rule picks exactly those up
    while leaving the four JS-state ones blocked. So the flags double as the
    discriminator: **flagged ⇒ the value is a plain widget value the API can
    carry; unflagged bespoke type ⇒ pack JS state.** Confirmed over 60 of the
    user's real saved workflows — the surviving `js-widget-input` blocks are all
    and only the unflagged ones. If a pack ever flags a genuinely JS-resolved
    input, that rule breaks and this becomes a real bug; there is no test that
    can catch it locally, so it is written here.
- **[OPEN] `multiple_of` unset for wan, qwen_image, krea2.** Round 23
  investigated all three: LTX's 32 came from Lightricks' own docs, and
  sd15/sdxl/sd35/flux came straight from their empty-latent node's own live
  `/object_info` step (8 or 16) - but WAN's ComfyUI tutorial docs don't state
  a divisibility requirement, and neither qwen_image nor krea2 have a
  dedicated empty-latent node in the trimmed test fixture to check against a
  live schema. Chroma got `16` anyway on strong architectural inference
  (explicitly FLUX-schnell-derived, reuses FLUX's own VAE file) but is flagged
  as inferred, not schema-verified, in its own YAML comment. Verify each
  against a live instance's `/object_info` (whichever Empty*LatentImage/Video
  node the family's template uses) or the model's own release notes before
  adding - do not guess from general video-DiT conventions.

Recently closed:

- **[DONE, round 23] Family detection anchored to the diffusion model;
  organize_workflow overlap fix; reader-priority reorganization; layout/group
  edit ops; force-socket creation; knob cards; curated sources; resolution
  alignment.** From a live bug report building a MiniMax H3 R2V+Voice
  workflow (`organize_workflow` claimed the graph was LTX Video from a
  placeholder LoRA's filename, and shipped a layout with 49 overlapping node
  pairs it had already diagnosed as broken in its own returned lint) plus a
  separate design review of the default organization itself. See the Gotchas
  above for the mechanics of each; summary:
  - `detect_family` now scores only the diffusion-model loader widget
    (primary refs), never a LoRA/VAE/CLIP filename (aux refs) - see the
    family-detection gotcha above.
  - `annotate()` restructured into four ordered phases (layout → place notes
    → resolve overlaps → group bounds from final positions) with a new
    `layout.resolve_overlaps` sweep and a park-only-if-colliding pass for
    human-authored notes - `organize_workflow` never again ships a layout its
    own lint has flagged as broken; it says so in a `warning` if one survives.
  - `_note_text`'s `Safe ranges:` line and output-medium line are now graph-
    aware (`_graph_knobs`, `_output_medium`) instead of asserting whatever a
    family's guidance carried regardless of what's actually on the canvas.
  - `edit_workflow` gained `set_pos`/`add_group`/`set_group`/`remove_group` -
    the reported session's #1 pain point was no way to fix layout without
    `export_workflow_json` → hand-edit → re-`import_workflow`, the single
    largest token cost of that session. Docstrings were compressed to hold
    the tool-schema token ceiling despite the four new ops.
  - `Workflow.connect(force=True)` can now create a socket `/object_info`
    never declared (rgthree's Any Switch and similar frontend-JS-built
    inputs) - distinct from the `js-widget-input` TODO below; see the gotcha.
  - Default organization reorganized from six pipeline-order bands into seven
    reader-priority bands (Inputs, Prompt Building, Models & LoRAs,
    Conditioning, Sampling, Post-Processing, Output): an unwired encoder
    prompt box (the classic hand-typed box) now lives in the leftmost Inputs
    band instead of a middle "Conditioning" band, while a multi-step prompt-
    building pipeline (wildcard bank → concatenator → LLM step) stays
    together in its own band even when its root producer is itself unwired.
  - New `graph/knobs.py` + `knowledge/techniques.yaml`: every note now
    renders a markdown table of that band's editable knobs (current value,
    live-schema range/choices, a tradeoff sentence) plus any matching
    technique's tradeoff (EasyCache/TeaCache, SageAttention, TorchCompile,
    FreeU, LCM/Lightning/Hyper) - range/choices always come straight from
    `/object_info` via the same cap `get_node_info` uses, never invented, and
    a wired knob is shown as `(wired)` with no claim made.
  - `knowledge.matching_sources` surfaces a family's curated model-file
    download links into the Models note - never synthesizes a URL; two
    families seeded with URLs verified reachable before commit.
  - New `resolution-not-aligned` lint check for families with a confirmed
    alignment requirement (`multiple_of`), gated behind `lint()`'s
    `learned_dir` parameter so no untouched caller starts seeing it.
- **[DONE, round 22] Optional-input muted-source check, queue attribution.**
  From a live Chroma-HD-Flash troubleshooting session's bug report (one real
  bug, several UX gaps; two reported items were host/ComfyUI behavior, not
  draftsman defects). See Gotchas for the muted/dead-source mechanics. Also:
  `get_node_info`'s "not installed" error now suggests installed lookalikes
  (a live report hit `SigmasRescale` vs. the real `Sigmas Rescale`, with a
  space); `manage_queue(status)`/`get_run_status` now attribute prompt_ids this
  process itself queued via `run_workflow` (`draftsman_submitted` /
  `workflow_id`), closing the "was that timeout my run or the user's queue"
  ambiguity; `list_models` names `get_node_info` as the loader-specific escape
  hatch when a search comes up empty (a loader node can scan folders beyond
  the standard type-to-folder mapping); `run_workflow`'s docstring names the
  text-only-caller path (`return_preview=False` + `save_dir`) explicitly. Full
  writeup in the CHANGELOG, including what was deliberately NOT changed and why.
- **[DONE, round 21] Widget flags + autogrow authoring.** Closed four of round
  20's five open TODOs, two of which rested on a premise that turned out to be
  false — a reminder to re-check a TODO's *claim* before implementing around it:
  - *"No flag in `/object_info` distinguishes a widget from a socket."* There are
    two, `socketless` and `widgetType`. Using them fixed 26 classes.
  - *"Core autogrow nodes declare their full slot range up front, so connecting
    works."* They declare only the marker; connecting was impossible and the
    marker was reported as an unconnected required input.
  - *"Nested primitive chains are unhandled."* They are **unrepresentable** — a
    `PrimitiveNode` has no inputs at all, so `connect` refuses by name with the
    available list. Better than resolving it would have been; closed, not built.
  - *"`roll_primitive_control`'s randomize should maybe be value-weighted."*
    Decided WONTFIX: it mirrors the frontend's `addValueControlWidget`, which is
    index-uniform, and a combo with uneven option counts per category has no
    defensible notion of "fair". Matching the frontend IS the correct behaviour,
    so this was never pending work.

    Measured over 60 of the user's real saved workflows: 15 blocking false errors
    removed (`unconnected-input` 24→10, `js-widget-input` 16→15, lint
    `unconnected-input` 30→24) with every other finding count unchanged —
    `missing-node-class` 176, `invalid-combo-value` 87, `null-widget-value` 32,
    `out-of-range` 1 — and every surviving error verified genuine.
- **[DONE, round 20] V3 meta types + honest subgraph remedies.** From a live
  session that logged its snags while building a Krea-2 + LM Studio workflow;
  three of seven were real, two more surfaced while verifying those.
  `COMFY_MATCHTYPE_V3` was treated as a concrete type, so every wire touching a
  core `ComfySwitchNode` raised a blocking `link-type-mismatch` and made the
  bundled Krea-2 templates neither runnable nor savable; the V3 meta types were
  being counted as custom JS widgets and shifting `widgets_values`; every
  subgraph-inner finding claimed `edit_workflow` couldn't reach it (false — and
  the direct cause of a 14-node hand-rebuild); `inspect_workflow` listed
  unreachable boundary inputs as if connectable; and `no-prompt-preview` fired
  on graphs whose Show Text was tapped rather than inline. Plus `list_templates`
  silently dropping ~390 of ~450 templates. See Gotchas for the mechanics;
  `tests/test_round20_v3_types_and_subgraph_edits.py` pins all of it. Three
  reported items were **not** defects and were deliberately not "fixed" — the
  CHANGELOG's *Not changed* section records why, so they don't get relitigated.
- **[DONE, round 19] PrimitiveNode/Reroute authoring, socket type checking, data
  outputs.** From a live session that hit three walls building a
  character-cycling workflow, plus one found in the ensuing audit: `connect`
  treated COMBO as a wildcard (a STRING wired into a converted combo widget
  validated clean, then silently partial-ran on the live server);
  `PrimitiveNode`/`Reroute` were in `VIRTUAL_TYPES` but unauthorable via
  `add_node`, blocking the one correct idiom for a value that mirrors and can
  cycle a dropdown; `run_workflow`/`get_run_status` only ever surfaced FILE
  outputs, dropping a custom node's other return values entirely; and
  `OutputSlot` silently lost a primitive's widget marker on every save
  round-trip, including in the bundled SDXL template. All four fixed - see
  Gotchas for the mechanics. Net tool-schema cost went DOWN (~19.6k vs ~20.0k
  chars) despite the new capability, by collapsing `edit_workflow`'s five
  near-duplicate `*_in_definition` doc lines into one rule.
- **[DONE, round 18] Token efficiency** — `lint`'s overlap report collapsed from
  one finding per overlapping *pair* (quadratic: 1,326 findings / ~30k tokens on
  a 52-node graph) to one finding for the set; `save_workflow`/
  `organize_workflow`/`port_workflow` now cap their findings like the other
  tools (~35k → ~5.4k tokens on a messy graph); `list_models` and
  `search_nodes(detail=True)` capped; the `node-disabled` note collapsed to one
  finding. Plus two bugs: `_cap_findings` could return more than it received with
  a false "…0 more omitted" marker, and `diagnose_workflow`'s `ok` flipped on a
  purely informational finding. Ceilings pinned in
  `tests/test_round18_tokens.py`.
- **[DONE, round 17] Repo-audit remediation** — `set_widget` no longer destroys
  a neighbouring widget's value on pack nodes with custom JS-widget inputs
  (`socket_names` threaded through the whole write path); muted/bypassed nodes
  no longer emit blocking validation errors, with the new `dead-input-source`
  check covering what bypass actually breaks; relocation covers video/audio, not
  just images; `save_dir` on a background run says so instead of no-opping; the
  Comfy Registry degrades to a structured error instead of throwing away a
  diagnose's local findings (and resolves concurrently); import parse failures
  are actionable and API prompts with non-numeric node ids import; session
  writes are atomic; `detect_family` stopped rebuilding its YAML index per
  model reference. See the CHANGELOG for the full list.

- **[DONE, round 14] Layout companions + queue etiquette** — display nodes
  (Show Text / PreviewImage) glued beneath their source instead of a far-away
  Output group; empty-latent canvas nodes moved to the Inputs band;
  `run_workflow(front=...)` queue-busy gate and front-of-queue submits;
  `get_run_status` partial-accept detection (the round-13 `[MAYBE]` — the
  stored history entry's submitted prompt vs `outputs_to_execute`).
- **[DONE, round 13] Live-testing fixes** — `view_output` list-form image return
  (was a dict-wrapped `Image` that never rendered); partial-accept `node_errors`
  surfaced through `run_and_wait`/`run_workflow` (was a silent "success" with
  empty outputs); `widget-count-drift` suppressed on display/output nodes;
  big-int seed rounding traced to host transport (saved files are exact). See
  Gotchas.
- **[DONE, round 12] Headless API-submission parity** — custom JS-widget input
  serialization, `%date:%` token substitution, seed `control_after_generate`
  re-roll, case-insensitive connect, epsilon-`min` step alignment, and
  core-vs-custom combo severity (see Gotchas). All had been failing silently for
  custom-node-heavy workflows driven through `run_workflow`.
- **[DONE] Edit inside subgraph definitions** — flattening covers
  run/validate/export; targeted edits of definition internals are implemented
  (parsed into Workflow objects internally). Nested definitions (depth > 1)
  raise NotImplementedError.
- **[DONE] `step` constraint on INT/FLOAT widgets** — surfaced by
  `get_node_info` and enforced by validation during set_widget/add_node ops.
- **[DIAGNOSTIC ADDED] Inner nodes omitting `inputs` arrays** — lint checker
  detects missing `inputs` arrays on subgraph definition inner nodes and
  reports them as a diagnostic (with the node id and definition uuid); a
  synthetic fallback would still guess wrong, so this is surfaced rather than
  silently fixed.
