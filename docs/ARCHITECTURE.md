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
- **Display/output nodes overflow `widgets_values` on purpose.** ShowText,
  rgthree "Display Any", and preview nodes stash the text/data they display into
  `widgets_values` beyond their declared schema widgets. A count *overflow* on a
  node whose schema sets `output_node` is expected and suppressed (not
  `widget-count-drift`); a shortfall, or any mismatch on a non-output node, still
  reports.

## Remaining TODOs

Open:

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
  frontend. (Code note: validate.py:388 currently blocks on any widget-backed
  unconnected custom input regardless of value type; the surrounding docs' hint
  at value-awareness is aspirational, not implemented — see above for why.)
Recently closed:

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
