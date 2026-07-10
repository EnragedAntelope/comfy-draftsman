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
- **Image-metadata metadata** — `view_output` returns a `meta` dict alongside
  images so text-only models can describe renders.
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
    widget-backed slot (carries a `widget` marker) and its value is pack-specific
    JS state (a dict/list, not a scalar), it genuinely can't be serialized to the
    raw API — `validate` blocks it with a `js-widget-input` error (with the
    remediation) rather than silently no-opping the branch. A generic tool cannot
    replay a pack's client-side JS; the honest outcome is a loud, actionable stop.
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

## Remaining TODOs

None open. Recently closed:

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
