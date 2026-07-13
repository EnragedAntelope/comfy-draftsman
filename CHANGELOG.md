# Changelog

## 0.4.2 — Round 13: live-testing fixes

A long custom-node-heavy testing session (krea2 speed optimization) surfaced a handful of correctness and noise issues in the execution/inspection path. None change the workflow model; all make what draftsman *reports* match what ComfyUI actually did.

### Fixed

- **Inline images now render (`view_output`)** — `view_output` returned a dict *containing* an `Image` object, which FastMCP serializes as a Python `repr` string (`<...Image object at 0x...>`) rather than an image content block, so the picture never displayed. It now returns the list form `[{"meta": {...}}, Image(...)]` — the same shape `run_workflow`'s preview already uses — so the render is actually visible while text-only models still get the dimensions/filename `meta`.
- **Partial runs no longer masquerade as success (`run_workflow`)** — ComfyUI can return **HTTP 200 with `node_errors`** (not 400): it queues the prompt, runs the still-valid subgraph, and drops the rejected nodes' branches. Those node_errors were swallowed, so a run that executed only a few text-utility nodes in ~50 ms reported bare `status: success` with empty outputs. `run_and_wait` now threads the submit-time node_errors onto the result; `run_workflow` downgrades `status` to `"partial"` with a loud `warning`, and `wait=False` surfaces them on the `queued` response.
- **Display-node validation noise removed** — `widget-count-drift` fired on nearly every ShowText / rgthree "Display Any" / preview node, which stash the text they display into `widgets_values` beyond their declared schema widgets. A count overflow on an `output_node` is now recognized as expected and suppressed; shortfalls and non-output-node mismatches still report.

### Notes

- **Big-int seeds** — confirmed that `save_workflow`/`export` preserve seeds `> 2^53` exactly (Python `json` keeps arbitrary-precision ints). A rounded seed in a tool *response* is the MCP host's JS-side `JSON.parse` coercing to a double (display-only, not in the saved file); draftsman intentionally does not alter seed values to match a rounded readback.
- **Custom widget-backed JS inputs** (LoraManager `text`, StyleStringInjector2 `gallery`) remain a loud `js-widget-input` stop by design — a generic scalar-emit fix was considered and rejected because the live server rejected hand-serialized values that weren't rebuilt by the pack's own client-side JS. Tracked as an OPEN item in `docs/ARCHITECTURE.md`.

### Changed

- **Version bump** — `0.4.1` → `0.4.2`.

## 0.4.1 — Round 12: headless API-submission parity

Live testing against a custom-node-heavy workflow surfaced gaps where a graph that runs in the browser could not be driven through `run_workflow`, because several behaviors are implemented by ComfyUI's frontend JS and the raw `/prompt` backend never performs them. draftsman now mirrors them at submit time (as it already mirrors subgraph flattening).

### Fixed

- **Custom JS-widget inputs no longer fail silently** — inputs with a pack-specific type that the node's own frontend renders as a widget (e.g. LoraManager's `AUTOCOMPLETE_TEXT_LORAS`, style-gallery buttons) were silently dropped from the UI→API conversion, leaving their downstream chain unrunnable while ComfyUI still reported success. Two cases now handled: (a) a plain-scalar custom widget the node did not serialize as a socket is recognized per-instance and its value flows into the `/prompt` payload; (b) a custom widget the node exposes as a widget-backed slot whose value is pack-specific JS state (an object the raw API can't send) is now blocked at validation with a clear, actionable `js-widget-input` error (connect it, or swap for the pack's plain-STRING variant) instead of silently no-opping the branch or reporting a misleading "not connected".
- **`%date:FORMAT%` filename tokens** — `filename_prefix` tokens like `%date:yyyy-MM-dd%` (substituted by a frontend extension, never by the backend) are now substituted at API-serialization time, fixing an `OSError` on Windows (the literal `:` is an illegal filename char). The saved UI document keeps the literal token for the browser.
- **Step-alignment false positives** — the `step` check used a schema's `min` as the grid origin, so an epsilon `min` (e.g. `0.0001`) rejected every normal value (even a workflow's own saved `denoise=0.36`). Alignment now accepts either origin `0` or `min` with a step-relative tolerance.
- **Case-insensitive connect** — `edit_workflow`'s `connect` no longer rejects `STRING → string` as a type mismatch (litegraph slot typing is case-insensitive).
- **Combo false-positive flood** — a combo value absent from the `/object_info` snapshot is now a blocking `error` only for on-disk file listings and core-node enums; for third-party nodes that repopulate combos client-side (wildcard/LoRA/style pickers) it is a non-blocking `warning`. Model-installed checks and core-enum typos still block; the noise that forced `allow_invalid=True` is gone.

### Added

- **Seed re-roll on run** — `run_workflow` now honors `control_after_generate` (which only ever fired in the browser): seeds set to `randomize`/`increment`/`decrement` are re-rolled before submit and the new value persisted, so headless runs vary instead of repeating one seed. Pass `roll_seeds=False` for a deterministic re-run.
- **Findings cap** — `validate_workflow`/`diagnose_workflow` cap returned findings (most-severe first, every error kept) with a truncation marker, bounding token cost on noisy graphs.

### Changed

- **Version bump** — `0.4.0` → `0.4.1`.

## 0.4.0 — Round 11 improvements

### Added

- **Step constraint enforcement** — `INT`/`FLOAT` widget values are now validated against the `step` field exposed by `/object_info`. Misaligned values produce warning-level findings with two-sided float tolerance.
- **Subgraph definition editing** — `edit_workflow` now supports six ops for modifying inner subgraph definitions without unwrapping the parent workflow:
  - `add_node_to_definition`
  - `remove_node_from_definition`
  - `set_title_in_definition`
  - `set_mode_in_definition`
  - `connect_in_definition`
  - `set_widget_in_definition`
  Nested definitions remain unsupported and raise `NotImplementedError`.
- **Subgraph materialization diagnostics** — `subgraph.flatten()` returns a third `diagnostics` element that reports boundary links dropped during flattening, and `validate()` warns when inner nodes lack an `inputs` array.
- **`view_output` metadata** — the tool now returns `{"image": <Image>, "meta": {...}}` so text-only or metadata-bearing outputs can carry filename, format, dimensions, and subfolder alongside the image bytes.
- **Comfy Org API key support** — when `COMFY_API_KEY` is set in the environment, `run_workflow` injects it into the prompt payload as `extra_data.api_key_comfy_org`. Omitting the variable leaves the payload unchanged.

### Changed

- **Version bump** — `0.3.0` → `0.4.0`.
- **Documentation** — updated `ARCHITECTURE.md` TODOs and tightened `.gitignore` hygiene.

### Fixed

- Removed a duplicate handler for `add_node_to_definition`.
- Corrected integration-test assertions to match the new `view_output` return shape.
- Cleaned up unused test variables and trailing-newline lint.

### Tests

- Added unit coverage for step constraints, subgraph definition edits, dropped boundary links, and missing inner inputs.
- Integration tests pass against a live ComfyUI instance: **9 passed, 1 skipped** (Depth-Anything-3 nodes not installed on the test instance).
- Full suite: **291 unit tests passed**, **10 integration tests deselected**, `ruff check .` clean.
