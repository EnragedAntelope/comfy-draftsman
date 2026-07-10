# Changelog

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
