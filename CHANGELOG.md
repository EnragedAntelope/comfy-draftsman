# Changelog

## 0.6.0 — Round 15: Cowork/Code delivery readiness up front

A sandboxed client (Claude Cowork, Claude Desktop) can only be *handed* a finished render if `COMFYUI_MOUNT_DIR` points at a folder both this server and the caller can see. Until now that was discovered reactively — after spending a whole render — via a late error, and a natural relative `dest_dir` from an agent silently resolved against the server's own working directory (often `System32` on an MCP host). This round makes relocation readiness visible before the render, and turns the relative-path footgun into a clear refusal.

### Added

- **`get_instance_info` reports relocation readiness** — the "call first" tool now returns a `relocation` block (`configured` / `writable` / `path`, or an actionable `hint` when unset). An agent can see up front whether a render can be delivered to the user and, if not, ask them to set `COMFYUI_MOUNT_DIR` before wasting a render.
- **`check_setup` diagnostic tool** — a one-shot doctor for a fresh install or a sandboxed client: is ComfyUI reachable, can renders be relocated to a caller-reachable folder, is the partner-node key present. Unlike `get_instance_info` it never raises — a down instance is a failed check, not an error — so it's the right first call when a render can't be delivered or the instance seems unreachable. `ok` is gated on ComfyUI being reachable; relocation is a soft check surfaced via `hint`.
- **`draftsman://capabilities` resource** — a machine-readable snapshot of what the process can do right now: relocation status, background runs, and whether the partner-node API key (`COMFY_API_KEY`) is present. Same `relocation` block as `get_instance_info`, without a tool round-trip.
- **Mount-dir write probe** — relocation readiness is verified, not assumed: the mount dir is resolved, created, and a probe file is written+read+removed, so a configured-but-unwritable mount is reported as `writable: false` with the OS error rather than failing mid-render.

### Fixed

- **Relative `dest_dir` / `save_dir` is refused, not silently misplaced** — `run_workflow(save_dir=...)` and `save_output(dest_dir=...)` now reject a relative path with a clear error explaining that the server's working directory is not the agent's, so a relative path would land somewhere invisible. Absolute paths and `~`-expansions are unaffected. Previously a value like `./renders` resolved against the server's cwd (`System32` on Windows MCP hosts) and either failed opaquely or wrote out of sight.
- **Unreachable-instance errors are now actionable** — every ComfyUI HTTP call routes through one wrapper that turns a transport-level failure (instance down, wrong `COMFYUI_URL`, DNS/connect timeout) into a `ComfyConnectionError` naming the URL and the likely fixes ("is ComfyUI running… a remote instance must be started with `--listen`"), instead of surfacing a raw `httpx.ConnectError` an agent can't reason about. A real HTTP response (even 4xx/5xx) is untouched — only failures-to-connect are reclassified.

### Changed

- **Version is single-sourced** — `pyproject.toml` now derives the version dynamically from `comfy_draftsman.__version__` (`[tool.hatch.version]`), so the two can no longer drift. This session found them already diverged: `__init__` sat at `0.4.2` while `pyproject` had moved to `0.5.0` (Round 14 bumped one, not the other). A packaging test guards the config.
- **Removed a stray duplicate** of the `_WRITE_INSTANCE` / `_DESTRUCTIVE_INSTANCE` tool-annotation constants in `server.py` (defined twice, identically).

### Docs

- **"Using with Claude Cowork / Code" README section** — explains the shared-folder requirement for `COMFYUI_MOUNT_DIR` (the server and the sandbox must see the same directory), the absolute-path rule, and how to check readiness via `get_instance_info` / `check_setup`.

### Notes

- **Version bump** — `0.5.0` → `0.6.0` (now single-sourced from `__init__.__version__`).

## 0.5.0 — Round 14: readable layouts by default + queue etiquette

User feedback from real sessions: the organized layout swept every Show Text and PreviewImage node into one far-away Output group (pairing six previews with six samplers meant tracing wires across the whole canvas), and a test render had to wait behind a long existing queue.

### Changed

- **Display nodes stay beside their source (`organize_workflow`)** — Show Text-style nodes and `PreviewImage`-style nodes are now *companions*: they inherit the pipeline stage of the node they display and are glued directly beneath it, so the preview for a sampler chain sits inside that chain and a wildcard's Show Text sits under the wildcard. Chains of display nodes resolve to the real source; an unwired preview keeps its old Output placement. SaveImage-style disk writers still group under Output — they are real outputs, not displays.
- **Resolution is an input, not a sampler detail** — empty-latent canvas nodes (`EmptyLatentImage` & family) now classify into the leftmost **Inputs** band (titled `📐 Image Size` when that's all it holds), with a guidance note, so everything a user typically tweaks — source media, canvas size, models/LoRAs, prompts — reads left-to-right before the tuned machinery. They were previously buried in Sampling.

### Added

- **Front-of-queue runs without touching the queue (`run_workflow(front=...)`)** — `front` defaults to `None`: if ≥2 prompts are already pending, **nothing is queued** and `{status: "queue_busy"}` comes back with the pending count so the user can choose. `front=True` queues the run to go *next* after the current job — existing pending jobs are never deleted or interrupted — and `front=False` waits at the back of the line. Works for both `wait=True` and `wait=False` runs.
- **`get_run_status` detects partial accepts** — a `wait=False` run polled through `get_run_status` can now see queue-time partial accepts: the stored history entry's full submitted prompt is compared against ComfyUI's `outputs_to_execute`, and output nodes dropped at queue time downgrade the status to `"partial"` with `dropped_output_nodes` and the usual warning. Closes the round-13 `[MAYBE]` TODO.

### Notes

- **Version bump** — `0.4.2` → `0.5.0` (layout defaults changed; new `front` run parameter).

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
