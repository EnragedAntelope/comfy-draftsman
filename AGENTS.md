# AGENTS.md — comfy-draftsman

A local-first MCP server that drafts, organizes, validates, and runs ComfyUI workflows. Delivers clean, labeled, annotated UI-format graphs that humans can read — computed layout, colored stage groups, titled nodes, green-highlighted knobs, and generated guidance notes. Python ≥3.11, hatchling build, httpx + websockets + mcp stack.

**Deep reference: `docs/ARCHITECTURE.md` (read it before engine/graph/validation changes)**

## Current state

_Last verified: 2026-10-26_

- **Status:** in development; v0.25.1 (`src/comfy_draftsman/__init__.py` is the single source of truth for the version). CI green. **Published to PyPI** as `comfy-draftsman` via Trusted Publishing (`.github/workflows/release.yml`); a `v*` tag push publishes and cuts the GitHub Release - see README -> "Publishing a release".
- **Works:** the full draft → organize → validate → run → save loop against a live ComfyUI instance; schema 0.4 and 1.0 graphs including subgraph flatten/rebuild; V3 dynamic combos, autogrow inputs and match types round-tripping through the API's dotted-key form; `organize_workflow`'s labeled stage groups, knob cards and guidance notes; workflow import straight from ComfyUI's browser (`list_workflows` + `import_workflow(name=…)`); the per-family knowledge floor plus a persistent learned overlay written by `record_learning`; `to_api` bypass routing that mirrors the frontend's; `organize_workflow` staging that uses graph position (switches, primitives, MODEL patches, i2i preprocessing) with row-wrapped bands.
- **In progress:** nothing half-built - each round lands complete. 0.25.1 flags widget values of the wrong type (shifted by a pack update) and reports them on import as `widget_drift`, stops reading a newly added core socket as a JS widget, extends the sampling-size lint to every size-tuned model patch (JigsawHiRes) per sampler, and adds the `set_resolution` edit op for sweeps. 0.25.0 adds the `fibo2`, `flux3_klein`, `minimax_h3` and `anima` families, a spectral color-stripe banding score (checked on real MOSS SFT renders), `above-native-resolution` and `hidden-troubleshooting` lints, headless fresh LoRA Manager loaders (text and `loras` kept in sync), off-step values accepted at edit time, unwrapped text/JSON outputs, `record_learning` shape hints, and sweep `sampling_s` and crop guide. 0.24.0 adds the `moss_image` family with a `family-avoid-node` lint, guards `record_learning` shapes, maps imported LoRA Manager widgets by their saved names, makes organize text plain (no emoji), refreshes object_info on edits naming an unlisted model, and gives sweeps fractional crops, per-crop sheets and cached-run marking. 0.23.0 fixes upload cache refresh, partial-edit reporting and template findings, adds sweep output relocation/reuse hints and sampling-size lint. 0.22.0 fixes a LoRA Manager feedback round: JS-widget inputs added as widget-backed slots, stale link slots repaired on import, collapsed-node overlap lint, `disconnect`/`set_color`/group `bounding` ops, and organize guidance that keeps hand-arranged layouts. `CHANGELOG.md` records older rounds and what was deliberately not changed.
- **Known gaps / next steps:** confirmation dialogs fall back to the explicit confirm flag on the 2026-07-28 MCP protocol (no back-channel for `ctx.elicit`; pre-2026 handshakes still get the dialog); misaligned pack nodes are reported, not realigned, so they can't run headlessly; no confirmation popup appears in the Claude Code VS Code extension (verified 2026-10-11 on 0.21.0); gates take the ask-in-chat fallback, and overwrite proceeds; `add_node` treats a custom type nothing outputs as a JS widget, so a real socket whose producer pack is missing gets a widget marker too; some families carry no VRAM data and correctly stay silent; acceleration variants match the UNet filename only, not `lora_name`; `COMFY_DYNAMICSLOT_V3` is classified but never exercised (no live instance declares one - do not implement it speculatively); widget-backed custom-JS inputs stay a deliberate loud stop rather than a silently-wrong emit (the LM Lora Loader is the one pack resolved by name); `flux3_klein`, `minimax_h3` and `anima` floors come from public sources and have not been rendered locally. All written up in full under `docs/ARCHITECTURE.md` -> "Remaining TODOs".
- **Deep docs:** `docs/ARCHITECTURE.md` (module map, data flow, subgraph mechanics, hard-won gotchas, open TODOs), `docs/PERMISSIONS.md` (which tools are read-only), `CHANGELOG.md`.

## Architecture in 60 seconds

- **Thin MCP wiring layer.** `server.py` exposes tools/prompts; all logic lives in tested modules underneath. Ground truth is the live ComfyUI instance's `/object_info`.
- **Graph model with schema 0.4/1.0 support.** `graph/` handles workflow ↔ internal model conversion, widget mapping, subgraph flattening, live-instance validation, and layout.
- **ComfyUI client layer.** `comfy/` provides the httpx REST client, object_info catalog, websocket progress tracker, and Comfy Registry lookups for missing node packs.
- **Two-layer knowledge system.** Per-family tuning floor (YAML) + persistent learned overlay. `record_learning` saves researched settings so future sessions start smarter. The floor also carries optional, sourced VRAM requirements behind `knowledge.fit_verdict`.
- **Gates that teach reactively.** The VRAM fit verdict and the partner-node spend gate both fire before anything irreversible, so each explains itself in its own response instead of in a docstring everyone pays for.
- **V3 dynamic combos first-class.** `COMFY_DYNAMICCOMBO_V3`, `COMFY_AUTOGROW_V3`, match types, and socketless widgets are handled natively — values round-trip through the API's dotted-key form.
- **Token discipline.** Every tool returns bounded lists; findings are severity-capped; summaries clip long strings. Full object_info is never returned to the model. The tool surface — the one budget paid on *every* request — is stripped of pydantic's auto-generated schema titles *and* of docstring indentation at import (`_trim_published_surface`), and held under a ceiling by `tests/test_round18_tokens.py`. Keep that ceiling interpreter-independent: only Python 3.13+ strips docstring indentation at compile time.

## Layout

| Directory | Purpose |
|-----------|---------|
| `src/comfy_draftsman/server.py` | MCP tools/prompts — thin wiring only, no logic |
| `src/comfy_draftsman/graph/` | Workflow model, widgets, subgraph flattening, validate, lint, annotate, layout, knobs, port, spend |
| `src/comfy_draftsman/comfy/` | httpx client, object_info catalog, websocket progress, Comfy Registry |
| `src/comfy_draftsman/knowledge/` | Per-family tuning floor (YAML) + learned overlay |
| `src/comfy_draftsman/config.py` | Env-driven config (COMFYUI_URL, DRAFTSMAN_SESSION_DIR, ...) |
| `src/comfy_draftsman/session.py` | workflow_id → Workflow store, persisted under session dir |
| `tests/` | pytest + pytest-asyncio; integration tests require running ComfyUI |
| `docs/` | Architecture (deep reference), images, showcase |

## Build / test / run

```bash
# Install (editable)
uv sync

# Run the MCP server
comfy-draftsman
# or: uv run comfy-draftsman

# Test (excludes integration tests by default)
uv run pytest

# Test with integration tests (requires COMFYUI_TEST_URL)
uv run pytest -m integration

# Lint
uv run ruff check src tests

# Typecheck
uv run mypy

# Wheel data check (CI's `package` job) - --refresh, or uv serves a cached
# extract of the previous build with the same version number
uv build && uv run --isolated --no-project --refresh --with "$(ls dist/*.whl)" python -c "
import comfy_draftsman.knowledge as k; assert k.get_guidance('flux')['sampling'] and k.get_guidance('qwen_image21')['sampling']"
```

## Conventions & gotchas

- Dynamic nodes serialize only in-use widgets — never pad widgets_values with `None`.
- Frontend runs `.replace()` over every string widget at queue time — `null` crashes the editor.
- Seed control widgets are a name heuristic (INT literally named `seed`/`noise_seed`).
- Never default paths off `Path.cwd()` — MCP hosts launch from arbitrary directories. Session state lives under `~/.comfy-draftsman`.
- `object_info` is multi-megabyte — never return it or full combo lists to the model. Everything recurring must be capped or digested.
- Validation gates: `run_workflow` and `save_workflow` refuse on `validate()` errors unless `allow_invalid=True`.
- `lint()` is advisory only — it never blocks.
- `edit_workflow` ops deliberately do NOT reach inside subgraph definitions — rebuild flat to modify internals.
- `organize_workflow` never synthesizes a download URL or an alignment (`multiple_of`) requirement — both only ever come from a curated family YAML or a `record_learning` call; a guessed one is worse than none.
- **A green test suite says nothing about what a new user resolves.** Dev and every CI *test* job install from `uv.lock`; only the `package` job resolves against the index. That is why `mcp>=1.10` (unbounded) shipped an unstartable 0.15.0 — mcp 2.0.0 dropped `mcp.server.fastmcp` — with 729 tests passing on six legs. Runtime pins whose major would break a module-scope import need a ceiling, and both wheel checks import `comfy_draftsman.server`, not just a dependency-free subpackage.
- **Never invent VRAM numbers.** `hardware.vram_gb` requires a `hardware.source` that states the figure; `tests/test_hardware_fit.py` fails the build otherwise.
- **Unknown is not paid.** No VRAM data, or an instance too old to flag `api_node`, means *say nothing* / *don't bill* — never a default, never a nag. A warning repeated on every call is one an agent learns to skip.
- **New behavior is taught by responses, not by docstrings.** The tool surface is re-sent on every request (`test_round18_tokens.py` caps it); a gate that fires before the irreversible act can explain itself in its own return value, paid for only by the caller who hit it.

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing, re-read this file and `CLAUDE.md` against that rule. Maintainers run a denylist checker over both; it lives outside the repo (it is a machine-local tool, and a `pwsh` script would not run on the Linux CI runner anyway), so there is deliberately no in-repo command to invoke here.

Deep architecture, data flow, subgraph mechanics, and gotchas: `docs/ARCHITECTURE.md`.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines. Link to `docs/ARCHITECTURE.md` for detail.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; `docs/ARCHITECTURE.md` = deep reference. Identical build/test/run commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
