# AGENTS.md — comfy-draftsman

A local-first MCP server that drafts, organizes, validates, and runs ComfyUI workflows. Delivers clean, labeled, annotated UI-format graphs that humans can read — computed layout, colored stage groups, titled nodes, green-highlighted knobs, and generated guidance notes. Python ≥3.11, hatchling build, httpx + websockets + mcp stack.

**Deep reference: `docs/ARCHITECTURE.md` (read it before engine/graph/validation changes)**

## Architecture in 60 seconds

- **Thin MCP wiring layer.** `server.py` exposes tools/prompts; all logic lives in tested modules underneath. Ground truth is the live ComfyUI instance's `/object_info`.
- **Graph model with schema 0.4/1.0 support.** `graph/` handles workflow ↔ internal model conversion, widget mapping, subgraph flattening, live-instance validation, and layout.
- **ComfyUI client layer.** `comfy/` provides the httpx REST client, object_info catalog, websocket progress tracker, and Comfy Registry lookups for missing node packs.
- **Two-layer knowledge system.** Per-family tuning floor (YAML) + persistent learned overlay. `record_learning` saves researched settings so future sessions start smarter.
- **V3 dynamic combos first-class.** `COMFY_DYNAMICCOMBO_V3`, `COMFY_AUTOGROW_V3`, match types, and socketless widgets are handled natively — values round-trip through the API's dotted-key form.
- **Token discipline.** Every tool returns bounded lists; findings are severity-capped; summaries clip long strings. Full object_info is never returned to the model.

## Layout

| Directory | Purpose |
|-----------|---------|
| `src/comfy_draftsman/server.py` | MCP tools/prompts — thin wiring only, no logic |
| `src/comfy_draftsman/graph/` | Workflow model, widgets, subgraph flattening, validate, lint, annotate, layout, knobs, port |
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

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: `pwsh scripts/check-agents-md.ps1 AGENTS.md CLAUDE.md` — must exit 0.

Deep architecture, data flow, subgraph mechanics, and gotchas: `docs/ARCHITECTURE.md`.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines. Link to `docs/ARCHITECTURE.md` for detail.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; `docs/ARCHITECTURE.md` = deep reference. Identical build/test/run commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
