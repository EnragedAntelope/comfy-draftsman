# Round-3 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address round-3 testing feedback: truthful stage notes, room for nodes that grow after a run, quieter dynamic-node warnings, a Show-Text-preview lint rule, and clobber-safe saving.

**Architecture:** All behavior changes live in the tested modules (`graph/annotate.py`, `graph/layout.py`, `graph/validate.py`, `graph/lint.py`, `comfy/client.py`); `server.py` stays thin wiring (save rename loop + guidance strings). No new dependencies.

**Tech Stack:** Python 3.11+, FastMCP, httpx/respx, pytest (asyncio), live-instance integration tests on :8288.

## Global Constraints

- ruff check must pass; do NOT blanket-reformat (repo is not ruff-format-clean).
- Never pad `widgets_values` with `None` (frontend crashes on null string widgets).
- Unit tests: `uv run pytest -m "not integration"`. Integration: `uv run pytest -m integration` (needs the `.comfyui-test` instance on :8288).
- Commit per task; conventional-commit style messages as in git log.

---

### Task 1: Truthful post-processing note

**Files:**
- Modify: `src/comfy_draftsman/graph/annotate.py` (post branch of `_note_text`)
- Test: `tests/test_annotate.py`

**Interfaces:**
- Produces: post-stage fallback note text listing actual member node display names; never claims tuning or spatial position.

- [x] **Step 1: Write failing test** — post note with no matching techniques must name the actual nodes and must not contain "above" or "tuned".
- [x] **Step 2: Run test, verify fails.**
- [x] **Step 3: Implement** — replace the fallback line with:

```python
if not lines:
    steps = list(dict.fromkeys(
        n.title or (object_info.get(n.type) or {}).get("display_name") or n.type
        for n in members
    ))
    listed = ", ".join(steps[:4]) + (", …" if len(steps) > 4 else "")
    lines.append(f"⚙️ Extra image steps applied after generation: {listed}.")
```

- [x] **Step 4: Tests pass.**
- [x] **Step 5: Commit** `fix(annotate): post-processing note describes actual member nodes`

### Task 2: Layout — reserve space for populate-later nodes + real widget counts

**Files:**
- Modify: `src/comfy_draftsman/graph/layout.py`
- Test: `tests/test_layout.py`

**Interfaces:**
- Produces: `estimate_size(class_type, object_info, widget_count: int | None = None)`; callers (`apply_layout`, `apply_staged_layout`) pass `len(node.widgets_values)` when it is a list.
- New constants: `PREVIEW_RESERVE_H = 320.0`, `TEXT_PREVIEW_RESERVE_H = 150.0`; predicate `_is_text_display(class_type)` (regex `show.?text|display.?text|show.?anything`, case-insensitive).

- [x] **Step 1: Failing tests** — (a) node whose schema declares 20 optional widgets but instance has 3 widgets_values gets a much shorter estimate than schema-only; (b) `PreviewImage`/`SaveImage` (output_node with IMAGE input) reserve ≥ 300px extra height; (c) ShowText-like class reserves extra height and text width.
- [x] **Step 2: Verify fail.**
- [x] **Step 3: Implement** — widget rows: `rows = min(widget_count, len(slots)) if widget_count is not None else min(len(slots), MAX_WIDGET_ROWS)`; after base height, add reserves for image-output nodes (`schema.get("output_node")` and an IMAGE input) and text-display nodes; widen image nodes to ≥ 340.
- [x] **Step 4: Tests pass (incl. existing overlap tests).**
- [x] **Step 5: Commit** `feat(layout): reserve space for preview/save/show-text nodes; size from actual widget counts`

### Task 3: Quieter widget-count-drift on dynamic nodes

**Files:**
- Modify: `src/comfy_draftsman/graph/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Produces: drift finding downgraded to `level="info"` when the node is short of the schema count AND the schema declares ≥ 6 optional widget inputs (dynamic node signature). Overage or non-dynamic drift stays `warning`.

- [x] **Step 1: Failing tests** — dynamic node (12 optional STRING inputs, 3 values) → info; static node drift → warning stays.
- [x] **Step 2: Verify fail.**
- [x] **Step 3: Implement** in the drift branch:

```python
optional_widgets = sum(
    1 for spec in (schema.get("input", {}).get("optional", {}) or {}).values()
    if w.is_widget_input(spec)
)
dynamic_short = len(node.widgets_values) < len(slots) and optional_widgets >= 6
```

info message: "serializes only the widgets in use (dynamic node) - usually harmless".
- [x] **Step 4: Tests pass.**
- [x] **Step 5: Commit** `fix(validate): widget-count-drift is info-level for dynamic nodes`

### Task 4: `no-prompt-preview` lint rule + guidance strings

**Files:**
- Modify: `src/comfy_draftsman/graph/lint.py`, `src/comfy_draftsman/server.py` (instructions + build_workflow prompt)
- Test: `tests/test_lint.py` (new)

**Interfaces:**
- Produces: lint finding `{"code": "no-prompt-preview", "node_id": <encoder id>}` when a text-encode node's `text` input is wired from upstream and no text-display node (regex above) exists in the upstream string chain (BFS ≤ 4 hops). Uses `_prompt_role` imported from `.annotate` to restrict to positive prompts.

- [x] **Step 1: Failing tests** — wildcard→encoder with no ShowText fires; wildcard→ShowText→encoder does not; hand-typed (unwired) prompt does not.
- [x] **Step 2: Verify fail.**
- [x] **Step 3: Implement** rule + add one sentence to server instructions and build_workflow step 4: generated positive prompts should pass through a Show Text node so the user can see the final prompt.
- [x] **Step 4: Tests pass.**
- [x] **Step 5: Commit** `feat(lint): flag wired positive prompts lacking a Show Text preview`

### Task 5: Clobber-safe save_workflow

**Files:**
- Modify: `src/comfy_draftsman/comfy/client.py` (`save_userdata_workflow`), `src/comfy_draftsman/server.py` (`save_workflow`)
- Test: `tests/test_client.py`, `tests/test_mcp_e2e.py`

**Interfaces:**
- `save_userdata_workflow(name, document, overwrite: bool = False)` sends `?overwrite=false` and raises `FileExistsError(filename)` on HTTP 409.
- `save_workflow(workflow_id, name, allow_invalid=False, overwrite=False)`: on conflict, retries `"{name} (draftsman)"`, `"{name} (draftsman 2)"` … `(draftsman 20)`; result gains `"saved": True` and `"renamed_from"` when renamed. `overwrite=True` restores old clobber behavior.

- [x] **Step 1: Failing tests** — client: 409 → FileExistsError; overwrite=True sends overwrite=true. Server: first name conflicts → saved under "(draftsman)" suffix with renamed_from set.
- [x] **Step 2: Verify fail.**
- [x] **Step 3: Implement.**
- [x] **Step 4: Tests pass.**
- [x] **Step 5: Commit** `feat(save): never overwrite an existing workflow file by default`

### Task 6: Integration tests — save refusal + no-clobber

**Files:**
- Modify: `tests/test_integration_live.py`

- [x] **Step 1: Add** `test_save_refuses_invalid_workflow` (bogus ckpt_name → `saved is False`, findings include invalid-combo-value) and `test_save_never_clobbers` (save same name twice → two distinct filenames), wiring `server._State` to the live client via monkeypatch.
- [x] **Step 2: Run against :8288, verify pass.**
- [x] **Step 3: Commit** `test(integration): save-refusal and no-clobber paths against live instance`

### Task 7: Docs, memory, delivery

- [x] README: document new save default, `no-prompt-preview` lint code, Show Text guidance.
- [x] Run full unit suite + ruff check; run integration suite.
- [x] Push to main; verify CI.
- [x] Update project memory (round-3 section; clear resolved TODOs).
- [x] Local MCP install serves repo source via `uv run --project` — no reinstall needed; note that a session restart / `/mcp` reconnect picks up the new code.

## Self-review notes

- Item 4 (spacing) and item 3's estimate_size TODO are both Task 2 — one mechanism.
- Item 1 (Show Text) implemented as lint + agent guidance rather than auto-insertion: ShowText classes come from third-party packs which may not be installed; silently inserting third-party nodes contradicts the server's own "installing packs is the user's call" policy.
- Item 2 uses ComfyUI's native `overwrite` query param on POST /userdata (409 on conflict) — verified against the live instance in Task 6.
