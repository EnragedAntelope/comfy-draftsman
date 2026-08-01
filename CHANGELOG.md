# Changelog

## 0.12.0 — Widget flags, autogrow authoring

Closes four of 0.11.0's five open TODOs. Two of them rested on a premise that
turned out to be false, so the "fix" was smaller and much more valuable than the
TODO predicted — worth re-checking a TODO's claim before implementing around it.

Measured over 60 real saved workflows on a live instance: **15 blocking false
errors removed** (`unconnected-input` 24→10, `js-widget-input` 16→15, lint
`unconnected-input` 30→24) with every other finding count unchanged, and every
surviving error verified genuine.

### Fixed

- **ComfyUI's own `socketless` / `widgetType` flags are now believed.** V3's
  `WidgetInput` serializes both, and draftsman ignored them — so on a stock
  instance **26 classes had a ComfyUI-declared widget treated as a required
  connection socket.** `ColorToRGBInt`, whose only parameter is a socketless
  `COLOR`, got a phantom socket, no widget value at all, and a *blocking*
  `unconnected-input` error for a graph that was perfectly fine; `TextOverlay`
  silently dropped its `color` value and shifted every later widget up a slot.
  `forceInput` still wins where a node declares both (7 inputs do), and
  `socketless: false` is honoured as the real value it is. Widget value checks
  now use `widgetType` as the effective kind, since a union like `"FLOAT,INT"`
  names no single checkable type.
- **Autogrow inputs are authorable.** `COMFY_AUTOGROW_V3` (66 inputs on a stock
  instance) declares only a *marker* plus a template; the real sockets are
  synthesized from `prefix`+index or an explicit `names` list. Three consequences,
  all previously wrong:
  - The marker was reported as an unconnected required input — a blocking error
    on 56 required markers. `validate` now checks what the backend actually
    requires (at least `min` synthesized slots connected,
    `autogrow-underfilled`), and `lint` mirrors the exemption so the two tools
    stop contradicting each other.
  - The slot names were undiscoverable, so `connect` had nothing to aim at.
    `get_node_info` now expands them (capped) with a hint, and `add_node`
    materializes the mandatory ones.
  - **The prompt key is dotted — `images.image0`, not `image0`.** Confirmed from
    ComfyUI's `parse_class_inputs`/`finalize_prefix` and independently from the
    frontend bundle. This one is silent when wrong: the backend does not error on
    an unrecognized key, it just runs the node with that input missing. `to_api`
    normalizes, `connect` accepts either spelling, and an imported socket is
    never renamed — canonicalizing one would rewrite the user's file. Gaps are
    legal, so `image0`+`image2` runs as written with no renumbering.

### Closed without code

- **Nested primitive chains (`primitive → Reroute → primitive`)** were carried as
  unhandled. They are *unrepresentable*: a `PrimitiveNode` is a pure source with
  no inputs, so `connect` refuses by name with the available list. That is a
  better outcome than resolving the chain would have been.
- **`roll_primitive_control`'s index-uniform randomize** is correct, not a
  shortfall: it mirrors the frontend's `addValueControlWidget`, and a combo with
  uneven option counts per category has no defensible notion of "fair".
- **The `js-widget-input` block is now evidence-backed rather than assumed.** The
  genuinely-unemittable types (`AUTOCOMPLETE_TEXT_LORAS`, `RANDOMIZER_CONFIG`,
  `LORA_POOL_CONFIG`, `LORAS`) carry **no schema flags at all**, while
  LoraManager's ordinary prompt widget declares `widgetType` — so the new rule
  picks up the latter and leaves the former blocked, exactly as intended.
  Verified across the same 60 workflows: every surviving block is an unflagged
  bespoke type.

## 0.11.0 — V3 meta types, honest subgraph remedies

From a live session that built a Krea-2 + LM Studio workflow and logged every
snag as it went (report kept locally, not in the repo). Three of the seven
reported snags were real draftsman defects; two more surfaced while verifying
them. The rest belonged to other tools or were misreadings — see *Not changed*.

### Fixed

- **`COMFY_MATCHTYPE_V3` is a wildcard, not a concrete type.** ComfyUI's own
  executor short-circuits `validate_node_input` to `True` whenever either end of
  a wire is a MatchType ("validation for this is handled by the frontend"), and
  core nodes rely on it — `ComfySwitchNode` (If/Else Switch) is MatchType in and
  MatchType out. `types_compatible` treated the marker as an ordinary type, so
  every wire into or out of a switch raised a `link-type-mismatch` **error**.
  Because that check gates both `run_workflow` and `save_workflow`, any workflow
  containing a switch — including every bundled Krea-2 template — was neither
  runnable nor savable through draftsman, with the finding confidently blaming
  the template. A live session hand-rebuilt a 14-node graph rather than use one.
- **V3 meta types are no longer counted as custom JS widgets.** Round 17's
  per-instance heuristic reads "a custom-typed input the node did not serialize
  as a socket" as a pack-rendered widget. ComfyUI's five own `COMFY_*_V3` meta
  types (`MATCHTYPE`, `AUTOGROW`, `DYNAMICSLOT`, `DYNAMICCOMBO`, `MULTITYPED`)
  fit that shape exactly while being no widget at all — an autogrow node emits
  `value0..valueN` in place of its `values` marker, and an unconnected MatchType
  slot is simply absent. Each one invented a widget slot, which shifted every
  later `widgets_values` entry up a position on read and on rebuild.
- **Findings inside a subgraph now name the op that fixes them.** Every inner
  finding ended with "edit_workflow can't reach inside; rebuild flat to change
  it" — false since the `*_in_definition` ops shipped. It is the single line
  that caused the rebuild above. Inner findings now carry `definition_id` and
  `inner_node_id` as structured fields (the exact arguments those ops take), and
  the results that return findings carry one `subgraph_edit` sentence explaining
  how to use them. Only a genuinely unreachable node — nested deeper than
  `subgraph_as_workflow` can parse — still says "rebuild flat", and
  `flatten`'s provenance computes that honestly rather than assuming.
- **`inspect_workflow` distinguishes exposed from internal subgraph inputs.** A
  definition's boundary inputs are not all reachable from the parent graph: the
  instance node exposes only some as real sockets, and `connect` addresses
  instance sockets. Listing all of them unqualified read as "these are
  connectable" and sent a session chasing a `value` socket that did not exist
  (the bundled Z-Image template declares six boundary inputs and exposes one).
  Unexposed inputs now report as `name (internal)`, and the subgraph note says
  what to do instead.
- **`no-prompt-preview` no longer fires on a previewed workflow.** The rule
  walked only the encoder's upstream chain, so it missed the more common
  hand-wired shape — a Show Text *tapped off* the generator's output, in
  parallel with the encoder rather than in series. That displays the identical
  string, and arguably better (the display isn't in the path it reports on). A
  lint that fires on correct work teaches callers to ignore the rule.

### Changed

- **`list_templates` returns a bounded object instead of a silently truncated
  list.** The bundled catalog is ~450 templates; the tool returned a bare
  `list[:60]` with no indication that ~390 were dropped, so a caller who found
  no match reasonably concluded none existed — the one failure mode the
  repo's own bounded-list rule exists to prevent. It now returns
  `{count, templates, hint}` with the true match count and a `search=` hint
  whenever the list is cut. Cap tightened to 40 and the description clip to 110
  chars now that the response is honest about what it omits (~18KB → ~7KB on the
  common no-search call); `search` still matches against the **full** template
  record, so a model name or a detail past the clip stays findable.

### Not changed (reported, but not defects)

- *Bundled templates reference model files by flat filename while the user's
  models live in subfolders.* ComfyUI fails on this identically — the template
  isn't wrong about anything draftsman can fix. `validate` already reports the
  installed path as a closest-match `suggestion`; what was missing was any way
  to apply it, which is the subgraph-remedy fix above.
- *`StringConcatenate` delimiter double-escaping.* Nothing in the write path
  escapes string widget values — `set_widget` stores exactly what it is given
  and `json.dump` writes it — and a real newline **is** `\n` in a JSON file.
  Auto-unescaping was considered and rejected: it would corrupt Windows paths
  and any prompt containing a literal backslash.
- *`look_at` timeouts and Playwright sidebar behavior.* Different MCP servers.

## 0.10.0 — PrimitiveNode authoring, socket typing, data outputs

From a live session building a character-cycling workflow (report kept in
project memory, not the repo) that hit three walls, plus one more found in the
repo audit that followed. All four are behavior changes, hence minor.

### Fixed

- **`connect` no longer treats `COMBO` as a wildcard.** A STRING output wired
  into a converted combo widget used to pass `validate_workflow` clean and then
  get rejected by ComfyUI's own executor at queue time
  (`return_type_mismatch`) — which queues the prompt anyway and runs only the
  rest of the graph, so the failure was silent. `connect` now checks every
  wire with `model.types_compatible` (shared with the new `validate` check
  below) and refuses a mismatch unless `"force": true`. `validate_workflow`
  gained the matching `link-type-mismatch` (error) check, so the same mistake
  arriving via `import_workflow` is no longer invisible until the failed run.
- **`PrimitiveNode` and `Reroute` are now authorable.** Both were already
  recognized as virtual nodes (`to_api` inlined a primitive's value and traced
  through reroutes), but `add_node`'s installed-class gate only special-cased
  `Note`/`MarkdownNote`, so adding either failed with "not installed on this
  instance." This blocked the correct ComfyUI idiom for "a value that adapts
  to and can cycle whatever it's wired to" — a dropdown, a checkpoint, a LoRA
  — forcing a strictly worse same-type-only workaround. A primitive is
  typeless until `connect`ed, then mirrors its target's type (COMBO included)
  the way the frontend does on first connection; `set_widget` addresses it by
  the mirrored widget's name or the alias `"value"`, plus
  `"control_after_generate"`; `apply_seed_control` (used by
  `run_workflow(roll_seeds=True)`) now rolls a primitive's control mode too —
  this is the only way a headless caller can cycle a COMBO across runs, since
  `control_after_generate` is browser JS the raw `/prompt` API never applies.
  `validate` checks a primitive's value against every widget it drives (the
  one place anything does, since its consumer's own widget check is skipped
  because that slot is connected); `organize_workflow` places it as a green
  Inputs-band knob and keeps a Reroute glued to its source as a layout
  companion.
- **`run_workflow`/`get_run_status` surface non-file output values.** Only the
  four FILE keys (`images`/`gifs`/`videos`/`audio`) were ever harvested from a
  finished job's history, so a custom node's other return values — a save
  node's `filenames`/`path`/`saved_count`, a ShowText-style node's `text` —
  were silently dropped. Both tools now include `data_outputs` (per node id,
  clipped and budgeted, omitted entirely when empty) alongside the unchanged
  file `outputs`.
- **`OutputSlot` no longer drops a primitive's widget marker on save.** A real
  `PrimitiveNode` output serializes as `{"widget": {"name": "steps"}, ...}`
  (confirmed against the bundled `sdxl_simple_example.json` fixture, which
  itself uses primitives); `OutputSlot` had no field for it, so every
  save/load round-trip of a workflow using primitives silently lost what each
  one mirrored — including that bundled template.

### Also

- `edit_workflow`'s docstring collapsed five near-duplicate
  `*_in_definition` op lines into one rule (`_check_op` already names an op's
  required keys on error, so the schema doesn't need repeating six times).
  Net tool-schema cost across all 29 tools went down (~19.6k vs ~20.0k chars)
  despite the new capability.

## 0.9.0 — Token efficiency

A second audit, asking specifically whether the server is token-efficient to
use. The architecture already had good instincts — the `edit_workflow` compact
delta, combo capping in `get_node_info`, digested safetensors metadata — but
five paths bypassed that discipline, the worst able to return ~30k tokens from a
single call. Every number below was measured by running the code.

Minor rather than patch: several tool outputs change shape (capped lists, one
collapsed finding instead of many, a corrected `ok`).

### Fixed — token leaks

- **`lint` no longer reports overlap pairwise.** The check is O(n²) and emitted
  one finding *per overlapping pair*. Because `add_node` leaves every node at the
  default position until `organize_workflow` runs, anything built through
  `edit_workflow` is N-way self-overlapping — so the report was quadratic in the
  node count while saying one thing: "not laid out yet". Now a single finding
  naming the nodes, with the full id list in `node_ids`.

  | graph | before | after |
  |---|---|---|
  | 20-node mid-build | 292 findings, 27,108 chars (~6,777 tok) | 103 findings, ~2,884 tok (capped: ~1,120) |
  | 52-node messy | 1,428 findings, 122,754 chars (~30,688 tok) | 103 findings, ~2,832 tok |

- **`save_workflow`, `organize_workflow` and `port_workflow` now cap their
  findings.** `_cap_findings` existed for exactly this and was applied by
  `validate_workflow`/`diagnose_workflow` only; the other three returned
  everything. `save_workflow` measured **~35,000 tok → ~5,400 tok** on a messy
  graph. Advisory lint gets its own `_cap_lint` (lint findings carry no severity
  and never block, so a straight cap is right).
- **`list_models` is capped at 60 files** with the true `count`, a `truncated`
  count and a `search=` hint. It served the same filenames an `object_info` combo
  does — and those are capped at 24 — so an instance with 400 LoRAs returned
  ~5,283 tok per call, against the repo's own documented rule.
- **`search_nodes(detail=True)` folds a schema into the top 8 hits only**, and
  says so. At the default `limit=25` it measured ~13,068 chars (~3,267 tok) on
  the *trimmed* test fixture; a real instance is larger.
- **A disabled branch is one finding, not one per node.** The `node-disabled`
  note added in 0.8.0 repeated ~174 chars per node — the per-item repetition this
  project explicitly forbids. Now one note listing the ids (~64 tok flat,
  whatever the count), with every id in `node_ids`.

### Fixed — bugs

- **`_cap_findings` could return MORE than it was given, and lie about it.** When
  errors alone filled the cap, nothing was dropped but the truncation marker was
  appended anyway: 88 findings in → 89 out, ending with *"…0 more finding(s)
  omitted"*. The marker is now only added when something was actually trimmed.
- **`diagnose_workflow`'s `ok` flipped on an informational note.** It used
  `not findings`, so *any* finding — including a purely informational one —
  reported the workflow as broken. A clean, fully-wired graph with one muted
  PreviewImage returned `ok: false` while `validate_workflow` correctly returned
  `ok: true`. Now both use the same errors-only predicate. Pre-existing (the
  `subgraph-instance` info could already trigger it), but 0.8.0's `node-disabled`
  made it fire on any graph with a disabled node.
- **`lint` no longer contradicts `validate` about disabled nodes.** 0.8.0 taught
  `validate` to skip muted/bypassed nodes but left `lint` reporting
  `unconnected-input`/`orphan-node` on them, so the two tools disagreed about the
  same graph and `save_workflow`'s *"lint is not clean"* nag fired over a
  deliberately muted branch.

### Audited, not changed

- **Tool descriptions stay as they are.** The 29 tool schemas cost ~5,673 tokens
  of fixed context per session (plus ~221 for the server instructions).
  `edit_workflow` (1,813 chars) and `run_workflow` (1,676) dominate, but
  `edit_workflow`'s bulk *is* the op-schema list an agent needs to call it
  correctly at all — a misused tool costs far more than the prose saves. Measured
  and left alone deliberately.
- **`add_node` still places every node at `[0, 0]`.** This is the root cause of
  the overlap noise above, but `organize_workflow` overwrites positions anyway,
  so fixing the *reporting* was the right-sized change. Noted so it is not
  rediscovered as a bug.
- **Errors are still never dropped by the cap**, by design — a graph with 88
  missing node classes returns 88 errors. Hiding actionable errors to save tokens
  would be the wrong trade.

### Dev

- New `tests/test_round18_tokens.py` (13 tests) asserting measured *ceilings* on
  the payloads most likely to blow up, so an unbounded list fails in CI rather
  than in someone's conversation. Ceilings are ~2x current size: regression
  guards, not golden masters.

## 0.8.0 — Full repo audit remediation

A second full-repo audit, this time reproducing each defect by running the code
before fixing it. Two of these were silent: one corrupted a saved workflow, the
other refused to run a perfectly good one. Minor rather than patch because
validation behavior changes and relocation gains a capability.

### Fixed

- **`set_widget` no longer destroys values on pack nodes with custom JS-widget
  inputs.** `socket_names` — the per-instance signal that tells a pack's
  JS-rendered widget (LoraManager's `AUTOCOMPLETE_TEXT_LORAS`,
  StyleStringInjector2's `ZIPN_STYLE_GALLERY_BUTTON`) apart from a connection
  socket — was threaded into `to_api`, `apply_seed_control` and `validate`, but
  **not** into the write path. So the slot walk missed the custom widget and
  `named_to_widgets` rebuilt a shorter array: setting *any* widget on such a
  node dropped the pack's value and shifted the one being set into its slot.
  Observed on a LoraManager-shaped node: `['<lora:foo:0.8>', 1.0]` →
  `set_widget('strength', 2.0)` → `[2.0]`, with `to_api` then submitting
  `lora_text=2.0`. `set_widget`, `get_widget`, `check_widget_value`,
  `named_to_widgets` and subgraph widget promotion now all carry the node's real
  socket set. Schema-only paths (fresh-node defaults, `add_node`) stay
  conservative as before.
- **Muted and bypassed nodes no longer block runs and saves.** `to_api` drops
  mode-2/mode-4 nodes, but `validate` walked them anyway — so a muted branch (the
  standard way to disable one) produced `unconnected-input` **errors**, and
  `run_workflow` answered `status: invalid` / `save_workflow` refused, for a graph
  whose prompt document didn't contain those nodes at all. Disabled nodes are now
  reported once as `info` (`node-disabled`) and otherwise skipped.
- **A bypassed dead-end is now caught (new `dead-input-source` error).** Bypass is
  a passthrough, so a bypassed node with its own input unconnected forwards a
  hole: `to_api` silently dropped the *consumer's* input and ComfyUI rejected the
  prompt with nothing in validate pointing at it. The consumer now reports it,
  naming the input — the bypass counterpart to `muted-input-source`.
- **Video and audio renders can finally be handed to the caller.**
  `save_output` and `run_workflow(save_dir=...)` filtered outputs to
  `kind == "images"`, so a Wan / LTX / AnimateDiff result was permanently stuck
  in ComfyUI's output tree and `save_output(prompt_id=...)` reported *"produced
  no output images"* for a job that had produced a video. All kinds now relocate;
  the inline preview stays image-only (it needs a decodable still).
- **`run_workflow(save_dir=..., wait=False)` no longer silently ignores
  `save_dir`.** It created the destination directory and then discarded it — the
  response now carries `save_dir_ignored` naming the exact follow-up
  `save_output(prompt_id=...)` call.
- **The Comfy Registry degrades instead of crashing the tool.** A transport
  failure raised a raw `httpx.ConnectError` out of `diagnose_workflow`, throwing
  away the local validation findings it had already computed — on an offline or
  firewalled host, which is a normal state for a local-first tool. It now raises
  a named `RegistryUnavailableError` that the tools convert into a structured
  `{error, hint}`, with the local findings intact.
- **Import failures are actionable.** `import_workflow` leaked
  `JSONDecodeError`, `KeyError: 'type'` and `invalid literal for int()` to the
  caller; it now returns the standard `{error, hint}` shape, and `from_ui` names
  exactly which node entry is malformed.
- **API prompts with non-numeric node ids import instead of crashing.**
  `from_api` did `int(node_id)` unconditionally. Non-numeric keys are now mapped
  to fresh ids (original preserved in `properties['api_node_id']`), with
  connections remapped to match.
- **Session writes are atomic.** `persist` runs on every seed roll, so an
  interrupted write was not hypothetical — and a half-written file made that
  workflow id permanently unloadable behind an opaque decoder traceback. Writes
  now go through a temp file + `os.replace`, and an unreadable file reads back as
  a named `KeyError` explaining the fix.
- **`organize_workflow` clears a stale "touch me" highlight.** `_paint_knobs`
  only ever added green, so a prompt node that later got wired from upstream kept
  telling the reader to type into a box they can't type into. Only draftsman's own
  swatch is cleared; a colour a human picked is left alone.

### Changed

- **`detect_family` builds its knowledge index once per call** instead of once
  per model reference — it globs the learned dir and parses every YAML in it, and
  `find_workflow` calls it for each of up to 400 saved workflows.
- **Registry lookups run concurrently** (bounded at 6): a diagnose on a workflow
  with a dozen missing classes was a dozen serial round trips to a remote host.
- **Mount readiness is cached against the configured path.**
  `get_instance_info`, `check_setup` and the capabilities resource each wrote and
  deleted a probe file on every call. `check_setup` still re-probes — it is the
  tool you run precisely because something on disk changed.
- **The model `folder` path segment is URL-quoted** in `list_models` /
  `get_model_metadata`, matching every other path in the client.

### Docs

- `docs/PERMISSIONS.md` listed 16 read-only tools; the server annotates 18 —
  `check_setup` and `find_workflow` were missing from the allowlist.
- New ARCHITECTURE gotchas for the `socket_names` invariant, disabled-node
  validation, and relocation scope.

### Dev

- New `tests/test_round17_audit.py` (25 tests) pinning each fix, plus video/audio
  relocation and background-`save_dir` coverage in `test_output_relocation.py`.
- CI now builds the wheel and imports the knowledge data from it (nothing
  verified that `knowledge/families/*.yaml` actually shipped), and installs with
  `--frozen` so the committed `uv.lock` is validated.

### Audited, not changed

- The lazy `_State` accessors in `server.py` are **not** subject to a
  construct-twice race: they are synchronous, and a sync body cannot be preempted
  mid-way by the event loop. An invariant comment now records that this is load
  bearing — adding an `await` inside one would introduce the race.

## 0.7.2 — Cycle-safe auto-layout

### Fixed

- **Auto-layout no longer piles cyclic nodes at the far-left edge.** The layout
  ranker (`graph/layout._ranks`) used a plain topological sort, so a workflow
  containing a feedback edge (rare — an unusual custom node that loops output
  back to an upstream input) left every node in the cycle at rank 0, stacked and
  overlapping. The ranker now breaks a cycle by releasing its most-resolved node
  and continues, spreading the nodes across columns. Acyclic workflows — every
  normal image pipeline — hit an identical code path and are byte-for-byte
  unchanged (verified: the existing left-to-right, no-overlap, real-template, and
  determinism layout tests all still pass).

## 0.7.1 — Repo audit remediation

Correctness and hygiene fixes surfaced by a full-repo audit. No new tools or
behavior changes to a well-formed workflow; the fixes turn three latent failure
modes into early, actionable errors (or preserved data) and tighten the
developer tooling.

### Fixed

- **API-format import no longer misreads a 2-element list widget value as a
  wire.** `Workflow.from_api` decided "connection vs. literal value" purely by
  list length, so a widget value that happens to be a 2-element list (e.g. a
  coordinate pair `[512, 512]`) was dropped from the widgets *and* turned into a
  bogus connection — and a non-numeric first element (`["a", "b"]`) crashed the
  whole import on `int(...)`. It now treats a value as a link only when it is
  `[existing_node_id, integer_slot]`, preserving genuine list-valued widgets.
  Scope is limited to API-format import (the `name=`/UI-format path is
  unaffected).
- **Subgraph flatten no longer keeps a malformed inner link.** When an inner
  link's origin slot index exceeded the producing node's outputs,
  `graph/subgraph.py`'s expander recorded a "dropped" diagnostic but still
  created the link, leaving a dangling `[origin_id, bad_slot]` reference in the
  API prompt. It now drops the link (matching the target-slot-out-of-range path)
  so the API document stays consistent.
- **A required input fed by a muted node is caught before the run.** A `MODE_MUTE`
  producer is skipped at execution and isn't traced through (unlike Reroute /
  bypass), so a consumer's input serialized to a reference ComfyUI rejects — yet
  `validate` reported the input as "connected" and passed. `validate` now emits a
  `muted-input-source` error naming the muted node, so the failure is early and
  actionable instead of a confusing run-time error on a graph that "validated".

### Changed

- **Graceful shutdown.** The server now closes its lazily-created ComfyUI and
  Comfy Registry HTTP clients and stops the progress-tracker websocket loop on
  exit (via a FastMCP `lifespan`), instead of leaking "unclosed client session"
  warnings.
- **`COMFY_API_KEY` moved onto `Config`.** It was the one setting read at import
  time rather than through `load_config()`; centralizing it makes configuration
  uniform and testable.

### Dev

- **Committed `uv.lock`** for reproducible dev/CI installs (unpinned transitive
  dependencies before).
- **Added `mypy` to the dev group and CI** (default mode over `src`, plus
  `types-PyYAML`); the package now type-checks clean.

Asked to "make an image", an agent almost always builds from scratch even when a saved workflow already does exactly that — because `list_workflows` returns only filenames, so checking for a match means importing and inspecting each one, and it isn't worth the tokens. This round adds a discovery tool that does the matching server-side and hands back only a few compact, ranked candidates.

### Added

- **`find_workflow(intent)` tool** — describe the goal in words ("flux portrait at 1024 with a face detailer") and get back a few **ranked** matches, each a compact profile: model family, base model, resolution, feature tags (`lora` / `controlnet` / `detailer` / `upscale` / `inpaint` / `img2img` / …), and *why* it matched. Profiles are extracted straight from the saved JSON, so **hand-built workflows are covered too** — not just ones this tool authored. It returns summaries only, never full graphs; `import_workflow(name=...)` loads the one you pick. This keeps the token cost bounded no matter how large the workflow library is: the fetch-and-parse of every saved file happens server-side, and only the top handful of small profiles cross back to the agent.

### Notes

- Reuses the existing family detector, widget-name mapping, and userdata client (which already guards against path traversal); no new configuration. Saved files are fetched concurrently and each is profiled defensively — an unreadable or malformed file is skipped and reported in a `skipped` count rather than failing the call.
- **Version bump** — `0.6.0` → `0.7.0`.

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
