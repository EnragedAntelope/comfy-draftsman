"""comfy-draftsman MCP server: tools, prompts, and resources.

All heavy lifting lives in tested modules (graph/, comfy/, knowledge/); this
file is thin wiring. State: one ComfyClient + RegistryClient + Session per
process, created lazily.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import difflib
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import knowledge
from .comfy.catalog import metadata_digest, node_summary
from .comfy.catalog import search_nodes as catalog_search
from .comfy.client import ComfyClient, ComfyConnectionError, ComfyValidationError
from .comfy.progress import ProgressTracker
from .comfy.registry import RegistryClient, RegistryUnavailableError
from .config import Config, load_config
from .graph.annotate import annotate
from .graph.lint import lint
from .graph.model import NOTE_TYPES, PRIMITIVE_TYPE, VIRTUAL_TYPES, Workflow
from .graph.port import port_workflow as port_engine
from .graph.spend import api_nodes
from .graph.validate import check_primitive_value, check_widget_value, validate
from .graph.widgets import SYNTHETIC_SUFFIXES, all_slot_names, widgets_to_named
from .imaging import downscale_image
from .session import Session

# Tool annotations let clients reason about safety and, where supported,
# auto-approve safe calls. Read tools that query the live instance are
# read-only + open-world; session-local reads are read-only + closed-world.
# See docs/PERMISSIONS.md for the recommended Claude Code allowlist.
_READ_INSTANCE = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)
_READ_LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True)
_EDIT_LOCAL = ToolAnnotations(readOnlyHint=False, openWorldHint=False, destructiveHint=False)
_WRITE_INSTANCE = ToolAnnotations(readOnlyHint=False, openWorldHint=True, destructiveHint=False)
_DESTRUCTIVE_INSTANCE = ToolAnnotations(readOnlyHint=False, openWorldHint=True, destructiveHint=True)

class _State:
    config: Config | None = None
    client: ComfyClient | None = None
    registry: RegistryClient | None = None
    session: Session | None = None
    tracker: ProgressTracker | None = None
    # prompt_id -> workflow_id for prompts THIS process queued via run_workflow.
    # A live session reported the queue as unattributable - a run_workflow
    # timeout was ambiguous between "queued behind the user's own jobs" and
    # "actually hung" - because manage_queue/get_run_status only ever had the raw
    # ComfyUI prompt_id to go on. Bounded (_SUBMITTED_CAP) and best-effort: a
    # server restart, or a prompt queued from the ComfyUI UI directly, is simply
    # unattributed, not an error.
    submitted: ClassVar[dict[str, str]] = {}
    # /system_stats devices, cached for the process lifetime. VRAM *total* is
    # what drives the fit verdict and cannot change while ComfyUI is running, so
    # this is one HTTP call per session rather than one per guidance lookup.
    # run_workflow's preflight refreshes it, because free VRAM does change.
    devices: list[dict[str, Any]] | None = None


# INVARIANT: the lazy accessors below (_config/_client/_registry/_session/
# _tracker) must stay SYNCHRONOUS. FastMCP dispatches tool calls concurrently on
# one event loop, and a sync function body cannot be preempted mid-way - that is
# the only reason "check None, then construct" is safe here without a lock. Add
# an `await` inside one of them and two racing cold calls could each build a
# client, with _lifespan closing only whichever landed in _State and leaking the
# other's connection pool. Construct eagerly-but-synchronously; await elsewhere.


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Close the lazily-created HTTP clients and stop the progress tracker on
    shutdown, so a clean exit doesn't leak 'unclosed client session' warnings or
    leave the websocket reconnect loop running."""
    try:
        yield
    finally:
        if _State.tracker is not None:
            with contextlib.suppress(Exception):
                await _State.tracker.stop()
        for handle in (_State.client, _State.registry):
            if handle is not None:
                with contextlib.suppress(Exception):
                    await handle.close()


mcp = FastMCP(
    "comfy-draftsman",
    lifespan=_lifespan,
    instructions=(
        "Draft, repair, port, validate, run, and SAVE ComfyUI workflows against the "
        "user's local ComfyUI instance. The finished artifact is always an organized, "
        "labeled workflow: run organize_workflow before save_workflow. Ground truth is "
        "the live instance (search_nodes/get_node_info/list_models); templates "
        "(list_templates) are the best starting points for current models. Batch "
        "get_node_info lookups (it takes a list) in ONE call. For a workflow already "
        "saved in ComfyUI, use list_workflows then import_workflow(name=...) - never "
        "ask for pasted JSON. get_model_guidance has tuned per-family settings; "
        "persist anything you research with record_learning. A GENERATED positive "
        "prompt (wildcards/concatenators) should pass through a Show Text node before "
        "the encoder (lint: no-prompt-preview). When modernizing, spell out any "
        "capability that would be LOST before dropping nodes - never silently."
    ),
)


def _config() -> Config:
    if _State.config is None:
        _State.config = load_config()
    return _State.config


def _client() -> ComfyClient:
    if _State.client is None:
        _State.client = ComfyClient(_config())
    return _State.client


def _registry() -> RegistryClient:
    if _State.registry is None:
        _State.registry = RegistryClient(_config())
    return _State.registry


def _session() -> Session:
    if _State.session is None:
        _State.session = Session(_config().session_dir)
    return _State.session


def _tracker() -> ProgressTracker:
    if _State.tracker is None:
        _State.tracker = ProgressTracker(_client()._ws_url)
    return _State.tracker


async def _load_devices(refresh: bool = False) -> list[dict[str, Any]]:
    """Cached device list from /system_stats (see _State.devices).

    Async on purpose: the INVARIANT above keeps the lazy accessors synchronous,
    so this must not become one. Two racing cold calls can both fetch, which is
    harmless - the result is idempotent and holds no resource."""
    if _State.devices is None or refresh:
        stats = await _client().get_system_stats()
        _State.devices = list(stats.get("devices") or [])
    return _State.devices


def _fit(guidance: dict[str, Any], live: bool = False) -> dict[str, Any] | None:
    """knowledge.fit_verdict against the cached devices - None (emit nothing)
    whenever the devices haven't been loaded yet.

    ``live=False`` strips vram_free before comparing. The cache is only sound
    for VRAM *total*, which cannot change while ComfyUI is running; a free-VRAM
    figure snapshotted during someone else's render would otherwise still be
    reporting "only 1GB of your 24GB is free" hours later. Only a caller that
    just re-read /system_stats (run_workflow's preflight) passes live=True.
    """
    devices = _State.devices or []
    if not live:
        devices = [{k: v for k, v in d.items() if k != "vram_free"} for d in devices]
    return knowledge.fit_verdict(guidance, devices)


class _Confirmation(BaseModel):
    """Elicitation form for anything irreversible or billable. One boolean:
    MCP elicitation only allows primitive fields, and anything richer would be
    a decision the user has already been asked to make in prose.

    REQUIRED, deliberately - no default. An optional field is one a client may
    omit from its form entirely, and then a user who pressed Accept comes back
    as ``confirm: false`` and is told they declined, which is the opposite of
    what they chose. Required means the client has to collect an answer; a
    response missing it fails validation and lands on the cannot-ask path,
    which is honest about not knowing rather than inventing a refusal.
    """

    confirm: bool = Field(description="Yes, go ahead")


async def _confirm(
    ctx: Context | None, message: str, fallback_hint: str | None, prefix: str
) -> dict[str, Any] | None:
    """Ask the user before doing something irreversible. None = go ahead.

    Three-way degrade, because elicitation support varies by client and the
    no-capability path is the common case on some of them:

    1. client elicits and the user accepts -> None, the caller proceeds
    2. client elicits and the user declines -> ``{prefix}_declined``
    3. client cannot elicit at all -> ``{prefix}_confirmation_required`` carrying
       ``fallback_hint``, the instructions for re-running once the user really
       has agreed - or, when ``fallback_hint`` is None, simply proceed.

    That last choice is per call site, not a global policy. Spending the user's
    money and discarding someone else's queued render both refuse by default:
    the cost of a wrong "yes" is unrecoverable. save_workflow(overwrite=True)
    proceeds, because the caller already passed an explicit destructive flag and
    refusing would make the feature unusable on every non-eliciting client.

    Cases 2 and 3 are normal returns, not exceptions - same shape as the
    queue_busy result, because "nothing happened, here is why" is an outcome
    the calling agent has to read and act on, not an error to retry.
    """
    if ctx is not None:
        try:
            answer = await ctx.elicit(message=message, schema=_Confirmation)
        except Exception:
            answer = None  # client has no elicitation capability
        if answer is not None:
            if answer.action == "accept" and getattr(answer.data, "confirm", False):
                return None
            return {
                "status": f"{prefix}_declined",
                "hint": "the user did not confirm - nothing was done. Ask what they "
                "want instead; do not re-issue the same call.",
            }
    if fallback_hint is None:
        return None
    return {"status": f"{prefix}_confirmation_required", "hint": fallback_hint}


_SPEND_HINT = (
    "NOTHING WAS QUEUED. This graph contains partner/API nodes, which run on the "
    "provider's hardware and charge the user's Comfy Org account for every submit - "
    "a failed or unwanted render still costs. Show them the api_nodes list, get an "
    "explicit yes, and only then re-run with confirm_spend=True. confirm_spend "
    "authorizes THIS submit, not a series of retries."
)

_API_NODES_CAP = 10


def _spend_payload(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """The api_nodes list, bounded - a graph with thirty partner nodes must not
    return thirty rows just to say "this costs money"."""
    payload: dict[str, Any] = {"api_nodes": nodes[:_API_NODES_CAP]}
    if len(nodes) > _API_NODES_CAP:
        payload["api_nodes_truncated"] = len(nodes) - _API_NODES_CAP
    return payload


async def _capacity(wf: Workflow, object_info: dict[str, Any]) -> dict[str, Any] | None:
    """Pre-render fit verdict for the workflow's detected family, or None.

    Best-effort in every direction: an undetectable family, an unreachable
    instance, or a comfortable fit all return None. It NEVER blocks a run -
    draftsman gates only on validate() errors, and a curated VRAM floor is not
    authoritative enough to refuse someone their own hardware."""
    with contextlib.suppress(Exception):
        family = knowledge.detect_family(wf, object_info, learned_dir=_config().learned_dir)
        if not family:
            return None
        # refresh: unlike get_model_guidance, this cares about FREE VRAM too,
        # and that changes with every job the instance runs
        await _load_devices(refresh=True)
        guidance = knowledge.get_guidance(family, learned_dir=_config().learned_dir)
        verdict = _fit(guidance, live=True)
        if verdict:
            return {"family": family, **verdict}
    return None


def _check_output_ref(filename: str, subfolder: str) -> str | None:
    """Refuse refs that could escape ComfyUI's output/input/temp dirs."""
    for part in (filename, subfolder):
        clean = part.replace("\\", "/")
        if clean.startswith("/") or ".." in clean.split("/"):
            return f"invalid path component: {part!r}"
    return None


async def _object_info(refresh: bool = False) -> dict[str, Any]:
    return await _client().get_object_info(refresh=refresh)


def _wf(workflow_id: str) -> Workflow:
    return _session().get(workflow_id)


def _find_group(wf: Workflow, group_id: int):
    for g in wf.groups:
        if g.id == group_id:
            return g
    # ValueError, not KeyError: the batch's KeyError handler says "unknown
    # node id", which would misname a missing GROUP id
    raise ValueError(f"no group #{group_id}; available: {[g.id for g in wf.groups]}")


def _clip(v: Any) -> Any:
    return v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v


_LEVEL_RANK = {"error": 0, "warning": 1, "info": 2}
_FINDINGS_CAP = 40

# How many prompt_id -> workflow_id mappings to remember (see _State.submitted).
# A handful of in-flight/recent runs is all manage_queue/get_run_status ever need
# to attribute; older entries are dropped oldest-first.
_SUBMITTED_CAP = 200


def _record_submission(prompt_id: str, workflow_id: str) -> None:
    if len(_State.submitted) >= _SUBMITTED_CAP:
        _State.submitted.pop(next(iter(_State.submitted)))
    _State.submitted[prompt_id] = workflow_id


def _workflow_tag(prompt_id: str) -> dict[str, str]:
    """`{"workflow_id": ...}` if THIS process queued prompt_id, else `{}` - splat
    into a result dict so an unattributed prompt (someone else's job, or queued
    before this server started) costs nothing extra."""
    workflow_id = _State.submitted.get(prompt_id)
    return {"workflow_id": workflow_id} if workflow_id else {}

# search_nodes(detail=True) folds a full node schema into each hit; only the top
# few get one, or a default-limit search becomes a multi-thousand-token response.
_DETAIL_SCHEMA_CAP = 8

# list_models serves the same filenames a combo does, and object_info combos are
# capped at 24 (catalog.MAX_COMBO_CHOICES) for exactly this reason - an instance
# with hundreds of LoRAs would otherwise return the lot on every call.
_MODEL_FILES_CAP = 60

# The bundled template catalog is ~450 entries and growing (452 on ComfyUI
# 0.29 / comfyui_workflow_templates 0.11.2); an unsearched list_templates was
# returning 60 of them - ~18KB - while silently dropping the other ~390, so a
# caller who found no match concluded none existed. Capped lower now that the
# response carries a true `count` and names `search=`, and the description clip
# is tighter: title + models identify a template, the description only has to
# disambiguate. ~18KB -> ~7KB on the common no-search call.
_TEMPLATES_CAP = 40
_TEMPLATE_DESC_CAP = 110

# Curated per-family download links (knowledge.matching_sources). A handful of
# hand-curated entries per family in practice, but this is user-writable via
# record_learning, so it gets the same bounded-list treatment as everything
# else a session can grow without limit.
_SOURCES_CAP = 12
_SOURCE_URL_CAP = 300


def _cap_sources(guidance: dict[str, Any]) -> dict[str, Any]:
    sources = guidance.get("sources")
    if not isinstance(sources, list):
        return guidance
    clipped = [
        {**s, "url": s["url"][:_SOURCE_URL_CAP] + "…" if len(s.get("url", "")) > _SOURCE_URL_CAP else s.get("url", "")}
        if isinstance(s, dict)
        else s
        for s in sources[:_SOURCES_CAP]
    ]
    guidance["sources"] = clipped
    if len(sources) > _SOURCES_CAP:
        guidance["sources_truncated"] = len(sources)
    return guidance


def _cap_lint(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap advisory lint output. Lint findings carry no severity (they never
    block), so this is a straight cap - but it must exist: lint is returned by
    organize_workflow and save_workflow, and an unorganized graph can produce
    hundreds of findings."""
    if len(warnings) <= _FINDINGS_CAP:
        return warnings
    return [
        *warnings[:_FINDINGS_CAP],
        {
            "code": "lint-truncated",
            "message": f"…{len(warnings) - _FINDINGS_CAP} more lint finding(s) "
            "omitted; organize_workflow usually clears most of them at once",
        },
    ]


def _cap_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort findings most-severe first and cap the number returned to the model,
    appending a marker if truncated. Every error is always kept - only lower
    levels are trimmed - so nothing blocking is hidden and tokens stay bounded."""
    ordered = sorted(findings, key=lambda f: _LEVEL_RANK.get(str(f.get("level")), 3))
    if len(ordered) <= _FINDINGS_CAP:
        return ordered
    errors = [f for f in ordered if f.get("level") == "error"]
    keep = max(_FINDINGS_CAP, len(errors))  # never drop an error to make room
    capped = ordered[:keep]
    omitted = len(ordered) - keep
    if not omitted:
        # errors alone filled the cap, so nothing was actually dropped. Appending
        # the marker anyway made the "capped" list LONGER than its input and
        # announced "…0 more finding(s) omitted".
        return capped
    capped.append(
        {
            "level": "info",
            "code": "findings-truncated",
            "message": f"…{omitted} more finding(s) omitted; fix "
            "the ones above first, then re-validate",
        }
    )
    return capped


def _similar_class_hint(name: str, object_info: dict[str, Any]) -> str:
    """`get_node_info` naming an installed lookalike saves the round-trip through
    search_nodes a live session needed for "SigmasRescale" (not installed) vs
    the real "Sigmas Rescale" (RES4LYF, with a space) - a plain substring search
    doesn't bridge that gap either, since neither name contains the other."""
    close = difflib.get_close_matches(name.lower(), [c.lower() for c in object_info], n=3, cutoff=0.6)
    if not close:
        return ""
    by_lower = {c.lower(): c for c in object_info}
    names = [by_lower[c] for c in close if c in by_lower]
    return "; installed classes with a similar name: " + ", ".join(names)


def _subgraph_edit_hint(findings: list[dict[str, Any]]) -> dict[str, str]:
    """A single `subgraph_edit` sentence when any finding sits on an editable
    inner node - `{}` otherwise, so it costs nothing on a flat graph.

    Stated once per RESULT, never per finding: the findings themselves already
    carry `definition_id` + `inner_node_id` as structured fields, and repeating
    the how-to on each is exactly the per-item repetition this server treats as a
    bug. Findings inside a subgraph are common (one wrong model path in a bundled
    template produces several), and until round 20 they all ended in "edit_workflow
    can't reach inside; rebuild flat" - which is false, and cost a live session a
    hand-rebuild of a 14-node graph."""
    if not any(f.get("definition_id") for f in findings):
        return {}
    return {
        "subgraph_edit": (
            "Findings carrying definition_id + inner_node_id are fixable in place: "
            "pass those to edit_workflow's definition-scoped ops "
            "(set_widget_in_definition, connect_in_definition, "
            "add_node_to_definition, remove_node_from_definition, "
            "set_title_in_definition, set_mode_in_definition). No rebuild needed."
        )
    }


def _widget_preview(n) -> Any:
    # prompts/wildcards/note text can be multi-KB and summaries are re-sent on
    # every inspect/edit - truncate the preview, the graph content is intact
    # (export_workflow_json shows full values)
    if isinstance(n.widgets_values, list):
        return [_clip(v) for v in n.widgets_values]
    return {k: _clip(v) for k, v in dict(n.widgets_values).items()}


def _subgraph_summary(sg: dict[str, Any], wf: Workflow | None = None) -> dict[str, Any]:
    """Readable view of one subgraph definition: inner nodes with widget
    previews, and inner wiring - enough to edit it in place or rebuild it flat.

    A definition's boundary inputs are NOT all reachable from the parent graph:
    an instance node exposes only some of them as real sockets, and `connect`
    addresses instance sockets. Listing the definition's full input list
    unqualified read as "these are connectable" and sent a live session chasing
    a `value` socket the instance never exposed. Inputs are reported as
    `"name (internal)"` when no instance exposes them, which needs the parent
    workflow - omit `wf` and every input is reported unqualified."""
    nodes = {n["id"]: n for n in sg.get("nodes", []) or [] if "id" in n}
    exposed: set[str] | None = None
    if wf is not None:
        exposed = {
            slot.name
            for node in wf.nodes.values()
            if node.type == sg.get("id")
            for slot in node.inputs
        }

    def link_str(ln: Any) -> str | None:
        if isinstance(ln, dict):
            oid, oslot, tid, tslot = (
                ln.get("origin_id"), ln.get("origin_slot"),
                ln.get("target_id"), ln.get("target_slot"),
            )
        else:
            oid, oslot, tid, tslot = ln[1], ln[2], ln[3], ln[4]
        target = nodes.get(tid)
        tname = tslot
        if target:
            inputs = target.get("inputs", []) or []
            if isinstance(tslot, int) and tslot < len(inputs):
                tname = inputs[tslot].get("name", tslot)
        # -10/-20 are the subgraph's own input/output boundary pseudo-nodes
        left = f"#{oid}[{oslot}]" if oid in nodes else f"<subgraph input {oslot}>"
        right = f"#{tid}.{tname}" if tid in nodes else f"<subgraph output {tslot}>"
        return f"{left} -> {right}"

    return {
        "id": sg.get("id"),
        "name": sg.get("name"),
        "inputs": [
            name
            if exposed is None or name in exposed
            else f"{name} (internal)"
            for i in sg.get("inputs", []) or []
            if (name := i.get("name")) is not None
        ],
        "outputs": [o.get("name") for o in sg.get("outputs", []) or []],
        "nodes": [
            {
                "id": nid,
                "class_type": n.get("type"),
                "title": n.get("title"),
                "widgets": [_clip(v) for v in n.get("widgets_values") or []]
                if isinstance(n.get("widgets_values"), list)
                else n.get("widgets_values"),
            }
            for nid, n in sorted(nodes.items(), key=lambda kv: str(kv[0]))
        ],
        "links": [s for ln in sg.get("links", []) or [] if (s := link_str(ln))],
    }


def _summary(workflow_id: str, wf: Workflow) -> dict[str, Any]:
    subgraphs = wf.subgraph_defs()
    return {
        "workflow_id": workflow_id,
        "title": _session().title(workflow_id),
        "nodes": [
            {
                "id": n.id,
                "class_type": n.type,
                "title": n.title,
                "widgets": _widget_preview(n),
                # notes/reroutes/primitives are UI-only: kept in the graph and
                # saved, but never sent to /prompt
                **({"virtual": True} if n.type in VIRTUAL_TYPES else {}),
                **(
                    {"subgraph": subgraphs[n.type].get("name", n.type)}
                    if n.type in subgraphs
                    else {}
                ),
            }
            for n in sorted(wf.nodes.values(), key=lambda x: x.id)
        ],
        # keep summaries light: every create/edit/import re-sends this. Full
        # subgraph internals are folded in by inspect_workflow only.
        **(
            {
                "subgraphs": {
                    sid: f"{sg.get('name', sid)} ({len(sg.get('nodes', []) or [])} inner "
                    "nodes; runs flattened; inspect_workflow shows internals)"
                    for sid, sg in subgraphs.items()
                }
            }
            if subgraphs
            else {}
        ),
        "links": [
            f"#{ln.origin_id}[{ln.origin_slot}] -> #{ln.target_id}.{wf.nodes[ln.target_id].inputs[ln.target_slot].name}"
            for ln in sorted(wf.links.values(), key=lambda x: x.id)
            if ln.origin_id in wf.nodes
            and ln.target_id in wf.nodes
            and ln.target_slot < len(wf.nodes[ln.target_id].inputs)
            # Notes are the only virtual nodes with no wiring to show. Primitives
            # and Reroutes DO carry links, and hiding them made an authored
            # primitive look unconnected in the very summary used to verify it.
            and wf.nodes[ln.origin_id].type not in NOTE_TYPES
            and wf.nodes[ln.target_id].type not in NOTE_TYPES
        ],
        # "#N title" - the id is what set_group/remove_group address
        "groups": [f"#{g.id} {g.title}" for g in wf.groups],
    }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


@mcp.tool(annotations=_READ_INSTANCE)
async def get_instance_info() -> dict[str, Any]:
    """ComfyUI version, OS, VRAM, queue length, and render-relocation readiness of
    the connected instance. Call first. The `relocation` block reports whether
    COMFYUI_MOUNT_DIR is set and writable - if it isn't, renders can't be handed to
    the user automatically, so surface that to them before spending a render."""
    stats = await _client().get_system_stats()
    queue = await _client().get_queue()
    # seed the process-lifetime device cache while we're here - this is the
    # "call first" tool, so the fit verdict usually costs no extra HTTP call
    _State.devices = list(stats.get("devices") or [])
    devices = [
        {
            "name": d.get("name"),
            "vram_total": d.get("vram_total"),
            "vram_free": d.get("vram_free"),
            # raw bytes are what ComfyUI reports; GB is what a human (or an
            # agent comparing against a model's requirement) actually reasons in
            "vram_total_gb": knowledge.bytes_to_gb(d.get("vram_total")),
            "vram_free_gb": knowledge.bytes_to_gb(d.get("vram_free")),
        }
        for d in stats.get("devices", [])
    ]
    return {
        "url": _config().comfyui_url,
        "comfyui_version": stats.get("system", {}).get("comfyui_version"),
        "os": stats.get("system", {}).get("os"),
        "devices": devices,
        "queue_running": len(queue.get("queue_running", [])),
        "queue_pending": len(queue.get("queue_pending", [])),
        "relocation": _mount_status(),
        "knowledge_families": knowledge.list_families(_config().learned_dir),
    }


@mcp.tool(annotations=_READ_INSTANCE)
async def check_setup() -> dict[str, Any]:
    """One-shot setup diagnostic for a fresh install or a sandboxed client (Cowork/
    Desktop/Code): can I reach ComfyUI, can I hand finished renders back to the user
    (COMFYUI_MOUNT_DIR), is the partner-node key present. Unlike get_instance_info it
    never raises - a down instance is a failed check, not an error - so run it first
    when a render can't be delivered or the instance seems unreachable. Returns
    {ok, checks:[{name, ok, detail}], hint?}; `ok` gates on ComfyUI being reachable,
    relocation is a soft check surfaced via `hint`."""
    cfg = _config()
    checks: list[dict[str, Any]] = []

    # Hard requirement: can we talk to ComfyUI at all?
    try:
        stats = await _client().get_system_stats()
        version = stats.get("system", {}).get("comfyui_version") or "unknown"
        checks.append(
            {"name": "comfyui", "ok": True, "detail": f"reachable at {cfg.comfyui_url} (v{version})"}
        )
        reachable = True
    except ComfyConnectionError as e:
        checks.append({"name": "comfyui", "ok": False, "detail": str(e)})
        reachable = False
    except Exception as e:  # reached it, but it answered oddly - still actionable
        checks.append(
            {"name": "comfyui", "ok": False, "detail": f"error talking to {cfg.comfyui_url}: {e}"}
        )
        reachable = False

    # Soft requirement: can finished renders be relocated to a caller-reachable
    # folder? Re-probe rather than trusting the cache - this is the doctor tool,
    # run precisely when the operator has just changed something.
    reloc = _mount_status(recheck=True)
    checks.append(
        {
            "name": "relocation",
            "ok": bool(reloc.get("writable")),
            "detail": reloc.get("path") or reloc.get("error") or reloc.get("hint"),
        }
    )

    # Informational: partner/* nodes (Luma, Kling, Runway, ...) need COMFY_API_KEY
    checks.append(
        {
            "name": "partner_node_api_key",
            "ok": True,
            "detail": "set" if cfg.comfy_api_key else "unset (only needed for partner/* nodes)",
        }
    )

    result: dict[str, Any] = {"ok": reachable, "checks": checks}
    if not reachable:
        result["hint"] = "fix ComfyUI connectivity first - the other checks assume it's up"
    elif not reloc.get("writable"):
        result["hint"] = reloc.get("hint") or reloc.get("error")
    return result


@mcp.tool(annotations=_READ_INSTANCE)
async def search_nodes(
    query: str, category: str = "", limit: int = 25, detail: bool = False
) -> list[dict[str, Any]]:
    """Search node classes installed on the instance (name/display-name/description).

    Use category to narrow (e.g. 'loaders', 'conditioning', 'sampling', 'ImpactPack').
    Set detail=True to fold each hit's full input/output schema in-line (use a
    specific query + small limit) so you can skip the follow-up get_node_info.
    """
    object_info = await _object_info()
    results = catalog_search(object_info, query, category=category or None, limit=limit)
    if detail:
        # A folded schema is ~300-700 tokens each, so detail=True at the default
        # limit of 25 was a multi-thousand-token response. Fold into the top hits
        # only and say so - the rest still come back as normal search results.
        for hit in results[:_DETAIL_SCHEMA_CAP]:
            hit["schema"] = node_summary(object_info, hit["class_type"])
        if len(results) > _DETAIL_SCHEMA_CAP:
            results.append(
                {
                    "note": f"schemas folded into the first {_DETAIL_SCHEMA_CAP} hits "
                    f"only ({len(results) - _DETAIL_SCHEMA_CAP} more matched); narrow "
                    "the query, pass a smaller limit, or call get_node_info "
                    "(it batches: class_types=[...]) for specific classes"
                }
            )
    return results


@mcp.tool(annotations=_READ_INSTANCE)
async def get_node_info(
    class_type: str = "",
    class_types: list[str] | None = None,
    choices_filter: str = "",
    max_choices: int = 0,
) -> dict[str, Any]:
    """Full input/output schema for node classes: slot names, types, widget
    defaults/ranges, combo choices, tooltips.

    BATCH your lookups: pass class_types=["A", "B", "C"] to fetch many in ONE
    call (returns {class_type: schema}) instead of one call per node. A single
    class_type=... still returns that one node's schema directly.

    Long combo lists (fonts, model files...) are capped at 24 choices by
    default; to browse the rest, pass choices_filter='substring'
    (case-insensitive, applies to every combo of the node) and/or
    max_choices=N to raise the cap.
    """
    names = list(class_types or [])
    if class_type:
        names.insert(0, class_type)
    if not names:
        return {"error": "pass class_type=... or class_types=[...]"}
    object_info = await _object_info()
    results: dict[str, Any] = {}
    for name in names:
        if name in VIRTUAL_TYPES:
            results[name] = {
                "class_type": name,
                "virtual": True,
                "note": "UI-only display node, not in ComfyUI object_info",
            }
            continue
        try:
            results[name] = node_summary(
                object_info, name, choices_filter=choices_filter, max_choices=max_choices
            )
        except KeyError:
            results[name] = {
                "error": f"'{name}' is not installed on this instance",
                "hint": "resolve_missing_nodes can find which pack provides it"
                + _similar_class_hint(name, object_info),
            }
    if class_types is None and class_type:  # single-lookup back-compat shape
        return results[class_type]
    return results


@mcp.tool(annotations=_READ_INSTANCE)
async def list_models(
    folder: str = "checkpoints", search: str = "", metadata_for: str = ""
) -> dict[str, Any]:
    """Model files installed on the instance. `folder` picks the model type:
    checkpoints, loras, vae, diffusion_models, text_encoders, upscale_models,
    controlnet, embeddings, ... (unknown folder -> the full available list).
    `search` filters filenames (case-insensitive substring). `metadata_for`
    (a .safetensors filename from this folder) returns its embedded training
    metadata instead - base model + top trigger tags, key for using a LoRA."""
    folders = await _client().list_model_folders()
    if folder not in folders:
        return {"error": f"unknown folder '{folder}'", "available": folders}
    if metadata_for:
        try:
            meta = await _client().get_model_metadata(folder, metadata_for)
        except FileNotFoundError:
            return {
                "error": f"no embedded metadata for {metadata_for!r} in '{folder}' "
                "(file not found, not .safetensors, or trained without metadata)"
            }
        except ValueError as e:
            return {"error": str(e)}
        return {"folder": folder, "file": metadata_for, "metadata": metadata_digest(meta)}
    files = await _client().list_models(folder)
    if search:
        needle = search.lower()
        files = [f for f in files if needle in f.lower()]
    # cap like a combo list: `count` still reports the true total, so the agent
    # knows to narrow rather than assuming it has seen everything
    result: dict[str, Any] = {
        "folder": folder,
        "count": len(files),
        "files": files[:_MODEL_FILES_CAP],
    }
    if search:
        result["search"] = search
    if len(files) > _MODEL_FILES_CAP:
        result["truncated"] = len(files) - _MODEL_FILES_CAP
        result["hint"] = (
            f"showing {_MODEL_FILES_CAP} of {len(files)}; pass search='substring' "
            "to narrow (matching is case-insensitive on the filename)"
        )
    elif search and not files:
        # a live session lost time here: a workflow referenced a Chroma checkpoint
        # invisible to list_models(folder="checkpoints"), but ClownModelLoader's
        # own combo listed it fine - some loader nodes scan additional folders
        # (diffusion_models, unet, ...) beyond this type's standard listing, so an
        # empty result here does not mean the file is missing from the instance.
        result["hint"] = (
            f"no '{folder}' file matched {search!r} - if a specific loader node "
            "should see this file, get_node_info(class_type=<that loader>) shows "
            "its own combo, which can scan additional folders this listing doesn't"
        )
    return result


@mcp.tool(annotations=_READ_INSTANCE)
async def list_templates(search: str = "") -> dict[str, Any]:
    """ComfyUI's bundled workflow templates - the best starting points for current
    models (they ship with every release). Seed one via create_workflow(template=...).
    Narrow with search= (matched against title/description/models); the catalog is
    ~450 templates, far more than one response should carry."""
    index = await _client().get_template_index()
    needle = search.lower()
    out = []
    for module in index:
        for template in module.get("templates", []):
            entry = {
                "name": template.get("name"),
                "title": template.get("title"),
                "description": (template.get("description") or "")[:_TEMPLATE_DESC_CAP],
                "models": template.get("models", []),
                "category": module.get("title"),
            }
            # match on the FULL record, not the clipped entry: a model name or a
            # detail past the description clip is exactly what a caller searches
            # for, and dropping it from the haystack would make it unfindable
            if not needle or needle in json.dumps([entry, template]).lower():
                out.append(entry)
    shown = out[:_TEMPLATES_CAP]
    result: dict[str, Any] = {"count": len(out), "templates": shown}
    if len(out) > len(shown):
        # the previous bare `out[:60]` said nothing about the other ~390, so a
        # caller who saw no match reasonably concluded the catalog had none
        result["hint"] = (
            f"showing {len(shown)} of {len(out)} matches - narrow with "
            "search= (model family, modality, or template name)"
        )
    return result


# --------------------------------------------------------------------------
# Authoring
# --------------------------------------------------------------------------


@mcp.tool(annotations=_EDIT_LOCAL)
async def create_workflow(title: str, template: str = "") -> dict[str, Any]:
    """Start a workflow: blank, or seeded from a bundled template (recommended for
    current model families - see list_templates). Returns workflow_id + node summary."""
    if template:
        document = await _client().get_template_workflow(template)
        wf = Workflow.from_ui(document)
    else:
        wf = Workflow.new()
    workflow_id = _session().create(wf, title=title)
    return _summary(workflow_id, wf)


@mcp.tool(annotations=_READ_INSTANCE)
async def list_workflows(search: str = "") -> dict[str, Any]:
    """Workflows already saved in ComfyUI's workflow browser (userdata). Use a
    returned name with import_workflow(name=...) to load one WITHOUT pasting its
    JSON. `search` filters names (case-insensitive substring)."""
    names = [n[:-5] if n.endswith(".json") else n for n in await _client().list_userdata_workflows()]
    if search:
        needle = search.lower()
        names = [n for n in names if needle in n.lower()]
    result = {"count": len(names), "workflows": sorted(names)}
    if search:
        result["search"] = search
    return result


# --------------------------------------------------------------------------
# find_workflow: cheap "what do I already have that fits?" over saved workflows
# --------------------------------------------------------------------------
#
# list_workflows returns only names, so an agent that wants to REUSE a saved
# workflow would have to import+inspect each one - expensive enough that it just
# rebuilds from scratch. find_workflow does the fetch+parse SERVER-side and hands
# back only a handful of ranked, compact profiles (family, resolution, feature
# tags), so the token cost to the caller is bounded no matter how large the
# library is. It never returns full workflow JSON - import_workflow(name=...)
# loads the one the agent picks.

_FIND_CONCURRENCY = 8      # parallel userdata GETs; a big library shouldn't serialize
_FIND_SCAN_CAP = 400       # hard ceiling on files profiled per call
_FIND_PROMPT_HINT = 100    # chars of a positive prompt echoed back for context

# A base-model value is identified by its widget NAME (positional widgets are
# mapped to names via the live schema). Only the two unambiguous diffusion-model
# widgets - so a VAE/CLIP/upscale filename (upscalers use "model_name") isn't
# miscounted as the base model; upscalers surface as the "upscale" feature instead.
_BASE_MODEL_WIDGETS = {"ckpt_name", "unet_name"}
_MODEL_EXTS = (".safetensors", ".ckpt", ".sft", ".gguf", ".pt", ".pth", ".bin")

# feature tag -> substrings matched against a node's (lowercased) class_type.
# Custom nodes vary in prefix, so match on substring, not exact class name.
_FEATURE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lora", ("lora",)),
    ("controlnet", ("controlnet",)),
    ("ipadapter", ("ipadapter",)),
    ("detailer", ("detailer", "facerestore", "facedetail")),
    ("upscale", ("upscale",)),
    ("inpaint", ("inpaint",)),
    ("flux-guidance", ("fluxguidance",)),
    ("video", ("vhs_", "animatediff", "svd_", "wanvideo")),
)

# intent words -> a feature the workflow actually has, so "fix the face" matches a
# detailer and "hi-res" matches an upscale without the literal tag in the request.
_INTENT_SYNONYMS: dict[str, frozenset[str]] = {
    "detailer": frozenset({"detail", "detailer", "face", "adetailer"}),
    "upscale": frozenset({"upscale", "upscaler", "hires", "highres", "enlarge", "4k", "2x"}),
    "lora": frozenset({"lora", "loras"}),
    "controlnet": frozenset({"controlnet", "control", "pose", "openpose", "depth", "canny"}),
    "inpaint": frozenset({"inpaint", "inpainting", "mask"}),
    "video": frozenset({"video", "animation", "animate", "motion"}),
}

# dropped from an intent before matching - too generic to discriminate.
_FIND_STOPWORDS = frozenset(
    {
        "make", "create", "generate", "render", "want", "need", "using", "use",
        "with", "and", "the", "for", "that", "this", "some", "image", "images",
        "picture", "workflow", "please", "give", "get", "one",
    }
)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication of non-empty strings."""
    seen: dict[str, None] = {}
    for it in items:
        if it:
            seen.setdefault(it, None)
    return list(seen)


def _intent_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.split(r"[^a-z0-9]+", text.lower())
        if len(t) >= 2 and t not in _FIND_STOPWORDS
    }


def _profile_workflow(
    name: str, data: dict[str, Any], object_info: dict[str, Any], learned_dir: Path
) -> dict[str, Any]:
    """Compact, searchable profile of one saved workflow, extracted straight from
    its stored JSON (so hand-built graphs are covered too). May raise on a
    malformed file; find_workflow skips those."""
    wf = Workflow.from_ui(data)

    # every class_type, including inside subgraph definitions, drives feature tags
    class_types = [n.type for n in wf.nodes.values()]
    for sg in wf.subgraph_defs().values():
        class_types += [str(n.get("type", "")) for n in sg.get("nodes", []) or []]
    lowered = [c.lower() for c in class_types if c]

    base_models: list[str] = []
    loras: list[str] = []
    resolutions: list[str] = []
    prompts: list[str] = []
    for node in wf.nodes.values():
        if object_info.get(node.type) is None:
            continue  # unknown/custom node: features still counted via class_type
        try:
            named = widgets_to_named(node.type, node.widgets_values, object_info)
        except Exception:
            continue
        for key, val in named.items():
            if not isinstance(val, str) or not val:
                continue
            if "lora" in key and val.lower().endswith(_MODEL_EXTS):
                loras.append(_basename(val))
            elif key in _BASE_MODEL_WIDGETS:
                base_models.append(_basename(val))
        t = node.type.lower()
        if "empty" in t and "latent" in t:  # the resolution knob
            wv, hv = named.get("width"), named.get("height")
            if isinstance(wv, (int, float)) and isinstance(hv, (int, float)):
                resolutions.append(f"{int(wv)}x{int(hv)}")
        if "textencode" in t:
            txt = named.get("text")
            if isinstance(txt, str) and txt.strip():
                prompts.append(txt.strip())

    features: list[str] = []
    for tag, markers in _FEATURE_MARKERS:
        if tag not in features and any(m in c for c in lowered for m in markers):
            features.append(tag)
    if "loadimage" in lowered and "vaeencode" in lowered and "inpaint" not in features:
        features.append("img2img")

    try:
        family = knowledge.detect_family(wf, object_info, learned_dir=learned_dir)
    except Exception:
        family = None

    return {
        "name": name,
        "family": family,
        "base_models": _dedupe(base_models),
        "loras": _dedupe(loras),
        "resolutions": _dedupe(resolutions),
        "features": features,
        "nodes": sum(1 for n in wf.nodes.values() if n.type not in VIRTUAL_TYPES),
        "prompts": _dedupe(prompts),
    }


def _profile_haystack(p: dict[str, Any]) -> str:
    parts = [p["name"], p.get("family") or ""]
    parts += p["base_models"] + p["loras"] + p["resolutions"] + p["features"] + p["prompts"]
    return " ".join(parts).lower()


def _score_profile(p: dict[str, Any], tokens: set[str]) -> tuple[int, list[str]]:
    """Heuristic relevance of a profile to the intent tokens: +1 per intent token
    that appears anywhere in the profile, +2 when an intent word maps (via
    synonyms) to a feature the workflow actually has. Returns (score, why)."""
    haystack = _profile_haystack(p)
    matched: set[str] = set()
    score = 0
    for tok in tokens:
        if tok in haystack:
            score += 1
            matched.add(tok)
    for tag in p["features"]:
        syns = _INTENT_SYNONYMS.get(tag)
        if syns and tokens & syns:
            score += 2
            matched.add(tag)
    return score, sorted(matched)


def _present_match(p: dict[str, Any], score: int, matched: list[str]) -> dict[str, Any]:
    """The compact, token-frugal view returned to the caller - the useful bits for
    deciding whether to reuse, never the full graph."""
    out: dict[str, Any] = {"name": p["name"], "score": score, "matched": matched}
    if p.get("family"):
        out["family"] = p["family"]
    if p["base_models"]:
        out["base_models"] = p["base_models"]
    if p["loras"]:
        out["loras"] = p["loras"]
    if p["resolutions"]:
        out["resolution"] = ", ".join(p["resolutions"])
    if p["features"]:
        out["features"] = p["features"]
    out["nodes"] = p["nodes"]
    if p["prompts"]:
        hint = p["prompts"][0]
        out["prompt_hint"] = (
            hint[:_FIND_PROMPT_HINT] + "…" if len(hint) > _FIND_PROMPT_HINT else hint
        )
    return out


@mcp.tool(annotations=_READ_INSTANCE)
async def find_workflow(intent: str, limit: int = 5) -> dict[str, Any]:
    """Find saved workflows that already DO what you're about to build, so reuse
    beats rebuilding from scratch. Describe the goal in words - model, subject,
    resolution, extras - e.g. "flux portrait at 1024 with a face detailer", and get
    back a few RANKED, compact matches: family, base model, resolution, feature tags
    (detailer / upscale / lora / controlnet / inpaint / img2img), and why each
    matched. Profiles are extracted from the saved JSON, so hand-built workflows are
    covered too. Returns summaries only, never full graphs - load the one you want
    with import_workflow(name=...). Prefer this over importing+inspecting each result
    of list_workflows."""
    intent = (intent or "").strip()
    if not intent:
        return {"error": "describe what you want, e.g. 'flux portrait with a face detailer at 1024'"}
    names = [n[:-5] if n.endswith(".json") else n for n in await _client().list_userdata_workflows()]
    if not names:
        return {
            "intent": intent,
            "matches": [],
            "scanned": 0,
            "hint": "no saved workflows in ComfyUI's workflow browser yet",
        }
    object_info = await _object_info()
    learned = _config().learned_dir
    scan = names[:_FIND_SCAN_CAP]

    sem = asyncio.Semaphore(_FIND_CONCURRENCY)

    async def _load(nm: str) -> tuple[str, Any]:
        async with sem:
            try:
                return nm, await _client().get_userdata_workflow(nm)
            except Exception:
                return nm, None  # unreachable/renamed/corrupt: skipped below

    loaded = await asyncio.gather(*(_load(n) for n in scan))

    tokens = _intent_tokens(intent)
    ranked: list[tuple[int, dict[str, Any], list[str]]] = []
    skipped = 0
    for nm, data in loaded:
        if not isinstance(data, dict):
            skipped += 1
            continue
        try:
            profile = _profile_workflow(nm, data, object_info, learned)
        except Exception:
            skipped += 1
            continue
        score, matched = _score_profile(profile, tokens)
        if score > 0:
            ranked.append((score, profile, matched))

    # best score first; break ties toward the simpler (fewer-node) graph, then name
    ranked.sort(key=lambda t: (-t[0], t[1]["nodes"], t[1]["name"].lower()))
    matches = [_present_match(p, s, m) for s, p, m in ranked[: max(1, limit)]]

    result: dict[str, Any] = {"intent": intent, "scanned": len(scan) - skipped, "matches": matches}
    if skipped:
        result["skipped"] = skipped
    if len(names) > len(scan):
        result["note"] = f"profiled the first {len(scan)} of {len(names)} saved workflows"
    if matches:
        result["hint"] = "import_workflow(name=...) loads a match into the session - no pasting"
    else:
        result["hint"] = (
            "nothing clearly matched; try model/feature keywords "
            "('flux', 'sdxl', 'detailer', 'upscale', 'inpaint', 'lora'), "
            "or list_workflows to browse everything by name"
        )
    return result


@mcp.tool(annotations=_EDIT_LOCAL)
async def import_workflow(
    workflow_json: str = "", name: str = "", title: str = ""
) -> dict[str, Any]:
    """Import an existing workflow into the session. EITHER paste JSON as
    `workflow_json` (UI format with nodes/links, or API format
    {id: {class_type, inputs}}), OR pass `name` to load one straight from
    ComfyUI's workflow browser (see list_workflows) - preferred for large files,
    no pasting needed. Use for beautifying/diagnosing/porting outside work."""
    if bool(workflow_json) == bool(name):
        return {"error": "pass exactly one of workflow_json (pasted JSON) or name (see list_workflows)"}
    if name:
        try:
            data = await _client().get_userdata_workflow(name)
        except FileNotFoundError:
            return {
                "error": f"no workflow named {name!r} in ComfyUI's workflow browser",
                "hint": "list_workflows shows what's available",
            }
        except ValueError as e:
            return {"error": str(e)}
        title = title or name.replace("\\", "/").rsplit("/", 1)[-1]
    else:
        try:
            data = json.loads(workflow_json)
        except json.JSONDecodeError as e:
            return {
                "error": f"workflow_json is not valid JSON: {e}",
                "hint": "paste the file's exact contents, or use name=... to load a "
                "workflow straight from ComfyUI (list_workflows shows them)",
            }
    if not isinstance(data, dict):
        return {
            "error": f"expected a JSON object, got {type(data).__name__}",
            "hint": "a workflow is either UI format (a 'nodes' list) or API format "
            "({node_id: {class_type, inputs}})",
        }
    try:
        if "nodes" in data:
            wf = Workflow.from_ui(data)
        else:
            wf = Workflow.from_api(data, await _object_info())
    except (ValueError, KeyError, TypeError) as e:
        return {
            "error": f"could not parse this workflow: {e}",
            "hint": "UI format needs a 'nodes' list with integer ids; API format is "
            "{node_id: {class_type, inputs}}. export_workflow_json shows both shapes",
        }
    workflow_id = _session().create(wf, title=title or "imported")
    return _summary(workflow_id, wf)


@mcp.tool(annotations=_READ_LOCAL)
async def inspect_workflow(workflow_id: str) -> dict[str, Any]:
    """Compact view of a session workflow: nodes (id/class/title/widgets), links,
    groups - plus full inner node/wiring detail for any subgraph definitions
    (newer bundled templates package their graph as a subgraph)."""
    wf = _wf(workflow_id)
    summary = _summary(workflow_id, wf)
    subgraphs = wf.subgraph_defs()
    if subgraphs:
        summary["subgraphs"] = [_subgraph_summary(sg, wf) for sg in subgraphs.values()]
        summary["subgraph_note"] = (
            "subgraph instances run FLATTENED - validate/run/export expand them "
            "automatically. Edit internals in place with edit_workflow's "
            "definition-scoped ops (set_widget_in_definition, connect_in_definition, "
            "add_node_to_definition, remove_node_from_definition, "
            "set_title_in_definition, set_mode_in_definition), passing the "
            "subgraph 'id' as definition_id and the inner node's own id; only a "
            "definition that itself contains an instance needs a flat rebuild. "
            "An input marked '(internal)' is not a socket on any instance, so "
            "`connect` can't target it - wire it inside the definition instead."
        )
    return summary


# edit_workflow op schemas: op -> (required keys, optional keys). Validated
# up front so a malformed op fails with the schema spelled out instead of a
# raw KeyError, and misspelled keys (widgets_values, node, ...) are rejected
# instead of silently ignored.
_OP_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "add_node": ({"class_type"}, {"title", "widgets", "force"}),
    "remove_node": ({"node_id"}, set()),
    "connect": ({"from_node", "from_output", "to_node", "to_input"}, {"force"}),
    "set_widget": ({"node_id", "input", "value"}, {"force"}),
    "set_title": ({"node_id", "title"}, set()),
    "set_mode": ({"node_id", "mode"}, set()),
    "set_pos": ({"node_id", "pos"}, {"size"}),
    "add_group": ({"title", "node_ids"}, {"color"}),
    "set_group": ({"group_id"}, {"title", "node_ids", "color"}),
    "remove_group": ({"group_id"}, set()),
    "add_node_to_definition": (
        {"definition_id", "class_type"},
        {"title", "widgets", "force"},
    ),
    "connect_in_definition": (
        {"definition_id", "from_node", "from_output", "to_node", "to_input"},
        set(),
    ),
    "remove_node_from_definition": ({"definition_id", "node_id"}, set()),
    "set_title_in_definition": ({"definition_id", "node_id", "title"}, set()),
    "set_mode_in_definition": ({"definition_id", "node_id", "mode"}, set()),
    "set_widget_in_definition": (
        {"definition_id", "node_id", "input", "value"},
        {"force"},
    ),
}


def _check_op(index: int, op: dict[str, Any]) -> str:
    """Validate one op against _OP_SPECS; returns the op kind or raises with
    the exact schema of the failing op."""
    kind = op.get("op")
    if kind not in _OP_SPECS:
        raise ValueError(
            f"operation {index}: unknown op {kind!r}; valid ops: {sorted(_OP_SPECS)}"
        )
    required, optional = _OP_SPECS[kind]
    allowed = required | optional | {"op"}
    problems = []
    missing = sorted(required - op.keys())
    if missing:
        problems.append(f"missing required key(s) {missing}")
    for key in sorted(op.keys() - allowed):
        close = difflib.get_close_matches(key, allowed, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        problems.append(f"unexpected key {key!r}{hint}")
    if problems:
        schema = f"'{kind}' requires {sorted(required)}"
        if optional:
            schema += f", optional {sorted(optional)}"
        raise ValueError(f"operation {index} ({kind}): {'; '.join(problems)}. Schema: {schema}")
    return kind


@mcp.tool(annotations=_EDIT_LOCAL)
async def edit_workflow(
    workflow_id: str, operations: list[dict[str, Any]], summary: bool = False
) -> dict[str, Any]:
    """Apply batched edits. Each op is a dict with 'op' plus:

    - {"op": "add_node", "class_type": str, "title"?: str, "widgets"?: {name: value}}
    - {"op": "remove_node", "node_id": int}
    - {"op": "connect", "from_node": int, "from_output": str|int, "to_node": int, "to_input": str}
    - {"op": "set_widget", "node_id": int, "input": str, "value": any}
    - {"op": "set_title", "node_id": int, "title": str}
    - {"op": "set_mode", "node_id": int, "mode": int}  # 0 normal, 2 mute, 4 bypass

    All six have a definition-scoped twin taking an extra "definition_id", for
    editing inside a subgraph definition: add_node_to_definition,
    remove_node_from_definition, and connect/set_widget/set_title/
    set_mode_in_definition. A malformed op reports its own required keys.

    Layout/group ops (no definition twin): set_pos {node_id, pos:[x,y], size?:[w,h]};
    add_group {title, node_ids:[int,...], color?}; set_group {group_id, title?,
    node_ids?, color?}; remove_group {group_id}. Groups are addressed by member
    node_ids - bounding comes from their own extents. organize_workflow re-lays
    out and re-groups everything, so run these AFTER it, not before.

    Slot/widget names come from get_node_info. Virtual classes: Note/MarkdownNote
    take one widget 'text'; Reroute/PrimitiveNode take none at add - connect a
    PrimitiveNode to a widget input to mirror its type, then set_widget 'value'
    (+ 'control_after_generate' for number/combo, to advance each run).

    Ops apply in order; a failing op stops the batch (graph unchanged past that
    point). Widget values and link types are checked live - "force": true on
    set_widget/add_node/connect overrides; on connect it also lets a
    frontend-only input (no /object_info entry - rgthree switches, dynamic
    collectors) be wired by creating the socket.

    Result is a compact delta (applied ops + changed nodes); pass summary=true
    or call inspect_workflow for the full graph.
    """
    wf = _wf(workflow_id)
    object_info = await _object_info()
    applied: list[str] = []
    touched: set[int] = set()
    try:
        for index, op in enumerate(operations):
            kind = _check_op(index, op)
            if kind == "add_node":
                class_type = op["class_type"]
                widgets = op.get("widgets") or {}
                # validate class + widget names BEFORE touching the graph so a
                # failed add leaves no half-built stub node behind
                if class_type in NOTE_TYPES:
                    if bad := sorted(set(widgets) - {"text"}):
                        raise ValueError(
                            f"{class_type} has a single widget 'text'; got {bad}"
                        )
                elif class_type in VIRTUAL_TYPES:
                    # PrimitiveNode/Reroute are frontend-only nodes, absent from
                    # object_info - the "installed?" check below would reject them
                    # even though ComfyUI has them. A primitive has no widgets until
                    # it adopts a socket's type on its first connection.
                    if widgets:
                        raise ValueError(
                            f"{class_type} takes no widgets at add time: connect it to "
                            "a widget input first (it then mirrors that socket's type), "
                            "then set_widget 'value' and, for a number/combo, "
                            "'control_after_generate'; graph unchanged"
                        )
                elif class_type not in object_info:
                    raise ValueError(
                        f"unknown node class {class_type!r} - not installed on this "
                        "instance (search_nodes finds classes, resolve_missing_nodes "
                        "finds packs); graph unchanged"
                    )
                else:
                    # accept any widget that exists under some dynamic-combo
                    # selection; set_widget enforces option ordering at apply time
                    slots = all_slot_names(class_type, object_info)
                    if bad := sorted(set(widgets) - set(slots)):
                        raise ValueError(
                            f"{class_type} has no widget(s) {bad}; widgets: {slots}; "
                            "graph unchanged"
                        )
                    if not op.get("force"):
                        for name, value in widgets.items():
                            problem = check_widget_value(
                                class_type, name, value, object_info
                            )
                            if problem:
                                raise ValueError(
                                    f"{class_type}: {problem}; graph unchanged"
                                )
                node = wf.add_node(class_type, object_info=object_info, title=op.get("title"))
                for name, value in widgets.items():
                    wf.set_widget(node.id, name, value, object_info)
                touched.add(node.id)
                applied.append(f"added {node.type} as #{node.id}")
            elif kind == "remove_node":
                wf.remove_node(int(op["node_id"]))
                applied.append(f"removed #{op['node_id']}")
            elif kind == "connect":
                from_output = op["from_output"]
                if isinstance(from_output, str) and from_output.isdigit():
                    from_output = int(from_output)
                # a link already feeding the target input gets replaced - say so
                replaced = ""
                target = wf.nodes.get(int(op["to_node"]))
                pre_slot = None
                if target is not None:
                    pre_slot = target.input_by_name(op["to_input"])
                    if pre_slot is not None and pre_slot.link is not None:
                        old = wf.links.get(pre_slot.link)
                        if old is not None:
                            replaced = f" (replaced existing link from #{old.origin_id}[{old.origin_slot}])"
                wf.connect(
                    int(op["from_node"]),
                    from_output,
                    int(op["to_node"]),
                    op["to_input"],
                    object_info,
                    force=bool(op.get("force")),
                )
                touched.update((int(op["from_node"]), int(op["to_node"])))
                # a slot that didn't exist before and isn't a declared widget
                # was force-created as an undeclared (frontend-only) socket -
                # flag it so the caller knows to verify with a test run
                undeclared_note = ""
                if target is not None and pre_slot is None and op.get("force"):
                    try:
                        is_declared_widget = op["to_input"] in all_slot_names(
                            target.type, object_info
                        )
                    except ValueError:
                        is_declared_widget = False
                    if not is_declared_widget:
                        undeclared_note = (
                            f" - created undeclared input '{op['to_input']}' on "
                            f"{target.type} (frontend-created slot, verify with a test run)"
                        )
                applied.append(
                    f"connected #{op['from_node']}.{op['from_output']} -> "
                    f"#{op['to_node']}.{op['to_input']}{replaced}{undeclared_note}"
                )
            elif kind == "set_widget":
                node_id = int(op["node_id"])
                node = wf.nodes[node_id]
                if not op.get("force"):
                    if node.type == PRIMITIVE_TYPE and op["input"] != "control_after_generate":
                        # a primitive's value must satisfy the widget it mirrors -
                        # nothing else validates it, since its consumer's slot is
                        # connected and so skips its own widget check
                        problem = check_primitive_value(
                            wf, node, op["value"], object_info
                        )
                    elif node.type in VIRTUAL_TYPES:
                        problem = None  # notes/reroutes/primitive control slots
                    else:
                        problem = check_widget_value(
                            node.type, op["input"], op["value"], object_info,
                            node.widgets_values,
                            {slot.name for slot in node.inputs},
                        )
                    if problem:
                        raise ValueError(f"{node.type} #{node_id}: {problem}")
                wf.set_widget(node_id, op["input"], op["value"], object_info)
                touched.add(node_id)
                applied.append(f"set #{op['node_id']}.{op['input']} = {op['value']!r}")
            elif kind == "set_title":
                wf.nodes[int(op["node_id"])].title = op["title"]
                touched.add(int(op["node_id"]))
                applied.append(f"titled #{op['node_id']}")
            elif kind == "set_mode":
                wf.nodes[int(op["node_id"])].mode = int(op["mode"])
                touched.add(int(op["node_id"]))
                applied.append(f"mode #{op['node_id']} = {op['mode']}")
            elif kind == "set_pos":
                node_id = int(op["node_id"])
                node = wf.nodes[node_id]
                pos = op["pos"]
                if not (isinstance(pos, list | tuple) and len(pos) == 2):
                    raise ValueError(f"set_pos: 'pos' must be [x, y]; got {pos!r}")
                node.pos = [float(pos[0]), float(pos[1])]
                if "size" in op:
                    size = op["size"]
                    if not (isinstance(size, list | tuple) and len(size) == 2):
                        raise ValueError(f"set_pos: 'size' must be [w, h]; got {size!r}")
                    node.size = [float(size[0]), float(size[1])]
                touched.add(node_id)
                applied.append(
                    f"moved #{node_id} to {node.pos}"
                    + (f", size {node.size}" if "size" in op else "")
                )
            elif kind == "add_group":
                node_ids = [int(nid) for nid in op["node_ids"]]
                group = wf.group_from_nodes(op["title"], node_ids, color=op.get("color", "#3f789e"))
                applied.append(f"added group #{group.id} {group.title!r} ({len(node_ids)} nodes)")
            elif kind == "set_group":
                group = _find_group(wf, int(op["group_id"]))
                if "title" in op:
                    group.title = op["title"]
                if "color" in op:
                    group.color = op["color"]
                if "node_ids" in op:
                    node_ids = [int(nid) for nid in op["node_ids"]]
                    group.bounding = wf.group_bounding_for(node_ids)
                applied.append(f"updated group #{group.id} {group.title!r}")
            elif kind == "remove_group":
                group = _find_group(wf, int(op["group_id"]))
                wf.groups = [g for g in wf.groups if g.id != group.id]
                applied.append(f"removed group #{group.id}")
            elif kind == "add_node_to_definition":
                definition_id = op["definition_id"]
                class_type = op["class_type"]
                widgets = op.get("widgets") or {}
                inner = wf.subgraph_as_workflow(definition_id)
                if class_type in NOTE_TYPES:
                    if bad := sorted(set(widgets) - {"text"}):
                        raise ValueError(
                            f"{class_type} has a single widget 'text'; got {bad}"
                        )
                elif class_type not in object_info:
                    raise ValueError(
                        f"unknown node class {class_type!r} - not installed on this "
                        "instance; definition unchanged"
                    )
                else:
                    slots = all_slot_names(class_type, object_info)
                    if bad := sorted(set(widgets) - set(slots)):
                        raise ValueError(
                            f"{class_type} has no widget(s) {bad}; widgets: {slots}; "
                            "definition unchanged"
                        )
                    if not op.get("force"):
                        for name, value in widgets.items():
                            problem = check_widget_value(
                                class_type, name, value, object_info
                            )
                            if problem:
                                raise ValueError(
                                    f"{class_type}: {problem}; definition unchanged"
                                )
                new_node = inner.add_node(
                    class_type,
                    object_info=object_info,
                    title=op.get("title"),
                )
                for name, value in widgets.items():
                    inner.set_widget(new_node.id, name, value, object_info)
                wf.update_subgraph(definition_id, inner)
                touched.add(new_node.id)
                applied.append(
                    f"added {class_type} as #{new_node.id} in definition {definition_id}"
                )
            elif kind == "connect_in_definition":
                definition_id = op["definition_id"]
                from_node = int(op["from_node"])
                from_output = op["from_output"]
                to_node = int(op["to_node"])
                to_input = op["to_input"]
                if from_node in (-10, -20) or to_node in (-10, -20):
                    raise ValueError(
                        "cannot connect to boundary pseudo-nodes (-10/-20) directly"
                    )
                if isinstance(from_output, str) and from_output.isdigit():
                    from_output = int(from_output)
                inner = wf.subgraph_as_workflow(definition_id)
                replaced = ""
                target = inner.nodes.get(to_node)
                if target is not None:
                    slot = target.input_by_name(to_input)
                    if slot is not None and slot.link is not None:
                        old = inner.links.get(slot.link)
                        if old is not None:
                            replaced = (
                                f" (replaced existing link from "
                                f"#{old.origin_id}[{old.origin_slot}])"
                            )
                inner.connect(
                    from_node, from_output, to_node, to_input, object_info,
                )
                wf.update_subgraph(definition_id, inner)
                applied.append(
                    f"connected #{from_node}.{from_output} -> "
                    f"#{to_node}.{to_input} in definition {definition_id}{replaced}"
                )

            elif kind == "remove_node_from_definition":
                def_id = op["definition_id"]
                inner_nid = int(op["node_id"])
                inner_wf = wf.subgraph_as_workflow(def_id)
                inner_wf.remove_node(inner_nid)
                wf.update_subgraph(def_id, inner_wf)
                warnings = []
                for node in wf.nodes.values():
                    if node.type != def_id:
                        continue
                    proxy = (node.properties or {}).get("proxyWidgets") or {}
                    found = False
                    if isinstance(proxy, dict):
                        found = str(inner_nid) in proxy or inner_nid in proxy
                    elif isinstance(proxy, list):
                        found = any(
                            isinstance(p, (list, tuple)) and len(p) >= 1
                            and str(p[0]) == str(inner_nid)
                            for p in proxy
                        )
                    if found:
                        warnings.append(
                            f"removed inner node #{inner_nid} but instance "
                            f"#{node.id} has proxyWidgets for it; those widget "
                            f"overrides will be dropped during flatten"
                        )
                result_msg = f"remove_node_from_definition: removed #{inner_nid} from definition {def_id}"
                if warnings:
                    result_msg += f"; warnings: {'; '.join(warnings)}"
                applied.append(result_msg)
            elif kind == "set_title_in_definition":
                def_id = op["definition_id"]
                inner_nid = int(op["node_id"])
                inner_wf = wf.subgraph_as_workflow(def_id)
                inner_wf.nodes[inner_nid].title = op["title"]
                wf.update_subgraph(def_id, inner_wf)
                applied.append(f"set_title_in_definition: titled #{inner_nid} in definition {def_id}")
            elif kind == "set_mode_in_definition":
                def_id = op["definition_id"]
                inner_nid = int(op["node_id"])
                inner_wf = wf.subgraph_as_workflow(def_id)
                inner_wf.nodes[inner_nid].mode = int(op["mode"])
                wf.update_subgraph(def_id, inner_wf)
                applied.append(
                    f"set_mode_in_definition: mode #{inner_nid} = {op['mode']} in definition {def_id}"
                )
            elif kind == "set_widget_in_definition":
                def_id = op["definition_id"]
                inner_nid = int(op["node_id"])
                input_name = op["input"]
                value = op["value"]
                if any(input_name.endswith(s) for s in SYNTHETIC_SUFFIXES):
                    raise ValueError(
                        f"cannot set synthetic control slot '{input_name}' on "
                        "definition-internal node"
                    )
                inner_wf = wf.subgraph_as_workflow(def_id)
                inner_node = inner_wf.nodes[inner_nid]
                if not op.get("force") and inner_node.type not in NOTE_TYPES:
                    problem = check_widget_value(
                        inner_node.type, input_name, value, object_info,
                        inner_node.widgets_values,
                        {slot.name for slot in inner_node.inputs},
                    )
                    if problem:
                        raise ValueError(
                            f"{inner_node.type} #{inner_nid} in definition "
                            f"{def_id}: {problem}"
                        )
                inner_wf.set_widget(inner_nid, input_name, value, object_info)
                wf.update_subgraph(def_id, inner_wf)
                touched.add(inner_nid)
                applied.append(
                    f"set_widget_in_definition: set #{inner_nid}.{input_name} = {value!r} in definition {def_id}"
                )
    except KeyError as e:
        return {
            "applied": applied,
            "error": f"unknown node id {e}",
            "hint": "inspect_workflow lists current node ids",
        }
    except ValueError as e:
        return {
            "applied": applied,
            "error": str(e),
            "hint": "get_node_info gives slot/widget names; the op schemas are in this tool's description",
        }
    if summary:
        return {"applied": applied, "summary": _summary(workflow_id, wf)}
    # compact delta: re-sending the whole graph after every edit batch was the
    # single biggest recurring token cost; inspect_workflow has the full view
    return {
        "applied": applied,
        "nodes": len(wf.nodes),
        "links": len(wf.links),
        "changed": [
            {
                "id": n.id,
                "class_type": n.type,
                "title": n.title,
                "widgets": _widget_preview(n),
            }
            for nid in sorted(touched)
            if (n := wf.nodes.get(nid)) is not None
        ],
    }


@mcp.tool(annotations=_EDIT_LOCAL)
async def organize_workflow(workflow_id: str) -> dict[str, Any]:
    """THE finishing step: auto-layout into pipeline stage bands, colored groups,
    human titles, green highlights on user-editable knobs, and markdown guidance
    notes (model-family aware, two registers: 'touch this' vs 'leave alone').
    Run after wiring is done and before save_workflow. Idempotent.

    MUTATES the session workflow in place - the `applied` block in the result
    summarizes the layout/group/note changes; inspect_workflow or
    export_workflow_json shows the full reorganized graph."""
    wf = _wf(workflow_id)
    object_info = await _object_info()
    learned_dir = _config().learned_dir
    report = annotate(wf, object_info, learned_dir=learned_dir)
    report["lint"] = _cap_lint(lint(wf, object_info, learned_dir=learned_dir))
    return report


@mcp.tool(annotations=_READ_INSTANCE)
async def lint_workflow(workflow_id: str) -> list[dict[str, Any]]:
    """Readability/wiring lint: unlabeled prompts, missing groups/notes, orphan
    nodes, unconnected required inputs, overlapping nodes, misaligned
    resolution (when a family with a known alignment requirement is detected).
    Empty list = clean."""
    return lint(_wf(workflow_id), await _object_info(), learned_dir=_config().learned_dir)


@mcp.tool(annotations=_READ_INSTANCE)
async def validate_workflow(workflow_id: str) -> dict[str, Any]:
    """Validate against the LIVE instance: node classes installed, widget values in
    range, combo/model-file values actually present (with closest-match suggestions),
    required inputs connected. Fix errors before run_workflow."""
    findings = validate(_wf(workflow_id), await _object_info(refresh=True))
    capped = _cap_findings(findings)
    return {
        "ok": not any(f["level"] == "error" for f in findings),
        "findings": capped,
        **_subgraph_edit_hint(capped),
    }


@mcp.tool(annotations=_READ_INSTANCE)
async def diagnose_workflow(workflow_id: str) -> dict[str, Any]:
    """Deep-check an old/broken workflow and propose fixes: everything from
    validate_workflow PLUS Comfy Registry resolution for missing custom-node
    classes (which pack provides them, how to install). Apply fixes via
    edit_workflow, or port_workflow for model-family moves."""
    wf = _wf(workflow_id)
    findings = validate(wf, await _object_info(refresh=True))
    missing = sorted({f["class_type"] for f in findings if f["code"] == "missing-node-class"})
    registry_result: dict[str, Any] = {}
    capability_impact: list[dict[str, Any]] = []
    if missing:
        try:
            registry_result = await _registry().resolve_node_classes(missing)
        except RegistryUnavailableError as e:
            # the registry is remote and optional; the local findings above are
            # the valuable part of a diagnose - never throw them away for it
            registry_result = {
                "error": str(e),
                "hint": "the findings above are complete; pack resolution needs "
                "internet access - retry resolve_missing_nodes when online",
                "unresolved": missing,
            }
        resolved = registry_result.get("resolved", {})
        capability_impact = [
            {"class_type": cls, "provided_by": resolved.get(cls)} for cls in missing
        ]
    result: dict[str, Any] = {
        # errors only, same predicate as validate_workflow: an informational
        # note (a disabled node, a subgraph instance) is not a problem, and
        # reporting ok=False for one sent agents hunting for a nonexistent fault
        "ok": not any(f["level"] == "error" for f in findings),
        "findings": (capped := _cap_findings(findings)),
        "missing_node_packs": registry_result,
        "capability_impact": capability_impact,
        "family": knowledge.detect_family(
            wf, await _object_info(), learned_dir=_config().learned_dir
        ),
        **_subgraph_edit_hint(capped),
    }
    if missing:
        # stated once, not per missing node
        result["capability_notice"] = (
            "Per missing node: install its pack (runs third-party code), replace "
            "it with a core/installed equivalent, or drop it and LOSE what it "
            "does. Tell the user what function is lost BEFORE they choose - "
            "never drop a feature silently."
        )
    return result


@mcp.tool(annotations=_EDIT_LOCAL)
async def port_workflow(workflow_id: str, target_family: str) -> dict[str, Any]:
    """CROSS-FAMILY MODEL PORT ONLY (e.g. 'sdxl' -> 'flux'): swaps loader
    topology when needed, retunes CFG/steps/sampler/scheduler and technique
    nodes (FaceDetailer etc.) from family knowledge, swaps latent node class,
    picks installed model files. NOT for fixing missing/uninstalled nodes -
    that's diagnose_workflow + resolve_missing_nodes. Returns changes + flags
    for anything that needs your judgment. Families: get_model_guidance /
    get_instance_info."""
    wf = _wf(workflow_id)
    report = port_engine(wf, target_family, await _object_info(refresh=True), _config().learned_dir)
    report["validate"] = _cap_findings(validate(wf, await _object_info()))
    return report


# --------------------------------------------------------------------------
# Execution & saving
# --------------------------------------------------------------------------


# run/view/upload/queue tool concepts inspired by KerbalTheGathering/ComfyUI_MCP
# (independently implemented; see README acknowledgments).

PREVIEW_MAX_DIM = 768  # inline previews are thumbnails; view_output serves full size

_PARTIAL_RUN_WARNING = (
    "ComfyUI accepted the prompt but REJECTED one or more nodes at queue time and "
    "executed only the rest of the graph - expected outputs (images/video) may be "
    "missing. Inspect node_errors, fix the offending nodes (diagnose_workflow / "
    "edit_workflow), and re-run. Do NOT treat this as a complete render."
)


"""When run_workflow finds this many (or more) prompts already pending and the
caller didn't say where in line to go, it returns queue_busy instead of
queuing, so the user can choose front-of-queue vs waiting."""
_QUEUE_BUSY_THRESHOLD = 2


@mcp.tool(annotations=_WRITE_INSTANCE)
async def run_workflow(
    workflow_id: str,
    timeout_seconds: float = 600,
    return_preview: bool = True,
    wait: bool = True,
    allow_invalid: bool = False,
    save_dir: str = "",
    roll_seeds: bool = True,
    front: bool | None = None,
    confirm_spend: bool = False,
    ctx: Context | None = None,
) -> Any:
    """Queue the workflow and (by default) wait for completion. Returns status,
    node errors on failure, output file refs, any non-file return values
    (data_outputs: generated text, paths a save node wrote), and an inline preview
    thumbnail so you can SEE the result (view_output fetches full size).
    wait=False returns {status: queued, prompt_id} - poll get_run_status. Prove a
    workflow works before saving/delivering.

    Text-only caller (no image input)? Pass return_preview=False - the result then
    carries a file path instead of a thumbnail if save_dir/COMFYUI_MOUNT_DIR is set.

    roll_seeds=True (default) mirrors the browser: every seed/PrimitiveNode set to
    randomize/increment/decrement is re-rolled and persisted before submit - the
    raw /prompt API never does, so headless runs repeat forever. False re-runs
    the stored values.

    allow_invalid=True submits despite local validation errors (ComfyUI is the
    final judge; use it if a valid graph is wrongly blocked). save_dir (or the
    configured COMFYUI_MOUNT_DIR) relocates every finished output file - images,
    video, audio alike - into a folder the caller can reach, returning
    saved_paths. Needs finished files (wait=True); a background run relocates
    later via save_output(prompt_id=...).

    front: None (default) refuses to queue when >=2 prompts are already pending
    and returns {status: queue_busy} so the USER can choose; True runs next
    (pending jobs untouched); False waits at the back of the line.

    confirm_spend: partner/API nodes charge the user's account per submit, so a
    graph containing one is gated - pass True only after they have agreed.

    LONG RENDERS: a timeout cancels the caller's wait, not the ComfyUI job.
    Submit wait=False, front=False, then poll get_run_status(prompt_id) until
    success/error/partial and call save_output. prompt_id survives in
    manage_queue(status).draftsman_submitted if your session dies mid-poll."""
    wf = _wf(workflow_id)
    if front is None:
        # best-effort etiquette check; an unreachable /queue never blocks a run
        with contextlib.suppress(Exception):
            queue = await _client().get_queue()
            pending = len(queue.get("queue_pending", []))
            if pending >= _QUEUE_BUSY_THRESHOLD:
                return {
                    "status": "queue_busy",
                    "queue_running": len(queue.get("queue_running", [])),
                    "queue_pending": pending,
                    "hint": (
                        "nothing was queued - ASK THE USER how to proceed, then re-run "
                        "with front=True to go next after the current job (pending jobs "
                        "stay queued, untouched) or front=False to wait in line"
                    ),
                }
    # refresh: combo choices embed the installed model files, so a stale cache
    # can wave through (or wrongly block) model-name widgets
    object_info = await _object_info(refresh=True)
    if not allow_invalid:
        errors = [f for f in validate(wf, object_info) if f["level"] == "error"]
        if errors:
            capped = _cap_findings(errors)
            return {
                "status": "invalid",
                "findings": capped,
                "hint": "fix with edit_workflow (diagnose_workflow for missing node "
                "classes), or run_workflow(allow_invalid=True) to submit anyway",
                **_subgraph_edit_hint(capped),
            }
    try:
        api = wf.to_api(object_info)
    except ValueError as e:
        return {"status": "invalid", "error": str(e)}
    # advisory only, and silent unless it has something to say. Both checks sit
    # here, after to_api: they describe what would actually be submitted, and
    # neither is worth computing for a graph that cannot be.
    capacity = await _capacity(wf, object_info)
    warn = {"capacity": capacity} if capacity else {}
    # Partner/API nodes spend the user's money, so this gate fires BEFORE
    # anything is queued and returns rather than raises.
    billable = api_nodes(api, object_info)
    if billable:
        if not _config().comfy_api_key:
            # today this surfaces as an opaque queue-time "Unauthorized"; the
            # same condition deserves a name and a fix
            return {
                "status": "missing_api_key",
                **_spend_payload(billable),
                "hint": "this graph needs partner/API nodes, which require COMFY_API_KEY "
                "in the server's environment. Nothing was queued - ask the user to set it "
                "(their Comfy Org account key) and restart the MCP server.",
            }
        if not confirm_spend:
            names = ", ".join(f"#{n['node_id']} {n['class_type']}" for n in billable[:_API_NODES_CAP])
            refusal = await _confirm(
                ctx,
                f"This workflow queues {len(billable)} partner/API node(s) ({names}), "
                "which charge your Comfy Org account. Run it?",
                _SPEND_HINT,
                "spend",
            )
            if refusal is not None:
                return {**refusal, **_spend_payload(billable), **warn}
    # Only now that the run is definitely going ahead: rolling seeds MUTATES and
    # persists the stored workflow, and a refused run that still advanced the
    # seed would drift the user's graph without ever queueing anything.
    if roll_seeds and wf.apply_seed_control(object_info):
        # persist so inspect_workflow reflects what ran and increment/decrement
        # advance across calls; best-effort (a read-only session dir shouldn't
        # block the run)
        with contextlib.suppress(OSError):
            _session().persist(workflow_id)
        api = wf.to_api(object_info)  # re-serialize with the rolled values
    # Where to relocate finished renders: an explicit save_dir, else the
    # configured mount dir (auto-relocate). None -> leave outputs in ComfyUI.
    dest_root: Path | None = None
    mount_error: str | None = None
    if save_dir:
        dest_root, dest_error = _resolve_dest(save_dir)
        if dest_error:
            return {"status": "invalid", "error": dest_error}
    elif wait and _config().mount_dir is not None:
        dest_root, mount_error = _resolve_dest("")  # resolves + creates the mount dir
    extra_data: dict[str, Any] | None = None
    if _config().comfy_api_key:
        extra_data = {"api_key_comfy_org": _config().comfy_api_key}
    if not wait:
        tracker = _tracker()
        tracker.ensure_running()
        try:
            queued = await _client().queue_prompt(
                api, extra_data=extra_data, client_id=tracker.client_id, front=bool(front)
            )
        except ComfyValidationError as e:
            return {"status": "rejected", "error": str(e), "node_errors": e.node_errors}
        _record_submission(queued["prompt_id"], workflow_id)
        response: dict[str, Any] = {
            "status": "queued", "prompt_id": queued["prompt_id"], **warn
        }
        if save_dir:
            # relocation happens after a run finishes, and this call returns
            # before that - say so rather than silently ignoring save_dir
            response["save_dir_ignored"] = (
                f"save_dir {save_dir!r} does not apply to a background run: nothing has "
                "rendered yet. When get_run_status reports success, relocate with "
                f"save_output(prompt_id={queued['prompt_id']!r}, dest_dir={save_dir!r})"
            )
        if queued.get("node_errors"):
            response["node_errors"] = queued["node_errors"]
            response["warning"] = _PARTIAL_RUN_WARNING
        return response
    try:
        result = await _client().run_and_wait(
            api, timeout=timeout_seconds, extra_data=extra_data, front=bool(front)
        )
    except ComfyValidationError as e:
        return {"status": "rejected", "error": str(e), "node_errors": e.node_errors}
    _record_submission(result["prompt_id"], workflow_id)
    result.update(warn)
    # ComfyUI ran only part of the graph (some nodes rejected at queue time): keep
    # relocating/previewing whatever DID render, but downgrade to "partial" so the
    # dropped outputs aren't mistaken for a clean run.
    node_errors = result.pop("node_errors", None)
    ran_ok = result["status"] == "success"
    if node_errors:
        result["status"] = "partial"
        result["node_errors"] = node_errors
        result["warning"] = _PARTIAL_RUN_WARNING
    if dest_root is not None and ran_ok:
        # every kind, not just images: a video/audio render (Wan, LTX,
        # AnimateDiff...) is just as undeliverable while it sits in ComfyUI's
        # output tree, and /view serves all of them the same way
        saved, save_errors = await _relocate_outputs(_client(), result["outputs"], dest_root)
        if saved:
            result["saved_paths"] = saved
            result["dest_dir"] = str(dest_root)
        if save_errors:
            result["save_errors"] = save_errors
    elif mount_error and ran_ok:
        # COMFYUI_MOUNT_DIR is configured but unusable - say so instead of
        # silently skipping the relocation the user asked for
        result["save_errors"] = [mount_error]
    if return_preview and ran_ok:
        image_items = [o for o in result["outputs"] if o.get("kind") == "images"]
        if image_items:
            data = await _client().fetch_output(image_items[0])
            try:
                thumb, fmt, _, _ = downscale_image(data, PREVIEW_MAX_DIM)
            except ValueError:
                return result  # first "image" output isn't decodable; refs still returned
            result["preview"] = (
                f"inline image is a <={PREVIEW_MAX_DIM}px thumbnail of "
                f"{image_items[0].get('filename')} - view_output(filename=..., "
                "max_dim=None) for full size or other outputs"
            )
            return [result, Image(data=thumb, format=fmt)]
    return result


@mcp.tool(annotations=_READ_INSTANCE)
async def view_output(
    filename: str,
    subfolder: str = "",
    type: Literal["output", "temp", "input"] = "output",
    max_dim: int | None = 1024,
) -> Any:
    """Fetch a rendered image so you (and the user) can SEE it - refs come from
    run_workflow/get_run_status outputs. Downscaled to max_dim px to keep the
    conversation light; max_dim=None for full resolution."""
    problem = _check_output_ref(filename, subfolder)
    if problem:
        return {"error": problem}
    try:
        data = await _client().fetch_output(
            {"filename": filename, "subfolder": subfolder, "type": type}
        )
    except Exception as e:
        return {"error": f"could not fetch {filename!r}: {e}"}
    try:
        data, fmt, width, height = downscale_image(data, max_dim)
    except ValueError as e:
        return {"error": str(e), "hint": "only image outputs can be viewed inline"}
    # FastMCP serializes an Image only as a standalone return or a list element -
    # a dict *containing* an Image gets repr'd into text and never renders. Return
    # the image block plus a sibling meta dict (same list form as run_workflow's
    # preview) so text-only models still get the dimensions/filename.
    return [
        {"meta": {
            "filename": filename,
            "width": width,
            "height": height,
            "format": fmt,
            "subfolder": subfolder,
            "type": type,
        }},
        Image(data=data, format=fmt),
    ]


def _resolve_dest(dest_dir: str) -> tuple[Path | None, str | None]:
    """Resolve+create the relocation directory. dest_dir empty -> the configured
    COMFYUI_MOUNT_DIR. A relative path is refused: this server's cwd is NOT the
    caller's (MCP hosts often launch it from a system dir like System32), so a
    relative path would resolve somewhere invisible. Returns (path, None) or
    (None, error)."""
    root = Path(dest_dir) if dest_dir else _config().mount_dir
    if root is None:
        return None, (
            "no destination: pass save_dir/dest_dir, or set COMFYUI_MOUNT_DIR so "
            "outputs relocate to a folder the caller can reach"
        )
    root = root.expanduser()  # ~ expands to an absolute path; ./foo does not
    if not root.is_absolute():
        return None, (
            f"destination must be an absolute path (got {str(root)!r}): the server's "
            "working directory is not the agent's, so a relative path would resolve "
            "somewhere invisible - pass an absolute save_dir/dest_dir (or set "
            "COMFYUI_MOUNT_DIR to an absolute folder both sides can reach)"
        )
    try:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return None, f"destination directory unusable: {e}"
    return root, None


_MOUNT_STATUS_CACHE: tuple[Path | None, dict[str, Any]] | None = None


def _mount_status(recheck: bool = False) -> dict[str, Any]:
    """Relocation readiness for a sandboxed caller (Cowork/Desktop/Code). A render
    can only be handed to the user if COMFYUI_MOUNT_DIR points at a folder BOTH
    this server and the caller can see. We verify our half (configured, resolves,
    writable via a probe file); the shared-view half is the operator's to set up.
    Returned by get_instance_info and the draftsman://capabilities resource so an
    agent can check up front instead of discovering it after a wasted render.

    The result is cached against the configured mount path: the mount is fixed
    by env var at start, but get_instance_info / check_setup / the capabilities
    resource are all read repeatedly during a session and each was writing and
    deleting a probe file. Caching on the path (not just "once per process")
    means a reconfigured mount re-probes on its own. check_setup passes
    recheck=True - it is the tool you run precisely because something on disk
    changed."""
    global _MOUNT_STATUS_CACHE
    mount = _config().mount_dir
    if not recheck and _MOUNT_STATUS_CACHE is not None and _MOUNT_STATUS_CACHE[0] == mount:
        return _MOUNT_STATUS_CACHE[1]
    status = _probe_mount()
    _MOUNT_STATUS_CACHE = (mount, status)
    return status


def _probe_mount() -> dict[str, Any]:
    mount = _config().mount_dir
    if mount is None:
        return {
            "configured": False,
            "writable": False,
            "hint": (
                "COMFYUI_MOUNT_DIR is unset: run_workflow(save_dir=...) / save_output "
                "need an explicit absolute dest_dir, and renders can't be handed to the "
                "user automatically. Ask the user to set COMFYUI_MOUNT_DIR to a folder "
                "both ComfyUI's host and this agent can reach."
            ),
        }
    root, error = _resolve_dest("")  # resolves + creates the configured mount dir
    if error:
        return {"configured": True, "writable": False, "path": str(mount), "error": error}
    assert root is not None  # _resolve_dest returns a path whenever error is None
    probe = root / ".draftsman-write-probe"
    try:
        probe.write_bytes(b"ok")
        probe.read_bytes()
    except OSError as e:
        return {
            "configured": True,
            "writable": False,
            "path": str(root),
            "error": f"COMFYUI_MOUNT_DIR exists but isn't writable: {e}",
        }
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    return {"configured": True, "writable": True, "path": str(root)}


def _dedupe_path(path: Path) -> Path:
    """`path`, or `path` with a numeric suffix if it already exists."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for i in range(1, 10000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    return path


def _safe_dest(root: Path, name: str) -> Path | None:
    """Join a filename under root, refusing anything that escapes it."""
    target = (root / Path(name).name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


async def _relocate_outputs(
    client: ComfyClient,
    items: list[dict[str, Any]],
    dest_root: Path,
    dest_filename: str | None = None,
    overwrite: bool = False,
) -> tuple[list[str], list[str]]:
    """Fetch each output image's bytes and write them under dest_root. Returns
    (saved_paths, errors)."""
    saved: list[str] = []
    errors: list[str] = []
    for item in items:
        filename = item.get("filename", "")
        try:
            data = await client.fetch_output(item)
        except Exception as e:  # surface the fetch failure per file, keep going
            errors.append(f"fetch {filename!r}: {e}")
            continue
        target = _safe_dest(dest_root, dest_filename or filename)
        if target is None:
            errors.append(f"unsafe destination name: {dest_filename or filename!r}")
            continue
        if not overwrite:
            target = _dedupe_path(target)
        try:
            target.write_bytes(data)
        except OSError as e:
            errors.append(f"write {target}: {e}")
            continue
        saved.append(str(target))
    return saved, errors


@mcp.tool(annotations=_WRITE_INSTANCE)
async def save_output(
    prompt_id: str = "",
    filename: str = "",
    subfolder: str = "",
    type: Literal["output", "temp", "input"] = "output",
    dest_dir: str = "",
    dest_filename: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a finished render out of ComfyUI's output tree into a folder the caller
    (e.g. a Claude Desktop / Cowork sandbox) can reach. ComfyUI's save nodes only
    write inside its own output/ dir and reject absolute paths, so a render must be
    relocated before it can be presented or edited.

    Pass prompt_id (relocates every output FILE of that finished job - images,
    video, audio) OR an explicit filename (+subfolder/type, as reported in a run's
    outputs). dest_dir defaults to COMFYUI_MOUNT_DIR; dest_filename renames a
    single file. Returns {saved_paths, dest_dir}."""
    if not prompt_id and not filename:
        return {"error": "provide prompt_id or filename - nothing to relocate otherwise"}
    client = _client()
    if prompt_id:
        history = await client.get_history(prompt_id)
        if not history:
            return {"error": f"no finished job {prompt_id!r} in history (still running?)"}
        # all kinds (images, gifs, videos, audio) - a video render is exactly as
        # stuck inside ComfyUI's output tree as an image is
        items = client._collect_outputs(history)
        if not items:
            return {"error": f"job {prompt_id!r} produced no output files to relocate"}
    else:
        problem = _check_output_ref(filename, subfolder)
        if problem:
            return {"error": problem}
        items = [{"filename": filename, "subfolder": subfolder, "type": type}]
    if dest_filename and len(items) > 1:
        return {
            "error": "dest_filename can't rename a multi-file batch; relocate one "
            "at a time (pass an explicit filename) to rename"
        }
    dest_root, dest_error = _resolve_dest(dest_dir)
    if dest_error or dest_root is None:
        return {"error": dest_error}
    saved, errors = await _relocate_outputs(
        client, items, dest_root, dest_filename or None, overwrite
    )
    result: dict[str, Any] = {"saved_paths": saved, "dest_dir": str(dest_root)}
    if errors:
        result["errors"] = errors
    return result


def _history_error(history: dict[str, Any]) -> dict[str, Any] | None:
    for name, data in history.get("status", {}).get("messages", []) or []:
        if name == "execution_error":
            return {
                "node_id": data.get("node_id"),
                "node_type": data.get("node_type"),
                "message": data.get("exception_message"),
                "type": data.get("exception_type"),
            }
    return None


@mcp.tool(annotations=_READ_INSTANCE)
async def get_run_status(prompt_id: str) -> dict[str, Any]:
    """Polling tool for runs queued with `run_workflow(wait=False)`. For long/paid renders, see `run_workflow`'s long-render pattern.
    Status of a run queued with run_workflow(wait=False): queue position, live
    step progress while sampling, and outputs (+ error details) once finished."""
    client = _client()
    history = await client.get_history(prompt_id)
    if history:
        error = _history_error(history)
        result: dict[str, Any] = {
            "status": "error" if error else "success",
            "prompt_id": prompt_id,
            "outputs": client._collect_outputs(history),
            **_workflow_tag(prompt_id),
        }
        # a pure parser over the history document, so it is called on the CLASS:
        # tests (and any sandboxed caller) substitute lightweight fake clients that
        # only implement what they need, and reaching for an instance attribute
        # here would break them for no benefit
        if data := ComfyClient._collect_data_outputs(history):
            result["data_outputs"] = data
        if error:
            result["error"] = error
        else:
            # queue-time partial accept: node_errors aren't stored in /history,
            # but the stored entry keeps both the FULL submitted prompt ([2]) and
            # the validated outputs_to_execute ([4]) - output nodes present in
            # the former but missing from the latter were dropped at queue time
            entry = history.get("prompt") or []
            if len(entry) > 4 and isinstance(entry[2], dict):
                info = await _object_info()
                executed = {str(x) for x in (entry[4] or [])}
                dropped = [
                    nid
                    for nid, n in entry[2].items()
                    if isinstance(n, dict)
                    and (info.get(str(n.get("class_type"))) or {}).get("output_node")
                    and str(nid) not in executed
                ]
                if dropped:
                    result["status"] = "partial"
                    result["dropped_output_nodes"] = dropped
                    result["warning"] = _PARTIAL_RUN_WARNING
            if result["status"] == "success":
                result["hint"] = "view_output(filename=...) to see an image output"
        return result
    queue = await client.get_queue()
    running = [entry[1] for entry in queue.get("queue_running", [])]
    pending = [entry[1] for entry in queue.get("queue_pending", [])]
    snapshot = _tracker().snapshot(prompt_id)
    if prompt_id in running:
        return {"status": "running", "prompt_id": prompt_id, **snapshot, **_workflow_tag(prompt_id)}
    if prompt_id in pending:
        return {
            "status": "pending",
            "prompt_id": prompt_id,
            "queue_position": pending.index(prompt_id) + 1,
            "queue_pending": len(pending),
            **_workflow_tag(prompt_id),
        }
    return {
        "status": "unknown",
        "prompt_id": prompt_id,
        "note": "not in queue or history - wrong prompt_id, or history was cleared",
        **snapshot,
    }


@mcp.tool(annotations=_WRITE_INSTANCE)
async def upload_image(
    image_path: str | None = None,
    image_base64: str | None = None,
    name: str | None = None,
    subfolder: str = "",
    overwrite: bool = False,
    mask_for: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload a source image into ComfyUI's input folder so LoadImage can use it
    (img2img / inpaint / ControlNet). Exactly one of image_path (local file) or
    image_base64. mask_for={filename, subfolder?, type?} uploads this as a MASK
    for that already-uploaded image instead."""
    if (image_path is None) == (image_base64 is None):
        return {"error": "pass exactly one of image_path or image_base64"}
    if image_path is not None:
        path = Path(image_path)
        if not path.is_file():
            return {"error": f"not a file: {image_path}"}
        data = path.read_bytes()
        name = name or path.name
    else:
        assert image_base64 is not None  # exactly-one guard above ensures this
        try:
            data = base64.b64decode(image_base64, validate=True)
        except Exception as e:
            return {"error": f"invalid base64: {e}"}
        name = name or "upload.png"
    if ".." in name or any(sep in name for sep in ("/", "\\")):
        return {"error": "name must be a plain filename - no path separators or '..'"}
    problem = _check_output_ref(name, subfolder)
    if problem:
        return {"error": problem}
    client = _client()
    if mask_for is not None:
        if not mask_for.get("filename"):
            return {"error": "mask_for needs at least {filename: ...}"}
        ref = {
            "filename": mask_for["filename"],
            "subfolder": mask_for.get("subfolder", ""),
            "type": mask_for.get("type", "input"),
        }
        uploaded = await client.upload_mask(data, name, ref, subfolder=subfolder, overwrite=overwrite)
    else:
        uploaded = await client.upload_image(data, name, subfolder=subfolder, overwrite=overwrite)
    served_name = uploaded.get("name", name)
    served_sub = uploaded.get("subfolder", "")
    return {
        "uploaded": uploaded,
        "hint": (
            "reference it in a LoadImage node's image widget as "
            f"{(served_sub + '/' if served_sub else '') + served_name!r}"
        ),
    }


async def _confirm_destroys_others(
    ctx: Context | None,
    client: ComfyClient,
    action: str,
    prompt_ids: list[str] | None,
    confirm: bool,
) -> dict[str, Any] | None:
    """Confirm interrupt/clear/delete, but ONLY when it would destroy work this
    session did not queue. None = go ahead.

    Precision is the whole point. Draftsman already knows which prompt_ids it
    submitted (_State.submitted), so cleaning up after itself stays silent and
    a blanket confirmation never trains the user to click through the one
    prompt that matters - somebody else's render about to be dropped. An
    unreachable /queue is not a reason to block: the same best-effort posture
    as run_workflow's queue etiquette check.
    """
    if confirm:
        return None  # the user has already been asked
    try:
        queue = await client.get_queue()
        running = [entry[1] for entry in queue.get("queue_running", [])]
        pending = [entry[1] for entry in queue.get("queue_pending", [])]
    except Exception:
        return None
    if action == "interrupt":
        affected = running
    elif action == "clear":
        affected = pending
    else:
        # only ids actually queued can be destroyed; an unknown id is a no-op
        affected = [pid for pid in (prompt_ids or []) if pid in (*running, *pending)]
    foreign = [pid for pid in affected if pid not in _State.submitted]
    if not foreign:
        return None
    return await _confirm(
        ctx,
        f"manage_queue(action='{action}') will discard {len(foreign)} queued render(s) "
        "this session did not submit - most likely the user's own jobs. Continue?",
        f"NOTHING WAS DISCARDED. {len(foreign)} of the affected prompt(s) were queued "
        "outside this session (the user's own jobs, or another agent's). "
        "manage_queue(action='status') lists what is queued and which prompts are "
        "draftsman's; show the user, get an explicit yes, then re-run with confirm=True.",
        "queue",
    )


@mcp.tool(annotations=_DESTRUCTIVE_INSTANCE)
async def manage_queue(
    action: Literal["status", "interrupt", "clear", "delete", "free"],
    prompt_ids: list[str] | None = None,
    unload_models: bool = False,
    confirm: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Inspect or manage the instance's run queue: status (queued prompt ids;
    draftsman_submitted maps the ones THIS session queued to their workflow_id -
    the rest are someone else's job), interrupt (stop the running prompt), clear
    (drop ALL pending), delete (drop given pending prompt_ids), free (release
    cached VRAM/RAM; unload_models=True also unloads models). clear/delete/
    interrupt are gated when they'd discard prompts this session didn't queue;
    confirm=True once the user agrees."""
    client = _client()
    if action == "status":
        queue = await client.get_queue()
        running = [entry[1] for entry in queue.get("queue_running", [])]
        pending = [entry[1] for entry in queue.get("queue_pending", [])]
        result: dict[str, Any] = {"running": running, "pending": pending, "pending_count": len(pending)}
        # attribute whichever of these prompt_ids THIS process queued via
        # run_workflow - the rest are someone else's job (the user's own queue,
        # or queued before this server started), which is exactly the ambiguity
        # that made a run_workflow timeout unreadable without it
        mine = {pid: wf for pid in (*running, *pending) if (wf := _State.submitted.get(pid))}
        if mine:
            result["draftsman_submitted"] = mine
            unattributed = len(running) + len(pending) - len(mine)
            if unattributed:
                result["note"] = (
                    f"{unattributed} queued prompt(s) not in draftsman_submitted - "
                    "not queued by this session (the user's own job, another agent, "
                    "or queued before this server started)"
                )
        return result
    if action in ("interrupt", "clear", "delete"):
        if action == "delete" and not prompt_ids:
            return {"error": "delete requires prompt_ids"}
        refusal = await _confirm_destroys_others(ctx, client, action, prompt_ids, confirm)
        if refusal is not None:
            return refusal
    if action == "interrupt":
        await client.interrupt()
        return {"done": "interrupt sent to the running prompt"}
    if action == "clear":
        await client.clear_queue()
        return {"done": "pending queue cleared"}
    if action == "delete":
        targets = prompt_ids or []
        await client.delete_queue_items(targets)
        return {"done": f"deleted {len(targets)} pending prompt(s)"}
    await client.free(unload_models=unload_models)
    return {"done": "freed memory" + (" and unloaded models" if unload_models else "")}


@mcp.tool(annotations=_WRITE_INSTANCE)
async def save_workflow(
    workflow_id: str,
    name: str,
    allow_invalid: bool = False,
    overwrite: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Save the workflow (UI format, with layout/groups/notes) into ComfyUI's
    workflow browser + the session dir. Run organize_workflow first - this is the
    deliverable. REFUSES to save with validation errors unless allow_invalid=True.
    Never overwrites by default: a taken name saves as '<name> (draftsman)'
    (result.renamed_from says so); overwrite=True replaces deliberately."""
    if ".." in name or any(sep in name for sep in ("/", "\\")):
        return {"error": "name must be a plain filename - no path separators or '..'"}
    wf = _wf(workflow_id)
    object_info = await _object_info(refresh=True)
    findings = validate(wf, object_info)
    errors = [f for f in findings if f["level"] == "error"]
    if errors and not allow_invalid:
        return {
            "saved": False,
            "error": (
                "refusing to save: the workflow has validation errors that would "
                "break for the user - fix them with edit_workflow, or pass "
                "allow_invalid=True to save a known-broken draft anyway"
            ),
            "findings": (capped := _cap_findings(errors)),
            **_subgraph_edit_hint(capped),
        }
    document = wf.to_ui()
    candidates = [name, f"{name} (draftsman)"] + [f"{name} (draftsman {i})" for i in range(2, 21)]
    filename = renamed_from = None
    for candidate in candidates:
        try:
            # Always try without overwrite first, even when overwrite=True: a free
            # name destroys nothing, so there is nothing to confirm. Only the
            # FileExistsError below proves a real file is about to be replaced -
            # asking before that would claim "the current file is lost" about a
            # file that does not exist, and a decline would refuse a harmless save.
            filename = await _client().save_userdata_workflow(candidate, document, overwrite=False)
            renamed_from = None if candidate == name else name
            break
        except FileExistsError:
            if overwrite and candidate == name:
                refusal = await _confirm(
                    ctx,
                    f"Replace the existing workflow '{name}' in ComfyUI's browser? "
                    "The current file is lost.",
                    # no fallback hint: a client that cannot ask goes ahead -
                    # overwrite=True is already an explicit act by the caller
                    None,
                    "overwrite",
                )
                if refusal is not None:
                    return {"saved": False, **refusal}
                filename = await _client().save_userdata_workflow(
                    candidate, document, overwrite=True
                )
                renamed_from = None
                break
            continue
    if filename is None:
        return {
            "saved": False,
            "error": (
                f"'{name}' and 20 draftsman-suffixed variants already exist - "
                "pass a different name, or overwrite=True to replace deliberately"
            ),
        }
    # the ComfyUI-side save above succeeded; a local backup-copy failure
    # (unwritable session dir on a locked-down machine) must not fail the tool
    try:
        local: str | None = str(_session().persist(workflow_id))
        persist_note = ""
    except OSError as e:
        local = None
        persist_note = (
            f"local session copy could not be written ({e}) - the ComfyUI save above "
            "still succeeded; set DRAFTSMAN_SESSION_DIR to a writable path to fix. "
        )
    warnings = lint(wf, object_info, learned_dir=_config().learned_dir)
    return {
        "saved": True,
        "saved_to_comfyui": f"workflows/{filename} (visible in the ComfyUI workflow browser)",
        "renamed_from": renamed_from,
        "local_copy": local,
        "validation": _cap_findings(findings),
        "lint": _cap_lint(warnings),
        "note": persist_note
        + (
            f"'{name}' already existed, so this saved as '{filename}' - the original file is untouched. "
            if renamed_from
            else ""
        )
        + ("" if not warnings else "lint is not clean - consider organize_workflow before delivering"),
    }


@mcp.tool(annotations=_READ_LOCAL)
async def export_workflow_json(
    workflow_id: str, format: Literal["ui", "api"] = "ui"
) -> dict[str, Any]:
    """The workflow as JSON: 'ui' (shareable, opens in the editor, keeps layout &
    notes) or 'api' (for POST /prompt automation)."""
    wf = _wf(workflow_id)
    if format == "api":
        return wf.to_api(await _object_info())
    return wf.to_ui()


# --------------------------------------------------------------------------
# Ecosystem & knowledge
# --------------------------------------------------------------------------


@mcp.tool(annotations=_READ_INSTANCE)
async def resolve_missing_nodes(class_types: list[str]) -> dict[str, Any]:
    """Find which installable node packs provide these node class names (official
    Comfy Registry). THIS is the tool for missing/uninstalled nodes (port_workflow
    is for model-family moves, not missing nodes). Returns pack ids, repos, and
    install hints. Installing custom nodes runs third-party code - surface the
    choice to the user."""
    try:
        return await _registry().resolve_node_classes(class_types)
    except RegistryUnavailableError as e:
        return {"error": str(e), "unresolved": class_types}


@mcp.tool(annotations=_READ_INSTANCE)
async def search_node_packs(query: str) -> list[dict[str, Any]]:
    """Search the Comfy Registry for node packs by capability (e.g. 'face detailer',
    'wildcards', 'video interpolation')."""
    try:
        return await _registry().search_packs(query)
    except RegistryUnavailableError as e:
        return [{"error": str(e)}]


@mcp.tool(annotations=_READ_INSTANCE)
async def get_model_guidance(family: str = "", model_filename: str = "") -> dict[str, Any]:
    """Tuned settings for a model family: sampling (CFG/steps/samplers), native
    resolutions, technique blocks (face_detailer, hires_fix...), prompt style notes.
    Variant-aware: pass model_filename so turbo/lightning/distill overrides apply.
    Includes any learned overlay from past research plus a research directive -
    for brand-new models, verify online and record_learning what you find. A `fit`
    block appears only when this GPU can't comfortably hold the model."""
    learned = _config().learned_dir
    if not family:
        return {"families": knowledge.list_families(learned)}
    try:
        guidance = knowledge.get_guidance(family, model_filename or None, learned_dir=learned)
    except KeyError:
        return {
            "error": f"no knowledge for '{family}'",
            "families": knowledge.list_families(learned),
            "hint": "research current best settings online, then record_learning them",
        }
    # The verdict is best-effort: guidance is the valuable part, and an
    # unreachable instance must never turn a knowledge lookup into an error.
    with contextlib.suppress(Exception):
        await _load_devices()
    fit = _fit(guidance)
    # The raw hardware block (both numbers, prose notes, a URL) would otherwise
    # ride along on EVERY guidance call while being useful only when the verdict
    # is bad. fit_verdict folds what matters into `fit`; the rest is dropped.
    guidance.pop("hardware", None)
    return _cap_sources({**guidance, **({"fit": fit} if fit else {})})


@mcp.tool(annotations=_EDIT_LOCAL)
async def record_learning(family: str, updates: dict[str, Any], source: str) -> dict[str, Any]:
    """Persist researched settings so FUTURE sessions start smarter. updates uses
    the guidance shape, e.g. {"sampling": {"cfg": {"default": 3.5}}} or
    {"techniques": {"face_detailer": {"denoise": 0.4}}}. source = URL/model page.
    Any family name works; for a NEW family also include a "detect" block so it's
    auto-recognized next session: {"detect": {"checkpoint_patterns": ["mymodel"]},
    "loader": "unet_clip_vae"}.

    A "sources" list teaches organize_workflow's Models note where to download
    each file - it never invents a URL, so this is the only way one appears:
    {"sources": [{"match": ["mymodel_v1.safetensors"], "what": "checkpoint",
    "url": "https://..."}]}. Verify the URL resolves before recording it."""
    path = knowledge.save_learning(_config().learned_dir, family, updates, source)
    return {"saved": str(path), "guidance_now": knowledge.get_guidance(family, learned_dir=_config().learned_dir)}


# --------------------------------------------------------------------------
# Prompts & resources
# --------------------------------------------------------------------------


@mcp.prompt()
def build_workflow(request: str) -> str:
    """Guided flow for building a working, optimized, human-readable workflow."""
    return f"""Build a ComfyUI workflow for: {request}

Follow this sequence with the comfy-draftsman tools:
1. get_instance_info - confirm the instance and see known model families.
2. list_templates(search=...) - templates ship with every ComfyUI release and are
   the correct starting topology for current models. Seed with
   create_workflow(template=...) when one fits; blank only for truly custom graphs.
3. list_models to pick from what is actually installed; get_model_guidance(family,
   model_filename) for tuned CFG/steps/sampler/resolution and technique settings.
   If the model is newer than the guidance floor, research current recommendations
   online and record_learning them.
4. Wire with edit_workflow; check unfamiliar nodes via get_node_info first. If a
   needed capability is missing (e.g. FaceDetailer, wildcards), use
   resolve_missing_nodes / search_node_packs and ask the user before installing.
   If the positive prompt is generated (wildcards/concatenators) rather than
   hand-typed, route it through a Show Text node into the encoder so the user
   sees the final prompt text.
5. validate_workflow until ok; fix with edit_workflow.
6. run_workflow - actually render; inspect the preview. Iterate if wrong.
7. organize_workflow - REQUIRED finishing step (layout, groups, notes, knobs).
8. save_workflow - it lands in the user's ComfyUI workflow browser.

The deliverable is a workflow a non-technical person can read: green nodes are
theirs to touch, notes explain what everything does and which settings to leave
alone."""


@mcp.prompt()
def modernize_workflow(problem: str = "an old workflow that no longer works") -> str:
    """Guided flow for repairing or porting an outdated workflow."""
    return f"""Modernize {problem}.

1. import_workflow with the old JSON (UI or API format both work).
2. diagnose_workflow - every incompatibility with the live instance, with fixes:
   renamed/removed nodes, changed widget schemas (widget-count-drift), missing
   model files (closest installed suggestion), missing custom-node packs (registry
   resolution + install hints), and a capability_impact list for missing nodes.
3. Apply fixes via edit_workflow. BEFORE choosing "core nodes only" vs installing
   packs, spell out for the user exactly what each missing node DOES and what
   capability is lost if it's dropped (use diagnose_workflow's capability_impact).
   Get explicit confirmation - never silently drop a feature. Custom nodes execute
   third-party code, so installing is always the user's call.
4. To move to a newer model family (e.g. sdxl -> flux/krea): port_workflow, then
   review its flags - it retunes samplers/techniques and swaps loader topology
   mechanically, and tells you what needs judgment.
5. validate_workflow until ok, run_workflow to prove it renders,
   organize_workflow, then save_workflow."""


@mcp.resource("draftsman://workflow-format")
def workflow_format_cheatsheet() -> str:
    """How ComfyUI workflow JSON works (UI vs API format)."""
    return (
        "ComfyUI has two workflow JSON formats:\n"
        "- UI format (schema 0.4/1.0): nodes[] with pos/size/title/color, links[],\n"
        "  groups[], notes. What the editor loads/saves; keeps all visual organization.\n"
        "- API format: {node_id: {class_type, inputs}} - what POST /prompt executes.\n"
        "  No layout. Widget inputs are named values; connections are [origin_id, slot].\n"
        "comfy-draftsman edits an internal graph and serializes to both; virtual nodes\n"
        "(Note, MarkdownNote, PrimitiveNode, Reroute) exist only in UI format and are\n"
        "resolved away for execution. Muted (mode 2) nodes are skipped; bypassed\n"
        "(mode 4) pass their matching-type inputs through."
    )


@mcp.resource("draftsman://knowledge/{family}")
def knowledge_resource(family: str) -> str:
    """Raw guidance YAML for a model family (floor + learned overlay merged)."""
    return yaml.safe_dump(
        knowledge.get_guidance(family, learned_dir=_config().learned_dir), sort_keys=False
    )


@mcp.resource("draftsman://capabilities")
def capabilities_resource() -> str:
    """What this draftsman process can do for a client right now: whether finished
    renders can be relocated to a caller-reachable folder (the key question for a
    sandboxed Cowork/Desktop client), background runs, and the partner-node API key.
    Read this - or call get_instance_info - before a render you intend to show the
    user, so a missing COMFYUI_MOUNT_DIR is caught before the render, not after."""
    cfg = _config()
    return json.dumps(
        {
            "comfyui_url": cfg.comfyui_url,
            "relocation": _mount_status(),
            # run_workflow(wait=False) queues in the background; poll get_run_status
            "background_runs": True,
            # partner/* nodes (Luma, Kling, Runway, ...) need COMFY_API_KEY set
            "partner_node_api_key": bool(cfg.comfy_api_key),
            "session_dir": str(cfg.session_dir),
            "learned_dir": str(cfg.learned_dir),
        },
        indent=2,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
