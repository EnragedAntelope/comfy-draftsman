"""comfy-draftsman MCP server: tools, prompts, and resources.

All heavy lifting lives in tested modules (graph/, comfy/, knowledge/); this
file is thin wiring. State: one ComfyClient + RegistryClient + Session per
process, created lazily.
"""

from __future__ import annotations

import base64
import contextlib
import difflib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations

from . import knowledge
from .comfy.catalog import metadata_digest, node_summary
from .comfy.catalog import search_nodes as catalog_search
from .comfy.client import ComfyClient, ComfyConnectionError, ComfyValidationError
from .comfy.progress import ProgressTracker
from .comfy.registry import RegistryClient
from .config import Config, load_config
from .graph.annotate import annotate
from .graph.lint import lint
from .graph.model import NOTE_TYPES, VIRTUAL_TYPES, Workflow
from .graph.port import port_workflow as port_engine
from .graph.validate import check_widget_value, validate
from .graph.widgets import SYNTHETIC_SUFFIXES, all_slot_names
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

# Comfy Org API key for partner/* nodes (Luma, Seedance, Kling, Runway).
# The frontend normally injects this into the prompt payload's extra_data;
# without it, headless MCP queues fail with "Unauthorized" on partner nodes.
_COMFY_API_KEY = os.environ.get("COMFY_API_KEY", "")

mcp = FastMCP(
    "comfy-draftsman",
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


class _State:
    config: Config | None = None
    client: ComfyClient | None = None
    registry: RegistryClient | None = None
    session: Session | None = None
    tracker: ProgressTracker | None = None


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


def _clip(v: Any) -> Any:
    return v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v


_LEVEL_RANK = {"error": 0, "warning": 1, "info": 2}
_FINDINGS_CAP = 40


def _cap_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort findings most-severe first and cap the number returned to the model,
    appending a marker if truncated. Every error is always kept - only lower
    levels are trimmed - so nothing blocking is hidden and tokens stay bounded."""
    ordered = sorted(findings, key=lambda f: _LEVEL_RANK.get(f.get("level"), 3))
    if len(ordered) <= _FINDINGS_CAP:
        return ordered
    errors = [f for f in ordered if f.get("level") == "error"]
    keep = max(_FINDINGS_CAP, len(errors))  # never drop an error to make room
    capped = ordered[:keep]
    capped.append(
        {
            "level": "info",
            "code": "findings-truncated",
            "message": f"…{len(ordered) - len(capped)} more finding(s) omitted; fix "
            "the ones above first, then re-validate",
        }
    )
    return capped


def _widget_preview(n) -> Any:
    # prompts/wildcards/note text can be multi-KB and summaries are re-sent on
    # every inspect/edit - truncate the preview, the graph content is intact
    # (export_workflow_json shows full values)
    if isinstance(n.widgets_values, list):
        return [_clip(v) for v in n.widgets_values]
    return {k: _clip(v) for k, v in dict(n.widgets_values).items()}


def _subgraph_summary(sg: dict[str, Any]) -> dict[str, Any]:
    """Readable view of one subgraph definition: inner nodes with widget
    previews, and inner wiring - enough to use the subgraph as reference or
    rebuild it flat."""
    nodes = {n["id"]: n for n in sg.get("nodes", []) or [] if "id" in n}

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
        "inputs": [i.get("name") for i in sg.get("inputs", []) or []],
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
            and wf.nodes[ln.origin_id].type not in VIRTUAL_TYPES
            and wf.nodes[ln.target_id].type not in VIRTUAL_TYPES
        ],
        "groups": [g.title for g in wf.groups],
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
    devices = [
        {"name": d.get("name"), "vram_total": d.get("vram_total"), "vram_free": d.get("vram_free")}
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
    (COMFYUI_MOUNT_DIR), and is the partner-node key present. Unlike get_instance_info
    this never raises - a down instance is reported as a failed check, not an error -
    so run it first when a render can't be delivered or the instance seems unreachable.
    Returns {ok, checks:[{name, ok, detail}], hint?}. `ok` is gated on ComfyUI being
    reachable; relocation is a soft check (workflows still run, but auto-delivery needs
    it) surfaced via `hint`."""
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

    # Soft requirement: can finished renders be relocated to a caller-reachable folder?
    reloc = _mount_status()
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
            "detail": "set" if _COMFY_API_KEY else "unset (only needed for partner/* nodes)",
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
        for hit in results:
            hit["schema"] = node_summary(object_info, hit["class_type"])
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
                "hint": "resolve_missing_nodes can find which pack provides it",
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
    result = {"folder": folder, "count": len(files), "files": files}
    if search:
        result["search"] = search
    return result


@mcp.tool(annotations=_READ_INSTANCE)
async def list_templates(search: str = "") -> list[dict[str, Any]]:
    """ComfyUI's bundled workflow templates - the best starting points for current
    models (they ship with every release). Seed one via create_workflow(template=...)."""
    index = await _client().get_template_index()
    out = []
    for module in index:
        for template in module.get("templates", []):
            entry = {
                "name": template.get("name"),
                "title": template.get("title"),
                "description": (template.get("description") or "")[:160],
                "models": template.get("models", []),
                "category": module.get("title"),
            }
            haystack = json.dumps(entry).lower()
            if not search or search.lower() in haystack:
                out.append(entry)
    return out[:60]


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
        data = json.loads(workflow_json)
    if "nodes" in data:
        wf = Workflow.from_ui(data)
    else:
        wf = Workflow.from_api(data, await _object_info())
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
        summary["subgraphs"] = [_subgraph_summary(sg) for sg in subgraphs.values()]
        summary["subgraph_note"] = (
            "subgraph instances run FLATTENED - validate/run/export expand them "
            "automatically; edit_workflow ops don't reach inside, rebuild flat "
            "to modify internals"
        )
    return summary


# edit_workflow op schemas: op -> (required keys, optional keys). Validated
# up front so a malformed op fails with the schema spelled out instead of a
# raw KeyError, and misspelled keys (widgets_values, node, ...) are rejected
# instead of silently ignored.
_OP_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "add_node": ({"class_type"}, {"title", "widgets", "force"}),
    "remove_node": ({"node_id"}, set()),
    "connect": ({"from_node", "from_output", "to_node", "to_input"}, set()),
    "set_widget": ({"node_id", "input", "value"}, {"force"}),
    "set_title": ({"node_id", "title"}, set()),
    "set_mode": ({"node_id", "mode"}, set()),
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
    - {"op": "connect_in_definition", "definition_id": str, "from_node": int, "from_output": str|int, "to_node": int, "to_input": str}
    - {"op": "add_node_to_definition", "definition_id": str, "class_type": str, "title"?: str, "widgets"?: {name: value}, "force"?: bool}
    - {"op": "remove_node_from_definition", "definition_id": str, "node_id": int}
    - {"op": "set_title_in_definition", "definition_id": str, "node_id": int, "title": str}
    - {"op": "set_mode_in_definition", "definition_id": str, "node_id": int, "mode": int}
    - {"op": "set_widget_in_definition", "definition_id": str, "node_id": int, "input": str, "value": any, "force"?: bool}


    Output slot names and widget names come from get_node_info. Annotation nodes
    (class_type "Note" or "MarkdownNote") are supported with a single widget
    'text'. Ops apply in order; a failing op stops the batch, reports what
    succeeded, and leaves the graph unchanged by the failing op.

    Widget VALUES are checked against the live schema at write time (combo
    choices, ranges, types) - an invalid value fails the op with suggestions;
    add "force": true to a set_widget/add_node op to skip that check.

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
                if target is not None:
                    slot = target.input_by_name(op["to_input"])
                    if slot is not None and slot.link is not None:
                        old = wf.links.get(slot.link)
                        if old is not None:
                            replaced = f" (replaced existing link from #{old.origin_id}[{old.origin_slot}])"
                wf.connect(
                    int(op["from_node"]),
                    from_output,
                    int(op["to_node"]),
                    op["to_input"],
                    object_info,
                )
                touched.update((int(op["from_node"]), int(op["to_node"])))
                applied.append(
                    f"connected #{op['from_node']}.{op['from_output']} -> "
                    f"#{op['to_node']}.{op['to_input']}{replaced}"
                )
            elif kind == "set_widget":
                node_id = int(op["node_id"])
                node = wf.nodes[node_id]
                if not op.get("force") and node.type not in NOTE_TYPES:
                    problem = check_widget_value(
                        node.type, op["input"], op["value"], object_info,
                        node.widgets_values,
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
    report = annotate(wf, object_info, learned_dir=_config().learned_dir)
    report["note"] = (
        "workflow layout updated in place; save_workflow persists it, "
        "export_workflow_json shows the reorganized graph"
    )
    report["lint"] = lint(wf, object_info)
    return report


@mcp.tool(annotations=_READ_INSTANCE)
async def lint_workflow(workflow_id: str) -> list[dict[str, Any]]:
    """Readability/wiring lint: unlabeled prompts, missing groups/notes, orphan
    nodes, unconnected required inputs, overlapping nodes. Empty list = clean."""
    return lint(_wf(workflow_id), await _object_info())


@mcp.tool(annotations=_READ_INSTANCE)
async def validate_workflow(workflow_id: str) -> dict[str, Any]:
    """Validate against the LIVE instance: node classes installed, widget values in
    range, combo/model-file values actually present (with closest-match suggestions),
    required inputs connected. Fix errors before run_workflow."""
    findings = validate(_wf(workflow_id), await _object_info(refresh=True))
    return {
        "ok": not any(f["level"] == "error" for f in findings),
        "findings": _cap_findings(findings),
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
        registry_result = await _registry().resolve_node_classes(missing)
        resolved = registry_result.get("resolved", {})
        capability_impact = [
            {"class_type": cls, "provided_by": resolved.get(cls)} for cls in missing
        ]
    result: dict[str, Any] = {
        "ok": not findings,
        "findings": _cap_findings(findings),
        "missing_node_packs": registry_result,
        "capability_impact": capability_impact,
        "family": knowledge.detect_family(
            wf, await _object_info(), learned_dir=_config().learned_dir
        ),
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
    report["validate"] = validate(wf, await _object_info())
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
) -> Any:
    """Queue the workflow and (by default) wait for completion. Returns status,
    node errors if it failed, output file refs, and an inline preview thumbnail so
    you can SEE the result (view_output fetches full size / other outputs).
    wait=False returns {status: queued, prompt_id} immediately - poll
    get_run_status(prompt_id). Prove a workflow works before saving/delivering.

    roll_seeds=True (default) mirrors the browser: any seed whose
    control_after_generate is randomize/increment/decrement is re-rolled before
    submit and the new value persisted (the raw /prompt API never does this, so
    headless runs would otherwise repeat the same seed forever). Pass
    roll_seeds=False for a deterministic re-run of the exact stored seeds.

    allow_invalid=True submits even when the local validator reports errors
    (ComfyUI is the final judge; use it if a valid graph is being wrongly
    blocked). save_dir (or, when empty, the configured COMFYUI_MOUNT_DIR)
    relocates every finished image out of ComfyUI's output tree into a folder
    the caller can reach, returning saved_paths - so a render is presentable in
    one call without a separate save_output step.

    front: None (default) checks the queue first - if >=2 prompts are already
    pending, NOTHING is queued and {status: queue_busy} comes back so the user
    can choose. front=True queues this run to go next (existing pending jobs are
    untouched, never deleted); front=False waits at the back of the line."""
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
    if roll_seeds and wf.apply_seed_control(object_info):
        # persist so inspect_workflow reflects what ran and increment/decrement
        # advance across calls; best-effort (a read-only session dir shouldn't
        # block the run)
        with contextlib.suppress(OSError):
            _session().persist(workflow_id)
    if not allow_invalid:
        errors = [f for f in validate(wf, object_info) if f["level"] == "error"]
        if errors:
            return {
                "status": "invalid",
                "findings": errors,
                "hint": "fix with edit_workflow (diagnose_workflow for missing node "
                "classes), or run_workflow(allow_invalid=True) to submit anyway",
            }
    try:
        api = wf.to_api(object_info)
    except ValueError as e:
        return {"status": "invalid", "error": str(e)}
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
    if _COMFY_API_KEY:
        extra_data = {"api_key_comfy_org": _COMFY_API_KEY}
    if not wait:
        tracker = _tracker()
        tracker.ensure_running()
        try:
            queued = await _client().queue_prompt(
                api, extra_data=extra_data, client_id=tracker.client_id, front=bool(front)
            )
        except ComfyValidationError as e:
            return {"status": "rejected", "error": str(e), "node_errors": e.node_errors}
        response = {"status": "queued", "prompt_id": queued["prompt_id"]}
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
        image_items = [o for o in result["outputs"] if o.get("kind") == "images"]
        saved, save_errors = await _relocate_outputs(_client(), image_items, dest_root)
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


def _mount_status() -> dict[str, Any]:
    """Relocation readiness for a sandboxed caller (Cowork/Desktop/Code). A render
    can only be handed to the user if COMFYUI_MOUNT_DIR points at a folder BOTH
    this server and the caller can see. We verify our half (configured, resolves,
    writable via a probe file); the shared-view half is the operator's to set up.
    Returned by get_instance_info and the draftsman://capabilities resource so an
    agent can check up front instead of discovering it after a wasted render."""
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
    """Copy a finished render out of ComfyUI's output tree into a folder the
    caller (e.g. a Claude Desktop / Cowork sandbox) can reach. ComfyUI's save
    nodes only write inside its own output/ dir and reject absolute paths, so a
    relocation step is needed before an image can be presented or edited.

    Provide prompt_id (relocates every output image of that finished job) OR an
    explicit filename (+subfolder/type, as reported in a run's outputs).
    dest_dir defaults to the configured COMFYUI_MOUNT_DIR. dest_filename renames
    a single saved image. Returns {saved_paths, dest_dir}."""
    if not prompt_id and not filename:
        return {"error": "provide prompt_id or filename - nothing to relocate otherwise"}
    client = _client()
    if prompt_id:
        history = await client.get_history(prompt_id)
        if not history:
            return {"error": f"no finished job {prompt_id!r} in history (still running?)"}
        items = [o for o in client._collect_outputs(history) if o.get("kind") == "images"]
        if not items:
            return {"error": f"job {prompt_id!r} produced no output images to relocate"}
    else:
        problem = _check_output_ref(filename, subfolder)
        if problem:
            return {"error": problem}
        items = [{"filename": filename, "subfolder": subfolder, "type": type}]
    if dest_filename and len(items) > 1:
        return {
            "error": "dest_filename can't rename a multi-image batch; relocate one "
            "at a time (pass an explicit filename) to rename"
        }
    dest_root, dest_error = _resolve_dest(dest_dir)
    if dest_error:
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
    """Status of a run queued with run_workflow(wait=False): queue position, live
    step progress while sampling, and outputs (+ error details) once finished."""
    client = _client()
    history = await client.get_history(prompt_id)
    if history:
        error = _history_error(history)
        result: dict[str, Any] = {
            "status": "error" if error else "success",
            "prompt_id": prompt_id,
            "outputs": client._collect_outputs(history),
        }
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
                    and (info.get(n.get("class_type")) or {}).get("output_node")
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
        return {"status": "running", "prompt_id": prompt_id, **snapshot}
    if prompt_id in pending:
        return {
            "status": "pending",
            "prompt_id": prompt_id,
            "queue_position": pending.index(prompt_id) + 1,
            "queue_pending": len(pending),
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


@mcp.tool(annotations=_DESTRUCTIVE_INSTANCE)
async def manage_queue(
    action: Literal["status", "interrupt", "clear", "delete", "free"],
    prompt_ids: list[str] | None = None,
    unload_models: bool = False,
) -> dict[str, Any]:
    """Inspect or manage the instance's run queue: status (prompt ids only),
    interrupt (stop the currently running prompt), clear (drop ALL pending),
    delete (drop specific pending prompt_ids), free (release cached VRAM/RAM;
    unload_models=True also unloads models). clear/delete/interrupt discard
    other queued work - make sure that's what the user wants."""
    client = _client()
    if action == "status":
        queue = await client.get_queue()
        running = [entry[1] for entry in queue.get("queue_running", [])]
        pending = [entry[1] for entry in queue.get("queue_pending", [])]
        return {"running": running, "pending": pending, "pending_count": len(pending)}
    if action == "interrupt":
        await client.interrupt()
        return {"done": "interrupt sent to the running prompt"}
    if action == "clear":
        await client.clear_queue()
        return {"done": "pending queue cleared"}
    if action == "delete":
        if not prompt_ids:
            return {"error": "delete requires prompt_ids"}
        await client.delete_queue_items(prompt_ids)
        return {"done": f"deleted {len(prompt_ids)} pending prompt(s)"}
    await client.free(unload_models=unload_models)
    return {"done": "freed memory" + (" and unloaded models" if unload_models else "")}


@mcp.tool(annotations=_WRITE_INSTANCE)
async def save_workflow(
    workflow_id: str, name: str, allow_invalid: bool = False, overwrite: bool = False
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
            "findings": errors,
        }
    document = wf.to_ui()
    candidates = [name, f"{name} (draftsman)"] + [f"{name} (draftsman {i})" for i in range(2, 21)]
    filename = renamed_from = None
    for candidate in candidates:
        try:
            filename = await _client().save_userdata_workflow(candidate, document, overwrite=overwrite)
            renamed_from = None if candidate == name else name
            break
        except FileExistsError:
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
    warnings = lint(wf, object_info)
    return {
        "saved": True,
        "saved_to_comfyui": f"workflows/{filename} (visible in the ComfyUI workflow browser)",
        "renamed_from": renamed_from,
        "local_copy": local,
        "validation": findings,
        "lint": warnings,
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
    return await _registry().resolve_node_classes(class_types)


@mcp.tool(annotations=_READ_INSTANCE)
async def search_node_packs(query: str) -> list[dict[str, Any]]:
    """Search the Comfy Registry for node packs by capability (e.g. 'face detailer',
    'wildcards', 'video interpolation')."""
    return await _registry().search_packs(query)


@mcp.tool(annotations=_READ_LOCAL)
async def get_model_guidance(family: str = "", model_filename: str = "") -> dict[str, Any]:
    """Tuned settings for a model family: sampling (CFG/steps/samplers), native
    resolutions, technique blocks (face_detailer, hires_fix...), prompt style notes.
    Variant-aware: pass model_filename so turbo/lightning/distill overrides apply.
    Includes any learned overlay from past research plus a research directive -
    for brand-new models, verify online and record_learning what you find."""
    learned = _config().learned_dir
    if not family:
        return {"families": knowledge.list_families(learned)}
    try:
        return knowledge.get_guidance(family, model_filename or None, learned_dir=learned)
    except KeyError:
        return {
            "error": f"no knowledge for '{family}'",
            "families": knowledge.list_families(learned),
            "hint": "research current best settings online, then record_learning them",
        }


@mcp.tool(annotations=_EDIT_LOCAL)
async def record_learning(family: str, updates: dict[str, Any], source: str) -> dict[str, Any]:
    """Persist researched settings so FUTURE sessions start smarter. updates uses
    the guidance shape, e.g. {"sampling": {"cfg": {"default": 3.5}}} or
    {"techniques": {"face_detailer": {"denoise": 0.4}}}. source = URL/model page.
    Any family name works; for a NEW family also include a "detect" block so it's
    auto-recognized next session: {"detect": {"checkpoint_patterns": ["mymodel"]},
    "loader": "unet_clip_vae"}."""
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
            "partner_node_api_key": bool(_COMFY_API_KEY),
            "session_dir": str(cfg.session_dir),
            "learned_dir": str(cfg.learned_dir),
        },
        indent=2,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
