"""comfy-draftsman MCP server: tools, prompts, and resources.

All heavy lifting lives in tested modules (graph/, comfy/, knowledge/); this
file is thin wiring. State: one ComfyClient + RegistryClient + Session per
process, created lazily.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, Literal

import yaml
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations

from . import knowledge
from .comfy.catalog import node_summary
from .comfy.catalog import search_nodes as catalog_search
from .comfy.client import ComfyClient, ComfyValidationError
from .comfy.registry import RegistryClient
from .config import Config, load_config
from .graph.annotate import annotate
from .graph.lint import lint
from .graph.model import NOTE_TYPES, VIRTUAL_TYPES, Workflow
from .graph.port import port_workflow as port_engine
from .graph.validate import validate
from .graph.widgets import widget_slot_names
from .session import Session

# Tool annotations let clients reason about safety and, where supported,
# auto-approve safe calls. Read tools that query the live instance are
# read-only + open-world; session-local reads are read-only + closed-world.
# See docs/PERMISSIONS.md for the recommended Claude Code allowlist.
_READ_INSTANCE = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)
_READ_LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False, idempotentHint=True)
_EDIT_LOCAL = ToolAnnotations(readOnlyHint=False, openWorldHint=False, destructiveHint=False)
_WRITE_INSTANCE = ToolAnnotations(readOnlyHint=False, openWorldHint=True, destructiveHint=False)

mcp = FastMCP(
    "comfy-draftsman",
    instructions=(
        "Draft, repair, port, validate, run, and SAVE ComfyUI workflows against the "
        "user's local ComfyUI instance. The finished artifact is always an organized, "
        "labeled workflow: run organize_workflow before save_workflow so a human gets "
        "groups, notes, and highlighted knobs. Ground truth is the live instance "
        "(search_nodes/get_node_info/list_models); templates (list_templates) are the "
        "best starting points for current models. get_node_info accepts a list of "
        "class_types - batch your lookups in ONE call instead of one per node. "
        "To work on a workflow the user already saved in ComfyUI, use "
        "list_workflows then import_workflow(name=...) - never ask them to paste "
        "JSON that's already on the instance. "
        "get_model_guidance returns tuned settings per model family; when you research "
        "better settings online, persist them with record_learning (include a 'detect' "
        "block for brand-new families so they're recognized next session). "
        "When a positive prompt is GENERATED upstream (wildcards, concatenators) "
        "instead of hand-typed, wire it through a Show Text node before the encoder "
        "so the user can see the final prompt (lint flags this as no-prompt-preview). "
        "When modernizing a workflow, if some nodes have no core/installed equivalent, "
        "tell the user exactly what capability would be LOST before they choose "
        "'core nodes only' vs installing a pack - don't drop features silently."
    ),
)


class _State:
    config: Config | None = None
    client: ComfyClient | None = None
    registry: RegistryClient | None = None
    session: Session | None = None


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


async def _object_info(refresh: bool = False) -> dict[str, Any]:
    return await _client().get_object_info(refresh=refresh)


def _wf(workflow_id: str) -> Workflow:
    return _session().get(workflow_id)


def _widget_preview(n) -> Any:
    values = n.widgets_values if isinstance(n.widgets_values, list) else dict(n.widgets_values)
    if n.type in VIRTUAL_TYPES and isinstance(values, list):
        # note text can be long - truncate the preview, the content is intact
        values = [
            v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v for v in values
        ]
    return values


def _summary(workflow_id: str, wf: Workflow) -> dict[str, Any]:
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
            }
            for n in sorted(wf.nodes.values(), key=lambda x: x.id)
        ],
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
    """ComfyUI version, OS, VRAM, queue length of the connected instance. Call first."""
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
        "knowledge_families": knowledge.list_families(_config().learned_dir),
    }


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
    class_type: str = "", class_types: list[str] | None = None
) -> dict[str, Any]:
    """Full input/output schema for node classes: slot names, types, widget
    defaults/ranges, combo choices (truncated), tooltips.

    BATCH your lookups: pass class_types=["A", "B", "C"] to fetch many in ONE
    call (returns {class_type: schema}) instead of one call per node. A single
    class_type=... still returns that one node's schema directly.
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
            results[name] = node_summary(object_info, name)
        except KeyError:
            results[name] = {
                "error": f"'{name}' is not installed on this instance",
                "hint": "resolve_missing_nodes can find which pack provides it",
            }
    if class_types is None and class_type:  # single-lookup back-compat shape
        return results[class_type]
    return results


@mcp.tool(annotations=_READ_INSTANCE)
async def list_models(folder: str = "checkpoints", search: str = "") -> dict[str, Any]:
    """Model files installed on the instance. `folder` picks the model type:
    checkpoints, loras, vae, diffusion_models, text_encoders, upscale_models,
    controlnet, embeddings, ... (unknown folder -> the full available list).
    `search` filters filenames (case-insensitive substring)."""
    folders = await _client().list_model_folders()
    if folder not in folders:
        return {"error": f"unknown folder '{folder}'", "available": folders}
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
    """Compact view of a session workflow: nodes (id/class/title/widgets), links, groups."""
    return _summary(workflow_id, _wf(workflow_id))


# edit_workflow op schemas: op -> (required keys, optional keys). Validated
# up front so a malformed op fails with the schema spelled out instead of a
# raw KeyError, and misspelled keys (widgets_values, node, ...) are rejected
# instead of silently ignored.
_OP_SPECS: dict[str, tuple[set[str], set[str]]] = {
    "add_node": ({"class_type"}, {"title", "widgets"}),
    "remove_node": ({"node_id"}, set()),
    "connect": ({"from_node", "from_output", "to_node", "to_input"}, set()),
    "set_widget": ({"node_id", "input", "value"}, set()),
    "set_title": ({"node_id", "title"}, set()),
    "set_mode": ({"node_id", "mode"}, set()),
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
async def edit_workflow(workflow_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply batched edits. Each op is a dict with 'op' plus:

    - {"op": "add_node", "class_type": str, "title"?: str, "widgets"?: {name: value}}
    - {"op": "remove_node", "node_id": int}
    - {"op": "connect", "from_node": int, "from_output": str|int, "to_node": int, "to_input": str}
    - {"op": "set_widget", "node_id": int, "input": str, "value": any}
    - {"op": "set_title", "node_id": int, "title": str}
    - {"op": "set_mode", "node_id": int, "mode": int}  # 0 normal, 2 mute, 4 bypass

    Output slot names and widget names come from get_node_info. Annotation nodes
    (class_type "Note" or "MarkdownNote") are supported with a single widget
    'text'. Ops apply in order; a failing op stops the batch, reports what
    succeeded, and leaves the graph unchanged by the failing op.
    """
    wf = _wf(workflow_id)
    object_info = await _object_info()
    applied: list[str] = []
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
                    slots = widget_slot_names(class_type, object_info)
                    if bad := sorted(set(widgets) - set(slots)):
                        raise ValueError(
                            f"{class_type} has no widget(s) {bad}; widgets: {slots}; "
                            "graph unchanged"
                        )
                node = wf.add_node(class_type, object_info=object_info, title=op.get("title"))
                for name, value in widgets.items():
                    wf.set_widget(node.id, name, value, object_info)
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
                applied.append(
                    f"connected #{op['from_node']}.{op['from_output']} -> "
                    f"#{op['to_node']}.{op['to_input']}{replaced}"
                )
            elif kind == "set_widget":
                wf.set_widget(int(op["node_id"]), op["input"], op["value"], object_info)
                applied.append(f"set #{op['node_id']}.{op['input']} = {op['value']!r}")
            elif kind == "set_title":
                wf.nodes[int(op["node_id"])].title = op["title"]
                applied.append(f"titled #{op['node_id']}")
            elif kind == "set_mode":
                wf.nodes[int(op["node_id"])].mode = int(op["mode"])
                applied.append(f"mode #{op['node_id']} = {op['mode']}")
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
    return {"applied": applied, "summary": _summary(workflow_id, wf)}


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
        "findings": findings,
    }


@mcp.tool()
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
        for cls in missing:
            pack = resolved.get(cls)
            capability_impact.append(
                {
                    "class_type": cls,
                    "provided_by": pack,
                    "options": (
                        f"install pack '{pack}' (runs third-party code) to keep this "
                        "node's capability, OR replace it with a core/already-installed "
                        "equivalent, OR drop it and lose what it does"
                        if pack
                        else "not found in the registry - find a core/installed "
                        "equivalent, or dropping it loses what it does"
                    ),
                }
            )
    return {
        "ok": not findings,
        "findings": findings,
        "missing_node_packs": registry_result,
        "capability_impact": capability_impact,
        "capability_notice": (
            "For each missing node, tell the user WHAT FUNCTION is lost if it's dropped "
            "and whether a core/installed equivalent exists, BEFORE they choose "
            "'core nodes only' vs installing a pack. Never drop a feature silently."
        )
        if missing
        else "",
        "family": knowledge.detect_family(
            wf, await _object_info(), learned_dir=_config().learned_dir
        ),
    }


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


@mcp.tool()
async def run_workflow(
    workflow_id: str, timeout_seconds: float = 600, return_preview: bool = True
) -> Any:
    """Queue the workflow on the instance and wait for completion. Returns status,
    node errors if it failed, output files, and (optionally) a preview image so you
    can SEE the result. Prove a workflow works before saving/delivering it."""
    wf = _wf(workflow_id)
    object_info = await _object_info()
    errors = [f for f in validate(wf, object_info) if f["level"] == "error"]
    if errors:
        return {
            "status": "invalid",
            "findings": errors,
            "hint": "fix with edit_workflow (diagnose_workflow for missing node classes)",
        }
    try:
        api = wf.to_api(object_info)
    except ValueError as e:
        return {"status": "invalid", "error": str(e)}
    try:
        result = await _client().run_and_wait(api, timeout=timeout_seconds)
    except ComfyValidationError as e:
        return {"status": "rejected", "error": str(e), "node_errors": e.node_errors}
    if return_preview and result["status"] == "success":
        image_items = [o for o in result["outputs"] if o.get("kind") == "images"]
        if image_items:
            data = await _client().fetch_output(image_items[0])
            if len(data) < 1_500_000:
                return [result, Image(data=data, format="png")]
    return result


@mcp.tool(annotations=_WRITE_INSTANCE)
async def save_workflow(
    workflow_id: str, name: str, allow_invalid: bool = False, overwrite: bool = False
) -> dict[str, Any]:
    """Save the workflow (UI format, with all layout/groups/notes) into ComfyUI's
    workflow browser and to the session dir. Run organize_workflow first so the
    saved artifact is readable. This is the deliverable.

    NEVER overwrites an existing workflow file by default: if `name` is taken
    (e.g. you're saving an edited copy of the user's workflow), the save lands
    under '<name> (draftsman)' so their original is preserved - the result's
    renamed_from tells you when that happened. Pass overwrite=True only when the
    user explicitly wants the existing file replaced.

    Validates against the live instance first and REFUSES to save if there are
    validation errors (a broken deliverable is worse than no save) - fix them
    with edit_workflow, or pass allow_invalid=True to save a known-broken draft."""
    if ".." in name or any(sep in name for sep in ("/", "\\")):
        return {"error": "name must be a plain filename - no path separators or '..'"}
    wf = _wf(workflow_id)
    object_info = await _object_info()
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
    local = _session().persist(workflow_id)
    warnings = lint(wf, object_info)
    return {
        "saved": True,
        "saved_to_comfyui": f"workflows/{filename} (visible in the ComfyUI workflow browser)",
        "renamed_from": renamed_from,
        "local_copy": str(local),
        "validation": findings,
        "lint": warnings,
        "note": (
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
    """Persist researched settings so FUTURE sessions start smarter. updates uses the
    guidance shape, e.g. {"sampling": {"cfg": {"default": 3.5}}} or
    {"techniques": {"face_detailer": {"denoise": 0.4, "cfg": 1.0}}} or
    {"notes": {"sampling": "..."}}. source = where you learned it (URL/model page).

    Works for brand-new families too - pass any family name. For a NEW family,
    also include a "detect" block so the server RECOGNIZES it automatically next
    session (otherwise it'll re-misdetect as a lookalike family):
    {"detect": {"checkpoint_patterns": ["mymodel"]}, "loader": "unet_clip_vae"}."""
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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
