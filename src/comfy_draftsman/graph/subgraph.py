"""Flatten schema-1.0 subgraph instances for API serialization and validation.

ComfyUI's frontend expands subgraph instances client-side at queue time; the
backend only ever sees plain nodes. This module mirrors that expansion so
run_workflow / export(api) / validate work on subgraph-packaged workflows.

Boundary semantics (verified against the bundled subgraph-packaged templates,
e.g. 01_get_started_text_to_image):

- A definition's "inputs"/"outputs" lists are its boundary slots. Inner links
  use pseudo node ids -10 (input boundary) / -20 (output boundary), with the
  boundary-side slot index pointing into those lists.
- The instance node exposes only SOME boundary inputs as sockets, so instance
  input slots are matched to def inputs BY NAME (never by position). Instance
  outputs mirror def outputs by name (positional fallback).
- Widget promotion: instance properties.proxyWidgets is a list of
  [innerNodeId(str), widgetName]; a non-empty instance widgets_values zips
  positionally over proxyWidgets and overrides the inner nodes' own widget
  values. Bundled templates ship an empty list (inner defaults hold).
- A boundary input feeding an inner *widget* input that has no external link
  simply falls back to the inner node's own widget value.
"""

from __future__ import annotations

from typing import Any

from . import widgets as w
from .model import MODE_NORMAL, Link, Workflow

BOUNDARY_INPUT = -10
BOUNDARY_OUTPUT = -20
MAX_DEPTH = 10


def has_subgraph_instances(wf: Workflow) -> bool:
    """True if any *active* node is a subgraph instance. Muted/bypassed
    instances don't count: to_api skips/traces through them unexpanded."""
    defs = wf.subgraph_defs()
    return bool(defs) and any(
        n.type in defs and n.mode == MODE_NORMAL for n in wf.nodes.values()
    )


def flatten(
    wf: Workflow, object_info: dict[str, Any]
) -> tuple[Workflow, dict[int, dict[str, Any]]]:
    """Expanded copy of ``wf`` with every active subgraph instance replaced by
    its inner nodes, plus provenance for reporting:
    {new node id: {"path": "104:90", "subgraph": name, "instance": 104}}
    ("path" uses the frontend's instanceId:innerId convention).

    Nested subgraphs expand iteratively (an inner node whose type is another
    definition uuid becomes an instance in the next pass). Raises ValueError
    on malformed definitions or nesting deeper than MAX_DEPTH.
    """
    defs = wf.subgraph_defs()
    flat = Workflow.from_ui(wf.to_ui())
    provenance: dict[int, dict[str, Any]] = {}
    depth = 0
    while True:
        instances = [
            n for n in list(flat.nodes.values())
            if n.type in defs and n.mode == MODE_NORMAL
        ]
        if not instances:
            break
        if depth >= MAX_DEPTH:
            raise ValueError(f"subgraphs nested deeper than {MAX_DEPTH} levels")
        for inst in instances:
            _expand(flat, inst.id, defs[inst.type], object_info, provenance)
        depth += 1
    return flat, provenance


def _expand(
    flat: Workflow,
    inst_id: int,
    sg: dict[str, Any],
    object_info: dict[str, Any],
    provenance: dict[int, dict[str, Any]],
) -> None:
    inst = flat.nodes[inst_id]
    name = sg.get("name") or sg.get("id")
    if not isinstance(sg.get("nodes"), list) or not sg["nodes"]:
        raise ValueError(f"subgraph '{name}': definition has no inner nodes")
    inner = Workflow.from_ui({"nodes": sg["nodes"], "links": sg.get("links") or []})
    parent = provenance.get(inst.id)
    prefix = parent["path"] if parent else str(inst.id)

    # widget promotion: instance values override inner widgets positionally
    proxies = inst.properties.get("proxyWidgets") or []
    values = inst.widgets_values if isinstance(inst.widgets_values, list) else []
    for pair, value in zip(proxies, values, strict=False):
        try:
            inner_id, widget_name = int(pair[0]), str(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        node = inner.nodes.get(inner_id)
        if node is None or node.type not in object_info:
            continue
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        key = widget_name if widget_name in named else next(
            (k for k in named if k.endswith("__" + widget_name)), None
        )
        if key is not None:
            named[key] = value
            node.widgets_values = w.named_to_widgets(node.type, named, object_info)

    # what feeds each boundary input, from the instance's external wiring
    # (instance sockets are a named subset of the def's inputs)
    ext_feed: dict[int, tuple[int, int]] = {}
    for k, bslot in enumerate(sg.get("inputs") or []):
        socket = inst.input_by_name(str(bslot.get("name", "")))
        if socket is not None and socket.link is not None:
            ext = flat.links.get(socket.link)
            if ext is not None:
                ext_feed[k] = (ext.origin_id, ext.origin_slot)

    # move inner nodes into flat under fresh ids, link refs rebuilt below
    id_map: dict[int, int] = {}
    for node in list(inner.nodes.values()):
        new_id = flat._next_node_id
        flat._next_node_id += 1
        id_map[node.id] = new_id
        provenance[new_id] = {
            "path": f"{prefix}:{node.id}",
            "subgraph": name,
            "instance": inst.id,
        }
        node.id = new_id
        for slot in node.inputs:
            slot.link = None
        for out in node.outputs:
            out.links = []
        flat.nodes[new_id] = node

    def add_link(
        origin_id: int, origin_slot: int, target_id: int, target_slot: int, ltype: str
    ) -> None:
        origin = flat.nodes.get(origin_id)
        target = flat.nodes.get(target_id)
        if origin is None or target is None or target_slot >= len(target.inputs):
            return
        lid = flat._next_link_id
        flat._next_link_id += 1
        flat.links[lid] = Link(lid, origin_id, origin_slot, target_id, target_slot, ltype)
        target.inputs[target_slot].link = lid
        if origin_slot < len(origin.outputs):
            origin.outputs[origin_slot].links.append(lid)

    boundary_out: dict[int, tuple[int, int]] = {}
    for ln in inner.links.values():
        if ln.origin_id == BOUNDARY_INPUT:
            feed = ext_feed.get(ln.origin_slot)
            if feed is not None and ln.target_id in id_map:
                add_link(feed[0], feed[1], id_map[ln.target_id], ln.target_slot, ln.type)
            # no external feed: an inner widget input keeps its widget value;
            # a non-widget required input stays unconnected (validate reports it)
        elif ln.target_id == BOUNDARY_OUTPUT:
            if ln.origin_id in id_map:
                boundary_out[ln.target_slot] = (id_map[ln.origin_id], ln.origin_slot)
        elif ln.origin_id in id_map and ln.target_id in id_map:
            add_link(
                id_map[ln.origin_id], ln.origin_slot,
                id_map[ln.target_id], ln.target_slot, ln.type,
            )

    # rewire external consumers of the instance's outputs to the inner producer
    def_outputs = sg.get("outputs") or []
    for k, out_slot in enumerate(inst.outputs):
        b = next(
            (i for i, o in enumerate(def_outputs) if o.get("name") == out_slot.name), k
        )
        producer = boundary_out.get(b)
        for lid in list(out_slot.links):
            ext = flat.links.get(lid)
            if ext is None:
                continue
            if producer is None:
                # no inner producer for this boundary output: drop the dangler
                flat.links.pop(lid, None)
                tgt = flat.nodes.get(ext.target_id)
                if (
                    tgt is not None
                    and ext.target_slot < len(tgt.inputs)
                    and tgt.inputs[ext.target_slot].link == lid
                ):
                    tgt.inputs[ext.target_slot].link = None
                continue
            ext.origin_id, ext.origin_slot = producer
            pnode = flat.nodes[producer[0]]
            if producer[1] < len(pnode.outputs):
                pnode.outputs[producer[1]].links.append(lid)
        # rewired consumer links no longer belong to the instance, so
        # remove_node's link sweep below must not see them here
        out_slot.links = []
    flat.remove_node(inst.id)
