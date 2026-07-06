"""Internal workflow graph model with dual serialization.

Parses ComfyUI UI-format workflows (schema 0.4, which the frontend and every
bundled template still emit), edits them as a typed graph, and serializes to:

- UI format 0.4 (``to_ui``): positions, sizes, titles, colors, groups, notes -
  what gets saved and opened in the editor
- API format (``to_api``): the {id: {class_type, inputs}} prompt document that
  POST /prompt executes

Frontend-only virtual nodes (Note, MarkdownNote, PrimitiveNode, Reroute) exist
in the UI graph but are resolved away during API serialization.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import widgets as w

VIRTUAL_TYPES = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}

# UI-only annotation nodes: never in object_info, single 'text' widget
NOTE_TYPES = {"Note", "MarkdownNote"}

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

MODE_NORMAL = 0
MODE_MUTE = 2
MODE_BYPASS = 4


@dataclass
class InputSlot:
    name: str
    type: str
    link: int | None = None
    widget_name: str | None = None


@dataclass
class OutputSlot:
    name: str
    type: str
    links: list[int] = field(default_factory=list)
    slot_index: int | None = None


@dataclass
class Link:
    id: int
    origin_id: int
    origin_slot: int
    target_id: int
    target_slot: int
    type: str


@dataclass
class Group:
    id: int
    title: str
    bounding: list[float]
    color: str = "#3f789e"
    font_size: int = 24
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    id: int
    type: str
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0])
    size: list[float] = field(default_factory=lambda: [270.0, 100.0])
    title: str | None = None
    color: str | None = None
    bgcolor: str | None = None
    mode: int = MODE_NORMAL
    order: int = 0
    flags: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    widgets_values: Any = field(default_factory=list)
    inputs: list[InputSlot] = field(default_factory=list)
    outputs: list[OutputSlot] = field(default_factory=list)

    def input_by_name(self, name: str) -> InputSlot | None:
        for slot in self.inputs:
            if slot.name == name:
                return slot
        return None

    def output_by_name(self, name: str) -> OutputSlot | None:
        for slot in self.outputs:
            if slot.name == name:
                return slot
        return None


def _as_pair(value: Any) -> list[float]:
    """pos/size may be [x, y] or {'0': x, '1': y}."""
    if isinstance(value, dict):
        return [float(value.get("0", 0)), float(value.get("1", 0))]
    return [float(value[0]), float(value[1])]


class Workflow:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.links: dict[int, Link] = {}
        self.groups: list[Group] = []
        self.extra: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        # ComfyUI's UI schema wants a uuid at top-level "id"; an empty string
        # trips its zod validator ("Invalid uuid at id"). Mint one per workflow
        # and keep it stable across re-exports.
        self.uuid: str = str(uuid.uuid4())
        self._next_node_id = 1
        self._next_link_id = 1

    # --- construction ---

    @classmethod
    def new(cls) -> Workflow:
        return cls()

    @classmethod
    def from_ui(cls, data: dict[str, Any]) -> Workflow:
        wf = cls()
        wf.extra = data.get("extra", {}) or {}
        wf.config = data.get("config", {}) or {}
        # preserve an existing valid workflow uuid if the source has one;
        # otherwise keep the freshly minted one from __init__
        existing_id = data.get("id") or wf.extra.get("workflow_id")
        if isinstance(existing_id, str) and _UUID_RE.match(existing_id):
            wf.uuid = existing_id
        for raw in data.get("nodes", []):
            node = Node(
                id=int(raw["id"]),
                type=raw["type"],
                pos=_as_pair(raw.get("pos", [0, 0])),
                size=_as_pair(raw.get("size", [270, 100])),
                title=raw.get("title"),
                color=raw.get("color"),
                bgcolor=raw.get("bgcolor"),
                mode=raw.get("mode", MODE_NORMAL),
                order=raw.get("order", 0),
                flags=raw.get("flags", {}) or {},
                properties=raw.get("properties", {}) or {},
                widgets_values=raw.get("widgets_values", []),
            )
            for i in raw.get("inputs", []) or []:
                node.inputs.append(
                    InputSlot(
                        name=i.get("name", ""),
                        type=str(i.get("type", "*")),
                        link=i.get("link"),
                        widget_name=(i.get("widget") or {}).get("name"),
                    )
                )
            for idx, o in enumerate(raw.get("outputs", []) or []):
                links = o.get("links") or []
                node.outputs.append(
                    OutputSlot(
                        name=o.get("name", ""),
                        type=str(o.get("type", "*")),
                        links=list(links),
                        slot_index=o.get("slot_index", idx),
                    )
                )
            wf.nodes[node.id] = node
        for raw_link in data.get("links", []) or []:
            if isinstance(raw_link, dict):  # schema 1.0 style
                link = Link(
                    id=raw_link["id"],
                    origin_id=raw_link["origin_id"],
                    origin_slot=raw_link["origin_slot"],
                    target_id=raw_link["target_id"],
                    target_slot=raw_link["target_slot"],
                    type=str(raw_link.get("type", "*")),
                )
            else:
                link = Link(
                    id=raw_link[0],
                    origin_id=raw_link[1],
                    origin_slot=raw_link[2],
                    target_id=raw_link[3],
                    target_slot=raw_link[4],
                    type=str(raw_link[5]) if len(raw_link) > 5 else "*",
                )
            wf.links[link.id] = link
        for raw_group in data.get("groups", []) or []:
            wf.groups.append(
                Group(
                    id=raw_group.get("id", len(wf.groups) + 1),
                    title=raw_group.get("title", "Group"),
                    bounding=list(raw_group.get("bounding", [0, 0, 100, 100])),
                    color=raw_group.get("color", "#3f789e"),
                    font_size=raw_group.get("font_size", 24),
                    flags=raw_group.get("flags", {}) or {},
                )
            )
        wf._next_node_id = max(wf.nodes, default=0) + 1
        wf._next_link_id = max(wf.links, default=0) + 1
        return wf

    @classmethod
    def from_api(cls, api: dict[str, Any], object_info: dict[str, Any]) -> Workflow:
        """Reconstruct an editable graph from an API-format prompt document."""
        wf = cls()
        # first pass: create nodes with widget values
        for nid_str, entry in api.items():
            class_type = entry["class_type"]
            node = wf.add_node(
                class_type,
                object_info=object_info if class_type in object_info else None,
                node_id=int(nid_str),
                raw_widgets=[] if class_type not in object_info else None,
            )
            if class_type in object_info:
                named = {
                    k: v
                    for k, v in entry.get("inputs", {}).items()
                    if not isinstance(v, list)
                }
                node.widgets_values = w.named_to_widgets(class_type, named, object_info)
        # second pass: connections
        for nid_str, entry in api.items():
            for input_name, value in entry.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2:
                    origin_id, origin_slot = int(value[0]), int(value[1])
                    origin = wf.nodes.get(origin_id)
                    if origin is None:
                        continue
                    while len(origin.outputs) <= origin_slot:
                        origin.outputs.append(
                            OutputSlot(name=f"out_{len(origin.outputs)}", type="*")
                        )
                    target = wf.nodes[int(nid_str)]
                    if target.input_by_name(input_name) is None:
                        target.inputs.append(InputSlot(name=input_name, type="*"))
                    wf._add_link(origin_id, origin_slot, int(nid_str), input_name)
        return wf

    # --- editing ---

    def add_node(
        self,
        class_type: str,
        object_info: dict[str, Any] | None = None,
        title: str | None = None,
        node_id: int | None = None,
        raw_widgets: list[Any] | None = None,
    ) -> Node:
        nid = node_id if node_id is not None else self._next_node_id
        self._next_node_id = max(self._next_node_id, nid + 1)
        node = Node(id=nid, type=class_type, title=title)
        if object_info is not None and class_type in object_info:
            schema = object_info[class_type]
            for name, spec in w._iter_schema_inputs(schema):
                if w.is_widget_input(spec):
                    continue
                slot_type = spec[0] if isinstance(spec[0], str) else "COMBO"
                node.inputs.append(InputSlot(name=name, type=slot_type))
            out_names = schema.get("output_name") or schema.get("output") or []
            out_types = schema.get("output") or []
            for idx, out_type in enumerate(out_types):
                name = out_names[idx] if idx < len(out_names) else str(out_type)
                node.outputs.append(
                    OutputSlot(name=str(name), type=str(out_type), slot_index=idx)
                )
            node.widgets_values = w.widget_defaults(class_type, object_info)
            node.properties = {"Node name for S&R": class_type}
        elif class_type in NOTE_TYPES:
            node.widgets_values = [""]
            node.size = [380.0, 180.0]
        if raw_widgets is not None:
            node.widgets_values = raw_widgets
        self.nodes[nid] = node
        return node

    def remove_node(self, node_id: int) -> None:
        self.nodes.pop(node_id, None)
        dead = [
            lid
            for lid, link in self.links.items()
            if link.origin_id == node_id or link.target_id == node_id
        ]
        for lid in dead:
            link = self.links.pop(lid)
            origin = self.nodes.get(link.origin_id)
            if origin and link.origin_slot < len(origin.outputs):
                out = origin.outputs[link.origin_slot]
                out.links = [x for x in out.links if x != lid]
            target = self.nodes.get(link.target_id)
            if target and link.target_slot < len(target.inputs):
                slot = target.inputs[link.target_slot]
                if slot.link == lid:
                    slot.link = None

    def connect(
        self,
        origin_id: int,
        origin_out: str | int,
        target_id: int,
        target_input: str,
        object_info: dict[str, Any] | None = None,
    ) -> Link:
        origin = self.nodes[origin_id]
        target = self.nodes[target_id]
        if isinstance(origin_out, int):
            out_index = origin_out
        else:
            out_index = next(
                (i for i, o in enumerate(origin.outputs) if o.name == origin_out),
                None,
            )
            if out_index is None:
                raise ValueError(
                    f"node {origin_id} ({origin.type}) has no output '{origin_out}'; "
                    f"available: {[o.name for o in origin.outputs]}"
                )
        out_slot = origin.outputs[out_index]
        in_slot = target.input_by_name(target_input)
        if in_slot is None:
            # a widget input (STRING/INT/FLOAT/...) not yet exposed as a socket:
            # convert it to an input, exactly like the ComfyUI "convert widget to
            # input" action. Keeps the widgets_values slot; the link overrides it.
            in_slot = self._materialize_widget_input(target, target_input, object_info)
        if in_slot is None:
            widget_hint = ""
            if object_info is None:
                widget_hint = " (pass object_info to connect into widget inputs)"
            raise ValueError(
                f"node {target_id} ({target.type}) has no input '{target_input}'; "
                f"available: {[i.name for i in target.inputs]}{widget_hint}"
            )
        if in_slot.type not in ("*", "COMBO") and out_slot.type != "*" and in_slot.type != out_slot.type:
            raise ValueError(
                f"type mismatch: {origin.type}.{out_slot.name} ({out_slot.type}) -> "
                f"{target.type}.{target_input} ({in_slot.type})"
            )
        return self._add_link(origin_id, out_index, target_id, target_input)

    def _materialize_widget_input(
        self, node: Node, name: str, object_info: dict[str, Any] | None
    ) -> InputSlot | None:
        """Expose a primitive/combo widget as a real input slot so a link can feed
        it. Returns the new slot, or None if the name isn't a convertible widget."""
        if object_info is None:
            return None
        schema = object_info.get(node.type)
        if schema is None:
            return None
        for input_name, spec in w._iter_schema_inputs(schema):
            if input_name != name or not w.is_widget_input(spec):
                continue
            kind = spec[0]
            slot_type = "COMBO" if isinstance(kind, list) or kind == "COMBO" else str(kind)
            slot = InputSlot(name=name, type=slot_type, widget_name=name)
            node.inputs.append(slot)
            return slot
        return None

    def _add_link(
        self, origin_id: int, origin_slot: int, target_id: int, target_input: str
    ) -> Link:
        target = self.nodes[target_id]
        in_index = next(i for i, s in enumerate(target.inputs) if s.name == target_input)
        in_slot = target.inputs[in_index]
        if in_slot.link is not None:  # replace existing connection
            old = self.links.pop(in_slot.link, None)
            if old:
                old_origin = self.nodes.get(old.origin_id)
                if old_origin and old.origin_slot < len(old_origin.outputs):
                    out = old_origin.outputs[old.origin_slot]
                    out.links = [x for x in out.links if x != old.id]
        origin = self.nodes[origin_id]
        out_slot = origin.outputs[origin_slot]
        link = Link(
            id=self._next_link_id,
            origin_id=origin_id,
            origin_slot=origin_slot,
            target_id=target_id,
            target_slot=in_index,
            type=out_slot.type,
        )
        self._next_link_id += 1
        self.links[link.id] = link
        out_slot.links.append(link.id)
        in_slot.link = link.id
        return link

    def set_widget(
        self, node_id: int, input_name: str, value: Any, object_info: dict[str, Any]
    ) -> None:
        node = self.nodes[node_id]
        if node.type in NOTE_TYPES:
            # UI-only note nodes aren't in object_info; they hold one text widget
            if input_name != "text":
                raise ValueError(f"{node.type} has a single widget: 'text'")
            node.widgets_values = [value]
            return
        slots = w.widget_slot_names(node.type, object_info, node.widgets_values)
        if input_name not in slots:
            real_widgets = [s for s in slots if not s.endswith(w.SYNTHETIC_SUFFIXES)]
            control_slots = [s for s in slots if s.endswith(w.CONTROL_SUFFIX)]
            if input_name in w.all_slot_names(node.type, object_info):
                # a sub-widget of a dynamic combo whose option isn't selected
                raise ValueError(
                    f"{node.type}: '{input_name}' is a sub-widget of a dynamic "
                    "combo whose option isn't selected. Set its parent combo to "
                    f"the option that owns it first. Active widgets: {real_widgets}"
                )
            raise ValueError(
                f"{node.type} has no widget '{input_name}'.\n"
                f"Widgets: {real_widgets}.\n"
                f"Synthetic control slots: {control_slots}. "
                f"Use 'seed__control_after_generate' to set randomize/fixed/increment/decrement."
            )
        if not isinstance(node.widgets_values, list):
            node.widgets_values[input_name] = value
            return
        # Round-trip through the named form so that setting a dynamic combo's
        # main key rebuilds its sub-widget slots (seeded with the new option's
        # defaults) exactly as the ComfyUI frontend does, and a short array gets
        # padded with schema defaults - never None: the frontend crashes on null
        # string widgets when queueing ("Cannot read properties of null").
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        named[input_name] = value
        node.widgets_values = w.named_to_widgets(node.type, named, object_info)

    def get_widget(self, node_id: int, input_name: str, object_info: dict[str, Any]) -> Any:
        node = self.nodes[node_id]
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        return named.get(input_name)

    # --- serialization ---

    def to_ui(self) -> dict[str, Any]:
        nodes_out = []
        for node in sorted(self.nodes.values(), key=lambda n: n.id):
            raw: dict[str, Any] = {
                "id": node.id,
                "type": node.type,
                "pos": list(node.pos),
                "size": list(node.size),
                "flags": node.flags,
                "order": node.order,
                "mode": node.mode,
                "inputs": [
                    {
                        "name": s.name,
                        "type": s.type,
                        "link": s.link,
                        **({"widget": {"name": s.widget_name}} if s.widget_name else {}),
                    }
                    for s in node.inputs
                ],
                "outputs": [
                    {
                        "name": s.name,
                        "type": s.type,
                        "links": list(s.links),
                        **({"slot_index": s.slot_index} if s.slot_index is not None else {}),
                    }
                    for s in node.outputs
                ],
                "properties": node.properties,
                "widgets_values": node.widgets_values,
            }
            if node.title is not None:
                raw["title"] = node.title
            if node.color is not None:
                raw["color"] = node.color
            if node.bgcolor is not None:
                raw["bgcolor"] = node.bgcolor
            nodes_out.append(raw)
        return {
            "id": self.uuid,
            "revision": 0,
            "last_node_id": max(self.nodes, default=0),
            "last_link_id": max(self.links, default=0),
            "nodes": nodes_out,
            "links": [
                [ln.id, ln.origin_id, ln.origin_slot, ln.target_id, ln.target_slot, ln.type]
                for ln in sorted(self.links.values(), key=lambda x: x.id)
            ],
            "groups": [
                {
                    "id": g.id,
                    "title": g.title,
                    "bounding": list(g.bounding),
                    "color": g.color,
                    "font_size": g.font_size,
                    "flags": g.flags,
                }
                for g in self.groups
            ],
            "config": self.config,
            "extra": self.extra,
            "version": 0.4,
        }

    def to_api(self, object_info: dict[str, Any]) -> dict[str, Any]:
        """Serialize executable nodes to the /prompt API document."""
        resolved = self._resolve_link_origins()
        primitive_values = {
            n.id: (n.widgets_values[0] if n.widgets_values else None)
            for n in self.nodes.values()
            if n.type == "PrimitiveNode"
        }
        api: dict[str, Any] = {}
        for node in self.nodes.values():
            if node.type in VIRTUAL_TYPES or node.mode in (MODE_MUTE, MODE_BYPASS):
                continue
            if node.type not in object_info:
                raise ValueError(
                    f"node {node.id}: class '{node.type}' is not available on this "
                    "ComfyUI instance (missing custom node?)"
                )
            named_widgets = w.named_for_api(node.type, node.widgets_values, object_info)
            inputs: dict[str, Any] = {
                k: v
                for k, v in named_widgets.items()
                if not k.endswith(w.SYNTHETIC_SUFFIXES)
            }
            for in_index, slot in enumerate(node.inputs):
                if slot.link is None:
                    continue
                origin = resolved.get((node.id, in_index))
                if origin is None:
                    continue
                origin_id, origin_slot = origin
                if origin_id in primitive_values:
                    inputs[slot.name] = primitive_values[origin_id]
                else:
                    inputs[slot.name] = [str(origin_id), origin_slot]
            api[str(node.id)] = {"class_type": node.type, "inputs": inputs}
        return api

    def _resolve_link_origins(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Map (target_id, target_slot) -> real (origin_id, origin_slot), seeing
        through Reroute and bypassed nodes."""
        resolved: dict[tuple[int, int], tuple[int, int]] = {}
        for link in self.links.values():
            origin = self._trace_origin(link.origin_id, link.origin_slot, depth=0)
            if origin is not None:
                resolved[(link.target_id, link.target_slot)] = origin
        return resolved

    def _trace_origin(
        self, origin_id: int, origin_slot: int, depth: int
    ) -> tuple[int, int] | None:
        if depth > 100:
            return None
        node = self.nodes.get(origin_id)
        if node is None:
            return None
        passthrough = node.type == "Reroute" or node.mode == MODE_BYPASS
        if not passthrough:
            return (origin_id, origin_slot)
        # find this node's upstream feed: for Reroute, its single input; for
        # bypass, the first input whose type matches the requested output type
        wanted_type = (
            node.outputs[origin_slot].type if origin_slot < len(node.outputs) else "*"
        )
        for slot in node.inputs:
            if slot.link is None:
                continue
            if node.type == "Reroute" or wanted_type in ("*",) or slot.type == wanted_type:
                upstream = self.links.get(slot.link)
                if upstream:
                    return self._trace_origin(upstream.origin_id, upstream.origin_slot, depth + 1)
        return None
