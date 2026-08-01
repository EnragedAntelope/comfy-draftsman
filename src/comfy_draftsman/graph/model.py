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

import random
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import widgets as w

VIRTUAL_TYPES = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}

PRIMITIVE_TYPE = "PrimitiveNode"
REROUTE_TYPE = "Reroute"

# Slot types that accept anything: litegraph's own wildcard plus the spellings
# custom packs use for the same idea.
_WILDCARD_SLOT_TYPES = {"*", "", "ANY", "ANYTYPE", "?"}

# ComfyUI's V3 io system declares a handful of META types that are markers for
# the frontend, never a concrete socket type a value can be. `COMFY_MATCHTYPE_V3`
# is the one that reaches `types_compatible`: the backend's own
# `comfy_execution.validation.validate_node_input` short-circuits True whenever
# either side is a MatchType ("validation for this is handled by the frontend"),
# because a match-type node adopts whatever it is wired to. Core nodes use it -
# `ComfySwitchNode` (If/Else Switch) is MatchType in and MatchType out.
MATCH_TYPE = "COMFY_MATCHTYPE_V3"

CONTROL_MODES = ("fixed", "randomize", "increment", "decrement")


def types_compatible(out_type: str | None, in_type: str | None) -> bool:
    """Whether an output slot may feed an input slot.

    Mirrors ComfyUI's executor (a mismatch answers ``return_type_mismatch``) and
    litegraph's case-insensitive comparison, with the wildcards above and the
    comma-joined union types a few packs declare ("IMAGE,LATENT" - which is also
    how a V3 ``MultiType`` input serializes).

    **COMBO is not a wildcard.** Wiring a STRING into a converted combo widget
    used to pass every local check and then be rejected by the live server, which
    queues the prompt anyway and executes only the rest of the graph - a silent
    partial render. A combo input does accept a COMBO-typed producer (the
    frontend's PrimitiveNode adopts exactly that type when it mirrors a combo).

    **MatchType IS a wildcard**, exactly as the backend treats it - see
    ``MATCH_TYPE``. Treating it as a concrete type refused every wire into or out
    of a core ``ComfySwitchNode``, which made any workflow containing one
    unrunnable *and* unsavable through draftsman (validate's link-type-mismatch
    is a blocking error), with no way to force past it from a template.
    """
    out = str(out_type or "").strip().upper()
    inp = str(in_type or "").strip().upper()
    if out in _WILDCARD_SLOT_TYPES or inp in _WILDCARD_SLOT_TYPES:
        return True
    if MATCH_TYPE in (out, inp):
        return True
    if out == inp:
        return True
    if "," in out or "," in inp:
        return bool(
            {p.strip() for p in out.split(",") if p.strip()}
            & {p.strip() for p in inp.split(",") if p.strip()}
        )
    return False

# %date% / %date:FORMAT% filename-prefix tokens are substituted by a frontend
# extension (pysssss / Custom-Scripts) before the browser submits; the backend
# /prompt endpoint never processes them, so a headless run would pass the literal
# token (illegal ':' on Windows) to the filesystem. Mirror the extension here at
# API-serialization time. .NET-style tokens, longest-first so 'yyyy' beats 'yy'.
_DATE_TOKEN_RE = re.compile(r"%date(?::([^%]*))?%")
_DATE_TOKEN_MAP = (
    ("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
    ("hh", "%H"), ("mm", "%M"), ("ss", "%S"),
)


def _substitute_filename_tokens(value: str, now: datetime) -> str:
    """Replace %date%/%date:FORMAT% tokens in a string with the current time.
    Non-token strings are returned unchanged."""
    if "%date" not in value:
        return value

    def _replace(match: re.Match[str]) -> str:
        fmt = match.group(1) or "yyyy-MM-dd"
        for token, directive in _DATE_TOKEN_MAP:
            fmt = fmt.replace(token, now.strftime(directive))
        return fmt

    return _DATE_TOKEN_RE.sub(_replace, value)

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
    # A PrimitiveNode's output records which widget it mirrors
    # ({"widget": {"name": "steps"}}). Dropping it on import silently degraded
    # every round-trip of a workflow using primitives - including the bundled
    # SDXL template, whose primitives drive steps/end_at_step/text.
    widget_name: str | None = None


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
        # schema-1.0 "definitions" (subgraph packaging): kept verbatim so a
        # subgraph workflow round-trips intact; wrapper nodes' type is the
        # definition's uuid. See subgraph_defs().
        self.definitions: dict[str, Any] = {}
        # ComfyUI's UI schema wants a uuid at top-level "id"; an empty string
        # trips its zod validator ("Invalid uuid at id"). Mint one per workflow
        # and keep it stable across re-exports.
        self.uuid: str = str(uuid.uuid4())
        self._next_node_id = 1
        self._next_link_id = 1
        # populated by subgraph_as_workflow; restored by update_subgraph
        self._subgraph_meta: dict[str, Any] = {}

    # --- construction ---

    @classmethod
    def new(cls) -> Workflow:
        return cls()

    @classmethod
    def from_ui(cls, data: dict[str, Any]) -> Workflow:
        wf = cls()
        wf.extra = data.get("extra", {}) or {}
        wf.config = data.get("config", {}) or {}
        wf.definitions = data.get("definitions", {}) or {}
        # preserve an existing valid workflow uuid if the source has one;
        # otherwise keep the freshly minted one from __init__
        existing_id = data.get("id") or wf.extra.get("workflow_id")
        if isinstance(existing_id, str) and _UUID_RE.match(existing_id):
            wf.uuid = existing_id
        for raw in data.get("nodes", []):
            if not isinstance(raw, dict) or "id" not in raw or "type" not in raw:
                raise ValueError(
                    f"malformed node entry {str(raw)[:80]!r}: every UI-format node "
                    "needs an 'id' and a 'type'"
                )
            try:
                node_id = int(raw["id"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"node id {raw['id']!r} is not a number - UI-format workflows "
                    "use integer node ids"
                ) from e
            node = Node(
                id=node_id,
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
                        widget_name=(o.get("widget") or {}).get("name"),
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

    @staticmethod
    def _is_api_connection(value: Any, node_ids: set[str]) -> bool:
        """True if an API-format input value is a LINK rather than a literal
        widget value.

        In API format a connection is exactly ``[source_node_id, output_slot]``:
        a 2-element list whose first element names a node present in this prompt
        and whose second is an integer slot index. A widget value that merely
        *happens* to be a 2-element list (e.g. a coordinate pair like [512, 512],
        or ["a", "b"]) is not a connection and must be preserved verbatim - the
        old length-2 heuristic silently turned such values into bogus wires (and
        int(value[0]) on a non-numeric first element could even crash the import).
        Booleans are excluded because bool is a subclass of int.
        """
        return (
            isinstance(value, list)
            and len(value) == 2
            and not isinstance(value[0], bool)
            and isinstance(value[0], (str, int))
            and str(value[0]) in node_ids
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        )

    @classmethod
    def from_api(cls, api: dict[str, Any], object_info: dict[str, Any]) -> Workflow:
        """Reconstruct an editable graph from an API-format prompt document.

        Node ids are usually numeric strings, but a prompt exported by a newer
        frontend (or hand-written) can key nodes by an arbitrary string. The UI
        graph model needs int ids, so non-numeric keys are mapped to fresh ints
        and the original is kept in ``properties['api_node_id']`` rather than
        crashing the whole import on ``int()``.
        """
        wf = cls()
        node_ids = set(api.keys())
        id_map = cls._api_id_map(node_ids)
        # first pass: create nodes with widget values
        for nid_str, entry in api.items():
            if not isinstance(entry, dict) or "class_type" not in entry:
                raise ValueError(
                    f"node {nid_str!r} is not a valid API-format entry: expected "
                    '{"class_type": ..., "inputs": {...}} - is this actually a '
                    "UI-format workflow (it would have a top-level 'nodes' list)?"
                )
            class_type = entry["class_type"]
            node = wf.add_node(
                class_type,
                object_info=object_info if class_type in object_info else None,
                node_id=id_map[nid_str],
                raw_widgets=[] if class_type not in object_info else None,
            )
            if str(id_map[nid_str]) != nid_str:
                node.properties["api_node_id"] = nid_str
            if class_type in object_info:
                # only real connections are excluded from widgets; a literal
                # list-valued widget (rare, but legal) is kept, not dropped
                named = {
                    k: v
                    for k, v in entry.get("inputs", {}).items()
                    if not cls._is_api_connection(v, node_ids)
                }
                node.widgets_values = w.named_to_widgets(class_type, named, object_info)
        # second pass: connections
        for nid_str, entry in api.items():
            for input_name, value in entry.get("inputs", {}).items():
                if cls._is_api_connection(value, node_ids):
                    origin_id, origin_slot = id_map[str(value[0])], int(value[1])
                    origin = wf.nodes.get(origin_id)
                    if origin is None:
                        continue
                    while len(origin.outputs) <= origin_slot:
                        origin.outputs.append(
                            OutputSlot(name=f"out_{len(origin.outputs)}", type="*")
                        )
                    target = wf.nodes[id_map[nid_str]]
                    if target.input_by_name(input_name) is None:
                        target.inputs.append(InputSlot(name=input_name, type="*"))
                    wf._add_link(origin_id, origin_slot, id_map[nid_str], input_name)
        return wf

    @staticmethod
    def _api_id_map(node_ids: set[str]) -> dict[str, int]:
        """API node key -> int node id. Numeric keys keep their value; anything
        else gets a fresh id above the numeric ones, so a prompt mixing both
        never collides."""
        mapping: dict[str, int] = {}
        numeric = {}
        for key in node_ids:
            try:
                numeric[key] = int(key)
            except (TypeError, ValueError):
                continue
        mapping.update(numeric)
        next_id = max(numeric.values(), default=0) + 1
        for key in sorted(node_ids - numeric.keys()):
            mapping[key] = next_id
            next_id += 1
        return mapping

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
        elif class_type == REROUTE_TYPE:
            # a wire-tidying passthrough: one untyped input, one untyped output
            node.inputs.append(InputSlot(name="", type="*"))
            node.outputs.append(OutputSlot(name="", type="*", slot_index=0))
            node.size = [75.0, 26.0]
            node.properties = {"showOutputText": False, "horizontal": False}
        elif class_type == PRIMITIVE_TYPE:
            # Typeless until connected: the frontend gives a primitive its type,
            # widget and value list from the first widget input it feeds. connect()
            # mirrors that here (see _mirror_primitive) because there is no browser
            # to do it for a headless author.
            node.outputs.append(OutputSlot(name="*", type="*", slot_index=0))
            node.properties = {"Run widget replace on values": False}
            node.size = [210.0, 82.0]
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
        force: bool = False,
    ) -> Link:
        origin = self.nodes[origin_id]
        target = self.nodes[target_id]
        if isinstance(origin_out, int):
            out_index = origin_out
        else:
            found = next(
                (i for i, o in enumerate(origin.outputs) if o.name == origin_out),
                None,
            )
            if found is None:
                raise ValueError(
                    f"node {origin_id} ({origin.type}) has no output '{origin_out}'; "
                    f"available: {[o.name for o in origin.outputs]}"
                )
            out_index = found
        out_slot = origin.outputs[out_index]
        in_slot = target.input_by_name(target_input)
        materialized = False
        if in_slot is None:
            # a widget input (STRING/INT/FLOAT/...) not yet exposed as a socket:
            # convert it to an input, exactly like the ComfyUI "convert widget to
            # input" action. Keeps the widgets_values slot; the link overrides it.
            in_slot = self._materialize_widget_input(target, target_input, object_info)
            materialized = in_slot is not None
        if in_slot is None:
            widget_hint = ""
            if object_info is None:
                widget_hint = " (pass object_info to connect into widget inputs)"
            raise ValueError(
                f"node {target_id} ({target.type}) has no input '{target_input}'; "
                f"available: {[i.name for i in target.inputs]}{widget_hint}"
            )
        if not force and not types_compatible(out_slot.type, in_slot.type):
            if materialized:
                # a REFUSED connect must leave nothing behind: converting the
                # widget to an input is a visible change (the node grows an empty
                # socket and stops accepting a typed value), so undo it
                target.inputs.remove(in_slot)
            raise ValueError(
                f"type mismatch: {origin.type}.{out_slot.name} ({out_slot.type}) -> "
                f"{target.type}.{target_input} ({in_slot.type}). ComfyUI rejects this "
                "at queue time (return_type_mismatch) and then runs only the REST of "
                f"the graph, so the failure is silent. To drive a {in_slot.type} "
                "widget: set_widget it directly, or feed it a PrimitiveNode (it "
                "mirrors the socket's own type and can carry control_after_generate). "
                'Add "force": true to wire it anyway.'
            )
        # a primitive takes its type from the socket it first feeds - do that
        # before the link is made, so the link records the resolved type
        if origin.type == PRIMITIVE_TYPE:
            self._mirror_primitive(origin, out_slot, target, in_slot, object_info)
        return self._add_link(origin_id, out_index, target_id, target_input)

    def _mirror_primitive(
        self,
        primitive: Node,
        out_slot: OutputSlot,
        target: Node,
        in_slot: InputSlot,
        object_info: dict[str, Any] | None,
    ) -> None:
        """Adopt the target widget's type onto a PrimitiveNode, as the frontend
        does on a primitive's first connection (widgetInputs.ts
        ``#onFirstConnection``/``setType``).

        That behaviour is browser JS, so a headless author must resolve it at
        authoring time or the node stays typeless and unusable. Only an
        *unresolved* primitive adopts: one already bound keeps its type, so wiring
        a second consumer of the same value never silently retypes it.

        Shape mirrors a real export (see tests/fixtures/sdxl_simple_example.json):
        output name+type are the widget type ("COMBO" for any combo), the output
        carries the widget marker, the title is the widget's name, and
        widgets_values is ``[value]`` plus ``["fixed"]`` for number/combo widgets
        only - a STRING primitive gets no control widget.
        """
        if out_slot.type not in ("*", ""):
            return
        widget_name = in_slot.widget_name or in_slot.name
        spec = None
        if object_info is not None and target.type in object_info:
            spec = w.widget_specs(
                target.type,
                object_info,
                target.widgets_values,
                {s.name for s in target.inputs},
            ).get(widget_name)
        if spec is not None:
            kind = spec[0]
            slot_type = (
                "COMBO"
                if isinstance(kind, list) or kind in ("COMBO", w.DYNAMIC_COMBO_TYPE)
                else str(kind)
            )
        else:
            slot_type = in_slot.type if in_slot.type not in ("*", "") else "COMBO"
        out_slot.type = slot_type
        out_slot.name = slot_type
        out_slot.widget_name = widget_name
        if primitive.title is None:
            primitive.title = widget_name
        values: list[Any] = [w._default_for(spec) if spec is not None else ""]
        if spec is not None and w.primitive_takes_control(spec):
            values.append("fixed")
        primitive.widgets_values = values
        if spec is not None and w._opts(spec).get("multiline"):
            primitive.size = [300.0, 160.0]
        else:
            primitive.size = [210.0, 82.0]

    def _forward_widget_targets(
        self, node_id: int, depth: int = 0
    ) -> Iterator[tuple[Node, InputSlot]]:
        """(target node, input slot) for everything this node feeds, seeing
        forward through Reroutes the way _trace_origin sees backward."""
        if depth > 10:
            return
        node = self.nodes.get(node_id)
        if node is None:
            return
        for out in node.outputs:
            for lid in out.links:
                link = self.links.get(lid)
                if link is None:
                    continue
                target = self.nodes.get(link.target_id)
                if target is None or link.target_slot >= len(target.inputs):
                    continue
                if target.type == REROUTE_TYPE:
                    yield from self._forward_widget_targets(target.id, depth + 1)
                else:
                    yield target, target.inputs[link.target_slot]

    def primitive_targets(
        self, node_id: int, object_info: dict[str, Any]
    ) -> list[tuple[Any, Node, str]]:
        """(widget spec, target node, widget name) for every widget a
        PrimitiveNode drives. Empty when it is unconnected or feeds no known
        widget - i.e. it has no type and nothing to validate against."""
        results: list[tuple[Any, Node, str]] = []
        seen: set[tuple[int, str]] = set()
        for target, slot in self._forward_widget_targets(node_id):
            if target.type not in object_info:
                continue
            name = slot.widget_name or slot.name
            if (target.id, name) in seen:
                continue
            spec = w.widget_specs(
                target.type,
                object_info,
                target.widgets_values,
                {s.name for s in target.inputs},
            ).get(name)
            if spec is not None:
                seen.add((target.id, name))
                results.append((spec, target, name))
        return results

    def primitive_target(
        self, node_id: int, object_info: dict[str, Any]
    ) -> tuple[Any, Node, str] | None:
        """The first widget a PrimitiveNode drives, or None."""
        targets = self.primitive_targets(node_id, object_info)
        return targets[0] if targets else None

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
        if node.type == PRIMITIVE_TYPE:
            self._set_primitive_widget(node, input_name, value, object_info)
            return
        # instance context: a custom JS-widget input is only recognizable from
        # the sockets this node actually serialized (widgets._is_custom_widget).
        # Without it the slot walk misses that widget, so the round-trip below
        # would drop its value and shift every later one up a slot.
        socket_names = {slot.name for slot in node.inputs}
        slots = w.widget_slot_names(node.type, object_info, node.widgets_values, socket_names)
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
        named = w.widgets_to_named(
            node.type, node.widgets_values, object_info, socket_names
        )
        named[input_name] = value
        node.widgets_values = w.named_to_widgets(
            node.type, named, object_info, socket_names
        )

    def _primitive_widget_names(self, node: Node) -> tuple[str | None, set[str]]:
        """(mirrored widget name, accepted control-slot names) for a primitive."""
        bound = node.outputs[0].widget_name if node.outputs else None
        control = {"control_after_generate"}
        if bound:
            control.add(bound + w.CONTROL_SUFFIX)
        return bound, control

    def _set_primitive_widget(
        self, node: Node, input_name: str, value: Any, object_info: dict[str, Any]
    ) -> None:
        """Set a PrimitiveNode's value or control mode.

        A primitive's widgets are created by the frontend from whatever socket it
        mirrors, so its slot names come from the GRAPH, not object_info: slot 0 is
        the mirrored widget - addressable by its real name or the alias ``value`` -
        and slot 1, when the mirrored widget is a number or combo, is
        ``control_after_generate``.
        """
        bound, control_names = self._primitive_widget_names(node)
        values = list(node.widgets_values) if isinstance(node.widgets_values, list) else []
        if not values:
            values = [""]
        if input_name in control_names:
            if value not in CONTROL_MODES:
                raise ValueError(
                    f"control_after_generate must be one of {list(CONTROL_MODES)}, "
                    f"got {value!r}"
                )
            resolved = self.primitive_target(node.id, object_info)
            if resolved is not None and not w.primitive_takes_control(resolved[0]):
                kind = resolved[0][0]
                raise ValueError(
                    f"PrimitiveNode #{node.id} mirrors a {kind if isinstance(kind, str) else 'COMBO'} "
                    "widget, and the ComfyUI frontend only gives number and combo "
                    "primitives a control_after_generate widget - there is nothing to set"
                )
            if len(values) < 2:
                values.append(value)
            else:
                values[1] = value
            node.widgets_values = values
            return
        if input_name not in ("value", bound):
            usable = "'value'" + (f" (mirrors '{bound}')" if bound else "")
            hint = (
                ""
                if bound
                else " - connect it to a widget input first so it adopts that "
                "socket's type"
            )
            raise ValueError(
                f"PrimitiveNode #{node.id} has widgets {usable} and "
                f"'control_after_generate'; got {input_name!r}{hint}"
            )
        values[0] = value
        node.widgets_values = values

    def get_widget(self, node_id: int, input_name: str, object_info: dict[str, Any]) -> Any:
        node = self.nodes[node_id]
        named = w.widgets_to_named(
            node.type,
            node.widgets_values,
            object_info,
            {slot.name for slot in node.inputs},
        )
        return named.get(input_name)

    # --- subgraphs ---

    def subgraph_defs(self) -> dict[str, dict[str, Any]]:
        """{definition uuid: subgraph definition} from the schema-1.0
        "definitions" block. A node whose type equals one of these uuids is a
        subgraph instance (opaque wrapper around the definition's inner graph)."""
        return {
            sg["id"]: sg
            for sg in self.definitions.get("subgraphs", []) or []
            if isinstance(sg, dict) and sg.get("id")
        }

    def subgraph_as_workflow(self, def_id: str) -> Workflow:
        """Extract one subgraph definition as a standalone Workflow.

        The returned Workflow contains the definition's inner nodes and links,
        suitable for editing with the normal graph operations. Definition
        metadata (name, inputs, outputs) is preserved on the Workflow object
        as ``_subgraph_meta`` and restored by :meth:`update_subgraph`.

        Raises:
            KeyError: if def_id is not in the definitions.
            NotImplementedError: if the definition contains nested subgraph
                instance nodes (editing those is not supported yet).
        """
        defs = self.subgraph_defs()
        if def_id not in defs:
            raise KeyError(f"no subgraph definition with id {def_id!r}")
        definition = defs[def_id]

        # check for nested subgraph instances inside this definition
        known_def_ids = set(defs.keys())
        for node in definition.get("nodes", []):
            ntype = node.get("type", "")
            if ntype.startswith("subgraph.") or ntype in known_def_ids:
                raise NotImplementedError(
                    "editing nested subgraph definitions is not supported yet"
                )

        # build a minimal UI-format document and parse it via from_ui
        nodes = definition.get("nodes", [])
        links = definition.get("links", [])
        max_nid = max((n.get("id", 0) for n in nodes), default=0)
        max_lid = 0
        for ln in links:
            if isinstance(ln, dict):
                max_lid = max(max_lid, ln.get("id", 0))
            elif isinstance(ln, (list, tuple)) and ln:
                max_lid = max(max_lid, ln[0])
        ui_doc = {
            "last_node_id": max_nid,
            "last_link_id": max_lid,
            "nodes": nodes,
            "links": links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }
        wf = Workflow.from_ui(ui_doc)
        # attach definition metadata so update_subgraph can restore it
        wf._subgraph_meta = {
            "name": definition.get("name", ""),
            "inputs": definition.get("inputs", []),
            "outputs": definition.get("outputs", []),
        }
        return wf

    def update_subgraph(self, def_id: str, wf: Workflow) -> None:
        """Write an edited Workflow back into a subgraph definition.

        Updates the definition's nodes and links from the serialized form of
        ``wf``. Definition metadata (name, inputs, outputs) is preserved.

        Raises:
            KeyError: if def_id is not in the definitions.
        """
        defs = self.subgraph_defs()
        if def_id not in defs:
            raise KeyError(f"no subgraph definition with id {def_id!r}")

        ui = wf.to_ui()
        definition = defs[def_id]
        definition["nodes"] = ui["nodes"]
        definition["links"] = ui["links"]
        # preserve existing name, inputs, outputs — do not overwrite


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
                        **({"widget": {"name": s.widget_name}} if s.widget_name else {}),
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
            **({"definitions": self.definitions} if self.definitions else {}),
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
        """Serialize executable nodes to the /prompt API document. Subgraph
        instances are flattened first (mirroring the frontend's queue-time
        expansion; see graph/subgraph.py)."""
        from .subgraph import flatten, has_subgraph_instances

        if has_subgraph_instances(self):
            flat, _, _ = flatten(self, object_info)
            return flat.to_api(object_info)
        resolved = self._resolve_link_origins()
        now = datetime.now()
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
                subgraph = self.subgraph_defs().get(node.type)
                if subgraph is not None:
                    # active instances were flattened above; only reachable if
                    # the definition is too malformed to expand
                    raise ValueError(
                        f"node {node.id} is an instance of subgraph "
                        f"'{subgraph.get('name', node.type)}' that could not be "
                        "flattened (malformed definition?). Rebuild the graph from "
                        "the subgraph's internals (inspect_workflow lists its nodes "
                        "and wiring), or run it from the ComfyUI frontend."
                    )
                raise ValueError(
                    f"node {node.id}: class '{node.type}' is not available on this "
                    "ComfyUI instance (missing custom node?)"
                )
            socket_names = {slot.name for slot in node.inputs}
            named_widgets = w.named_for_api(
                node.type, node.widgets_values, object_info, socket_names
            )
            inputs: dict[str, Any] = {
                k: (_substitute_filename_tokens(v, now) if isinstance(v, str) else v)
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

    def apply_seed_control(self, object_info: dict[str, Any]) -> bool:
        """Re-roll seed widgets per their control_after_generate mode (the
        frontend does this between queues; the raw API never does). Mutates
        widgets_values in place. Returns True if any seed changed."""
        changed = False
        for node in self.nodes.values():
            if node.type == PRIMITIVE_TYPE:
                if self.roll_primitive_control(node, object_info):
                    changed = True
                continue
            if node.type not in object_info:
                continue
            new_values, node_changed = w.roll_seed_controls(
                node.type,
                node.widgets_values,
                object_info,
                socket_names={slot.name for slot in node.inputs},
            )
            if node_changed:
                node.widgets_values = new_values
                changed = True
        return changed

    def roll_primitive_control(
        self,
        node: Node,
        object_info: dict[str, Any],
        rng: random.Random | None = None,
    ) -> bool:
        """Re-roll one PrimitiveNode per its ``control_after_generate``, the way
        the browser does between queues (ComfyUI's ``addValueControlWidgets``).

        This is the ONLY way a headless caller can cycle a COMBO across runs:
        control_after_generate is applied by browser JS, never by ``/prompt``, so
        without this every repeated run re-submits the same dropdown value. It is
        what makes "cycle a character/checkpoint/LoRA dropdown across runs"
        possible through the MCP at all.

        Faithful to the frontend, including the parts that feel wrong: an index
        walk **clamps** at the ends of the option list rather than wrapping, and
        numbers clamp to min/max. Returns True if the value changed.
        """
        values = node.widgets_values
        if not isinstance(values, list) or len(values) < 2:
            return False  # no control widget (a STRING primitive never has one)
        mode = values[1]
        if mode not in ("randomize", "increment", "decrement"):
            return False
        resolved = self.primitive_target(node.id, object_info)
        if resolved is None:
            return False
        spec = resolved[0]
        roller = rng or random
        current = values[0]
        choices = w.combo_choices(spec)
        if choices:
            if mode == "randomize":
                index = roller.randrange(len(choices))
            else:
                try:
                    index = choices.index(current)
                except ValueError:
                    index = 0
                index += 1 if mode == "increment" else -1
            new: Any = choices[max(0, min(len(choices) - 1, index))]
        elif spec[0] in ("INT", "FLOAT") and isinstance(current, int | float) and not isinstance(current, bool):
            opts = w._opts(spec)
            step = opts.get("step") or 1
            low = opts.get("min")
            high = opts.get("max")
            if mode == "randomize":
                lo = float(low if low is not None else 0)
                hi = float(high if high is not None else w.MAX_SEED)
                # the frontend randomizes on the step grid: floor(rand * span/step) * step + min
                span = max(int((hi - lo) / step), 0) if step else 0
                new = lo + roller.randrange(span + 1) * step
            else:
                new = current + (step if mode == "increment" else -step)
            if low is not None:
                new = max(low, new)
            if high is not None:
                new = min(high, new)
            new = int(new) if spec[0] == "INT" else round(float(new), 6)
        else:
            return False
        if new == current:
            return False
        node.widgets_values = [new, *values[1:]]
        return True

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
