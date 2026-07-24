"""Readability and wiring lint for workflows.

Findings are dicts: {"code", "message", "node_id"?}. An annotated, correctly
wired workflow lints clean.
"""

from __future__ import annotations

from typing import Any

from . import widgets as w
from .layout import is_text_display
from .model import Node, Workflow

NOTE_TYPES = {"Note", "MarkdownNote"}
_PROMPT_WIDGETS = ("text", "prompt", "wildcard_text")


def _finding(code: str, message: str, node_id: int | None = None) -> dict[str, Any]:
    finding: dict[str, Any] = {"code": code, "message": message}
    if node_id is not None:
        finding["node_id"] = node_id
    return finding


def _upstream_nodes(wf: Workflow, node: Node, slot_name: str, depth: int = 4) -> list[Node]:
    """Nodes on the chain feeding an input slot (BFS upstream, bounded)."""
    found: list[Node] = []
    slot = node.input_by_name(slot_name)
    if slot is None or slot.link is None or slot.link not in wf.links:
        return found
    frontier = [wf.links[slot.link].origin_id]
    visited: set[int] = set()
    for _ in range(depth):
        next_frontier: list[int] = []
        for nid in frontier:
            if nid in visited or nid not in wf.nodes:
                continue
            visited.add(nid)
            upstream = wf.nodes[nid]
            found.append(upstream)
            for inp in upstream.inputs:
                if inp.link is not None and inp.link in wf.links:
                    next_frontier.append(wf.links[inp.link].origin_id)
        frontier = next_frontier
    return found


def _missing_prompt_previews(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    """A positive prompt built upstream (wildcards, concatenators) is invisible
    to the user unless a Show Text-style node displays the final string."""
    from .annotate import _prompt_role

    findings = []
    for node in wf.nodes.values():
        try:
            slots = set(w.widget_slot_names(node.type, object_info))
        except (ValueError, KeyError):
            continue
        if node.type != "CLIPTextEncode" and not (slots & set(_PROMPT_WIDGETS)):
            continue
        wired = [
            name
            for name in _PROMPT_WIDGETS
            if name in slots
            and (slot := node.input_by_name(name)) is not None
            and slot.link is not None
        ]
        if not wired or _prompt_role(wf, node) != "positive":
            continue
        chain = _upstream_nodes(wf, node, wired[0])
        if not any(is_text_display(n.type) for n in chain):
            findings.append(
                _finding(
                    "no-prompt-preview",
                    f"{node.type} #{node.id}: the positive prompt is generated "
                    "upstream, so the user never sees the final text - insert a "
                    "Show Text node (e.g. ShowText|pys) inline before this encoder",
                    node.id,
                )
            )
    return findings


def lint(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    real_nodes = [n for n in wf.nodes.values() if n.type not in NOTE_TYPES]

    if not wf.groups:
        findings.append(_finding("no-groups", "no groups: stages are not visually organized"))
    if not any(n.type in NOTE_TYPES for n in wf.nodes.values()):
        findings.append(_finding("no-notes", "no guidance notes for human readers"))

    linked_ids: set[int] = set()
    for link in wf.links.values():
        linked_ids.update((link.origin_id, link.target_id))

    for node in real_nodes:
        schema = object_info.get(node.type)
        if schema is not None:
            for name, spec in schema.get("input", {}).get("required", {}).items():
                if w.is_widget_input(spec):
                    continue
                slot = node.input_by_name(name)
                if slot is None or slot.link is None:
                    findings.append(
                        _finding(
                            "unconnected-input",
                            f"{node.type} #{node.id}: required input '{name}' is not connected",
                            node.id,
                        )
                    )
        if node.id not in linked_ids and len(real_nodes) > 1:
            findings.append(
                _finding("orphan-node", f"{node.type} #{node.id} is connected to nothing", node.id)
            )
        try:
            slots = set(w.widget_slot_names(node.type, object_info))
        except (ValueError, KeyError):
            slots = set()
        if "text" in slots and node.title is None:
            findings.append(
                _finding(
                    "untitled-prompts",
                    f"{node.type} #{node.id} holds prompt text but has no descriptive title",
                    node.id,
                )
            )

    findings.extend(_missing_prompt_previews(wf, object_info))

    # overlapping nodes make workflows unreadable
    boxes = [
        (n.id, (n.pos[0], n.pos[1], n.pos[0] + n.size[0], n.pos[1] + n.size[1]))
        for n in wf.nodes.values()
    ]
    for i, (id_a, a) in enumerate(boxes):
        for id_b, b in boxes[i + 1 :]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                findings.append(
                    _finding("overlap", f"nodes #{id_a} and #{id_b} overlap visually", id_a)
                )
    return findings
