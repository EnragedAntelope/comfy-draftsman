"""Readability and wiring lint for workflows.

Findings are dicts: {"code", "message", "node_id"?}. An annotated, correctly
wired workflow lints clean.
"""

from __future__ import annotations

from typing import Any

from . import widgets as w
from .model import Workflow

NOTE_TYPES = {"Note", "MarkdownNote"}


def _finding(code: str, message: str, node_id: int | None = None) -> dict[str, Any]:
    finding: dict[str, Any] = {"code": code, "message": message}
    if node_id is not None:
        finding["node_id"] = node_id
    return finding


def lint(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    real_nodes = [n for n in wf.nodes.values() if n.type not in NOTE_TYPES]

    if not wf.groups:
        findings.append(_finding("no-groups", "no groups: stages are not visually organized"))
    if not any(n.type in NOTE_TYPES for n in wf.nodes.values()):
        findings.append(_finding("no-notes", "no guidance notes for human readers"))

    linked_ids = set()
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
        slots = set()
        try:
            slots = set(w.widget_slot_names(node.type, object_info))
        except (ValueError, KeyError):
            pass
        if "text" in slots and node.title is None:
            findings.append(
                _finding(
                    "untitled-prompts",
                    f"{node.type} #{node.id} holds prompt text but has no descriptive title",
                    node.id,
                )
            )

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
