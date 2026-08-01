"""Readability and wiring lint for workflows.

Findings are dicts: {"code", "message", "node_id"?}. An annotated, correctly
wired workflow lints clean.
"""

from __future__ import annotations

from typing import Any

from . import widgets as w
from .layout import is_text_display
from .model import MODE_BYPASS, MODE_MUTE, Node, Workflow

NOTE_TYPES = {"Note", "MarkdownNote"}
_PROMPT_WIDGETS = ("text", "prompt", "wildcard_text")

# Overlap is reported as ONE finding naming at most this many nodes. Every node
# added through edit_workflow sits at the default position until
# organize_workflow runs, so a freshly built graph is N-way self-overlapping:
# reporting it pairwise produced 190 findings for 20 nodes (and 1,326 for 52),
# which is both unreadable and the single largest token sink in the server.
_OVERLAP_IDS_SHOWN = 8


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


def _has_text_display(wf: Workflow, chain: list[Node]) -> bool:
    """Whether the user can see the text produced by this upstream chain.

    Two shapes count, because they show the identical string:

    - *inline*: a Show Text node sits IN the chain feeding the encoder.
    - *tapped*: a Show Text node hangs off an output of a node in the chain
      (generator -> Show Text, generator -> encoder). This is the more common
      hand-wired shape and arguably the better one - the display doesn't sit in
      the path it reports on - but the chain walk alone never saw it, so a
      correctly-previewed workflow was told to "insert a Show Text node" that was
      already there. A false lint teaches callers to ignore the rule.
    """
    if any(is_text_display(n.type) for n in chain):
        return True
    chain_ids = {n.id for n in chain}
    return any(
        is_text_display(consumer.type)
        for link in wf.links.values()
        if link.origin_id in chain_ids
        and (consumer := wf.nodes.get(link.target_id)) is not None
    )


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
        if not _has_text_display(wf, chain):
            findings.append(
                _finding(
                    "no-prompt-preview",
                    f"{node.type} #{node.id}: the positive prompt is generated "
                    "upstream, so the user never sees the final text - add a Show "
                    "Text node (e.g. ShowText|pys), either inline before this "
                    "encoder or tapped off the generator's text output",
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
        if node.mode in (MODE_MUTE, MODE_BYPASS):
            # a disabled node is dropped before the prompt is submitted, so its
            # own wiring is not a defect - validate skips it for the same reason,
            # and lint contradicting validate on the same graph (plus
            # save_workflow's "lint is not clean" nag over a deliberately muted
            # branch) was pure noise
            continue
        schema = object_info.get(node.type)
        if schema is not None:
            socket_names = {slot.name for slot in node.inputs}
            for name, spec in schema.get("input", {}).get("required", {}).items():
                # mirror validate's exemptions exactly - lint contradicting
                # validate on the same graph is pure noise, and save_workflow
                # nags about an unclean lint. A widget, a pack's JS widget, and
                # an autogrow container are all "not a socket"; the last is
                # checked as a group by validate's autogrow-underfilled instead.
                if (
                    w.is_widget_input(spec)
                    or w._is_custom_widget(name, spec, socket_names)
                    or w.autogrow_template(spec) is not None
                ):
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
    findings.extend(_overlap_findings(wf))
    return findings


def _overlap_findings(wf: Workflow) -> list[dict[str, Any]]:
    """Overlapping nodes make workflows unreadable - reported as ONE finding.

    Every node involved in any overlap is collected and named once. The previous
    per-pair report was quadratic in output: a graph whose nodes are all still at
    the default position (i.e. anything built with edit_workflow and not yet
    organized) produced N*(N-1)/2 findings that all said the same thing.
    """
    boxes = [
        (n.id, (n.pos[0], n.pos[1], n.pos[0] + n.size[0], n.pos[1] + n.size[1]))
        for n in wf.nodes.values()
    ]
    involved: set[int] = set()
    pairs = 0
    for i, (id_a, a) in enumerate(boxes):
        for id_b, b in boxes[i + 1 :]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                involved.update((id_a, id_b))
                pairs += 1
    if not involved:
        return []
    ids = sorted(involved)
    shown = ", ".join(f"#{nid}" for nid in ids[:_OVERLAP_IDS_SHOWN])
    more = len(ids) - _OVERLAP_IDS_SHOWN
    finding = _finding(
        "overlap",
        f"{len(ids)} node(s) overlap visually ({pairs} overlapping pair(s)): {shown}"
        + (f" (+{more} more)" if more > 0 else "")
        + " - organize_workflow lays the whole graph out in one pass",
    )
    finding["node_ids"] = ids
    return [finding]
