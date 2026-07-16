"""Layered auto-layout (Sugiyama-style) for workflow graphs.

Columns = longest-path rank from source nodes, so data always flows left to
right. Within a column, nodes are ordered by the barycenter of their upstream
neighbors to keep wires short and reduce crossings. Sizes are estimated from
slot and widget counts so text-heavy nodes get room to breathe.

Note/MarkdownNote nodes are not positioned here - the annotator places them
relative to the groups it creates.
"""

from __future__ import annotations

import re
from typing import Any

from . import widgets as w
from .model import Workflow

X_GUTTER = 90.0
Y_GAP = 40.0
HEADER_H = 30.0
SLOT_H = 21.0
WIDGET_H = 24.0
MIN_H = 60.0
DEFAULT_W = 250.0
TEXT_W = 400.0
NOTE_TYPES = {"Note", "MarkdownNote"}
# Dynamic nodes (text concatenators, switches...) declare dozens of optional
# widgets in their schema but render only the handful in use; counting them
# all would estimate meter-tall nodes and blow up group bounds.
MAX_WIDGET_ROWS = 16
# Wrap a stage band's node columns once they exceed this height, so a stage
# with many parallel nodes stays roughly rectangular instead of one very tall
# column next to a short pipeline row (= a group full of empty space).
WRAP_TARGET_H = 900.0

# Nodes that render nearly empty when drafted but grow content after the
# first run (image thumbnails, generated text) - reserve room up front so the
# populated node doesn't cover its neighbors.
PREVIEW_RESERVE_H = 320.0  # image thumbnail on PreviewImage/SaveImage/...
PREVIEW_MIN_W = 340.0
TEXT_PREVIEW_RESERVE_H = 150.0  # generated text on Show Text-style nodes
_TEXT_DISPLAY_RE = re.compile(r"show.?text|display.?text|show.?anything", re.IGNORECASE)

# widget names whose nodes want extra width for editing comfort
_TEXTY_WIDGETS = {"text", "prompt", "wildcard_text", "populated_text", "string"}


def is_text_display(class_type: str) -> bool:
    """Show Text-style display nodes (populated with generated text at run time)."""
    return _TEXT_DISPLAY_RE.search(class_type) is not None


def estimate_size(
    class_type: str, object_info: dict[str, Any], widget_count: int | None = None
) -> tuple[float, float]:
    """Estimated (width, height). widget_count is the node's actual serialized
    widgets_values length when known - dynamic nodes declare dozens of optional
    widgets in their schema but render only the ones in use."""
    schema = object_info.get(class_type)
    if schema is None:
        return (DEFAULT_W, 120.0)
    try:
        slots = w.widget_slot_names(class_type, object_info)
    except ValueError:
        slots = []
    inputs = schema.get("input", {})
    connection_count = sum(
        0 if w.is_widget_input(spec) else 1
        for section in ("required", "optional")
        for spec in inputs.get(section, {}).values()
    )
    output_count = len(schema.get("output") or [])
    texty = any(s in _TEXTY_WIDGETS for s in slots)
    width = TEXT_W if texty else DEFAULT_W
    if widget_count is not None:
        rows = min(widget_count, len(slots))
    else:
        rows = min(len(slots), MAX_WIDGET_ROWS)
    height = (
        HEADER_H
        + max(connection_count, output_count) * SLOT_H
        + rows * WIDGET_H
        + (90.0 if texty else 10.0)  # multiline text area
    )
    takes_image = any(
        isinstance(spec, list | tuple) and spec and str(spec[0]).upper() == "IMAGE"
        for section in ("required", "optional")
        for spec in inputs.get(section, {}).values()
    )
    if schema.get("output_node") and takes_image:
        width = max(width, PREVIEW_MIN_W)
        height += PREVIEW_RESERVE_H
    elif is_text_display(class_type):
        width = max(width, TEXT_W)
        height += TEXT_PREVIEW_RESERVE_H
    return (width, max(height, MIN_H))


def _node_widget_count(node: Any) -> int | None:
    return len(node.widgets_values) if isinstance(node.widgets_values, list) else None


def _ranks(wf: Workflow) -> dict[int, int]:
    """Longest-path rank per linked node; unlinked non-note nodes go to rank 0."""
    downstream: dict[int, list[int]] = {}
    indegree: dict[int, int] = {}
    linked = set()
    for link in wf.links.values():
        if link.origin_id not in wf.nodes or link.target_id not in wf.nodes:
            continue
        downstream.setdefault(link.origin_id, []).append(link.target_id)
        indegree[link.target_id] = indegree.get(link.target_id, 0) + 1
        linked.update((link.origin_id, link.target_id))
    rank = {nid: 0 for nid in linked}
    frontier = [nid for nid in sorted(linked) if indegree.get(nid, 0) == 0]
    remaining_in = dict(indegree)
    order: list[int] = []
    while frontier:
        nid = frontier.pop(0)
        order.append(nid)
        for child in downstream.get(nid, []):
            rank[child] = max(rank[child], rank[nid] + 1)
            remaining_in[child] -= 1
            if remaining_in[child] == 0:
                frontier.append(child)
    for node in wf.nodes.values():
        if node.id not in rank and node.type not in NOTE_TYPES:
            rank[node.id] = 0
    return rank


def apply_layout(
    wf: Workflow, object_info: dict[str, Any], origin: tuple[float, float] = (0.0, 0.0)
) -> None:
    rank = _ranks(wf)
    if not rank:
        return
    for nid in rank:
        node = wf.nodes[nid]
        node.size = list(estimate_size(node.type, object_info, _node_widget_count(node)))

    columns: dict[int, list[int]] = {}
    for nid, r in rank.items():
        columns.setdefault(r, []).append(nid)

    upstream: dict[int, list[int]] = {}
    for link in wf.links.values():
        upstream.setdefault(link.target_id, []).append(link.origin_id)

    # order each column by barycenter of upstream y-order (two sweeps)
    y_order: dict[int, float] = {}
    for r in sorted(columns):
        col = columns[r]
        col.sort(key=lambda nid: (y_order.get(nid, 0.0), nid))
        if r > 0:
            def bary(nid: int) -> tuple[float, int]:
                ups = [y_order[u] for u in upstream.get(nid, []) if u in y_order]
                return (sum(ups) / len(ups) if ups else 1e9, nid)

            col.sort(key=bary)
        for i, nid in enumerate(col):
            y_order[nid] = float(i)

    # x per column from max width of previous columns
    x_cursor = origin[0]
    col_x: dict[int, float] = {}
    for r in sorted(columns):
        col_x[r] = x_cursor
        widest = max(wf.nodes[nid].size[0] for nid in columns[r])
        x_cursor += widest + X_GUTTER

    # stack nodes in each column, vertically centered around the tallest column
    col_heights = {
        r: sum(wf.nodes[nid].size[1] for nid in col) + Y_GAP * (len(col) - 1)
        for r, col in columns.items()
    }
    max_height = max(col_heights.values())
    for r, col in columns.items():
        y_cursor = origin[1] + (max_height - col_heights[r]) / 2.0
        for nid in col:
            node = wf.nodes[nid]
            node.pos = [col_x[r], y_cursor]
            y_cursor += node.size[1] + Y_GAP

    # execution order metadata roughly matches visual order
    for i, nid in enumerate(sorted(rank, key=lambda n: (rank[n], y_order.get(n, 0)))):
        wf.nodes[nid].order = i

    # park note nodes in their own column left of the graph; the annotator
    # will reposition the ones it adopts next to their groups
    notes = sorted(
        (n for n in wf.nodes.values() if n.type in NOTE_TYPES), key=lambda n: n.id
    )
    if notes:
        note_x = origin[0] - max(n.size[0] for n in notes) - X_GUTTER
        y_cursor = origin[1]
        for note in notes:
            note.pos = [note_x, y_cursor]
            y_cursor += note.size[1] + Y_GAP


def apply_staged_layout(
    wf: Workflow,
    object_info: dict[str, Any],
    stage_of: dict[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
    companion_of: dict[int, int] | None = None,
) -> dict[int, tuple[float, float, float, float]]:
    """Lay nodes out in left-to-right stage bands (models | prompts | sampling | ...).

    Within a band, nodes form sub-columns by their global dataflow rank, so
    chains like checkpoint -> lora -> lora stay ordered. Bands are top-aligned,
    which leaves the space above every band free for the annotator's notes.

    companion_of maps display nodes (Show Text, PreviewImage...) to the node
    they visualize; a companion is glued directly beneath its source so the
    reader never hunts for which sampler/wildcard produced which output.

    Returns {stage_index: (x, y, width, height)} bounding box per band.
    """
    companion_of = {
        nid: src
        for nid, src in (companion_of or {}).items()
        if nid in stage_of and src in stage_of and nid != src
    }
    companions: dict[int, list[int]] = {}
    for nid, src in sorted(companion_of.items()):
        companions.setdefault(src, []).append(nid)

    rank = _ranks(wf)
    for nid in stage_of:
        node = wf.nodes[nid]
        if node.type not in NOTE_TYPES:
            node.size = list(
                estimate_size(node.type, object_info, _node_widget_count(node))
            )

    # bands hold cluster heads only; companions ride along with their source
    bands: dict[int, list[int]] = {}
    for nid, stage in sorted(stage_of.items()):
        if wf.nodes[nid].type not in NOTE_TYPES and nid not in companion_of:
            bands.setdefault(stage, []).append(nid)

    def cluster(head: int) -> list[int]:
        return [head, *companions.get(head, [])]

    def cluster_h(head: int) -> float:
        return sum(wf.nodes[nid].size[1] + Y_GAP for nid in cluster(head))

    boxes: dict[int, tuple[float, float, float, float]] = {}
    x_cursor = origin[0]
    for stage in sorted(bands):
        heads = bands[stage]
        # sub-columns by global rank (of the cluster head)
        sub_ranks = sorted({rank.get(nid, 0) for nid in heads})
        sub_index = {r: i for i, r in enumerate(sub_ranks)}
        sub_cols: dict[int, list[int]] = {}
        for nid in heads:
            sub_cols.setdefault(sub_index[rank.get(nid, 0)], []).append(nid)
        # wrap each rank-column into height-limited visual columns: a stage
        # with many parallel same-rank nodes should grow sideways, not into
        # one tall column beside a short pipeline row
        target_h = max(
            WRAP_TARGET_H, max(cluster_h(head) for head in heads)
        )
        columns: list[list[int]] = []
        for i in sorted(sub_cols):
            col = sorted(sub_cols[i], key=lambda nid: (rank.get(nid, 0), nid))
            chunk: list[int] = []
            chunk_h = 0.0
            for head in col:
                unit_h = cluster_h(head)
                if chunk and chunk_h + unit_h > target_h:
                    columns.append(chunk)
                    chunk, chunk_h = [], 0.0
                chunk.extend(cluster(head))
                chunk_h += unit_h
            if chunk:
                columns.append(chunk)
        col_widths = [
            max(wf.nodes[nid].size[0] for nid in col) for col in columns
        ]
        band_x = x_cursor
        band_h = 0.0
        for i, col in enumerate(columns):
            col_x = band_x + sum(col_widths[:i]) + i * (X_GUTTER / 2)
            y_cursor = origin[1]
            for nid in col:
                node = wf.nodes[nid]
                node.pos = [col_x, y_cursor]
                y_cursor += node.size[1] + Y_GAP
            band_h = max(band_h, y_cursor - Y_GAP - origin[1])
        band_w = sum(col_widths) + (len(col_widths) - 1) * (X_GUTTER / 2)
        boxes[stage] = (band_x, origin[1], band_w, band_h)
        x_cursor = band_x + band_w + X_GUTTER
    # execution order roughly left-to-right, top-to-bottom
    ordered = sorted(
        (nid for nid in stage_of if wf.nodes[nid].type not in NOTE_TYPES),
        key=lambda nid: (rank.get(nid, 0), wf.nodes[nid].pos[0], wf.nodes[nid].pos[1]),
    )
    for i, nid in enumerate(ordered):
        wf.nodes[nid].order = i
    return boxes
