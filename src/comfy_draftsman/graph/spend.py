"""Which nodes in a graph cost real money to run.

Partner/API nodes (Luma, Kling, Runway, Seedance, ...) execute on someone
else's hardware against the user's Comfy Org account, so queueing one is a
purchase, not a render. This module answers "does submitting this graph spend
credits?" from the /object_info snapshot alone - no I/O, no live probe.

The bias is deliberately toward under-reporting: an instance too old to emit
the ``api_node`` flag, or a pack that does not set it, is treated as free.
Flagging everything unknown as billable would train users to click through the
one confirmation that actually matters.
"""

from __future__ import annotations

from typing import Any

from .model import MODE_NORMAL, VIRTUAL_TYPES, Workflow


def is_api_node(schema: dict[str, Any] | None) -> bool:
    """True when this class's /object_info entry says it bills the user.

    Primary signal is ComfyUI's own per-class ``api_node`` boolean. The category
    fallback exists because that flag is comparatively recent - a partner pack
    on an older instance still files itself under an "api node/..." category.
    """
    if not isinstance(schema, dict):
        return False
    if schema.get("api_node") is True:
        return True
    category = schema.get("category")
    return isinstance(category, str) and category.lower().startswith("api node")


def api_nodes(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    """[{node_id, class_type, title?}] for every node in the graph that spends.

    Disabled nodes are excluded: a muted or bypassed partner node never reaches
    the executor, so confirming a spend for it would be a prompt for something
    that cannot happen.
    """
    found: list[dict[str, Any]] = []
    for node in sorted(wf.nodes.values(), key=lambda n: n.id):
        if node.type in VIRTUAL_TYPES or node.mode != MODE_NORMAL:
            continue
        if not is_api_node(object_info.get(node.type)):
            continue
        entry: dict[str, Any] = {"node_id": node.id, "class_type": node.type}
        if node.title and node.title != node.type:
            entry["title"] = node.title
        found.append(entry)
    return found
