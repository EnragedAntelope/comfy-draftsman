"""Which nodes in a graph cost real money to run.

Partner/API nodes (Luma, Kling, Runway, Seedance, ...) execute on someone
else's hardware against the user's Comfy Org account, so queueing one is a
purchase, not a render. This module answers "does submitting this graph spend
credits?" from the /object_info snapshot alone - no I/O, no live probe.

It reads the **API prompt**, not the editing graph, because the API prompt is
by definition exactly what POST /prompt will execute: subgraph instances are
already flattened into their inner nodes, and muted/bypassed nodes are already
gone. Scanning the graph's top-level nodes instead would miss a partner node
packaged inside a subgraph - whose node type is a definition uuid that appears
nowhere in object_info - and bill the user silently.

The bias is deliberately toward under-reporting: an instance too old to emit
the ``api_node`` flag, or a pack that does not set it, is treated as free.
Flagging everything unknown as billable would train users to click through the
one confirmation that actually matters.
"""

from __future__ import annotations

from typing import Any


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


def api_nodes(api_prompt: dict[str, Any], object_info: dict[str, Any]) -> list[dict[str, Any]]:
    """[{node_id, class_type}] for every node in the API prompt that spends."""
    found: list[dict[str, Any]] = []
    for node_id, node in api_prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not is_api_node(object_info.get(class_type)):
            continue
        found.append({"node_id": node_id, "class_type": class_type})
    # API prompt keys are stringified node ids; sort numerically where possible
    # so #2 doesn't sort before #10 in the list shown to the user
    found.sort(key=lambda n: (0, int(n["node_id"])) if str(n["node_id"]).isdigit() else (1, 0))
    return found
