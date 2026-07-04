"""Compact search and summaries over ComfyUI's object_info document.

object_info is multi-megabyte (combo inputs embed every model filename on the
instance), so agents must never receive it raw. Search returns lightweight
descriptors; node_summary returns one node's schema with giant combo lists
truncated.
"""

from __future__ import annotations

from typing import Any

from ..graph import widgets as w

MAX_COMBO_CHOICES = 24


def search_nodes(
    object_info: dict[str, Any],
    query: str,
    category: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Case-insensitive substring search over class name, display name, description."""
    q = query.lower().strip()
    hits: list[tuple[int, dict[str, Any]]] = []
    for class_type, schema in object_info.items():
        cat = schema.get("category", "") or ""
        if category and category.lower() not in cat.lower():
            continue
        display = schema.get("display_name", "") or ""
        description = schema.get("description", "") or ""
        haystacks = (class_type.lower(), display.lower(), description.lower())
        if q:
            if q in haystacks[0]:
                rank = 0
            elif q in haystacks[1]:
                rank = 1
            elif q in haystacks[2]:
                rank = 2
            else:
                continue
        else:
            rank = 3
        hits.append(
            (
                rank,
                {
                    "class_type": class_type,
                    "display_name": display,
                    "description": description[:200],
                    "category": cat,
                    "output_node": bool(schema.get("output_node")),
                },
            )
        )
    hits.sort(key=lambda pair: (pair[0], pair[1]["class_type"]))
    return [h for _, h in hits[:limit]]


def node_summary(object_info: dict[str, Any], class_type: str) -> dict[str, Any]:
    """One node's full slot schema, sized for an agent to wire it correctly."""
    schema = object_info[class_type]
    inputs = []
    for section in ("required", "optional"):
        for name, spec in schema.get("input", {}).get(section, {}).items():
            entry: dict[str, Any] = {
                "name": name,
                "required": section == "required",
                "widget": w.is_widget_input(spec),
            }
            kind = spec[0] if isinstance(spec, list | tuple) and spec else "*"
            opts = spec[1] if isinstance(spec, list | tuple) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(kind, list):
                entry["type"] = "COMBO"
                entry["choices"] = kind[:MAX_COMBO_CHOICES]
                if len(kind) > MAX_COMBO_CHOICES:
                    entry["choices_truncated"] = len(kind)
            elif kind == "COMBO":
                options = opts.get("options", [])
                entry["type"] = "COMBO"
                entry["choices"] = options[:MAX_COMBO_CHOICES]
                if len(options) > MAX_COMBO_CHOICES:
                    entry["choices_truncated"] = len(options)
            else:
                entry["type"] = str(kind)
            for key in ("default", "min", "max", "tooltip", "control_after_generate"):
                if key in opts:
                    entry[key] = opts[key]
            inputs.append(entry)
    out_names = schema.get("output_name") or []
    outputs = [
        {"name": str(out_names[i]) if i < len(out_names) else str(t), "type": str(t)}
        for i, t in enumerate(schema.get("output") or [])
    ]
    return {
        "class_type": class_type,
        "display_name": schema.get("display_name", class_type),
        "description": (schema.get("description") or "")[:500],
        "category": schema.get("category", ""),
        "output_node": bool(schema.get("output_node")),
        "inputs": inputs,
        "outputs": outputs,
    }
