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
MAX_TOOLTIP_CHARS = 160  # tooltips are rarely needed to wire a node correctly


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
                    "description": description[:120],
                    "category": cat,
                    "output_node": bool(schema.get("output_node")),
                },
            )
        )
    hits.sort(key=lambda pair: (pair[0], pair[1]["class_type"]))
    return [h for _, h in hits[:limit]]


def _sub_widget_descriptors(parent: str, option: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe the conditional sub-widgets a dynamic-combo option reveals, with
    their dotted set_widget names (e.g. 'output.normalization')."""
    descriptors: list[dict[str, Any]] = []
    inputs = (option or {}).get("inputs", {}) or {}
    for section in ("required", "optional"):
        for name, spec in (inputs.get(section) or {}).items():
            if not w.is_widget_input(spec):
                continue
            opts = spec[1] if isinstance(spec, list | tuple) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            kind = spec[0]
            desc: dict[str, Any] = {"name": f"{parent}.{name}"}
            if w.is_dynamic_combo(spec):
                desc["type"] = "COMBO"
                desc["dynamic_combo"] = True
                desc["choices"] = [o.get("key") for o in w.dynamic_options(spec)]
                desc["default"] = w.dynamic_default_key(spec)
            elif isinstance(kind, list):
                desc["type"] = "COMBO"
                desc["choices"] = kind[:MAX_COMBO_CHOICES]
            elif kind == "COMBO":
                desc["type"] = "COMBO"
                desc["choices"] = opts.get("options", [])[:MAX_COMBO_CHOICES]
            else:
                desc["type"] = str(kind)
            if "default" in opts:
                desc["default"] = opts["default"]
            descriptors.append(desc)
    return descriptors


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
            elif w.is_dynamic_combo(spec):
                # a V3 dynamic combo: the main value is one of the option keys,
                # and the selected key reveals dotted sub-widgets. Surface both
                # so the agent can set_widget them (main first, then subs).
                entry["type"] = "COMBO"
                entry["dynamic_combo"] = True
                entry["choices"] = [o.get("key") for o in w.dynamic_options(spec)]
                entry["default"] = w.dynamic_default_key(spec)
                entry["options"] = {
                    o.get("key"): _sub_widget_descriptors(name, o)
                    for o in w.dynamic_options(spec)
                }
            else:
                entry["type"] = str(kind)
            for key in ("default", "min", "max", "tooltip", "control_after_generate"):
                if key in opts:
                    entry[key] = opts[key]
            if isinstance(entry.get("tooltip"), str) and len(entry["tooltip"]) > MAX_TOOLTIP_CHARS:
                entry["tooltip"] = entry["tooltip"][:MAX_TOOLTIP_CHARS] + "…"
            if opts.get("control_after_generate"):
                entry["control_slot"] = f"{name}__control_after_generate"
            inputs.append(entry)
    out_names = schema.get("output_name") or []
    outputs = [
        {"name": str(out_names[i]) if i < len(out_names) else str(t), "type": str(t)}
        for i, t in enumerate(schema.get("output") or [])
    ]
    return {
        "class_type": class_type,
        "display_name": schema.get("display_name", class_type),
        "description": (schema.get("description") or "")[:300],
        "category": schema.get("category", ""),
        "output_node": bool(schema.get("output_node")),
        "inputs": inputs,
        "outputs": outputs,
    }
