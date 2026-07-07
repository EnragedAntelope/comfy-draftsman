"""Compact search and summaries over ComfyUI's object_info document.

object_info is multi-megabyte (combo inputs embed every model filename on the
instance), so agents must never receive it raw. Search returns lightweight
descriptors; node_summary returns one node's schema with giant combo lists
truncated.
"""

from __future__ import annotations

import json
from typing import Any

from ..graph import widgets as w

MAX_COMBO_CHOICES = 24
MAX_TOOLTIP_CHARS = 160  # tooltips are rarely needed to wire a node correctly
MAX_TRIGGER_TAGS = 15


def metadata_digest(meta: dict[str, Any], max_tags: int = MAX_TRIGGER_TAGS) -> dict[str, Any]:
    """Trim a safetensors __metadata__ dict to what matters for USING the model:
    base-model/architecture keys and the top training tags by frequency (a
    LoRA's trigger words live in ss_tag_frequency). Raw headers can be tens of
    KB of per-image tag counts - never return them whole."""
    out: dict[str, Any] = {}
    for key in (
        "ss_base_model_version",
        "ss_sd_model_name",
        "modelspec.architecture",
        "modelspec.title",
        "ss_network_module",
        "ss_network_dim",
        "ss_network_alpha",
        "ss_output_name",
        "ss_resolution",
    ):
        if meta.get(key):
            out[key] = meta[key]
    freq_raw = meta.get("ss_tag_frequency")
    if freq_raw:
        try:
            data = json.loads(freq_raw) if isinstance(freq_raw, str) else freq_raw
            counts: dict[str, int] = {}
            for tags in data.values():  # {dataset_name: {tag: count}}
                if isinstance(tags, dict):
                    for tag, n in tags.items():
                        counts[tag.strip()] = counts.get(tag.strip(), 0) + int(n)
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_tags]
            out["top_training_tags"] = [f"{tag} ({n})" for tag, n in top]
        except (ValueError, TypeError, AttributeError):
            pass
    if not out:
        out["note"] = (
            "no recognizable training metadata; raw keys: "
            + ", ".join(sorted(meta)[:20])
        )
    return out


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


def _apply_choices(
    entry: dict[str, Any],
    options: list[Any],
    choices_filter: str,
    max_choices: int,
) -> None:
    """Attach a combo's choices to an input entry, filtered and capped. When the
    list is cut, say how to see the rest (agents can't guess hidden entries)."""
    total = len(options)
    if choices_filter:
        needle = choices_filter.lower()
        options = [o for o in options if needle in str(o).lower()]
    cap = max_choices if max_choices > 0 else MAX_COMBO_CHOICES
    entry["choices"] = options[:cap]
    if choices_filter:
        entry["choices_matched"] = len(options)
        entry["choices_total"] = total
    if len(options) > cap:
        entry["choices_truncated"] = len(options)
        entry["choices_hint"] = (
            f"showing {cap} of {len(options)}; re-call get_node_info with "
            "choices_filter='substring' or a larger max_choices"
        )


def node_summary(
    object_info: dict[str, Any],
    class_type: str,
    choices_filter: str = "",
    max_choices: int = 0,
) -> dict[str, Any]:
    """One node's full slot schema, sized for an agent to wire it correctly.

    choices_filter / max_choices control combo-choice listing: filter is a
    case-insensitive substring over every combo input's choices; max_choices
    raises the per-combo cap (default MAX_COMBO_CHOICES)."""
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
                _apply_choices(entry, kind, choices_filter, max_choices)
            elif kind == "COMBO":
                entry["type"] = "COMBO"
                _apply_choices(entry, opts.get("options", []), choices_filter, max_choices)
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
            if entry["widget"] and w.has_control_slot(name, spec):
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
