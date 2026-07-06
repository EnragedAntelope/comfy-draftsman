"""Workflow validation against the live instance's node catalog.

Because object_info comes from the running ComfyUI, combo choices embed the
actual model files present on disk - so combo membership checks double as
"is this model installed" checks. Findings carry fix suggestions, which makes
this the engine behind both validate_workflow and diagnose_workflow.
"""

from __future__ import annotations

import difflib
from typing import Any

from . import widgets as w
from .model import VIRTUAL_TYPES, Workflow


def _finding(
    level: str, code: str, message: str, node_id: int | None = None, **extra: Any
) -> dict[str, Any]:
    finding: dict[str, Any] = {"level": level, "code": code, "message": message}
    if node_id is not None:
        finding["node_id"] = node_id
    finding.update(extra)
    return finding


def _combo_choices(spec: Any) -> list[Any] | None:
    kind = spec[0]
    if isinstance(kind, list):
        return kind
    if w.is_dynamic_combo(spec):
        # the main value must be one of the option keys
        return [o.get("key") for o in w.dynamic_options(spec)]
    if kind == "COMBO":
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        return opts.get("options", [])
    return None


def validate(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for node in wf.nodes.values():
        if node.type in VIRTUAL_TYPES:
            continue
        schema = object_info.get(node.type)
        if schema is None:
            findings.append(
                _finding(
                    "error",
                    "missing-node-class",
                    f"node #{node.id}: class '{node.type}' is not installed on this "
                    "instance - resolve it via the Comfy Registry (resolve_missing_nodes)",
                    node.id,
                    class_type=node.type,
                )
            )
            continue

        slots = w.widget_slot_names(node.type, object_info, node.widgets_values)
        if isinstance(node.widgets_values, list) and len(node.widgets_values) != len(slots):
            # dynamic nodes (text concatenators, switches...) declare dozens of
            # optional widgets in their schema but the frontend serializes only
            # the ones in use - a shortfall there is normal, not drift
            optional_widgets = sum(
                1
                for spec in (schema.get("input", {}).get("optional", {}) or {}).values()
                if w.is_widget_input(spec)
            )
            dynamic_short = len(node.widgets_values) < len(slots) and optional_widgets >= 6
            if dynamic_short:
                findings.append(
                    _finding(
                        "info",
                        "widget-count-drift",
                        f"{node.type} #{node.id}: {len(node.widgets_values)} of {len(slots)} "
                        "schema widgets serialized - this node declares many optional "
                        "widgets and serializes only the ones in use; usually harmless",
                        node.id,
                        expected=slots,
                    )
                )
            else:
                findings.append(
                    _finding(
                        "warning",
                        "widget-count-drift",
                        f"{node.type} #{node.id}: has {len(node.widgets_values)} widget values "
                        f"but current schema expects {len(slots)} ({slots}) - the node's "
                        "parameters changed since this workflow was made; re-check each value",
                        node.id,
                        expected=slots,
                    )
                )

        # real widget slots for the current selection, incl. dotted sub-widgets
        # of a dynamic combo's chosen option - so their values get validated too
        specs = w.widget_specs(node.type, object_info, node.widgets_values)
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        for name, value in named.items():
            if value is None:
                # the frontend runs string replacement over every widget value
                # when queueing, so a null crashes it even if the slot is
                # connected or optional
                findings.append(
                    _finding(
                        "error",
                        "null-widget-value",
                        f"{node.type} #{node.id}: widget '{name}' is null - the ComfyUI "
                        "editor crashes on null widget values (\"Cannot read properties "
                        "of null\"); set a concrete value (empty string is fine)",
                        node.id,
                        input=name,
                    )
                )
                continue
            if name.endswith(w.SYNTHETIC_SUFFIXES) or name not in specs:
                continue
            spec = specs[name]
            slot = node.input_by_name(name)
            if slot is not None and slot.link is not None:
                continue  # connected: widget value is overridden
            choices = _combo_choices(spec)
            if choices is not None and choices and value not in choices:
                close = difflib.get_close_matches(str(value), [str(c) for c in choices], n=1, cutoff=0.4)
                findings.append(
                    _finding(
                        "error",
                        "invalid-combo-value",
                        f"{node.type} #{node.id}: '{name}' = {value!r} is not available "
                        + (f"- closest installed option: {close[0]!r}" if close else
                           "- list options with get_node_info / list_models"),
                        node.id,
                        input=name,
                        suggestion=close[0] if close else None,
                    )
                )
                continue
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(value, int | float) and not isinstance(value, bool):
                low, high = opts.get("min"), opts.get("max")
                if (low is not None and value < low) or (high is not None and value > high):
                    findings.append(
                        _finding(
                            "error",
                            "out-of-range",
                            f"{node.type} #{node.id}: '{name}' = {value} outside "
                            f"[{low}, {high}]",
                            node.id,
                            input=name,
                        )
                    )

        for name, spec in schema.get("input", {}).get("required", {}).items():
            if w.is_widget_input(spec):
                continue
            slot = node.input_by_name(name)
            if slot is None or slot.link is None:
                findings.append(
                    _finding(
                        "error",
                        "unconnected-input",
                        f"{node.type} #{node.id}: required input '{name}' is not connected",
                        node.id,
                        input=name,
                    )
                )
    return findings
