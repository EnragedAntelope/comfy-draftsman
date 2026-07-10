"""Workflow validation against the live instance's node catalog.

Because object_info comes from the running ComfyUI, combo choices embed the
actual model files present on disk - so combo membership checks double as
"is this model installed" checks. Findings carry fix suggestions, which makes
this the engine behind both validate_workflow and diagnose_workflow.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from . import widgets as w
from .model import MODE_NORMAL, VIRTUAL_TYPES, Workflow

# A combo's option list is authoritative when we can trust the /object_info
# snapshot: either it lists on-disk files (an "is this installed" check) or it
# belongs to a core node (baked enums like sampler_name/scheduler). Third-party
# nodes commonly repopulate their combos client-side (wildcard/LoRA/style
# pickers), so a saved value absent from their snapshot isn't necessarily wrong.
_FILE_COMBO_RE = re.compile(
    r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|sft|vae|pkl|yaml|yml)$", re.IGNORECASE
)


def _looks_like_file_combo(choices: list[Any]) -> bool:
    return any(
        isinstance(c, str) and (_FILE_COMBO_RE.search(c) or "/" in c or "\\" in c)
        for c in choices
    )


def _is_custom_node(class_type: str, object_info: dict[str, Any]) -> bool:
    """True if this class comes from a third-party pack (python_module under
    ``custom_nodes``) rather than core/bundled ComfyUI. Missing -> treated as
    core (strict), so an unknown never silently relaxes validation."""
    module = str((object_info.get(class_type) or {}).get("python_module") or "")
    return module.startswith("custom_nodes")


def _authoritative_combo(
    class_type: str, choices: list[Any], object_info: dict[str, Any]
) -> bool:
    """Whether a combo-membership failure should block (error) rather than warn:
    on-disk file listings always, and any combo on a core node."""
    return _looks_like_file_combo(choices) or not _is_custom_node(class_type, object_info)


def _step_aligned(value: float, min_val: float, step: float) -> bool:
    """True if ``value`` sits on the widget's step grid. Accepts alignment to
    origin 0 OR to ``min``: a schema often sets ``min`` to a tiny epsilon (e.g.
    0.0001, meaning ">0") that would otherwise offset the whole grid and reject
    every normal value. Tolerance is step-relative to survive float error."""
    if step is None or step <= 0:
        return True
    for origin in {0.0, float(min_val or 0)}:
        k = round((value - origin) / step)
        if abs(origin + k * step - value) <= step * 1e-4:
            return True
    return False


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


def check_widget_value(
    class_type: str,
    input_name: str,
    value: Any,
    object_info: dict[str, Any],
    widgets_values: Any = None,
) -> str | None:
    """Actionable error string if ``value`` is invalid for this widget, else
    None. Used by edit ops to reject made-up values at WRITE time - validate()
    catches the same problems later, but late feedback wastes a round trip.
    Widget-NAME checks live in set_widget/add_node; unknown names pass here."""
    if class_type not in object_info or input_name.endswith(w.SYNTHETIC_SUFFIXES):
        return None
    spec = w.widget_specs(class_type, object_info, widgets_values).get(input_name)
    if spec is None:
        return None
    if value is None:
        return (
            f"'{input_name}' cannot be null - the ComfyUI editor crashes on null "
            "widget values (empty string is fine)"
        )
    choices = _combo_choices(spec)
    if choices:
        if value in choices:
            return None
        # A value absent from a non-authoritative combo (a third-party node that
        # populates its own list client-side: wildcard/LoRA/style picker) is most
        # likely legitimate - don't reject the write. validate() still surfaces it
        # as a non-blocking warning; core enums and file lists stay strict.
        if not _authoritative_combo(class_type, choices, object_info):
            return None
        close = difflib.get_close_matches(
            str(value), [str(c) for c in choices], n=3, cutoff=0.4
        )
        listing = (
            f"close matches: {close}" if close else f"e.g. {[str(c) for c in choices[:8]]}"
        )
        browse = (
            f"; browse all {len(choices)} via get_node_info('{class_type}', "
            "choices_filter=...)"
            if len(choices) > 8
            else ""
        )
        return (
            f"'{input_name}' = {value!r} is not an available option on this "
            f"instance - {listing}{browse}. Only listed values run; "
            '"force": true overrides if you know better'
        )
    kind = spec[0]
    if kind == "INT" and (isinstance(value, bool) or not isinstance(value, int)):
        return f"'{input_name}' expects an integer, got {type(value).__name__} {value!r}"
    if kind == "FLOAT" and (isinstance(value, bool) or not isinstance(value, int | float)):
        return f"'{input_name}' expects a number, got {type(value).__name__} {value!r}"
    if kind == "STRING" and not isinstance(value, str):
        return f"'{input_name}' expects a string, got {type(value).__name__} {value!r}"
    if kind == "BOOLEAN" and not isinstance(value, bool):
        return f"'{input_name}' expects true/false, got {type(value).__name__} {value!r}"
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(value, int | float) and not isinstance(value, bool):
        low, high = opts.get("min"), opts.get("max")
        if (low is not None and value < low) or (high is not None and value > high):
            return f"'{input_name}' = {value} is outside the allowed range [{low}, {high}]"
        step = opts.get("step")
        if step is not None and step > 0:
            min_val = opts.get("min", 0) or 0
            if not _step_aligned(value, min_val, step):
                return f"'{input_name}' = {value} is not aligned to step {step} (min {min_val})"
    return None


def validate(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a workflow; subgraph instances are flattened first so inner
    nodes get the same checks, with findings carrying subgraph provenance."""
    from .subgraph import flatten, has_subgraph_instances

    if not has_subgraph_instances(wf):
        return _validate_nodes(wf, object_info)
    try:
        flat, provenance, diagnostics = flatten(wf, object_info)
    except ValueError as e:
        findings = _validate_nodes(wf, object_info)
        findings.append(
            _finding(
                "error",
                "subgraph-flatten-failed",
                f"could not flatten subgraph instances for validation/run: {e}",
            )
        )
        return findings
    findings = _validate_nodes(flat, object_info)
    for d in diagnostics:
        findings.append(
            _finding(
                "warning",
                "subgraph-missing-inner-inputs",
                f"subgraph '{d['subgraph']}': inner node #{d.get('inner_node_id', '?')} "
                f"dropped boundary link ({d.get('reason', 'unknown')}); "
                "the node may be missing its inputs/outputs arrays",
                d.get("inner_node_id"),
                subgraph=d.get("subgraph"),
            )
        )
    for f in findings:
        origin = provenance.get(f.get("node_id", -1))
        if origin:
            f["subgraph"] = origin["subgraph"]
            f["inner_node"] = origin["path"]
            f["message"] += (
                f" [inner node {origin['path']} of subgraph '{origin['subgraph']}' - "
                "edit_workflow can't reach inside; rebuild flat to change it]"
            )
    defs = wf.subgraph_defs()
    for node in wf.nodes.values():
        sg = defs.get(node.type)
        if sg is not None and node.mode == MODE_NORMAL:
            findings.append(
                _finding(
                    "info",
                    "subgraph-instance",
                    f"node #{node.id} is an instance of subgraph "
                    f"'{sg.get('name', node.type)}' - flattened automatically at "
                    "validate/run time; its inner findings (if any) are listed above",
                    node.id,
                    subgraph=sg.get("name"),
                )
            )
    return findings


def _validate_nodes(wf: Workflow, object_info: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for node in wf.nodes.values():
        if node.type in VIRTUAL_TYPES:
            continue
        schema = object_info.get(node.type)
        if schema is None:
            subgraph = wf.subgraph_defs().get(node.type)
            if subgraph is not None:
                # reached only for muted/bypassed instances (never executed) or
                # when flattening failed - active instances validate flattened
                findings.append(
                    _finding(
                        "warning",
                        "subgraph-instance",
                        f"node #{node.id} is an instance of subgraph "
                        f"'{subgraph.get('name', node.type)}' "
                        f"({len(subgraph.get('nodes', []) or [])} inner nodes), "
                        "left unflattened here (muted/bypassed or malformed "
                        "definition) - its internals aren't validated",
                        node.id,
                        subgraph=subgraph.get("name"),
                    )
                )
                continue
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

        socket_names = {slot.name for slot in node.inputs}
        slots = w.widget_slot_names(node.type, object_info, node.widgets_values, socket_names)
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
        specs = w.widget_specs(node.type, object_info, node.widgets_values, socket_names)
        named = w.widgets_to_named(node.type, node.widgets_values, object_info, socket_names)
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
                if _authoritative_combo(node.type, choices, object_info):
                    # on-disk listing or a core node's baked enum: genuinely wrong
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
                else:
                    # third-party node that likely fills this combo client-side
                    # (wildcard/LoRA/style picker): advisory, non-blocking -
                    # ComfyUI is the final judge.
                    findings.append(
                        _finding(
                            "warning",
                            "combo-value-unlisted",
                            f"{node.type} #{node.id}: '{name}' = {value!r} is not in this "
                            "instance's schema options - fine if this custom node fills the "
                            "list client-side (wildcard/LoRA/style picker); otherwise "
                            "list options with get_node_info",
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
                step = opts.get("step")
                if step is not None and step > 0:
                    min_val = opts.get("min", 0) or 0
                    if not _step_aligned(value, min_val, step):
                        findings.append(
                            _finding(
                                "warning",
                                "step-misaligned",
                                f"{node.type} #{node.id}: '{name}' = {value} is not aligned to step "
                                f"{step} (min {min_val})",
                                node.id,
                                input=name,
                            )
                        )

        for name, spec in schema.get("input", {}).get("required", {}).items():
            # base widgets, and custom JS-widget inputs the node didn't serialize
            # as a socket, are populated from widgets_values - not "unconnected"
            if w.is_widget_input(spec) or w._is_custom_widget(name, spec, socket_names):
                continue
            slot = node.input_by_name(name)
            if slot is not None and slot.link is None and slot.widget_name:
                # a custom-typed input the node exposes as a widget-backed slot
                # (carries a `widget` marker): its value is pack-specific frontend
                # JS state (e.g. LoraManager's autocomplete), not a plain scalar
                # the raw /prompt API can send. Driving it headlessly would silently
                # no-op the node's whole branch - so block loudly with the fix.
                kind = spec[0] if isinstance(spec, list | tuple) and spec else spec
                findings.append(
                    _finding(
                        "error",
                        "js-widget-input",
                        f"{node.type} #{node.id}: required input '{name}' (type "
                        f"{kind}) is a custom widget its pack's frontend JS fills in "
                        "the browser; the raw API can't, so a headless run would "
                        "no-op this node's branch. Connect it, or swap this node for "
                        "the pack's plain-STRING variant / a core equivalent",
                        node.id,
                        input=name,
                    )
                )
            elif slot is None or slot.link is None:
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
