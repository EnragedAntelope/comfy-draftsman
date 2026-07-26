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
from .model import (
    MODE_BYPASS,
    MODE_MUTE,
    MODE_NORMAL,
    PRIMITIVE_TYPE,
    VIRTUAL_TYPES,
    Node,
    Workflow,
    types_compatible,
)

# A combo's option list is authoritative when we can trust the /object_info
# snapshot: either it lists on-disk files (an "is this installed" check) or it
# belongs to a core node (baked enums like sampler_name/scheduler). Third-party
# nodes commonly repopulate their combos client-side (wildcard/LoRA/style
# pickers), so a saved value absent from their snapshot isn't necessarily wrong.
_FILE_COMBO_RE = re.compile(
    r"\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|sft|vae|pkl|yaml|yml)$", re.IGNORECASE
)

# How many disabled node ids to name in the single collapsed `node-disabled`
# note; the rest are counted. The full list is always in its `node_ids` field.
_DISABLED_IDS_SHOWN = 12


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


_combo_choices = w.combo_choices


def check_widget_value(
    class_type: str,
    input_name: str,
    value: Any,
    object_info: dict[str, Any],
    widgets_values: Any = None,
    socket_names: set[str] | None = None,
) -> str | None:
    """Actionable error string if ``value`` is invalid for this widget, else
    None. Used by edit ops to reject made-up values at WRITE time - validate()
    catches the same problems later, but late feedback wastes a round trip.
    Widget-NAME checks live in set_widget/add_node; unknown names pass here.

    ``socket_names`` (an existing node's declared sockets) makes custom
    JS-widget inputs visible to the slot walk, so their spec is found and the
    positions of the widgets after them line up. Omit it for a fresh node."""
    if class_type not in object_info or input_name.endswith(w.SYNTHETIC_SUFFIXES):
        return None
    spec = w.widget_specs(class_type, object_info, widgets_values, socket_names).get(
        input_name
    )
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


def check_primitive_value(
    wf: Workflow, node: Node, value: Any, object_info: dict[str, Any]
) -> str | None:
    """Actionable error string if a PrimitiveNode's value is invalid for the
    widget(s) it mirrors, else None.

    A primitive's value is checked HERE and nowhere else: ``to_api`` inlines it
    into its consumer's input, and the consumer's own widget check is skipped
    because that slot is *connected*. Reuses ``check_widget_value``, so the same
    confidence gate applies - a client-populated third-party combo doesn't block.
    """
    for _spec, target, name in wf.primitive_targets(node.id, object_info):
        problem = check_widget_value(
            target.type,
            name,
            value,
            object_info,
            target.widgets_values,
            {s.name for s in target.inputs},
        )
        if problem:
            return f"drives {target.type} #{target.id}.{name}: {problem}"
    return None


def _primitive_findings(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for node in wf.nodes.values():
        if node.type != PRIMITIVE_TYPE or node.mode in (MODE_MUTE, MODE_BYPASS):
            continue
        if not wf.primitive_targets(node.id, object_info):
            # harmless to the run (to_api drops virtual nodes), but it has no type
            # and drives nothing - so a warning, not a blocking error
            findings.append(
                _finding(
                    "warning",
                    "primitive-unbound",
                    f"PrimitiveNode #{node.id} drives no widget, so it has no type and "
                    "does nothing. Connect it to a widget input (it then mirrors that "
                    "socket's type) or remove it",
                    node.id,
                )
            )
            continue
        value = (
            node.widgets_values[0]
            if isinstance(node.widgets_values, list) and node.widgets_values
            else None
        )
        problem = check_primitive_value(wf, node, value, object_info)
        if problem:
            findings.append(
                _finding(
                    "error",
                    "primitive-value-invalid",
                    f"PrimitiveNode #{node.id} {problem}",
                    node.id,
                )
            )
    return findings


def _schema_output_type(
    node: Node, slot_index: int, object_info: dict[str, Any]
) -> str:
    """An output slot's type, preferring the live schema over the stored slot: a
    workflow saved against an older version of a pack can carry a type the node
    no longer declares."""
    outs = (object_info.get(node.type) or {}).get("output") or []
    if isinstance(outs, list) and 0 <= slot_index < len(outs):
        return str(outs[slot_index])
    if 0 <= slot_index < len(node.outputs):
        return node.outputs[slot_index].type
    return "*"


def _schema_input_type(node: Node, slot: Any, object_info: dict[str, Any]) -> str:
    """An input slot's type from the live schema, with any combo flavour reported
    as COMBO (what the frontend types a converted combo widget as)."""
    schema = object_info.get(node.type)
    if schema is not None:
        wanted = slot.widget_name or slot.name
        for name, spec in w._iter_schema_inputs(schema):
            if name != wanted:
                continue
            if w.combo_choices(spec) is not None:
                return "COMBO"
            kind = spec[0] if isinstance(spec, list | tuple) and spec else spec
            return str(kind)
    return slot.type


def _link_type_findings(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    """Mirror ComfyUI's executor type check on every wire.

    The server answers ``return_type_mismatch``, queues the prompt anyway, and
    executes only the rest of the graph - so a mistyped wire reads as a run that
    "succeeded" with missing outputs. Nothing else in draftsman checked link
    types at all, which is how a STRING wired into a COMBO widget validated
    clean. Virtual endpoints (Reroute/PrimitiveNode) adopt their neighbour's type
    and disabled nodes never run, so both are skipped.
    """
    findings: list[dict[str, Any]] = []
    for link in sorted(wf.links.values(), key=lambda x: x.id):
        origin, target = wf.nodes.get(link.origin_id), wf.nodes.get(link.target_id)
        if origin is None or target is None:
            continue
        if origin.type in VIRTUAL_TYPES or target.type in VIRTUAL_TYPES:
            continue
        if origin.type not in object_info or target.type not in object_info:
            continue  # missing-node-class already reports this
        if origin.mode != MODE_NORMAL or target.mode != MODE_NORMAL:
            continue
        if link.target_slot >= len(target.inputs):
            continue
        slot = target.inputs[link.target_slot]
        out_type = _schema_output_type(origin, link.origin_slot, object_info)
        in_type = _schema_input_type(target, slot, object_info)
        if types_compatible(out_type, in_type):
            continue
        findings.append(
            _finding(
                "error",
                "link-type-mismatch",
                f"{origin.type} #{origin.id} ({out_type}) -> {target.type} #{target.id}"
                f".{slot.name} ({in_type}): ComfyUI rejects this wire at queue time "
                "(return_type_mismatch) and runs only the rest of the graph. "
                + (
                    "Drive a COMBO widget with set_widget, or with a PrimitiveNode "
                    "(it mirrors the socket's type and can cycle it via "
                    "control_after_generate)"
                    if in_type == "COMBO"
                    else "Rewire it to a source of the expected type"
                ),
                target.id,
                input=slot.name,
            )
        )
    return findings


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
    disabled: list[int] = []
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

        # Muted (mode 2) and bypassed (mode 4) nodes are dropped by to_api and
        # never reach ComfyUI, so their own widget values and unconnected inputs
        # cannot break the run - and muting a branch is THE standard way to
        # disable it. Checking them anyway produced blocking errors that refused
        # run_workflow/save_workflow for a graph whose prompt document doesn't
        # even contain those nodes. Collected and reported as ONE finding below
        # (a 25-node muted branch used to emit 25 near-identical notes); what a
        # disabled node does to its *consumers* is checked on the active consumer
        # itself (muted-input-source / dead-input-source).
        if node.mode in (MODE_MUTE, MODE_BYPASS):
            disabled.append(node.id)
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
            # display/output nodes (ShowText, rgthree "Display Any", preview nodes)
            # stash the text/data they show into widgets_values beyond their declared
            # schema widgets - an overflow there is the norm, not schema drift.
            display_overflow = len(node.widgets_values) > len(slots) and schema.get("output_node")
            if display_overflow:
                pass  # expected for display nodes - stay silent (pure noise otherwise)
            elif dynamic_short:
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
            if slot is not None and slot.link is not None:
                # Connected - but a MUTE (mode 2) producer is dropped entirely at
                # run time (to_api skips it; _trace_origin sees through Reroute and
                # bypass but not mute), leaving a dangling [muted_id, slot] reference
                # ComfyUI rejects. Catch it here so the failure is early and named
                # instead of a confusing run-time error on a graph that "validated".
                link = wf.links.get(slot.link)
                origin = (
                    wf._trace_origin(link.origin_id, link.origin_slot, 0)
                    if link is not None
                    else None
                )
                src = wf.nodes.get(origin[0]) if origin is not None else None
                if origin is None:
                    # The slot is wired, but the chain resolves to no producer:
                    # a BYPASSED node with nothing feeding its matching input
                    # (bypass is a passthrough, so it forwards a hole), or a
                    # link whose origin node is gone. to_api drops the input
                    # entirely and ComfyUI rejects the prompt - name it here.
                    findings.append(
                        _finding(
                            "error",
                            "dead-input-source",
                            f"{node.type} #{node.id}: required input '{name}' is wired "
                            "but the chain feeding it resolves to nothing - a bypassed "
                            "node passes its input through, so a bypassed node with "
                            "that input unconnected forwards a hole. Connect the "
                            "upstream source, or unbypass the node that should "
                            "produce this value",
                            node.id,
                            input=name,
                        )
                    )
                    continue
                if src is not None and src.mode == MODE_MUTE:
                    findings.append(
                        _finding(
                            "error",
                            "muted-input-source",
                            f"{node.type} #{node.id}: required input '{name}' is fed by "
                            f"muted node #{src.id} ({src.type}) - a muted node never "
                            "runs, so the graph would fail at execution with a missing "
                            f"input. Unmute #{src.id} (set_mode 0) or rewire '{name}' to "
                            "an active source",
                            node.id,
                            input=name,
                        )
                    )
                continue
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
    findings.extend(_primitive_findings(wf, object_info))
    findings.extend(_link_type_findings(wf, object_info))
    if disabled:
        # one note for the whole set: repeating it per node was pure token cost
        # on a graph with a disabled branch, and said nothing new each time
        shown = ", ".join(f"#{nid}" for nid in sorted(disabled)[:_DISABLED_IDS_SHOWN])
        more = len(disabled) - _DISABLED_IDS_SHOWN
        findings.append(
            _finding(
                "info",
                "node-disabled",
                f"{len(disabled)} node(s) muted/bypassed - skipped at run time and "
                f"not validated: {shown}"
                + (f" (+{more} more)" if more > 0 else "")
                + ". set_mode 0 re-enables one",
                node_ids=sorted(disabled),
            )
        )
    return findings
