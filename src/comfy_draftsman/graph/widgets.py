"""Mapping between UI-format widgets_values arrays and named inputs.

ComfyUI's UI format stores widget values as a positional array whose order is the
node schema's input declaration order (required section first, then optional),
counting only *widget* inputs (primitives / combos), and inserting synthetic
slots the frontend adds:

- ``control_after_generate`` (e.g. 'randomize'/'fixed') right after any input
  whose schema options set ``control_after_generate: true``
- an upload-button slot after inputs with ``image_upload: true``

Usage: to set a KSampler's seed to randomize:
``set_widget(node_id, "seed__control_after_generate", "randomize")``
Connection-typed inputs (MODEL, CLIP, LATENT, ...) never consume a slot, but
widget inputs that have been *converted to inputs* (connected) still do.

V3 dynamic combos (``COMFY_DYNAMICCOMBO_V3``) are combo widgets whose selected
key reveals a set of conditional sub-widgets. The frontend serializes the main
key followed immediately by the selected option's sub-widget values, all flat in
the same positional ``widgets_values`` array; the /prompt API keys the
sub-widgets with a dotted path (e.g. ``output.normalization``). Because which
sub-widgets are present depends on the selected key, slot computation is
value-aware: pass the node's ``widgets_values`` so the right option expands.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# V3 dynamic combo: a combo whose selected key ("options"[i].key) reveals that
# option's conditional sub-widgets (options[i].inputs). Serialized flat in
# widgets_values; keyed with a dotted path in the API.
DYNAMIC_COMBO_TYPE = "COMFY_DYNAMICCOMBO_V3"

CONTROL_SUFFIX = "__control_after_generate"
UPLOAD_SUFFIX = "__upload"
SYNTHETIC_SUFFIXES = (CONTROL_SUFFIX, UPLOAD_SUFFIX)

# key_for(prefix, position) -> selected option key for a dynamic combo whose
# main widget sits at the given flat position / dotted prefix (or None -> use
# the schema default option).
KeyResolver = Callable[[str, int], Any]


def _iter_schema_inputs(schema: dict[str, Any]):
    """Yield (name, spec) over required then optional inputs, in declaration order."""
    inputs = schema.get("input", {})
    for section in ("required", "optional"):
        yield from inputs.get(section, {}).items()


def _opts(spec: Any) -> dict[str, Any]:
    if isinstance(spec, list | tuple) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1]
    return {}


def is_dynamic_combo(spec: Any) -> bool:
    """True if this input spec is a V3 dynamic combo (COMFY_DYNAMICCOMBO_V3)."""
    return isinstance(spec, list | tuple) and bool(spec) and spec[0] == DYNAMIC_COMBO_TYPE


def is_widget_input(spec: Any) -> bool:
    """True if this input spec renders as a widget (consumes a widgets_values slot)."""
    if not isinstance(spec, list | tuple) or not spec:
        return False
    kind = spec[0]
    opts = _opts(spec)
    if opts.get("forceInput"):
        return False
    if isinstance(kind, list):  # legacy COMBO: list of choices
        return True
    if kind in ("COMBO", DYNAMIC_COMBO_TYPE):  # V3-style COMBO / dynamic combo
        return True
    return kind in PRIMITIVE_TYPES


# --- dynamic combo helpers ---------------------------------------------------


def dynamic_options(spec: Any) -> list[dict[str, Any]]:
    """The options list of a dynamic combo (each {'key', 'inputs'})."""
    return _opts(spec).get("options", []) or []


def dynamic_default_key(spec: Any) -> Any:
    """The default selected key of a dynamic combo: explicit 'default' or the
    first option's key (which is how ComfyUI seeds a freshly created node)."""
    opts = _opts(spec)
    if "default" in opts:
        return opts["default"]
    options = dynamic_options(spec)
    return options[0].get("key") if options else None


def _dynamic_option_for(spec: Any, key: Any) -> dict[str, Any] | None:
    for option in dynamic_options(spec):
        if option.get("key") == key:
            return option
    options = dynamic_options(spec)
    return options[0] if options else None  # unknown key -> default option


def _dynamic_sub_inputs(option: dict[str, Any] | None) -> Iterator[tuple[str, Any]]:
    """(name, spec) pairs for an option's conditional sub-widgets, in order."""
    inputs = (option or {}).get("inputs", {}) or {}
    for section in ("required", "optional"):
        yield from (inputs.get(section) or {}).items()


# --- positional slot / value model -------------------------------------------


def _entries(schema: dict[str, Any], key_for: KeyResolver) -> Iterator[tuple[str, Any]]:
    """Yield (slot_name, spec) over widget slots in positional order.

    ``spec`` is None for synthetic control/upload slots. A dynamic combo expands
    to its main slot (spec = the dynamic spec) immediately followed by the
    selected option's sub-widget slots, whose names are dotted
    (``parent.child``) and may recurse. ``key_for`` picks the selected key for
    each dynamic combo from its dotted prefix / flat position.
    """
    pos = 0

    def walk(name: str, spec: Any) -> Iterator[tuple[str, Any]]:
        nonlocal pos
        if is_dynamic_combo(spec):
            main_pos = pos
            yield name, spec
            pos += 1
            key = key_for(name, main_pos)
            if key is None:
                key = dynamic_default_key(spec)
            option = _dynamic_option_for(spec, key)
            for sub_name, sub_spec in _dynamic_sub_inputs(option):
                if is_widget_input(sub_spec):
                    yield from walk(f"{name}.{sub_name}", sub_spec)
            return
        yield name, spec
        pos += 1
        opts = _opts(spec)
        if opts.get("control_after_generate"):
            yield name + CONTROL_SUFFIX, None
            pos += 1
        if opts.get("image_upload"):
            yield name + UPLOAD_SUFFIX, None
            pos += 1

    for name, spec in _iter_schema_inputs(schema):
        if is_widget_input(spec):
            yield from walk(name, spec)


def _positional_resolver(widgets_values: Any) -> KeyResolver:
    vals = widgets_values if isinstance(widgets_values, list) else []

    def key_for(_prefix: str, position: int) -> Any:
        return vals[position] if position < len(vals) else None

    return key_for


def _named_resolver(named: dict[str, Any]) -> KeyResolver:
    def key_for(prefix: str, _position: int) -> Any:
        return named.get(prefix)

    return key_for


def _schema(class_type: str, object_info: dict[str, Any]) -> dict[str, Any]:
    schema = object_info.get(class_type)
    if schema is None:
        raise ValueError(f"unknown node class: {class_type}")
    return schema


def widget_slot_names(
    class_type: str, object_info: dict[str, Any], widgets_values: Any = None
) -> list[str]:
    """Ordered widgets_values slot names for a node, including synthetic slots.

    Dynamic combos expand per ``widgets_values`` (the selected key picks which
    sub-widgets appear); without it, each dynamic combo expands to its default
    option - matching a freshly created node.
    """
    schema = _schema(class_type, object_info)
    return [name for name, _ in _entries(schema, _positional_resolver(widgets_values))]


def _default_for(spec: Any) -> Any:
    kind = spec[0]
    opts = _opts(spec)
    if is_dynamic_combo(spec):
        return dynamic_default_key(spec)
    if "default" in opts:
        return opts["default"]
    if isinstance(kind, list):
        return kind[0] if kind else None
    if kind == "COMBO":
        options = opts.get("options", [])
        return options[0] if options else None
    if kind == "INT":
        return 0
    if kind == "FLOAT":
        return 0.0
    if kind == "BOOLEAN":
        return False
    return ""


def widget_defaults(
    class_type: str, object_info: dict[str, Any], widgets_values: Any = None
) -> list[Any]:
    """Default widgets_values array. Dynamic combos expand per ``widgets_values``
    (or their default option when it is absent)."""
    schema = _schema(class_type, object_info)
    values: list[Any] = []
    for name, spec in _entries(schema, _positional_resolver(widgets_values)):
        if spec is None:  # synthetic slot
            values.append("fixed" if name.endswith(CONTROL_SUFFIX) else "image")
        else:
            values.append(_default_for(spec))
    return values


def widget_specs(
    class_type: str, object_info: dict[str, Any], widgets_values: Any = None
) -> dict[str, Any]:
    """{slot_name: spec} for the real (non-synthetic) widget slots, dynamic
    combos expanded per the current selection. Used to validate values,
    including dotted sub-widgets of the selected option."""
    schema = _schema(class_type, object_info)
    return {
        name: spec
        for name, spec in _entries(schema, _positional_resolver(widgets_values))
        if spec is not None
    }


def all_slot_names(class_type: str, object_info: dict[str, Any]) -> list[str]:
    """Union of every widget slot across all dynamic-combo selections - for
    lenient name pre-checks, since a sub-widget may belong to an option that is
    not currently selected (select its parent combo first, then set it)."""
    schema = _schema(class_type, object_info)
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            names.append(name)

    def walk(name: str, spec: Any) -> None:
        if is_dynamic_combo(spec):
            add(name)
            for option in dynamic_options(spec):
                for sub_name, sub_spec in _dynamic_sub_inputs(option):
                    if is_widget_input(sub_spec):
                        walk(f"{name}.{sub_name}", sub_spec)
            return
        add(name)
        opts = _opts(spec)
        if opts.get("control_after_generate"):
            add(name + CONTROL_SUFFIX)
        if opts.get("image_upload"):
            add(name + UPLOAD_SUFFIX)

    for name, spec in _iter_schema_inputs(schema):
        if is_widget_input(spec):
            walk(name, spec)
    return names


def widgets_to_named(
    class_type: str, widgets_values: list[Any], object_info: dict[str, Any]
) -> dict[str, Any]:
    """Map a positional widgets_values array to {input_name: value}.

    Synthetic slots are included under their suffixed names; dynamic-combo
    sub-widgets under dotted names. A short or long array is tolerated (custom
    frontend versions drift): missing slots are omitted, extras ignored.
    """
    if isinstance(widgets_values, dict):  # some nodes serialize as dict already
        return dict(widgets_values)
    named: dict[str, Any] = {}
    slots = widget_slot_names(class_type, object_info, widgets_values)
    for slot, value in zip(slots, widgets_values or [], strict=False):
        named[slot] = value
    return named


def named_for_api(
    class_type: str, widgets_values: Any, object_info: dict[str, Any]
) -> dict[str, Any]:
    """Named widget inputs for the /prompt API. Like widgets_to_named, but every
    dynamic-combo slot (the main key and its selected option's dotted
    sub-widgets) is guaranteed present - defaulted when an older save dropped
    it - so a graph containing V3 combos stays runnable. Regular optional
    widgets are left as-is (dynamic nodes legitimately omit unused ones)."""
    named = widgets_to_named(class_type, widgets_values, object_info)
    if isinstance(widgets_values, dict):
        return named
    schema = _schema(class_type, object_info)
    for name, spec in _entries(schema, _positional_resolver(widgets_values)):
        if spec is None or name in named:
            continue
        if is_dynamic_combo(spec) or "." in name:
            named[name] = _default_for(spec)
    return named


def named_to_widgets(
    class_type: str, named: dict[str, Any], object_info: dict[str, Any]
) -> list[Any]:
    """Build a positional widgets_values array from named values, defaults
    filling gaps. Dynamic-combo expansion follows the selected keys present in
    ``named`` (dotted sub-widgets), so an API-format prompt round-trips to the
    UI array the frontend expects."""
    schema = _schema(class_type, object_info)
    values: list[Any] = []
    for name, spec in _entries(schema, _named_resolver(named)):
        if name in named:
            values.append(named[name])
        elif spec is None:
            values.append("fixed" if name.endswith(CONTROL_SUFFIX) else "image")
        else:
            values.append(_default_for(spec))
    return values
