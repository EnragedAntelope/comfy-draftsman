"""Mapping between UI-format widgets_values arrays and named inputs.

ComfyUI's UI format stores widget values as a positional array whose order is the
node schema's input declaration order (required section first, then optional),
counting only *widget* inputs (primitives / combos), and inserting synthetic
slots the frontend adds:

- ``control_after_generate`` (e.g. 'randomize'/'fixed') right after any input
  whose schema options set ``control_after_generate: true`` — and ALSO after any
  INT input literally named ``seed``/``noise_seed`` even without the flag: the
  frontend adds the control widget by name for legacy (V1) nodes, so their
  serialized widgets_values carry the extra slot despite the schema
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

import random
from collections.abc import Callable, Iterator
from typing import Any

PRIMITIVE_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# ComfyUI seeds are unsigned 64-bit; the frontend randomizes within this range.
MAX_SEED = 0xFFFFFFFFFFFFFFFF

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


def is_v3_meta_type(kind: Any) -> bool:
    """True for ComfyUI's own V3 *meta* io types - schema markers, never a
    concrete value type and never a widget.

    `comfy_api.latest._io` declares five (`COMFY_MATCHTYPE_V3`,
    `COMFY_AUTOGROW_V3`, `COMFY_DYNAMICSLOT_V3`, `COMFY_DYNAMICCOMBO_V3`,
    `COMFY_MULTITYPED_V3`). They matter here because a node usually does NOT
    serialize them into its `inputs` socket array - an autogrow node emits
    `value0..valueN` instead of the `values` marker, and an unconnected MatchType
    slot is simply absent - which is precisely the shape `_is_custom_widget` reads
    as "a pack's JS-rendered widget". Counting one as a widget invents a slot that
    doesn't exist and shifts every later `widgets_values` entry up by one.

    Matching the `COMFY_*_V3` shape rather than a fixed set keeps new core meta
    types from silently reintroducing the same corruption.
    """
    return (
        isinstance(kind, str) and kind.startswith("COMFY_") and kind.endswith("_V3")
    )


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


def has_control_slot(name: str, spec: Any) -> bool:
    """True if the frontend appends a control_after_generate widget slot after
    this input: the schema opts ask for it, or (legacy name heuristic, matching
    the frontend) it's an INT literally named seed/noise_seed. An explicit
    ``control_after_generate: false`` opts out."""
    opts = _opts(spec)
    if "control_after_generate" in opts:
        return bool(opts["control_after_generate"])
    leaf = name.rsplit(".", 1)[-1]
    return spec[0] == "INT" and leaf in ("seed", "noise_seed")


def combo_choices(spec: Any) -> list[Any] | None:
    """The option list of any combo flavour (legacy list, V3 COMBO, dynamic
    combo), else None. Dynamic combos answer with their option KEYS, which is
    what the main widget's value must be."""
    if not isinstance(spec, list | tuple) or not spec:
        return None
    kind = spec[0]
    if isinstance(kind, list):
        return list(kind)
    if is_dynamic_combo(spec):
        return [o.get("key") for o in dynamic_options(spec)]
    if kind == "COMBO":
        return list(_opts(spec).get("options") or [])
    return None


def primitive_takes_control(spec: Any) -> bool:
    """Whether a PrimitiveNode mirroring this widget gets a
    ``control_after_generate`` slot.

    The frontend calls ``addValueControlWidget`` only for widget types "number"
    and "combo", so a STRING or BOOLEAN primitive serializes a single value and
    nothing else. Distinct from ``has_control_slot``, which answers the same
    question for a REAL node's own seed widget (a name/flag heuristic)."""
    if not isinstance(spec, list | tuple) or not spec:
        return False
    if combo_choices(spec) is not None:
        return True
    return spec[0] in ("INT", "FLOAT")


def widget_kind(spec: Any) -> Any:
    """The widget this input actually renders as.

    V3's ``widgetType`` overrides the io type for widget purposes - ComfyUI's own
    ``WidgetInput.get_io_type`` does exactly this. It matters when the declared
    type is a union: ``LTXVEmptyLatentAudio.frame_rate`` is ``"FLOAT,INT"`` with
    ``widgetType: "FLOAT"``, and only the latter says how to range-check it."""
    opts = _opts(spec)
    declared = opts.get("widgetType")
    if declared:
        return declared
    return spec[0] if isinstance(spec, list | tuple) and spec else spec


def is_widget_input(spec: Any) -> bool:
    """True if this input spec renders as a widget (consumes a widgets_values slot).

    Primitives, combos, and anything ComfyUI itself flags as widget-rendered -
    all schema-level truth, so this works without instance context (``add_node``
    and fresh-node defaults depend on that).

    **``socketless`` / ``widgetType`` are ComfyUI's own declarations** and are
    believed over the io type. `socketless: true` means the frontend renders the
    input as a widget and never draws a socket for it; `widgetType` names the
    widget that renders a bespoke io type. Without them, 26 classes on a stock
    instance had a widget treated as a required connection socket - `TextOverlay`
    lost its `color` value and shifted every later widget up a slot, and
    `ColorToRGBInt` (whose only parameter is a socketless COLOR) validated as a
    blocking `unconnected-input` error for a graph that was perfectly fine.

    `forceInput` still wins: 7 inputs declare `socketless` *and* `forceInput`
    (a type whose class defaults to socketless, overridden per-node), and
    `forceInput` is the node author's explicit "draw this as a socket".

    A pack that declares neither flag is still invisible here - a bespoke type
    like ``AUTOCOMPLETE_TEXT_LORAS`` (no flags at all on a stock instance) is
    recognized per-instance via ``socket_names`` in the slot walk
    (``_is_custom_widget``), not here."""
    if not isinstance(spec, list | tuple) or not spec:
        return False
    kind = spec[0]
    opts = _opts(spec)
    if opts.get("forceInput"):
        return False
    if opts.get("socketless") or opts.get("widgetType"):
        return True
    if isinstance(kind, list):  # legacy COMBO: list of choices
        return True
    if kind in ("COMBO", DYNAMIC_COMBO_TYPE):  # V3-style COMBO / dynamic combo
        return True
    return kind in PRIMITIVE_TYPES


AUTOGROW_TYPE = "COMFY_AUTOGROW_V3"


def autogrow_template(spec: Any) -> dict[str, Any] | None:
    """The synthesized socket list behind a `COMFY_AUTOGROW_V3` marker input, or
    None if this spec isn't one.

    An autogrow input is a *container*, not a socket: `/object_info` declares only
    the marker (e.g. `BatchImagesNode.images`) plus a `template`, and the real
    sockets are synthesized from it - `prefix`+index (`image0`..`image49`), or an
    explicit `names` list. The frontend synthesizes them as wires are attached;
    the backend re-derives them at execution from whatever the submitted prompt
    actually contains (`Autogrow._expand_schema_for_dynamic`), making the first
    `min` required and the rest optional.

    Two consequences draftsman has to mirror:

    - **The marker is never connectable.** It was being reported as a required
      input that is "not connected" - a blocking error on 56 required markers
      across a stock instance, for graphs that are fine.
    - **Gaps are legal.** The backend walks the name list and collects whichever
      names the prompt carries, so `image0` + `image2` with no `image1` executes
      exactly as written. Nothing needs renumbering.

    Note the prefix is NOT derivable from the marker name (`images` -> `image`),
    so it must be read from the template.

    **The API key is dotted: `{marker}.{slot}` (`images.image0`).** The canvas
    label is the bare slot name, but the prompt document must carry the dotted
    form - confirmed twice over, from ComfyUI's own
    `parse_class_inputs`/`finalize_prefix` (which prefixes the marker id onto
    every expanded name) and from the frontend bundle, which builds each
    autogrow input as ``{name: `${marker}.${slot}`, display_name: slot}``.
    Emitting the bare name would not error - the backend simply would not match
    it, and the node would run with that input silently missing.
    """
    if not isinstance(spec, list | tuple) or not spec or spec[0] != AUTOGROW_TYPE:
        return None
    template = _opts(spec).get("template") or {}
    if "names" in template:
        names = [str(n) for n in template.get("names") or []]
    elif "prefix" in template:
        try:
            maximum = int(template.get("max") or 0)
        except (TypeError, ValueError):
            return None
        names = [f"{template['prefix']}{i}" for i in range(maximum)]
    else:
        return None
    if not names:
        return None
    try:
        minimum = max(0, int(template.get("min") or 0))
    except (TypeError, ValueError):
        minimum = 0
    # the per-slot spec: the template's single declared input, whichever section
    # it sits in ("required" only means `min` of them are mandatory)
    item_spec: Any = ["*", {}]
    for section in ("required", "optional"):
        entries = (template.get("input") or {}).get(section) or {}
        if entries:
            item_spec = next(iter(entries.values()))
            break
    return {"names": names, "min": min(minimum, len(names)), "item_spec": item_spec}


def autogrow_slots(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{marker input name: autogrow_template(...)} for one node schema."""
    found: dict[str, dict[str, Any]] = {}
    for name, spec in _iter_schema_inputs(schema):
        template = autogrow_template(spec)
        if template is not None:
            found[name] = template
    return found


def autogrow_resolve(
    schema: dict[str, Any], input_name: str
) -> tuple[str, Any] | None:
    """``(dotted API key, per-slot spec)`` for an autogrow socket, else None.

    Accepts either spelling - the bare canvas name (``image0``) or the dotted API
    name (``images.image0``) - and always answers with the dotted one. Both are
    accepted deliberately: a caller naturally writes what `get_node_info` shows on
    the canvas, while an imported workflow carries whatever the frontend wrote,
    and renaming an imported socket to "canonicalize" it would silently rewrite
    the user's file. Normalizing at the one place that matters (`to_api`) keeps
    both spellings runnable and neither rewritten.
    """
    for marker, template in autogrow_slots(schema).items():
        for slot in template["names"]:
            if input_name in (slot, f"{marker}.{slot}"):
                return f"{marker}.{slot}", template["item_spec"]
    return None


def autogrow_api_key(schema: dict[str, Any], input_name: str) -> str | None:
    """The dotted key `to_api` must emit for this socket, or None if it isn't an
    autogrow slot (in which case its own name is already the key)."""
    resolved = autogrow_resolve(schema, input_name)
    return resolved[0] if resolved is not None else None


def _is_custom_widget(name: str, spec: Any, socket_names: set[str] | None) -> bool:
    """True if this input is a custom widget-like type the frontend JS renders as
    a widget rather than a socket. Recognized only with instance context: a
    custom-typed input that the node did NOT serialize as a socket
    (``name not in socket_names``) can only be a JS widget (the frontend always
    emits real connection sockets in the node's ``inputs`` array). Without
    ``socket_names`` (schema/fresh context) we can't tell, so return False.

    ComfyUI's own V3 meta types are the exception to "absent socket => widget":
    they are schema markers the frontend expands into other slots, so they are
    routinely absent while being no widget at all (see ``is_v3_meta_type``)."""
    if socket_names is None or is_widget_input(spec):
        return False
    if not isinstance(spec, list | tuple) or not spec:
        return False
    kind = spec[0]
    if _opts(spec).get("forceInput") or is_v3_meta_type(kind):
        return False
    return isinstance(kind, str) and kind != "*" and name not in socket_names


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


def _entries(
    schema: dict[str, Any], key_for: KeyResolver, socket_names: set[str] | None = None
) -> Iterator[tuple[str, Any]]:
    """Yield (slot_name, spec) over widget slots in positional order.

    ``spec`` is None for synthetic control/upload slots. A dynamic combo expands
    to its main slot (spec = the dynamic spec) immediately followed by the
    selected option's sub-widget slots, whose names are dotted
    (``parent.child``) and may recurse. ``key_for`` picks the selected key for
    each dynamic combo from its dotted prefix / flat position. ``socket_names``
    (the node instance's declared input sockets) lets custom widget-like inputs
    be counted - see ``_is_custom_widget``.
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
        if has_control_slot(name, spec):
            yield name + CONTROL_SUFFIX, None
            pos += 1
        if opts.get("image_upload"):
            yield name + UPLOAD_SUFFIX, None
            pos += 1

    for name, spec in _iter_schema_inputs(schema):
        if is_widget_input(spec) or _is_custom_widget(name, spec, socket_names):
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
    class_type: str,
    object_info: dict[str, Any],
    widgets_values: Any = None,
    socket_names: set[str] | None = None,
) -> list[str]:
    """Ordered widgets_values slot names for a node, including synthetic slots.

    Dynamic combos expand per ``widgets_values`` (the selected key picks which
    sub-widgets appear); without it, each dynamic combo expands to its default
    option - matching a freshly created node. Pass ``socket_names`` (the node's
    declared input sockets) so custom JS-widget inputs are counted.
    """
    schema = _schema(class_type, object_info)
    return [
        name
        for name, _ in _entries(schema, _positional_resolver(widgets_values), socket_names)
    ]


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
    class_type: str,
    object_info: dict[str, Any],
    widgets_values: Any = None,
    socket_names: set[str] | None = None,
) -> dict[str, Any]:
    """{slot_name: spec} for the real (non-synthetic) widget slots, dynamic
    combos expanded per the current selection. Used to validate values,
    including dotted sub-widgets of the selected option."""
    schema = _schema(class_type, object_info)
    return {
        name: spec
        for name, spec in _entries(schema, _positional_resolver(widgets_values), socket_names)
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
        if has_control_slot(name, spec):
            add(name + CONTROL_SUFFIX)
        if opts.get("image_upload"):
            add(name + UPLOAD_SUFFIX)

    for name, spec in _iter_schema_inputs(schema):
        if is_widget_input(spec):
            walk(name, spec)
    return names


def widgets_to_named(
    class_type: str,
    widgets_values: list[Any],
    object_info: dict[str, Any],
    socket_names: set[str] | None = None,
) -> dict[str, Any]:
    """Map a positional widgets_values array to {input_name: value}.

    Synthetic slots are included under their suffixed names; dynamic-combo
    sub-widgets under dotted names. A short or long array is tolerated (custom
    frontend versions drift): missing slots are omitted, extras ignored.
    ``socket_names`` (the node's declared sockets) lets custom JS-widget inputs
    be mapped too.
    """
    if isinstance(widgets_values, dict):  # some nodes serialize as dict already
        return dict(widgets_values)
    named: dict[str, Any] = {}
    slots = widget_slot_names(class_type, object_info, widgets_values, socket_names)
    for slot, value in zip(slots, widgets_values or [], strict=False):
        named[slot] = value
    return named


def named_for_api(
    class_type: str,
    widgets_values: Any,
    object_info: dict[str, Any],
    socket_names: set[str] | None = None,
) -> dict[str, Any]:
    """Named widget inputs for the /prompt API. Like widgets_to_named, but every
    dynamic-combo slot (the main key and its selected option's dotted
    sub-widgets) is guaranteed present - defaulted when an older save dropped
    it - so a graph containing V3 combos stays runnable. Regular optional
    widgets are left as-is (dynamic nodes legitimately omit unused ones).
    ``socket_names`` lets custom JS-widget inputs (e.g. LoraManager's autocomplete
    box) survive UI->API conversion instead of being dropped."""
    named = widgets_to_named(class_type, widgets_values, object_info, socket_names)
    if isinstance(widgets_values, dict):
        return named
    schema = _schema(class_type, object_info)
    for name, spec in _entries(schema, _positional_resolver(widgets_values), socket_names):
        if spec is None or name in named:
            continue
        if is_dynamic_combo(spec) or "." in name:
            named[name] = _default_for(spec)
    return named


def roll_seed_controls(
    class_type: str,
    widgets_values: Any,
    object_info: dict[str, Any],
    rng: random.Random | None = None,
    socket_names: set[str] | None = None,
) -> tuple[Any, bool]:
    """Mirror the frontend's ``control_after_generate`` re-roll for one node.

    The backend /prompt endpoint runs whatever literal seed is submitted; only
    the browser bumps/randomizes the value between queues. So a headless run
    leaves seeds fixed unless we roll them ourselves. For each seed widget whose
    adjacent ``__control_after_generate`` slot is randomize/increment/decrement,
    update the seed in place: randomize -> fresh int in [min, max]; increment/
    decrement -> +/- step, clamped. Returns (new_values, changed)."""
    if not isinstance(widgets_values, list):
        return widgets_values, False
    slots = widget_slot_names(class_type, object_info, widgets_values, socket_names)
    specs = widget_specs(class_type, object_info, widgets_values, socket_names)
    index = {name: i for i, name in enumerate(slots)}
    values = list(widgets_values)
    roller = rng or random
    changed = False
    for ctrl_name in slots:
        if not ctrl_name.endswith(CONTROL_SUFFIX):
            continue
        seed_name = ctrl_name[: -len(CONTROL_SUFFIX)]
        ci, si = index.get(ctrl_name), index.get(seed_name)
        if ci is None or si is None or ci >= len(values) or si >= len(values):
            continue
        mode = values[ci]
        if mode not in ("randomize", "increment", "decrement"):
            continue
        current = values[si]
        if not isinstance(current, int) or isinstance(current, bool):
            continue
        opts = _opts(specs.get(seed_name))
        low = int(opts.get("min") or 0)
        high = min(int(opts.get("max") or MAX_SEED), MAX_SEED)
        step = int(opts.get("step") or 1) or 1
        if mode == "randomize":
            new = roller.randint(low, high)
        elif mode == "increment":
            new = current + step
        else:
            new = current - step
        new = max(low, min(high, new))
        if new != current:
            values[si] = new
            changed = True
    return values, changed


def named_to_widgets(
    class_type: str,
    named: dict[str, Any],
    object_info: dict[str, Any],
    socket_names: set[str] | None = None,
) -> list[Any]:
    """Build a positional widgets_values array from named values, defaults
    filling gaps. Dynamic-combo expansion follows the selected keys present in
    ``named`` (dotted sub-widgets), so an API-format prompt round-trips to the
    UI array the frontend expects.

    ``socket_names`` (the node instance's declared sockets) must be passed
    whenever this rebuilds an EXISTING node's array: without it a custom
    JS-widget input is invisible to the slot walk, so its value is dropped and
    every later value shifts up a slot. Omit it only for schema-only contexts
    (a fresh node with no instance to read sockets from).
    """
    schema = _schema(class_type, object_info)
    values: list[Any] = []
    for name, spec in _entries(schema, _named_resolver(named), socket_names):
        if name in named:
            values.append(named[name])
        elif spec is None:
            values.append("fixed" if name.endswith(CONTROL_SUFFIX) else "image")
        else:
            values.append(_default_for(spec))
    return values
