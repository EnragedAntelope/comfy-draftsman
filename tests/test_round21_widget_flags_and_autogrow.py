"""Round 21: believe ComfyUI's own widget flags, and author autogrow sockets.

Closing round 20's two open TODOs, both of which turned out to rest on a wrong
premise:

- "There is no flag in /object_info that distinguishes a widget from a socket."
  There are two - `socketless` and `widgetType`, serialized by V3's
  `WidgetInput.as_dict`. Without them, 26 classes on a stock instance had a
  ComfyUI-declared widget treated as a required connection socket:
  `ColorToRGBInt` (whose only parameter is a socketless COLOR) validated as a
  blocking `unconnected-input` for a graph that was fine, and `TextOverlay` lost
  its `color` value and shifted every later widget up a slot.

- "Autogrow is tolerated but not authorable; core nodes declare their full slot
  range up front, so connecting works." They do not - `/object_info` declares
  only the marker plus a template, so the real slot names were undiscoverable
  and unconnectable, and the marker itself was reported as an unconnected
  required input (56 required markers on a stock instance).

The autogrow API key is DOTTED (`images.image0`) while the canvas shows the bare
name - confirmed from ComfyUI's `parse_class_inputs`/`finalize_prefix` and
independently from the frontend bundle, which builds each slot as
``{name: `${marker}.${slot}`, display_name: slot}``. Emitting the bare name does
not error; the backend just never matches it and the input goes missing.
"""

import json

import pytest

from comfy_draftsman.comfy.catalog import node_summary
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import PRIMITIVE_TYPE, REROUTE_TYPE, Workflow
from comfy_draftsman.graph.validate import validate

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"

# Real shapes from a stock ComfyUI 0.29 instance.
COLOR_NODE = {
    "input": {"required": {"color": ["COLOR", {"socketless": True, "default": "#ffffff"}]}},
    "output": ["INT"],
    "name": "ColorToRGBInt",
    "python_module": "comfy_extras.nodes_color",
}
# TextOverlay's `color` sits BETWEEN font_size and position - the slot whose
# omission shifted every later widget.
TEXT_OVERLAY = {
    "input": {
        "required": {
            "images": ["IMAGE", {}],
            "text": ["STRING", {"multiline": True}],
            "font_size": ["FLOAT", {"default": 5.0}],
            "color": ["COLOR", {"socketless": True, "default": "#ffffff"}],
            "position": ["COMBO", {"options": ["top", "bottom"]}],
        }
    },
    "output": ["IMAGE"],
    "name": "TextOverlay",
}
# A union type whose only checkable kind is its widgetType.
UNION_WIDGET = {
    "input": {
        "required": {"frame_rate": ["FLOAT,INT", {"widgetType": "FLOAT", "default": 30.0,
                                                  "min": 1.0, "max": 120.0}]}
    },
    "output": ["LATENT"],
    "name": "LTXVEmptyLatentAudio",
}
# socketless AND forceInput: the node author overriding a type whose class
# defaults to socketless. forceInput wins - it is the explicit "draw a socket".
FORCED_SOCKET = {
    "input": {"required": {"bboxes": ["BOUNDING_BOX", {"socketless": True, "forceInput": True}]}},
    "output": ["IMAGE"],
    "name": "DrawBBoxes",
}
# The deliberately-blocked case: a bespoke JS-state type with NO flags at all.
LORA_JS = {
    "input": {"required": {"text": ["AUTOCOMPLETE_TEXT_LORAS", {}], "model": ["MODEL", {}]}},
    "output": ["MODEL"],
    "name": "Lora Loader (LoraManager)",
    "python_module": "custom_nodes.ComfyUI-Lora-Manager",
}
BATCH_IMAGES = {
    "input": {
        "required": {
            "images": [
                "COMFY_AUTOGROW_V3",
                {"template": {"input": {"required": {"image": ["IMAGE", {}]}},
                              "prefix": "image", "min": 1, "max": 50}},
            ]
        }
    },
    "output": ["IMAGE"],
    "name": "BatchImagesNode",
    "python_module": "comfy_extras.nodes_post_processing",
}
# TemplateNames form, and min=0 (nothing is mandatory).
NAMED_AUTOGROW = {
    "input": {
        "optional": {
            "subjects": [
                "COMFY_AUTOGROW_V3",
                {"template": {"input": {"optional": {"subject": ["STRING", {}]}},
                              "names": ["alpha", "beta", "gamma"], "min": 0}},
            ]
        }
    },
    "output": ["STRING"],
    "name": "NamedAutogrow",
}


@pytest.fixture(scope="module")
def base_oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def oi(base_oi):
    return {
        **base_oi,
        "ColorToRGBInt": COLOR_NODE,
        "TextOverlay": TEXT_OVERLAY,
        "LTXVEmptyLatentAudio": UNION_WIDGET,
        "DrawBBoxes": FORCED_SOCKET,
        "Lora Loader (LoraManager)": LORA_JS,
        "BatchImagesNode": BATCH_IMAGES,
        "NamedAutogrow": NAMED_AUTOGROW,
    }


def _errors(findings, code=None, node_id=None):
    return [
        f for f in findings
        if f["level"] == "error"
        and (code is None or f["code"] == code)
        and (node_id is None or f.get("node_id") == node_id)
    ]


# --- ComfyUI's own widget flags ----------------------------------------------


def test_socketless_input_is_a_widget():
    assert w.is_widget_input(["COLOR", {"socketless": True}])


def test_widget_type_input_is_a_widget():
    assert w.is_widget_input(["FLOAT,INT", {"widgetType": "FLOAT"}])


def test_force_input_beats_socketless():
    """7 inputs on a stock instance declare both; forceInput is the node author's
    explicit override of a type whose class defaults to socketless."""
    assert not w.is_widget_input(["BOUNDING_BOX", {"socketless": True, "forceInput": True}])


def test_socketless_false_is_not_a_widget():
    """`socketless: false` is a real serialized value, not an absent key."""
    assert not w.is_widget_input(["BOUNDING_BOXES", {"socketless": False}])


def test_socketless_widget_is_not_a_phantom_socket(oi):
    """ColorToRGBInt's only parameter is a socketless COLOR. It used to become a
    connection socket with no widget value - so the node had nothing to set, and
    validate refused the graph over an input that can never be connected."""
    wf = Workflow.new()
    node = wf.add_node("ColorToRGBInt", object_info=oi)
    assert [s.name for s in node.inputs] == []
    assert node.widgets_values == ["#ffffff"]
    assert _errors(validate(wf, oi), "unconnected-input", node.id) == []


def test_socketless_widget_keeps_later_widgets_aligned(oi):
    """`color` sits between font_size and position; dropping it shifted every
    later value up a slot."""
    wf = Workflow.new()
    node = wf.add_node("TextOverlay", object_info=oi)
    named = w.widgets_to_named("TextOverlay", node.widgets_values, oi, {"images"})
    assert named["font_size"] == 5.0
    assert named["color"] == "#ffffff"
    assert named["position"] == "top"


def test_widget_type_drives_the_value_check(oi):
    """"FLOAT,INT" names no single checkable kind; its widgetType does."""
    from comfy_draftsman.graph.validate import check_widget_value

    assert check_widget_value("LTXVEmptyLatentAudio", "frame_rate", 24.0, oi) is None
    problem = check_widget_value("LTXVEmptyLatentAudio", "frame_rate", "fast", oi)
    assert problem and "number" in problem


def test_unflagged_js_widget_still_blocks(oi):
    """The carve-out must not unblock the case round 13 deliberately stopped:
    AUTOCOMPLETE_TEXT_LORAS carries no flags on a stock instance, so it is still
    invisible to the schema-level check."""
    assert not w.is_widget_input(["AUTOCOMPLETE_TEXT_LORAS", {}])
    wf = Workflow.new()
    node = wf.add_node("Lora Loader (LoraManager)", object_info=oi)
    assert _errors(validate(wf, oi), node_id=node.id)


# --- autogrow: the template ---------------------------------------------------


def test_prefix_template_expands():
    spec = BATCH_IMAGES["input"]["required"]["images"]
    t = w.autogrow_template(spec)
    assert t["names"][:3] == ["image0", "image1", "image2"]
    assert len(t["names"]) == 50
    assert t["min"] == 1
    assert t["item_spec"][0] == "IMAGE"


def test_names_template_expands():
    t = w.autogrow_template(NAMED_AUTOGROW["input"]["optional"]["subjects"])
    assert t["names"] == ["alpha", "beta", "gamma"]
    assert t["min"] == 0


def test_prefix_is_not_the_marker_name():
    """`images` -> `image0`: the prefix must be read from the template, never
    derived from the marker."""
    t = w.autogrow_template(BATCH_IMAGES["input"]["required"]["images"])
    assert not t["names"][0].startswith("images")


def test_non_autogrow_spec_is_none():
    assert w.autogrow_template(["IMAGE", {}]) is None
    assert w.autogrow_template(["COMFY_MATCHTYPE_V3", {}]) is None


def test_api_key_is_dotted(oi):
    assert w.autogrow_api_key(BATCH_IMAGES, "image0") == "images.image0"
    assert w.autogrow_api_key(BATCH_IMAGES, "images.image0") == "images.image0"
    assert w.autogrow_api_key(BATCH_IMAGES, "not_a_slot") is None


# --- autogrow: authoring ------------------------------------------------------


def test_marker_is_never_an_unconnected_socket(oi):
    """The marker names a container. Asking whether it is connected is a category
    error, and it blocked 56 required markers on a stock instance."""
    wf = Workflow.new()
    wf.add_node("BatchImagesNode", object_info=oi)
    assert not [
        f for f in validate(wf, oi)
        if f["code"] == "unconnected-input" and f.get("input") == "images"
    ]


def test_add_node_materializes_the_mandatory_slots(oi):
    wf = Workflow.new()
    node = wf.add_node("BatchImagesNode", object_info=oi)
    assert [s.name for s in node.inputs] == ["images.image0"]


def test_min_zero_creates_no_slots_and_never_complains(oi):
    wf = Workflow.new()
    node = wf.add_node("NamedAutogrow", object_info=oi)
    assert [s.name for s in node.inputs] == []
    assert _errors(validate(wf, oi), "autogrow-underfilled", node.id) == []


def test_underfilled_reports_the_real_requirement(oi):
    wf = Workflow.new()
    node = wf.add_node("BatchImagesNode", object_info=oi)
    bad = _errors(validate(wf, oi), "autogrow-underfilled", node.id)
    assert len(bad) == 1
    assert "images.image0" in bad[0]["message"]
    assert "gaps are allowed" in bad[0]["message"].lower()


def test_connect_accepts_both_spellings_without_duplicating(oi):
    wf = Workflow.new()
    batch = wf.add_node("BatchImagesNode", object_info=oi)
    a = wf.add_node("LoadImage", object_info=oi)
    b = wf.add_node("LoadImage", object_info=oi)
    wf.connect(a.id, 0, batch.id, "image0", oi)  # bare, onto the existing socket
    wf.connect(b.id, 0, batch.id, "images.image2", oi)  # dotted, a new one
    assert [s.name for s in batch.inputs] == ["images.image0", "images.image2"]


def test_gaps_are_legal(oi):
    """The backend collects whichever names the prompt carries, so image0+image2
    runs as written - nothing needs renumbering."""
    wf = Workflow.new()
    batch = wf.add_node("BatchImagesNode", object_info=oi)
    a = wf.add_node("LoadImage", object_info=oi)
    b = wf.add_node("LoadImage", object_info=oi)
    wf.connect(a.id, 0, batch.id, "images.image0", oi)
    wf.connect(b.id, 0, batch.id, "images.image2", oi)
    assert _errors(validate(wf, oi), node_id=batch.id) == []
    assert set(wf.to_api(oi)[str(batch.id)]["inputs"]) == {"images.image0", "images.image2"}


def test_synthesized_slot_is_still_type_checked(oi):
    wf = Workflow.new()
    batch = wf.add_node("BatchImagesNode", object_info=oi)
    enc = wf.add_node("CLIPTextEncode", object_info=oi)
    with pytest.raises(ValueError, match="type mismatch"):
        wf.connect(enc.id, 0, batch.id, "images.image3", oi)


def test_imported_bare_names_are_preserved_but_emit_dotted(oi):
    """Renaming an imported socket to "canonicalize" it would silently rewrite the
    user's file; normalizing only at to_api keeps both spellings runnable."""
    wf = Workflow.new()
    batch = wf.add_node("BatchImagesNode", object_info=oi)
    src = wf.add_node("LoadImage", object_info=oi)
    wf.connect(src.id, 0, batch.id, "images.image0", oi)
    doc = wf.to_ui()
    for n in doc["nodes"]:
        if n["type"] == "BatchImagesNode":
            for slot in n["inputs"]:
                slot["name"] = slot["name"].split(".")[-1]
    reloaded = Workflow.from_ui(doc)
    node = next(n for n in reloaded.nodes.values() if n.type == "BatchImagesNode")
    assert [s.name for s in node.inputs] == ["image0"]  # untouched
    assert _errors(validate(reloaded, oi), node_id=node.id) == []
    assert "images.image0" in reloaded.to_api(oi)[str(node.id)]["inputs"]


def test_lint_agrees_with_validate_on_the_marker(oi):
    """lint contradicting validate is pure noise, and save_workflow nags about an
    unclean lint."""
    wf = Workflow.new()
    batch = wf.add_node("BatchImagesNode", object_info=oi)
    src = wf.add_node("LoadImage", object_info=oi)
    wf.connect(src.id, 0, batch.id, "images.image0", oi)
    assert not [
        f for f in lint(wf, oi)
        if f["code"] == "unconnected-input" and f.get("node_id") == batch.id
    ]


def test_get_node_info_makes_the_slots_discoverable(oi):
    """/object_info names only the marker, so without this the real slot names
    cannot be found and connect has nothing to aim at."""
    entry = next(
        i for i in node_summary(oi, "BatchImagesNode")["inputs"] if i["name"] == "images"
    )
    assert entry["autogrow"] is True
    assert entry["connect_to"][0] == "images.image0"
    assert entry["slot_count"] == 50
    assert entry["min_connected"] == 1
    assert "never to" in entry["hint"]


def test_get_node_info_caps_the_slot_names(oi):
    """A marker can declare 50 names in an obvious arithmetic pattern; listing
    them all is repetition this server treats as a bug."""
    from comfy_draftsman.comfy import catalog

    entry = next(
        i for i in node_summary(oi, "BatchImagesNode")["inputs"] if i["name"] == "images"
    )
    assert len(entry["connect_to"]) == catalog._AUTOGROW_NAMES_SHOWN


# --- TODO #4: a primitive chain is unrepresentable, not unhandled -------------


def test_primitive_cannot_be_chained(oi):
    """`primitive -> Reroute -> primitive` was carried as an open TODO. A
    PrimitiveNode is a pure source with NO inputs, so the pattern cannot be
    built at all - connect refuses by name with the available list, which is a
    better outcome than resolving it would have been."""
    wf = Workflow.new()
    first = wf.add_node(PRIMITIVE_TYPE, object_info=oi)
    reroute = wf.add_node(REROUTE_TYPE, object_info=oi)
    second = wf.add_node(PRIMITIVE_TYPE, object_info=oi)
    wf.connect(first.id, 0, reroute.id, "", oi)
    assert second.inputs == []
    with pytest.raises(ValueError, match="has no input"):
        wf.connect(reroute.id, 0, second.id, "value", oi)
