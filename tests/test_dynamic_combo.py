"""V3 dynamic combos (COMFY_DYNAMICCOMBO_V3): schema parsing, value-aware slot
expansion, dotted-key round-trips, set_widget, and validation.

Fixtures DA3Inference / DA3Render / SaveImageAdvanced are real schemas extracted
from ComfyUI 0.26.0's /object_info. Their native serialization (verified against
the bundled depth-anything-3 template) is:
    DA3Render      widgets_values = ["depth", "v2_style", False]
    DA3Inference   widgets_values = [504, "upper_bound_resize", "mono"]
    SaveImageAdvanced (fresh) = ["ComfyUI", "png", "8-bit", "sRGB"]
i.e. the main combo key followed, flat, by the selected option's sub-widgets.
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def oi():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


# --- schema recognition ------------------------------------------------------


def test_dynamic_combo_is_a_widget(oi):
    spec = oi["DA3Render"]["input"]["required"]["output"]
    assert w.is_dynamic_combo(spec)
    assert w.is_widget_input(spec)  # the core fix: no longer a link-only input


def test_dynamic_default_key_is_first_option_when_no_default(oi):
    spec = oi["DA3Inference"]["input"]["required"]["mode"]
    assert w.dynamic_default_key(spec) == "mono"


# --- value-aware slot expansion ---------------------------------------------


def test_fresh_defaults_match_native(oi):
    assert w.widget_defaults("DA3Render", oi) == ["depth", "v2_style", False]
    assert w.widget_defaults("DA3Inference", oi) == [504, "upper_bound_resize", "mono"]
    assert w.widget_defaults("SaveImageAdvanced", oi) == ["ComfyUI", "png", "8-bit", "sRGB"]


def test_slots_depend_on_selected_option(oi):
    mono = w.widget_slot_names("DA3Inference", oi, [504, "upper_bound_resize", "mono"])
    multi = w.widget_slot_names("DA3Inference", oi, [504, "upper_bound_resize", "multiview"])
    assert mono == ["resolution", "resize_method", "mode"]
    assert multi == [
        "resolution",
        "resize_method",
        "mode",
        "mode.ref_view_strategy",
        "mode.pose_method",
    ]


def test_widgets_to_named_uses_dotted_keys(oi):
    named = w.widgets_to_named("DA3Render", ["depth", "v2_style", False], oi)
    assert named == {
        "output": "depth",
        "output.normalization": "v2_style",
        "output.apply_sky_clip": False,
    }


def test_all_slot_names_is_union_across_options(oi):
    names = w.all_slot_names("DA3Render", oi)
    # depth/depth_colored subs plus the sky_mask/confidence 'colored' sub
    assert "output.normalization" in names
    assert "output.apply_sky_clip" in names
    assert "output.colored" in names


# --- set_widget --------------------------------------------------------------


def test_set_sub_widget(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Render", object_info=oi)
    wf.set_widget(n.id, "output.normalization", "min_max", oi)
    assert n.widgets_values == ["depth", "min_max", False]


def test_switching_main_key_rebuilds_sub_widgets(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Render", object_info=oi)
    wf.set_widget(n.id, "output", "sky_mask", oi)
    # sky_mask has a single 'colored' sub-widget (default False), not normalization
    assert n.widgets_values == ["sky_mask", False]
    assert w.widgets_to_named("DA3Render", n.widgets_values, oi) == {
        "output": "sky_mask",
        "output.colored": False,
    }


def test_set_widget_multiview_then_sub(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Inference", object_info=oi)
    wf.set_widget(n.id, "mode", "multiview", oi)
    wf.set_widget(n.id, "mode.pose_method", "ray_pose", oi)
    named = w.widgets_to_named("DA3Inference", n.widgets_values, oi)
    assert named["mode"] == "multiview"
    assert named["mode.pose_method"] == "ray_pose"
    assert named["mode.ref_view_strategy"] == "saddle_balanced"  # defaulted


def test_set_unselected_option_sub_widget_is_rejected(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Render", object_info=oi)  # default option 'depth'
    with pytest.raises(ValueError, match="option isn't selected"):
        wf.set_widget(n.id, "output.colored", True, oi)  # belongs to sky_mask


# --- API serialization -------------------------------------------------------


def test_to_api_emits_dotted_keys(oi):
    wf = Workflow.new()
    inf = wf.add_node("DA3Inference", object_info=oi)
    ren = wf.add_node("DA3Render", object_info=oi)
    wf.connect(inf.id, 0, ren.id, "da3_geometry", oi)
    api = wf.to_api(oi)
    inputs = api[str(ren.id)]["inputs"]
    assert inputs["output"] == "depth"
    assert inputs["output.normalization"] == "v2_style"
    assert inputs["output.apply_sky_clip"] is False


def test_api_roundtrip_stable(oi):
    wf = Workflow.new()
    inf = wf.add_node("DA3Inference", object_info=oi)
    wf.set_widget(inf.id, "mode", "multiview", oi)
    ren = wf.add_node("DA3Render", object_info=oi)
    wf.connect(inf.id, 0, ren.id, "da3_geometry", oi)
    api = wf.to_api(oi)
    rebuilt = Workflow.from_api(api, oi).to_api(oi)
    assert rebuilt == api


def test_ui_roundtrip_preserves_widget_values(oi):
    wf = Workflow.new()
    sav = wf.add_node("SaveImageAdvanced", object_info=oi)
    wf.set_widget(sav.id, "format", "exr", oi)
    before = list(wf.nodes[sav.id].widgets_values)
    wf2 = Workflow.from_ui(wf.to_ui())
    assert wf2.nodes[sav.id].widgets_values == before


def test_to_api_fills_dropped_dynamic_values(oi):
    """A graph whose V3 node lost its widget values (older buggy save) still
    reaches ComfyUI with the schema defaults rather than a missing dotted key."""
    wf = Workflow.new()
    ren = wf.add_node("DA3Render", object_info=oi)
    inf = wf.add_node("DA3Inference", object_info=oi)
    wf.connect(inf.id, 0, ren.id, "da3_geometry", oi)
    wf.nodes[ren.id].widgets_values = []  # simulate the drop
    inputs = wf.to_api(oi)[str(ren.id)]["inputs"]
    assert inputs["output"] == "depth"
    assert inputs["output.normalization"] == "v2_style"
    assert inputs["output.apply_sky_clip"] is False


# --- validation --------------------------------------------------------------


def test_validate_no_false_unconnected_on_dynamic_combo(oi):
    wf = Workflow.new()
    inf = wf.add_node("DA3Inference", object_info=oi)
    ren = wf.add_node("DA3Render", object_info=oi)
    wf.connect(inf.id, 0, ren.id, "da3_geometry", oi)
    codes = {(f["code"], f.get("input")) for f in validate(wf, oi) if f["level"] == "error"}
    # 'output' (the V3 combo) must NOT be flagged as an unconnected input
    assert ("unconnected-input", "output") not in codes


def test_validate_flags_genuinely_unconnected_link_input(oi):
    wf = Workflow.new()
    wf.add_node("DA3Inference", object_info=oi)  # da3_model + image unset
    errors = [f for f in validate(wf, oi) if f["code"] == "unconnected-input"]
    inputs = {f["input"] for f in errors}
    assert "da3_model" in inputs and "image" in inputs
    assert "mode" not in inputs  # the V3 combo is a widget, not a required link


def test_validate_rejects_invalid_combo_key(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Inference", object_info=oi)
    wf.nodes[n.id].widgets_values[2] = "bogus_mode"
    errors = [f for f in validate(wf, oi) if f["code"] == "invalid-combo-value"]
    assert any(f.get("input") == "mode" for f in errors)


def test_validate_no_count_drift_for_expanded_multiview(oi):
    wf = Workflow.new()
    n = wf.add_node("DA3Inference", object_info=oi)
    wf.set_widget(n.id, "mode", "multiview", oi)
    drift = [f for f in validate(wf, oi) if f["code"] == "widget-count-drift"]
    assert not drift
