"""0.25.6: session feedback - subgraph instance writes point at
set_widget_in_definition, a missing LoRA suggests bypass instead of a
name-alike, and Comfy-Org model families detect on their own files."""

from __future__ import annotations

import json

import pytest

from comfy_draftsman import knowledge, server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import check_widget_value, validate

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"
SUBGRAPH_TEMPLATE = "tests/fixtures/subgraph_real_template.json"


@pytest.fixture(scope="module")
def oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def sub_wf():
    with open(SUBGRAPH_TEMPLATE, encoding="utf-8") as fh:
        wf = Workflow.from_ui(json.load(fh))
    inst = next(n for n in wf.nodes.values() if n.type in wf.subgraph_defs())
    return wf, inst


@pytest.mark.parametrize("op", [
    {"op": "set_widget", "input": "text", "value": "a cat"},
    {"op": "set_widget", "input": "width", "value": 512},  # an "(internal)" input
    {"op": "replace_in_widget", "input": "text", "old": "a", "new": "b"},
])
def test_subgraph_instance_write_names_the_definition_op(sub_wf, oi, op):
    wf, inst = sub_wf
    with pytest.raises(ValueError) as err:
        server._apply_ops(wf, [{**op, "node_id": inst.id}], oi, [], set())
    msg = str(err.value)
    assert "set_widget_in_definition" in msg and f"definition_id={inst.type!r}" in msg
    assert "node_id=" in msg and "NoneType" not in msg


def test_subgraph_raw_index_write_on_a_live_proxy_still_lands(sub_wf, oi):
    wf, inst = sub_wf
    inst.widgets_values = ["old prompt"]  # zips over proxyWidgets[0] = ['90', 'text']
    applied: list[str] = []
    server._apply_ops(wf, [{"op": "set_widget", "node_id": inst.id, "index": 0, "value": "new"}], oi, applied, set())
    assert inst.widgets_values == ["new"]


def test_missing_lora_suggests_bypass_not_a_name_alike(oi):
    wf = Workflow.new()
    lora = wf.add_node("LoraLoader", object_info=oi)
    wf.set_widget(lora.id, "lora_name", "my_style_lora.safetensors", oi)
    found = [f for f in validate(wf, oi) if f.get("node_id") == lora.id and f.get("input") == "lora_name"]
    assert found and "bypass" in found[0]["message"]
    assert "closest installed" not in found[0]["message"] and not found[0].get("suggestion")
    assert "bypass" in (check_widget_value("LoraLoader", "lora_name", "nope.safetensors", oi) or "")


@pytest.mark.parametrize("filename, family", [
    ("minimax_h3_image_pruned_int8_convrot.safetensors", "minimax_h3_image"),
    ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3"),
    ("flux2_dev_fp8mixed.safetensors", "flux2"),
    ("flux-2-klein-base-9b-fp8.safetensors", "flux2"),
    ("flux1-dev-fp8.safetensors", "flux"),
    ("z_image_turbo_bf16.safetensors", "z_image"),
    ("wan2.2_bernini_r_high_noise_fp8_scaled.safetensors", "bernini_r"),
    ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan"),
    ("hidream_i1_dev_fp8.safetensors", "hidream"),
    ("mage_flow_turbo_int8_convrot.safetensors", "mage_flow"),
    ("lens_turbo_bf16.safetensors", "lens"),
    ("fisheye_lens_xl.safetensors", None),
])
def test_new_families_detect_on_their_files(filename, family):
    assert knowledge.detect_family_from_filename(filename)[0] == family


def test_hidream_o1_checkpoint_beats_hidream():
    refs = [("ckpt_name", "hidream_o1_image_dev_fp8_scaled.safetensors", "primary")]
    assert knowledge._detect_over_refs(refs, knowledge._detect_index(None))[0] == "hidream_o1"


@pytest.mark.parametrize("family, filename, steps", [
    ("flux2", "flux-2-klein-4b.safetensors", 4),
    ("flux2", "flux-2-klein-base-4b.safetensors", 20),
    ("z_image", "z_image_turbo_bf16.safetensors", 8),
    ("hunyuan_video", "hunyuanvideo1.5_720p_t2v_fp16.safetensors", 20),
])
def test_variants_pick_their_settings(family, filename, steps):
    assert knowledge.get_guidance(family, filename)["sampling"]["steps"]["default"] == steps


def test_every_floor_has_a_readable_shape():
    for family in knowledge.list_families():
        assert knowledge.shape_errors(knowledge.get_guidance(family)) == [], family
