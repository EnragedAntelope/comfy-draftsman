"""0.25.7: H3-Image session feedback - sweep cache flag, allow_invalid and crop
guide, family banding calibration, bare LoRA metadata names, all malformed ops
in one error, group lint, panel staging, locked knobs and primitive knob rows."""

from __future__ import annotations

import pytest
from PIL import Image

from comfy_draftsman import knowledge, server
from comfy_draftsman.graph import knobs
from comfy_draftsman.graph.annotate import _panel_stage
from comfy_draftsman.graph.lint import group_findings, lint
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.imaging import crop_guide
from test_round27_qwen21 import server_state, sweep_env  # noqa: F401 - fixtures

OI = {
    "KSamplerSelect": {"input": {"required": {"sampler_name": [["euler", "lcm"]]}}, "output": ["SAMPLER"]},
    "SamplerCustomAdvanced": {"input": {"required": {"noise": ["NOISE"], "sigmas": ["SIGMAS"]}}, "output": ["LATENT", "LATENT"]},
    "BasicScheduler": {
        "input": {"required": {
            "model": ["MODEL"],
            "scheduler": [["simple", "beta"]],
            "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
            "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
        }},
        "output": ["SIGMAS"],
    },
    "RandomNoise": {"input": {"required": {"noise_seed": ["INT", {"min": 0, "max": 2**64}]}}, "output": ["NOISE"]},
    "PrimitiveFloat": {"input": {"required": {"value": ["FLOAT", {}]}}, "output": ["FLOAT"], "category": "utilities/primitive"},
    "UNETLoader": {"input": {"required": {"unet_name": [["h3.safetensors"]]}}, "output": ["MODEL"], "category": "loaders"},
    "LLMNode": {"input": {"required": {"prompt": ["STRING", {}], "seed": ["INT", {}]}}, "output": ["STRING"]},
    "SaveImageWithMetaDataUniversal": {
        "input": {"required": {"images": ["IMAGE"]}, "optional": {"extra_metadata": ["EXTRA_METADATA"]}},
        "output": [], "output_node": True,
    },
}
H3_GUIDANCE = {"sampling": {"cfg": {"min": 1, "max": 1}, "steps": {"min": 6, "max": 6}, "schedulers": ["simple"]}}


def test_sampler_object_nodes_do_not_count_as_cached_sampling():
    api = {"1": {"class_type": "KSamplerSelect"}, "2": {"class_type": "SamplerLCM"}, "3": {"class_type": "SamplerCustomAdvanced"}}
    assert not server._sampling_cached({"cached_nodes": ["1"]}, api, OI)
    assert not server._sampling_cached({"cached_nodes": ["2"]}, api)  # unknown Sampler*: a SAMPLER object
    assert server._sampling_cached({"cached_nodes": ["3"]}, api, OI)
    assert server._sampling_cached({"cached_nodes": ["3"]}, api)


async def test_sweep_honours_allow_invalid(sweep_env, monkeypatch):  # noqa: F811
    srv, client, _wf, _node, wf_id, out = sweep_env
    monkeypatch.setattr(srv, "validate", lambda wf, oi: [{"level": "error", "message": "bad"}])
    refused = await srv.run_workflow(wf_id, front=True, save_dir=str(out), sweep={"seeds": [1]}, return_preview=False)
    assert refused["status"] == "invalid" and client.apis == []
    ran = await srv.run_workflow(
        wf_id, front=True, save_dir=str(out), sweep={"seeds": [1]}, allow_invalid=True, return_preview=False
    )
    assert ran["runs"][0]["status"] == "success" and len(client.apis) == 1
    assert "crop_guide" in ran  # every sweep returns it now


def test_family_grain_suppresses_banding_on_its_period():
    grain = {"family": "minimax_h3_image", "grain_period_px": 16, "grain_max": 25}
    assert server._is_family_grain(20.0, "vertical 16px", grain)
    assert server._is_family_grain(20.0, "horizontal 17px", grain)
    assert not server._is_family_grain(30.0, "vertical 16px", grain)  # past the calibrated ceiling
    assert not server._is_family_grain(20.0, "vertical 48px", grain)  # a real stripe
    assert not server._is_family_grain(20.0, "vertical 16px", None)
    assert knowledge.get_guidance("minimax_h3_image")["banding"]["grain_period_px"] == 16


def test_crop_guide_outlines_requested_crops():
    guide = crop_guide(Image.new("RGB", (1000, 1000), (0, 0, 0)), size=100, boxes=[[0.2, 0.2, 0.6, 0.6]])
    assert guide.getpixel((20, 40)) == (255, 40, 40)  # left edge of the box


def test_every_malformed_op_is_reported_at_once_and_aliases_apply():
    wf = Workflow.new()
    ops = [
        {"op": "connect", "from_node": 1, "from_slot": 0, "to_node": 2, "to_input": "x"},  # alias: fine
        {"op": "set_widget", "node_id": 1, "name": "steps", "value": 4},  # alias: fine
        {"op": "set_widget", "node_id": 1, "inptu": "steps", "value": 4},
        {"op": "set_title", "node_id": 1},
    ]
    with pytest.raises(ValueError) as err:
        server._apply_ops(wf, ops, {}, [], set())
    msg = str(err.value)
    assert "operation 2" in msg and "operation 3" in msg
    assert "operation 0" not in msg and "operation 1" not in msg
    assert server._alias_keys(ops[0])["from_output"] == 0 and server._alias_keys(ops[1])["input"] == "steps"


def test_group_lint_flags_empty_and_partly_overlapping_groups():
    wf = Workflow.new()
    node = wf.add_node("PrimitiveFloat", object_info=OI)
    node.pos, node.size = [0, 0], [100, 60]
    wf.group_from_nodes("Holds it", [node.id])
    a = wf.group_from_nodes("Empty", [node.id])
    a.bounding = [500, 500, 200, 200]
    b = wf.group_from_nodes("Overlaps", [node.id])
    b.bounding = [600, 600, 300, 300]
    codes = [f["code"] for f in group_findings(wf)]
    assert codes.count("empty-group") == 2 and codes.count("group-overlap") == 1
    inner = wf.group_from_nodes("Nested", [node.id])
    inner.bounding = [620, 620, 50, 50]  # fully inside "Overlaps": a nest, not an overlap
    assert [f["code"] for f in group_findings(wf)].count("group-overlap") == 1


def test_locked_knobs_drop_out_and_primitives_get_rows():
    wf = Workflow.new()
    sched = wf.add_node("BasicScheduler", object_info=OI)
    sched.widgets_values = ["simple", 6, 1.0]
    noise = wf.add_node("RandomNoise", object_info=OI)
    noise.widgets_values = [42]
    prim = wf.add_node("PrimitiveFloat", object_info=OI)
    prim.widgets_values, prim.title = [16.0], "Megapixels"
    locked: list[str] = []
    rows = knobs.knob_rows([sched, noise, prim], OI, H3_GUIDANCE, locked)
    assert locked == ["scheduler simple", "steps 6", "denoise 1.0"]
    assert any("Megapixels #" in r and "16.0" in r for r in rows)
    assert any(r.startswith("| noise_seed") for r in rows)
    assert not any("steps" in r for r in rows)
    sched.widgets_values = ["beta", 8, 0.5]  # off the floor: a real knob again
    assert len(knobs.knob_rows([sched], OI, H3_GUIDANCE)) == 3
    # a one-entry scheduler list without pinned steps is a template default, not a lock
    sched.widgets_values = ["simple", 20, 1.0]
    assert len(knobs.knob_rows([sched], OI, {"sampling": {"schedulers": ["simple"]}})) == 3


def test_lcm_note_is_dropped_for_a_distilled_family():
    assert knobs.technique_note("SamplerLCM")
    assert knobs.technique_note("SamplerLCM", distilled=True) is None
    assert knobs.technique_note("EasyCache", distilled=True)


def test_panel_puts_loaders_and_llm_seeds_in_plumbing():
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=OI)
    llm = wf.add_node("LLMNode", object_info=OI)
    llm.widgets_values = ["idea", 7]
    noise = wf.add_node("RandomNoise", object_info=OI)
    noise.widgets_values = [42]
    sampler = wf.add_node("SamplerCustomAdvanced", object_info=OI)
    wf.connect(noise.id, 0, sampler.id, "noise", OI)
    sched = wf.add_node("BasicScheduler", object_info=OI)
    sched.widgets_values = ["simple", 6, 1.0]
    wf.connect(sched.id, 0, sampler.id, "sigmas", OI)
    assert _panel_stage(wf, unet, OI, "models", H3_GUIDANCE) == "plumbing"
    assert _panel_stage(wf, llm, OI, "prompt_build", H3_GUIDANCE) == "plumbing"
    assert _panel_stage(wf, noise, OI, "sampling", H3_GUIDANCE) == "controls"
    assert _panel_stage(wf, sched, OI, "sampling", H3_GUIDANCE) == "plumbing"
    assert _panel_stage(wf, sched, OI, "sampling", None) == "controls"


def test_metadata_saver_without_an_encoder_suggests_extra_metadata():
    wf = Workflow.new()
    wf.add_node("SaveImageWithMetaDataUniversal", object_info=OI)
    assert any(f["code"] == "metadata-prompt-uncaptured" for f in lint(wf, OI))
