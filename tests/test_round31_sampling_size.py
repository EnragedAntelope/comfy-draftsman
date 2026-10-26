"""Sampling resolution feedback: warn only when the sampled latent size is unambiguous."""

import pytest

from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, MODE_MUTE, InputSlot, Workflow

_SIZE = {"width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}]}


@pytest.fixture
def size_graph():
    info = {
        "EmptyLatentImage": {"input": {"required": dict(_SIZE)}, "output": ["LATENT"]},
        "ModelSamplingFlux": {
            "input": {"required": {"model": ["MODEL"], **_SIZE}},
            "output": ["MODEL"],
        },
        "Loader": {"input": {"required": {}}, "output": ["MODEL"]},
        "LatentUpscale": {"input": {"required": {"samples": ["LATENT"]}}, "output": ["LATENT"]},
        "KSampler": {
            "input": {"required": {"model": ["MODEL"], "latent_image": ["LATENT"]}},
            "output": ["LATENT"],
        },
    }
    wf = Workflow.new()
    loader = wf.add_node("Loader", object_info=info)
    canvas = wf.add_node("EmptyLatentImage", object_info=info)
    sampling = wf.add_node("ModelSamplingFlux", object_info=info)
    sampler = wf.add_node("KSampler", object_info=info)
    wf.connect(loader.id, 0, sampling.id, "model", info)
    wf.connect(sampling.id, 0, sampler.id, "model", info)
    wf.connect(canvas.id, 0, sampler.id, "latent_image", info)
    sampling.widgets_values = [512, 768]
    return wf, info, canvas, sampling, sampler


def _size_findings(wf, info):
    return [f for f in lint(wf, info) if f["code"] == "model-sampling-size-mismatch"]


def test_sampling_size_mismatch(size_graph):
    wf, info, _, sampling, _ = size_graph
    findings = _size_findings(wf, info)
    assert len(findings) == 1
    assert findings[0]["node_id"] == sampling.id
    assert "latent units" not in findings[0]["message"]
    assert sampling.widgets_values == [512, 768]


def test_any_size_tuned_model_patch_is_checked_and_latent_units_named(size_graph):
    wf, info, _, sampling, _ = size_graph
    info["JigsawHiRes"] = info.pop("ModelSamplingFlux")
    sampling.type = "JigsawHiRes"
    sampling.widgets_values = [128, 128]  # 1024 / 8: typed in latent units
    (finding,) = _size_findings(wf, info)
    assert finding["message"].startswith("JigsawHiRes #")
    assert "latent units /8" in finding["message"]


@pytest.mark.parametrize(
    "edge",
    ["matching", "linked_sampling", "linked_canvas", "bypass", "mute",
     "disabled_canvas", "nonliteral", "missing_schema", "hires_pass", "unrelated_canvas"],
)
def test_sampling_size_uncertain_or_disabled_stays_silent(size_graph, edge):
    wf, info, canvas, sampling, sampler = size_graph
    if edge == "matching":
        sampling.widgets_values = [1024, 1024]
    elif edge == "linked_sampling":
        sampling.inputs.append(InputSlot(name="width", type="INT", link=999))
    elif edge == "linked_canvas":
        canvas.inputs.append(InputSlot(name="width", type="INT", link=999))
    elif edge == "bypass":
        sampling.mode = MODE_BYPASS
    elif edge == "mute":
        sampling.mode = MODE_MUTE
    elif edge == "disabled_canvas":
        canvas.mode = MODE_BYPASS
    elif edge == "nonliteral":
        sampling.widgets_values = ["dynamic", 768]
    elif edge == "missing_schema":
        del info["ModelSamplingFlux"]
    elif edge == "hires_pass":
        # the patched model samples an upscaled latent, not the canvas
        up = wf.add_node("LatentUpscale", object_info=info)
        wf.connect(canvas.id, 0, up.id, "samples", info)
        wf.connect(up.id, 0, sampler.id, "latent_image", info)
    elif edge == "unrelated_canvas":
        sampling.widgets_values = [1024, 1024]
        other = wf.add_node("EmptyLatentImage", object_info=info)
        other.widgets_values = [2048, 2048]
    assert _size_findings(wf, info) == []
