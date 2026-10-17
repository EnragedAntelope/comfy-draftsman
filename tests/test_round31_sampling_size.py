"""Sampling resolution feedback: warn only when literal sizes are unambiguous."""

import pytest

from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, MODE_MUTE, InputSlot, Workflow


@pytest.fixture
def size_graph():
    info = {
        name: {
            "input": {"required": {"width": ["INT", {"default": 1024}],
                                   "height": ["INT", {"default": 1024}]}},
            "output": [],
        }
        for name in ("EmptyLatentImage", "ModelSamplingFlux")
    }
    wf = Workflow.new()
    canvas = wf.add_node("EmptyLatentImage", object_info=info)
    sampling = wf.add_node("ModelSamplingFlux", object_info=info)
    sampling.widgets_values = [512, 768]
    return wf, info, canvas, sampling


def _size_findings(wf, info):
    return [f for f in lint(wf, info) if f["code"] == "model-sampling-size-mismatch"]


def test_sampling_size_mismatch(size_graph):
    wf, info, _, sampling = size_graph
    findings = _size_findings(wf, info)
    assert len(findings) == 1
    assert findings[0]["node_id"] == sampling.id
    assert sampling.widgets_values == [512, 768]


@pytest.mark.parametrize(
    "edge",
    ["matching", "linked_sampling", "linked_canvas", "ambiguous", "bypass",
     "mute", "disabled_canvas", "nonliteral", "no_canvas", "missing_schema",
     "linked_second_canvas"],
)
def test_sampling_size_uncertain_or_disabled_stays_silent(size_graph, edge):
    wf, info, canvas, sampling = size_graph
    if edge == "matching":
        sampling.widgets_values = [1024, 1024]
    elif edge == "linked_sampling":
        sampling.inputs.append(InputSlot(name="width", type="INT", link=999))
    elif edge == "linked_canvas":
        canvas.inputs.append(InputSlot(name="width", type="INT", link=999))
    elif edge == "ambiguous":
        other = wf.add_node("EmptyLatentImage", object_info=info)
        other.widgets_values = [2048, 2048]
    elif edge == "bypass":
        sampling.mode = MODE_BYPASS
    elif edge == "mute":
        sampling.mode = MODE_MUTE
    elif edge == "disabled_canvas":
        canvas.mode = MODE_BYPASS
    elif edge == "nonliteral":
        sampling.widgets_values = ["dynamic", 768]
    elif edge == "no_canvas":
        del wf.nodes[canvas.id]
    elif edge == "missing_schema":
        del info["ModelSamplingFlux"]
    elif edge == "linked_second_canvas":
        other = wf.add_node("EmptyLatentImage", object_info=info)
        other.inputs.append(InputSlot(name="width", type="INT", link=999))
    assert _size_findings(wf, info) == []
