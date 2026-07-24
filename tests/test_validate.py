"""Validator: structural + live-catalog checks with actionable fix suggestions."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _codes(findings):
    return {f["code"] for f in findings}


def test_valid_minimal_graph_passes(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    # choose a value that actually exists in the live combo choices
    choices = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    wf.set_widget(ckpt.id, "ckpt_name", choices[0], object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    findings = validate(wf, object_info)
    errors = [f for f in findings if f["level"] == "error"]
    assert errors == []


def test_unknown_class_reported_with_registry_hint(object_info):
    wf = Workflow.new()
    wf.add_node("FaceDetailer", raw_widgets=[])
    findings = validate(wf, object_info)
    missing = [f for f in findings if f["code"] == "missing-node-class"]
    assert missing and missing[0]["class_type"] == "FaceDetailer"


def test_bad_combo_value_gets_closest_suggestion(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    choices = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    target = choices[0]
    # simulate an old workflow referencing a renamed/moved file
    typo = target.replace(".safetensors", "_old.ckpt")
    wf.set_widget(ckpt.id, "ckpt_name", typo, object_info)
    findings = validate(wf, object_info)
    combo = [f for f in findings if f["code"] == "invalid-combo-value"]
    assert combo
    assert combo[0]["suggestion"] == target


def test_out_of_range_numeric_flagged(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "steps", 100000, object_info)
    findings = validate(wf, object_info)
    assert any(f["code"] == "out-of-range" and f["node_id"] == sampler.id for f in findings)


def test_unconnected_required_input_is_error(object_info):
    wf = Workflow.new()
    wf.add_node("KSampler", object_info=object_info)
    findings = validate(wf, object_info)
    dangling = [f for f in findings if f["code"] == "unconnected-input"]
    assert dangling and all(f["level"] == "error" for f in dangling)


def test_widget_count_drift_reported(object_info):
    wf = Workflow.new()
    node = wf.add_node("KSampler", object_info=object_info)
    node.widgets_values = [42, "randomize"]  # ancient workflow with fewer widgets
    findings = validate(wf, object_info)
    assert any(f["code"] == "widget-count-drift" for f in findings)


def test_widget_count_drift_static_node_stays_warning(object_info):
    wf = Workflow.new()
    node = wf.add_node("KSampler", object_info=object_info)
    node.widgets_values = [42, "randomize"]
    findings = validate(wf, object_info)
    drift = [f for f in findings if f["code"] == "widget-count-drift"]
    assert drift and drift[0]["level"] == "warning"


def test_widget_count_drift_dynamic_node_is_info():
    """Dynamic nodes (text concatenators, switches) declare dozens of optional
    widgets but serialize only the ones in use - that drift is normal and must
    not be reported at warning level."""
    oi = {
        "DynamicConcat": {
            "input": {
                "required": {"delimiter": ["STRING", {"default": ", "}]},
                "optional": {
                    f"text_{c}": ["STRING", {"default": ""}] for c in "abcdefghijkl"
                },
            },
            "output": ["STRING"],
        }
    }
    wf = Workflow.new()
    node = wf.add_node("DynamicConcat")
    node.widgets_values = [", ", "cat", "dog"]  # only 3 of 13 slots serialized
    findings = validate(wf, oi)
    drift = [f for f in findings if f["code"] == "widget-count-drift"]
    assert drift and drift[0]["level"] == "info", findings


def test_null_widget_value_is_error(object_info):
    """A null widget value crashes the ComfyUI editor when queueing - validate
    must flag it even though the count matches and the type is right."""
    wf = Workflow.new()
    node = wf.add_node("CLIPTextEncode", object_info=object_info)
    node.widgets_values = [None]
    findings = validate(wf, object_info)
    nulls = [f for f in findings if f["code"] == "null-widget-value"]
    assert nulls and nulls[0]["level"] == "error" and nulls[0]["node_id"] == node.id



def test_step_aligned_value_passes(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "steps", 20, object_info)
    findings = validate(wf, object_info)
    assert not any(f["code"] == "step-misaligned" for f in findings)


def test_step_misaligned_value_fails(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "cfg", 1.23, object_info)
    findings = validate(wf, object_info)
    assert any(f["code"] == "step-misaligned" for f in findings)


def test_step_float_tolerance_passes(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "cfg", 1.0, object_info)
    findings = validate(wf, object_info)
    assert not any(f["code"] == "step-misaligned" for f in findings)


def test_step_absent_no_flag(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "seed", 42, object_info)
    findings = validate(wf, object_info)
    assert not any(f["code"] == "step-misaligned" for f in findings)


def test_muted_producer_feeding_required_input_is_flagged(object_info):
    """A required input wired to a MUTE (mode 2) node validated 'connected' but the
    run failed on a dangling reference; validate now flags it up front."""
    from comfy_draftsman.graph.model import MODE_MUTE, MODE_NORMAL

    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    choices = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    wf.set_widget(ckpt.id, "ckpt_name", choices[0], object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")

    # connected to an active source -> no muted finding
    assert not any(f["code"] == "muted-input-source" for f in validate(wf, object_info))

    ckpt.mode = MODE_MUTE
    findings = validate(wf, object_info)
    muted = [f for f in findings if f["code"] == "muted-input-source"]
    assert muted, findings
    m = next(f for f in muted if f["input"] == "model")
    assert m["level"] == "error"
    assert m["node_id"] == sampler.id
    # 'model' is wired, so it must NOT also be reported as unconnected
    assert not any(
        f["code"] == "unconnected-input" and f["node_id"] == sampler.id and f["input"] == "model"
        for f in findings
    )

    ckpt.mode = MODE_NORMAL
    assert not any(f["code"] == "muted-input-source" for f in validate(wf, object_info))


def test_muted_source_seen_through_reroute(object_info):
    """The mute is detected even when the link reaches the sampler via a Reroute."""
    from comfy_draftsman.graph.model import MODE_MUTE, InputSlot, OutputSlot

    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    choices = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    wf.set_widget(ckpt.id, "ckpt_name", choices[0], object_info)
    reroute = wf.add_node("Reroute", raw_widgets=[])
    reroute.inputs.append(InputSlot(name="", type="*"))
    reroute.outputs.append(OutputSlot(name="", type="*"))
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", reroute.id, "")
    wf.connect(reroute.id, 0, sampler.id, "model")

    ckpt.mode = MODE_MUTE
    findings = validate(wf, object_info)
    assert any(
        f["code"] == "muted-input-source" and f["node_id"] == sampler.id for f in findings
    ), findings
