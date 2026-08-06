"""Round 23 / Part D4: resolutions rendered into the Inputs note, and a
resolution-not-aligned lint warning for families with a known alignment
requirement - never an auto-rewrite, never an injected math node, and never a
guess for a family with no declared requirement."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.annotate import _resolution_text, annotate
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _note_texts(wf) -> list[str]:
    return [
        n.widgets_values[0]
        for n in wf.nodes.values()
        if n.type == "MarkdownNote" and n.widgets_values
    ]


# --- _resolution_text: list and dict forms, never invents ---


def test_resolution_text_list_form():
    text = _resolution_text(["1024x1024", "1152x896"])
    assert "1024x1024" in text
    assert "1152x896" in text


def test_resolution_text_dict_form_with_note_and_native():
    text = _resolution_text(
        {
            "note": "Native canvas is a 768px short edge.",
            "native": {"16:9": [1344, 768], "1:1": [1024, 1024]},
        }
    )
    assert "768px short edge" in text
    assert "1344x768" in text


def test_resolution_text_none_when_absent():
    assert _resolution_text(None) is None
    assert _resolution_text({}) is None
    assert _resolution_text([]) is None


def test_resolution_text_caps_long_lists():
    long_list = [f"{100 + i}x{100 + i}" for i in range(20)]
    text = _resolution_text(long_list)
    assert "more)" in text


# --- integration: resolution text reaches the Inputs note ---


def test_organize_renders_resolutions_in_inputs_note(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    annotate(wf, object_info)
    joined = "\n".join(_note_texts(wf))
    assert "Native resolutions" in joined
    assert "1024x1024" in joined


# --- lint: resolution-not-aligned ---


def _sdxl_wf_with_width(object_info, width):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    wf.set_widget(latent.id, "width", width, object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    return wf


def test_lint_flags_misaligned_resolution_when_learned_dir_given(object_info, tmp_path):
    wf = _sdxl_wf_with_width(object_info, 1001)  # not a multiple of 8
    findings = lint(wf, object_info, learned_dir=tmp_path)
    codes = [f["code"] for f in findings]
    assert "resolution-not-aligned" in codes


def test_lint_clean_when_resolution_aligned(object_info, tmp_path):
    wf = _sdxl_wf_with_width(object_info, 1024)  # multiple of 8
    findings = lint(wf, object_info, learned_dir=tmp_path)
    codes = [f["code"] for f in findings]
    assert "resolution-not-aligned" not in codes


def test_lint_skips_check_entirely_when_learned_dir_is_none(object_info):
    """learned_dir=None (lint()'s default) is a deliberate opt-out - existing
    callers that don't thread config through must never start seeing this
    finding just because a family YAML gained a multiple_of."""
    wf = _sdxl_wf_with_width(object_info, 1001)
    findings = lint(wf, object_info)  # no learned_dir passed
    codes = [f["code"] for f in findings]
    assert "resolution-not-aligned" not in codes


def test_lint_silent_for_undetected_family(object_info, tmp_path):
    """No family detected -> no claim, even with a misaligned-looking size."""
    wf = Workflow.new()
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    wf.set_widget(latent.id, "width", 1001, object_info)
    findings = lint(wf, object_info, learned_dir=tmp_path)
    assert not any(f["code"] == "resolution-not-aligned" for f in findings)


def test_lint_silent_for_family_with_no_multiple_of(object_info, tmp_path):
    """A family that hasn't declared multiple_of (most of them, currently)
    must never trigger a guessed constraint."""
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "krea2\\krea2_turbo_mxfp8.safetensors", object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    wf.set_widget(latent.id, "width", 1001, object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(unet.id, "MODEL", sampler.id, "model")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    findings = lint(wf, object_info, learned_dir=tmp_path)
    assert not any(f["code"] == "resolution-not-aligned" for f in findings)


def test_lint_never_mutates_the_widget_value(object_info, tmp_path):
    wf = _sdxl_wf_with_width(object_info, 1001)
    lint(wf, object_info, learned_dir=tmp_path)
    # find the EmptyLatentImage node and confirm its width is untouched
    latent = next(n for n in wf.nodes.values() if n.type == "EmptyLatentImage")
    from comfy_draftsman.graph import widgets as w

    named = w.widgets_to_named(latent.type, latent.widgets_values, object_info)
    assert named["width"] == 1001


def test_resolution_finding_names_nearest_legal_value(object_info, tmp_path):
    wf = _sdxl_wf_with_width(object_info, 1001)
    findings = lint(wf, object_info, learned_dir=tmp_path)
    finding = next(f for f in findings if f["code"] == "resolution-not-aligned")
    assert "1000" in finding["message"] or "1008" in finding["message"]
