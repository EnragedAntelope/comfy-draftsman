"""Round 23 / A3: notes may only assert what the graph actually contains.

A live session's learned minimax_h3.yaml overlay (sampling.cfg has a prose
'note', no numeric min/max) made the sampling band literally render
'Safe ranges: CFG None-None, steps None-None.' - and the fallback output note
was hardcoded to 'Finished images land here' on a video workflow with no
CFG anywhere."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import knowledge
from comfy_draftsman.graph.annotate import annotate
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


def test_sampling_note_omits_none_ranges(object_info, tmp_path):
    """A learned overlay whose sampling block has notes but no numeric
    min/max (exactly the shape of the user's real minimax_h3.yaml) must never
    render the literal string 'None' into a note."""
    knowledge.save_learning(
        tmp_path,
        "minimax_h3",
        {
            "detect": {"checkpoint_patterns": ["minimax_h3"]},
            "loader": "unet_clip_vae",
            "sampling": {
                "cfg": {"note": "No CFG. H3 is guidance-distilled."},
                "steps": {"default": 20, "range": [15, 30]},
            },
        },
        source="unit-test",
    )
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "minimax_h3_ref2va_pruned_int8.safetensors", object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(unet.id, "MODEL", sampler.id, "model")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    annotate(wf, object_info, learned_dir=tmp_path)
    for text in _note_texts(wf):
        assert "None" not in text, f"note rendered a None value: {text!r}"


def test_sampling_note_shows_cfg_prose_when_no_numeric_range(object_info, tmp_path):
    """When the family's cfg block is a prose note (H3: 'No CFG...'), that
    prose should still reach the reader even though there's no numeric range
    to render."""
    knowledge.save_learning(
        tmp_path,
        "minimax_h3",
        {
            "detect": {"checkpoint_patterns": ["minimax_h3"]},
            "loader": "unet_clip_vae",
            "sampling": {"cfg": {"note": "No CFG. H3 is guidance-distilled."}},
        },
        source="unit-test",
    )
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "minimax_h3_ref2va_pruned_int8.safetensors", object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(unet.id, "MODEL", sampler.id, "model")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    annotate(wf, object_info, learned_dir=tmp_path)
    joined = " ".join(_note_texts(wf))
    assert "No CFG" in joined or "guidance-distilled" in joined


def test_sampling_range_still_renders_with_real_numeric_bounds(object_info):
    """Un-regress the common case: a family (sdxl) whose cfg block DOES carry
    real min/max must still render the safe-ranges line."""
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    annotate(wf, object_info)
    joined = " ".join(_note_texts(wf))
    assert "None" not in joined
    assert "CFG" in joined


def test_output_note_names_video_not_images(object_info):
    """core SaveVideo isn't in the trimmed fixture, so inject a minimal
    synthetic schema shaped like the real one (single VIDEO input, no IMAGE) -
    the note must say 'video', never fall back to the old hardcoded 'images'."""
    synthetic = dict(object_info)
    synthetic["FakeSaveVideo"] = {
        "input": {"required": {"video": ["VIDEO", {}]}},
        "output": [],
        "output_node": True,
        "category": "video",
        "display_name": "Save Video",
    }
    wf = Workflow.new()
    wf.add_node("CheckpointLoaderSimple", object_info=synthetic)
    wf.add_node("FakeSaveVideo", object_info=synthetic)
    annotate(wf, synthetic)
    output_notes = [t for t in _note_texts(wf) if "Output" in t or "land here" in t]
    assert output_notes, "expected an output-band note"
    assert any("video" in t.lower() for t in output_notes)
    assert not any("image" in t.lower() for t in output_notes)


def test_output_note_still_says_images_for_image_workflow(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    annotate(wf, object_info)
    output_notes = [t for t in _note_texts(wf) if "Output" in t or "land here" in t]
    assert output_notes
    assert any("image" in t.lower() for t in output_notes)
