"""Round 23 / Part D1: default organization ordered by how OFTEN a reader
touches something, not purely by dataflow rank.

- a bare hand-typed prompt box is the single most commonly edited thing in
  almost any workflow, so it lives in the leftmost Inputs band, not buried in
  a middle "Conditioning" band
- a WIRED prompt (fed by a wildcard bank, concatenator, or LLM/VLM step)
  keeps that whole prompt-building pipeline together in its own band, ahead
  of Models/Conditioning, so the reader can see what produces the final text
- pure wiring (an encoder/apply node with nothing user-editable) is its own
  "Conditioning" band, explicitly labeled "nothing to edit here"
- data still flows strictly left to right: inputs -> prompt_build ->
  conditioning; models -> conditioning and -> sampling
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.annotate import _STAGE_INDEX, STAGES, annotate, classify
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"

_SHOWTEXT_SCHEMA = {
    "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
    "output": ["STRING"],
    "output_node": True,
    "python_module": "custom_nodes.pysssss",
}


@pytest.fixture(scope="module")
def object_info():
    info = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    info["ShowText|pysssss"] = _SHOWTEXT_SCHEMA
    return info


def _note_texts(wf) -> list[str]:
    return [
        n.widgets_values[0]
        for n in wf.nodes.values()
        if n.type == "MarkdownNote" and n.widgets_values
    ]


# --- classify(): the core routing decision ---


def test_unwired_clip_text_encode_classifies_as_inputs(object_info):
    wf = Workflow.new()
    enc = wf.add_node("CLIPTextEncode", object_info=object_info)
    assert classify(enc, object_info) == "inputs"


def test_wired_clip_text_encode_classifies_as_conditioning(object_info):
    wf = Workflow.new()
    wildcard_schema = {
        "category": "custom/text",
        "input": {"required": {"text": ["STRING", {"default": "", "multiline": True}]}},
        "output": ["STRING"],
    }
    info = dict(object_info)
    info["WildcardBank"] = wildcard_schema
    wf = Workflow.new()
    bank = wf.add_node("WildcardBank", object_info=info)
    enc = wf.add_node("CLIPTextEncode", object_info=info)
    wf.connect(bank.id, "STRING", enc.id, "text", info)
    assert classify(enc, info) == "conditioning"


def test_wildcard_producer_classifies_as_prompt_build_even_when_unwired(object_info):
    """A root wildcard/LLM producer stays in prompt_build regardless of its
    own wired state - splitting it from the rest of a prompt-building
    pipeline (concatenator, encoder) would defeat the point of the band."""
    info = dict(object_info)
    info["WildcardBank"] = {
        "category": "custom/text",
        "input": {"required": {"text": ["STRING", {"default": "", "multiline": True}]}},
        "output": ["STRING"],
    }
    wf = Workflow.new()
    bank = wf.add_node("WildcardBank", object_info=info)
    assert classify(bank, info) == "prompt_build"


def test_text_overlay_with_incidental_text_widget_stays_post(object_info):
    """A node whose PRIMARY job is image-in/image-out (e.g. a text overlay
    effect) must not be hijacked into Inputs just because it happens to also
    carry a widget literally named 'text'."""
    info = dict(object_info)
    info["TextOverlay"] = {
        "category": "custom/overlay",
        "input": {"required": {"image": ["IMAGE"], "text": ["STRING", {"default": ""}]}},
        "output": ["IMAGE"],
    }
    wf = Workflow.new()
    node = wf.add_node("TextOverlay", object_info=info)
    assert classify(node, info) == "post"


# --- end-to-end: band order and content ---


def _prompt_pipeline_wf(object_info):
    """checkpoint -> wildcard -> encoder -> sampler -> decode -> save, with a
    Show Text tapped off the wildcard (a real prompt-building pipeline)."""
    info = dict(object_info)
    info["WildcardBank"] = {
        "category": "custom/text",
        "input": {"required": {"text": ["STRING", {"default": "", "multiline": True}]}},
        "output": ["STRING"],
    }
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=info)
    bank = wf.add_node("WildcardBank", object_info=info)
    show = wf.add_node("ShowText|pysssss", object_info=info)
    pos = wf.add_node("CLIPTextEncode", object_info=info)
    neg = wf.add_node("CLIPTextEncode", object_info=info)
    latent = wf.add_node("EmptyLatentImage", object_info=info)
    sampler = wf.add_node("KSampler", object_info=info)
    decode = wf.add_node("VAEDecode", object_info=info)
    save = wf.add_node("SaveImage", object_info=info)
    wf.connect(bank.id, "STRING", pos.id, "text", info)
    wf.connect(bank.id, "STRING", show.id, "text", info)
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    return wf, info, {
        "ckpt": ckpt.id, "bank": bank.id, "show": show.id, "pos": pos.id,
        "neg": neg.id, "latent": latent.id, "sampler": sampler.id,
    }


def test_band_x_order_matches_reader_priority(object_info):
    wf, info, ids = _prompt_pipeline_wf(object_info)
    annotate(wf, info)
    # latent + neg (unwired) -> inputs; bank -> prompt_build; ckpt -> models;
    # pos (wired) -> conditioning; sampler -> sampling
    x = {name: wf.nodes[nid].pos[0] for name, nid in ids.items()}
    assert x["latent"] < x["bank"] < x["ckpt"] < x["pos"] < x["sampler"]


def test_wildcard_and_its_show_text_share_prompt_build_band(object_info):
    wf, info, ids = _prompt_pipeline_wf(object_info)
    annotate(wf, info)
    assert wf.nodes[ids["bank"]].pos[0] == wf.nodes[ids["show"]].pos[0]
    bank_group = next(
        g for g in wf.groups
        if g.bounding[0] <= wf.nodes[ids["bank"]].pos[0] < g.bounding[0] + g.bounding[2]
    )
    assert "prompt" in bank_group.title.lower()


def test_unwired_negative_prompt_lands_left_of_wired_positive(object_info):
    wf, info, ids = _prompt_pipeline_wf(object_info)
    annotate(wf, info)
    # neg is unwired (classic hand-typed negative box) -> Inputs (leftmost);
    # pos is wired (fed by the wildcard bank) -> Conditioning, further right
    assert wf.nodes[ids["neg"]].pos[0] < wf.nodes[ids["pos"]].pos[0]


def test_conditioning_note_says_nothing_to_edit(object_info):
    wf, info, _ids = _prompt_pipeline_wf(object_info)
    annotate(wf, info)
    joined = " ".join(_note_texts(wf))
    assert "nothing to edit" in joined.lower()


def test_prompt_build_note_explains_the_pipeline(object_info):
    wf, info, _ids = _prompt_pipeline_wf(object_info)
    annotate(wf, info)
    joined = " ".join(_note_texts(wf))
    assert "builds the final prompt" in joined.lower()


def test_stages_declares_seven_bands_in_reader_priority_order():
    keys = [key for key, _, _ in STAGES]
    assert keys == [
        "inputs", "prompt_build", "models", "conditioning", "sampling", "post", "output",
    ]
    assert _STAGE_INDEX["inputs"] < _STAGE_INDEX["prompt_build"] < _STAGE_INDEX["models"]
    assert _STAGE_INDEX["models"] < _STAGE_INDEX["conditioning"] < _STAGE_INDEX["sampling"]
