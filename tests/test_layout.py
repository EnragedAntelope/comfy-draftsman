"""Layered auto-layout: left-to-right data flow, no overlaps, deterministic."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.layout import apply_layout, estimate_size
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture
def txt2img(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    return wf, {"ckpt": ckpt.id, "sampler": sampler.id, "save": save.id, "pos": pos.id}


def _boxes(wf):
    return {
        n.id: (n.pos[0], n.pos[1], n.pos[0] + n.size[0], n.pos[1] + n.size[1])
        for n in wf.nodes.values()
    }


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_dataflow_goes_left_to_right(txt2img, object_info):
    wf, ids = txt2img
    apply_layout(wf, object_info)
    for link in wf.links.values():
        origin = wf.nodes[link.origin_id]
        target = wf.nodes[link.target_id]
        assert origin.pos[0] + origin.size[0] <= target.pos[0], (
            f"{origin.type} should be left of {target.type}"
        )


def test_no_overlapping_nodes(txt2img, object_info):
    wf, _ = txt2img
    apply_layout(wf, object_info)
    boxes = _boxes(wf)
    items = list(boxes.items())
    for i, (id_a, box_a) in enumerate(items):
        for id_b, box_b in items[i + 1 :]:
            assert not _overlaps(box_a, box_b), f"nodes {id_a} and {id_b} overlap"


def test_relayout_of_real_template_has_no_overlaps(object_info):
    template = json.loads((FIXTURES / "sdxl_simple_example.json").read_text(encoding="utf-8"))
    wf = Workflow.from_ui(template)
    apply_layout(wf, object_info)
    boxes = _boxes(wf)
    items = list(boxes.items())
    for i, (id_a, box_a) in enumerate(items):
        for id_b, box_b in items[i + 1 :]:
            assert not _overlaps(box_a, box_b), f"nodes {id_a} and {id_b} overlap"


def test_layout_is_deterministic(txt2img, object_info):
    wf, _ = txt2img
    apply_layout(wf, object_info)
    first = {n.id: tuple(n.pos) for n in wf.nodes.values()}
    apply_layout(wf, object_info)
    second = {n.id: tuple(n.pos) for n in wf.nodes.values()}
    assert first == second


def test_estimate_size_scales_with_widget_count(object_info):
    ksampler = estimate_size("KSampler", object_info)
    decode = estimate_size("VAEDecode", object_info)
    assert ksampler[1] > decode[1]
    assert ksampler[0] >= 200


def test_text_nodes_get_wider_boxes(object_info):
    encode = estimate_size("CLIPTextEncode", object_info)
    decode = estimate_size("VAEDecode", object_info)
    assert encode[0] > decode[0]
