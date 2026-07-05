"""Graph model: parse UI format, round-trip, serialize to API format, build programmatically.

Fixtures are real: sdxl_simple_example.json fetched from ComfyUI 0.27.0's bundled
templates; object_info_trimmed.json extracted from the live /object_info.
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture
def template():
    return json.loads((FIXTURES / "sdxl_simple_example.json").read_text(encoding="utf-8"))


# --- Parsing UI format ---


def test_parse_ui_template(template):
    wf = Workflow.from_ui(template)
    assert len(wf.nodes) == 25
    assert len(wf.links) == 23
    assert len(wf.groups) == 10
    ksa = wf.nodes[10]
    assert ksa.type == "KSamplerAdvanced"
    assert ksa.widgets_values[1] == 721897303308196


def test_parse_preserves_titles_and_colors(template):
    wf = Workflow.from_ui(template)
    note42 = wf.nodes[42]
    assert note42.type == "Note"
    assert note42.title == "Note - Empty Latent Image"
    assert note42.color == "#323"


# --- Round-trip UI -> model -> UI ---


def test_roundtrip_ui(template):
    wf = Workflow.from_ui(template)
    out = wf.to_ui()
    assert out["version"] == 0.4
    assert len(out["nodes"]) == 25
    assert len(out["links"]) == 23
    assert len(out["groups"]) == 10
    by_id = {n["id"]: n for n in out["nodes"]}
    orig_by_id = {n["id"]: n for n in template["nodes"]}
    for nid, orig in orig_by_id.items():
        got = by_id[nid]
        assert got["type"] == orig["type"]
        assert got["widgets_values"] == orig.get("widgets_values", got.get("widgets_values"))
        assert [round(p) for p in got["pos"]] == [round(p) for p in orig["pos"]]
    # links preserved as 6-tuples
    orig_links = {tuple(link[:6]) for link in template["links"]}
    got_links = {tuple(link[:6]) for link in out["links"]}
    assert got_links == orig_links


# --- API serialization ---


def test_to_api_excludes_frontend_only_nodes(template, object_info):
    wf = Workflow.from_ui(template)
    api = wf.to_api(object_info)
    types = {entry["class_type"] for entry in api.values()}
    assert "Note" not in types
    assert "MarkdownNote" not in types
    assert "PrimitiveNode" not in types


def test_to_api_maps_widgets_with_control_after_generate(template, object_info):
    wf = Workflow.from_ui(template)
    api = wf.to_api(object_info)
    ksa = api["10"]["inputs"]
    assert ksa["add_noise"] == "enable"
    assert ksa["noise_seed"] == 721897303308196
    assert ksa["cfg"] == 8
    assert ksa["sampler_name"] == "euler"
    assert ksa["scheduler"] == "normal"
    assert ksa["start_at_step"] == 0
    assert ksa["return_with_leftover_noise"] == "enable"
    # 'randomize' control value must NOT leak into inputs
    assert "randomize" not in ksa.values()


def test_to_api_bakes_primitive_values_into_converted_widgets(template, object_info):
    wf = Workflow.from_ui(template)
    api = wf.to_api(object_info)
    # steps and end_at_step are connected from PrimitiveNodes titled 'steps' (25)
    # and 'end_at_step' (20) - values baked, not connection refs
    assert api["10"]["inputs"]["steps"] == 25
    assert api["10"]["inputs"]["end_at_step"] == 20


def test_to_api_connections_are_origin_refs(template, object_info):
    wf = Workflow.from_ui(template)
    api = wf.to_api(object_info)
    model_ref = api["10"]["inputs"]["model"]
    assert isinstance(model_ref, list)
    origin_id, origin_slot = model_ref
    assert api[origin_id]["class_type"] == "CheckpointLoaderSimple"
    assert origin_slot == 0


def test_to_api_bakes_primitive_text_into_clip_encode(template, object_info):
    wf = Workflow.from_ui(template)
    api = wf.to_api(object_info)
    texts = [
        e["inputs"]["text"]
        for e in api.values()
        if e["class_type"] == "CLIPTextEncode"
        and isinstance(e["inputs"].get("text"), str)
    ]
    assert "evening sunset scenery blue sky nature, glass bottle with a galaxy in it" in texts


def test_to_api_unknown_node_raises(object_info):
    wf = Workflow.new()
    wf.add_node("TotallyMadeUpNode", raw_widgets=[])
    with pytest.raises(ValueError, match="TotallyMadeUpNode"):
        wf.to_api(object_info)


# --- Programmatic building ---


def test_build_minimal_txt2img(object_info):
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

    wf.set_widget(pos.id, "text", "a red fox", object_info)
    wf.set_widget(sampler.id, "steps", 12, object_info)
    wf.set_widget(sampler.id, "seed", 42, object_info)

    api = wf.to_api(object_info)
    s = api[str(sampler.id)]["inputs"]
    assert s["steps"] == 12
    assert s["seed"] == 42
    assert s["model"] == [str(ckpt.id), 0]
    # defaults filled from schema
    assert s["cfg"] == 8.0
    ui = wf.to_ui()
    assert ui["version"] == 0.4
    assert any(n["type"] == "KSampler" for n in ui["nodes"])


def test_get_widget_reads_named_value(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "cfg", 3.5, object_info)
    assert wf.get_widget(sampler.id, "cfg", object_info) == 3.5


def test_connect_type_mismatch_raises(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    with pytest.raises(ValueError, match="MODEL"):
        wf.connect(ckpt.id, "MODEL", sampler.id, "positive")


def test_reconnect_replaces_existing_link(object_info):
    wf = Workflow.new()
    a = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    b = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(a.id, "MODEL", sampler.id, "model")
    wf.connect(b.id, "MODEL", sampler.id, "model")
    api_ready_links = [
        link for link in wf.links.values() if link.target_id == sampler.id and link.target_slot == 0
    ]
    assert len(api_ready_links) == 1
    assert api_ready_links[0].origin_id == b.id


def test_remove_node_drops_its_links(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.remove_node(ckpt.id)
    assert ckpt.id not in wf.nodes
    assert all(
        link.origin_id != ckpt.id and link.target_id != ckpt.id for link in wf.links.values()
    )


# --- API format import (for beautifying external workflows) ---


def test_from_api_reconstructs_graph(object_info):
    api = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "hi"}},
    }
    wf = Workflow.from_api(api, object_info)
    assert len(wf.nodes) == 2
    out = wf.to_api(object_info)
    assert out["2"]["inputs"]["clip"] == ["1", 1]
    assert out["2"]["inputs"]["text"] == "hi"
    assert out["1"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"


def test_mode_mute_excluded_from_api(template, object_info):
    wf = Workflow.from_ui(template)
    wf.nodes[19].mode = 2  # mute the SaveImage
    api = wf.to_api(object_info)
    assert str(19) not in api


def test_set_widget_error_includes_synthetic_slots(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    with pytest.raises(ValueError, match="seed__control_after_generate"):
        wf.set_widget(sampler.id, "nonexistent_widget", 42, object_info)


def test_set_widget_control_after_generate_succeeds(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "seed__control_after_generate", "randomize", object_info)
    assert wf.get_widget(sampler.id, "seed__control_after_generate", object_info) == "randomize"


def test_set_widget_pads_short_array_with_defaults_not_none(object_info):
    """Dynamic nodes serialize fewer widgets_values than the schema declares;
    padding must use schema defaults - a None here crashes the ComfyUI editor
    ("Cannot read properties of null (reading 'replace')")."""
    wf = Workflow.new()
    node = wf.add_node("KSampler", object_info=object_info)
    node.widgets_values = node.widgets_values[:2]  # simulate a short/old array
    wf.set_widget(node.id, "seed", 42, object_info)
    from comfy_draftsman.graph.widgets import widget_slot_names

    assert None not in node.widgets_values
    assert len(node.widgets_values) == len(widget_slot_names("KSampler", object_info))
    assert wf.get_widget(node.id, "seed", object_info) == 42
