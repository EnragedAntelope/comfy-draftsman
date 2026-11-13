"""0.25.9: MOSS session feedback - a bypassed source on an optional input is a
warning, the CLIP loader's type detects the family, encoder-only files feeding
TextGenerate are linted, and set_widget reads a value from a file."""

from __future__ import annotations

import pytest

from comfy_draftsman import knowledge, server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.config import Config
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, Workflow
from comfy_draftsman.graph.validate import validate
from comfy_draftsman.graph.widgets import named_for_api, widgets_to_named

OI = {
    "LoadImage": {
        "input": {"required": {"image": [["a.png"], {}]}},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
    },
    "CLIPLoader": {
        "input": {"required": {
            "clip_name": [["moss_te.safetensors", "moss_full.safetensors", "x.safetensors"], {}],
            "type": [["moss_image", "sdxl"], {}],
        }},
        "output": ["CLIP"],
        "output_name": ["CLIP"],
    },
    "Gen": {
        "input": {
            "required": {"clip": ["CLIP", {}]},
            "optional": {"image": ["IMAGE", {}]},
        },
        "output": ["STRING"],
        "output_name": ["STRING"],
    },
}


def _graph(image_mode: int = 0, clip_name: str = "x.safetensors", gen: str = "Gen"):
    info = {**OI, gen: OI["Gen"]}
    wf = Workflow.new()
    img = wf.add_node("LoadImage", object_info=info)
    clip = wf.add_node("CLIPLoader", raw_widgets=[clip_name, "moss_image"], object_info=info)
    node = wf.add_node(gen, object_info=info)
    wf.connect(clip.id, 0, node.id, "clip", info)
    wf.connect(img.id, 0, node.id, "image", info)
    img.mode = image_mode
    return wf, info, node


def test_bypassed_source_on_optional_input_is_a_warning():
    wf, info, node = _graph(MODE_BYPASS)
    dead = [f for f in validate(wf, info) if f["code"] == "dead-input-source"]
    assert [f["level"] for f in dead] == ["warning"]
    assert "image" not in wf.to_api(info)[str(node.id)]["inputs"]


@pytest.mark.parametrize(("live", "underfilled"), [(True, False), (False, True)])
def test_dead_autogrow_socket_does_not_fill_min(live, underfilled):
    # Batch.images needs min 1: a bypassed LoadImage on image0 alone leaves it
    # empty in the prompt, so the container error must still fire
    info = {**OI, "Batch": {
        "input": {"required": {"images": ["COMFY_AUTOGROW_V3", {"template": {
            "input": {"required": {"image": ["IMAGE", {}]}}, "prefix": "image", "min": 1, "max": 4}}]}},
        "output": ["IMAGE"], "output_name": ["IMAGE"],
    }}
    wf = Workflow.new()
    dead = wf.add_node("LoadImage", object_info=info)
    batch = wf.add_node("Batch", object_info=info)
    wf.connect(dead.id, 0, batch.id, "images.image0", info)
    dead.mode = MODE_BYPASS
    if live:
        wf.connect(wf.add_node("LoadImage", object_info=info).id, 0, batch.id, "images.image1", info)
    findings = validate(wf, info)
    assert any(f["code"] == "autogrow-underfilled" for f in findings) is underfilled
    assert not any(f["code"] == "dead-input-source" and f["level"] == "error" for f in findings)


def test_clip_loader_type_detects_the_family():
    wf, info, _ = _graph()
    detail = knowledge.detect_family_detail(wf, info)
    assert detail["family"] == "moss_image" and detail["widget"] == "type"
    # a type no family claims in detect.clip_types detects nothing, even if it
    # equals a family name: shared types (sdxl, flux2, wan) would only guess
    wf.nodes[2].widgets_values[1] = "sdxl"
    assert knowledge.detect_family(wf, info) is None


@pytest.mark.parametrize(("name", "flagged"), [
    ("moss_te.safetensors", True),
    ("moss_full.safetensors", False),
    ("x.safetensors", False),
])
def test_text_only_encoder_feeding_textgenerate(name, flagged):
    wf, info, _ = _graph(clip_name=name, gen="TextGenerate")
    codes = {f["code"] for f in lint(wf, info)}
    assert ("text-only-encoder-generate" in codes) is flagged


TEXT_GENERATE = {"input": {
    "required": {
        "clip": ["CLIP", {}],
        "prompt": ["STRING", {"multiline": True}],
        "max_length": ["INT", {"default": 512}],
        "sampling_mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "on", "inputs": {
                "required": {
                    "temperature": ["FLOAT", {}], "top_k": ["INT", {}], "top_p": ["FLOAT", {}],
                    "min_p": ["FLOAT", {}], "repetition_penalty": ["FLOAT", {}], "seed": ["INT", {}],
                },
                "optional": {"presence_penalty": ["FLOAT", {}]},
            }},
            {"key": "off", "inputs": {"required": {}}},
        ]}],
    },
    "optional": {"thinking": ["BOOLEAN", {}], "use_default_template": ["BOOLEAN", {}]},
}, "output": ["STRING", "STRING"]}


def test_nested_seed_has_no_control_slot():
    # widgets_values exactly as ComfyUI's bundled Krea-2 Turbo blueprint saves a
    # TextGenerate: no control_after_generate value after sampling_mode.seed
    saved = ["", 512, "on", 0.7, 64, 0.95, 0.05, 1.05, 7, 0, False, True]
    named = widgets_to_named("TextGenerate", saved, {"TextGenerate": TEXT_GENERATE})
    assert named["sampling_mode.seed"] == 7
    assert (named["thinking"], named["use_default_template"]) == (False, True)


async def test_node_schemas_refresh_only_the_requested_classes(monkeypatch):
    client = ComfyClient(Config(comfyui_url="http://x"))
    client._object_info = {"CLIPLoader": {"input": {"required": {"clip_name": [["old_te.safetensors"], {}]}}},
                           "KSampler": {"stale": True}}
    fresh = {"CLIPLoader": {"input": {"required": {"clip_name": [["new.safetensors"], {}]}}}}
    calls = []

    async def fake_get_json(path):
        calls.append(path)
        return fresh if path == "/object_info/CLIPLoader" else {}

    monkeypatch.setattr(client, "_get_json", fake_get_json)
    info = await client.get_node_schemas(["CLIPLoader", "NoSuchNode", "CLIPLoader"])
    assert calls == ["/object_info/CLIPLoader", "/object_info/NoSuchNode"]
    assert info["CLIPLoader"] == fresh["CLIPLoader"]
    assert info["KSampler"] == {"stale": True} and "NoSuchNode" not in info


def test_old_save_gets_missing_required_widget_defaults():
    # saved before refine_steps (required) and extra (optional) existed: the
    # browser fills required defaults, and /prompt rejects the run without them
    info = {"Moge": {"input": {
        "required": {"level": ["INT", {"default": 9}], "refine_steps": ["INT", {"default": 3}]},
        "optional": {"extra": ["INT", {"default": 5}]},
    }, "output": []}}
    named = named_for_api("Moge", [7], info)
    assert named == {"level": 7, "refine_steps": 3}


@pytest.mark.parametrize(("saved", "index"), [
    (["Video Editing", 0, "Default", "Text to Image", "Video Editing", ""], 2),
    (["Music", 0, "Music", "SFX", ""], 0),
    (["gone", 3, "a", "b", ""], -1),
])
def test_custom_combo_sends_the_frontend_index(saved, index):
    info = {"CustomCombo": {"input": {"required": {"choice": ["COMBO", {"options": []}]}}, "output": ["STRING", "INT"]}}
    named = named_for_api("CustomCombo", saved, info)
    assert (named["choice"], named["index"]) == (saved[0], index)


def test_set_widget_reads_value_from_file(tmp_path):
    wf, info, _ = _graph()
    prompt = tmp_path / "p.txt"
    prompt.write_text("long system prompt\nline 2", encoding="utf-8")
    server._apply_ops(wf, [{"op": "set_widget", "node_id": 2, "input": "type", "force": True, "file_path": str(prompt)}], info, [], set())
    assert wf.nodes[2].widgets_values[1] == "long system prompt\nline 2"
    with pytest.raises(ValueError, match="not both"):
        server._apply_ops(wf, [{"op": "set_widget", "node_id": 2, "input": "type", "value": "a", "file_path": str(prompt)}], info, [], set())
