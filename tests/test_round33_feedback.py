"""0.25 feedback round: headless LoRA Manager loader, off-step edits, unwrapped
text outputs, record_learning shape errors, sampler timing, banding score,
crop guide, FIBO family and the above-native-resolution lint."""

import math

from PIL import Image

from comfy_draftsman import knowledge, server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph import validate as v
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, Workflow
from comfy_draftsman.imaging import banding_score, crop_guide

# --- LoRA Manager: a fresh loader runs headlessly, text and loras stay in sync -------

LM = "Lora Loader (LoraManager)"
LM_OI = {
    LM: {
        "input": {
            "required": {
                "model": ["MODEL"],
                "text": ["AUTOCOMPLETE_TEXT_LORAS", {}],
                "loras": ["LORAS", {}],
            },
            "optional": {},
        },
        "output": ["MODEL", "CLIP", "STRING", "STRING"],
        "output_name": ["MODEL", "CLIP", "trigger_words", "loaded_loras"],
        "python_module": "custom_nodes.ComfyUI-Lora-Manager",
    }
}


def test_fresh_lm_loader_runs_from_its_text():
    wf = Workflow.new()
    node = wf.add_node(LM, object_info=LM_OI)
    wf.set_widget(node.id, "text", "portrait <lora:skin:0.8> <lora:light:0.5:0.3>", LM_OI)
    loras = node.widgets_values[2]
    assert [(e["name"], e["strength"], e["clipStrength"], e["active"]) for e in loras] == [
        ("skin", 0.8, 0.8, True),
        ("light", 0.5, 0.3, True),
    ]
    assert not [f for f in v.validate(wf, LM_OI) if f["code"] == "js-widget-input"]
    inputs = wf.to_api(LM_OI)[str(node.id)]["inputs"]
    assert inputs["loras"]["__value__"] == loras  # the list is what LoRA Manager runs


def test_text_edit_keeps_entry_state_and_drops_removed_loras():
    wf = Workflow.new()
    node = wf.add_node(LM, object_info=LM_OI)
    node.widgets_values[2] = [
        {"name": "skin", "strength": 0.8, "active": False, "expanded": True},
        {"name": "gone", "strength": 1.0, "active": True},
    ]
    wf.set_widget(node.id, "text", "<lora:skin:0.6>", LM_OI)
    assert node.widgets_values[2] == [
        {"name": "skin", "strength": 0.6, "active": False, "expanded": True, "clipStrength": 0.6}
    ]


def test_setting_loras_adds_missing_tags_and_strips_last_accepted():
    wf = Workflow.new()
    node = wf.add_node(LM, object_info=LM_OI)
    node.widgets_values[0] = {"version": 1, "textWidgetName": "text", "lastAccepted": {"start": 0, "end": 9}}
    node.widgets_values[1] = "a cat"
    wf.set_widget(node.id, "loras", [{"name": "fur", "strength": 0.7, "active": True}], LM_OI)
    assert node.widgets_values[1] == "a cat <lora:fur:0.7>"  # the browser rebuilds the list from text
    assert node.widgets_values[0] == {"version": 1, "textWidgetName": "text"}


# --- off-step values are accepted at edit time -----------------------------------------


def test_off_step_value_is_accepted_and_noted_as_info():
    oi = {"ScaleROPE": {"input": {"required": {
        "model": ["MODEL"], "scale_x": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}]}},
        "output": ["MODEL"]}}
    assert v.check_widget_value("ScaleROPE", "scale_x", 0.85, oi) is None
    wf = Workflow.new()
    node = wf.add_node("ScaleROPE", object_info=oi)
    wf.set_widget(node.id, "scale_x", 0.85, oi)
    found = [f for f in v.validate(wf, oi) if f["code"] == "step-misaligned"]
    assert found and found[0]["level"] == "info"


# --- text outputs read as text, JSON answers as JSON ---------------------------------


def test_show_text_output_is_unwrapped_and_json_parsed():
    history = {"outputs": {
        "5": {"text": ['{"short_description": "a fox", "objects": []}']},
        "6": {"text": ["plain words"]},
        "7": {"filenames": ["one.png"]},
    }}
    data = ComfyClient._collect_data_outputs(history)
    assert data["5"]["text"] == {"short_description": "a fox", "objects": []}
    assert list(data["5"]["text"]) == ["short_description", "objects"]  # key order kept
    assert data["6"]["text"] == "plain words"
    assert data["7"]["filenames"] == ["one.png"]


def test_unwired_llm_troubleshooting_output_is_flagged_once():
    oi = {
        "EA_LMStudio": {"input": {"required": {"prompt": ["STRING", {"multiline": True}]}},
                        "output": ["STRING", "STRING", "STRING"],
                        "output_name": ["response", "reasoning", "troubleshooting"]},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}],
                                                  "clip": ["CLIP"]}}, "output": ["CONDITIONING"]},
        "ShowText|pys": {"input": {"required": {"text": ["STRING", {"forceInput": True}]}},
                         "output": ["STRING"]},
        "KSampler": {"input": {"required": {"positive": ["CONDITIONING"], "negative": ["CONDITIONING"]}},
                     "output": ["LATENT"]},
    }
    wf = Workflow.new()
    llm = wf.add_node("EA_LMStudio", object_info=oi)
    show = wf.add_node("ShowText|pys", object_info=oi)
    sampler = wf.add_node("KSampler", object_info=oi)
    for role in ("positive", "negative"):  # two encoders off one generator
        enc = wf.add_node("CLIPTextEncode", object_info=oi)
        wf.connect(llm.id, "response", enc.id, "text", oi)
        wf.connect(enc.id, "CONDITIONING", sampler.id, role, oi)
    wf.connect(llm.id, "response", show.id, "text", oi)

    def hidden():
        return [f for f in lint(wf, oi) if f["code"] == "hidden-troubleshooting"]

    assert [f["node_id"] for f in hidden()] == [llm.id]
    wf.connect(llm.id, "troubleshooting", show.id, "text", oi)
    assert not hidden()


# --- record_learning teaches its shape ---------------------------------------------------


def test_shape_errors_name_unread_detect_keys_and_sampling_knobs():
    errors = dict(knowledge.shape_errors({"detect": {"unet": ["fibo"], "checkpoint_patterns": ["fibo"]},
                                          "sampling": {"default": 8}}))
    assert "checkpoint_patterns" in errors["detect.unet"]
    assert "cfg" in errors["sampling.default"]


async def test_record_learning_without_updates_returns_the_shape(monkeypatch, tmp_path):
    from comfy_draftsman.config import Config

    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path, learned_dir=tmp_path / "l"))
    out = await server.record_learning("fibo2")
    assert "nothing saved" in out["error"] and "updates" in out["example"]
    assert not (tmp_path / "l").exists()


# --- sampler timing ------------------------------------------------------------------


def test_sampling_seconds_come_from_step_ticks_only():
    api = {"3": {"class_type": "KSampler"}, "8": {"class_type": "VAEDecodeTiled"}}
    # model load before step 1 (t=0..10) is excluded: 2 s/step x 20 steps
    ticks = {"3": [(10.0, 1, 20), (48.0, 20, 20)], "8": [(50.0, 1, 4), (90.0, 4, 4)]}
    assert ComfyClient._sampling_s(ticks, api) == 40.0
    assert ComfyClient._sampling_s({}, api) is None


# --- banding score -----------------------------------------------------------------


def _image(fn, size=512):
    img = Image.new("L", (size, size))
    img.putdata([max(0, min(255, round(fn(x, y)))) for y in range(size) for x in range(size)])
    return img


def test_banding_score_sees_grid_stripes_not_gradients_or_texture():
    clean = banding_score(_image(lambda x, y: 60 + 120 * y / 512))
    striped = banding_score(_image(lambda x, y: 128 + 2 * math.sin(y * 2 * math.pi / 8)))
    assert clean == 0.0
    assert striped is not None and striped > 1.0  # RMS of a 2-level sine is 1.41
    # texture everywhere: nothing smooth to judge
    assert banding_score(_image(lambda x, y: 128 + 60 * ((x * 7919 + y * 104729) % 13 - 6) / 6)) is None


def test_crop_guide_is_a_small_thumbnail():
    guide = crop_guide(Image.new("RGB", (2048, 1024), (90, 90, 90)))
    assert guide.size == (384, 192)


# --- FIBO family and the above-native-resolution lint ---------------------------------

FIBO_OI = {
    "UNETLoader": {"input": {"required": {"unet_name": [["fibo2_dev_bf16.safetensors"]],
                                          "weight_dtype": [["default"]]}}, "output": ["MODEL"]},
    "TextEncodeFibo2": {"input": {"required": {
        "prompt": ["STRING", {"multiline": True}],
        "width": ["INT", {"default": 0, "min": 0, "max": 16384, "step": 16}],
        "height": ["INT", {"default": 0, "min": 0, "max": 16384, "step": 16}]}},
        "output": ["CONDITIONING", "CONDITIONING", "LATENT"]},
    "ScaleROPE": {"input": {"required": {"model": ["MODEL"]}}, "output": ["MODEL"]},
}


def _fibo_graph(width, height):
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=FIBO_OI)
    unet.widgets_values = ["fibo2_dev_bf16.safetensors", "default"]
    enc = wf.add_node("TextEncodeFibo2", object_info=FIBO_OI)
    enc.widgets_values = ["a market", width, height]
    return wf, enc


def _native(wf, tmp_path):
    return [f for f in lint(wf, FIBO_OI, learned_dir=tmp_path) if f["code"] == "above-native-resolution"]


def test_fibo_above_1mp_without_scale_rope_is_flagged(tmp_path):
    wf, enc = _fibo_graph(2048, 2048)
    assert knowledge.detect_family(wf, FIBO_OI, learned_dir=tmp_path) == "fibo2"
    found = _native(wf, tmp_path)
    assert [f["node_id"] for f in found] == [enc.id] and "0.50" in found[0]["message"]
    rope = wf.add_node("ScaleROPE", object_info=FIBO_OI)
    assert not _native(wf, tmp_path)
    rope.mode = MODE_BYPASS  # a bypassed fix is no fix
    assert _native(wf, tmp_path)


def test_fibo_native_and_default_sizes_stay_quiet(tmp_path):
    for size in ((1248, 832), (0, 0)):
        wf, _ = _fibo_graph(*size)
        assert not _native(wf, tmp_path)


def test_flux3_klein_and_minimax_h3_floors(tmp_path):
    oi = {"UNETLoader": FIBO_OI["UNETLoader"]}
    for name, family in (("flux3-image-klein-int8-convrot.safetensors", "flux3_klein"),
                         ("flux-2-klein-9b.safetensors", "flux"),  # FLUX.2 Klein is not FLUX 3
                         ("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3")):
        wf = Workflow.new()
        wf.add_node("UNETLoader", object_info=oi).widgets_values = [name, "default"]
        assert knowledge.detect_family(wf, oi, learned_dir=tmp_path) == family, name
    klein = knowledge.get_guidance("flux3_klein", "flux3-image-klein-bf16.safetensors", learned_dir=tmp_path)
    base = knowledge.get_guidance("flux3_klein", "flux3-image-klein-base-bf16.safetensors", learned_dir=tmp_path)
    assert (klein["sampling"]["cfg"], klein["sampling"]["steps"]["default"]) == ({"default": 1}, 16)
    assert (base["sampling"]["cfg"], base["sampling"]["steps"]["default"]) == ({"default": 3.5}, 50)


def test_native_resolution_without_fix_node_gives_no_scale_factor(tmp_path):
    oi = {"UNETLoader": FIBO_OI["UNETLoader"], "EmptyMiniMaxH3LatentAV": {"input": {"required": {
        "width": ["INT", {"default": 1344}], "height": ["INT", {"default": 768}],
        "length": ["INT", {"default": 124}]}}, "output": ["LATENT"]}}
    wf = Workflow.new()
    wf.add_node("UNETLoader", object_info=oi).widgets_values = ["minimax_h3_fl2va_bf16.safetensors", "default"]
    canvas = wf.add_node("EmptyMiniMaxH3LatentAV", object_info=oi)

    def native():
        return [f for f in lint(wf, oi, learned_dir=tmp_path) if f["code"] == "above-native-resolution"]

    assert not native()  # 1344x768 is H3's native canvas
    canvas.widgets_values = [1920, 1088, 124]
    found = native()
    assert found and "sqrt" not in found[0]["message"]


def test_fibo_variants_split_dev_and_turbo():
    dev = knowledge.get_guidance("fibo2", "fibo2_dev_bf16.safetensors")
    turbo = knowledge.get_guidance("fibo2", "fibo2_edit_turbo_bf16.safetensors")
    assert dev["sampling"]["cfg"]["default"] == 5 and turbo["sampling"]["cfg"]["default"] == 1
