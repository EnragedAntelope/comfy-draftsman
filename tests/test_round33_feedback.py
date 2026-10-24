"""0.25 feedback round: headless LoRA Manager loader, off-step edits, unwrapped
text outputs, record_learning shape errors, sampler timing, banding score,
crop guide, FIBO family and the above-native-resolution lint."""

import math
import random

from PIL import Image

from comfy_draftsman import knowledge, server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph import validate as v
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, Workflow
from comfy_draftsman.imaging import BANDING_VISIBLE, banding_score, crop_guide

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


def test_display_only_preview_widget_takes_no_saved_value():
    oi = {"ResolutionSelector": {"input": {
        "required": {"aspect_ratio": ["COMBO", {"options": ["1:1 (Square)", "2:3 (Portrait Photo)"]}],
                     "megapixels": ["FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0}],
                     "multiple": ["INT", {"default": 8, "min": 8, "max": 128}]},
        "optional": {"preview": ["RESOLUTION_PREVIEW", {"socketless": True}]}},
        "output": ["INT", "INT"]}}
    wf = Workflow.new()
    node = wf.add_node("ResolutionSelector", object_info=oi)
    assert len(node.widgets_values) == 3  # what the frontend saves
    node.widgets_values = ["2:3 (Portrait Photo)", 2, 16]
    assert not [f for f in v.validate(wf, oi) if f["code"] == "widget-count-drift"]
    assert "preview" not in wf.to_api(oi)[str(node.id)]["inputs"]


def test_image_compare_view_is_not_saved_but_is_sent_to_the_api():
    # ComfyUI rejects an ImageCompare prompt without compare_view (checked live)
    oi = {"ImageCompare": {"input": {
        "required": {"compare_view": ["IMAGECOMPARE", {"socketless": True}]},
        "optional": {"image_a": ["IMAGE"], "image_b": ["IMAGE"]}},
        "output": [], "output_node": True}}
    wf = Workflow.new()
    node = wf.add_node("ImageCompare", object_info=oi)
    assert node.widgets_values == []  # what the frontend saves
    assert wf.to_api(oi)[str(node.id)]["inputs"] == {"compare_view": {"__value__": ["", ""]}}


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


def _render(stripe=lambda x, y: (0, 0, 0), size=512):
    """A grainy sky gradient, plus a per-pixel (r, g, b) offset."""
    rnd = random.Random(0)
    img = Image.new("RGB", (size, size))
    px = []
    for y in range(size):
        for x in range(size):
            base, grain, (dr, dg, db) = 90 + 120 * y / size, rnd.gauss(0, 4), stripe(x, y)
            px.append(tuple(max(0, min(255, round(c))) for c in (base + grain + dr, base + 30 + grain + dg, 220 + grain + db)))
    img.putdata(px)
    return img


def test_banding_score_finds_color_and_brightness_stripes_not_grain():
    clean, _ = banding_score(_render())
    assert clean < BANDING_VISIBLE
    # MOSS-Image SFT style: faint vertical color bands, brightness untouched
    wave = lambda x: 1.5 * math.sin(x * 2 * math.pi / 16)  # noqa: E731
    color, where = banding_score(_render(lambda x, y: (wave(x), -wave(x), wave(x))))
    assert color >= BANDING_VISIBLE and where == "vertical 16px"
    luma, where = banding_score(_render(lambda x, y: (wave(y * 16 / 24),) * 3))
    assert luma >= BANDING_VISIBLE and where == "horizontal 24px"
    # 8px is every VAE's normal texture, out of range on purpose
    vae, _ = banding_score(_render(lambda x, y: (wave(x * 2),) * 3))
    assert vae < BANDING_VISIBLE


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
    "ScaleROPE": {"input": {"required": {
        "model": ["MODEL"],
        **{k: ["FLOAT", {"default": d, "min": -100.0, "max": 100.0, "step": 0.1}]
           for k, d in (("scale_x", 1.0), ("shift_x", 0.0), ("scale_y", 1.0), ("shift_y", 0.0))}}},
        "output": ["MODEL"]},
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
    assert _native(wf, tmp_path)  # at its defaults (scale 1.0) ScaleROPE does nothing
    rope.widgets_values = [0.5, 0.0, 0.5, 0.0]
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


def test_anima_floor_and_turbo_variant_without_claiming_sdxl_anime(tmp_path):
    oi = {"UNETLoader": FIBO_OI["UNETLoader"]}
    for name, family in (("anima-turbo-v1.1.safetensors", "anima"),
                         ("anima-aesthetic-v1.0b.safetensors", "anima"),
                         ("animagine-xl-4.0.safetensors", None),
                         ("anima_pencil-XL-v5.safetensors", None)):
        wf = Workflow.new()
        wf.add_node("UNETLoader", object_info=oi).widgets_values = [name, "default"]
        assert knowledge.detect_family(wf, oi, learned_dir=tmp_path) == family, name
    turbo = knowledge.get_guidance("anima", "anima-turbo-v1.1.safetensors", learned_dir=tmp_path)
    base = knowledge.get_guidance("anima", "anima-base-v1.0.safetensors", learned_dir=tmp_path)
    assert (turbo["sampling"]["cfg"], turbo["sampling"]["steps"]) == ({"default": 1}, {"default": 8})
    assert (base["sampling"]["cfg"], base["sampling"]["steps"]) == ({"default": 4}, {"default": 30})


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
