"""0.24 feedback round: MOSS family, learned-shape guard, LoRA Manager saved widget
names, PreviewAny, plain-text notes, control alias, model refresh, cached runs."""

import yaml

from comfy_draftsman import knowledge, server
from comfy_draftsman.graph import annotate as an
from comfy_draftsman.graph import validate as v
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_BYPASS, Workflow

# --- learned-shape guard --------------------------------------------------------------


def test_shape_errors_name_the_bad_section():
    bad = knowledge.shape_errors({"notes": "flat string", "sampling": {"sampler": {"x": 1}}})
    assert {path for path, _ in bad} == {"notes", "sampling.sampler"}
    good = {"notes": {"sampling": "x"}, "sampling": {"cfg": {"default": 2}, "samplers": ["euler"], "notes": "ok"}}
    assert knowledge.shape_errors(good) == []


def test_a_bad_learned_section_is_dropped_not_fatal(tmp_path):
    learned = tmp_path / "learned"
    learned.mkdir()
    path = knowledge._learned_path(learned, "flux")
    path.write_text(
        yaml.safe_dump({"data": {
            "notes": "flat string",
            # a real pre-0.24 learned file had this: one bad child, good siblings
            "sampling": {"cfg": {"default": 9.5}, "scheduler": {"default": "simple"}},
        }}),
        encoding="utf-8",
    )
    g = knowledge.get_guidance("flux", learned_dir=learned)
    assert isinstance(g["notes"], dict)  # the floor's notes survive the bad overlay
    assert g["sampling"]["cfg"]["default"] == 9.5  # the bad child's siblings still merge
    assert all(isinstance(s, str) for s in g["sampling"].get("schedulers", []))


async def test_record_learning_refuses_a_bad_shape_and_saves_nothing(monkeypatch, tmp_path):
    from comfy_draftsman.config import Config

    config = Config(comfyui_url="http://comfy.test", session_dir=tmp_path / "s", learned_dir=tmp_path / "learned")
    monkeypatch.setattr(server._State, "config", config)
    out = await server.record_learning("flux", {"notes": "flat string"}, source="x")
    assert "nothing saved" in out["error"]
    assert not knowledge._learned_path(tmp_path / "learned", "flux").exists()


# --- MOSS family / avoid_nodes lint ---------------------------------------------------

MOSS_OI = {
    "UNETLoader": {
        "input": {"required": {"unet_name": [["moss_image_v1.safetensors"]], "weight_dtype": [["default"]]}},
        "output": ["MODEL"],
    },
    "ModelSamplingSD3": {
        "input": {"required": {"model": ["MODEL"], "shift": ["FLOAT", {"default": 3.0}]}},
        "output": ["MODEL"],
    },
}


def _moss_graph():
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=MOSS_OI)
    unet.widgets_values = ["moss_image_v1.safetensors", "default"]
    shift = wf.add_node("ModelSamplingSD3", object_info=MOSS_OI)
    wf.connect(unet.id, "MODEL", shift.id, "model")
    return wf, shift


def test_moss_is_detected_and_model_sampling_sd3_is_flagged(tmp_path):
    wf, shift = _moss_graph()
    assert knowledge.detect_family(wf, MOSS_OI, learned_dir=tmp_path) == "moss_image"
    found = [f for f in lint(wf, MOSS_OI, learned_dir=tmp_path) if f["code"] == "family-avoid-node"]
    assert [f["node_id"] for f in found] == [shift.id]
    shift.mode = MODE_BYPASS
    assert not [f for f in lint(wf, MOSS_OI, learned_dir=tmp_path) if f["code"] == "family-avoid-node"]


# --- PreviewAny counts as a prompt preview --------------------------------------------


def test_preview_any_is_a_text_display():
    from comfy_draftsman.graph.layout import is_text_display

    assert is_text_display("PreviewAny")


# --- LoRA Manager saved widget names --------------------------------------------------

LM = "Lora Loader (LoraManager)"
LM_OI = {
    LM: {
        "input": {
            "required": {
                "model": ["MODEL"],
                "text": ["AUTOCOMPLETE_TEXT_LORAS", {}],
                "loras": ["LORAS", {}],
            }
        },
        "input_order": {"required": ["model", "text", "loras"]},
        "output": ["MODEL", "CLIP", "STRING", "STRING"],
        "output_name": ["MODEL", "CLIP", "trigger_words", "loaded_loras"],
    }
}


def _lm_workflow():
    ui = {
        "version": 0.4,
        "last_node_id": 4,
        "last_link_id": 0,
        "links": [],
        "groups": [],
        "nodes": [
            {
                "id": 4, "type": LM, "pos": [0, 0], "size": [370, 430], "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "text", "type": "AUTOCOMPLETE_TEXT_LORAS", "widget": {"name": "text"}, "link": None},
                    {"name": "loras", "type": "LORAS", "widget": {"name": "loras"}, "link": None},
                ],
                "outputs": [],
                "properties": {"__lm_widget_ids": ["__lm_autocomplete_meta_text", "text", "loras"]},
                "widgets_values": [
                    {"version": 1, "textWidgetName": "text"},
                    "<lora:a:0.8>",
                    [{"name": "a", "strength": 0.8, "active": True}],
                ],
            }
        ],
    }
    return Workflow.from_ui(ui)


def test_lora_manager_widgets_set_by_saved_name_and_reach_the_api():
    wf = _lm_workflow()
    wf.set_widget(4, "text", "<lora:b:1>", LM_OI)
    wf.set_widget(4, "loras", [{"name": "b", "strength": 1.0, "active": True}], LM_OI)
    node = wf.nodes[4]
    assert node.widgets_values[0] == {"version": 1, "textWidgetName": "text"}  # meta untouched
    inputs = wf.to_api(LM_OI)["4"]["inputs"]
    assert inputs["text"] == "<lora:b:1>"
    assert inputs["loras"] == {"__value__": [{"name": "b", "strength": 1.0, "active": True}]}


def test_lora_manager_saved_names_skip_the_positional_drift_errors():
    wf = _lm_workflow()
    codes = {f["code"] for f in v.validate(wf, LM_OI) if f["level"] == "error"}
    assert not codes & {"widget-count-mismatch", "widget-layout-unmapped", "js-widget-input"}


# --- control_after_generate alias -----------------------------------------------------


def test_control_after_generate_alias_sets_the_seed_control():
    oi = {"KSampler": {"input": {"required": {"seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}]}},
                       "output": ["LATENT"]}}
    wf = Workflow.new()
    node = wf.add_node("KSampler", object_info=oi)
    wf.set_widget(node.id, "control_after_generate", "fixed", oi)
    assert "fixed" in node.widgets_values


# --- the size note points at the node that sets the size ------------------------------


def test_size_note_names_the_wired_resolution_node():
    oi = {
        "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}]}}, "output": ["LATENT"]},
        "Res": {"input": {"required": {}}, "output": ["INT", "INT"], "output_name": ["width", "height"],
                "display_name": "Resolution Picker"},
    }
    wf = Workflow.new()
    canvas = wf.add_node("EmptyLatentImage", object_info=oi)
    res = wf.add_node("Res", object_info=oi)
    canvas.widgets_values = [1024, 1024]
    wf.connect(res.id, "width", canvas.id, "width", object_info=oi)
    assert an._size_sources(wf, [canvas], oi) == ["Resolution Picker"]


# --- model list refresh on edit, cached sweep runs ------------------------------------


def test_an_unlisted_model_file_in_an_edit_triggers_a_refresh():
    oi = {"UNETLoader": {"input": {"required": {"unet_name": [["old.safetensors"]]}}, "output": ["MODEL"]}}
    ops = [{"op": "set_widget", "node_id": 1, "input": "unet_name", "value": "new.safetensors"}]
    assert server._names_unlisted_file(ops, oi)
    ops[0]["value"] = "old.safetensors"
    assert not server._names_unlisted_file(ops, oi)
    assert not server._names_unlisted_file([{"op": "set_widget", "value": 7}], oi)


def test_cached_sampler_is_reported_and_popped():
    api = {"3": {"class_type": "KSampler"}, "9": {"class_type": "SaveImage"}}
    res = {"cached_nodes": ["3"]}
    assert server._sampling_cached(res, api) and "cached_nodes" not in res
    assert not server._sampling_cached({"cached_nodes": ["9"]}, api)
    assert not server._sampling_cached({}, api)
