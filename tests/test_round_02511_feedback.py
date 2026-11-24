"""0.25.11: photo-angles session feedback - LoadImage subfolder paths, sweep axes,
dead-end lint, unknown API input keys, multi-word node search, new/updated floors."""

from __future__ import annotations

from comfy_draftsman import knowledge, server
from comfy_draftsman.comfy.catalog import search_nodes
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import check_widget_value

OI = {
    "LoadImage": {
        "input": {"required": {"image": [["root.png"], {"image_upload": True}]}},
        "output": ["IMAGE", "MASK"],
    },
    "UNETLoader": {"input": {"required": {"unet_name": [["m.safetensors"]]}}, "output": ["MODEL"]},
    "LoraLoaderModelOnly": {
        "input": {"required": {"model": ["MODEL"], "lora_name": [["l.safetensors"]], "strength_model": ["FLOAT", {}]}},
        "output": ["MODEL"],
    },
    "PreviewImage": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
    "MediaPipeFaceLandmarker": {"input": {"required": {}}, "output": ["BBOX"], "display_name": "Face Landmarker"},
    "CropByBBoxes": {"input": {"required": {}}, "output": ["IMAGE"], "display_name": "Crop By BBoxes"},
}


def test_loadimage_accepts_subfolder_path():
    assert check_widget_value("LoadImage", "image", "abtest/x.jpg", OI) is None
    assert check_widget_value("LoadImage", "image", "x.png [output]", OI) is None
    assert check_widget_value("LoadImage", "image", "../x.jpg", OI) is not None
    assert check_widget_value("LoadImage", "image", "missing.jpg", OI) is not None


def test_sweep_axes_build_the_cartesian_product():
    a = {"op": "set_widget", "node_id": 1, "input": "text", "value": "A"}
    b = {"op": "set_widget", "node_id": 2, "input": "image", "value": "b.png"}
    spec = server._sweep_spec({"axes": {
        "subject": [{"label": "s1", "ops": [a]}, {"label": "s2", "ops": [b]}],
        "shot": [{"label": "front", "ops": []}, {"ops": [a]}],
    }})
    assert not isinstance(spec, str)
    variants = spec[0]
    assert [v["label"] for v in variants] == ["s1 / front", "s1 / shot2", "s2 / front", "s2 / shot2"]
    assert variants[3]["ops"] == [b, a]
    assert "not both" in server._sweep_spec({"axes": {"x": [{"ops": []}]}, "variants": []})
    assert "over 24" in server._sweep_spec({"axes": {"x": [{"ops": []}] * 5, "y": [{"ops": []}] * 5}})
    assert isinstance(server._sweep_spec({"axes": {"x": []}}), str)


def test_dead_end_node_is_linted():
    wf = Workflow()
    unet = wf.add_node("UNETLoader", object_info=OI)
    lora = wf.add_node("LoraLoaderModelOnly", object_info=OI)
    wf.connect(unet.id, 0, lora.id, "model")
    codes = {(f["code"], f.get("node_id")) for f in lint(wf, OI)}
    assert ("dead-end-node", lora.id) in codes
    assert ("dead-end-node", unet.id) not in codes


def test_unknown_api_input_keys_are_reported():
    oi = {"TextEncode": {"input": {"required": {"text": ["STRING", {}]}, "optional": {"images": ["*"]}}, "output": ["X"]}}
    api = {"5": {"class_type": "TextEncode", "inputs": {"text": "a", "resolution": 1024, "images.image_2": ["4", 0]}}}
    assert server._unknown_api_inputs(api, oi) == ["TextEncode #5: 'resolution'"]


def test_multi_word_search_adds_any_word_hits_core_first():
    oi = {
        **OI,
        "PackFaceCrop": {"description": "smart face crop", "python_module": "custom_nodes.pack"},
        "PackCropper": {"description": "crop", "python_module": "custom_nodes.pack"},
    }
    hits = [h["class_type"] for h in search_nodes(oi, "face crop")]
    assert hits[0] == "PackFaceCrop"  # the exact phrase still leads
    assert set(hits[1:3]) == {"MediaPipeFaceLandmarker", "CropByBBoxes"} and hits[3] == "PackCropper"
    assert search_nodes(OI, "zebra unicorn") == []


def test_floors():
    assert knowledge.get_guidance("sensenova_u16")["sampling"]["steps"]["default"] == 50
    h3 = knowledge.get_guidance("minimax_h3_image")
    assert h3["native_resolution"]["max_mp"] == 40 and "edit_resolution" in h3["notes"]
    assert "edit" in knowledge.get_guidance("fibo2")["notes"]
