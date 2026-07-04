"""Porting workflows across model families (e.g. SDXL -> FLUX/Krea)."""

import copy
import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.port import port_workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _build_sdxl_txt2img(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "cfg", 6.0, object_info)
    wf.set_widget(sampler.id, "steps", 28, object_info)
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
    return wf, {"ckpt": ckpt.id, "sampler": sampler.id, "latent": latent.id, "decode": decode.id}


def test_port_sdxl_to_flux_retunes_sampler(object_info):
    wf, ids = _build_sdxl_txt2img(object_info)
    report = port_workflow(wf, "flux", object_info)
    assert wf.get_widget(ids["sampler"], "cfg", object_info) == 1.0
    assert wf.get_widget(ids["sampler"], "sampler_name", object_info) == "euler"
    assert wf.get_widget(ids["sampler"], "scheduler", object_info) == "simple"
    assert report["target_family"] == "flux"
    assert any("cfg" in c for c in report["changes"])


def test_port_sdxl_to_flux_swaps_loader_topology(object_info):
    wf, ids = _build_sdxl_txt2img(object_info)
    port_workflow(wf, "flux", object_info)
    types = {n.type for n in wf.nodes.values()}
    assert "CheckpointLoaderSimple" not in types
    assert "UNETLoader" in types
    assert "DualCLIPLoader" in types
    assert "VAELoader" in types
    # consumers stay wired: sampler.model, decode.vae, both text encoders' clip
    api_ok = wf.to_api(object_info)  # raises if graph is broken
    sampler_inputs = api_ok[str(ids["sampler"])]["inputs"]
    assert isinstance(sampler_inputs["model"], list)
    unet_id = sampler_inputs["model"][0]
    assert api_ok[unet_id]["class_type"] == "UNETLoader"
    assert api_ok[str(ids["decode"])]["inputs"]["vae"][0] != str(ids["ckpt"])
    dual = next(v for v in api_ok.values() if v["class_type"] == "DualCLIPLoader")
    assert dual["inputs"]["type"] == "flux"


def test_port_swaps_latent_node_class(object_info):
    wf, ids = _build_sdxl_txt2img(object_info)
    port_workflow(wf, "flux", object_info)
    assert wf.nodes[ids["latent"]].type == "EmptySD3LatentImage"


def test_port_flags_when_no_matching_model_file(object_info):
    # strip flux-looking files from the UNETLoader choices so selection must flag
    oi = copy.deepcopy(object_info)
    spec = oi["UNETLoader"]["input"]["required"]["unet_name"]
    spec[0] = [f for f in spec[0] if "flux" not in f.lower() and "krea" not in f.lower()]
    wf, _ = _build_sdxl_txt2img(oi)
    report = port_workflow(wf, "flux", oi)
    assert any("unet" in flag.lower() or "model file" in flag.lower() for flag in report["flags"])


def test_port_applies_technique_settings_to_detailer(object_info):
    oi = copy.deepcopy(object_info)
    # realistic subset of Impact Pack's FaceDetailer schema
    oi["FaceDetailer"] = {
        "input": {
            "required": {
                "image": ["IMAGE", {}],
                "model": ["MODEL", {}],
                "guide_size": ["FLOAT", {"default": 512.0, "min": 64, "max": 8192}],
                "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                "sampler_name": [["euler", "dpmpp_2m"], {}],
                "scheduler": [["normal", "simple", "karras"], {}],
                "denoise": ["FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["image"],
        "category": "ImpactPack/Detailer",
        "output_node": False,
    }
    wf, _ = _build_sdxl_txt2img(oi)
    detailer = wf.add_node("FaceDetailer", object_info=oi)
    report = port_workflow(wf, "flux", oi)
    assert wf.get_widget(detailer.id, "cfg", oi) == 1.0  # flux face_detailer technique
    assert wf.get_widget(detailer.id, "denoise", oi) == 0.40
    assert any("facedetailer" in c.lower() or "face_detailer" in c.lower() for c in report["changes"])
