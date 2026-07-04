"""Integration tests against a live ComfyUI instance (COMFYUI_TEST_URL).

Run with: pytest -m integration
These prove the full loop: discover -> build -> validate -> render -> outputs.
"""

import os

import pytest

from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate

pytestmark = pytest.mark.integration

LIVE_URL = os.environ.get("COMFYUI_TEST_URL", "http://127.0.0.1:8288")


@pytest.fixture
async def live_client(tmp_path):
    client = ComfyClient(Config(comfyui_url=LIVE_URL, session_dir=tmp_path, request_timeout=60))
    yield client
    await client.close()


async def test_discovery_endpoints(live_client):
    stats = await live_client.get_system_stats()
    assert "comfyui_version" in stats["system"]
    folders = await live_client.list_model_folders()
    assert "checkpoints" in folders
    index = await live_client.get_template_index()
    assert index


async def test_build_validate_run_real_render(live_client):
    object_info = await live_client.get_object_info()
    checkpoints = await live_client.list_models("checkpoints")
    sdxl = [c for c in checkpoints if "sdxl" in c.lower() or "xl" in c.lower()]
    assert sdxl, f"no SDXL-ish checkpoint available in {checkpoints}"

    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", sdxl[0], object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    wf.set_widget(latent.id, "width", 640, object_info)
    wf.set_widget(latent.id, "height", 640, object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.set_widget(sampler.id, "steps", 6, object_info)
    wf.set_widget(sampler.id, "seed", 7, object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    wf.set_widget(save.id, "filename_prefix", "draftsman_e2e", object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    wf.set_widget(pos.id, "text", "a tiny red fox, watercolor", object_info)
    wf.set_widget(neg.id, "text", "text, watermark", object_info)
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")

    findings = validate(wf, object_info)
    assert [f for f in findings if f["level"] == "error"] == []

    result = await live_client.run_and_wait(wf.to_api(object_info), timeout=300)
    assert result["status"] == "success", result
    images = result["outputs"]
    assert images and images[0]["filename"].startswith("draftsman_e2e")
