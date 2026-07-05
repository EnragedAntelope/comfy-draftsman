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
from comfy_draftsman.session import Session

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


@pytest.fixture
async def live_server(live_client, tmp_path, monkeypatch):
    """server module wired to the live instance (tool functions called directly)."""
    from comfy_draftsman import server

    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url=LIVE_URL, session_dir=tmp_path, request_timeout=60)
    )
    monkeypatch.setattr(server._State, "client", live_client)
    monkeypatch.setattr(server._State, "session", Session(tmp_path / "sessions"))
    return server


async def test_save_refuses_invalid_workflow(live_server):
    """save_workflow must refuse a workflow whose model file does not exist."""
    object_info = await live_server._client().get_object_info()
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    ckpt.widgets_values = ["does-not-exist-draftsman-itest.safetensors"]
    wf_id = live_server._session().create(wf, title="refusal test")

    result = await live_server.save_workflow(wf_id, "draftsman-itest-refusal")
    assert result["saved"] is False
    codes = {f["code"] for f in result["findings"]}
    assert "invalid-combo-value" in codes or "unconnected-input" in codes, result


async def test_save_never_clobbers(live_server, live_client):
    """Saving the same name twice must produce two distinct files."""
    from urllib.parse import quote

    wf = Workflow.new()  # empty workflow validates clean
    session = live_server._session()
    wf_id = session.create(wf, title="no-clobber test")

    try:
        first = await live_server.save_workflow(wf_id, "draftsman-itest-noclobber")
        second = await live_server.save_workflow(wf_id, "draftsman-itest-noclobber")
        assert first["saved"] is True and second["saved"] is True
        assert first["saved_to_comfyui"] != second["saved_to_comfyui"], (first, second)
        assert second["renamed_from"] == "draftsman-itest-noclobber"
    finally:
        # userdata persists across test runs - clean up so the draftsman
        # suffix pool never exhausts on the test instance
        candidates = ["draftsman-itest-noclobber", "draftsman-itest-noclobber (draftsman)"]
        candidates += [f"draftsman-itest-noclobber (draftsman {i})" for i in range(2, 21)]
        for name in candidates:
            await live_client._http.delete(
                f"/api/userdata/{quote(f'workflows/{name}.json', safe='')}"
            )
