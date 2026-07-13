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


def _build_txt2img(object_info, checkpoint: str, prefix: str = "draftsman_e2e") -> Workflow:
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", checkpoint, object_info)
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
    wf.set_widget(save.id, "filename_prefix", prefix, object_info)
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
    return wf


async def _pick_checkpoint(client) -> str:
    checkpoints = await client.list_models("checkpoints")
    sdxl = [c for c in checkpoints if "sdxl" in c.lower() or "xl" in c.lower()]
    assert sdxl, f"no SDXL-ish checkpoint available in {checkpoints}"
    return sdxl[0]


async def test_build_validate_run_real_render(live_client):
    object_info = await live_client.get_object_info()
    wf = _build_txt2img(object_info, await _pick_checkpoint(live_client))

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
    monkeypatch.setattr(server._State, "tracker", None)  # fresh per event loop
    yield server
    if server._State.tracker is not None:
        await server._State.tracker.stop()


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


async def test_run_thumbnail_then_view_full_size(live_server, live_client):
    """run_workflow inlines a downscaled thumbnail; view_output serves full size."""
    import io

    from mcp.server.fastmcp.utilities.types import Image
    from PIL import Image as PILImage

    object_info = await live_client.get_object_info()
    wf = _build_txt2img(object_info, await _pick_checkpoint(live_client), prefix="draftsman_r8")
    wf_id = live_server._session().create(wf, title="round8 thumbnail test")

    result = await live_server.run_workflow(wf_id, timeout_seconds=300)
    assert isinstance(result, list), result
    payload, preview = result
    assert payload["status"] == "success"
    assert isinstance(preview, Image)
    thumb = PILImage.open(io.BytesIO(preview.data))
    assert max(thumb.size) <= live_server.PREVIEW_MAX_DIM

    ref = payload["outputs"][0]
    full = (await live_server.view_output(ref["filename"], ref["subfolder"], max_dim=None))[1]
    assert isinstance(full, Image)
    assert max(PILImage.open(io.BytesIO(full.data)).size) == 640  # render size


async def test_background_run_reports_status(live_server, live_client):
    """wait=False queues; get_run_status ends at success with outputs."""
    import asyncio

    object_info = await live_client.get_object_info()
    wf = _build_txt2img(object_info, await _pick_checkpoint(live_client), prefix="draftsman_r8bg")
    wf_id = live_server._session().create(wf, title="round8 background test")

    queued = await live_server.run_workflow(wf_id, wait=False)
    assert queued["status"] == "queued", queued
    prompt_id = queued["prompt_id"]

    seen_statuses = set()
    for _ in range(150):
        status = await live_server.get_run_status(prompt_id)
        seen_statuses.add(status["status"])
        if status["status"] in ("success", "error"):
            break
        await asyncio.sleep(2)
    assert status["status"] == "success", status
    assert status["outputs"][0]["filename"].startswith("draftsman_r8bg")


async def test_upload_image_roundtrip(live_server):
    """upload_image lands in the input folder and can be viewed back."""
    import base64
    import io

    from mcp.server.fastmcp.utilities.types import Image
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (32, 32), (10, 200, 10)).save(buf, format="PNG")
    result = await live_server.upload_image(
        image_base64=base64.b64encode(buf.getvalue()).decode(),
        name="draftsman-itest-upload.png",
        overwrite=True,
    )
    assert result["uploaded"]["name"] == "draftsman-itest-upload.png", result

    back = (await live_server.view_output(
        result["uploaded"]["name"], result["uploaded"].get("subfolder", ""), type="input"
    ))[1]
    assert isinstance(back, Image)
    assert PILImage.open(io.BytesIO(back.data)).size == (32, 32)


async def test_subgraph_workflow_runs_flattened(live_client):
    """Round-10: a subgraph-packaged workflow (schema-1.0 definitions) renders
    end-to-end - to_api flattens the instance the way the frontend would."""
    object_info = await live_client.get_object_info()
    inner = _build_txt2img(object_info, await _pick_checkpoint(live_client))
    save = next(n for n in inner.nodes.values() if n.type == "SaveImage")
    decode = next(n for n in inner.nodes.values() if n.type == "VAEDecode")
    inner.remove_node(save.id)  # SaveImage lives OUTSIDE the subgraph
    ui = inner.to_ui()
    sg_id = "12345678-1234-4123-8123-123456789abc"
    doc = {
        "id": "11111111-2222-4333-8444-555555555555",
        "revision": 0,
        "nodes": [
            {
                "id": 100, "type": sg_id, "pos": [0, 0], "size": [200, 100],
                "inputs": [], "widgets_values": [], "properties": {},
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [500]}],
            },
            {
                "id": 101, "type": "SaveImage", "pos": [400, 0], "size": [300, 200],
                "inputs": [{"name": "images", "type": "IMAGE", "link": 500}],
                "outputs": [], "widgets_values": ["draftsman_r10_subgraph"],
            },
        ],
        "links": [[500, 100, 0, 101, 0, "IMAGE"]],
        "groups": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": sg_id,
                    "name": "TXT2IMG (draftsman itest)",
                    "inputs": [],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "linkIds": [999]}],
                    "nodes": ui["nodes"],
                    "links": [*ui["links"], [999, decode.id, 0, -20, 0, "IMAGE"]],
                }
            ]
        },
        "config": {}, "extra": {}, "version": 0.4,
    }
    wf = Workflow.from_ui(doc)
    findings = validate(wf, object_info)
    assert [f for f in findings if f["level"] == "error"] == [], findings
    assert any(f["code"] == "subgraph-instance" for f in findings)

    api = wf.to_api(object_info)
    assert "SaveImage" in {e["class_type"] for e in api.values()}
    result = await live_client.run_and_wait(api, timeout=300)
    assert result["status"] == "success", result
    assert result["outputs"][0]["filename"].startswith("draftsman_r10_subgraph")
