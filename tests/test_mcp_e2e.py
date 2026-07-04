"""End-to-end through the MCP protocol layer against a live ComfyUI instance.

Exercises the real tool surface the way an agent would: discover -> build ->
validate -> organize -> run -> save.
"""

import json
import os

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

pytestmark = pytest.mark.integration

LIVE_URL = os.environ.get("COMFYUI_TEST_URL", "http://127.0.0.1:8288")


@pytest.fixture
def draftsman_server(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", LIVE_URL)
    monkeypatch.setenv("DRAFTSMAN_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("DRAFTSMAN_LEARNED_DIR", str(tmp_path / "learned"))
    from comfy_draftsman import server

    server._State.config = None
    server._State.client = None
    server._State.registry = None
    server._State.session = None
    return server


def _json(result):
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


async def test_full_agent_flow(draftsman_server):
    # session lives entirely inside the test body: anyio cancel scopes must
    # enter and exit in the same task
    async with create_connected_server_and_client_session(
        draftsman_server.mcp._mcp_server
    ) as mcp_session:
        await _flow(mcp_session)


async def _flow(mcp_session):
    info = _json(await mcp_session.call_tool("get_instance_info", {}))
    assert info["comfyui_version"]

    models = _json(await mcp_session.call_tool("list_models", {"folder": "checkpoints"}))
    sdxl = [f for f in models["files"] if "xl" in f.lower()]
    assert sdxl

    guidance = _json(
        await mcp_session.call_tool(
            "get_model_guidance", {"family": "sdxl", "model_filename": sdxl[0]}
        )
    )
    cfg = guidance["sampling"]["cfg"]["default"]
    steps = 6  # keep the smoke render fast

    created = _json(await mcp_session.call_tool("create_workflow", {"title": "e2e"}))
    wf_id = created["workflow_id"]

    ops = [
        {"op": "add_node", "class_type": "CheckpointLoaderSimple", "widgets": {"ckpt_name": sdxl[0]}},
        {"op": "add_node", "class_type": "CLIPTextEncode", "widgets": {"text": "misty forest, morning light"}},
        {"op": "add_node", "class_type": "CLIPTextEncode", "widgets": {"text": "watermark"}},
        {"op": "add_node", "class_type": "EmptyLatentImage", "widgets": {"width": 640, "height": 640}},
        {"op": "add_node", "class_type": "KSampler", "widgets": {"steps": steps, "cfg": cfg, "seed": 3}},
        {"op": "add_node", "class_type": "VAEDecode"},
        {"op": "add_node", "class_type": "SaveImage", "widgets": {"filename_prefix": "draftsman_mcp_e2e"}},
        {"op": "connect", "from_node": 1, "from_output": "MODEL", "to_node": 5, "to_input": "model"},
        {"op": "connect", "from_node": 1, "from_output": "CLIP", "to_node": 2, "to_input": "clip"},
        {"op": "connect", "from_node": 1, "from_output": "CLIP", "to_node": 3, "to_input": "clip"},
        {"op": "connect", "from_node": 2, "from_output": "CONDITIONING", "to_node": 5, "to_input": "positive"},
        {"op": "connect", "from_node": 3, "from_output": "CONDITIONING", "to_node": 5, "to_input": "negative"},
        {"op": "connect", "from_node": 4, "from_output": "LATENT", "to_node": 5, "to_input": "latent_image"},
        {"op": "connect", "from_node": 5, "from_output": "LATENT", "to_node": 6, "to_input": "samples"},
        {"op": "connect", "from_node": 1, "from_output": "VAE", "to_node": 6, "to_input": "vae"},
        {"op": "connect", "from_node": 6, "from_output": "IMAGE", "to_node": 7, "to_input": "images"},
    ]
    edited = _json(await mcp_session.call_tool("edit_workflow", {"workflow_id": wf_id, "operations": ops}))
    assert "error" not in edited, edited

    valid = _json(await mcp_session.call_tool("validate_workflow", {"workflow_id": wf_id}))
    assert valid["ok"], valid

    organized = _json(await mcp_session.call_tool("organize_workflow", {"workflow_id": wf_id}))
    assert organized["family"] == "sdxl"
    assert organized["lint"] == []

    run = await mcp_session.call_tool(
        "run_workflow", {"workflow_id": wf_id, "timeout_seconds": 300}
    )
    assert not run.isError
    payload = json.loads(run.content[0].text)
    assert payload["status"] == "success", payload
    # preview image comes back as MCP image content
    assert any(c.type == "image" for c in run.content)

    saved = _json(await mcp_session.call_tool("save_workflow", {"workflow_id": wf_id, "name": "draftsman-e2e-test"}))
    assert "workflow browser" in saved["saved_to_comfyui"]
    assert saved["lint"] == []
