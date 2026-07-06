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


async def test_dynamic_combo_flow(draftsman_server):
    """Regression: a graph containing COMFY_DYNAMICCOMBO_V3 nodes (Depth-Anything-3
    + SaveImageAdvanced) validates, exports dotted-key API inputs, and is accepted
    by the live /prompt - end-to-end through the draftsman alone, no manual glue."""
    async with create_connected_server_and_client_session(
        draftsman_server.mcp._mcp_server
    ) as mcp_session:
        info = _json(await mcp_session.call_tool(
            "get_node_info", {"class_types": ["LoadDA3Model", "DA3Inference", "DA3Render"]}
        ))
        if any(info.get(c, {}).get("error") for c in ("DA3Inference", "DA3Render")) or \
                info.get("LoadDA3Model", {}).get("error"):
            pytest.skip("Depth-Anything-3 nodes not installed on this instance")

        def _choices(summary, input_name):
            entry = next((i for i in summary["inputs"] if i["name"] == input_name), None)
            return entry.get("choices") if entry else None

        model_opts = _choices(info["LoadDA3Model"], "model_name")
        if not model_opts:
            pytest.skip("no Depth-Anything-3 model installed")
        model_name = model_opts[0]

        img_info = _json(await mcp_session.call_tool("get_node_info", {"class_type": "LoadImage"}))
        img_choices = _choices(img_info, "image")
        if not img_choices:
            pytest.skip("no input images available")
        image_name = img_choices[0]

        # get_node_info must surface the V3 combo's keys + dotted sub-widgets
        render_info = info["DA3Render"]
        out_entry = next(i for i in render_info["inputs"] if i["name"] == "output")
        assert out_entry.get("dynamic_combo") and "depth" in out_entry["choices"]
        assert any(
            d["name"] == "output.normalization" for d in out_entry["options"]["depth"]
        )

        wf_id = _json(await mcp_session.call_tool("create_workflow", {"title": "da3-e2e"}))["workflow_id"]
        ops = [
            {"op": "add_node", "class_type": "LoadDA3Model", "widgets": {"model_name": model_name}},
            {"op": "add_node", "class_type": "LoadImage", "widgets": {"image": image_name}},
            {"op": "add_node", "class_type": "DA3Inference", "widgets": {"mode": "mono"}},
            {"op": "add_node", "class_type": "DA3Render", "widgets": {"output": "depth"}},
            {"op": "add_node", "class_type": "SaveImageAdvanced",
             "widgets": {"filename_prefix": "draftsman_da3_e2e", "format": "png"}},
            {"op": "connect", "from_node": 1, "from_output": 0, "to_node": 3, "to_input": "da3_model"},
            {"op": "connect", "from_node": 2, "from_output": 0, "to_node": 3, "to_input": "image"},
            {"op": "connect", "from_node": 3, "from_output": 0, "to_node": 4, "to_input": "da3_geometry"},
            {"op": "connect", "from_node": 4, "from_output": 0, "to_node": 5, "to_input": "images"},
        ]
        edited = _json(await mcp_session.call_tool(
            "edit_workflow", {"workflow_id": wf_id, "operations": ops}
        ))
        assert "error" not in edited, edited

        valid = _json(await mcp_session.call_tool("validate_workflow", {"workflow_id": wf_id}))
        assert valid["ok"], valid  # no false unconnected-input on the V3 combos

        api = _json(await mcp_session.call_tool(
            "export_workflow_json", {"workflow_id": wf_id, "format": "api"}
        ))
        render = next(v for v in api.values() if v["class_type"] == "DA3Render")
        assert render["inputs"]["output"] == "depth"
        assert render["inputs"]["output.normalization"] == "v2_style"  # dotted sub-widget

        # the live /prompt must accept it (queue it) - proves ComfyUI parses the
        # dotted V3 inputs. wait=False so we don't depend on the render succeeding.
        queued = _json(await mcp_session.call_tool(
            "run_workflow", {"workflow_id": wf_id, "wait": False}
        ))
        assert queued["status"] == "queued", queued
        await mcp_session.call_tool("manage_queue", {"action": "clear"})
