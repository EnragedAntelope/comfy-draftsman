"""Round-13: fixes from live testing (worklog krea2-speedup, 2026-07-11).

1. run_workflow silently reported success when ComfyUI accepted the prompt (HTTP
   200) but rejected part of the graph (node_errors) and ran only the rest -
   node_errors are now surfaced and the status downgraded to "partial".
2. view_output returned a dict *containing* an Image (which FastMCP repr's into
   text instead of rendering) - now returns the image block in list form.
3. widget-count-drift warnings fired on every display/output node (ShowText,
   rgthree previews) that stashes shown text into widgets_values - suppressed.
"""

import io

import httpx
import pytest
import respx
from mcp.server.fastmcp.utilities.types import Image
from PIL import Image as PILImage

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate
from comfy_draftsman.session import Session

BASE = "http://comfy.test"


def _png_bytes(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


# --- 1. partial-execution node_errors ----------------------------------------


class StubTracker:
    client_id = "tracker-client"

    def ensure_running(self):
        pass

    def snapshot(self, prompt_id):
        return {}


class PartialRunClient:
    """Fake client: the queue accepts the prompt but rejects some nodes."""

    def __init__(self, node_errors):
        self._node_errors = node_errors
        self.output_bytes = _png_bytes(64, 64)

    async def get_object_info(self, refresh=False):
        return {}

    async def get_queue(self):
        return {"queue_running": [], "queue_pending": []}

    async def queue_prompt(self, api, extra_data=None, client_id=None, front=False):
        return {"prompt_id": "p1", "node_errors": self._node_errors}

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        # run_and_wait threads the queue-time node_errors through onto the result
        return {"status": "success", "prompt_id": "p1", "outputs": [], "node_errors": self._node_errors}

    async def fetch_output(self, item):
        return self.output_bytes


@pytest.fixture
def wired_partial(monkeypatch, tmp_path):
    client = PartialRunClient({"12": {"errors": [{"message": "required input missing"}]}})
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    wf_id = session.create(Workflow.new(), title="t")
    return client, wf_id


async def test_run_workflow_flags_partial_execution(wired_partial):
    _client, wf_id = wired_partial
    result = await server.run_workflow(wf_id)
    # no image rendered (rejected branch), so a bare dict comes back
    assert isinstance(result, dict)
    assert result["status"] == "partial"
    assert "12" in result["node_errors"]
    assert "REJECTED" in result["warning"]


async def test_run_workflow_wait_false_surfaces_node_errors(wired_partial):
    _client, wf_id = wired_partial
    result = await server.run_workflow(wf_id, wait=False)
    assert result["status"] == "queued"
    assert "12" in result["node_errors"]
    assert "warning" in result


@respx.mock
async def test_queue_prompt_returns_node_errors_on_200(monkeypatch):
    # ComfyUI returns 200 (not 400) with node_errors when it partially accepts
    respx.post(f"{BASE}/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "p1", "node_errors": {"7": {}}})
    )
    client = ComfyClient(Config(comfyui_url=BASE))
    try:
        queued = await client.queue_prompt({"1": {"class_type": "X", "inputs": {}}})
        assert queued["node_errors"] == {"7": {}}
    finally:
        await client.close()


# --- 2. view_output image serialization --------------------------------------


class ViewClient:
    def __init__(self):
        self.output_bytes = _png_bytes(640, 480)

    async def fetch_output(self, item):
        return self.output_bytes


@pytest.fixture
def wired_view(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
    monkeypatch.setattr(server._State, "client", ViewClient())


async def test_view_output_returns_image_block_not_dict(wired_view):
    result = await server.view_output("out_00001_.png", max_dim=None)
    # list form: [{"meta": {...}}, Image] - NOT a dict wrapping an Image
    assert isinstance(result, list) and len(result) == 2
    meta, image = result
    assert isinstance(image, Image)
    assert meta["meta"]["filename"] == "out_00001_.png"
    assert meta["meta"]["width"] == 640


# --- 3. widget-count-drift on display/output nodes ---------------------------

_DISPLAY_OI = {
    "ShowText": {  # rgthree/pysssss-style display node: OUTPUT_NODE, no widgets
        "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
        "output": ["STRING"],
        "output_node": True,
        "python_module": "custom_nodes.pysssss",
    },
    "PlainNode": {  # a normal node whose schema declares one widget
        "input": {"required": {"value": ["INT", {"default": 0}]}},
        "output": ["INT"],
        "python_module": "nodes",
    },
}


def _display_ui():
    return {
        "nodes": [{
            "id": 1,
            "type": "ShowText",
            "inputs": [{"name": "text", "type": "STRING", "link": None}],
            "outputs": [{"name": "STRING", "type": "STRING", "links": []}],
            "widgets_values": ["shown text captured at run time"],  # 1 value, 0 slots
        }],
        "links": [],
    }


def test_widget_count_drift_suppressed_for_display_node():
    findings = validate(Workflow.from_ui(_display_ui()), _DISPLAY_OI)
    assert not [f for f in findings if f["code"] == "widget-count-drift"]


def test_widget_count_drift_still_warns_for_ordinary_node():
    ui = {
        "nodes": [{
            "id": 1,
            "type": "PlainNode",
            "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "links": []}],
            "widgets_values": [5, 99],  # 2 values, schema expects 1
        }],
        "links": [],
    }
    findings = validate(Workflow.from_ui(ui), _DISPLAY_OI)
    drift = [f for f in findings if f["code"] == "widget-count-drift"]
    assert drift and drift[0]["level"] == "warning"
