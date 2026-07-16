"""Round-8: execution-side tools (view_output, get_run_status, upload_image,
manage_queue), non-blocking runs, downscaled previews, ProgressTracker.

Tool concepts credit KerbalTheGathering/ComfyUI_MCP; all code here is
independently implemented.
"""

import base64
import io

import httpx
import pytest
import respx
from mcp.server.fastmcp.utilities.types import Image
from PIL import Image as PILImage

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.comfy.progress import ProgressTracker
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.imaging import downscale_image
from comfy_draftsman.session import Session

BASE = "http://comfy.test"


def _png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    PILImage.new(mode, (width, height), (200, 30, 30) if mode == "RGB" else None).save(
        buf, format="PNG"
    )
    return buf.getvalue()


# --- imaging.downscale_image ---


def test_downscale_resizes_and_jpegs_opaque():
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000), 768)
    assert fmt == "jpeg"
    img = PILImage.open(io.BytesIO(data))
    assert max(img.size) == 768
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000), 768)
    assert fmt == "jpeg"
    img = PILImage.open(io.BytesIO(data))
    assert max(img.size) == 768
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000), 768)
    assert fmt == "jpeg"
    img = PILImage.open(io.BytesIO(data))
    assert max(img.size) == 768
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000), 768)
    assert fmt == "jpeg"
    img = PILImage.open(io.BytesIO(data))
    assert max(img.size) == 768


def test_downscale_keeps_alpha_as_png():
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000, mode="RGBA"), 768)
    assert fmt == "png"
    assert PILImage.open(io.BytesIO(data)).mode == "RGBA"
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000, mode="RGBA"), 768)
    assert fmt == "png"
    assert PILImage.open(io.BytesIO(data)).mode == "RGBA"
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000, mode="RGBA"), 768)
    assert fmt == "png"
    assert PILImage.open(io.BytesIO(data)).mode == "RGBA"
    assert width == 768
    assert height == 384
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000, mode="RGBA"), 768)
    assert fmt == "png"
    assert PILImage.open(io.BytesIO(data)).mode == "RGBA"


def test_small_image_passes_through_untouched():
    original = _png_bytes(100, 80)
    data, fmt, width, height = downscale_image(original, 768)
    assert data == original
    assert fmt == "png"
    assert width == 100
    assert height == 80
    original = _png_bytes(100, 80)
    data, fmt, width, height = downscale_image(original, 768)
    assert data == original
    assert fmt == "png"
    assert width == 100
    assert height == 80
    original = _png_bytes(100, 80)
    data, fmt, width, height = downscale_image(original, 768)
    assert data == original
    assert fmt == "png"
    assert width == 100
    assert height == 80
    original = _png_bytes(100, 80)
    data, fmt, width, height = downscale_image(original, 768)
    assert data == original
    assert fmt == "png"


def test_max_dim_none_keeps_full_resolution():
    data, fmt, width, height = downscale_image(_png_bytes(2000, 1000), None)
    assert PILImage.open(io.BytesIO(data)).size == (2000, 1000)
    # 2000px flat-color PNG is small, so it passes through as-is
    assert fmt == "png"
    assert width == 2000
    assert height == 1000


def test_large_opaque_png_reencodes_to_jpeg_without_resize():
    import random

    rng = random.Random(0)
    img = PILImage.new("RGB", (700, 700))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(700 * 700)])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    original = buf.getvalue()
    assert len(original) > 256 * 1024  # noise PNG compresses badly
    data, fmt, width, height = downscale_image(original, 768)  # no resize needed
    assert fmt == "jpeg"
    assert len(data) < len(original) // 2  # noise is JPEG's worst case; real renders do far better
    assert PILImage.open(io.BytesIO(data)).size == (700, 700)
    assert width == 700
    assert height == 700
    data, fmt, width, height = downscale_image(original, 768)  # no resize needed
    assert fmt == "jpeg"
    assert len(data) < len(original) // 2  # noise is JPEG's worst case; real renders do far better
    assert PILImage.open(io.BytesIO(data)).size == (700, 700)
    assert width == 700
    assert height == 700
    assert fmt == "jpeg"
    assert len(data) < len(original) // 2  # noise is JPEG's worst case; real renders do far better
    assert PILImage.open(io.BytesIO(data)).size == (700, 700)
    assert width == 700
    assert height == 700
    assert fmt == "jpeg"
    assert len(data) < len(original) // 2  # noise is JPEG's worst case; real renders do far better
    assert PILImage.open(io.BytesIO(data)).size == (700, 700)


def test_undecodable_payload_raises():
    with pytest.raises(ValueError, match="not a decodable image"):
        downscale_image(b"\x00\x01not-an-image", 768)


# --- server-tool fixtures ---


class StubTracker:
    client_id = "tracker-client"
    connected = False

    def ensure_running(self):
        pass

    def snapshot(self, prompt_id):
        return {"ws_connected": self.connected}


class RunClient:
    def __init__(self):
        self.queued_with: str | None = None
        self.history: dict = {}
        self.queue: dict = {"queue_running": [], "queue_pending": []}
        self.output_bytes = _png_bytes(1600, 1600)

    async def get_object_info(self, refresh: bool = False):
        return {}

    async def queue_prompt(self, api, extra_data=None, client_id=None, front=False):
        self.queued_with = client_id
        self.queued_extra_data = extra_data
        self.queued_front = front
        return {"prompt_id": "p123"}

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        self.run_extra_data = extra_data
        return {
            "status": "success",
            "prompt_id": "p123",
            "outputs": [
                {"filename": "out_00001_.png", "subfolder": "", "type": "output",
                 "node_id": "9", "kind": "images"}
            ],
        }

    async def get_history(self, prompt_id):
        return self.history

    async def get_queue(self):
        return self.queue

    async def fetch_output(self, item):
        return self.output_bytes

    @staticmethod
    def _collect_outputs(history):
        return ComfyClient._collect_outputs(history)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    client = RunClient()
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    wf_id = session.create(Workflow.new(), title="t")
    return client, wf_id


# --- run_workflow ---


async def test_run_workflow_returns_downscaled_thumbnail(wired):
    _client, wf_id = wired
    result = await server.run_workflow(wf_id)
    assert isinstance(result, list) and len(result) == 2
    payload, preview = result
    assert payload["status"] == "success"
    assert payload["outputs"][0]["filename"] == "out_00001_.png"
    assert "view_output" in payload["preview"]
    assert isinstance(preview, Image)
    thumb = PILImage.open(io.BytesIO(preview.data))
    assert max(thumb.size) == server.PREVIEW_MAX_DIM


async def test_run_workflow_wait_false_queues_under_tracker_id(wired):
    client, wf_id = wired
    result = await server.run_workflow(wf_id, wait=False)
    assert result == {"status": "queued", "prompt_id": "p123"}
    assert client.queued_with == "tracker-client"


async def test_run_workflow_skips_preview_on_undecodable_output(wired):
    client, wf_id = wired
    client.output_bytes = b"mp4-ish junk"
    result = await server.run_workflow(wf_id)
    assert isinstance(result, dict)  # refs only, no inline image
    assert result["status"] == "success"


async def test_run_workflow_injects_comfy_api_key_when_set(monkeypatch, wired):
    monkeypatch.setattr(server, "_COMFY_API_KEY", "secret-key")
    client, wf_id = wired
    result = await server.run_workflow(wf_id, wait=False)
    assert result == {"status": "queued", "prompt_id": "p123"}
    assert client.queued_extra_data == {"api_key_comfy_org": "secret-key"}


async def test_run_workflow_wait_injects_comfy_api_key_when_set(monkeypatch, wired):
    monkeypatch.setattr(server, "_COMFY_API_KEY", "secret-key")
    client, wf_id = wired
    result = await server.run_workflow(wf_id, return_preview=False)
    assert result["status"] == "success"
    assert client.run_extra_data == {"api_key_comfy_org": "secret-key"}


async def test_run_workflow_no_extra_data_without_api_key(wired):
    client, wf_id = wired
    await server.run_workflow(wf_id, wait=False)
    assert client.queued_extra_data is None
    client, wf_id = wired
    client.output_bytes = b"mp4-ish junk"
    result = await server.run_workflow(wf_id)
    assert isinstance(result, dict)  # refs only, no inline image
    assert result["status"] == "success"


# --- view_output ---


async def test_view_output_rejects_traversal(wired):
    assert "invalid path" in (await server.view_output("../secrets.png"))["error"]
    assert "invalid path" in (await server.view_output("a.png", subfolder="../up"))["error"]


async def test_view_output_downscales_by_default(wired):
    result = await server.view_output("out_00001_.png")
    image = result[1]
    assert isinstance(image, Image)
    assert max(PILImage.open(io.BytesIO(image.data)).size) == 1024
    result = await server.view_output("out_00001_.png")
    image = result[1]
    assert isinstance(image, Image)
    assert max(PILImage.open(io.BytesIO(image.data)).size) == 1024
    result = await server.view_output("out_00001_.png")
    image = result[1]
    assert isinstance(image, Image)
    assert max(PILImage.open(io.BytesIO(image.data)).size) == 1024
    result = await server.view_output("out_00001_.png")
    image = result[1]
    assert isinstance(image, Image)
    assert max(PILImage.open(io.BytesIO(image.data)).size) == 1024


async def test_view_output_full_size(wired):
    client, _ = wired
    result = await server.view_output("out_00001_.png", max_dim=None)
    image = result[1]
    assert image.data == client.output_bytes
    client, _ = wired
    result = await server.view_output("out_00001_.png", max_dim=None)
    image = result[1]
    assert image.data == client.output_bytes
    client, _ = wired
    result = await server.view_output("out_00001_.png", max_dim=None)
    image = result[1]
    assert image.data == client.output_bytes
    client, _ = wired
    result = await server.view_output("out_00001_.png", max_dim=None)
    image = result[1]
    assert image.data == client.output_bytes


async def test_view_output_non_image_errors_cleanly(wired):
    client, _ = wired
    client.output_bytes = b"not an image"
    result = await server.view_output("clip.mp4")
    assert "not a decodable image" in result["error"]


# --- get_run_status ---


async def test_status_success_from_history(wired):
    client, _ = wired
    client.history = {
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
        "status": {"status_str": "success", "completed": True, "messages": []},
    }
    result = await server.get_run_status("p123")
    assert result["status"] == "success"
    assert result["outputs"][0]["filename"] == "a.png"


async def test_status_error_from_history_messages(wired):
    client, _ = wired
    client.history = {
        "outputs": {},
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ["execution_error", {"node_id": "3", "node_type": "KSampler",
                                     "exception_message": "boom", "exception_type": "RuntimeError"}]
            ],
        },
    }
    result = await server.get_run_status("p123")
    assert result["status"] == "error"
    assert result["error"]["message"] == "boom"


async def test_status_pending_reports_queue_position(wired):
    client, _ = wired
    client.queue = {"queue_running": [[0, "other"]], "queue_pending": [[1, "x"], [2, "p123"]]}
    result = await server.get_run_status("p123")
    assert result["status"] == "pending"
    assert result["queue_position"] == 2


async def test_status_unknown(wired):
    result = await server.get_run_status("nope")
    assert result["status"] == "unknown"
    assert result["ws_connected"] is False


# --- upload_image ---


async def test_upload_requires_exactly_one_source(wired):
    assert "exactly one" in (await server.upload_image())["error"]
    assert "exactly one" in (await server.upload_image("a.png", "aGk="))["error"]


async def test_upload_rejects_path_separators_in_name(wired, tmp_path):
    src = tmp_path / "pic.png"
    src.write_bytes(_png_bytes(4, 4))
    result = await server.upload_image(image_path=str(src), name="../pic.png")
    assert "plain filename" in result["error"]


async def test_upload_rejects_bad_base64(wired):
    assert "invalid base64" in (await server.upload_image(image_base64="!!!"))["error"]


async def test_upload_from_path(wired, tmp_path, monkeypatch):
    client, _ = wired
    calls = {}

    async def fake_upload(data, name, subfolder="", overwrite=False, image_type="input"):
        calls["name"] = name
        return {"name": name, "subfolder": subfolder, "type": "input"}

    client.upload_image = fake_upload
    src = tmp_path / "pic.png"
    src.write_bytes(_png_bytes(4, 4))
    result = await server.upload_image(image_path=str(src))
    assert calls["name"] == "pic.png"
    assert "'pic.png'" in result["hint"]


async def test_upload_mask_needs_reference_filename(wired):
    result = await server.upload_image(image_base64=base64.b64encode(b"x").decode(), mask_for={})
    assert "mask_for needs" in result["error"]


async def test_upload_mask_routes_to_mask_endpoint(wired):
    client, _ = wired
    seen = {}

    async def fake_mask(data, name, original_ref, subfolder="", overwrite=False):
        seen["ref"] = original_ref
        return {"name": name, "subfolder": "", "type": "input"}

    client.upload_mask = fake_mask
    await server.upload_image(
        image_base64=base64.b64encode(b"x").decode(), name="m.png",
        mask_for={"filename": "src.png"},
    )
    assert seen["ref"] == {"filename": "src.png", "subfolder": "", "type": "input"}


# --- manage_queue ---


async def test_manage_queue_status_is_compact(wired):
    client, _ = wired
    client.queue = {
        "queue_running": [[0, "r1", {"huge": "graph"}]],
        "queue_pending": [[1, "p1", {"huge": "graph"}]],
    }
    result = await server.manage_queue("status")
    assert result == {"running": ["r1"], "pending": ["p1"], "pending_count": 1}


async def test_manage_queue_delete_requires_ids(wired):
    assert "requires prompt_ids" in (await server.manage_queue("delete"))["error"]


async def test_manage_queue_actions_dispatch(wired):
    client, _ = wired
    done = []
    for verb in ("interrupt", "clear_queue", "free"):
        async def fake(*a, _v=verb, **k):
            done.append(_v)
        setattr(client, verb, fake)

    async def fake_delete(ids):
        done.append(("delete", tuple(ids)))

    client.delete_queue_items = fake_delete
    await server.manage_queue("interrupt")
    await server.manage_queue("clear")
    await server.manage_queue("delete", prompt_ids=["a", "b"])
    result = await server.manage_queue("free", unload_models=True)
    assert done == ["interrupt", "clear_queue", ("delete", ("a", "b")), "free"]
    assert "unloaded models" in result["done"]


# --- ProgressTracker event handling ---


def _tracker():
    return ProgressTracker(lambda cid: f"ws://comfy.test/ws?clientId={cid}")


def test_tracker_progress_percent():
    t = _tracker()
    t.handle_event({"type": "execution_start", "data": {"prompt_id": "p"}})
    t.handle_event({"type": "progress", "data": {"prompt_id": "p", "node": "3", "value": 5, "max": 20}})
    snap = t.snapshot("p")
    assert snap["status"] == "running"
    assert snap["percent"] == 25.0
    assert snap["ws_connected"] is False


def test_tracker_terminal_states():
    t = _tracker()
    t.handle_event({"type": "execution_error", "data": {"prompt_id": "a", "exception_message": "x"}})
    t.handle_event({"type": "execution_interrupted", "data": {"prompt_id": "b"}})
    t.handle_event({"type": "executing", "data": {"prompt_id": "c", "node": None}})
    assert t.snapshot("a")["status"] == "error"
    assert t.snapshot("b")["error"]["message"] == "interrupted"
    assert t.snapshot("c")["status"] == "finished"


def test_tracker_is_bounded():
    t = _tracker()
    for i in range(30):
        t.handle_event({"type": "execution_start", "data": {"prompt_id": f"p{i}"}})
    assert len(t._states) == 20
    assert t.snapshot("p0") == {"ws_connected": False}  # evicted
    assert t.snapshot("p29")["status"] == "running"


def test_tracker_ignores_events_without_prompt_id():
    t = _tracker()
    t.handle_event({"type": "status", "data": {"exec_info": {"queue_remaining": 0}}})
    assert t._states == {}


# --- client endpoints (respx) ---


@pytest.fixture
def http_client(config):
    return ComfyClient(config)


@respx.mock
async def test_client_upload_image_multipart(http_client):
    route = respx.post(f"{BASE}/upload/image").mock(
        return_value=httpx.Response(200, json={"name": "pic.png", "subfolder": "", "type": "input"})
    )
    result = await http_client.upload_image(b"bytes", "pic.png", overwrite=True)
    assert result["name"] == "pic.png"
    body = route.calls[0].request.content
    assert b'name="image"' in body and b'filename="pic.png"' in body
    assert b"true" in body  # overwrite flag made it into the form


@respx.mock
async def test_client_upload_mask_sends_original_ref(http_client):
    route = respx.post(f"{BASE}/upload/mask").mock(
        return_value=httpx.Response(200, json={"name": "m.png", "subfolder": "", "type": "input"})
    )
    await http_client.upload_mask(b"bytes", "m.png", {"filename": "src.png"})
    assert b"original_ref" in route.calls[0].request.content


@respx.mock
async def test_client_queue_maintenance_endpoints(http_client):
    clear = respx.post(f"{BASE}/queue").mock(return_value=httpx.Response(200))
    free = respx.post(f"{BASE}/free").mock(return_value=httpx.Response(200))
    await http_client.clear_queue()
    await http_client.delete_queue_items(["a"])
    await http_client.free(unload_models=True)
    import json as _json

    assert _json.loads(clear.calls[0].request.content) == {"clear": True}
    assert _json.loads(clear.calls[1].request.content) == {"delete": ["a"]}
    assert _json.loads(free.calls[0].request.content) == {
        "unload_models": True, "free_memory": True,
    }
