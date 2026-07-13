"""save_workflow must never overwrite an existing workflow file by default."""

import io

import pytest
from PIL import Image as PILImage

from comfy_draftsman import server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session


class FakeClient:
    """Mimics ComfyClient's userdata conflict behavior."""

    def __init__(self, existing: set[str]):
        self.existing = set(existing)
        self.saved: list[str] = []

    async def get_object_info(self, refresh: bool = False):
        return {}

    async def save_userdata_workflow(self, name, document, overwrite: bool = False):
        filename = name if name.endswith(".json") else f"{name}.json"
        if filename in self.existing and not overwrite:
            raise FileExistsError(filename)
        self.existing.add(filename)
        self.saved.append(filename)
        return filename


@pytest.fixture
def wired(tmp_path, config, monkeypatch):
    client = FakeClient(existing={"menu.json"})
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    return client, wf_id


async def test_save_renames_instead_of_clobbering(wired):
    client, wf_id = wired
    result = await server.save_workflow(wf_id, "menu")
    assert result["saved"] is True
    assert client.saved == ["menu (draftsman).json"]
    assert result["renamed_from"] == "menu"
    assert "menu.json" in client.existing  # original untouched


async def test_save_free_name_needs_no_rename(wired):
    client, wf_id = wired
    result = await server.save_workflow(wf_id, "fresh")
    assert result["saved"] is True
    assert client.saved == ["fresh.json"]
    assert result["renamed_from"] is None


async def test_save_overwrite_true_replaces(wired):
    client, wf_id = wired
    result = await server.save_workflow(wf_id, "menu", overwrite=True)
    assert result["saved"] is True
    assert client.saved == ["menu.json"]
    assert result["renamed_from"] is None


class FakeViewClient:
    """Mimics ComfyClient's fetch_output for view_output testing."""

    def __init__(self, image_bytes: bytes | None = None):
        self.image_bytes = image_bytes or _make_test_png()

    async def fetch_output(self, ref: dict) -> bytes:
        return self.image_bytes


def _make_test_png(width: int = 320, height: int = 200) -> bytes:
    """Create a small PNG for testing view_output."""
    img = PILImage.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def view_wired(config, monkeypatch):
    """Wire up a FakeViewClient for view_output tests."""
    client = FakeViewClient()
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", client)
    return client


async def test_view_output_returns_meta(view_wired):
    """view_output returns list form: a meta dict plus a renderable Image block."""
    result = await server.view_output("test.png")
    assert isinstance(result, list) and len(result) == 2
    meta_block, image = result
    from mcp.server.fastmcp.utilities.types import Image
    assert isinstance(image, Image), "image block must be present for vision models"
    meta = meta_block["meta"]
    assert meta["filename"] == "test.png"
    assert meta["width"] == 320
    assert meta["height"] == 200
    assert meta["format"] == "png"
    assert meta["subfolder"] == ""
    assert meta["type"] == "output"


async def test_view_output_meta_with_custom_ref(view_wired):
    """view_output meta reflects the passed filename, subfolder, and type."""
    result = await server.view_output("result.png", subfolder="sub", type="temp")
    meta = result[0]["meta"]
    assert meta["filename"] == "result.png"
    assert meta["subfolder"] == "sub"
    assert meta["type"] == "temp"




async def test_view_output_error_on_fetch_failure(config, monkeypatch):
    """Fetch failure returns error, no meta."""
    class FailingClient:
        async def fetch_output(self, ref):
            raise RuntimeError("connection refused")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", FailingClient())
    result = await server.view_output("broken.png")
    assert "error" in result
    assert "meta" not in result
