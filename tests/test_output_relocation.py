"""Relocating finished renders out of ComfyUI's output tree into a mount folder
the caller can reach: the save_output tool and run_workflow's save_dir/mount
auto-relocation. Addresses the second pain point - ComfyUI save nodes can only
write inside output/, so a copy step is needed before an image is presentable.
"""

import io

import pytest
from PIL import Image as PILImage

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

pytestmark = pytest.mark.asyncio

BASE = "http://comfy.test"


def _png(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


class RelocClient:
    def __init__(self):
        self.png = _png()
        self.history = {
            "outputs": {
                "9": {"images": [
                    {"filename": "out_00001_.png", "subfolder": "", "type": "output"},
                    {"filename": "out_00002_.png", "subfolder": "", "type": "output"},
                ]}
            }
        }

    async def get_object_info(self, refresh: bool = False):
        return {}

    async def run_and_wait(self, api, timeout=600.0):
        return {
            "status": "success",
            "prompt_id": "p1",
            "outputs": [
                {"filename": "out_00001_.png", "subfolder": "", "type": "output",
                 "node_id": "9", "kind": "images"}
            ],
        }

    async def get_history(self, prompt_id):
        return self.history

    async def fetch_output(self, item):
        return self.png

    @staticmethod
    def _collect_outputs(history):
        return ComfyClient._collect_outputs(history)


@pytest.fixture
def wired(monkeypatch, tmp_path):
    client = RelocClient()
    session = Session(tmp_path / "sessions")
    mount = tmp_path / "mount"
    monkeypatch.setattr(
        server._State, "config",
        Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=mount),
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    return client, wf_id, mount


# --- save_output -------------------------------------------------------------


async def test_save_output_requires_a_source(wired):
    result = await server.save_output()
    assert "error" in result and "prompt_id or filename" in result["error"]


async def test_save_output_by_prompt_id_relocates_all_images(wired):
    _client, _wf_id, mount = wired
    result = await server.save_output(prompt_id="p1")
    assert len(result["saved_paths"]) == 2
    assert result["dest_dir"] == str(mount.resolve())
    for path in result["saved_paths"]:
        assert (mount.resolve() / __import__("pathlib").Path(path).name).exists()


async def test_save_output_explicit_filename_and_rename(wired):
    _client, _wf_id, mount = wired
    result = await server.save_output(filename="out_00001_.png", dest_filename="hero.png")
    assert result["saved_paths"] == [str(mount.resolve() / "hero.png")]


async def test_save_output_rejects_traversal_source(wired):
    result = await server.save_output(filename="../escape.png")
    assert "error" in result and "invalid path" in result["error"]


async def test_save_output_rename_refused_for_batch(wired):
    result = await server.save_output(prompt_id="p1", dest_filename="one.png")
    assert "error" in result and "multi-image" in result["error"]


async def test_save_output_dedupes_instead_of_clobbering(wired):
    first = await server.save_output(filename="out_00001_.png")
    second = await server.save_output(filename="out_00001_.png")
    assert first["saved_paths"] != second["saved_paths"]  # second got a _1 suffix


async def test_save_output_needs_a_destination(monkeypatch, tmp_path):
    # no mount configured and no dest_dir -> a clear error, not a crash
    session = Session(tmp_path / "s")
    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=None)
    )
    monkeypatch.setattr(server._State, "client", RelocClient())
    monkeypatch.setattr(server._State, "session", session)
    result = await server.save_output(filename="out_00001_.png")
    assert "error" in result and "COMFYUI_MOUNT_DIR" in result["error"]


# --- run_workflow auto-relocation -------------------------------------------


async def test_run_workflow_auto_relocates_to_mount(wired):
    _client, wf_id, mount = wired
    result = await server.run_workflow(wf_id, return_preview=False)
    assert result["status"] == "success"
    assert result["dest_dir"] == str(mount.resolve())
    assert len(result["saved_paths"]) == 1
    assert (mount.resolve() / "out_00001_.png").exists()


async def test_run_workflow_explicit_save_dir(wired, tmp_path):
    _client, wf_id, _mount = wired
    dest = tmp_path / "elsewhere"
    result = await server.run_workflow(wf_id, return_preview=False, save_dir=str(dest))
    assert result["saved_paths"] == [str(dest.resolve() / "out_00001_.png")]


async def test_run_workflow_no_relocation_without_mount(monkeypatch, tmp_path):
    session = Session(tmp_path / "s")
    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=None)
    )
    monkeypatch.setattr(server._State, "client", RelocClient())
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    result = await server.run_workflow(wf_id, return_preview=False)
    assert result["status"] == "success"
    assert "saved_paths" not in result
