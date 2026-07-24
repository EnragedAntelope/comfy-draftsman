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

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
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
    assert "error" in result and "multi-file" in result["error"]


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


# --- round 17: every output kind relocates, not just images ------------------


class VideoClient(RelocClient):
    """A run that produced a video (Wan/LTX/AnimateDiff) and its audio track -
    just as stuck inside ComfyUI's output tree as an image, and previously
    impossible to hand to a sandboxed caller."""

    def __init__(self):
        super().__init__()
        self.history = {
            "outputs": {
                "12": {
                    "gifs": [{"filename": "clip.webp", "subfolder": "", "type": "output"}],
                    "videos": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}],
                    "audio": [{"filename": "clip.flac", "subfolder": "", "type": "output"}],
                }
            }
        }

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        return {
            "status": "success",
            "prompt_id": "p1",
            "outputs": ComfyClient._collect_outputs(self.history),
        }


@pytest.fixture
def wired_video(monkeypatch, tmp_path):
    client = VideoClient()
    session = Session(tmp_path / "sessions")
    mount = tmp_path / "mount"
    monkeypatch.setattr(
        server._State, "config",
        Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=mount),
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    return client, session.create(Workflow.new(), title="t"), mount


async def test_save_output_relocates_video_and_audio(wired_video):
    _client, _wf_id, _mount = wired_video
    result = await server.save_output(prompt_id="p1")
    assert {__import__("pathlib").Path(p).name for p in result["saved_paths"]} == {
        "clip.webp", "clip.mp4", "clip.flac",
    }


async def test_run_workflow_relocates_a_video_render(wired_video):
    _client, wf_id, mount = wired_video
    result = await server.run_workflow(wf_id, return_preview=False)
    assert result["status"] == "success"
    assert len(result["saved_paths"]) == 3
    assert (mount.resolve() / "clip.mp4").exists()


async def test_save_output_reports_no_files_not_no_images(monkeypatch, tmp_path):
    client = RelocClient()
    client.history = {"outputs": {"9": {}}}
    session = Session(tmp_path / "s")
    monkeypatch.setattr(
        server._State, "config",
        Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=tmp_path / "m"),
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    result = await server.save_output(prompt_id="p1")
    assert "no output files" in result["error"]


# --- round 17: save_dir on a background run is never silently dropped --------


class QueueOnlyClient(RelocClient):
    def _ws_url(self, client_id=None):
        return f"ws://comfy.test/ws?clientId={client_id}"

    async def queue_prompt(self, api, extra_data=None, client_id=None, front=False):
        return {"prompt_id": "queued-1"}


class _NoopTracker:
    client_id = "tracker-1"

    def ensure_running(self):
        pass


async def test_background_run_says_save_dir_does_not_apply(monkeypatch, tmp_path):
    session = Session(tmp_path / "s")
    monkeypatch.setattr(
        server._State, "config",
        Config(comfyui_url=BASE, session_dir=tmp_path, mount_dir=None),
    )
    monkeypatch.setattr(server._State, "client", QueueOnlyClient())
    monkeypatch.setattr(server._State, "tracker", _NoopTracker())
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    dest = tmp_path / "elsewhere"

    result = await server.run_workflow(wf_id, wait=False, save_dir=str(dest))
    assert result["status"] == "queued"
    # previously: the dir was created, then silently ignored
    assert "save_dir_ignored" in result
    assert "save_output(prompt_id='queued-1'" in result["save_dir_ignored"]


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
