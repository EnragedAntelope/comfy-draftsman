"""Round 15: sharper Cowork/Code/Desktop integration.

Sandboxed clients can only be handed a render if COMFYUI_MOUNT_DIR points at a
folder both this server and the caller can see. These tests cover the three ways
that readiness is now made visible up front, plus the relative-path footgun:

- `_resolve_dest` refuses a relative dest_dir (the server's cwd is not the
  agent's - on an MCP host it's often System32).
- `_mount_status` reports configured / writable, with an actionable hint.
- `get_instance_info` and the `draftsman://capabilities` resource surface it.
"""

import json

import pytest

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyConnectionError
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

BASE = "http://comfy.test"


class MiniClient:
    """Just enough client for save_output/run_workflow to reach dest resolution.

    The relative-dest rejection happens in _resolve_dest, before any output is
    fetched, so these never need real image bytes."""

    async def get_object_info(self, refresh: bool = False):
        return {}

    async def get_queue(self):
        return {"queue_running": [], "queue_pending": []}


class StatsClient:
    """Minimal client for get_instance_info: just the two live calls it makes."""

    async def get_system_stats(self):
        return {
            "system": {"comfyui_version": "0.3.0", "os": "nt"},
            "devices": [{"name": "cuda:0", "vram_total": 24_000, "vram_free": 20_000}],
        }

    async def get_queue(self):
        return {"queue_running": [], "queue_pending": []}


class DownClient:
    """A ComfyUI that can't be reached - get_system_stats raises the actionable
    connection error the client now produces for transport failures."""

    async def get_system_stats(self):
        raise ComfyConnectionError(f"can't reach ComfyUI at {BASE}: ConnectError.")


def _cfg(tmp_path, mount):
    return Config(comfyui_url=BASE, session_dir=tmp_path / "s", mount_dir=mount)


# --- #2 relative dest_dir is refused ----------------------------------------


def test_resolve_dest_rejects_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    root, error = server._resolve_dest("renders")  # relative -> refused
    assert root is None
    assert "absolute" in error


def test_resolve_dest_rejects_dot_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    root, error = server._resolve_dest("./out")
    assert root is None and "absolute" in error


def test_resolve_dest_accepts_absolute(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    dest = tmp_path / "abs"
    root, error = server._resolve_dest(str(dest))
    assert error is None
    assert root == dest.resolve()


@pytest.mark.asyncio
async def test_save_output_rejects_relative_dest(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    monkeypatch.setattr(server._State, "client", MiniClient())
    monkeypatch.setattr(server._State, "session", Session(tmp_path / "s"))
    result = await server.save_output(filename="out_00001_.png", dest_dir="renders")
    assert "error" in result and "absolute" in result["error"]


@pytest.mark.asyncio
async def test_run_workflow_rejects_relative_save_dir(monkeypatch, tmp_path):
    session = Session(tmp_path / "s")
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    monkeypatch.setattr(server._State, "client", MiniClient())
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    result = await server.run_workflow(wf_id, return_preview=False, save_dir="out")
    assert result["status"] == "invalid" and "absolute" in result["error"]


# --- #3 _mount_status --------------------------------------------------------


def test_mount_status_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    status = server._mount_status()
    assert status["configured"] is False
    assert status["writable"] is False
    assert "COMFYUI_MOUNT_DIR" in status["hint"]


def test_mount_status_writable(monkeypatch, tmp_path):
    mount = tmp_path / "mount"
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, mount))
    status = server._mount_status()
    assert status["configured"] is True
    assert status["writable"] is True
    assert status["path"] == str(mount.resolve())
    # the probe file must not be left behind
    assert not (mount.resolve() / ".draftsman-write-probe").exists()


# --- #1 get_instance_info surfaces relocation --------------------------------


@pytest.mark.asyncio
async def test_get_instance_info_includes_relocation(monkeypatch, tmp_path):
    mount = tmp_path / "mount"
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, mount))
    monkeypatch.setattr(server._State, "client", StatsClient())
    info = await server.get_instance_info()
    assert info["relocation"]["configured"] is True
    assert info["relocation"]["writable"] is True


@pytest.mark.asyncio
async def test_get_instance_info_flags_missing_mount(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    monkeypatch.setattr(server._State, "client", StatsClient())
    info = await server.get_instance_info()
    assert info["relocation"]["configured"] is False
    assert "COMFYUI_MOUNT_DIR" in info["relocation"]["hint"]


# --- #5 capabilities resource ------------------------------------------------


def test_capabilities_resource_is_json_with_relocation(monkeypatch, tmp_path):
    mount = tmp_path / "mount"
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, mount))
    monkeypatch.setattr(server, "_COMFY_API_KEY", "")
    caps = json.loads(server.capabilities_resource())
    assert caps["relocation"]["configured"] is True
    assert caps["background_runs"] is True
    assert caps["partner_node_api_key"] is False
    assert caps["comfyui_url"] == BASE


# --- #4 check_setup doctor tool ----------------------------------------------


@pytest.mark.asyncio
async def test_check_setup_all_green(monkeypatch, tmp_path):
    mount = tmp_path / "mount"
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, mount))
    monkeypatch.setattr(server._State, "client", StatsClient())
    result = await server.check_setup()
    assert result["ok"] is True
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["comfyui"]["ok"] is True
    assert by_name["relocation"]["ok"] is True
    assert "hint" not in result


@pytest.mark.asyncio
async def test_check_setup_reachable_but_no_mount_is_soft(monkeypatch, tmp_path):
    # ComfyUI up, mount unset: overall ok (workflows still run), but a hint nudges
    # the user to configure relocation so renders can be delivered.
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, None))
    monkeypatch.setattr(server._State, "client", StatsClient())
    result = await server.check_setup()
    assert result["ok"] is True
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["relocation"]["ok"] is False
    assert "COMFYUI_MOUNT_DIR" in result["hint"]


@pytest.mark.asyncio
async def test_check_setup_unreachable_fails_hard(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", _cfg(tmp_path, tmp_path / "mount"))
    monkeypatch.setattr(server._State, "client", DownClient())
    result = await server.check_setup()
    assert result["ok"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["comfyui"]["ok"] is False
    assert "can't reach ComfyUI" in by_name["comfyui"]["detail"]
    assert "hint" in result
