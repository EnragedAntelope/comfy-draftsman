"""Round-7 fixes: cwd-independent session dir, persist-failure tolerance,
session id hygiene, truncated widget previews, diagnose_workflow annotation.

Origin: a Claude Desktop user hit PermissionError because the MCP host launched
the server with cwd = C:\\Windows\\System32 and the session dir defaulted to
cwd-relative.
"""

from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.config import Config, load_config
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

# --- session dir default must never depend on cwd ---


def test_default_session_dir_is_home_based(monkeypatch, tmp_path):
    monkeypatch.delenv("DRAFTSMAN_SESSION_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    session_dir = load_config().session_dir
    assert session_dir == Path.home() / ".comfy-draftsman" / "sessions"
    assert not session_dir.is_relative_to(tmp_path)


def test_session_dir_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("DRAFTSMAN_SESSION_DIR", str(tmp_path / "elsewhere"))
    assert load_config().session_dir == tmp_path / "elsewhere"


# --- save_workflow: a failed local backup copy must not fail the save ---


class OkClient:
    async def get_object_info(self, refresh: bool = False):
        return {}

    async def save_userdata_workflow(self, name, document, overwrite: bool = False):
        return f"{name}.json"


async def test_save_survives_unwritable_session_dir(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed", encoding="utf-8")
    cfg = Config(comfyui_url="http://comfy.test", session_dir=blocker / "sessions")
    session = Session(cfg.session_dir)
    monkeypatch.setattr(server._State, "config", cfg)
    monkeypatch.setattr(server._State, "client", OkClient())
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")

    result = await server.save_workflow(wf_id, "menu")

    assert result["saved"] is True  # the ComfyUI-side save worked
    assert result["local_copy"] is None
    assert "DRAFTSMAN_SESSION_DIR" in result["note"]


# --- workflow ids must not be usable as relative paths ---


def test_load_rejects_path_like_ids(tmp_path):
    inside = tmp_path / "sessions"
    inside.mkdir()
    (tmp_path / "evil.json").write_text("{}", encoding="utf-8")
    session = Session(inside)
    with pytest.raises(KeyError):
        session.get("../evil")


# --- summary previews truncate long strings on ALL node types ---


class _ListNode:
    type = "DPRandomGenerator"  # not virtual

    def __init__(self):
        self.widgets_values = ["x" * 5000, 42]


class _DictNode:
    type = "SomeNode"

    def __init__(self):
        self.widgets_values = {"text": "y" * 300, "seed": 7}


def test_widget_preview_truncates_non_virtual_list_widgets():
    preview = server._widget_preview(_ListNode())
    assert preview[0].endswith("…") and len(preview[0]) == 121
    assert preview[1] == 42


def test_widget_preview_truncates_dict_widgets():
    preview = server._widget_preview(_DictNode())
    assert preview["text"].endswith("…") and len(preview["text"]) == 121
    assert preview["seed"] == 7


# --- diagnose_workflow is documented read-only; the annotation must say so ---


def test_diagnose_workflow_annotated_read_only():
    tool = server.mcp._tool_manager.get_tool("diagnose_workflow")
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
