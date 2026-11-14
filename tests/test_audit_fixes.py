"""2026-11-13 repo audit: save name suffix, sweep sheet pruning, dedupe fallback."""

import os
import time

from comfy_draftsman import server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.outputs import SWEEP_SHEET_MAX_AGE_S, dedupe_path, prune_sweep_sheets
from comfy_draftsman.session import Session


async def test_save_with_json_suffix_does_not_double_it(tmp_path, config, monkeypatch):
    tried: list[str] = []

    class Client:
        async def get_object_info(self, refresh=False):
            return {}

        async def save_userdata_workflow(self, name, document, overwrite=False):
            tried.append(name)
            if name == "portrait":
                raise FileExistsError(name)
            return f"{name}.json"

    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", Client())
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.new(), title="t")
    result = await server.save_workflow(wf_id, "portrait.json", allow_invalid=True)
    assert tried == ["portrait", "portrait (draftsman)"]
    assert result["saved_to_comfyui"].startswith("workflows/portrait (draftsman).json")
    assert "empty" in (await server.save_workflow(wf_id, ".json"))["error"]


def test_prune_removes_only_old_sweep_sheets(tmp_path):
    old = time.time() - SWEEP_SHEET_MAX_AGE_S - 60
    for name in ("contact-1.png", "crops-1-c1.png", "crop-guide-1.png", "mine.png"):
        (tmp_path / name).write_bytes(b"x")
        os.utime(tmp_path / name, (old, old))
    (tmp_path / "contact-new.png").write_bytes(b"x")
    prune_sweep_sheets(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["contact-new.png", "mine.png"]


def test_dedupe_never_returns_an_existing_path(tmp_path, monkeypatch):
    target = tmp_path / "a.png"
    target.write_bytes(b"x")
    assert dedupe_path(target) == tmp_path / "a_1.png"
    monkeypatch.setattr("comfy_draftsman.outputs.range", lambda *a: [], raising=False)
    fallback = dedupe_path(target)
    assert fallback != target and not fallback.exists() and fallback.suffix == ".png"
