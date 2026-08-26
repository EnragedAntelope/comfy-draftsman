"""Validation fixes round: coerced UI dicts, widget counts, run timeout
surfacing prompt_id, and export/import file paths."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import respx

import comfy_draftsman.server as server
from comfy_draftsman.comfy.catalog import node_summary
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph.model import Workflow

BASE = "http://comfy.test"

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))



# --- Bug 1: string-serialized UI dict fields must coerce, not crash ---------

def test_from_ui_coerces_string_properties_and_flags_to_dict():
    ui = {
        "nodes": [
            {"id": 1, "type": "Note", "properties": "{}", "flags": "", "widgets_values": ["hi"]},
        ],
        "links": [],
        "extra": "{\"a\": 1}",
        "config": "",
    }
    wf = Workflow.from_ui(ui)
    assert isinstance(wf.nodes[1].properties, dict)
    assert isinstance(wf.nodes[1].flags, dict)
    assert isinstance(wf.extra, dict)
    assert isinstance(wf.config, dict)


# --- Nice-to-have 1: node_summary exposes canonical widget count -----------

def test_node_summary_exposes_widget_and_input_counts(object_info):
    ks = node_summary(object_info, "KSampler")
    assert ks["input_count"] == len(ks["inputs"])
    assert ks["widget_count"] == sum(1 for i in ks["inputs"] if i["widget"])
    assert ks["widget_count"] > 0


# --- Bug 3: a wait=True timeout must surface the queued prompt_id ---------

@respx.mock
async def test_run_and_wait_timeout_returns_prompt_id(config, monkeypatch):
    client = ComfyClient(config)
    respx.post(f"{BASE}/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "p-1"})
    )
    respx.get(f"{BASE}/history/p-1").mock(return_value=httpx.Response(200, json={}))

    class _FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def recv(self):
            await asyncio.sleep(30)  # longer than the timeout below

    def _fake_connect(*a, **k):
        return _FakeWS()

    monkeypatch.setattr("comfy_draftsman.comfy.client.websockets.connect", _fake_connect)
    monkeypatch.setattr(client, "_ws_url", lambda client_id=None: "ws://comfy.test/ws")

    result = await client.run_and_wait({"1": {"class_type": "X", "inputs": {}}}, timeout=0.05)
    assert result["status"] == "timeout"
    assert result["prompt_id"] == "p-1"
    assert "get_run_status" in result["hint"]


# --- Bug 4: export_workflow_json can write to a file ----------------------

async def test_export_workflow_json_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_wf", lambda wid: Workflow.new())
    monkeypatch.setattr(server, "_object_info", lambda: {})
    out = tmp_path / "w.json"
    result = await server.export_workflow_json("x", path=str(out))
    assert result["saved_to"] == str(out)
    assert out.exists()
    assert isinstance(json.loads(out.read_text()), dict)


# --- Bug 5: import_workflow can read a local file -------------------------

async def test_import_workflow_reads_local_file(monkeypatch, tmp_path):
    ui = {"nodes": [{"id": 1, "type": "Note", "widgets_values": ["hi"]}], "links": []}
    p = tmp_path / "in.json"
    p.write_text(json.dumps(ui))

    created = {}

    class _Sess:
        def create(self, wf, title=None):
            created["wf"] = wf
            return "new-id"

    monkeypatch.setattr(server, "_session", lambda: _Sess())
    monkeypatch.setattr(server, "_object_info", lambda: {})
    monkeypatch.setattr(server, "_summary", lambda wid, wf: {"workflow_id": wid})

    result = await server.import_workflow(file_path=str(p))
    assert result["workflow_id"] == "new-id"
    assert created["wf"].nodes[1].type == "Note"
