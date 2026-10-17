import base64
import copy
import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from comfy_draftsman import server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

OI = {
    "LoadImage": {
        "input": {"required": {"image": [["old.png"], {"image_upload": True}]}},
        "output": ["IMAGE"],
        "output_node": True,
    },
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["installed.safetensors"]]}},
        "output": ["MODEL"],
    },
}


class Client:
    def __init__(self):
        self.oi = copy.deepcopy(OI)
        self.refreshes = []

    async def get_object_info(self, refresh=False):
        self.refreshes.append(refresh)
        if refresh:
            self.oi["LoadImage"]["input"]["required"]["image"][0].append("new.png")
        return self.oi

    async def upload_image(self, *args, **kwargs):
        return {"name": "new.png"}

    upload_mask = upload_image

    async def run_and_wait(self, api, **kwargs):
        return {
            "status": "success", "prompt_id": "p1",
            "outputs": [{"filename": "render.png", "type": "output", "kind": "images"}],
        }

    async def fetch_output(self, item):
        data = io.BytesIO()
        Image.new("RGB", (8, 8)).save(data, format="PNG")
        return data.getvalue()


@pytest.fixture
def env(config, monkeypatch):
    client = Client()
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "session", Session(config.session_dir))
    monkeypatch.setattr(server._State, "client", client)
    wf = Workflow.new()
    node = wf.add_node("LoadImage", object_info=OI)
    wf_id = server._session().create(wf, title="feedback")
    return client, wf, node, wf_id


@pytest.mark.parametrize("mask", [False, True])
async def test_upload_refreshes_and_edit_accepts_new_file(env, mask):
    client, wf, node, wf_id = env
    result = await server.upload_image(
        image_base64=base64.b64encode(b"image").decode(),
        mask_for={"filename": "old.png"} if mask else None,
    )
    assert "uploaded" in result
    assert client.refreshes == [True]
    result = await server.edit_workflow(wf_id, [
        {"op": "set_widget", "node_id": node.id, "input": "image", "value": "new.png"},
    ])
    assert "error" not in result
    assert wf.get_widget(node.id, "image", client.oi) == "new.png"


async def test_upload_refresh_failure_keeps_success(env, monkeypatch):
    client, *_ = env
    monkeypatch.setattr(client, "get_object_info", AsyncMock(side_effect=RuntimeError("offline")))
    assert "uploaded" in await server.upload_image(image_base64="aQ==")


@pytest.mark.parametrize("failure", [
    {"op": "add_node", "class_type": "Missing"},
    {"op": "set_widget", "node_id": 999, "input": "image", "value": "old.png"},
])
async def test_partial_batch_is_honest_and_persists(env, failure):
    _, wf, node, wf_id = env
    result = await server.edit_workflow(wf_id, [
        {"op": "set_widget", "node_id": node.id, "input": "image", "value": "render.png [output]"},
        failure,
        {"op": "remove_node", "node_id": node.id},
    ])
    assert len(result["applied"]) == 1
    assert result["note"]
    assert "graph unchanged" not in str(result)
    assert node.id in wf.nodes
    restored = Session(server._config().session_dir)
    assert node.id in restored.get(wf_id).nodes
    assert restored.get(wf_id).get_widget(node.id, "image", OI) == "render.png [output]"


@pytest.mark.parametrize("missing", [False, True])
async def test_template_reports_missing_files(env, monkeypatch, missing):
    client, *_ = env
    wf = Workflow.new()
    node = wf.add_node("CheckpointLoaderSimple", object_info=OI)
    if missing:
        wf.set_widget(node.id, "ckpt_name", "missing.safetensors", OI)
    monkeypatch.setattr(client, "get_template_workflow", AsyncMock(return_value=wf.to_ui()), raising=False)
    result = await server.create_workflow("template", template="test")
    assert bool(result.get("template_findings")) is missing
    assert client.refreshes == [True]


@pytest.mark.parametrize("save", [False, True])
async def test_sweep_refreshes_file_choices_and_saves_only_when_requested(env, tmp_path, monkeypatch, save):
    client, _, node, wf_id = env
    relocate = AsyncMock(return_value=([str(tmp_path / "render.png")], ["test save warning"]))
    monkeypatch.setattr(server, "_relocate_outputs", relocate)
    result = await server.run_workflow(
        wf_id, front=True, return_preview=False,
        save_dir=str(tmp_path / "renders") if save else "",
        sweep={"variants": [{"label": "new", "ops": [
            {"op": "set_widget", "node_id": node.id, "input": "image", "value": "new.png"},
        ]}]},
    )
    body = result[0] if isinstance(result, list) else result
    assert body["runs"][0]["status"] == "success"
    assert body["hint"]
    assert True in client.refreshes
    assert relocate.await_count == int(save)
    if save:
        assert body["runs"][0]["saved_paths"] == [str(tmp_path / "render.png")]
        assert body["runs"][0]["save_errors"] == ["test save warning"]
