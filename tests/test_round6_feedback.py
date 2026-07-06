"""Round-6 feedback fixes: workflow-browser import, list_models filtering,
organize_workflow change reporting."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.layout import apply_layout
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

BASE = "http://comfy.test"
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- client


@pytest.fixture
def client(config):
    return ComfyClient(config)


@respx.mock
async def test_list_userdata_workflows(client):
    route = respx.get(f"{BASE}/api/userdata").mock(
        return_value=httpx.Response(200, json=["menu.json", "sub\\old.json"])
    )
    names = await client.list_userdata_workflows()
    assert names == ["menu.json", "sub/old.json"]
    url = str(route.calls[0].request.url)
    assert "dir=workflows" in url and "recurse=true" in url


@respx.mock
async def test_list_userdata_workflows_missing_dir_is_empty(client):
    respx.get(f"{BASE}/api/userdata").mock(return_value=httpx.Response(404))
    assert await client.list_userdata_workflows() == []


@respx.mock
async def test_get_userdata_workflow_appends_json_and_quotes(client):
    route = respx.get(url__regex=rf"{BASE}/api/userdata/.*").mock(
        return_value=httpx.Response(200, json={"nodes": []})
    )
    wf = await client.get_userdata_workflow("my menu")
    assert wf == {"nodes": []}
    assert "workflows%2Fmy%20menu.json" in str(route.calls[0].request.url)


@respx.mock
async def test_get_userdata_workflow_404_raises_filenotfound(client):
    respx.get(url__regex=rf"{BASE}/api/userdata/.*").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(FileNotFoundError):
        await client.get_userdata_workflow("nope")


@pytest.mark.parametrize("bad", ["../secrets", "..\\secrets", "a/../../b", ""])
async def test_get_userdata_workflow_rejects_traversal(client, bad):
    with pytest.raises(ValueError):
        await client.get_userdata_workflow(bad)


# ---------------------------------------------------------------- server tools


class FakeClient:
    def __init__(self):
        self.workflows = {
            "menu.json": {"nodes": [], "links": []},
            "subdir/legacy.json": {"nodes": [], "links": []},
        }
        self.models = {
            "checkpoints": ["SDXL\\juggernaut.safetensors", "flux\\flux1-dev.safetensors"],
            "loras": ["detail-tweaker.safetensors", "sdxl\\offset.safetensors"],
        }

    async def get_object_info(self, refresh: bool = False):
        return {}

    async def list_model_folders(self):
        return list(self.models)

    async def list_models(self, folder):
        return self.models[folder]

    async def list_userdata_workflows(self):
        return list(self.workflows)

    async def get_userdata_workflow(self, name):
        clean = name.replace("\\", "/").strip("/")
        if not clean or ".." in clean.split("/"):
            raise ValueError(f"invalid workflow name: {name!r}")
        filename = clean if clean.endswith(".json") else f"{clean}.json"
        if filename not in self.workflows:
            raise FileNotFoundError(name)
        return self.workflows[filename]


@pytest.fixture
def wired(tmp_path, config, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", Session(tmp_path / "sessions"))
    return client


async def test_list_models_search_filters(wired):
    result = await server.list_models(folder="loras", search="detail")
    assert result["files"] == ["detail-tweaker.safetensors"]
    assert result["count"] == 1
    assert result["search"] == "detail"


async def test_list_models_search_is_case_insensitive(wired):
    result = await server.list_models(folder="checkpoints", search="FLUX")
    assert result["files"] == ["flux\\flux1-dev.safetensors"]


async def test_list_models_no_search_returns_all(wired):
    result = await server.list_models(folder="checkpoints")
    assert result["count"] == 2
    assert "search" not in result


async def test_list_workflows_and_search(wired):
    all_ = await server.list_workflows()
    assert all_["workflows"] == ["menu", "subdir/legacy"]
    hit = await server.list_workflows(search="LEG")
    assert hit["workflows"] == ["subdir/legacy"]
    assert hit["count"] == 1


async def test_import_workflow_by_name(wired):
    result = await server.import_workflow(name="menu")
    assert "workflow_id" in result


async def test_import_workflow_by_name_missing_hints_list_workflows(wired):
    result = await server.import_workflow(name="ghost")
    assert "list_workflows" in result["hint"]


async def test_import_workflow_requires_exactly_one_source(wired):
    neither = await server.import_workflow()
    both = await server.import_workflow(workflow_json="{}", name="menu")
    assert "exactly one" in neither["error"]
    assert "exactly one" in both["error"]


# ---------------------------------------------------------------- organize report


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def test_annotate_reports_applied_changes(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    apply_layout(wf, object_info)

    report = annotate(wf, object_info)
    applied = report["applied"]
    assert applied["groups"] == [g.title for g in wf.groups] and applied["groups"]
    assert applied["guidance_notes_added"] >= 1
    assert applied["nodes_retitled"] >= 1  # positive prompt got a role title
    assert applied["knobs_highlighted_green"] >= 1
    assert "reposition" in applied["layout"]
