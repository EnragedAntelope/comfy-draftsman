"""Comfy Registry client: resolve missing node classes to installable packs."""

import httpx
import pytest
import respx

from comfy_draftsman.comfy.registry import RegistryClient

BASE = "http://registry.test"


@pytest.fixture
def registry(config):
    return RegistryClient(config)


@respx.mock
async def test_resolve_node_class_to_pack(registry):
    respx.get(f"{BASE}/comfy-nodes/FaceDetailer/node").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "comfyui-impact-pack",
                "name": "ComfyUI Impact Pack",
                "description": "Detailer nodes...",
                "repository": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
                "downloads": 1000000,
                "latest_version": {"version": "8.1.0"},
                "publisher": {"id": "ltdrdata"},
            },
        )
    )
    result = await registry.resolve_node_class("FaceDetailer")
    assert result["pack_id"] == "comfyui-impact-pack"
    assert result["repository"].startswith("https://github.com")
    assert "registry.comfy.org" in result["registry_url"]
    assert "comfy node install comfyui-impact-pack" in result["install_hint"]


@respx.mock
async def test_resolve_unknown_class_returns_none(registry):
    respx.get(f"{BASE}/comfy-nodes/NotARealNode/node").mock(
        return_value=httpx.Response(404, json={"message": "not found"})
    )
    assert await registry.resolve_node_class("NotARealNode") is None


@respx.mock
async def test_resolve_many_dedupes_packs(registry):
    for cls in ("FaceDetailer", "SAMLoader"):
        respx.get(f"{BASE}/comfy-nodes/{cls}/node").mock(
            return_value=httpx.Response(
                200, json={"id": "comfyui-impact-pack", "name": "Impact", "repository": "r"}
            )
        )
    results = await registry.resolve_node_classes(["FaceDetailer", "SAMLoader"])
    assert set(results["resolved"]) == {"FaceDetailer", "SAMLoader"}
    assert len(results["packs"]) == 1


@respx.mock
async def test_search_packs(registry):
    respx.get(f"{BASE}/nodes/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "nodes": [
                    {"id": "comfyui-impact-pack", "name": "Impact Pack", "description": "d",
                     "downloads": 5, "repository": "r"}
                ],
                "total": 1,
            },
        )
    )
    hits = await registry.search_packs("face detail")
    assert hits[0]["pack_id"] == "comfyui-impact-pack"
