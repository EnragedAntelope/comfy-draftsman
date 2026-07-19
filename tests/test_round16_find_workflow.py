"""Round 16: find_workflow - cheap discovery over saved workflows.

list_workflows returns only names, so an agent reusing a saved workflow would have
to import+inspect each one (expensive) and usually just rebuilds. find_workflow
profiles each saved workflow SERVER-side and returns a few ranked, compact matches
so the token cost to the caller is bounded. These tests cover profile extraction
(family / base model / lora / resolution / feature tags, incl. hand-built graphs),
ranking against a natural-language intent, and the resilience/edge paths.
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.config import Config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _node(nid, ntype, widgets_values=None):
    return {"id": nid, "type": ntype, "widgets_values": widgets_values or []}


def _ui(nodes):
    return {"nodes": nodes, "links": [], "version": 0.4}


# A saved SDXL anime-portrait workflow with a lora and a (custom, not-in-schema)
# face detailer.
SDXL_ANIME = _ui(
    [
        _node(1, "CheckpointLoaderSimple", ["sd_xl_base_1.0.safetensors"]),
        _node(2, "LoraLoader", ["anime_style.safetensors", 1.0, 1.0]),
        _node(3, "CLIPTextEncode", ["anime portrait, highly detailed face"]),
        _node(4, "EmptyLatentImage", [832, 1216, 1]),
        _node(5, "KSampler", [123, 20, 7.0, "euler", "normal", 1.0]),
        _node(6, "VAEDecode", []),
        _node(7, "SaveImage", ["ComfyUI"]),
        _node(8, "FaceDetailer", []),  # custom node: not in object_info
    ]
)

# A saved FLUX landscape workflow with an upscale pass.
FLUX_UPSCALE = _ui(
    [
        _node(1, "UNETLoader", ["flux1-krea-dev.safetensors", "default"]),
        _node(2, "CLIPTextEncode", ["a wide mountain landscape at sunset"]),
        _node(3, "EmptyLatentImage", [1024, 1024, 1]),
        _node(4, "KSampler", [7, 20, 3.5, "euler", "simple", 1.0]),
        _node(5, "UpscaleModelLoader", ["4x-UltraSharp.pth"]),
        _node(6, "ImageUpscaleWithModel", []),
        _node(7, "VAEDecode", []),
        _node(8, "SaveImage", ["ComfyUI"]),
    ]
)

# A saved SD1.5 inpaint workflow - should not match a "flux portrait" intent.
SD15_INPAINT = _ui(
    [
        _node(1, "CheckpointLoaderSimple", ["realisticVision_v5.safetensors"]),
        _node(2, "LoadImage", ["photo.png", "image"]),
        _node(3, "VAEEncodeForInpaint", []),  # custom/inpaint node
        _node(4, "CLIPTextEncode", ["remove the object"]),
        _node(5, "KSampler", [1, 20, 7.0, "euler", "normal", 0.8]),
    ]
)


class FakeClient:
    """Serves canned userdata workflows + object_info to find_workflow."""

    def __init__(self, workflows, object_info):
        self._wfs = workflows  # {name: dict | Exception | non-dict}
        self._oi = object_info

    async def get_object_info(self, refresh: bool = False):
        return self._oi

    async def list_userdata_workflows(self):
        return list(self._wfs)

    async def get_userdata_workflow(self, name):
        data = self._wfs[name]
        if isinstance(data, Exception):
            raise data
        return data


def _install(monkeypatch, tmp_path, object_info, workflows):
    cfg = Config(
        comfyui_url="http://comfy.test",
        session_dir=tmp_path / "s",
        learned_dir=tmp_path / "learned",
    )
    monkeypatch.setattr(server._State, "config", cfg)
    monkeypatch.setattr(server._State, "client", FakeClient(workflows, object_info))


# --- profile extraction ------------------------------------------------------


def test_profile_sdxl_anime(object_info, tmp_path):
    p = server._profile_workflow("sdxl_anime", SDXL_ANIME, object_info, tmp_path / "learned")
    assert p["family"] == "sdxl"
    assert p["base_models"] == ["sd_xl_base_1.0.safetensors"]
    assert p["loras"] == ["anime_style.safetensors"]
    assert p["resolutions"] == ["832x1216"]
    assert "lora" in p["features"]
    assert "detailer" in p["features"]  # from the custom FaceDetailer class_type
    assert p["prompts"] and "anime portrait" in p["prompts"][0]


def test_profile_flux_upscale_and_no_model_misclassification(object_info, tmp_path):
    p = server._profile_workflow("flux_up", FLUX_UPSCALE, object_info, tmp_path / "learned")
    assert p["family"] == "flux"
    # the upscale model (model_name / .pth) must NOT be counted as a base model
    assert p["base_models"] == ["flux1-krea-dev.safetensors"]
    assert "4x-UltraSharp.pth" not in p["base_models"]
    assert "upscale" in p["features"]
    assert p["resolutions"] == ["1024x1024"]


def test_profile_inpaint_feature_not_img2img(object_info, tmp_path):
    p = server._profile_workflow("sd15_inpaint", SD15_INPAINT, object_info, tmp_path / "learned")
    assert "inpaint" in p["features"]
    # VAEEncodeForInpaint is not a plain VAEEncode, so this is inpaint, not img2img
    assert "img2img" not in p["features"]


def test_profile_malformed_raises(object_info, tmp_path):
    # a node without a type is malformed - _profile_workflow raises (KeyError on the
    # missing "type"); find_workflow turns that into a skip rather than a failure.
    with pytest.raises(KeyError):
        server._profile_workflow("bad", {"nodes": [{"id": 1}]}, object_info, tmp_path / "learned")


# --- ranking -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_ranks_relevant_first(monkeypatch, tmp_path, object_info):
    _install(
        monkeypatch,
        tmp_path,
        object_info,
        {"sdxl_anime": SDXL_ANIME, "flux_upscale": FLUX_UPSCALE, "sd15_inpaint": SD15_INPAINT},
    )
    result = await server.find_workflow("anime portrait at 1024 with a face detailer")
    names = [m["name"] for m in result["matches"]]
    assert names[0] == "sdxl_anime"  # portrait + face + detailer outweighs the rest
    top = result["matches"][0]
    assert "detailer" in top["matched"]
    assert top["family"] == "sdxl"
    assert "sd15_inpaint" not in names  # nothing about inpaint matched


@pytest.mark.asyncio
async def test_find_filters_out_nonmatching(monkeypatch, tmp_path, object_info):
    _install(
        monkeypatch,
        tmp_path,
        object_info,
        {"sdxl_anime": SDXL_ANIME, "flux_upscale": FLUX_UPSCALE},
    )
    result = await server.find_workflow("flux landscape at 1024")
    names = [m["name"] for m in result["matches"]]
    assert names == ["flux_upscale"]  # the sdxl portrait scores 0 and is dropped


@pytest.mark.asyncio
async def test_find_respects_limit(monkeypatch, tmp_path, object_info):
    _install(
        monkeypatch,
        tmp_path,
        object_info,
        {"sdxl_anime": SDXL_ANIME, "flux_upscale": FLUX_UPSCALE},
    )
    result = await server.find_workflow("1024", limit=1)
    assert len(result["matches"]) == 1


# --- resilience & edges ------------------------------------------------------


@pytest.mark.asyncio
async def test_find_skips_unreadable_and_malformed(monkeypatch, tmp_path, object_info):
    _install(
        monkeypatch,
        tmp_path,
        object_info,
        {
            "good": FLUX_UPSCALE,
            "gone": FileNotFoundError("gone"),  # fetch fails -> skipped
            "junk": ["not", "a", "workflow"],   # not a dict -> skipped
            "broken": {"nodes": [{"id": 1}]},    # malformed -> skipped
        },
    )
    result = await server.find_workflow("flux 1024")
    assert [m["name"] for m in result["matches"]] == ["good"]
    assert result["skipped"] == 3
    assert result["scanned"] == 1


@pytest.mark.asyncio
async def test_find_no_matches_gives_hint(monkeypatch, tmp_path, object_info):
    _install(monkeypatch, tmp_path, object_info, {"flux_upscale": FLUX_UPSCALE})
    result = await server.find_workflow("zzzznonsense qqqq")
    assert result["matches"] == []
    assert "list_workflows" in result["hint"]


@pytest.mark.asyncio
async def test_find_empty_intent_is_error(monkeypatch, tmp_path, object_info):
    _install(monkeypatch, tmp_path, object_info, {"flux_upscale": FLUX_UPSCALE})
    result = await server.find_workflow("   ")
    assert "error" in result


@pytest.mark.asyncio
async def test_find_no_saved_workflows(monkeypatch, tmp_path, object_info):
    _install(monkeypatch, tmp_path, object_info, {})
    result = await server.find_workflow("flux portrait")
    assert result["scanned"] == 0
    assert result["matches"] == []


@pytest.mark.asyncio
async def test_find_returns_only_compact_fields(monkeypatch, tmp_path, object_info):
    # guard the token budget: a match must not smuggle back the full graph.
    _install(monkeypatch, tmp_path, object_info, {"sdxl_anime": SDXL_ANIME})
    result = await server.find_workflow("anime portrait detailer")
    match = result["matches"][0]
    allowed = {
        "name", "score", "matched", "family", "base_models", "loras",
        "resolution", "features", "nodes", "prompt_hint",
    }
    assert set(match) <= allowed
    assert "nodes" not in match or isinstance(match["nodes"], int)  # a count, not a node list
