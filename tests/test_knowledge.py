"""Knowledge floor: per-family optimization guidance, variant-aware (turbo/lightning/...)."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.knowledge import (
    detect_family,
    get_guidance,
    list_families,
    save_learning,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def test_families_include_core_set():
    families = list_families()
    for expected in ("sd15", "sdxl", "flux", "wan", "qwen_image"):
        assert expected in families


def test_guidance_has_sampling_floor():
    g = get_guidance("sdxl")
    assert g["sampling"]["cfg"]["default"] > 1
    assert g["sampling"]["steps"]["default"] >= 20
    assert g["resolutions"]
    assert "1024x1024" in g["resolutions"]
    assert g["loader"] == "checkpoint"


def test_guidance_includes_research_directive():
    g = get_guidance("flux")
    assert "research" in g
    assert len(g["research"]) > 40  # a real instruction, not a stub


def test_variant_override_turbo_cfg():
    g = get_guidance("sdxl", model_filename="sd_xl_turbo_1.0_fp16.safetensors")
    assert g["sampling"]["cfg"]["default"] == 1.0
    assert g["sampling"]["steps"]["default"] <= 8
    assert g["variant"] == "turbo"


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        get_guidance("definitely_not_a_family")


def test_detect_family_from_checkpoint_widget(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    assert detect_family(wf, object_info) == "sdxl"


def test_detect_family_survives_lying_merge_names(object_info):
    # a real-world SDXL/Pony merge with 'Flux' in its marketing name, loaded
    # through a checkpoint loader - must detect as sdxl, not flux
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(
        ckpt.id, "ckpt_name", "SDXL\\gonzalomoXLFluxPony_v60PhotoXLDMD.safetensors", object_info
    )
    assert detect_family(wf, object_info) == "sdxl"


def test_detect_family_returns_none_when_unknown(object_info):
    wf = Workflow.new()
    wf.add_node("VAEDecode", object_info=object_info)
    assert detect_family(wf, object_info) is None


# --- technique blocks: settings beyond the sampler (e.g. FaceDetailer) ---


def test_face_detailer_settings_differ_per_family():
    sdxl = get_guidance("sdxl")["techniques"]["face_detailer"]
    flux = get_guidance("flux")["techniques"]["face_detailer"]
    assert sdxl["cfg"] != flux["cfg"]  # universal detailer settings don't exist
    assert flux["cfg"] == 1.0
    assert 0 < sdxl["denoise"] < 1


def test_variant_overrides_reach_techniques():
    turbo = get_guidance("sdxl", model_filename="realvisxl_turbo.safetensors")
    assert turbo["techniques"]["face_detailer"]["cfg"] == 1.0


# --- learned overlay: research recorded in one session persists to the next ---


def test_save_learning_then_guidance_reflects_it(tmp_path):
    save_learning(
        tmp_path,
        "sdxl",
        {"techniques": {"face_detailer": {"denoise": 0.42}}},
        source="civitai model page for XYZ",
    )
    g = get_guidance("sdxl", learned_dir=tmp_path)
    assert g["techniques"]["face_detailer"]["denoise"] == 0.42
    # untouched floor keys survive the merge
    assert g["sampling"]["steps"]["default"] >= 20
    assert g["learned_sources"]


def test_learning_merges_incrementally(tmp_path):
    save_learning(tmp_path, "flux", {"notes": {"sampling": "new finding A"}}, source="a")
    save_learning(tmp_path, "flux", {"techniques": {"upscale": {"denoise": 0.25}}}, source="b")
    g = get_guidance("flux", learned_dir=tmp_path)
    assert g["notes"]["sampling"] == "new finding A"
    assert g["techniques"]["upscale"]["denoise"] == 0.25
    assert len(g["learned_sources"]) == 2


def test_learning_for_new_family_creates_entry(tmp_path):
    save_learning(tmp_path, "brand_new_model", {"sampling": {"cfg": {"default": 2.0}}}, source="x")
    g = get_guidance("brand_new_model", learned_dir=tmp_path)
    assert g["sampling"]["cfg"]["default"] == 2.0
    assert "brand_new_model" in list_families(learned_dir=tmp_path)
