"""Node catalog: compact search/summarize over the (huge) object_info document."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.comfy.catalog import node_summary, search_nodes

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def test_search_by_name_fragment(object_info):
    hits = search_nodes(object_info, "ksampler")
    names = [h["class_type"] for h in hits]
    assert "KSampler" in names
    assert "KSamplerAdvanced" in names


def test_search_matches_display_name_and_description(object_info):
    hits = search_nodes(object_info, "load checkpoint")
    assert any(h["class_type"] == "CheckpointLoaderSimple" for h in hits)


def test_search_filters_by_category(object_info):
    hits = search_nodes(object_info, "", category="loaders")
    assert all("loaders" in h["category"] for h in hits)
    assert any(h["class_type"] == "LoraLoader" for h in hits)


def test_search_results_are_compact(object_info):
    hits = search_nodes(object_info, "sampler")
    for h in hits:
        # a search hit must never embed full input schemas (combo lists are megabytes)
        assert set(h) <= {"class_type", "display_name", "description", "category", "output_node"}


def test_search_limit(object_info):
    hits = search_nodes(object_info, "", limit=3)
    assert len(hits) == 3


def test_node_summary_has_named_slots(object_info):
    s = node_summary(object_info, "KSampler")
    assert s["class_type"] == "KSampler"
    input_names = [i["name"] for i in s["inputs"]]
    assert "model" in input_names and "seed" in input_names
    seed = next(i for i in s["inputs"] if i["name"] == "seed")
    assert seed["widget"] is True
    model = next(i for i in s["inputs"] if i["name"] == "model")
    assert model["widget"] is False
    assert s["outputs"][0]["type"] == "LATENT"


def test_node_summary_truncates_giant_combos(object_info):
    s = node_summary(object_info, "KSampler")
    sampler = next(i for i in s["inputs"] if i["name"] == "sampler_name")
    assert sampler["type"] == "COMBO"
    assert len(sampler["choices"]) <= 24
    assert "euler" in sampler["choices"]


def test_node_summary_unknown_class(object_info):
    with pytest.raises(KeyError):
        node_summary(object_info, "NopeNode")


def test_node_summary_surfaces_dynamic_combo(object_info):
    summary = node_summary(object_info, "DA3Render")
    out = next(i for i in summary["inputs"] if i["name"] == "output")
    # the V3 combo is a widget with discoverable option keys + a default
    assert out["widget"] is True
    assert out["dynamic_combo"] is True
    assert out["type"] == "COMBO"
    assert set(out["choices"]) >= {"depth", "depth_colored", "sky_mask", "confidence"}
    assert out["default"] == "depth"
    # each option lists its dotted sub-widgets so the agent can set them
    depth_subs = {d["name"] for d in out["options"]["depth"]}
    assert depth_subs == {"output.normalization", "output.apply_sky_clip"}
    normalization = next(
        d for d in out["options"]["depth"] if d["name"] == "output.normalization"
    )
    assert normalization["default"] == "v2_style"
    assert "min_max" in normalization["choices"]


def test_node_summary_dynamic_combo_default_from_first_option(object_info):
    summary = node_summary(object_info, "DA3Inference")
    mode = next(i for i in summary["inputs"] if i["name"] == "mode")
    # no explicit top-level default -> first option key
    assert mode["default"] == "mono"
    assert mode["choices"] == ["mono", "multiview"]
