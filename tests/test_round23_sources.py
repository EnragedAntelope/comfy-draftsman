"""Round 23 / Part D3: curated model-file download sources.

organize_workflow's Models note can name where to get a loader's files - but
ONLY when a family YAML (bundled or learned via record_learning) curated a
real URL for it. The hard rule: never synthesize a URL. If nothing was
curated for a filename, the note says nothing about it."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import knowledge
from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _note_texts(wf) -> list[str]:
    return [
        n.widgets_values[0]
        for n in wf.nodes.values()
        if n.type == "MarkdownNote" and n.widgets_values
    ]


# --- knowledge.matching_sources: pure matching logic ---


def test_matching_sources_hits_on_filename_pattern():
    guidance = {
        "sources": [
            {"match": ["ae.safetensors"], "what": "FLUX VAE", "url": "https://example.test/ae"},
            {"match": ["clip_l"], "what": "CLIP-L", "url": "https://example.test/clip_l"},
        ]
    }
    hits = knowledge.matching_sources(guidance, ["FLUX/ae.safetensors"])
    assert hits == [{"what": "FLUX VAE", "url": "https://example.test/ae"}]


def test_matching_sources_empty_when_nothing_curated():
    assert knowledge.matching_sources({}, ["some_model.safetensors"]) == []


def test_matching_sources_empty_when_no_filename_matches():
    guidance = {"sources": [{"match": ["ae.safetensors"], "what": "x", "url": "https://x.test"}]}
    assert knowledge.matching_sources(guidance, ["totally_unrelated.safetensors"]) == []


# --- annotate integration: only real, curated URLs ever appear ---


def test_models_note_lists_a_curated_source(object_info):
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "zonkmodel_v1.safetensors", object_info)

    class _StaticGuidance(dict):
        pass

    # patch knowledge.get_guidance indirectly via a learned overlay so the
    # real annotate() code path (not a mock) exercises matching_sources
    import comfy_draftsman.knowledge as k

    orig_get_guidance = k.get_guidance
    orig_detect = k.detect_family_detail

    def fake_detect(_wf, _oi, learned_dir=None):
        return {"family": "zonkmodel", "matched_on": "zonkmodel", "widget": "unet_name"}

    def fake_guidance(family, model_filename=None, learned_dir=None):
        return {
            "family": "zonkmodel",
            "display_name": "Zonk Model",
            "sources": [
                {
                    "match": ["zonkmodel_v1.safetensors"],
                    "what": "Zonk checkpoint",
                    "url": "https://huggingface.co/zonk/zonkmodel",
                }
            ],
        }

    import comfy_draftsman.graph.annotate as a

    a.knowledge.detect_family_detail = fake_detect
    a.knowledge.get_guidance = fake_guidance
    try:
        annotate(wf, object_info)
    finally:
        a.knowledge.detect_family_detail = orig_detect
        a.knowledge.get_guidance = orig_get_guidance

    joined = "\n".join(_note_texts(wf))
    assert "https://huggingface.co/zonk/zonkmodel" in joined
    assert "Zonk checkpoint" in joined


def test_models_note_never_invents_a_url_when_nothing_curated(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    annotate(wf, object_info)  # sdxl.yaml has no sources: block (as of this round)
    joined = "\n".join(_note_texts(wf))
    assert "http://" not in joined
    assert "https://" not in joined


# --- bundled family YAML: every URL that exists must actually resolve ---


def test_bundled_sources_urls_are_https_and_look_real():
    """A defense-in-depth shape check (not a network check - CI has none):
    every curated URL in the shipped floor must be a real-looking https link,
    not a placeholder. Real resolution is verified manually before commit,
    per the round-23 plan's 'never synthesize a URL' rule."""
    for family in knowledge.list_families():
        guidance = knowledge.get_guidance(family)
        for entry in guidance.get("sources") or []:
            assert entry.get("url", "").startswith("https://"), (family, entry)
            assert entry.get("what"), (family, entry)
            assert entry.get("match"), (family, entry)
