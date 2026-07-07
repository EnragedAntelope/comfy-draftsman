"""Round-9 testing feedback fixes.

1. Seed control_after_generate off-by-one: the frontend appends a control
   widget after any INT named seed/noise_seed even when the schema never asks
   for it (legacy V1 nodes: DPRandomGenerator, EA_LMStudio...), so imported
   workflows carry an extra widgets_values slot the schema doesn't declare.
   Slot mapping must account for it or every widget after the seed misreads.
2. Combo choice truncation: get_node_info capped combo lists at 24 with no way
   to see the rest -> choices_filter / max_choices.
3. Subgraph-packaged templates: "definitions" must survive round-trips, the
   internals must be inspectable, and to_api/validate must say "subgraph"
   instead of "missing node class".
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman.comfy.catalog import node_summary
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate
from comfy_draftsman.server import _subgraph_summary

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def oi():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


# --- 1. seed control_after_generate name heuristic ----------------------------


def test_unflagged_seed_gets_control_slot(oi):
    # DPRandomGenerator's schema has no control_after_generate flag, but the
    # frontend serializes [text, seed, <control>, autorefresh]
    slots = w.widget_slot_names("DPRandomGenerator", oi)
    assert slots == ["text", "seed", "seed__control_after_generate", "autorefresh"]


def test_imported_seed_workflow_aligns(oi):
    named = w.widgets_to_named(
        "DPRandomGenerator", ["{a|b}", 966, "randomize", "No"], oi
    )
    assert named == {
        "text": "{a|b}",
        "seed": 966,
        "seed__control_after_generate": "randomize",
        "autorefresh": "No",
    }


def test_validate_accepts_real_frontend_serialization(oi):
    wf = Workflow.new()
    n = wf.add_node("DPRandomGenerator", object_info=oi)
    n.widgets_values = ["{a|b}", 966, "randomize", "No"]  # as the frontend saves it
    findings = validate(wf, oi)
    # no drift, and autorefresh="No" must NOT be misread against another slot
    assert not [f for f in findings if f["code"] in ("widget-count-drift", "invalid-combo-value")]


def test_to_api_strips_the_synthetic_control_slot(oi):
    wf = Workflow.new()
    n = wf.add_node("DPRandomGenerator", object_info=oi)
    n.widgets_values = ["{a|b}", 966, "randomize", "No"]
    inputs = wf.to_api(oi)[str(n.id)]["inputs"]
    assert inputs == {"text": "{a|b}", "seed": 966, "autorefresh": "No"}


def test_set_widget_stays_aligned_past_the_seed(oi):
    wf = Workflow.new()
    n = wf.add_node("DPRandomGenerator", object_info=oi)
    wf.set_widget(n.id, "autorefresh", "Yes", oi)
    assert w.widgets_to_named("DPRandomGenerator", n.widgets_values, oi)["autorefresh"] == "Yes"
    assert n.widgets_values[2] in ("fixed", "randomize")  # control slot intact


def test_explicit_false_flag_opts_out():
    assert w.has_control_slot("seed", ["INT", {"default": 0}])
    assert w.has_control_slot("noise_seed", ["INT", {}])
    assert not w.has_control_slot("seed", ["INT", {"control_after_generate": False}])
    assert not w.has_control_slot("seed", ["FLOAT", {}])  # INT only
    assert not w.has_control_slot("steps", ["INT", {}])  # name must match


def test_flagged_seed_unchanged(oi):
    # KSampler declares the flag; behavior must not double up
    slots = w.widget_slot_names("KSampler", oi)
    assert slots.count("seed__control_after_generate") == 1


# --- 2. combo choice filtering / cap ------------------------------------------

_BIG_COMBO_OI = {
    "FontNode": {
        "input": {
            "required": {
                "font": [[f"font_{i:02d}.ttf" for i in range(88)], {}],
            }
        },
        "output": [],
    }
}


def _font_entry(**kwargs):
    summary = node_summary(_BIG_COMBO_OI, "FontNode", **kwargs)
    return summary["inputs"][0]


def test_truncated_combo_says_how_to_see_more():
    entry = _font_entry()
    assert len(entry["choices"]) == 24
    assert entry["choices_truncated"] == 88
    assert "choices_filter" in entry["choices_hint"]


def test_choices_filter_narrows():
    entry = _font_entry(choices_filter="_4")
    assert entry["choices"] == [f"font_4{i}.ttf" for i in range(10)]
    assert entry["choices_matched"] == 10
    assert entry["choices_total"] == 88
    assert "choices_truncated" not in entry


def test_max_choices_raises_the_cap():
    entry = _font_entry(max_choices=100)
    assert len(entry["choices"]) == 88
    assert "choices_truncated" not in entry


# --- 3. subgraph-packaged workflows -------------------------------------------

SUBGRAPH_ID = "e5cfe5ba-2ae0-4bc4-869f-ab2228cb44d3"


def _subgraph_doc():
    """Minimal schema-1.0 doc shaped like ComfyUI's subgraph-packaged templates."""
    return {
        "id": "11111111-2222-4333-8444-555555555555",
        "revision": 0,
        "nodes": [
            {
                "id": 1,
                "type": SUBGRAPH_ID,
                "pos": [0, 0],
                "size": [300, 200],
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": [],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "pos": [400, 0],
                "size": [300, 200],
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "widgets_values": ["ComfyUI"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        "groups": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": SUBGRAPH_ID,
                    "name": "Text to Image (Qwen-Image)",
                    "inputs": [{"name": "text"}, {"name": "seed"}],
                    "outputs": [{"name": "IMAGE"}],
                    "nodes": [
                        {
                            "id": 3,
                            "type": "KSampler",
                            "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
                            "inputs": [
                                {"name": "model", "type": "MODEL", "link": None},
                                {"name": "positive", "type": "CONDITIONING", "link": None},
                                {"name": "negative", "type": "CONDITIONING", "link": None},
                                {"name": "latent_image", "type": "LATENT", "link": None},
                            ],
                            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9]}],
                        },
                        {
                            "id": 8,
                            "type": "VAEDecode",
                            "widgets_values": [],
                            "inputs": [
                                {"name": "samples", "type": "LATENT", "link": 9},
                                {"name": "vae", "type": "VAE", "link": None},
                            ],
                            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                        },
                    ],
                    "links": [
                        {"id": 9, "origin_id": 3, "origin_slot": 0,
                         "target_id": 8, "target_slot": 0, "type": "LATENT"},
                        # boundary wiring: -10 = subgraph input, -20 = subgraph output
                        {"id": 16, "origin_id": 8, "origin_slot": 0,
                         "target_id": -20, "target_slot": 0, "type": "IMAGE"},
                    ],
                }
            ]
        },
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def test_definitions_survive_the_round_trip():
    wf = Workflow.from_ui(_subgraph_doc())
    out = wf.to_ui()
    assert out["definitions"] == _subgraph_doc()["definitions"]
    assert SUBGRAPH_ID in wf.subgraph_defs()


def test_plain_workflows_gain_no_definitions_key():
    assert "definitions" not in Workflow.new().to_ui()


def test_validate_reports_subgraph_not_missing_class(oi):
    wf = Workflow.from_ui(_subgraph_doc())
    findings = validate(wf, oi)
    codes = {f["code"] for f in findings}
    assert "subgraph-instance" in codes
    assert "missing-node-class" not in codes
    sub = next(f for f in findings if f["code"] == "subgraph-instance")
    # round 10: instances validate FLATTENED, so the per-instance finding is
    # informational, and inner-node findings carry subgraph provenance
    assert sub["level"] == "info"
    assert "Text to Image (Qwen-Image)" in sub["message"]
    inner = [f for f in findings if f.get("subgraph") == "Text to Image (Qwen-Image)" and f is not sub]
    assert inner, "inner nodes should be validated (fixture KSampler has unconnected inputs)"
    assert all(f["inner_node"].startswith("1:") for f in inner)


def test_to_api_flattens_the_subgraph(oi):
    wf = Workflow.from_ui(_subgraph_doc())
    api = wf.to_api(oi)
    # instance replaced by its inner KSampler+VAEDecode, wired through to SaveImage
    classes = {entry["class_type"] for entry in api.values()}
    assert {"KSampler", "VAEDecode", "SaveImage"} <= classes
    save = next(e for e in api.values() if e["class_type"] == "SaveImage")
    decode_id = next(k for k, e in api.items() if e["class_type"] == "VAEDecode")
    assert save["inputs"]["images"] == [decode_id, 0]


def test_subgraph_summary_exposes_internals():
    sg = _subgraph_doc()["definitions"]["subgraphs"][0]
    summary = _subgraph_summary(sg)
    assert summary["name"] == "Text to Image (Qwen-Image)"
    assert summary["inputs"] == ["text", "seed"]
    assert {n["class_type"] for n in summary["nodes"]} == {"KSampler", "VAEDecode"}
    assert any(s.startswith("#3[0] -> #8.") for s in summary["links"])  # wiring listed
