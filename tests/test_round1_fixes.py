"""Regression tests for the round-1 fixes (post first real port test):

- valid workflow uuid in to_ui (no more "Invalid uuid at id" on open)
- connecting a link into a freshly-added node's widget input (STRING/INT)
- ConditioningZeroOut-aware positive/negative prompt titling
- krea2 family detection winning over FLUX's generic "krea" pattern
- graceful port_workflow on an unknown family
- record_learning seeding detection for a brand-new family
- run_workflow output collection helper
"""

import json
import re
from pathlib import Path

import pytest

from comfy_draftsman import knowledge
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph.annotate import _prompt_role, _title_nodes
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.port import port_workflow

FIXTURES = Path(__file__).parent / "fixtures"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# minimal schemas the shared trimmed fixture doesn't ship, kept accurate to the
# real ComfyUI classes they stand in for
_EXTRA_SCHEMAS = {
    "ConditioningZeroOut": {
        "input": {"required": {"conditioning": ["CONDITIONING"]}},
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
        "category": "advanced/conditioning",
    },
    "TestStringSource": {
        "input": {"required": {}},
        "output": ["STRING"],
        "output_name": ["STRING"],
        "category": "utils",
    },
    "TestIntSource": {
        "input": {"required": {}},
        "output": ["INT"],
        "output_name": ["INT"],
        "category": "utils",
    },
}


@pytest.fixture(scope="module")
def object_info():
    base = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    return {**base, **_EXTRA_SCHEMAS}


# --- valid workflow uuid (Invalid-uuid-at-id fix) ---


def test_new_workflow_to_ui_has_valid_uuid():
    ui = Workflow.new().to_ui()
    assert _UUID_RE.match(ui["id"]), f"expected a uuid, got {ui['id']!r}"


def test_from_ui_preserves_existing_valid_uuid():
    existing = "12345678-1234-1234-1234-1234567890ab"
    wf = Workflow.from_ui({"id": existing, "nodes": [], "links": []})
    assert wf.to_ui()["id"] == existing


def test_from_ui_replaces_empty_or_invalid_id_with_uuid():
    wf = Workflow.from_ui({"id": "", "nodes": [], "links": []})
    assert _UUID_RE.match(wf.to_ui()["id"])
    wf2 = Workflow.from_ui({"id": "not-a-uuid", "nodes": [], "links": []})
    assert _UUID_RE.match(wf2.to_ui()["id"])


# --- connect into a newly-added node's widget input ---


def test_connect_into_new_node_string_widget(object_info):
    wf = Workflow.new()
    src = wf.add_node("TestStringSource", object_info=object_info)
    enc = wf.add_node("CLIPTextEncode", object_info=object_info)
    # 'text' is a STRING widget with no socket until we convert it
    assert enc.input_by_name("text") is None
    wf.connect(src.id, "STRING", enc.id, "text", object_info)
    slot = enc.input_by_name("text")
    assert slot is not None and slot.link is not None
    assert slot.widget_name == "text"
    api = wf.to_api(object_info)
    assert api[str(enc.id)]["inputs"]["text"] == [str(src.id), 0]


def test_connect_into_new_node_int_widget(object_info):
    wf = Workflow.new()
    src = wf.add_node("TestIntSource", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    assert latent.input_by_name("width") is None
    wf.connect(src.id, "INT", latent.id, "width", object_info)
    assert latent.input_by_name("width").link is not None
    api = wf.to_api(object_info)
    assert api[str(latent.id)]["inputs"]["width"] == [str(src.id), 0]


def test_connect_unknown_widget_still_errors(object_info):
    wf = Workflow.new()
    src = wf.add_node("TestStringSource", object_info=object_info)
    enc = wf.add_node("CLIPTextEncode", object_info=object_info)
    with pytest.raises(ValueError, match="no input 'nonesuch'"):
        wf.connect(src.id, "STRING", enc.id, "nonesuch", object_info)


# --- ConditioningZeroOut-aware prompt titling ---


def _turbo_graph(object_info):
    wf = Workflow.new()
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    zero = wf.add_node("ConditioningZeroOut", object_info=object_info)
    ksampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(pos.id, "CONDITIONING", ksampler.id, "positive")
    wf.connect(pos.id, "CONDITIONING", zero.id, "conditioning")
    wf.connect(zero.id, "CONDITIONING", ksampler.id, "negative")
    return wf, pos, zero


def test_prompt_role_positive_through_zeroout(object_info):
    wf, pos, _zero = _turbo_graph(object_info)
    assert _prompt_role(wf, pos) == "positive"


def test_title_nodes_labels_turbo_pattern(object_info):
    wf, pos, zero = _turbo_graph(object_info)
    _title_nodes(wf, object_info)
    assert pos.title == "✅ Positive Prompt"
    assert zero.title == "🚫 Negative (zeroed)"


def test_prompt_role_direct_negative(object_info):
    wf = Workflow.new()
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    ksampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(neg.id, "CONDITIONING", ksampler.id, "negative")
    assert _prompt_role(wf, neg) == "negative"


# --- krea2 detection wins over FLUX's generic "krea" ---


def test_detect_family_krea2_over_flux(object_info):
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "krea2\\krea2_turbo_mxfp8.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info) == "krea2"


# --- graceful unknown-family port ---


def test_port_unknown_family_returns_error(object_info):
    wf = Workflow.new()
    report = port_workflow(wf, "totally_unknown_family", object_info)
    assert "error" in report
    assert "families" in report
    assert report["changes"] == []


# --- record_learning seeds detection for a new family ---


def test_learned_detect_block_makes_new_family_detectable(object_info, tmp_path):
    knowledge.save_learning(
        tmp_path,
        "zonkmodel",
        {"detect": {"checkpoint_patterns": ["zonk"]}, "loader": "checkpoint"},
        source="unit-test",
    )
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "zonk_v1.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info) is None  # not in floor
    assert knowledge.detect_family(wf, object_info, learned_dir=tmp_path) == "zonkmodel"


# --- run_workflow output collection ---


def test_collect_outputs_maps_history():
    history = {
        "outputs": {
            "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
            "12": {"gifs": [{"filename": "b.webp", "subfolder": "sub", "type": "output"}]},
        }
    }
    out = ComfyClient._collect_outputs(history)
    assert {o["kind"] for o in out} == {"images", "gifs"}
    images = next(o for o in out if o["kind"] == "images")
    assert images["node_id"] == "9" and images["filename"] == "a.png"


def test_collect_outputs_empty_history():
    assert ComfyClient._collect_outputs({}) == []
