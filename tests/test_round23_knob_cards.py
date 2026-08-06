"""Round 23 / Part D2: knob cards - what to change, what it does, what it may
be set to. A widget-effect glossary (graph/knobs.py) plus a technique
tradeoff lookup (knowledge/techniques.yaml), rendered as a compact markdown
table per band in organize_workflow's notes. Range/choices always come
straight from /object_info - a knob wired from upstream is never claimed
editable."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph import knobs
from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


# --- knobs.knob_rows: table content ---


def test_unwired_numeric_knob_gets_range_and_effect(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    rows = knobs.knob_rows([sampler], object_info)
    steps_row = next(r for r in rows if r.startswith("| steps "))
    assert "1-10000" in steps_row
    assert "refinement" in steps_row.lower()


def test_wired_knob_shows_as_wired_not_editable(object_info):
    wf = Workflow.new()
    prim = wf.add_node("PrimitiveNode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(prim.id, 0, sampler.id, "steps", object_info)
    rows = knobs.knob_rows([sampler], object_info)
    steps_row = next(r for r in rows if r.startswith("| steps "))
    assert "wired" in steps_row
    # a wired knob's range/effect must not be asserted - it isn't editable here
    assert "refinement" not in steps_row.lower()


def test_seed_never_shows_its_astronomical_schema_max(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    rows = knobs.knob_rows([sampler], object_info)
    seed_row = next(r for r in rows if r.startswith("| seed "))
    assert "18446744073709551615" not in seed_row


def test_combo_choices_capped_with_more_marker(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    rows = knobs.knob_rows([sampler], object_info)
    sampler_row = next(r for r in rows if r.startswith("| sampler_name "))
    assert "more)" in sampler_row


def test_family_cfg_note_wins_over_generic_glossary(object_info):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=object_info)
    guidance = {"sampling": {"cfg": {"note": "No CFG. H3 is guidance-distilled."}}}
    rows = knobs.knob_rows([sampler], object_info, guidance)
    cfg_row = next(r for r in rows if r.startswith("| cfg "))
    assert "guidance-distilled" in cfg_row
    assert "oversaturate" not in cfg_row  # the generic glossary sentence is NOT used


def test_no_glossary_knobs_gives_empty_table(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    assert knobs.knob_rows([ckpt], object_info) == []


def test_lora_strength_knobs_present(object_info):
    wf = Workflow.new()
    lora = wf.add_node("LoraLoader", object_info=object_info)
    rows = knobs.knob_rows([lora], object_info)
    names = [r.split("|")[1].strip() for r in rows]
    assert "strength_model" in names
    assert "strength_clip" in names


# --- knobs.technique_note: class-name pattern lookup ---


@pytest.mark.parametrize(
    "class_type",
    ["EasyCache", "TeaCache", "SageAttention", "PatchSageAttentionKJ", "TorchCompileModel"],
)
def test_known_technique_classes_have_a_note(class_type):
    assert knobs.technique_note(class_type) is not None


def test_unknown_class_has_no_technique_note():
    assert knobs.technique_note("CheckpointLoaderSimple") is None


def test_technique_note_never_claims_a_specific_value():
    """Every technique note is a tradeoff, never a 'use X' recommendation -
    this file can't verify a specific number, only what moving it costs."""
    for entry in knobs._load_techniques():
        note = entry["note"]
        assert "use " not in note.lower()[:20]  # doesn't open with a command


# --- integration: _note_text renders the table, doesn't invent anything ---


def test_organize_produces_table_for_sampler_band(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    annotate(wf, object_info)
    note_texts = [
        n.widgets_values[0]
        for n in wf.nodes.values()
        if n.type == "MarkdownNote" and n.widgets_values
    ]
    joined = "\n".join(note_texts)
    assert "| knob | now | range / choices | effect |" in joined
    assert "| steps |" in joined


def test_organize_never_invents_a_range_absent_from_schema(object_info):
    """A widget the glossary knows about but whose schema declares no
    min/max (free-form) must render an EMPTY range cell, never a guess."""
    info = dict(object_info)
    info["FreeformDenoise"] = {
        "category": "custom",
        "input": {"required": {"denoise": ["FLOAT", {"default": 1.0}]}},  # no min/max
        "output": ["LATENT"],
    }
    wf = Workflow.new()
    node = wf.add_node("FreeformDenoise", object_info=info)
    rows = knobs.knob_rows([node], info)
    denoise_row = next(r for r in rows if r.startswith("| denoise "))
    cells = [c.strip() for c in denoise_row.split("|")]
    # cells: ['', 'denoise', '1.0', '', 'effect text', '']
    assert cells[3] == ""
