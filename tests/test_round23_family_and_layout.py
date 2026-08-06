"""Round 23: family detection anchored to the diffusion model, organize_workflow
overlap fix, honest notes, layout/group edit ops, force-socket creation, and the
reader-first default organization. From a live bug report building a MiniMax H3
workflow: a placeholder LTX LoRA made organize_workflow claim the graph was LTX
Video and inject CFG guidance into a graph with no CFG node."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import knowledge
from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.layout import apply_staged_layout, resolve_overlaps
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


# --- A1: family detection anchored to the diffusion model, never a LoRA ---


def test_lora_filename_does_not_hijack_family_detection(object_info, tmp_path):
    """The exact reported incident: a placeholder LTX LoRA on an H3-named UNet
    must not detect as 'ltx'. Without the H3 learned overlay, no family in the
    floor claims this UNet name, so detection must return None - not 'ltx'."""
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "minimax_h3_ref2va_pruned_int8.safetensors", object_info)
    lora = wf.add_node("LoraLoader", object_info=object_info)
    wf.set_widget(lora.id, "lora_name", "LTX23_The_Cook_Watermark_Remover_v1.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info) != "ltx"
    assert knowledge.detect_family(wf, object_info) is None


def test_learned_overlay_detects_family_from_primary_ref_only(object_info, tmp_path):
    knowledge.save_learning(
        tmp_path,
        "minimax_h3",
        {
            "detect": {"checkpoint_patterns": ["minimax_h3", "minimax-h3", "minimaxh3"]},
            "loader": "unet_clip_vae",
        },
        source="unit-test",
    )
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "minimax_h3_ref2va_pruned_int8.safetensors", object_info)
    lora = wf.add_node("LoraLoader", object_info=object_info)
    wf.set_widget(lora.id, "lora_name", "LTX23_The_Cook_Watermark_Remover_v1.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info, learned_dir=tmp_path) == "minimax_h3"


def test_lora_only_graph_detects_nothing(object_info):
    """A graph whose only model-shaped reference is a LoRA must not detect any
    family - there is no diffusion model reference to anchor on."""
    wf = Workflow.new()
    lora = wf.add_node("LoraLoader", object_info=object_info)
    wf.set_widget(lora.id, "lora_name", "SDXL\\real-humans-PublicPrompts.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info) is None


def test_detect_family_survives_lying_merge_names_still_passes(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(
        ckpt.id, "ckpt_name", "SDXL\\gonzalomoXLFluxPony_v60PhotoXLDMD.safetensors", object_info
    )
    assert knowledge.detect_family(wf, object_info) == "sdxl"


def test_detect_family_krea2_over_flux_still_passes(object_info):
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=object_info)
    wf.set_widget(unet.id, "unet_name", "krea2\\krea2_turbo_mxfp8.safetensors", object_info)
    assert knowledge.detect_family(wf, object_info) == "krea2"


def test_detect_family_detail_reports_matched_pattern(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    detail = knowledge.detect_family_detail(wf, object_info)
    assert detail["family"] == "sdxl"
    assert detail["matched_on"]
    assert detail["widget"] == "ckpt_name"


def test_lora_variant_name_does_not_select_variant(object_info):
    """A LoRA named '...turbo...' must not select the sdxl 'turbo' variant of
    an unrelated base checkpoint."""
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    lora = wf.add_node("LoraLoader", object_info=object_info)
    wf.set_widget(lora.id, "lora_name", "SDXL\\turbo_style_lora.safetensors", object_info)
    annotate(wf, object_info)
    assert knowledge.detect_family(wf, object_info) == "sdxl"
    filenames = knowledge.primary_model_filenames(wf, object_info)
    assert all("turbo" not in f.lower() for f in filenames)


# --- A2: organize_workflow must never ship a layout it has diagnosed as broken ---


def _minimal_wf(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    decode = wf.add_node("VAEDecode", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model")
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
    wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
    wf.connect(sampler.id, "LATENT", decode.id, "samples")
    wf.connect(ckpt.id, "VAE", decode.id, "vae")
    wf.connect(decode.id, "IMAGE", save.id, "images")
    return wf


def test_foreign_notes_do_not_survive_as_overlaps(object_info):
    """The exact reported shape: several oversized human-authored MarkdownNote
    nodes, all left at the default position, must not produce overlap findings
    after organize_workflow."""
    wf = _minimal_wf(object_info)
    for i in range(7):
        note = wf.add_node("MarkdownNote", title=f"Human note {i}")
        note.widgets_values = [f"Some hand-written note #{i} with real content."]
        note.size = [420.0, 260.0]
        # left at [0, 0] (default position) - exactly like every node added
        # through edit_workflow before organize_workflow runs
    report = annotate(wf, object_info)
    findings = lint(wf, object_info)
    overlaps = [f for f in findings if f["code"] == "overlap"]
    assert overlaps == [], f"organize_workflow shipped a broken layout: {overlaps}"
    assert report["applied"].get("foreign_notes_parked", 0) >= 1


def test_resolve_overlaps_is_noop_on_clean_layout(object_info):
    wf = _minimal_wf(object_info)
    apply_staged_layout(
        wf, object_info, {n.id: 0 for n in wf.nodes.values() if n.type != "MarkdownNote"}
    )
    before = {nid: tuple(n.pos) for nid, n in wf.nodes.items()}
    moved = resolve_overlaps(wf)
    after = {nid: tuple(n.pos) for nid, n in wf.nodes.items()}
    assert moved == 0
    assert before == after


def test_annotate_idempotent_after_overlap_resolution(object_info):
    """Draftsman-generated notes get fresh node ids each run (removed and
    re-added for idempotent CONTENT), so idempotency is about the STABLE
    (non-note) nodes' positions staying put, not raw node-id equality."""
    wf = _minimal_wf(object_info)
    for i in range(3):
        note = wf.add_node("MarkdownNote", title=f"Note {i}")
        note.widgets_values = [f"content {i}"]
        note.size = [400.0, 240.0]
    # ids present BEFORE the first organize pass are stable across passes
    # (pipeline nodes + the foreign notes); draftsman-generated notes are
    # removed and re-added each run and get fresh ids, deliberately excluded
    real_ids = set(wf.nodes)
    annotate(wf, object_info)
    first_positions = {nid: tuple(wf.nodes[nid].pos) for nid in real_ids}
    annotate(wf, object_info)
    second_positions = {nid: tuple(wf.nodes[nid].pos) for nid in real_ids}
    assert first_positions == second_positions
    assert all(f["code"] != "overlap" for f in lint(wf, object_info))


