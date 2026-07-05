"""Annotator: semantic groups, human titles, knob highlighting, guidance notes."""

import json
from pathlib import Path

import pytest

from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.layout import apply_layout
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture
def txt2img(object_info):
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
    wf.set_widget(ckpt.id, "ckpt_name", "SDXL\\juggernautXL_v9.safetensors", object_info)
    apply_layout(wf, object_info)
    return wf, {"pos": pos.id, "neg": neg.id, "sampler": sampler.id, "ckpt": ckpt.id}


def test_groups_created_per_stage(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)
    titles = [g.title for g in wf.groups]
    assert len(titles) >= 4
    joined = " ".join(titles).lower()
    for word in ("model", "prompt", "sampl", "output"):
        assert word in joined, f"expected a group about '{word}', got {titles}"


def test_groups_have_distinct_colors_and_contain_members(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)
    colors = [g.color for g in wf.groups]
    assert len(set(colors)) == len(colors)
    for group in wf.groups:
        x0, y0, w_, h_ = group.bounding
        members = [
            n
            for n in wf.nodes.values()
            if n.type not in ("Note", "MarkdownNote")
            and x0 <= n.pos[0] and y0 <= n.pos[1]
            and n.pos[0] + n.size[0] <= x0 + w_
            and n.pos[1] + n.size[1] <= y0 + h_
        ]
        assert members, f"group '{group.title}' contains no nodes"


def test_positive_negative_prompt_titles(txt2img, object_info):
    wf, ids = txt2img
    annotate(wf, object_info)
    assert "positive" in wf.nodes[ids["pos"]].title.lower()
    assert "negative" in wf.nodes[ids["neg"]].title.lower()


def test_user_knob_nodes_highlighted_green(txt2img, object_info):
    wf, ids = txt2img
    annotate(wf, object_info)
    assert wf.nodes[ids["pos"]].color == "#232"  # touch-me green
    assert wf.nodes[ids["sampler"]].color != "#232"  # tuned, leave alone


def test_notes_generated_with_model_aware_guidance(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)
    notes = [n for n in wf.nodes.values() if n.type == "MarkdownNote"]
    assert notes, "no guidance notes generated"
    all_text = " ".join(str(n.widgets_values[0]) for n in notes)
    assert "CFG" in all_text  # sampling guidance present
    assert "SDXL" in all_text  # family detected from checkpoint name
    # the two registers
    assert "👇" in all_text or "touch" in all_text.lower()
    assert "⚙" in all_text or "leave" in all_text.lower()


def test_notes_do_not_overlap_nodes(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)

    def box(n):
        return (n.pos[0], n.pos[1], n.pos[0] + n.size[0], n.pos[1] + n.size[1])

    notes = [n for n in wf.nodes.values() if n.type == "MarkdownNote"]
    others = [n for n in wf.nodes.values() if n.type != "MarkdownNote"]
    for note in notes:
        nb = box(note)
        for other in others:
            ob = box(other)
            overlap = nb[0] < ob[2] and ob[0] < nb[2] and nb[1] < ob[3] and ob[1] < nb[3]
            assert not overlap, f"note {note.id} overlaps node {other.id} ({other.type})"


def test_annotate_is_idempotent(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)
    group_count = len(wf.groups)
    note_count = sum(1 for n in wf.nodes.values() if n.type == "MarkdownNote")
    annotate(wf, object_info)
    assert len(wf.groups) == group_count
    assert sum(1 for n in wf.nodes.values() if n.type == "MarkdownNote") == note_count


# --- lint ---


def test_lint_flags_bare_workflow(txt2img, object_info):
    wf, _ = txt2img
    findings = lint(wf, object_info)
    codes = {f["code"] for f in findings}
    assert "no-groups" in codes
    assert "no-notes" in codes
    assert "untitled-prompts" in codes


def test_lint_passes_annotated_workflow(txt2img, object_info):
    wf, _ = txt2img
    annotate(wf, object_info)
    findings = lint(wf, object_info)
    assert findings == [], f"expected clean lint, got {findings}"


def test_lint_flags_dangling_required_input(object_info):
    wf = Workflow.new()
    wf.add_node("KSampler", object_info=object_info)
    findings = lint(wf, object_info)
    assert any(f["code"] == "unconnected-input" for f in findings)


def test_lint_flags_orphan_node(txt2img, object_info):
    wf, _ = txt2img
    wf.add_node("VAELoader", object_info=object_info)
    annotate(wf, object_info)
    findings = lint(wf, object_info)
    assert any(f["code"] == "orphan-node" for f in findings)


def test_models_group_title_no_lora(txt2img, object_info):
    """Workflow with no LoRA loader gets '🧠 Models' (no 'LoRA' in title)."""
    wf, _ = txt2img
    annotate(wf, object_info)
    model_groups = [g for g in wf.groups if "model" in g.title.lower()]
    assert model_groups, "no models group found"
    for g in model_groups:
        assert "lora" not in g.title.lower(), f"expected no LoRA in title, got '{g.title}'"
        assert g.title == "🧠 Models"


def test_models_group_title_with_lora(txt2img, object_info):
    """Workflow with a LoRA loader gets '🧠 Models & LoRAs' title."""
    wf, ids = txt2img
    lora = wf.add_node("LoraLoader", object_info=object_info)
    wf.connect(ids["ckpt"], "MODEL", lora.id, "model")
    wf.connect(ids["ckpt"], "CLIP", lora.id, "clip")
    wf.set_widget(lora.id, "lora_name", "SDXL\\dmd2_sdxl_4step_lora_fp16.safetensors", object_info)
    annotate(wf, object_info)
    model_groups = [g for g in wf.groups if "model" in g.title.lower()]
    assert model_groups, "no models group found"
    assert any("lora" in g.title.lower() for g in model_groups), (
        f"expected LoRA in group title, got {[g.title for g in model_groups]}"
    )


# --- classification of category-less utility nodes ---


def test_classify_string_builders_as_prompts():
    from comfy_draftsman.graph.annotate import classify
    from comfy_draftsman.graph.model import Node

    schema = {
        "category": "custom/text",
        "input": {"required": {"delimiter": ["STRING", {"default": ""}]}},
        "output": ["STRING"],
    }
    node = Node(id=1, type="TextConcat")
    assert classify(node, {"TextConcat": schema}) == "prompts"


def test_classify_image_in_image_out_as_post():
    from comfy_draftsman.graph.annotate import classify
    from comfy_draftsman.graph.model import Node

    schema = {
        "category": "custom/overlay",
        "input": {"required": {"image": ["IMAGE"], "text": ["STRING", {"default": ""}]}},
        "output": ["IMAGE"],
    }
    node = Node(id=1, type="TextOverlay")
    assert classify(node, {"TextOverlay": schema}) == "post"


def test_group_bounds_shrink_to_fit_members(txt2img, object_info):
    """Group boxes must tightly wrap their member nodes (+ note + padding),
    never trapping large empty areas."""
    wf, _ = txt2img
    annotate(wf, object_info)
    notes = {n.id: n for n in wf.nodes.values() if n.type == "MarkdownNote"}
    real = [n for n in wf.nodes.values() if n.type not in ("Note", "MarkdownNote")]
    for group in wf.groups:
        gx, gy, gw, gh = group.bounding
        inside = [
            n
            for n in list(real) + list(notes.values())
            if gx <= n.pos[0]
            and gy <= n.pos[1]
            and n.pos[0] + n.size[0] <= gx + gw
            and n.pos[1] + n.size[1] <= gy + gh
        ]
        assert inside, f"group '{group.title}' is empty"
        extent_w = max(n.pos[0] + n.size[0] for n in inside) - min(n.pos[0] for n in inside)
        extent_h = max(n.pos[1] + n.size[1] for n in inside) - min(n.pos[1] for n in inside)
        assert gw <= extent_w + 2 * 30.0 + 1.0, (
            f"group '{group.title}' is {gw - extent_w}px wider than its contents"
        )
        assert gh <= extent_h + 70.0 + 90.0 + 1.0, (
            f"group '{group.title}' is {gh - extent_h}px taller than its contents"
        )
