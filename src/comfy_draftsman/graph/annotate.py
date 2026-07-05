"""Turn a working graph into a workflow a regular person can read.

- classifies nodes into pipeline stages and lays them out in stage bands
- wraps each stage in a titled, colored group
- titles semantically important nodes (positive/negative prompts, loaders)
- paints "knobs you're meant to touch" green
- writes one MarkdownNote per stage in two registers: what to touch, and
  which tuned settings to leave alone - sourced from the knowledge floor
  (+ learned overlay) for the detected model family
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from .. import knowledge
from .layout import Y_GAP, apply_staged_layout
from .model import Group, Node, Workflow

NOTE_MARKER = "comfy-draftsman"

# node color swatches from the ComfyUI frontend palette
GREEN = ("#232", "#353")  # touch-me
NOTE_COLOR = ("#432", "#653")

STAGES: list[tuple[str, str, str]] = [
    # (key, group title, group color)
    ("inputs", "📥 Inputs", "#88553d"),
    ("models", "🧠 Models & LoRAs", "#3f5159"),
    ("prompts", "✍️ Prompts", "#335c33"),
    ("sampling", "🎛️ Sampling", "#42425c"),
    ("post", "✨ Post-Processing", "#5c5029"),
    ("output", "💾 Output", "#653d3d"),
]
_STAGE_INDEX = {key: i for i, (key, _, _) in enumerate(STAGES)}

_INPUT_CLASSES = {"LoadImage", "LoadImageMask", "LoadAudio", "LoadVideo", "VHS_LoadVideo"}
_POST_HINTS = ("detailer", "upscale", "facerestore", "interpolat", "rife", "segs", "postprocess")
_KNOB_WIDGETS = {"text", "prompt", "wildcard_text", "width", "height", "image"}


def classify(node: Node, object_info: dict[str, Any]) -> str:
    if node.type in _INPUT_CLASSES:
        return "inputs"
    schema = object_info.get(node.type)
    name = node.type.lower()
    if schema is not None:
        category = (schema.get("category") or "").lower()
        if schema.get("output_node"):
            return "output"
        if "loaders" in category:
            return "models"
        if "conditioning" in category:
            return "prompts"
        if "sampling" in category or "latent" in category:
            return "sampling"
        if category.startswith("image") or category.startswith("mask"):
            return "post"
    if any(hint in name for hint in _POST_HINTS):
        return "post"
    return "sampling"


ZEROOUT_TYPE = "ConditioningZeroOut"


def _reached_roles(
    wf: Workflow, node: Node, depth: int = 0, via_zeroout: bool = False
) -> set[tuple[str, bool]]:
    """Sampler roles ('positive'/'negative') this node's conditioning reaches,
    each tagged with whether the path passed through a ConditioningZeroOut."""
    results: set[tuple[str, bool]] = set()
    if depth > 5:
        return results
    for out in node.outputs:
        for lid in out.links:
            link = wf.links.get(lid)
            if link is None:
                continue
            target = wf.nodes.get(link.target_id)
            if target is None or link.target_slot >= len(target.inputs):
                continue
            input_name = target.inputs[link.target_slot].name.lower()
            if input_name in ("positive", "negative"):
                results.add((input_name, via_zeroout))
            downstream_zeroed = via_zeroout or target.type == ZEROOUT_TYPE
            results |= _reached_roles(wf, target, depth + 1, downstream_zeroed)
    return results


def _prompt_role(wf: Workflow, node: Node) -> str | None:
    """Positive/negative for a text-encode node. A ConditioningZeroOut in the
    path is the negative branch, so the text feeding it is the *positive* source
    (this is the turbo/distilled pattern: positive prompt -> ZeroOut -> negative)."""
    roles = _reached_roles(wf, node)
    if not roles:
        return None
    direct = {role for role, zeroed in roles if not zeroed}
    if "positive" in direct:  # prefer a real positive when a node feeds both
        return "positive"
    if "negative" in direct:
        return "negative"
    # only reaches a sampler through a ZeroOut -> it's the positive source
    if any(role == "negative" and zeroed for role, zeroed in roles):
        return "positive"
    return "positive" if any(role == "positive" for role, _ in roles) else None


def _title_nodes(wf: Workflow, object_info: dict[str, Any]) -> None:
    for node in wf.nodes.values():
        schema = object_info.get(node.type)
        if schema is None:
            continue
        if node.type == ZEROOUT_TYPE and node.title is None:
            node.title = "🚫 Negative (zeroed)"
            continue
        has_text_widget = any(s == "text" for s in _safe_slots(node, object_info))
        if node.type == "CLIPTextEncode" or has_text_widget:
            role = _prompt_role(wf, node)
            if role == "positive":
                node.title = "✅ Positive Prompt"
            elif role == "negative":
                node.title = "🚫 Negative Prompt"
        if "loaders" in (schema.get("category") or "") and node.title is None:
            filenames = [
                v
                for v in _named_widgets(node, object_info).values()
                if isinstance(v, str) and "." in v
            ]
            if filenames:
                stem = Path(filenames[0].replace("\\", "/")).stem
                display = schema.get("display_name") or node.type
                node.title = f"{display}: {stem[:32]}"


def _safe_slots(node: Node, object_info: dict[str, Any]) -> list[str]:
    from . import widgets as w

    try:
        return w.widget_slot_names(node.type, object_info)
    except (ValueError, KeyError):
        return []


def _named_widgets(node: Node, object_info: dict[str, Any]) -> dict[str, Any]:
    from . import widgets as w

    try:
        return w.widgets_to_named(node.type, node.widgets_values, object_info)
    except (ValueError, KeyError):
        return {}


def _wired_input(node: Node, name: str) -> bool:
    """True if this widget has been converted to an input and has a link feeding
    it - i.e. the value comes from upstream and is NOT hand-editable."""
    slot = node.input_by_name(name)
    return slot is not None and slot.link is not None


def _paint_knobs(wf: Workflow, object_info: dict[str, Any], stage_of_key: dict[int, str]) -> None:
    for node in wf.nodes.values():
        stage = stage_of_key.get(node.id)
        slots = set(_safe_slots(node, object_info))
        prompt_knobs = slots & {"text", "prompt", "wildcard_text"}
        # a text/prompt knob that is wired from upstream isn't editable - don't
        # paint it "touch me" green (that combination misleads a human reader)
        editable_prompt_knob = stage == "prompts" and any(
            not _wired_input(node, name) for name in prompt_knobs
        )
        is_knob = (
            node.type in _INPUT_CLASSES
            or editable_prompt_knob
            or node.type in ("EmptyLatentImage", "EmptySD3LatentImage")
        )
        if is_knob:
            node.color, node.bgcolor = GREEN


def _wrap(text: str, width: int = 58) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


def _note_text(
    stage: str,
    wf: Workflow,
    object_info: dict[str, Any],
    guidance: dict[str, Any] | None,
    members: list[Node],
) -> str | None:
    g = guidance or {}
    family = g.get("display_name", "this model")
    notes = g.get("notes", {})
    lines: list[str] = []
    if stage == "models":
        lines.append("👇 Swap models here to change the whole look.")
        if notes.get("loaders"):
            lines.append(notes["loaders"])
    elif stage == "prompts":
        text_nodes = [
            n for n in members if "text" in _safe_slots(n, object_info) or n.type == "CLIPTextEncode"
        ]
        any_editable = any(
            not _wired_input(n, name)
            for n in text_nodes
            for name in ("text", "prompt", "wildcard_text")
            if name in _safe_slots(n, object_info)
        )
        if any_editable:
            lines.append("👇 Type what you want in the green Positive Prompt node.")
        else:
            lines.append(
                "✍️ The prompt text here is built automatically from the upstream "
                "green string nodes — edit those (word banks / inputs) to change the "
                "result, not the prompt box (it's wired, so it can't be typed into)."
            )
        if notes.get("conditioning"):
            lines.append(notes["conditioning"])
    elif stage == "sampling":
        sampler = next(
            (n for n in members if "sampling" in (object_info.get(n.type, {}).get("category") or "")),
            None,
        )
        if sampler is not None:
            named = _named_widgets(sampler, object_info)
            current = ", ".join(
                f"{k}={named[k]}"
                for k in ("steps", "cfg", "sampler_name", "scheduler")
                if k in named
            )
            if current:
                lines.append(f"⚙️ Tuned for {family}: {current} — leave these alone.")
        if notes.get("sampling"):
            lines.append(notes["sampling"])
        if notes.get("latent"):
            lines.append("👇 " + notes["latent"])
        sampling = g.get("sampling", {})
        if sampling and "cfg" in sampling:
            cfg = sampling["cfg"]
            lines.append(
                f"Safe ranges: CFG {cfg.get('min')}-{cfg.get('max')}, "
                f"steps {sampling.get('steps', {}).get('min')}-{sampling.get('steps', {}).get('max')}."
            )
    elif stage == "post":
        for technique, settings in (g.get("techniques") or {}).items():
            hint = technique.replace("_", " ")
            if any(hint.split()[0] in n.type.lower() for n in members) and settings.get("note"):
                lines.append("⚙️ " + settings["note"])
        if not lines:
            lines.append("⚙️ Post-processing chain — tuned to match the model above.")
    elif stage == "output":
        lines.append("💾 Finished images land here (check the filename prefix).")
    elif stage == "inputs":
        lines.append("👇 Load your source image/media here.")
    if not lines:
        return None
    title = dict((k, t) for k, t, _ in STAGES)[stage]
    return f"### {title}\n\n" + "\n\n".join(_wrap(line) for line in lines)


def annotate(
    wf: Workflow,
    object_info: dict[str, Any],
    learned_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Organize, group, title, highlight, and annotate the workflow in place."""
    # drop notes we generated on a previous run (idempotency); keep human notes
    for nid in [
        n.id for n in wf.nodes.values() if n.properties.get("draftsman") == NOTE_MARKER
    ]:
        wf.remove_node(nid)
    wf.groups = []

    family = knowledge.detect_family(wf, object_info, learned_dir=learned_dir)
    guidance = None
    if family:
        filenames = knowledge.model_filenames(wf, object_info)
        guidance = knowledge.get_guidance(
            family, model_filename=filenames[0] if filenames else None, learned_dir=learned_dir
        )

    stage_of_key = {
        node.id: classify(node, object_info)
        for node in wf.nodes.values()
        if node.type not in ("Note", "MarkdownNote")
    }
    stage_of = {nid: _STAGE_INDEX[key] for nid, key in stage_of_key.items()}
    band_boxes = apply_staged_layout(wf, object_info, stage_of)

    _title_nodes(wf, object_info)
    _paint_knobs(wf, object_info, stage_of_key)

    members_by_stage: dict[int, list[Node]] = {}
    for nid, stage in stage_of.items():
        members_by_stage.setdefault(stage, []).append(wf.nodes[nid])

    for stage_index, (x, y, w_, h_) in sorted(band_boxes.items()):
        key, title, color = STAGES[stage_index]
        members = members_by_stage.get(stage_index, [])
        top = y
        text = _note_text(key, wf, object_info, guidance, members)
        if text:
            note_w = max(min(w_, 380.0), 300.0)
            # frontend renders markdown at ~17px/line; blank separator lines
            # collapse, headings add a little
            rendered_lines = sum(1 for line in text.splitlines() if line.strip())
            note_h = 17.0 * rendered_lines + 70.0
            note = wf.add_node("MarkdownNote", title=title)
            note.widgets_values = [text]
            note.size = [note_w, note_h]
            note.pos = [x, y - note_h - Y_GAP]
            note.color, note.bgcolor = NOTE_COLOR
            note.properties["draftsman"] = NOTE_MARKER
            top = y - note_h - Y_GAP
        pad = 30.0
        wf.groups.append(
            Group(
                id=len(wf.groups) + 1,
                title=title,
                bounding=[x - pad, top - 70.0, max(w_, 320.0) + 2 * pad, h_ + (y - top) + 90.0],
                color=color,
            )
        )
    return {
        "family": family,
        "variant": (guidance or {}).get("variant"),
        "stages": {STAGES[i][0]: len(m) for i, m in sorted(members_by_stage.items())},
    }
