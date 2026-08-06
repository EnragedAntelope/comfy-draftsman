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
from .layout import X_GUTTER, Y_GAP, apply_staged_layout, is_text_display, resolve_overlaps
from .model import PRIMITIVE_TYPE, REROUTE_TYPE, Node, Workflow

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


def _is_canvas_node(class_type: str) -> bool:
    """Empty-latent canvas nodes (EmptyLatentImage & family) - they ARE the
    resolution knob, so they belong with the user-facing inputs on the left."""
    name = class_type.lower()
    return "empty" in name and "latent" in name


def classify(node: Node, object_info: dict[str, Any]) -> str:
    if node.type in _INPUT_CLASSES:
        return "inputs"
    if _is_canvas_node(node.type):
        return "inputs"
    if node.type == PRIMITIVE_TYPE:
        # a primitive exists precisely to be adjusted by hand (that is why the
        # frontend gives it control_after_generate), so it belongs on the left
        # edge with the other tweakables - not in the middle of the sampler band
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
        # category didn't decide - infer from the data types flowing through
        out_types = {str(t).upper() for t in (schema.get("output") or [])}
        if out_types and out_types <= {"STRING"}:
            # pure text machinery (wildcards, concatenators, templates)
            # belongs with the prompts, not dumped into sampling
            return "prompts"
        in_types = {
            str(spec[0]).upper()
            for section in ("required", "optional")
            for spec in (schema.get("input", {}).get(section, {}) or {}).values()
            if isinstance(spec, list | tuple) and spec and isinstance(spec[0], str)
        }
        if "IMAGE" in in_types and "IMAGE" in out_types:
            return "post"  # image-in/image-out = post-processing (overlays, filters)
    if any(hint in name for hint in _POST_HINTS):
        return "post"
    return "sampling"


def _is_display_companion(node: Node, object_info: dict[str, Any]) -> bool:
    """Nodes that exist only in service of another node and must sit NEXT TO it,
    not be swept into a far-away group - a reader pairing six previews with six
    samplers by following wires across the canvas is exactly the layout failure
    this fixes.

    Display-only nodes (Show Text, PreviewImage...) and Reroutes qualify: a
    Reroute is pure wire-tidying, so stranding it in whatever band it happened to
    classify into lengthens the very wire it exists to shorten. SaveImage-style
    disk writers are NOT companions; they are real outputs."""
    if node.type == REROUTE_TYPE or is_text_display(node.type):
        return True
    schema = object_info.get(node.type) or {}
    return bool(schema.get("output_node")) and "preview" in node.type.lower()


def _companion_sources(
    wf: Workflow, object_info: dict[str, Any], stage_of_key: dict[int, str]
) -> dict[int, int]:
    """companion node id -> the (non-companion) node it displays, resolved
    through chains of display nodes; companions inherit their source's stage."""
    direct: dict[int, int] = {}
    for node in wf.nodes.values():
        if node.id not in stage_of_key or not _is_display_companion(node, object_info):
            continue
        src = next(
            (
                wf.links[s.link].origin_id
                for s in node.inputs
                if s.link is not None and s.link in wf.links
                and wf.links[s.link].origin_id in stage_of_key
            ),
            None,
        )
        if src is not None:
            direct[node.id] = src
    resolved: dict[int, int] = {}
    for nid, src in direct.items():
        seen = {nid}
        while src in direct and src not in seen:  # ShowText fed by ShowText
            seen.add(src)
            src = direct[src]
        if src not in seen:
            resolved[nid] = src
            stage_of_key[nid] = stage_of_key[src]
    return resolved


ZEROOUT_TYPE = "ConditioningZeroOut"

# titles we generate ourselves - safe to rewrite on a later organize pass;
# anything else is human-authored and must never be clobbered
ROLE_TITLES = {"✅ Positive Prompt", "🚫 Negative Prompt"}


def _outputs_conditioning(node: Node) -> bool:
    return any(o.type == "CONDITIONING" for o in node.outputs)


def _feeds_encoder_text(wf: Workflow, node: Node) -> bool:
    """True if this node's output is wired directly into a conditioning
    encoder's text/prompt input - i.e. it IS the prompt source, not a distant
    upstream fragment (wildcard bank, concatenator input, ...)."""
    for out in node.outputs:
        for lid in out.links:
            link = wf.links.get(lid)
            if link is None:
                continue
            target = wf.nodes.get(link.target_id)
            if target is None or link.target_slot >= len(target.inputs):
                continue
            slot_name = target.inputs[link.target_slot].name.lower()
            if slot_name in ("text", "prompt") and _outputs_conditioning(target):
                return True
    return False


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


def _title_nodes(wf: Workflow, object_info: dict[str, Any]) -> int:
    """Role-title prompt/loader nodes. Returns how many titles were written."""
    titled = 0
    for node in wf.nodes.values():
        schema = object_info.get(node.type)
        if schema is None:
            continue
        if node.type == ZEROOUT_TYPE and node.title is None:
            node.title = "🚫 Negative (zeroed)"
            titled += 1
            continue
        has_text_widget = any(s == "text" for s in _safe_slots(node, object_info))
        retitlable = node.title is None or node.title in ROLE_TITLES
        is_prompt_source = _outputs_conditioning(node) or _feeds_encoder_text(wf, node)
        if (node.type == "CLIPTextEncode" or has_text_widget) and retitlable and is_prompt_source:
            role = _prompt_role(wf, node)
            if role == "positive":
                node.title = "✅ Positive Prompt"
                titled += 1
            elif role == "negative":
                node.title = "🚫 Negative Prompt"
                titled += 1
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
                titled += 1
    return titled


def _safe_slots(node: Node, object_info: dict[str, Any]) -> list[str]:
    from . import widgets as w

    try:
        return w.widget_slot_names(node.type, object_info, node.widgets_values)
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


def _paint_knobs(wf: Workflow, object_info: dict[str, Any], stage_of_key: dict[int, str]) -> int:
    """Highlight user-editable knobs green. Returns how many nodes were painted."""
    painted = 0
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
            or _is_canvas_node(node.type)
            # a primitive IS a hand-set value, whatever socket it mirrors
            or node.type == PRIMITIVE_TYPE
        )
        if is_knob:
            node.color, node.bgcolor = GREEN
            painted += 1
        elif (node.color, node.bgcolor) == GREEN:
            # organize is idempotent, so a node that STOPPED being a knob (its
            # prompt got wired from upstream, say) must lose the "touch me"
            # green too - a stale highlight tells the reader to edit a box they
            # can't type into. Only our own swatch is cleared; a colour a human
            # picked is left alone.
            node.color, node.bgcolor = None, None
    return painted


def _wrap(text: str, width: int = 58) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


def _graph_knobs(members: list[Node], object_info: dict[str, Any]) -> set[str]:
    """Widget + connected-input names actually present on this stage's nodes -
    the ground truth for what a 'safe ranges' note is allowed to mention. A
    node without a 'cfg' widget in its schema (BasicGuider/SamplerCustomAdvanced
    - H3's guidance-distilled chain) must never have CFG guidance asserted
    about it, however confident the family guidance text is."""
    knobs: set[str] = set()
    for n in members:
        knobs |= set(_safe_slots(n, object_info))
        knobs |= {slot.name for slot in n.inputs}
    return knobs


_MEDIA_INPUT_TYPES = {"IMAGE": "images", "AUDIO": "audio", "VIDEO": "video"}


def _output_medium(members: list[Node], object_info: dict[str, Any]) -> str:
    """What an output-stage node actually writes, read from its OWN declared
    input types - never hardcoded. Mixed or undetermined falls back to the
    honest generic 'files' rather than guessing."""
    media: set[str] = set()
    for n in members:
        schema = object_info.get(n.type) or {}
        input_types = {
            str(spec[0]).upper()
            for section in ("required", "optional")
            for spec in (schema.get("input", {}).get(section, {}) or {}).values()
            if isinstance(spec, list | tuple) and spec and isinstance(spec[0], str)
        }
        for type_name, label in _MEDIA_INPUT_TYPES.items():
            if type_name in input_types:
                media.add(label)
    if len(media) == 1:
        return next(iter(media))
    return "files"


def _note_text(
    stage: str,
    wf: Workflow,
    object_info: dict[str, Any],
    guidance: dict[str, Any] | None,
    members: list[Node],
    title: str | None = None,
    wrap_width: int = 58,
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
        graph_knobs = _graph_knobs(members, object_info)
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
        cfg_block = sampling.get("cfg") or {}
        # a prose statement (H3: "No CFG - guidance-distilled") beats a numeric
        # range when there isn't one - but only when the graph actually has a
        # cfg knob to talk about (BasicGuider/SamplerCustomAdvanced don't)
        if isinstance(cfg_block.get("note"), str) and "cfg" in graph_knobs:
            lines.append(cfg_block["note"])
        # each clause requires BOTH a real numeric min/max AND the knob being
        # present on this graph's actual sampling nodes - the bug this fixes
        # rendered "Safe ranges: CFG None-None, steps None-None." from a
        # learned overlay whose cfg block was prose-only, with no numbers
        range_clauses = [
            f"{label} {block['min']}-{block['max']}"
            for knob_key, label in (("cfg", "CFG"), ("steps", "steps"))
            if knob_key in graph_knobs
            and isinstance((block := sampling.get(knob_key) or {}).get("min"), int | float)
            and isinstance(block.get("max"), int | float)
        ]
        if range_clauses:
            lines.append("Safe ranges: " + ", ".join(range_clauses) + ".")
    elif stage == "post":
        for technique, settings in (g.get("techniques") or {}).items():
            hint = technique.replace("_", " ")
            if any(hint.split()[0] in n.type.lower() for n in members) and settings.get("note"):
                lines.append("⚙️ " + settings["note"])
        if not lines:
            # no technique guidance matched: describe what's actually here
            # (never claim tuning or refer to spatial position - layouts vary)
            steps = list(
                dict.fromkeys(
                    n.title or (object_info.get(n.type) or {}).get("display_name") or n.type
                    for n in members
                )
            )
            listed = ", ".join(steps[:4]) + (", …" if len(steps) > 4 else "")
            lines.append(f"⚙️ Extra image steps applied after generation: {listed}.")
    elif stage == "output":
        medium = _output_medium(members, object_info)
        lines.append(f"💾 Finished {medium} land here (check the filename prefix).")
    elif stage == "inputs":
        if any(n.type in _INPUT_CLASSES for n in members):
            lines.append("👇 Load your source image/media here.")
        if any(_is_canvas_node(n.type) for n in members):
            lines.append("👇 Set the image size (width / height / batch) here.")
        if any(n.type == PRIMITIVE_TYPE for n in members):
            lines.append(
                "👇 The green value boxes here feed settings further along the "
                "graph — change them here, not at the node they connect to."
            )
    if not lines:
        return None
    note_title = title or dict((k, t) for k, t, _ in STAGES)[stage]
    return f"### {note_title}\n\n" + "\n\n".join(_wrap(line, wrap_width) for line in lines)


def _park_foreign_notes(wf: Workflow, stage_of: dict[int, int]) -> int:
    """Human-authored Note/MarkdownNote nodes are excluded from the staged
    layout (they aren't classified/positioned), so when the layout moves
    everything around them, they stay put and land on top of it - the 49
    overlapping pairs a real session hit with 7 hand-written MarkdownNotes.

    Park only the ones that actually collide with something. A foreign note
    sitting in clear space was placed there on purpose and must not move -
    only a note in the way gets relocated, into a column left of the graph."""
    foreign = [
        n
        for n in wf.nodes.values()
        if n.type in ("Note", "MarkdownNote") and n.properties.get("draftsman") != NOTE_MARKER
    ]
    if not foreign:
        return 0
    others = [wf.nodes[nid] for nid in stage_of]

    def collides(note: Node) -> bool:
        x0, y0 = note.pos[0], note.pos[1]
        x1, y1 = x0 + note.size[0], y0 + note.size[1]
        return any(
            x0 < o.pos[0] + o.size[0]
            and o.pos[0] < x1
            and y0 < o.pos[1] + o.size[1]
            and o.pos[1] < y1
            for o in others
        )

    to_park = [n for n in foreign if collides(n)]
    if not to_park:
        return 0
    park_x = -max(n.size[0] for n in to_park) - X_GUTTER
    y_cursor = 0.0
    for note in to_park:
        note.pos = [park_x, y_cursor]
        y_cursor += note.size[1] + Y_GAP
    return len(to_park)


def annotate(
    wf: Workflow,
    object_info: dict[str, Any],
    learned_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Organize, group, title, highlight, and annotate the workflow in place.

    Four ordered phases so a later one can always see (and fix) what an
    earlier one produced: (1) lay out nodes into stage bands, (2) place
    guidance notes above each band, (3) resolve any overlaps that survived -
    including human-authored notes the layout never touches, (4) compute group
    bounds from the FINAL positions, so a moved note's group never traps the
    empty space it left behind."""
    # drop notes we generated on a previous run (idempotency); keep human notes
    for nid in [
        n.id for n in wf.nodes.values() if n.properties.get("draftsman") == NOTE_MARKER
    ]:
        wf.remove_node(nid)
    wf.groups = []

    detail = knowledge.detect_family_detail(wf, object_info, learned_dir=learned_dir)
    family = detail["family"]
    guidance = None
    if family:
        # variant matching (turbo/lightning/...) must look at the DIFFUSION
        # MODEL's own filename, never a LoRA's - a LoRA named "...turbo..."
        # must not select the turbo variant of an unrelated base model
        filenames = knowledge.primary_model_filenames(wf, object_info)
        guidance = knowledge.get_guidance(
            family, model_filename=filenames[0] if filenames else None, learned_dir=learned_dir
        )

    stage_of_key = {
        node.id: classify(node, object_info)
        for node in wf.nodes.values()
        if node.type not in ("Note", "MarkdownNote")
    }
    # display nodes follow whatever they display (stage + position)
    companion_of = _companion_sources(wf, object_info, stage_of_key)
    stage_of = {nid: _STAGE_INDEX[key] for nid, key in stage_of_key.items()}
    # phase 1: lay out
    band_boxes = apply_staged_layout(wf, object_info, stage_of, companion_of=companion_of)
    foreign_parked = _park_foreign_notes(wf, stage_of)

    titled = _title_nodes(wf, object_info)
    painted = _paint_knobs(wf, object_info, stage_of_key)
    notes_added = 0

    members_by_stage: dict[int, list[Node]] = {}
    for nid, stage in stage_of.items():
        members_by_stage.setdefault(stage, []).append(wf.nodes[nid])

    # phase 2: place notes (group bounds are NOT computed yet - they depend on
    # final, post-resolve positions, which don't exist until phase 3 runs)
    stage_meta: list[tuple[int, str, str, int | None]] = []
    for stage_index in sorted(band_boxes):
        key, default_title, color = STAGES[stage_index]
        members = members_by_stage.get(stage_index, [])
        if not members:
            continue
        min_x = min(n.pos[0] for n in members)
        min_y = min(n.pos[1] for n in members)
        max_x = max(n.pos[0] + n.size[0] for n in members)
        # Dynamic title for models stage: only mention LoRAs if a LoRA loader is present
        title = default_title
        if key == "models":
            has_lora = any(
                "lora" in n.type.lower()
                and "loaders" in (object_info.get(n.type, {}).get("category") or "").lower()
                for n in members
            )
            if not has_lora:
                title = "\U0001f9e0 Models"
        elif key == "inputs" and all(_is_canvas_node(n.type) for n in members):
            title = "📐 Image Size"
        note_w = max(min(max_x - min_x, 380.0), 300.0)
        # wrap to the note's REAL width, not a fixed column count, so the
        # height estimate below matches what actually renders
        wrap_width = max(20, int(note_w // 7))
        text = _note_text(key, wf, object_info, guidance, members, title=title, wrap_width=wrap_width)
        note_id: int | None = None
        if text:
            # frontend renders markdown at ~17px/line; blank separator lines
            # collapse, headings add a little
            rendered_lines = sum(1 for line in text.splitlines() if line.strip())
            note_h = 17.0 * rendered_lines + 70.0
            note = wf.add_node("MarkdownNote", title=title)
            note.widgets_values = [text]
            note.size = [note_w, note_h]
            note.pos = [min_x, min_y - note_h - Y_GAP]
            note.color, note.bgcolor = NOTE_COLOR
            note.properties["draftsman"] = NOTE_MARKER
            notes_added += 1
            note_id = note.id
        stage_meta.append((stage_index, title, color, note_id))

    # phase 3: resolve overlaps - a too-wide note poking into a neighboring
    # band, a parked foreign note whose column undershot, or anything else
    # phases 1-2 couldn't rule out. Nodes only ever move down, never sideways.
    overlap_warning = None
    resolve_overlaps(wf)
    from .lint import lint  # local import: lint.py imports from this module

    surviving = [f for f in lint(wf, object_info) if f["code"] == "overlap"]
    if surviving:
        overlap_warning = surviving[0]["message"]

    # phase 4: group bounds from FINAL positions - computed last so a note (or
    # anything else) that phase 3 moved is never left outside its own group.
    # Same helper edit_workflow's add_group/set_group ops use, so a hand-made
    # group and a generated one are geometrically identical.
    for stage_index, title, color, note_id in stage_meta:
        members = members_by_stage[stage_index]
        boxed = list(members)
        if note_id is not None and note_id in wf.nodes:
            boxed.append(wf.nodes[note_id])
        wf.group_from_nodes(title, [n.id for n in boxed], color=color)
    report: dict[str, Any] = {
        "family": family,
        "variant": (guidance or {}).get("variant"),
        "stages": {STAGES[i][0]: len(m) for i, m in sorted(members_by_stage.items())},
        "applied": {
            "layout": "staged pipeline bands (nodes repositioned; preview/Show Text "
            "nodes sit beside their source)",
            "groups": [f"#{g.id} {g.title}" for g in wf.groups],
            "guidance_notes_added": notes_added,
            "nodes_retitled": titled,
            "knobs_highlighted_green": painted,
            "foreign_notes_parked": foreign_parked,
        },
    }
    if family:
        report["family_matched_on"] = detail["matched_on"]
    else:
        report["family_note"] = (
            "no model family detected - notes are generic; get_model_guidance + "
            "record_learning teach one"
        )
    if overlap_warning:
        # honest failure: organize_workflow must never silently ship a layout
        # it has itself diagnosed as broken
        report["warning"] = f"layout still has overlaps after auto-resolve: {overlap_warning}"
    return report
