"""Widget-effect glossary + technique tradeoff lookup for organize_workflow's
per-band notes: what a knob does, what it may be set to, and roughly what it
trades off (e.g. "raising EasyCache's threshold runs faster but softens
detail"). Two bounded data sources, deliberately generic - neither claims a
RECOMMENDED value, only what moving the knob trades off:

1. ``_KNOB_EFFECTS`` - a small in-code glossary of near-universal ComfyUI
   widget names (steps, cfg, seed, ...). Works across any node pack, since
   these names are conventions, not tied to one node's schema.
2. ``techniques.yaml`` - class-name-pattern -> one-line tradeoff for
   well-known speed/quality techniques (EasyCache, SageAttention,
   TorchCompile...) whose tradeoff belongs to the TECHNIQUE, not to any one
   widget name, so a per-widget glossary can't express it.

Range/choices in a rendered row always come straight from ``/object_info``
(via ``comfy.catalog._apply_choices``, the same cap used everywhere else) -
never invented.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from typing import Any

import yaml

from ..comfy.catalog import MAX_COMBO_CHOICES, _apply_choices
from . import widgets as w
from .model import Node

# One short, generic tradeoff sentence per widget name. Never a claim about
# what value to use - only what moving the knob costs/gains.
_KNOB_EFFECTS: dict[str, str] = {
    "steps": "More steps = more refinement, with diminishing returns past the "
    "model's sweet spot; fewer steps is faster but can look unfinished.",
    "cfg": "Higher CFG follows the prompt more literally but can oversaturate "
    "or burn; lower is more creative but may ignore the prompt.",
    "denoise": "Higher denoise changes the image more (closer to a fresh "
    "generation); lower preserves more of the input.",
    "sampler_name": "Changes the sampling algorithm - affects speed and "
    "sharpness, and how much a given step count buys you.",
    "scheduler": "Changes how noise is removed across steps - pairs with "
    "sampler_name; some samplers expect a specific scheduler.",
    "seed": "Same seed + same settings = same result. Change it for a new "
    "variation; keep it fixed to iterate on one image.",
    "control_after_generate": "How the seed changes between runs: fixed keeps "
    "it, increment/decrement steps it by 1, randomize picks a new one each run.",
    "width": "Output width in pixels - usually needs to match the model's "
    "trained resolution bucket, or composition degrades.",
    "height": "Output height in pixels - same caveat as width.",
    "batch_size": "How many images render per run - more takes proportionally "
    "longer and more VRAM.",
    "strength_model": "How strongly the LoRA affects the diffusion model - "
    "higher is a stronger stylistic pull, can overpower the base model.",
    "strength_clip": "How strongly the LoRA affects text understanding - "
    "usually kept close to strength_model unless the LoRA page says otherwise.",
    "guidance": "Distilled-guidance strength (FLUX-style) - higher follows "
    "the prompt more literally; unlike cfg, doesn't need a negative prompt.",
    "shift": "Shifts the noise schedule - affects how detail resolves across "
    "steps; usually a per-model tuned constant.",
    "start_at_step": "Skips early steps (starts denoising partway through) - "
    "used with an already-noised input, e.g. img2img/hires-fix.",
    "end_at_step": "Stops before the final step, leaving some noise - usually "
    "paired with a second sampler pass.",
    "upscale_by": "Scale multiplier applied on top of the current size - "
    "higher costs more time/VRAM for more detail.",
    "feather": "Blends the edges of a mask/crop over this many pixels - "
    "higher hides seams but softens detail near the edge.",
    "weight": "How strongly this effect/conditioning is applied - higher is "
    "a stronger pull, can overpower everything else.",
}

# Widgets whose numeric range is legally huge but never meaningfully a "pick
# within this range" choice (a seed's max is 2**64-1) - omit the range cell
# for these rather than print an unreadable number.
_UNBOUNDED_RANGE_NAMES = {"seed", "noise_seed"}

_ROW_VALUE_CAP = 60


@cache
def _load_techniques() -> list[dict[str, Any]]:
    path = resources.files("comfy_draftsman.knowledge") / "techniques.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def technique_note(class_type: str) -> str | None:
    """One-line tradeoff for a well-known technique node (EasyCache,
    SageAttention, TorchCompile...), matched by class-name pattern against
    techniques.yaml - or None if this class isn't a recognized technique."""
    name = class_type.lower()
    for entry in _load_techniques():
        patterns = entry.get("match") or []
        if any(re.search(p, name) for p in patterns):
            note = entry.get("note")
            return str(note) if note else None
    return None


def _clip(value: Any) -> str:
    text = str(value)
    return text[:_ROW_VALUE_CAP] + "…" if len(text) > _ROW_VALUE_CAP else text


def _range_or_choices(class_type: str, widget_name: str, object_info: dict[str, Any]) -> str:
    """Human-readable range/choices straight from the live schema, capped the
    same way get_node_info caps combo lists. Empty string if the schema
    doesn't declare bounds/choices (free-form) or the widget's range is
    unbounded in practice (seed-like names)."""
    if widget_name in _UNBOUNDED_RANGE_NAMES:
        return ""
    schema = object_info.get(class_type)
    if schema is None:
        return ""
    for section in ("required", "optional"):
        spec = (schema.get("input", {}).get(section) or {}).get(widget_name)
        if spec is None:
            continue
        if not (isinstance(spec, list | tuple) and spec):
            return ""
        kind = spec[0]
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(kind, list):
            entry: dict[str, Any] = {}
            _apply_choices(entry, kind, "", MAX_COMBO_CHOICES)
            choices = entry["choices"]
            text = ", ".join(str(c) for c in choices)
            hidden = entry.get("choices_truncated", 0) - len(choices)
            return text + (f" (+{hidden} more)" if hidden > 0 else "")
        if str(kind).upper() in ("INT", "FLOAT"):
            lo, hi = opts.get("min"), opts.get("max")
            if lo is not None and hi is not None:
                return f"{lo}-{hi}"
        return ""
    return ""


def _is_wired(node: Node, name: str) -> bool:
    slot = node.input_by_name(name)
    return slot is not None and slot.link is not None


def knob_rows(
    members: list[Node],
    object_info: dict[str, Any],
    guidance: dict[str, Any] | None = None,
) -> list[str]:
    """Markdown table rows (``| knob | now | range/choices | effect |``) for
    every glossary-known widget across these nodes. A knob wired from
    upstream is listed as ``<- wired`` with no range/effect claimed - it
    isn't hand-editable here, so a range/effect sentence would mislead.
    Family guidance (a per-knob ``note``, e.g. H3's "No CFG...") wins over
    the generic glossary sentence when both have something to say."""
    sampling_guidance = (guidance or {}).get("sampling") or {}
    rows: list[str] = []
    for node in members:
        try:
            named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        except (ValueError, KeyError):
            continue
        for name, value in named.items():
            if name not in _KNOB_EFFECTS:
                continue
            if _is_wired(node, name):
                rows.append(f"| {name} | *(wired)* | | |")
                continue
            range_str = _range_or_choices(node.type, name, object_info)
            effect = (sampling_guidance.get(name) or {}).get("note") or _KNOB_EFFECTS[name]
            rows.append(f"| {name} | {_clip(value)} | {range_str} | {effect} |")
    return rows
