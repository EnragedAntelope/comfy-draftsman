"""Model-family knowledge: bundled floor + persistent learned overlay.

Two layers, deep-merged at read time:

1. **Floor** - curated YAML per family in ``families/``. Sampling ranges,
   native resolutions, technique blocks (face_detailer, hires_fix, ...),
   note text, and variant overrides (turbo/lightning/... matched against the
   model filename). A floor, not a ceiling: last-reviewed dates and research
   directives are part of the data.

2. **Learned** - per-user overlay written by the ``record_learning`` MCP tool
   when the calling agent researches something (a model page's recommended
   FaceDetailer denoise, a new release's guidance value...). Stored as YAML
   in the configured learned dir, merged over the floor in every future
   session, sources tracked. This is how draftsman gets smarter over time
   without shipping stale mega-guides.
"""

from __future__ import annotations

import copy
import fnmatch
import re
from datetime import date
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

# Widgets that name the DIFFUSION MODEL itself - the only reference family
# detection may trust. A LoRA/VAE/CLIP/ControlNet filename can carry any other
# family's name in its own filename (a placeholder LoRA named "LTX23_..." on an
# H3 graph, a "SDXL" merge with "Flux" in its marketing name) without saying
# anything about what the checkpoint/UNet loader actually is.
_PRIMARY_MODEL_WIDGETS = {
    "ckpt_name", "unet_name", "model_name", "model", "checkpoint", "model_path", "diffusion_model",
}
# Auxiliary widgets that can NEVER name the diffusion model - excluded from
# detection even when their value happens to end in a model extension.
_AUX_WIDGET_RE = re.compile(
    r"(^|_)(lora|vae|clip|clip_vision|control_?net|style_model|upscale_model|"
    r"ipadapter|embedding|t5|text_encoder|gligen|photomaker)",
    re.IGNORECASE,
)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base (overlay wins); returns base."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


@cache
def _load_floor() -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    root = resources.files(__package__) / "families"
    for entry in root.iterdir():
        if entry.name.endswith(".yaml"):
            data = yaml.safe_load(entry.read_text(encoding="utf-8"))
            families[data["family"]] = data
    return families


def _learned_path(learned_dir: Path | str, family: str) -> Path:
    safe = re.sub(r"[^a-z0-9_\-]", "_", family.lower())
    return Path(learned_dir) / f"{safe}.yaml"


def _load_learned(learned_dir: Path | str | None, family: str) -> dict[str, Any] | None:
    if learned_dir is None:
        return None
    path = _learned_path(learned_dir, family)
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or None


def list_families(learned_dir: Path | str | None = None) -> list[str]:
    names = set(_load_floor())
    if learned_dir is not None:
        directory = Path(learned_dir)
        if directory.is_dir():
            for path in directory.glob("*.yaml"):
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                names.add(data.get("family", path.stem))
    return sorted(names)


def _matches(filename: str, patterns: list[str]) -> bool:
    name = filename.lower().replace("\\", "/")
    for pattern in patterns:
        p = pattern.lower()
        if ("*" in p and fnmatch.fnmatch(name, p)) or p in name:
            return True
    return False


def get_guidance(
    family: str,
    model_filename: str | None = None,
    learned_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Family guidance: floor <- learned overlay <- matching variant, merged."""
    floor = _load_floor()
    learned = _load_learned(learned_dir, family)
    if family not in floor and learned is None:
        raise KeyError(family)
    data = copy.deepcopy(floor.get(family, {"family": family}))
    if learned:
        deep_merge(data, learned.get("data", {}))
        data["learned_sources"] = learned.get("sources", [])
    variants = data.pop("variants", {}) or {}
    data["variant"] = None
    if model_filename:
        for variant_name, variant in variants.items():
            if _matches(model_filename, variant.get("patterns", [])):
                overlay = {k: v for k, v in variant.items() if k != "patterns"}
                deep_merge(data, overlay)
                data["variant"] = variant_name
                break
    return data


def save_learning(
    learned_dir: Path | str,
    family: str,
    updates: dict[str, Any],
    source: str,
) -> Path:
    """Merge researched findings into the persistent learned overlay for a family."""
    path = _learned_path(learned_dir, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    existing.setdefault("family", family)
    existing["data"] = deep_merge(existing.get("data", {}), updates)
    sources = existing.setdefault("sources", [])
    sources.append({"date": date.today().isoformat(), "source": source})
    path.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _model_refs(wf, object_info: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(widget_name, filename, role) triples for model file references.

    role is "primary" (names the diffusion model/checkpoint itself - the only
    thing family DETECTION may trust), "aux" (LoRA/VAE/CLIP/ControlNet/... -
    can carry any other family's name in its own filename), or "other" (a
    model-shaped filename on a widget that's neither - a fallback detection
    signal, weaker than primary but still real).
    """
    from ..graph import widgets as w  # local import to avoid cycle

    found: list[tuple[str, str, str]] = []
    for node in wf.nodes.values():
        schema = object_info.get(node.type)
        if schema is None:
            continue
        try:
            named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        except ValueError:
            continue
        for key, value in named.items():
            if not isinstance(value, str):
                continue
            is_model_ref = key in _PRIMARY_MODEL_WIDGETS or re.search(
                r"\.(safetensors|ckpt|sft|gguf|pt)$", value
            )
            if not is_model_ref:
                continue
            if key in _PRIMARY_MODEL_WIDGETS and not _AUX_WIDGET_RE.search(key):
                role = "primary"
            elif _AUX_WIDGET_RE.search(key):
                role = "aux"
            else:
                role = "other"
            found.append((key, value, role))
    return found


def matching_sources(guidance: dict[str, Any], filenames: list[str]) -> list[dict[str, str]]:
    """Curated download-source entries (``what``/``url``) whose ``match``
    patterns hit one of the given filenames - never invents a URL, only
    surfaces ones a family YAML or a record_learning call already curated.

    NOT the same key as ``learned_sources`` (research provenance - dates and
    a source string for what record_learning wrote, tracked separately by
    get_guidance). This is model-file download links, keyed ``sources`` on
    the family data itself, e.g.::

        sources:
          - match: ["ae.safetensors"]
            what: "FLUX VAE"
            url: "https://huggingface.co/..."
    """
    hits: list[dict[str, str]] = []
    for entry in guidance.get("sources") or []:
        patterns = entry.get("match") or []
        if any(_matches(fn, patterns) for fn in filenames):
            hits.append({"what": str(entry.get("what", "")), "url": str(entry.get("url", ""))})
    return hits


def model_filenames(wf, object_info: dict[str, Any]) -> list[str]:
    """String widget values that look like model file references (all roles -
    used for search/matching, where a LoRA name is a legitimate signal)."""
    return [filename for _, filename, _ in _model_refs(wf, object_info)]


def primary_model_filenames(wf, object_info: dict[str, Any]) -> list[str]:
    """Filenames that actually name the diffusion model - primary refs first,
    then other model-shaped refs, never aux (LoRA/VAE/CLIP/...). Use this for
    anything that should describe THE model (variant matching, guidance)."""
    primary = [f for _, f, role in _model_refs(wf, object_info) if role == "primary"]
    other = [f for _, f, role in _model_refs(wf, object_info) if role == "other"]
    return primary + other


def _detect_index(learned_dir: Path | str | None) -> dict[str, dict[str, Any]]:
    """Family -> {detect, loader} from the floor, overlaid with learned families.

    Learned overlays can carry their own ``detect``/``loader`` (written via
    record_learning with a detect block), so a model researched once becomes
    self-detecting in later sessions without editing the shipped floor.
    """
    index: dict[str, dict[str, Any]] = {
        name: copy.deepcopy(data) for name, data in _load_floor().items()
    }
    if learned_dir is not None:
        directory = Path(learned_dir)
        if directory.is_dir():
            for path in directory.glob("*.yaml"):
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                family = raw.get("family", path.stem)
                data = raw.get("data", {}) or {}
                if family in index:
                    deep_merge(index[family], data)
                else:
                    index[family] = {"family": family, **data}
    return index


def _detect_over_refs(
    refs: list[tuple[str, str, str]], index: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None, str | None]:
    """Best (family, matched_pattern, widget) over the given refs, or all Nones."""
    best_key: tuple[int, int] = (0, 0)
    best_family: str | None = None
    best_pattern: str | None = None
    best_widget: str | None = None
    for widget_name, filename, _role in refs:
        for family, data in index.items():
            patterns = (data.get("detect") or {}).get("checkpoint_patterns", [])
            matched = [p for p in patterns if _matches(filename, [p])]
            if not matched:
                continue
            loader = data.get("loader")
            loader_score = 1
            if (widget_name == "ckpt_name" and loader == "checkpoint") or (
                widget_name in ("unet_name", "model_name") and loader == "unet_clip_vae"
            ):
                loader_score = 2
            longest = max(matched, key=len)
            key = (loader_score, len(longest))
            if key > best_key:
                best_key, best_family, best_pattern, best_widget = key, family, longest, widget_name
    return best_family, best_pattern, best_widget


def detect_family_detail(
    wf, object_info: dict[str, Any], learned_dir: Path | str | None = None
) -> dict[str, str | None]:
    """Family detection from model filenames, disambiguated by loader topology
    and pattern specificity, anchored to the DIFFUSION MODEL - never a LoRA,
    VAE, CLIP, or other auxiliary reference.

    Merge names lie ("...XLFluxPony...DMD" is an SDXL merge, not FLUX), so a
    filename pattern match alone scores 1; a match whose loader widget agrees
    with the family's loader style (ckpt_name for checkpoint families,
    unet_name for split-loader families) scores 2. Ties break on the length of
    the matched pattern, so a specific pattern ("krea2") wins over a generic
    substring of it ("krea") when two families share a loader topology.

    Two passes: primary model references (ckpt_name/unet_name/...) are tried
    first; only if none of them match anything does the weaker "other"
    fallback run (a model-shaped filename on some widget that's neither a
    named primary slot nor a known auxiliary one). Auxiliary references
    (lora_name, vae_name, clip_name, ...) are never considered - a LoRA named
    "LTX23_..." on an H3 checkpoint's graph must not detect as LTX.

    Returns {"family", "matched_on", "widget"} - "matched_on" is the specific
    pattern that won, for a caller to sanity-check a surprising guess.
    """
    # built once: _detect_index globs the learned dir and parses every YAML in
    # it, and find_workflow calls this for each of up to 400 saved workflows -
    # rebuilding it per model reference made that quadratic in disk reads for
    # no benefit (the index can't change mid-call).
    index = _detect_index(learned_dir)
    refs = _model_refs(wf, object_info)
    primary = [r for r in refs if r[2] == "primary"]
    family, pattern, widget = _detect_over_refs(primary, index)
    if family is None:
        other = [r for r in refs if r[2] == "other"]
        family, pattern, widget = _detect_over_refs(other, index)
    return {"family": family, "matched_on": pattern, "widget": widget}


def detect_family(
    wf, object_info: dict[str, Any], learned_dir: Path | str | None = None
) -> str | None:
    """Family detection from model filenames. See detect_family_detail for the
    full mechanics and matched-pattern provenance."""
    return detect_family_detail(wf, object_info, learned_dir=learned_dir)["family"]


# Drivers report a little under the marketing capacity (a "16GB" card reports
# 15.99GB, and some reserve a slice for display), so a floor is met when the
# device is within this much of it. Without the slack every exactly-spec'd card
# would be told it is insufficient.
_VRAM_SLACK_GB = 0.5


def bytes_to_gb(value: Any) -> float | None:
    """Bytes -> GB, rounded to one decimal. None for anything non-numeric -
    /system_stats has been seen returning nulls for a CPU-only device."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return round(float(value) / (1024**3), 1)


def _largest_device(devices: list[dict[str, Any]]) -> tuple[float, float | None] | None:
    """(total_gb, free_gb) of the device with the most VRAM, or None if no
    device reports a usable total. Largest, not first: a laptop with an iGPU at
    index 0 must not be mistaken for the card the render will actually run on."""
    best: tuple[float, float | None] | None = None
    for device in devices or []:
        total = bytes_to_gb(device.get("vram_total"))
        if total is None or total <= 0:
            continue
        if best is None or total > best[0]:
            best = (total, bytes_to_gb(device.get("vram_free")))
    return best


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fit_verdict(
    guidance: dict[str, Any], devices: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Will this family's model actually run on this hardware? None when there
    is nothing worth saying.

    None - not "ok" - is the common answer, and deliberately so: no curated
    ``hardware`` block, no devices, or a comfortable fit all mean the caller
    emits no key at all. A block that says "everything is fine" is pure token
    cost on every call, forever.

    The comparison is against vram_TOTAL, never vram_free. A free-VRAM shortfall
    is "release the cached model first", not "this model does not fit your GPU" -
    conflating them produces a wrong verdict every time another job is resident.
    Free VRAM gets its own, separately actionable ``tight`` verdict.

    ``hardware`` deep-merges from a variant overlay like every other block, so a
    variant with a genuinely different floor must restate BOTH ``minimum`` and
    ``recommended`` - lowering only one leaves the family's other number in
    place (the same convention sdxl/turbo follows for min/max/default).
    """
    vram = ((guidance.get("hardware") or {}).get("vram_gb")) or {}
    minimum = _num(vram.get("minimum"))
    recommended = _num(vram.get("recommended"))
    if minimum is None and recommended is None:
        return None
    device = _largest_device(devices)
    if device is None:
        return None
    total, free = device
    hardware = guidance.get("hardware") or {}
    notes = str(hardware.get("notes") or "").strip()
    source = str(hardware.get("source") or "")

    def _verdict(kind: str, required: float, advice: str) -> dict[str, Any]:
        return {
            "verdict": kind,
            "vram_total_gb": total,
            "required_gb": required,
            "advice": f"{advice} {notes}".strip(),
            **({"source": source} if source else {}),
        }

    floor = minimum if minimum is not None else recommended
    if floor is None:  # unreachable - one of the two is set - but not asserted
        return None
    if total + _VRAM_SLACK_GB < floor:
        return _verdict(
            "insufficient",
            floor,
            f"{total}GB of VRAM is below the {floor}GB this family needs - expect "
            "heavy offloading to system RAM, or an out-of-memory failure.",
        )
    # Enough installed, but something else is holding it. Actionable right now,
    # so it wins over the headroom note below.
    if free is not None and free + _VRAM_SLACK_GB < floor:
        return _verdict(
            "tight",
            floor,
            f"only {free}GB of the card's {total}GB is free - free the cached model "
            "first with manage_queue(action='free', unload_models=True).",
        )
    if recommended is not None and total + _VRAM_SLACK_GB < recommended:
        return _verdict(
            "tight",
            recommended,
            f"{total}GB runs this family but is under the {recommended}GB recommended - "
            "expect offloading, and prefer the quantized weights.",
        )
    return None

