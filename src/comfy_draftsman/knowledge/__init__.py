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

_MODEL_WIDGETS = {"ckpt_name", "unet_name", "model_name", "model", "checkpoint"}


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


def _model_refs(wf, object_info: dict[str, Any]) -> list[tuple[str, str]]:
    """(widget_name, filename) pairs that look like model file references."""
    from ..graph import widgets as w  # local import to avoid cycle

    found: list[tuple[str, str]] = []
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
            if key in _MODEL_WIDGETS or re.search(r"\.(safetensors|ckpt|sft|gguf|pt)$", value):
                found.append((key, value))
    return found


def model_filenames(wf, object_info: dict[str, Any]) -> list[str]:
    """String widget values that look like model file references."""
    return [filename for _, filename in _model_refs(wf, object_info)]


def detect_family(wf, object_info: dict[str, Any]) -> str | None:
    """Family detection from model filenames, disambiguated by loader topology.

    Merge names lie ("...XLFluxPony...DMD" is an SDXL merge, not FLUX), so a
    filename pattern match alone scores 1; a match whose loader widget agrees
    with the family's loader style (ckpt_name for checkpoint families,
    unet_name for split-loader families) scores 2 and wins.
    """
    best_score, best_family = 0, None
    for widget_name, filename in _model_refs(wf, object_info):
        for family, data in _load_floor().items():
            patterns = (data.get("detect") or {}).get("checkpoint_patterns", [])
            if not _matches(filename, patterns):
                continue
            score = 1
            loader = data.get("loader")
            if (widget_name == "ckpt_name" and loader == "checkpoint") or (widget_name in ("unet_name", "model_name") and loader == "unet_clip_vae"):
                score += 1
            if score > best_score:
                best_score, best_family = score, family
    return best_family
