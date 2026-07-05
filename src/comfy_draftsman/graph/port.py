"""Retarget a workflow to a different model family (SDXL -> FLUX/Krea, ...).

Mechanical, knowledge-driven porting:
- sampler settings retuned to the target family's floor (CFG, steps,
  sampler, scheduler)
- loader topology swapped when the family requires it (checkpoint vs
  separate UNET/CLIP/VAE loaders), rewiring all consumers
- latent node class swapped (e.g. EmptyLatentImage -> EmptySD3LatentImage)
- technique nodes (FaceDetailer, ...) retuned from the family's technique
  blocks - detailer settings are family-specific, never universal
- model files picked from what is actually installed; anything that cannot
  be resolved mechanically comes back as an explicit flag for the agent

Everything the port cannot do safely is reported, never guessed silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import knowledge
from . import widgets as w
from .model import Workflow
from .validate import _combo_choices

_SAMPLER_WIDGET_MAP = {
    "cfg": ("cfg",),
    "steps": ("steps",),
    "samplers": ("sampler_name",),
    "schedulers": ("scheduler",),
}
_TECHNIQUE_KEY_MAP = {"sampler": "sampler_name"}


def _slots(wf: Workflow, node_id: int, object_info: dict[str, Any]) -> list[str]:
    try:
        return w.widget_slot_names(wf.nodes[node_id].type, object_info)
    except (ValueError, KeyError):
        return []


def _set_if_valid(
    wf: Workflow,
    node_id: int,
    name: str,
    value: Any,
    object_info: dict[str, Any],
    changes: list[str],
) -> None:
    node = wf.nodes[node_id]
    if name not in _slots(wf, node_id, object_info):
        return
    schema = object_info[node.type]
    spec = None
    for section in ("required", "optional"):
        spec = schema.get("input", {}).get(section, {}).get(name) or spec
    if spec is not None:
        choices = _combo_choices(spec)
        if choices is not None and value not in choices:
            return
    old = wf.get_widget(node_id, name, object_info)
    if old == value:
        return
    wf.set_widget(node_id, name, value, object_info)
    changes.append(f"{node.type} #{node_id}: {name} {old!r} -> {value!r}")


def _matching_files(choices: list[Any], patterns: list[str]) -> list[str]:
    return [
        str(choice)
        for choice in choices
        if any(p.lower() in str(choice).lower().replace("\\", "/") for p in patterns)
    ]


def _pick_file(choices: list[Any], patterns: list[str]) -> str | None:
    matches = _matching_files(choices, patterns)
    return matches[0] if matches else None


def _retune_samplers(wf, guidance, object_info, changes) -> None:
    sampling = guidance.get("sampling", {})
    for node in list(wf.nodes.values()):
        category = (object_info.get(node.type, {}).get("category") or "").lower()
        if "sampling" not in category:
            continue
        for knowledge_key, widget_names in _SAMPLER_WIDGET_MAP.items():
            source = sampling.get(knowledge_key)
            if source is None:
                continue
            value = source["default"] if isinstance(source, dict) else source[0]
            for widget_name in widget_names:
                _set_if_valid(wf, node.id, widget_name, value, object_info, changes)


def _swap_latent_nodes(wf, guidance, object_info, changes) -> None:
    target_class = guidance.get("latent_node")
    if not target_class or target_class not in object_info:
        return
    for node in wf.nodes.values():
        if node.type in ("EmptyLatentImage", "EmptySD3LatentImage") and node.type != target_class:
            named = w.widgets_to_named(node.type, node.widgets_values, object_info)
            node.type = target_class
            node.properties["Node name for S&R"] = target_class
            node.widgets_values = w.named_to_widgets(target_class, named, object_info)
            changes.append(f"latent node #{node.id} -> {target_class}")


def _swap_loader_topology(wf, guidance, object_info, changes, flags) -> None:
    loader_nodes = guidance.get("loader_nodes")
    checkpoints = [n for n in list(wf.nodes.values()) if n.type == "CheckpointLoaderSimple"]
    if guidance.get("loader") != "unet_clip_vae" or not loader_nodes:
        if checkpoints and guidance.get("detect"):
            # same topology: swap the checkpoint file to one from the target family
            patterns = guidance["detect"].get("checkpoint_patterns", [])
            for ckpt in checkpoints:
                spec = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
                picked = _pick_file(_combo_choices(spec) or [], patterns)
                if picked:
                    _set_if_valid(wf, ckpt.id, "ckpt_name", picked, object_info, changes)
                else:
                    flags.append(
                        f"no installed checkpoint matches {guidance['family']} "
                        f"(patterns {patterns}) - download one, then set ckpt_name on #{ckpt.id}"
                    )
        return

    role_for_output = {"MODEL": "model", "CLIP": "clip", "VAE": "vae"}
    for ckpt in checkpoints:
        replacements: dict[str, int] = {}
        for role, spec_def in loader_nodes.items():
            class_type = spec_def["class"]
            if class_type not in object_info:
                flags.append(f"loader node '{class_type}' not available on this instance")
                continue
            new_node = wf.add_node(class_type, object_info=object_info)
            for widget_name, value in (spec_def.get("widgets") or {}).items():
                _set_if_valid(wf, new_node.id, widget_name, value, object_info, changes)
            schema = object_info[class_type]
            all_specs = {
                name: spec
                for section in ("required", "optional")
                for name, spec in schema.get("input", {}).get(section, {}).items()
            }
            file_widget_patterns: dict[str, list[str]] = {}
            if spec_def.get("file_widget"):
                file_widget_patterns[spec_def["file_widget"]] = spec_def.get("file_patterns", [])
            for widget_name, patterns in (spec_def.get("file_widgets") or {}).items():
                file_widget_patterns[widget_name] = patterns
            for widget_name, patterns in file_widget_patterns.items():
                choices = _combo_choices(all_specs.get(widget_name, ["*"])) or []
                matches = _matching_files(choices, patterns)
                if matches:
                    _set_if_valid(wf, new_node.id, widget_name, matches[0], object_info, changes)
                    if len(matches) > 1:
                        flags.append(
                            f"{class_type}.{widget_name} (#{new_node.id}): picked "
                            f"{matches[0]!r} from {len(matches)} candidates - review "
                            f"alternatives: {matches[1:6]}"
                        )
                else:
                    flags.append(
                        f"no installed file matches {patterns} for {class_type}.{widget_name} "
                        f"(#{new_node.id}) - download the {guidance['family']} {role} file and set it"
                    )
            replacements[role] = new_node.id
            changes.append(f"added {class_type} #{new_node.id} ({role})")

        # rewire every consumer of the checkpoint's outputs
        for out in ckpt.outputs:
            role = role_for_output.get(out.type)
            new_id = replacements.get(role)
            for link_id in list(out.links):
                link = wf.links.get(link_id)
                if link is None:
                    continue
                target = wf.nodes[link.target_id]
                input_name = target.inputs[link.target_slot].name
                if new_id is None:
                    flags.append(
                        f"could not rewire {target.type}.{input_name} (was checkpoint {out.type})"
                    )
                    continue
                wf.connect(new_id, 0, link.target_id, input_name, object_info)
        try:
            ckpt_name = wf.get_widget(ckpt.id, "ckpt_name", object_info)
        except (ValueError, KeyError):
            ckpt_name = None
        wf.remove_node(ckpt.id)
        changes.append(
            f"replaced CheckpointLoaderSimple #{ckpt.id} ({ckpt_name}) with separate loaders"
        )


def _retune_techniques(wf, guidance, object_info, changes) -> None:
    for technique, settings in (guidance.get("techniques") or {}).items():
        if not isinstance(settings, dict):
            continue
        compact = technique.replace("_", "")
        for node in wf.nodes.values():
            if compact not in node.type.lower().replace("_", ""):
                continue
            for key, value in settings.items():
                if key in ("note",):
                    continue
                widget_name = _TECHNIQUE_KEY_MAP.get(key, key)
                before = len(changes)
                _set_if_valid(wf, node.id, widget_name, value, object_info, changes)
                if len(changes) > before:
                    changes[-1] += f" [technique {technique}]"


def port_workflow(
    wf: Workflow,
    target_family: str,
    object_info: dict[str, Any],
    learned_dir: Path | str | None = None,
) -> dict[str, Any]:
    try:
        guidance = knowledge.get_guidance(target_family, learned_dir=learned_dir)
    except KeyError:
        return {
            "target_family": target_family,
            "changes": [],
            "flags": [],
            "error": f"no knowledge for family '{target_family}'",
            "families": knowledge.list_families(learned_dir),
            "hint": (
                "research this family's loader topology + sampling online, then "
                "record_learning it (include a 'detect' block so it's recognized "
                "next time). Re-run port_workflow afterwards, or wire it manually "
                "with edit_workflow using get_model_guidance for settings."
            ),
        }
    changes: list[str] = []
    flags: list[str] = []

    _swap_loader_topology(wf, guidance, object_info, changes, flags)
    _retune_samplers(wf, guidance, object_info, changes)
    _swap_latent_nodes(wf, guidance, object_info, changes)
    _retune_techniques(wf, guidance, object_info, changes)

    if "guidance" in guidance.get("sampling", {}) and not any(
        n.type == "FluxGuidance" for n in wf.nodes.values()
    ):
        flags.append(
            "this family uses a guidance node (e.g. FluxGuidance "
            f"~{guidance['sampling']['guidance']['default']}) on the positive conditioning - "
            "add one with edit_workflow"
        )
    if guidance.get("research"):
        flags.append(f"verify against current sources: {guidance['research']}")
    return {"target_family": target_family, "changes": changes, "flags": flags}
