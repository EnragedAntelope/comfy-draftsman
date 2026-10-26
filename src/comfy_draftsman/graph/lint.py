"""Readability and wiring lint for workflows.

Findings are dicts: {"code", "message", "node_id"?}. An annotated, correctly
wired workflow lints clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import widgets as w
from .layout import is_text_display
from .model import MODE_BYPASS, MODE_MUTE, Node, Workflow

NOTE_TYPES = {"Note", "MarkdownNote"}
_PROMPT_WIDGETS = ("text", "prompt", "wildcard_text")

# Overlap is reported as ONE finding naming at most this many nodes. Every node
# added through edit_workflow sits at the default position until
# organize_workflow runs, so a freshly built graph is N-way self-overlapping:
# reporting it pairwise produced 190 findings for 20 nodes (and 1,326 for 52),
# which is both unreadable and the single largest token sink in the server.
_OVERLAP_IDS_SHOWN = 8


def _finding(code: str, message: str, node_id: int | None = None) -> dict[str, Any]:
    finding: dict[str, Any] = {"code": code, "message": message}
    if node_id is not None:
        finding["node_id"] = node_id
    return finding


def _upstream_nodes(wf: Workflow, node: Node, slot_name: str, depth: int = 4) -> list[Node]:
    """Nodes on the chain feeding an input slot (BFS upstream, bounded)."""
    found: list[Node] = []
    slot = node.input_by_name(slot_name)
    if slot is None or slot.link is None or slot.link not in wf.links:
        return found
    frontier = [wf.links[slot.link].origin_id]
    visited: set[int] = set()
    for _ in range(depth):
        next_frontier: list[int] = []
        for nid in frontier:
            if nid in visited or nid not in wf.nodes:
                continue
            visited.add(nid)
            upstream = wf.nodes[nid]
            found.append(upstream)
            for inp in upstream.inputs:
                if inp.link is not None and inp.link in wf.links:
                    next_frontier.append(wf.links[inp.link].origin_id)
        frontier = next_frontier
    return found


def _has_text_display(wf: Workflow, chain: list[Node]) -> bool:
    """Whether the user can see the text produced by this upstream chain.

    Two shapes count, because they show the identical string:

    - *inline*: a Show Text node sits IN the chain feeding the encoder.
    - *tapped*: a Show Text node hangs off an output of a node in the chain
      (generator -> Show Text, generator -> encoder). This is the more common
      hand-wired shape and arguably the better one - the display doesn't sit in
      the path it reports on - but the chain walk alone never saw it, so a
      correctly-previewed workflow was told to "insert a Show Text node" that was
      already there. A false lint teaches callers to ignore the rule.
    """
    if any(is_text_display(n.type) for n in chain):
        return True
    chain_ids = {n.id for n in chain}
    return any(
        is_text_display(consumer.type)
        for link in wf.links.values()
        if link.origin_id in chain_ids
        and (consumer := wf.nodes.get(link.target_id)) is not None
    )


def _missing_prompt_previews(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    """A positive prompt built upstream (wildcards, concatenators) is invisible
    to the user unless a Show Text-style node displays the final string."""
    from .annotate import _outputs_conditioning, _prompt_role

    findings: list[dict[str, Any]] = []
    for node in wf.nodes.values():
        if not _outputs_conditioning(node):
            continue  # an LLM/text node with a `prompt` input is not an encoder
        try:
            slots = set(w.widget_slot_names(node.type, object_info))
        except (ValueError, KeyError):
            continue
        if node.type != "CLIPTextEncode" and not (slots & set(_PROMPT_WIDGETS)):
            continue
        wired = [
            name
            for name in _PROMPT_WIDGETS
            if name in slots
            and (slot := node.input_by_name(name)) is not None
            and slot.link is not None
        ]
        if not wired or _prompt_role(wf, node) != "positive":
            continue
        chain = _upstream_nodes(wf, node, wired[0])
        findings.extend(f for f in _hidden_troubleshooting(wf, chain) if f not in findings)
        if not _has_text_display(wf, chain):
            findings.append(
                _finding(
                    "no-prompt-preview",
                    f"{node.type} #{node.id}: the positive prompt is generated "
                    "upstream, so the user never sees the final text - add a Show "
                    "Text node (e.g. ShowText|pys), either inline before this "
                    "encoder or tapped off the generator's text output",
                    node.id,
                )
            )
    return findings


_DIAGNOSTIC_OUTPUTS = {"troubleshooting", "status"}  # LLM/API node report outputs, by exact name


def _hidden_troubleshooting(wf: Workflow, chain: list[Node]) -> list[dict[str, Any]]:
    """A prompt generator's own report output (EA_LMStudio, Groq, OpenRouter...)
    left unwired: when the LLM fails or returns junk, the reason is never shown."""
    linked = {(link.origin_id, link.origin_slot) for link in wf.links.values()}
    return [
        _finding(
            "hidden-troubleshooting",
            f"{node.type} #{node.id} writes the positive prompt but its '{out.name}' output "
            "is not shown - wire it to a Show Text node so a failed or odd LLM answer explains itself",
            node.id,
        )
        for node in chain
        if node.mode not in (MODE_MUTE, MODE_BYPASS)
        for idx, out in enumerate(node.outputs)
        if out.name.lower() in _DIAGNOSTIC_OUTPUTS and out.type == "STRING" and (node.id, idx) not in linked
    ]


def _family_guidance(
    wf: Workflow, object_info: dict[str, Any], learned_dir: Path | str | None
) -> tuple[str | None, dict[str, Any]]:
    """(family, guidance) for the family-aware checks below; (None, {}) unless a
    caller threads learned_dir through (see _resolution_alignment_findings)."""
    if learned_dir is None:
        return None, {}
    from .. import knowledge  # local import: knowledge doesn't import lint

    family = knowledge.detect_family(wf, object_info, learned_dir=learned_dir)
    if not family:
        return None, {}
    return family, knowledge.get_guidance(family, learned_dir=learned_dir)


def _avoid_node_findings(
    wf: Workflow, family: str | None, guidance: dict[str, Any]
) -> list[dict[str, Any]]:
    """A node the detected family's YAML (or a record_learning call) says breaks
    it - e.g. ModelSamplingSD3 on MOSS-Image renders pure noise."""
    avoid = guidance.get("avoid_nodes") or {}
    return [
        _finding(
            "family-avoid-node",
            f"{node.type} #{node.id} on {guidance.get('display_name', family)}: {avoid[node.type]}",
            node.id,
        )
        for node in wf.nodes.values()
        if node.type in avoid and node.mode not in (MODE_MUTE, MODE_BYPASS)
    ]


def _resolution_alignment_findings(
    wf: Workflow, object_info: dict[str, Any], family: str | None, guidance: dict[str, Any]
) -> list[dict[str, Any]]:
    """A canvas node's width/height must be a multiple of the detected
    family's declared alignment (LTX pads/crops otherwise; DiT patchify
    architectures like FLUX/SD3 need it for the same structural reason).

    Fires ONLY when a caller explicitly threads a learned_dir through AND a
    family is detected AND it declares multiple_of - "a lint that fires on
    correct work is worse than no lint" (ARCHITECTURE.md): no family, no
    claim, rather than guessing at an unknown model's requirement.
    learned_dir=None (lint()'s default) is a deliberate opt-out, not "no
    learned overlay" - it keeps every caller that doesn't thread config
    through (most existing tests, any future direct lint() use) unaffected
    by this check regardless of what family YAMLs declare multiple_of; only
    the three server.py call sites (which always pass _config().learned_dir,
    a real path) turn it on. Never auto-rewrites the value; a workflow may
    deliberately use an odd size, and rewriting it or injecting a math node
    to force alignment would silently change the user's graph.
    """
    multiple_of = guidance.get("multiple_of")
    if not isinstance(multiple_of, int) or multiple_of <= 1:
        return []
    findings: list[dict[str, Any]] = []
    for node in wf.nodes.values():
        if not _is_size_node(node, object_info):
            continue
        try:
            named = w.widgets_to_named(node.type, node.widgets_values, object_info)
        except (ValueError, KeyError):
            continue
        for dim in ("width", "height"):
            value = named.get(dim)
            if not isinstance(value, int | float) or int(value) % multiple_of == 0:
                continue
            nearest = round(value / multiple_of) * multiple_of
            findings.append(
                _finding(
                    "resolution-not-aligned",
                    f"{node.type} #{node.id}: {dim}={int(value)} is not a multiple "
                    f"of {multiple_of} ({guidance.get('display_name', family)}) - "
                    f"nearest legal value is {nearest}",
                    node.id,
                )
            )
    return findings


def is_model_patch_size_node(node: Node, object_info: dict[str, Any]) -> bool:
    """MODEL in, MODEL out, with its own width/height widgets: the patch is
    tuned for a render size in pixels (ModelSamplingFlux, JigsawHiRes). One fed
    a LATENT (JigsawHiRes's optional input) sizes itself from it instead."""
    schema = object_info.get(node.type) or {}
    required = (schema.get("input") or {}).get("required") or {}
    return (
        not any(s.type == "LATENT" and s.link is not None for s in node.inputs)
        and list(schema.get("output") or []) == ["MODEL"]
        and any(isinstance(s, list | tuple) and s and s[0] == "MODEL" for s in required.values())
        and all(w.is_widget_input(required.get(dim)) for dim in ("width", "height"))
    )


def resolution_nodes(wf: Workflow, object_info: dict[str, Any]) -> list[Node]:
    """Active nodes whose width/height set, or are tuned for, the render size."""
    return [
        n for n in wf.nodes.values()
        if n.mode not in (MODE_MUTE, MODE_BYPASS)
        and (_is_size_node(n, object_info) or is_model_patch_size_node(n, object_info))
    ]


def _model_sampling_size_findings(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    """A size-tuned model patch upstream of a sampler whose latent comes straight
    from a canvas of another size. Traced per sampler, so a hi-res pass (latent
    from an upscale, not a canvas) is never compared to the first pass's canvas."""
    findings: list[dict[str, Any]] = []
    seen: set[int] = set()
    for sampler in wf.nodes.values():
        slot = sampler.input_by_name("latent_image")
        if sampler.mode in (MODE_MUTE, MODE_BYPASS) or slot is None or slot.link not in wf.links:
            continue
        canvas = wf.nodes.get(wf.links[slot.link].origin_id)
        if (
            canvas is None
            or canvas.mode in (MODE_MUTE, MODE_BYPASS)
            or not _is_size_node(canvas, object_info)
            or (size := _set_size(canvas, object_info)) is None
        ):
            continue
        for inp in sampler.inputs:
            for patch in _upstream_nodes(wf, sampler, inp.name, depth=8):
                if (
                    patch.id in seen
                    or patch.mode in (MODE_MUTE, MODE_BYPASS)
                    or not is_model_patch_size_node(patch, object_info)
                    or (patch_size := _set_size(patch, object_info)) is None
                    or patch_size == size
                ):
                    continue
                seen.add(patch.id)
                units = next(
                    (f" ({patch_size[0]}x{patch_size[1]} is that size in latent units /{f}; "
                     "these widgets take pixels)"
                     for f in (8, 16) if (patch_size[0] * f, patch_size[1] * f) == size),
                    "",
                )
                findings.append(
                    _finding(
                        "model-sampling-size-mismatch",
                        f"{patch.type} #{patch.id}: width/height {patch_size[0]}x{patch_size[1]} "
                        f"but {sampler.type} #{sampler.id} renders a {size[0]}x{size[1]} latent"
                        f"{units} - the model patch is tuned for the wrong resolution; set it "
                        "to the image size in pixels",
                        patch.id,
                    )
                )
    return findings


def lint(
    wf: Workflow,
    object_info: dict[str, Any],
    learned_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    real_nodes = [n for n in wf.nodes.values() if n.type not in NOTE_TYPES]

    if not wf.groups:
        findings.append(_finding("no-groups", "no groups: stages are not visually organized"))
    if not any(n.type in NOTE_TYPES for n in wf.nodes.values()):
        findings.append(_finding("no-notes", "no guidance notes for human readers"))

    linked_ids: set[int] = set()
    for link in wf.links.values():
        linked_ids.update((link.origin_id, link.target_id))

    for node in real_nodes:
        if node.mode in (MODE_MUTE, MODE_BYPASS):
            # a disabled node is dropped before the prompt is submitted, so its
            # own wiring is not a defect - validate skips it for the same reason,
            # and lint contradicting validate on the same graph (plus
            # save_workflow's "lint is not clean" nag over a deliberately muted
            # branch) was pure noise
            continue
        schema = object_info.get(node.type)
        if schema is not None:
            socket_names = {slot.name for slot in node.inputs}
            for name, spec in schema.get("input", {}).get("required", {}).items():
                # mirror validate's exemptions exactly - lint contradicting
                # validate on the same graph is pure noise, and save_workflow
                # nags about an unclean lint. A widget, a pack's JS widget, and
                # an autogrow container are all "not a socket"; the last is
                # checked as a group by validate's autogrow-underfilled instead.
                if (
                    w.is_widget_input(spec)
                    or w._is_custom_widget(name, spec, socket_names)
                    or w.autogrow_template(spec) is not None
                ):
                    continue
                slot = node.input_by_name(name)
                if slot is not None and slot.widget_name:
                    continue  # validate's js-widget-input already reports it
                if slot is None or slot.link is None:
                    findings.append(
                        _finding(
                            "unconnected-input",
                            f"{node.type} #{node.id}: required input '{name}' is not connected",
                            node.id,
                        )
                    )
        if node.id not in linked_ids and len(real_nodes) > 1:
            findings.append(
                _finding("orphan-node", f"{node.type} #{node.id} is connected to nothing", node.id)
            )
        try:
            slots = set(w.widget_slot_names(node.type, object_info))
        except (ValueError, KeyError):
            slots = set()
        if "text" in slots and node.title is None:
            findings.append(
                _finding(
                    "untitled-prompts",
                    f"{node.type} #{node.id} holds prompt text but has no descriptive title",
                    node.id,
                )
            )

    findings.extend(_missing_prompt_previews(wf, object_info))
    findings.extend(_output_behind_lazy_input(wf, object_info))
    findings.extend(_overlap_findings(wf))
    family, guidance = _family_guidance(wf, object_info, learned_dir)
    findings.extend(_avoid_node_findings(wf, family, guidance))
    findings.extend(_resolution_alignment_findings(wf, object_info, family, guidance))
    findings.extend(_native_resolution_findings(wf, object_info, family, guidance))
    findings.extend(_model_sampling_size_findings(wf, object_info))
    return findings


def _is_size_node(node: Node, object_info: dict[str, Any]) -> bool:
    """A node that sets the render size: an Empty*Latent canvas, or any node that
    outputs a LATENT from its own width/height widgets (TextEncodeFibo2)."""
    from .annotate import _is_canvas_node  # local import: annotate imports lint

    if _is_canvas_node(node.type):
        return True
    schema = object_info.get(node.type) or {}
    if "LATENT" not in (schema.get("output") or []):
        return False
    required = (schema.get("input") or {}).get("required") or {}
    return all(w.is_widget_input(required.get(dim)) for dim in ("width", "height"))


def _set_size(node: Node, object_info: dict[str, Any]) -> tuple[int, int] | None:
    """(width, height) typed on the node; None when wired, unreadable, or 0
    (0 means "the model's own size" on nodes like TextEncodeFibo2)."""
    if any((slot := node.input_by_name(dim)) is not None and slot.link is not None for dim in ("width", "height")):
        return None
    try:
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
    except (ValueError, KeyError):
        return None
    width, height = named.get("width"), named.get("height")
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return width, height
    return None


def _differs_from_defaults(node: Node, object_info: dict[str, Any]) -> bool:
    """A node doing something: a wired widget input, or a value off its schema
    default (ScaleROPE at scale 1.0 is a no-op). True when unreadable - unknown
    is not a finding."""
    if any(slot.link is not None and slot.widget_name is not None for slot in node.inputs):
        return True
    try:
        named = w.widgets_to_named(node.type, node.widgets_values, object_info)
    except (ValueError, KeyError):
        return True
    spec = (object_info.get(node.type) or {}).get("input") or {}
    defaults = {k: v[1].get("default") for part in spec.values() for k, v in part.items()
                if isinstance(v, list | tuple) and len(v) > 1 and isinstance(v[1], dict)}
    return not defaults or any(k in defaults and defaults[k] != v for k, v in named.items())


def _native_resolution_findings(
    wf: Workflow, object_info: dict[str, Any], family: str | None, guidance: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rendering above the size a family was trained at, when its YAML (or a
    record_learning call) says that breaks it - FIBO repeats subjects above
    ~1 MP - and the graph lacks the declared fix node. Silent with no data."""
    spec = guidance.get("native_resolution") or {}
    max_mp = spec.get("max_mp")
    if not isinstance(max_mp, int | float) or isinstance(max_mp, bool) or max_mp <= 0:
        return []
    active = [n for n in wf.nodes.values() if n.mode not in (MODE_MUTE, MODE_BYPASS)]
    fix = spec.get("fix_node")
    if fix and any(n.type == fix and _differs_from_defaults(n, object_info) for n in active):
        return []
    findings = []
    for node in active:
        if not _is_size_node(node, object_info) or (size := _set_size(node, object_info)) is None:
            continue
        mp = size[0] * size[1] / 2**20
        if mp <= max_mp * 1.15:  # trained sizes sit around max_mp, not exactly on it
            continue
        findings.append(
            _finding(
                "above-native-resolution",
                f"{node.type} #{node.id}: {size[0]}x{size[1]} is {mp:.1f} MP on "
                f"{guidance.get('display_name', family)} - {spec.get('note', '').strip()}"
                # the scale factor only means something when there is a fix node to set
                + (f" Here sqrt({max_mp:g}/{mp:.1f}) = {(max_mp / mp) ** 0.5:.2f}." if fix else ""),
                node.id,
            )
        )
    return findings


def _output_behind_lazy_input(
    wf: Workflow, object_info: dict[str, Any]
) -> list[dict[str, Any]]:
    """ComfyUI runs every ``output_node`` on every queue; laziness only prunes
    inputs. An output node wired straight into a lazy input (a switch's
    on_true/on_false) therefore executes whether or not that branch is selected,
    dragging its whole upstream in with it. Direct feeds only - a chain through
    another node is not followed."""
    findings = []
    for link in wf.links.values():
        origin, target = wf.nodes.get(link.origin_id), wf.nodes.get(link.target_id)
        if origin is None or target is None or link.target_slot >= len(target.inputs):
            continue
        if origin.mode in (MODE_MUTE, MODE_BYPASS):
            continue
        if not (object_info.get(origin.type) or {}).get("output_node"):
            continue
        name = target.inputs[link.target_slot].name
        inputs = (object_info.get(target.type) or {}).get("input") or {}
        spec = {**(inputs.get("required") or {}), **(inputs.get("optional") or {})}.get(name)
        if isinstance(spec, list | tuple) and len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("lazy"):
            findings.append(
                _finding(
                    "output-behind-lazy-input",
                    f"{origin.type} #{origin.id} is an output node feeding lazy input "
                    f"'{name}' of {target.type} #{target.id}; ComfyUI runs every output "
                    "node on every queue, so it executes whichever branch is selected",
                    origin.id,
                )
            )
    return findings


def _visible_box(n: Node) -> tuple[float, float, float, float]:
    """What the canvas draws: a collapsed node keeps its expanded ``size`` but
    shows only its title pill, which sits ABOVE ``pos`` (LiteGraph draws the
    title bar at pos.y - 30) and is about as wide as its title text."""
    x, y = n.pos[0], n.pos[1]
    if n.flags.get("collapsed"):
        # ponytail: ~8px per title character, not a font measurement
        width = min(n.size[0], 8.0 * len(n.title or n.type) + 40.0)
        return (x, y - 30.0, x + width, y)
    return (x, y, x + n.size[0], y + n.size[1])


def _overlap_findings(wf: Workflow) -> list[dict[str, Any]]:
    """Overlapping nodes make workflows unreadable - reported as ONE finding.

    Every node involved in any overlap is collected and named once. The previous
    per-pair report was quadratic in output: a graph whose nodes are all still at
    the default position (i.e. anything built with edit_workflow and not yet
    organized) produced N*(N-1)/2 findings that all said the same thing.
    """
    boxes = [(n.id, _visible_box(n)) for n in wf.nodes.values()]
    involved: set[int] = set()
    pairs = 0
    for i, (id_a, a) in enumerate(boxes):
        for id_b, b in boxes[i + 1 :]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                involved.update((id_a, id_b))
                pairs += 1
    if not involved:
        return []
    ids = sorted(involved)
    shown = ", ".join(f"#{nid}" for nid in ids[:_OVERLAP_IDS_SHOWN])
    more = len(ids) - _OVERLAP_IDS_SHOWN
    finding = _finding(
        "overlap",
        f"{len(ids)} node(s) overlap visually ({pairs} overlapping pair(s)): {shown}"
        + (f" (+{more} more)" if more > 0 else "")
        + " - organize_workflow lays the whole graph out in one pass",
    )
    finding["node_ids"] = ids
    return [finding]
