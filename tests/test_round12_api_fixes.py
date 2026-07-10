"""Round-12: API-submission gaps found during live testing (see the debug log
that motivated these). Each fix mirrors a ComfyUI *frontend* behavior the raw
/prompt backend never performs:

1. custom JS-widget input types (LoraManager autocomplete, style-gallery button)
   survive UI->API conversion instead of being silently dropped
2. %date:FORMAT% filename-prefix tokens substituted at submit time
3. control_after_generate re-rolls seeds on run
4. case-insensitive connect type check (STRING == string)
5. step-alignment tolerant of an epsilon `min` grid origin
6. combo-membership severity gated by core-vs-custom node (no false-positive
   flood on client-populated pickers), findings capped for token discipline
"""

import random
from datetime import datetime

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.model import Workflow, _substitute_filename_tokens
from comfy_draftsman.graph.validate import _step_aligned, check_widget_value, validate

# Minimal object_info covering a core sampler, a custom LoRA loader whose `text`
# input is a JS-only widget type, a custom client-populated combo, a custom file
# combo, and a SaveImage. python_module marks core ("nodes") vs third-party.
OI = {
    "MegaSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0, "min": 0, "max": 1000, "step": 1}],
                "sampler_name": [["euler", "dpmpp_2m", "ddim"]],
                "denoise": ["FLOAT", {"default": 1.0, "min": 0.0001, "max": 1.0, "step": 0.01}],
            }
        },
        "output": ["LATENT"],
        "python_module": "nodes",
    },
    "LoraTextLoader": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "clip": ["CLIP"],
                "text": ["AUTOCOMPLETE_TEXT_LORAS"],  # custom JS widget, not a socket
                "strength": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}],
            }
        },
        "output": ["MODEL", "CLIP"],
        "python_module": "custom_nodes.lora_manager",
    },
    "WildcardPicker": {  # custom node whose combo is repopulated client-side
        "input": {"required": {"wildcard": [["__hair__", "__eyes__"]]}},
        "output": ["STRING"],
        "python_module": "custom_nodes.impact",
    },
    "CustomLoraLoader": {  # custom node, but a real on-disk file listing
        "input": {"required": {"lora_name": [["a.safetensors", "b.safetensors"]]}},
        "output": ["MODEL"],
        "python_module": "custom_nodes.x",
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "output": [],
        "python_module": "nodes",
    },
    "StringSource": {"input": {"required": {}}, "output": ["STRING"], "python_module": "nodes"},
    "ImageSource": {"input": {"required": {}}, "output": ["IMAGE"], "python_module": "nodes"},
    "StringSink": {
        "input": {"required": {"value": ["string", {"forceInput": True}]}},
        "output": [],
        "python_module": "nodes",
    },
}


# --- 1. custom JS-widget inputs survive UI->API ------------------------------


def _lora_loader_ui():
    """A LoraTextLoader whose only serialized sockets are model/clip - `text`
    stays a JS widget (in widgets_values, not in the inputs array), as the real
    LoraManager node exports it."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "LoraTextLoader",
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "clip", "type": "CLIP", "link": None},
                ],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": []},
                    {"name": "CLIP", "type": "CLIP", "links": []},
                ],
                "widgets_values": ["mylora:0.6", 0.8],
            }
        ],
        "links": [],
    }


def test_custom_widget_survives_to_api():
    wf = Workflow.from_ui(_lora_loader_ui())
    inputs = wf.to_api(OI)["1"]["inputs"]
    assert inputs["text"] == "mylora:0.6"  # would be dropped before the fix
    assert inputs["strength"] == 0.8


def test_custom_widget_not_flagged_unconnected():
    wf = Workflow.from_ui(_lora_loader_ui())
    findings = validate(wf, OI)
    unconnected = {f.get("input") for f in findings if f["code"] == "unconnected-input"}
    assert "text" not in unconnected  # it's a widget, not a dangling socket
    assert {"model", "clip"} <= unconnected  # genuine sockets still checked


def test_fresh_defaults_ignore_custom_widget_without_instance():
    # schema/fresh context (no socket_names): custom type stays conservative, so a
    # freshly created node's defaults array doesn't invent a slot for `text`
    assert w.widget_defaults("LoraTextLoader", OI) == [1.0]  # just `strength`


def test_js_widget_backed_input_blocks_loudly_not_silently():
    # The real LoraManager shape: `text` IS serialized as an input (carries a
    # `widget` marker) but its value is pack-specific JS state - unrunnable via the
    # raw API. Validate must BLOCK with an actionable error, never silently drop it.
    ui = {
        "nodes": [
            {
                "id": 1,
                "type": "LoraTextLoader",
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "text", "type": "AUTOCOMPLETE_TEXT_LORAS", "link": None,
                     "widget": {"name": "text"}},
                ],
                "outputs": [],
                "widgets_values": [{"version": 1, "textSnapshot": "<lora:x:1>"}],
            }
        ],
        "links": [],
    }
    findings = validate(Workflow.from_ui(ui), OI)
    js = [f for f in findings if f["code"] == "js-widget-input" and f.get("input") == "text"]
    assert js and js[0]["level"] == "error"
    # and it's NOT mislabeled as a plain unconnected socket
    assert not any(f["code"] == "unconnected-input" and f.get("input") == "text" for f in findings)


# --- 2. %date:FORMAT% substitution -------------------------------------------


def test_substitute_filename_tokens_unit():
    dt = datetime(2026, 7, 10, 14, 5, 9)
    assert _substitute_filename_tokens("x_%date:yyyy-MM-dd%_%date:hhmmss%", dt) == "x_2026-07-10_140509"
    assert _substitute_filename_tokens("%date%", dt) == "2026-07-10"
    assert _substitute_filename_tokens("plain_prefix", dt) == "plain_prefix"


def test_date_tokens_substituted_in_api_not_in_ui():
    ui = {
        "nodes": [
            {
                "id": 1,
                "type": "SaveImage",
                "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
                "outputs": [],
                "widgets_values": ["shot_%date:yyyy-MM-dd%"],
            }
        ],
        "links": [],
    }
    wf = Workflow.from_ui(ui)
    api_prefix = wf.to_api(OI)["1"]["inputs"]["filename_prefix"]
    assert "%" not in api_prefix and ":" not in api_prefix
    assert api_prefix.startswith("shot_20")  # a real date
    # the saved document keeps the literal token for the browser
    assert wf.to_ui()["nodes"][0]["widgets_values"][0] == "shot_%date:yyyy-MM-dd%"


# --- 3. control_after_generate seed roll -------------------------------------


def test_roll_randomize_changes_seed_fixed_does_not():
    rng = random.Random(0)
    values = [42, "randomize", "euler", 1.0]
    rolled, changed = w.roll_seed_controls("MegaSampler", values, OI, rng, socket_names={"model"})
    assert changed and rolled[0] != 42 and 0 <= rolled[0] <= 1000
    same, changed = w.roll_seed_controls("MegaSampler", [42, "fixed", "euler", 1.0], OI, rng, socket_names={"model"})
    assert not changed and same[0] == 42


def test_roll_increment_advances_by_step():
    rolled, changed = w.roll_seed_controls(
        "MegaSampler", [42, "increment", "euler", 1.0], OI, socket_names={"model"}
    )
    assert changed and rolled[0] == 43


def test_apply_seed_control_persists_on_workflow():
    ui = {
        "nodes": [
            {
                "id": 1,
                "type": "MegaSampler",
                "inputs": [{"name": "model", "type": "MODEL", "link": None}],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
                "widgets_values": [7, "increment", "euler", 1.0],
            }
        ],
        "links": [],
    }
    wf = Workflow.from_ui(ui)
    assert wf.apply_seed_control(OI) is True
    assert wf.nodes[1].widgets_values[0] == 8


# --- 4. case-insensitive connect ---------------------------------------------


def _two_node_wf():
    wf = Workflow.new()
    src = wf.add_node("StringSource", object_info=OI)
    sink = wf.add_node("StringSink", object_info=OI)
    img = wf.add_node("ImageSource", object_info=OI)
    return wf, src, sink, img


def test_connect_case_insensitive_types():
    wf, src, sink, _ = _two_node_wf()
    link = wf.connect(src.id, "STRING", sink.id, "value", object_info=OI)  # STRING -> string
    assert link is not None


def test_connect_genuine_mismatch_still_raises():
    wf, _, sink, img = _two_node_wf()
    with pytest.raises(ValueError, match="type mismatch"):
        wf.connect(img.id, "IMAGE", sink.id, "value", object_info=OI)  # IMAGE -> string


# --- 5. step-alignment tolerance ---------------------------------------------


def test_step_aligned_epsilon_min():
    assert _step_aligned(0.36, 0.0001, 0.01)  # was a false positive
    assert _step_aligned(0.5, 0.0001, 0.01)
    assert _step_aligned(0.45, 0.0, 0.01)
    assert not _step_aligned(0.457, 0.0001, 0.01)  # genuinely off-grid


def test_check_widget_value_denoise_epsilon_min():
    assert check_widget_value("MegaSampler", "denoise", 0.36, OI) is None
    assert "step" in (check_widget_value("MegaSampler", "denoise", 0.457, OI) or "")


# --- 6. combo severity gate + findings cap -----------------------------------


def _single_node_wf(node_type, widgets):
    ui = {"nodes": [{"id": 1, "type": node_type, "inputs": [], "outputs": [], "widgets_values": widgets}], "links": []}
    return Workflow.from_ui(ui)


def test_core_combo_bad_value_is_error():
    wf = _single_node_wf("MegaSampler", [0, "fixed", "fake_sampler", 1.0])
    codes = {(f["code"], f["level"]) for f in validate(wf, OI) if f.get("input") == "sampler_name"}
    assert ("invalid-combo-value", "error") in codes


def test_custom_client_populated_combo_is_warning_not_error():
    wf = _single_node_wf("WildcardPicker", ["__typed_by_user__"])
    findings = [f for f in validate(wf, OI) if f.get("input") == "wildcard"]
    assert findings and findings[0]["level"] == "warning"
    assert findings[0]["code"] == "combo-value-unlisted"
    assert all(f["level"] != "error" for f in validate(wf, OI))


def test_custom_file_combo_missing_is_error():
    wf = _single_node_wf("CustomLoraLoader", ["missing.safetensors"])
    codes = {(f["code"], f["level"]) for f in validate(wf, OI) if f.get("input") == "lora_name"}
    assert ("invalid-combo-value", "error") in codes


def test_check_widget_value_combo_gate():
    assert check_widget_value("MegaSampler", "sampler_name", "fake", OI)  # core -> error string
    assert check_widget_value("WildcardPicker", "wildcard", "__x__", OI) is None  # custom -> accept
    assert check_widget_value("CustomLoraLoader", "lora_name", "missing.safetensors", OI)  # file -> error


def test_cap_findings_keeps_errors_and_truncates():
    findings = [{"level": "error", "code": f"e{i}", "message": "x"} for i in range(5)]
    findings += [{"level": "warning", "code": f"w{i}", "message": "x"} for i in range(60)]
    capped = server._cap_findings(findings)
    assert len(capped) == server._FINDINGS_CAP + 1  # + truncation marker
    assert sum(1 for f in capped if f["level"] == "error") == 5  # no error dropped
    assert capped[-1]["code"] == "findings-truncated"


def test_cap_findings_noop_when_small():
    findings = [{"level": "warning", "code": "w", "message": "x"}]
    assert server._cap_findings(findings) == findings
