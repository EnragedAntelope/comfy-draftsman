"""0.25.1: widget type mismatches, drift reported on import, and set_resolution."""

import json

import pytest

import comfy_draftsman.server as server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate

INFO = {
    "Patch": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "structure_freeze": ["BOOLEAN", {"default": False}],
                "normalize_strength": ["FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}],
                "mode": [["a", "b"]],
            }
        },
        "output": ["MODEL"],
        "python_module": "custom_nodes.pack",
    },
    "Loader": {"input": {"required": {}}, "output": ["MODEL"]},
    "EmptyLatentImage": {
        "input": {"required": {"width": ["INT", {"default": 1024, "min": 16, "max": 8192}],
                               "height": ["INT", {"default": 1024, "min": 16, "max": 8192}]}},
        "output": ["LATENT"],
        "python_module": "nodes",
    },
    "JigsawHiRes": {
        "input": {"required": {"model": ["MODEL"],
                               "width": ["INT", {"default": 1024, "min": 64, "max": 8192}],
                               "height": ["INT", {"default": 1024, "min": 64, "max": 8192}]}},
        "output": ["MODEL"],
    },
    "SizeCalc": {"input": {"required": {}}, "output": ["INT"]},
}


def _patch_wf(values):
    ui = {"nodes": [{"id": 1, "type": "Patch", "widgets_values": values,
                     "inputs": [{"name": "model", "type": "MODEL", "link": None}], "outputs": []}],
          "links": []}
    return Workflow.from_ui(ui)


def _by_input(findings):
    return {f.get("input"): (f["code"], f["level"]) for f in findings if f.get("input")}


def test_boolean_given_int_and_float_given_text_are_flagged():
    found = _by_input(validate(_patch_wf([1, "strong", "a"]), INFO))
    # bool(1) still runs, so a warning; float("strong") is rejected by ComfyUI
    assert found["structure_freeze"] == ("widget-type-mismatch", "warning")
    assert found["normalize_strength"] == ("widget-type-mismatch", "error")


def test_numeric_text_in_float_is_a_warning():
    found = _by_input(validate(_patch_wf([True, "0.5", "a"]), INFO))
    assert found["normalize_strength"] == ("widget-type-mismatch", "warning")


def test_number_in_custom_combo_is_a_type_mismatch_not_an_unlisted_option():
    found = _by_input(validate(_patch_wf([True, 0.5, 0.7]), INFO))
    assert found["mode"] == ("widget-type-mismatch", "warning")


def test_shifted_values_after_a_count_change_say_so():
    # a parameter was removed from the front: every value moved one slot left
    findings = validate(_patch_wf([0.5, "a"]), INFO)
    assert any(f["code"] == "widget-count-drift" and f["level"] == "warning" for f in findings)
    mismatch = [f for f in findings if f["code"] == "widget-type-mismatch"]
    assert mismatch and all("shifted" in f["message"] for f in mismatch)


def test_many_mismatches_on_one_node_merge_into_one_finding():
    # same count, every value in another widget's slot (a pack reordered them)
    findings = validate(_patch_wf([0.5, "a", True]), INFO)
    (merged,) = [f for f in findings if f["code"] == "widget-type-mismatch"]
    assert merged["inputs"] == ["structure_freeze", "normalize_strength", "mode"]
    # float("a") fails the run, but the editor may read a JS-reordered pack fine
    assert merged["level"] == "error" and merged["headless"] is True


def test_clean_node_has_no_type_findings():
    assert not [f for f in validate(_patch_wf([True, 0.5, "a"]), INFO) if f["code"] == "widget-type-mismatch"]


@pytest.fixture
def isolated(monkeypatch):
    store = {}

    class _Sess:
        def persist(self, wid):
            pass

        def create(self, wf, title=None):
            store["wf"] = wf
            return "wid"

    async def _info(refresh=False):
        return INFO

    monkeypatch.setattr(server, "_session", lambda: _Sess())
    monkeypatch.setattr(server, "_object_info", _info)
    monkeypatch.setattr(server, "_summary", lambda wid, wf: {"workflow_id": wid})
    return store


async def test_import_reports_drifted_nodes(isolated):
    ui = _patch_wf([0.5, "a"]).to_ui()
    result = await server.import_workflow(workflow_json=json.dumps(ui))
    codes = {f["code"] for f in result["widget_drift"]}
    assert {"widget-count-drift", "widget-type-mismatch"} <= codes


async def test_clean_import_has_no_drift_key(isolated):
    result = await server.import_workflow(workflow_json=json.dumps(_patch_wf([True, 0.5, "a"]).to_ui()))
    assert "widget_drift" not in result


# --- set_resolution ---------------------------------------------------------

def _size_graph():
    wf = Workflow.new()
    loader = wf.add_node("Loader", object_info=INFO)
    jig = wf.add_node("JigsawHiRes", object_info=INFO)
    canvas = wf.add_node("EmptyLatentImage", object_info=INFO)
    wf.connect(loader.id, 0, jig.id, "model", INFO)
    return wf, jig, canvas


def _named(wf, node):
    from comfy_draftsman.graph.widgets import widgets_to_named
    return widgets_to_named(node.type, node.widgets_values, INFO, {s.name for s in node.inputs})


def test_set_resolution_sets_canvas_and_size_tuned_patch():
    wf, jig, canvas = _size_graph()
    applied, touched = [], set()
    server._apply_ops(wf, [{"op": "set_resolution", "width": 1536, "height": 640}], INFO, applied, touched)
    for node in (jig, canvas):
        named = _named(wf, node)
        assert (named["width"], named["height"]) == (1536, 640)
    assert {jig.id, canvas.id} <= touched and "1536x640" in applied[0]


def test_set_resolution_sets_a_feeding_primitive_and_disconnects_computed_size():
    wf, _, canvas = _size_graph()
    prim = wf.add_node("PrimitiveNode", object_info=INFO)
    calc = wf.add_node("SizeCalc", object_info=INFO)
    wf.connect(prim.id, 0, canvas.id, "width", INFO)
    wf.connect(calc.id, 0, canvas.id, "height", INFO)
    server._apply_ops(wf, [{"op": "set_resolution", "width": 768, "height": 512}], INFO, [], set())
    assert prim.widgets_values[0] == 768
    assert canvas.input_by_name("width").link is not None  # the primitive stays wired
    assert canvas.input_by_name("height").link is None
    assert _named(wf, canvas)["height"] == 512


def test_set_resolution_checks_everything_before_changing_anything():
    wf, jig, canvas = _size_graph()
    before = (list(jig.widgets_values), list(canvas.widgets_values))
    with pytest.raises(ValueError, match="JigsawHiRes"):
        # 32 fits the canvas (min 16) but not Jigsaw (min 64)
        server._apply_ops(wf, [{"op": "set_resolution", "width": 32, "height": 512}], INFO, [], set())
    assert (jig.widgets_values, canvas.widgets_values) == before


def test_set_resolution_refuses_a_computed_size_that_also_supplies_the_latent():
    # Krea2 shape: one Resolution node feeds Jigsaw's width/height AND the latent
    info = {
        **INFO,
        "Resolution": {"input": {"required": {"megapixels": ["FLOAT", {}]}},
                       "output": ["INT", "INT", "LATENT"]},
        "KSampler": {"input": {"required": {"model": ["MODEL"], "latent_image": ["LATENT"]}},
                     "output": ["LATENT"]},
    }
    wf = Workflow.new()
    loader = wf.add_node("Loader", object_info=info)
    jig = wf.add_node("JigsawHiRes", object_info=info)
    res = wf.add_node("Resolution", object_info=info)
    sampler = wf.add_node("KSampler", object_info=info)
    wf.connect(loader.id, 0, jig.id, "model", info)
    wf.connect(res.id, 0, jig.id, "width", info)
    wf.connect(res.id, 1, jig.id, "height", info)
    wf.connect(res.id, 2, sampler.id, "latent_image", info)
    with pytest.raises(ValueError, match=r"Resolution #\d+ computes the size and also supplies"):
        server._apply_ops(wf, [{"op": "set_resolution", "width": 768, "height": 512}], info, [], set())
    assert jig.input_by_name("width").link is not None  # nothing changed


@pytest.mark.parametrize(("values", "ok"), [
    (["Manual", 1344, 768, False], True),   # fewer than the schema, but the prefix fits
    ([1344, 768, False, "Manual"], False),  # reordered: width lands in the combo
])
def test_set_resolution_on_a_drifted_canvas_needs_credible_positions(values, ok):
    info = {**INFO, "ResMaster": {"input": {"required": {
        "mode": [["Manual", "Auto"]],
        "width": ["INT", {"default": 512, "min": 0, "max": 8192}],
        "height": ["INT", {"default": 512, "min": 0, "max": 8192}],
        "auto": ["BOOLEAN", {}], "extra1": ["FLOAT", {}], "extra2": ["FLOAT", {}]}},
        "output": ["INT", "INT", "LATENT"]}}
    wf = Workflow.from_ui({"nodes": [{"id": 1, "type": "ResMaster", "widgets_values": values,
                                      "inputs": [], "outputs": []}], "links": []})
    op = [{"op": "set_resolution", "width": 1024, "height": 640}]
    if ok:
        server._apply_ops(wf, op, info, [], set())
        assert wf.nodes[1].widgets_values[1:3] == [1024, 640]
    else:
        with pytest.raises(ValueError, match="set_widget index="):
            server._apply_ops(wf, op, info, [], set())


def test_core_socket_added_after_save_is_unconnected_not_a_js_widget():
    # pack update added a required MODEL socket; the saved node predates it
    info = {"Balance": {"input": {"required": {"model": ["MODEL"], "balance": ["FLOAT", {"min": 0, "max": 1}]},
                                  "optional": {"debug": ["BOOLEAN", {}]}}, "output": ["MODEL"]}}
    wf = Workflow.from_ui({"nodes": [{"id": 1, "type": "Balance", "widgets_values": [0.5, False],
                                      "inputs": [], "outputs": []}], "links": []})
    findings = validate(wf, info)
    assert not [f for f in findings if f["code"] in ("widget-type-mismatch", "widget-count-drift")]
    assert any(f.get("input") == "model" and f["level"] == "error" for f in findings)


def test_set_resolution_without_a_size_node_says_what_to_do():
    wf = Workflow.new()
    wf.add_node("Loader", object_info=INFO)
    with pytest.raises(ValueError, match="set_widget"):
        server._apply_ops(wf, [{"op": "set_resolution", "width": 512, "height": 512}], INFO, [], set())
