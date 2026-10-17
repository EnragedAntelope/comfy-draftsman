"""Round-30: live-use feedback from a LoRA Manager build (0.22.0).

1. add_node gave a pack's JS widget types (nothing on the instance outputs them)
   bare sockets, so validate blocked save with unconnected-input
2. lint repeated validate's js-widget-input as unconnected-input
3. no way to remove a link without removing a node
4. a STRING -> COMBO mismatch now names the instance's bridge nodes
5. organize_workflow names the user's own groups it replaced
"""

import pytest

from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import Group, Workflow
from comfy_draftsman.graph.validate import validate

OI = {
    "Lora Stacker (LoraManager)": {
        "input": {
            "required": {
                "text": ["AUTOCOMPLETE_TEXT_LORAS", {"placeholder": "Search LoRAs"}],
                "loras": ["LORAS", {}],
            },
            "optional": {},
        },
        "output": ["LORA_STACK", "STRING", "STRING"],
        "output_name": ["LORA_STACK", "trigger_words", "active_loras"],
        "python_module": "custom_nodes.ComfyUI-Lora-Manager",
    },
    "ImageBlend": {
        "input": {"required": {"image": ["IMAGE"], "mode": [["a", "b"]]}},
        "output": ["IMAGE"],
        "python_module": "nodes",
    },
    "LoadImage": {
        "input": {"required": {"image": [["x.png"]]}},
        "output": ["IMAGE", "MASK"],
        "python_module": "nodes",
    },
    "StringNode": {
        "input": {"required": {"value": ["STRING", {"default": ""}]}},
        "output": ["STRING"],
        "python_module": "nodes",
    },
    "CR String To Combo": {
        "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
        "output": ["*", "STRING"],
        "python_module": "custom_nodes.ComfyUI_Comfyroll_CustomNodes",
    },
}


def test_unfed_custom_types_become_widget_backed_slots():
    wf = Workflow()
    node = wf.add_node("Lora Stacker (LoraManager)", object_info=OI)
    assert [(s.name, s.widget_name) for s in node.inputs] == [("text", "text"), ("loras", "loras")]
    assert node.widgets_values == []  # pack JS state, never guessed
    ui = wf.to_ui()["nodes"][0]["inputs"][0]
    assert ui["widget"] == {"name": "text"}
    findings = validate(wf, OI)
    blocking = [f for f in findings if f["level"] == "error" and not f.get("headless")]
    assert not blocking, blocking  # save_workflow no longer needs allow_invalid
    # lint no longer repeats validate's js-widget-input as unconnected-input
    assert not [f for f in lint(wf, OI) if f["code"] == "unconnected-input"]


def test_produced_type_stays_a_socket():
    wf = Workflow()
    node = wf.add_node("ImageBlend", object_info=OI)
    assert node.input_by_name("image").widget_name is None


def test_disconnect_removes_only_the_link():
    wf = Workflow()
    src = wf.add_node("LoadImage", object_info=OI)
    dst = wf.add_node("ImageBlend", object_info=OI)
    wf.connect(src.id, "IMAGE", dst.id, "image", OI)
    wf.disconnect(dst.id, "image")
    assert not wf.links and dst.input_by_name("image").link is None
    assert src.outputs[0].links == []
    with pytest.raises(ValueError, match="no linked input"):
        wf.disconnect(dst.id, "image")


def test_string_to_combo_mismatch_names_bridge_nodes():
    wf = Workflow()
    src = wf.add_node("StringNode", object_info=OI)
    dst = wf.add_node("ImageBlend", object_info=OI)
    with pytest.raises(ValueError, match="CR String To Combo"):
        wf.connect(src.id, "STRING", dst.id, "mode", OI)


def test_import_repoints_a_stale_link_slot():
    # a hand-edit dropped `loras`, shifting lora_stack from slot 2 to slot 1,
    # but the link table still said 2 - every reader skipped the link
    ui = {
        "nodes": [
            {"id": 1, "type": "A", "outputs": [{"name": "o", "type": "LORA_STACK", "links": [7]}]},
            {"id": 2, "type": "B", "inputs": [
                {"name": "text", "type": "T", "link": None},
                {"name": "lora_stack", "type": "LORA_STACK", "link": 7},
            ]},
        ],
        "links": [[7, 1, 0, 2, 2, "LORA_STACK"], [8, 1, 0, 2, 5, "LORA_STACK"]],
    }
    wf = Workflow.from_ui(ui)
    assert wf.links[7].target_slot == 1
    assert len(wf.import_notes) == 2 and "no such input" in wf.import_notes[1]


def test_collapsed_node_overlaps_only_by_its_title_pill():
    wf = Workflow()
    a = wf.add_node("LoadImage", object_info=OI)
    b = wf.add_node("ImageBlend", object_info=OI)
    a.pos, a.size, a.flags = [0.0, 120.0], [400.0, 170.0], {"collapsed": True}
    b.pos, b.size = [0.0, 210.0], [300.0, 400.0]
    assert not [f for f in lint(wf, OI) if f["code"] == "overlap"]
    a.flags = {}
    assert [f for f in lint(wf, OI) if f["code"] == "overlap"]


def test_layout_ops_disconnect_color_and_group_bounds():
    from comfy_draftsman import server

    wf = Workflow()
    src = wf.add_node("LoadImage", object_info=OI)
    dst = wf.add_node("ImageBlend", object_info=OI)
    wf.connect(src.id, "IMAGE", dst.id, "image", OI)
    wf.groups = [Group(id=1, title="g", bounding=[0, 0, 1, 1])]
    applied: list[str] = []
    server._apply_ops(wf, [
        {"op": "disconnect", "node_id": dst.id, "input": "image"},
        {"op": "set_color", "node_id": dst.id, "color": "#223", "bgcolor": "#335"},
        {"op": "set_group", "group_id": 1, "bounding": [5, 6, 70, 80]},
    ], OI, applied, set())
    assert not wf.links
    assert (dst.color, dst.bgcolor) == ("#223", "#335")
    assert wf.groups[0].bounding == [5.0, 6.0, 70.0, 80.0]


def test_organize_reports_replaced_user_groups():
    wf = Workflow()
    wf.add_node("LoadImage", object_info=OI)
    wf.groups = [
        Group(id=1, title="My loaders", bounding=[0, 0, 10, 10]),
        Group(id=2, title="📥 Inputs", bounding=[0, 0, 10, 10]),
    ]
    report = annotate(wf, OI)
    assert "My loaders" in report["groups_replaced"]
    assert "Inputs" not in report["groups_replaced"]
