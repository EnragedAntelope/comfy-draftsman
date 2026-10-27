"""0.25.3 session feedback.

- add_node accepts the bare 'control_after_generate' for a node's one seed control
- an 'input:substr' filter naming another node's input leaves this node's combos whole
- organize's title pass reports nothing on a re-run
- an AttributeError on a model object hints at the pack's supported models
"""

import test_round45_fixes as r45
from comfy_draftsman import server
from comfy_draftsman.comfy.catalog import node_summary
from comfy_draftsman.comfy.progress import execution_error
from comfy_draftsman.graph.annotate import _title_nodes

object_info = r45.object_info  # shared pytest fixtures
wired = r45.wired
_prompt_chain = r45._prompt_chain


async def test_add_node_accepts_bare_control_after_generate(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [{
            "op": "add_node",
            "class_type": "KSampler",
            "widgets": {"seed": 7, "control_after_generate": "fixed"},
        }],
    )
    assert "error" not in result, result
    (node,) = wf.nodes.values()
    assert node.widgets_values[:2] == [7, "fixed"]


def test_filter_scoped_to_other_nodes_input_keeps_combos(object_info):
    summary = node_summary(object_info, "KSampler", choices_filter="ckpt_name:xl")
    combos = {i["name"]: i for i in summary["inputs"] if i.get("type") == "COMBO"}
    assert combos["sampler_name"]["choices"]
    assert "choices_matched" not in combos["sampler_name"]


def test_filter_scoped_to_own_input_still_filters(object_info):
    summary = node_summary(object_info, "KSampler", choices_filter="sampler_name:euler")
    combos = {i["name"]: i for i in summary["inputs"] if i.get("type") == "COMBO"}
    assert all("euler" in c for c in combos["sampler_name"]["choices"])
    assert combos["scheduler"]["choices"]


def test_title_pass_is_idempotent(object_info):
    wf, *_ = _prompt_chain(object_info)
    assert _title_nodes(wf, object_info) > 0
    assert _title_nodes(wf, object_info) == 0


def test_model_attribute_error_gets_pack_hint():
    error = execution_error({
        "node_type": "SomeSampler",
        "exception_type": "AttributeError",
        "exception_message": "'QwenImage' object has no attribute 'double_blocks'",
    })
    assert "SomeSampler" in error["hint"]
    missing = execution_error({
        "exception_type": "AttributeError",
        "exception_message": "'NoneType' object has no attribute 'shape'",
    })
    assert "hint" not in missing
