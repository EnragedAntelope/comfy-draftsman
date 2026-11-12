"""0.25.8: pipeline staging - a pass-through display node counts as a consumer,
and an unwired number picker is an input."""

from __future__ import annotations

from comfy_draftsman.graph import annotate as an
from test_round29_feedback import _link, _node, _wf

OI = {
    "LLM": {"input": {"required": {}}, "output": ["STRING"], "category": "pack"},
    "Regex": {"input": {"required": {"string": ["STRING", {}]}}, "output": ["STRING"], "category": "text"},
    "PreviewAny": {"input": {"required": {"source": ["*", {}]}}, "output": ["STRING"], "output_node": True},
    "Meta": {"input": {"required": {"value1": ["STRING", {}]}}, "output": ["EXTRA_METADATA"], "category": "pack"},
    "Save": {"input": {"required": {"extra_metadata": ["EXTRA_METADATA", {}]}}, "output": [], "output_node": True},
    "Picker": {"input": {"required": {}}, "output": ["INT", "INT"], "category": "utilities"},
    "Enc": {
        "input": {"required": {"prompt": ["STRING", {}], "width": ["INT", {}]}},
        "output": ["CONDITIONING"],
        "category": "conditioning",
    },
}


def _out(type_, *links):
    return [{"name": type_, "type": type_, "links": list(links)}]


def _in(name, type_, link):
    return {"name": name, "type": type_, "link": link}


def test_prompt_chain_through_a_preview_stays_in_prompt_building():
    # LLM -> Regex -> PreviewAny -> Enc.prompt, and Regex -> Meta -> Save:
    # the metadata tap must not drag the prompt chain into Output
    wf = _wf(
        _node(1, "LLM", [], outputs=_out("STRING", 1)),
        _node(2, "Regex", [], inputs=[_in("string", "STRING", 1)], outputs=_out("STRING", 2, 3)),
        _node(3, "PreviewAny", [], inputs=[_in("source", "STRING", 2)], outputs=_out("STRING", 4)),
        _node(4, "Meta", [], inputs=[_in("value1", "STRING", 3)], outputs=_out("EXTRA_METADATA", 5)),
        _node(5, "Save", [], inputs=[_in("extra_metadata", "EXTRA_METADATA", 5)]),
        _node(6, "Picker", ["2:3"], outputs=_out("INT", 6)),
        _node(7, "Enc", [], inputs=[_in("prompt", "STRING", 4), _in("width", "INT", 6)]),
        links=[
            _link(1, 1, 2, "STRING"), _link(2, 2, 3, "STRING"), _link(3, 2, 4, "STRING"),
            _link(4, 3, 7, "STRING"), _link(5, 4, 5, "EXTRA_METADATA"), _link(6, 6, 7, "INT"),
        ],
    )
    stage_of_key, undecided = {}, set()
    for node in wf.nodes.values():
        key = an.classify(node, OI, default="")
        if not key:
            undecided.add(node.id)
        stage_of_key[node.id] = key or "sampling"
    an._restage_by_graph(wf, OI, stage_of_key, undecided)
    assert {n: stage_of_key[n] for n in (1, 2, 3)} == dict.fromkeys((1, 2, 3), "prompt_build")
    assert stage_of_key[4] == "output"
    assert stage_of_key[6] == "inputs"
