"""0.25.4 session feedback.

- add_node sets a LoRA Manager loader's text at creation, and gets LM's JS sockets
- a new group never reuses an imported group's id
- a scoped choices_filter on a connection says it matched nothing
- inspect_workflow returns full note text
- roll_seeds names every value it rolled (a combo on "increment" too)
- a sweep names nodes that re-ran with unchanged inputs
"""

import copy

import pytest

import test_round45_fixes as r45
from comfy_draftsman import server
from comfy_draftsman.comfy.catalog import node_summary
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

object_info = r45.object_info  # shared pytest fixture

LM_LOADER = {
    "input": {
        "required": {
            "model": ["MODEL"],
            "text": ["AUTOCOMPLETE_TEXT_LORAS", {"placeholder": "Search LoRAs"}],
            "loras": ["LORAS", {}],
        },
        "optional": {},
    },
    "output": ["MODEL", "CLIP", "STRING", "STRING"],
    "output_name": ["MODEL", "CLIP", "trigger_words", "loaded_loras"],
    "python_module": "custom_nodes.ComfyUI-Lora-Manager",
}


@pytest.fixture
def lm(tmp_path, config, monkeypatch, object_info):
    info = {**copy.deepcopy(object_info), "Lora Loader (LoraManager)": LM_LOADER}
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", r45.FakeClient(info))
    monkeypatch.setattr(server._State, "session", session)
    wf = Workflow.new()
    return wf, session.create(wf, title="t")


async def test_add_lm_loader_with_text_and_wire_clip(lm):
    wf, wf_id = lm
    result = await server.edit_workflow(
        wf_id,
        [
            {"op": "add_node", "class_type": "CheckpointLoaderSimple", "ref": "ck"},
            {
                "op": "add_node",
                "class_type": "Lora Loader (LoraManager)",
                "ref": "lm",
                "widgets": {"text": "<lora:a.safetensors:0.8>"},
            },
            {"op": "connect", "from_node": "ck", "from_output": "CLIP", "to_node": "lm", "to_input": "clip"},
        ],
    )
    assert "error" not in result, result
    assert not any("frontend-created" in a for a in result["applied"])
    node = next(n for n in wf.nodes.values() if n.type == "Lora Loader (LoraManager)")
    assert [s.name for s in node.inputs][-2:] == ["clip", "lora_stack"]
    assert node.widgets_values[1] == "<lora:a.safetensors:0.8>"
    assert node.widgets_values[2][0]["name"] == "a.safetensors"  # loras synced


def test_new_group_id_skips_imported_ids(object_info):
    wf = Workflow.from_ui({
        "nodes": [{"id": 1, "type": "Note", "pos": [0, 0], "size": [100, 100], "widgets_values": [""]}],
        "links": [],
        "groups": [{"id": 8, "title": "Identity Forge", "bounding": [0, 0, 10, 10]}],
    })
    assert wf.group_from_nodes("New", [1]).id == 9


def test_filter_on_a_connection_says_so(object_info):
    summary = node_summary(object_info, "KSampler", choices_filter="model:face")
    assert "not a dropdown" in summary["choices_filter_note"]
    plain = node_summary(object_info, "KSampler", choices_filter="sampler_name:euler")
    assert "choices_filter_note" not in plain


async def test_inspect_returns_full_note_text(lm):
    wf, wf_id = lm
    text = "# Test log\n" + "line of notes\n" * 40
    note = wf.add_node("MarkdownNote")
    note.widgets_values = [text]
    out = await server.inspect_workflow(wf_id)
    assert out["notes"][str(note.id)] == text


def test_roll_reports_each_rolled_value(object_info):
    wf = Workflow()
    node = wf.add_node("KSampler", object_info=object_info)
    node.widgets_values[1] = "increment"
    changes: list[str] = []
    assert wf.apply_seed_control(object_info, changes)
    assert changes and changes[0].startswith(f"#{node.id} seed: ")


def test_always_rerun_flags_only_uncached_identical_nodes():
    llm = {"class_type": "LLM", "inputs": {"prompt": "x"}}
    enc = {"class_type": "Enc", "inputs": {"text": ["1", 0]}}
    run = {"1": llm, "2": enc}
    history = [(run, set()), (copy.deepcopy(run), set())]
    assert server._always_rerun(history) == ["#1 LLM"]  # enc's feed re-ran, so not flagged
    assert server._always_rerun([(run, set()), (run, {"1", "2"})]) == []
