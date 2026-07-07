"""Round 4/5 testing-feedback fixes.

- organize must never clobber human-authored node titles with generic
  "Positive Prompt" titles, and only role-titles actual prompt sources
- edit_workflow add_node must be atomic: a failing op leaves no stub node
- Note/MarkdownNote are editable annotation nodes (single 'text' widget)
- imported MarkdownNote nodes must be visible in workflow summaries
- malformed ops fail with the op schema spelled out, not a raw KeyError
- connect reports when it replaced an existing link into the target input
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph.annotate import _title_nodes
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"

DPRANDOM_SCHEMA = {
    "input": {
        "required": {
            "text": ["STRING", {"multiline": True, "default": ""}],
            "seed": ["INT", {"default": 0}],
        }
    },
    "output": ["STRING"],
    "category": "custom/text",
    "display_name": "Random Prompts",
}
CONCAT_SCHEMA = {
    "input": {
        "required": {
            "text_a": ["STRING", {"default": ""}],
            "text_b": ["STRING", {"default": ""}],
        }
    },
    "output": ["STRING"],
    "category": "custom/text",
    "display_name": "Concatenate",
}


@pytest.fixture(scope="module")
def object_info():
    info = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    info["DPRandomGenerator"] = DPRANDOM_SCHEMA
    info["TextConcatenate"] = CONCAT_SCHEMA
    return info


class FakeClient:
    def __init__(self, object_info):
        self._object_info = object_info

    async def get_object_info(self, refresh: bool = False):
        return self._object_info


@pytest.fixture
def wired(tmp_path, config, monkeypatch, object_info):
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", FakeClient(object_info))
    monkeypatch.setattr(server._State, "session", session)
    wf = Workflow.new()
    wf_id = session.create(wf, title="t")
    return wf, wf_id


# --- organize title preservation (round 4/5: "✅ Positive Prompt" clobbering) ---


def _prompt_chain(object_info):
    """dp1 (custom title) + dp2 -> concat -> encoder.text -> sampler.positive"""
    wf = Workflow.new()
    dp1 = wf.add_node("DPRandomGenerator", object_info=object_info, title="🍔 Cuisine Bank")
    dp2 = wf.add_node("DPRandomGenerator", object_info=object_info)
    concat = wf.add_node("TextConcatenate", object_info=object_info)
    enc = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(dp1.id, 0, concat.id, "text_a", object_info)
    wf.connect(dp2.id, 0, concat.id, "text_b", object_info)
    wf.connect(concat.id, 0, enc.id, "text", object_info)
    wf.connect(enc.id, "CONDITIONING", sampler.id, "positive")
    return wf, dp1, dp2, concat, enc


def test_custom_titles_upstream_of_positive_prompt_survive(object_info):
    wf, dp1, dp2, _concat, enc = _prompt_chain(object_info)
    _title_nodes(wf, object_info)
    assert dp1.title == "🍔 Cuisine Bank"  # human title, never clobbered
    assert dp2.title is None  # distant fragment, not a prompt source
    assert enc.title == "✅ Positive Prompt"  # the actual encoder gets the role


def test_draftsman_generic_title_is_still_refreshable(object_info):
    wf, _dp1, _dp2, _concat, enc = _prompt_chain(object_info)
    enc.title = "✅ Positive Prompt"  # from a previous organize pass
    _title_nodes(wf, object_info)
    assert enc.title == "✅ Positive Prompt"


def test_custom_title_on_direct_prompt_source_survives(object_info):
    wf = Workflow.new()
    dp = wf.add_node("DPRandomGenerator", object_info=object_info, title="My Wildcards")
    enc = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(dp.id, 0, enc.id, "text", object_info)
    wf.connect(enc.id, "CONDITIONING", sampler.id, "positive")
    _title_nodes(wf, object_info)
    assert dp.title == "My Wildcards"


# --- edit_workflow: atomic add_node + note nodes ---


async def test_add_unknown_class_leaves_graph_unchanged(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id, [{"op": "add_node", "class_type": "NoSuchNode"}]
    )
    assert "unknown node class" in result["error"]
    assert result["applied"] == []
    assert wf.nodes == {}


async def test_add_node_with_bad_widget_name_leaves_graph_unchanged(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [{"op": "add_node", "class_type": "CLIPTextEncode", "widgets": {"nope": "x"}}],
    )
    assert "no widget" in result["error"]
    assert wf.nodes == {}


async def test_markdown_note_add_and_edit(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [
            {
                "op": "add_node",
                "class_type": "MarkdownNote",
                "title": "Read me",
                "widgets": {"text": "# How this works"},
            }
        ],
        summary=True,
    )
    assert "error" not in result, result
    (node_id,) = wf.nodes
    assert wf.nodes[node_id].widgets_values == ["# How this works"]
    summary_nodes = result["summary"]["nodes"]
    assert summary_nodes[0]["class_type"] == "MarkdownNote"
    assert summary_nodes[0]["virtual"] is True

    edited = await server.edit_workflow(
        wf_id, [{"op": "set_widget", "node_id": node_id, "input": "text", "value": "updated"}]
    )
    assert "error" not in edited, edited
    assert wf.nodes[node_id].widgets_values == ["updated"]


async def test_markdown_note_never_reaches_api(wired, object_info):
    wf, wf_id = wired
    await server.edit_workflow(
        wf_id,
        [{"op": "add_node", "class_type": "MarkdownNote", "widgets": {"text": "note"}}],
    )
    assert wf.to_api(object_info) == {}


# --- edit_workflow: op schema errors ---


async def test_missing_required_key_names_the_schema(wired):
    _wf, wf_id = wired
    result = await server.edit_workflow(wf_id, [{"op": "connect", "from_node": 1}])
    assert "connect" in result["error"]
    assert "to_input" in result["error"]
    assert "missing required" in result["error"]


async def test_unknown_key_is_rejected_with_suggestion(wired):
    _wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [{"op": "add_node", "class_type": "CLIPTextEncode", "widgets_values": ["x"]}],
    )
    assert "widgets_values" in result["error"]
    assert "did you mean 'widgets'" in result["error"]


async def test_unknown_op_lists_valid_ops(wired):
    _wf, wf_id = wired
    result = await server.edit_workflow(wf_id, [{"op": "retitle", "node_id": 1}])
    assert "unknown op" in result["error"]
    assert "set_title" in result["error"]


async def test_unknown_node_id_is_a_clean_error(wired):
    _wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id, [{"op": "set_title", "node_id": 99, "title": "x"}]
    )
    assert "unknown node id" in result["error"]


# --- edit_workflow: connect replacement reporting ---


async def test_connect_reports_replaced_link(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [
            {"op": "add_node", "class_type": "CheckpointLoaderSimple"},
            {"op": "add_node", "class_type": "CheckpointLoaderSimple"},
            {"op": "add_node", "class_type": "KSampler"},
            {"op": "connect", "from_node": 1, "from_output": "MODEL", "to_node": 3, "to_input": "model"},
            {"op": "connect", "from_node": 2, "from_output": "MODEL", "to_node": 3, "to_input": "model"},
        ],
    )
    assert "error" not in result, result
    assert "replaced existing link from #1[0]" in result["applied"][-1]
    assert "replaced" not in result["applied"][-2]
    assert len(wf.links) == 1


# --- import keeps notes visible ---


async def test_imported_markdown_notes_appear_in_summary(wired):
    _wf, _wf_id = wired
    long_text = "words " * 60
    ui = {
        "nodes": [
            {
                "id": 1,
                "type": "MarkdownNote",
                "pos": [0, 0],
                "size": [380, 180],
                "title": "Author note",
                "widgets_values": [long_text],
            },
            {"id": 2, "type": "KSampler", "pos": [400, 0], "size": [270, 262]},
        ],
        "links": [],
    }
    result = await server.import_workflow(json.dumps(ui), title="with-notes")
    notes = [n for n in result["nodes"] if n["class_type"] == "MarkdownNote"]
    assert len(notes) == 1
    assert notes[0]["virtual"] is True
    assert notes[0]["title"] == "Author note"
    preview = notes[0]["widgets"][0]
    assert preview.endswith("…") and len(preview) <= 121  # truncated, content intact
