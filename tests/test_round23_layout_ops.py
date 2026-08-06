"""Round 23 / Part B: layout and group edit ops - the escape hatch for when
organize_workflow's automatic layout isn't what the caller wants. Before this,
fixing layout meant export_workflow_json -> hand-edit in Python -> re-import,
which was the single largest token cost in a reported live session."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


class _FakeClient:
    def __init__(self, object_info):
        self._object_info = object_info

    async def get_object_info(self, refresh: bool = False):
        return self._object_info


@pytest.fixture
def wired(tmp_path, config, monkeypatch, object_info):
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", _FakeClient(object_info))
    monkeypatch.setattr(server._State, "session", session)
    wf = Workflow.new()
    a = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    b = wf.add_node("CLIPTextEncode", object_info=object_info)
    c = wf.add_node("KSampler", object_info=object_info)
    a.pos, b.pos, c.pos = [0.0, 0.0], [300.0, 0.0], [600.0, 0.0]
    wf_id = session.create(wf, title="t")
    return wf, wf_id, {"a": a.id, "b": b.id, "c": c.id}


@pytest.fixture
def wf(object_info):
    wf = Workflow.new()
    a = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    b = wf.add_node("CLIPTextEncode", object_info=object_info)
    c = wf.add_node("KSampler", object_info=object_info)
    a.pos = [0.0, 0.0]
    b.pos = [300.0, 0.0]
    c.pos = [600.0, 0.0]
    return wf, {"a": a.id, "b": b.id, "c": c.id}


# --- set_pos ---


def test_set_pos_moves_node(wf):
    workflow, ids = wf
    workflow.nodes[ids["a"]].pos = [10.0, 20.0]
    assert workflow.nodes[ids["a"]].pos == [10.0, 20.0]


def test_group_from_nodes_bounding_encloses_members(wf):
    workflow, ids = wf
    group = workflow.group_from_nodes("Test Group", [ids["a"], ids["b"]], color="#123456")
    x, y, w, h = group.bounding
    for nid in (ids["a"], ids["b"]):
        node = workflow.nodes[nid]
        assert x <= node.pos[0]
        assert y <= node.pos[1]
        assert node.pos[0] + node.size[0] <= x + w
        assert node.pos[1] + node.size[1] <= y + h
    assert group.color == "#123456"
    assert group.id == 1


def test_group_from_nodes_empty_raises(wf):
    workflow, _ids = wf
    with pytest.raises(ValueError):
        workflow.group_from_nodes("Empty", [])


def test_group_bounding_for_matches_group_from_nodes(wf):
    workflow, ids = wf
    bounding = workflow.group_bounding_for([ids["a"], ids["b"], ids["c"]])
    group = workflow.group_from_nodes("Full", [ids["a"], ids["b"], ids["c"]])
    assert bounding == group.bounding


def test_group_round_trips_through_to_ui_from_ui(wf, object_info):
    workflow, ids = wf
    workflow.group_from_nodes("Round Trip Group", [ids["a"], ids["b"]], color="#abcdef")
    ui = workflow.to_ui()
    reloaded = Workflow.from_ui(ui)
    assert len(reloaded.groups) == 1
    assert reloaded.groups[0].title == "Round Trip Group"
    assert reloaded.groups[0].color == "#abcdef"


# --- edit_workflow op dispatch (through the actual MCP tool function) ---


async def test_edit_workflow_set_pos(wired):
    _workflow, wf_id, ids = wired
    result = await server.edit_workflow(
        wf_id, [{"op": "set_pos", "node_id": ids["a"], "pos": [111.0, 222.0]}]
    )
    assert "error" not in result
    changed = {c["id"]: c for c in result["changed"]}
    assert ids["a"] in changed


async def test_edit_workflow_set_pos_with_size(wired):
    workflow, wf_id, ids = wired
    await server.edit_workflow(
        wf_id, [{"op": "set_pos", "node_id": ids["a"], "pos": [5.0, 6.0], "size": [400.0, 80.0]}]
    )
    assert workflow.nodes[ids["a"]].pos == [5.0, 6.0]
    assert workflow.nodes[ids["a"]].size == [400.0, 80.0]


async def test_edit_workflow_set_pos_bad_shape_errors(wired):
    _workflow, wf_id, ids = wired
    result = await server.edit_workflow(
        wf_id, [{"op": "set_pos", "node_id": ids["a"], "pos": [1.0]}]
    )
    assert "error" in result


async def test_edit_workflow_set_pos_unknown_node_is_clean_error(wired):
    _workflow, wf_id, _ids = wired
    result = await server.edit_workflow(
        wf_id, [{"op": "set_pos", "node_id": 99999, "pos": [0.0, 0.0]}]
    )
    assert "unknown node id" in result["error"]


async def test_edit_workflow_add_group(wired):
    workflow, wf_id, ids = wired
    result = await server.edit_workflow(
        wf_id,
        [{"op": "add_group", "title": "My Group", "node_ids": [ids["a"], ids["b"]]}],
    )
    assert "error" not in result
    assert len(workflow.groups) == 1
    assert workflow.groups[0].title == "My Group"
    assert any("added group #1" in a for a in result["applied"])


async def test_edit_workflow_add_group_unknown_node_errors(wired):
    _workflow, wf_id, ids = wired
    result = await server.edit_workflow(
        wf_id,
        [{"op": "add_group", "title": "Bad", "node_ids": [ids["a"], 99999]}],
    )
    assert "error" in result


async def test_edit_workflow_set_group_updates_title_and_bounds(wired):
    workflow, wf_id, ids = wired
    await server.edit_workflow(
        wf_id, [{"op": "add_group", "title": "Original", "node_ids": [ids["a"]]}]
    )
    group_id = workflow.groups[0].id
    result = await server.edit_workflow(
        wf_id,
        [
            {
                "op": "set_group",
                "group_id": group_id,
                "title": "Renamed",
                "node_ids": [ids["a"], ids["b"], ids["c"]],
            }
        ],
    )
    assert "error" not in result
    assert workflow.groups[0].title == "Renamed"
    x, y, w, h = workflow.groups[0].bounding
    for nid in (ids["a"], ids["b"], ids["c"]):
        node = workflow.nodes[nid]
        assert x <= node.pos[0] and node.pos[0] + node.size[0] <= x + w
        assert y <= node.pos[1] and node.pos[1] + node.size[1] <= y + h


async def test_edit_workflow_set_group_unknown_id_errors(wired):
    _workflow, wf_id, _ids = wired
    result = await server.edit_workflow(wf_id, [{"op": "set_group", "group_id": 42, "title": "x"}])
    assert "error" in result
    assert "no group #42" in result["error"]


async def test_edit_workflow_remove_group(wired):
    workflow, wf_id, ids = wired
    await server.edit_workflow(
        wf_id, [{"op": "add_group", "title": "ToRemove", "node_ids": [ids["a"]]}]
    )
    group_id = workflow.groups[0].id
    result = await server.edit_workflow(wf_id, [{"op": "remove_group", "group_id": group_id}])
    assert "error" not in result
    assert workflow.groups == []


async def test_edit_workflow_remove_group_unknown_id_errors(wired):
    _workflow, wf_id, _ids = wired
    result = await server.edit_workflow(wf_id, [{"op": "remove_group", "group_id": 7}])
    assert "error" in result


async def test_edit_workflow_group_ops_are_addressable_via_summary(wired):
    """_summary formats groups as '#N title' - the id set_group/remove_group
    take - so a caller can address a group without hand-tracking ids."""
    _workflow, wf_id, ids = wired
    await server.edit_workflow(
        wf_id, [{"op": "add_group", "title": "Addressable", "node_ids": [ids["a"]]}]
    )
    summary = await server.inspect_workflow(wf_id)
    assert any(g.startswith("#1 ") for g in summary["groups"])
