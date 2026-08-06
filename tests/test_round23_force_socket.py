"""Round 23 / Part C: force:true can wire a frontend-only input.

Some packs (rgthree's "Any Switch") build their inputs in frontend JS, so
/object_info declares none - connect refused unconditionally, even with
force:true, because the "no input" raise fired BEFORE the force check ever
ran. A live session hit this trying to wire rgthree's Any Switch, the
standard idiom a real community workflow (minimaxH3Ref2vaAdvanced_v10) uses
for auto/manual source switching."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"

# A stub shaped like rgthree's "Any Switch": its inputs are drawn by frontend
# JS, so /object_info declares none at all.
ANY_SWITCH_SCHEMA = {
    "input": {"required": {}},
    "output": ["*"],
    "output_name": ["any"],
    "category": "rgthree",
    "display_name": "Any Switch (rgthree)",
}


@pytest.fixture(scope="module")
def object_info():
    info = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    info["Any Switch (rgthree)"] = ANY_SWITCH_SCHEMA
    return info


def _wf_with_switch(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    switch = wf.add_node("Any Switch (rgthree)", object_info=object_info)
    return wf, ckpt, switch


# --- graph-level: Workflow.connect ---


def test_connect_without_force_still_refuses(object_info):
    wf, ckpt, switch = _wf_with_switch(object_info)
    with pytest.raises(ValueError, match="has no input"):
        wf.connect(ckpt.id, "MODEL", switch.id, "any_01", object_info)


def test_connect_error_mentions_force_escape_hatch(object_info):
    wf, ckpt, switch = _wf_with_switch(object_info)
    with pytest.raises(ValueError, match="force"):
        wf.connect(ckpt.id, "MODEL", switch.id, "any_01", object_info)


def test_connect_with_force_creates_the_socket(object_info):
    wf, ckpt, switch = _wf_with_switch(object_info)
    link = wf.connect(ckpt.id, "MODEL", switch.id, "any_01", object_info, force=True)
    slot = switch.input_by_name("any_01")
    assert slot is not None
    assert slot.type == "MODEL"  # adopts the origin's own output type
    assert slot.link == link.id


def test_force_created_socket_survives_to_api(object_info):
    wf, ckpt, switch = _wf_with_switch(object_info)
    wf.connect(ckpt.id, "MODEL", switch.id, "any_01", object_info, force=True)
    api = wf.to_api(object_info)
    assert str(switch.id) in api
    assert api[str(switch.id)]["inputs"]["any_01"] == [str(ckpt.id), 0]


def test_force_created_socket_round_trips_through_ui(object_info):
    wf, ckpt, switch = _wf_with_switch(object_info)
    wf.connect(ckpt.id, "MODEL", switch.id, "any_01", object_info, force=True)
    ui = wf.to_ui()
    reloaded = Workflow.from_ui(ui)
    slot = reloaded.nodes[switch.id].input_by_name("any_01")
    assert slot is not None
    assert slot.link is not None


# --- MCP-level: edit_workflow connect op ---


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
    wf, ckpt, switch = _wf_with_switch(object_info)
    wf_id = session.create(wf, title="t")
    return wf, wf_id, ckpt.id, switch.id


async def test_edit_workflow_connect_without_force_errors(wired):
    _wf, wf_id, ckpt_id, switch_id = wired
    result = await server.edit_workflow(
        wf_id,
        [
            {
                "op": "connect",
                "from_node": ckpt_id,
                "from_output": "MODEL",
                "to_node": switch_id,
                "to_input": "any_01",
            }
        ],
    )
    assert "error" in result


async def test_edit_workflow_connect_with_force_creates_socket_and_notes_it(wired):
    workflow, wf_id, ckpt_id, switch_id = wired
    result = await server.edit_workflow(
        wf_id,
        [
            {
                "op": "connect",
                "from_node": ckpt_id,
                "from_output": "MODEL",
                "to_node": switch_id,
                "to_input": "any_01",
                "force": True,
            }
        ],
    )
    assert "error" not in result
    joined = " ".join(result["applied"])
    assert "created undeclared input" in joined
    assert workflow.nodes[switch_id].input_by_name("any_01") is not None


async def test_edit_workflow_force_connect_into_declared_widget_is_not_flagged(
    tmp_path, config, monkeypatch, object_info
):
    """force:true on a perfectly ordinary widget-to-socket conversion must NOT
    get the "undeclared input" note - that note is only for genuinely
    frontend-only slots."""
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", _FakeClient(object_info))
    monkeypatch.setattr(server._State, "session", session)
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    prim = wf.add_node("PrimitiveNode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model", object_info)
    wf_id = session.create(wf, title="t")
    result = await server.edit_workflow(
        wf_id,
        [
            {
                "op": "connect",
                "from_node": prim.id,
                "from_output": 0,
                "to_node": sampler.id,
                "to_input": "steps",
                "force": True,
            }
        ],
    )
    assert "error" not in result
    joined = " ".join(result["applied"])
    assert "created undeclared input" not in joined
