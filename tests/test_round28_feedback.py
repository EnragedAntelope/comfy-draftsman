"""0.19: edit_workflow refs + connect-only deltas, write-through persistence, and
sampler/provenance folding in model guidance (all from a live identity-A/B session)."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.config import Config
from comfy_draftsman.knowledge import get_guidance, save_learning
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def oi():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture
def env(monkeypatch, tmp_path, oi):
    class StubClient:
        async def get_object_info(self, refresh=False):
            return oi

    config = Config(
        comfyui_url="http://comfy.test", session_dir=tmp_path / "sessions", learned_dir=tmp_path / "learned"
    )
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", StubClient())
    monkeypatch.setattr(server._State, "session", Session(config.session_dir))
    return config


async def test_edits_survive_a_server_restart(env):
    made = await server.create_workflow("keep me")
    wf_id = made["workflow_id"]
    await server.edit_workflow(wf_id, [{"op": "add_node", "class_type": "KSampler"}])
    # a resumed conversation is a NEW process: fresh Session over the same dir
    fresh = Session(env.session_dir)
    assert len(fresh.get(wf_id).nodes) == 1
    assert fresh.title(wf_id) == "keep me"


async def test_ops_that_landed_before_a_failure_are_persisted_too(env):
    wf_id = (await server.create_workflow("t"))["workflow_id"]
    result = await server.edit_workflow(
        wf_id, [{"op": "add_node", "class_type": "KSampler"}, {"op": "set_title", "node_id": 99, "title": "x"}]
    )
    assert "error" in result
    assert len(Session(env.session_dir).get(wf_id).nodes) == 1


async def test_connect_only_batch_does_not_echo_the_endpoints(env):
    wf_id = (await server.create_workflow("t"))["workflow_id"]
    await server.edit_workflow(
        wf_id,
        [{"op": "add_node", "class_type": "CheckpointLoaderSimple"}, {"op": "add_node", "class_type": "KSampler"}],
    )
    result = await server.edit_workflow(
        wf_id, [{"op": "connect", "from_node": 1, "from_output": "MODEL", "to_node": 2, "to_input": "model"}]
    )
    assert result["applied"] and result["changed"] == []


async def test_add_node_ref_can_be_used_by_later_ops_in_the_same_batch(env):
    wf_id = (await server.create_workflow("t"))["workflow_id"]
    result = await server.edit_workflow(
        wf_id,
        [
            {"op": "add_node", "class_type": "CheckpointLoaderSimple", "ref": "ckpt"},
            {"op": "add_node", "class_type": "KSampler", "ref": "ks"},
            {"op": "connect", "from_node": "ckpt", "from_output": "MODEL", "to_node": "ks", "to_input": "model"},
            {"op": "set_widget", "node_id": "ks", "input": "steps", "value": 7},
        ],
    )
    assert "error" not in result, result
    assert "(ref ckpt)" in result["applied"][0]
    wf = Session(env.session_dir).get(wf_id)
    assert len(wf.links) == 1


async def test_unknown_or_duplicate_ref_is_a_clear_error(env):
    wf_id = (await server.create_workflow("t"))["workflow_id"]
    bad = await server.edit_workflow(wf_id, [{"op": "set_widget", "node_id": "nope", "input": "steps", "value": 1}])
    assert "unknown ref 'nope'" in bad["error"]
    dup = await server.edit_workflow(
        wf_id,
        [
            {"op": "add_node", "class_type": "KSampler", "ref": "a"},
            {"op": "add_node", "class_type": "KSampler", "ref": "a"},
        ],
    )
    assert "must be a non-numeric string" in dup["error"]
    assert len(dup["applied"]) == 1  # the first add landed, the batch stopped


def test_learned_singular_sampler_folds_into_the_plural_list(tmp_path):
    save_learning(tmp_path, "krea2", {"sampling": {"sampler": "er_sde", "scheduler": "bong_tangent"}}, source="t")
    sampling = get_guidance("krea2", learned_dir=tmp_path)["sampling"]
    assert sampling["samplers"][0] == "er_sde" and "euler" in sampling["samplers"]
    assert sampling["schedulers"][0] == "bong_tangent"
    assert "sampler" not in sampling and "scheduler" not in sampling


async def test_guidance_reports_only_the_latest_learned_source(env):
    for i in range(3):
        save_learning(env.learned_dir, "flux", {"notes": {"sampling": f"n{i}"}}, source=f"src{i} " + "x" * 400)
    out = await server.get_model_guidance(family="flux")
    assert len(out["learned_sources"]) == 1
    assert out["learned_sources"][0]["source"].startswith("src2")
    assert len(out["learned_sources"][0]["source"]) <= 200
    assert out["learned_sources_total"] == 3


def test_a_variants_own_sampler_list_beats_a_learned_family_wide_pick(tmp_path):
    from comfy_draftsman import knowledge

    floor = knowledge._load_floor()
    name, variant = next(
        (f, v) for f, d in floor.items() for v in (d.get("variants") or {}).values() if "samplers" in (v.get("sampling") or {})
    )
    save_learning(tmp_path, name, {"sampling": {"sampler": "er_sde"}}, source="t")
    patterns = variant["patterns"][0]
    sampling = get_guidance(name, patterns, learned_dir=tmp_path)["sampling"]
    assert sampling["samplers"] == variant["sampling"]["samplers"]
