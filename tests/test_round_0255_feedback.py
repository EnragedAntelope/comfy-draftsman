"""0.25.5: session feedback - sweep freeze/timing/rerun reasons, atomic edit
batches, copy_nodes, organize's Unconnected band/plumbing/panel layout, knob
tables that name their node, browser-safe seed rolls."""

from __future__ import annotations

import asyncio
import json
import random

import pytest

import test_round45_fixes as r45
from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph import knobs
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.annotate import annotate
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import _bypass_hint
from comfy_draftsman.session import Session

STRING_NODE = {
    "input": {"required": {"text": ["STRING", {"default": ""}]}},
    "output": ["STRING"],
    "category": "utils/text",
}


@pytest.fixture(scope="module")
def object_info():
    info = json.loads((r45.FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    info["FilenameBuilder"] = STRING_NODE
    return info


@pytest.fixture
def wired(tmp_path, config, monkeypatch, object_info):
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", r45.FakeClient(object_info))
    monkeypatch.setattr(server._State, "session", session)
    return session


def _txt2img(oi):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    pos = wf.add_node("CLIPTextEncode", object_info=oi)
    latent = wf.add_node("EmptyLatentImage", object_info=oi)
    sampler = wf.add_node("KSampler", object_info=oi)
    decode = wf.add_node("VAEDecode", object_info=oi)
    save = wf.add_node("SaveImage", object_info=oi)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model", oi)
    wf.connect(ckpt.id, "CLIP", pos.id, "clip", oi)
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive", oi)
    wf.connect(latent.id, "LATENT", sampler.id, "latent_image", oi)
    wf.connect(sampler.id, "LATENT", decode.id, "samples", oi)
    wf.connect(ckpt.id, "VAE", decode.id, "vae", oi)
    wf.connect(decode.id, "IMAGE", save.id, "images", oi)
    return wf, {"ckpt": ckpt.id, "pos": pos.id, "sampler": sampler.id, "decode": decode.id, "save": save.id}


# --- edit_workflow: all or nothing, copy_nodes ---


async def test_copy_nodes_keeps_widgets_and_internal_links(wired, object_info):
    src, ids = _txt2img(object_info)
    src.set_widget(ids["sampler"], "steps", 33, object_info)
    src_id = wired.create(src, title="krea")
    dst_id = wired.create(Workflow.new(), title="target")
    result = await server.edit_workflow(dst_id, [
        {"op": "copy_nodes", "from_workflow": src_id, "node_ids": [ids["sampler"], ids["decode"]], "ref": "k"},
        {"op": "set_title", "node_id": f"k{ids['sampler']}", "title": "Face pass"},
    ])
    assert "error" not in result, result
    dst = server._wf(dst_id)
    sampler = next(n for n in dst.nodes.values() if n.type == "KSampler")
    assert sampler.title == "Face pass"
    assert dst.get_widget(sampler.id, "steps", object_info) == 33
    assert len(dst.links) == 1  # sampler -> decode came along
    assert "now unwired" in result["applied"][0] and ".model" in result["applied"][0]
    assert len(src.nodes) == 6  # the source is untouched


async def test_copy_nodes_rejects_unknown_ids(wired, object_info):
    src_id = wired.create(Workflow.new(), title="s")
    dst_id = wired.create(Workflow.new(), title="d")
    result = await server.edit_workflow(dst_id, [{"op": "copy_nodes", "from_workflow": src_id, "node_ids": [7]}])
    assert "no node(s) [7]" in result["error"]


async def test_bad_widget_name_late_in_batch_leaves_no_half_applied_ref(wired, object_info):
    wf_id = wired.create(Workflow.new(), title="t")
    bad = await server.edit_workflow(wf_id, [
        {"op": "add_node", "class_type": "KSampler", "ref": "fu"},
        {"op": "set_widget", "node_id": "fu", "input": "backend", "value": 1},
    ])
    assert bad["applied"] == [] and not server._wf(wf_id).nodes
    retry = await server.edit_workflow(wf_id, [
        {"op": "add_node", "class_type": "KSampler", "ref": "fu"},
        {"op": "set_widget", "node_id": "fu", "input": "steps", "value": 9},
    ])
    assert "error" not in retry, retry
    assert len(server._wf(wf_id).nodes) == 1


# --- run timing ---


def test_sampling_s_counts_every_detailer_pass():
    api = {"7": {"class_type": "FaceDetailer"}, "8": {"class_type": "VAEDecode"}}
    # two faces, 4 steps each at 1s/step; the counter restarts per face
    ticks = {"7": [(0.0, 1, 4), (3.0, 4, 4), (10.0, 1, 4), (13.0, 4, 4)], "8": [(20.0, 1, 2), (21.0, 2, 2)]}
    assert ComfyClient._sampling_s(ticks, api) == 8.0


def test_node_seconds_ranks_slowest_nodes():
    api = {"1": {"class_type": "UNETLoader"}, "2": {"class_type": "KSampler"}, "3": {"class_type": "SaveImage"}}
    starts = [(0.0, "1"), (4.0, "2"), (14.0, "3")]
    assert ComfyClient._node_seconds(starts, 14.05, api) == {"#2 KSampler": 10.0, "#1 UNETLoader": 4.0}


# --- sweep freeze ---


class FreezeClient(r45.FakeClient):
    def __init__(self, oi):
        super().__init__(oi)
        self.prompts = []

    async def run_and_wait(self, api, **kwargs):
        self.prompts.append(api)
        return {
            "status": "success", "prompt_id": "p",
            "outputs": [{"filename": "f.png", "subfolder": "", "type": "temp", "kind": "images",
                         "node_id": "freeze_preview"}],
        }


async def test_freeze_runs_upstream_once_and_rewires_cells(object_info, monkeypatch):
    wf, ids = _txt2img(object_info)
    client = FreezeClient(object_info)
    monkeypatch.setattr(server._State, "client", client)
    cell = wf.to_api(object_info)
    prepared = [("a", cell, None), ("bad", None, "invalid")]
    note = await server._freeze_stage(wf, ids["decode"], object_info, prepared, 60, None, False)
    probe = client.prompts[0]
    assert str(ids["save"]) not in probe and probe["freeze_preview"]["inputs"]["images"] == [str(ids["decode"]), 0]
    assert cell[str(ids["save"])]["inputs"]["images"] == ["freeze_load", 0]
    assert cell["freeze_load"]["inputs"]["image"] == "f.png [temp]"
    assert "held fixed" in note


async def test_freeze_needs_an_image_node(object_info):
    wf, ids = _txt2img(object_info)
    with pytest.raises(ValueError, match="IMAGE output"):
        await server._freeze_stage(wf, ids["sampler"], object_info, [], 60, None, False)
    assert "freeze must be" in server._sweep_spec({"freeze": "3"})


async def test_freeze_timeout_names_the_prompt(object_info, monkeypatch):
    wf, ids = _txt2img(object_info)
    client = FreezeClient(object_info)
    client.run_and_wait = lambda api, **kw: asyncio.sleep(0, {"status": "timeout", "prompt_id": "p9"})
    monkeypatch.setattr(server._State, "client", client)
    with pytest.raises(ValueError, match=r"prompt p9.*front-queued"):
        await server._freeze_stage(wf, ids["decode"], object_info, [], 5, None, True)


# --- validate: bypass hint ---


def test_bypass_hint_only_when_every_consumer_input_is_optional(object_info):
    oi = {**object_info, "Pool": {"input": {"required": {}}, "output": ["POOL"]},
          "Loader": {"input": {"required": {"model": ["MODEL"]}, "optional": {"pool": ["POOL"]}}, "output": ["MODEL"]}}
    wf = Workflow.new()
    pool = wf.add_node("Pool", object_info=oi)
    loader = wf.add_node("Loader", object_info=oi)
    wf.connect(pool.id, 0, loader.id, "pool", oi)
    assert "set_mode 4" in _bypass_hint(wf, pool, oi)
    oi["Loader"]["input"] = {"required": {"model": ["MODEL"], "pool": ["POOL"]}}
    assert _bypass_hint(wf, pool, oi) == ""


# --- seeds ---


def test_randomized_seed_stays_browser_safe(object_info):
    wf = Workflow.new()
    node = wf.add_node("KSampler", object_info=object_info)
    node.widgets_values[1] = "randomize"
    rng = random.Random(0)
    for _ in range(50):
        values, _ = w.roll_seed_controls("KSampler", node.widgets_values, object_info, rng=rng)
        assert 0 <= values[0] <= w.FRONTEND_SEED_MAX


# --- organize ---


def test_dead_branch_goes_to_unconnected_band_unpainted(object_info):
    wf, _ = _txt2img(object_info)
    stray = wf.add_node("LoadImage", object_info=object_info)
    report = annotate(wf, object_info)
    assert report["stages"]["unused"] == 1
    group = next(g for g in wf.groups if g.title == "Unconnected")
    x, _, gw, _ = group.bounding
    assert x <= stray.pos[0] <= x + gw
    assert (stray.color, stray.bgcolor) != ("#232", "#353")


def test_filename_string_sits_with_the_save_node(object_info):
    wf, ids = _txt2img(object_info)
    name = wf.add_node("FilenameBuilder", object_info=object_info)
    wf.connect(name.id, 0, ids["save"], "filename_prefix", object_info)
    report = annotate(wf, object_info)
    assert "prompt_build" not in report["stages"]
    output = next(g for g in wf.groups if g.title == "Output")
    assert output.bounding[0] <= name.pos[0] <= output.bounding[0] + output.bounding[2]


def test_panel_layout_groups_controls_and_pipeline(object_info):
    wf, _ = _txt2img(object_info)
    report = annotate(wf, object_info, layout="panel")
    assert set(report["stages"]) == {"controls", "plumbing"}
    controls = next(n for n in wf.nodes.values() if n.type == "MarkdownNote" and n.title == "Controls")
    assert "| steps (KSampler #" in controls.widgets_values[0]
    titles = {g.title for g in wf.groups}
    assert titles == {"Controls", "Pipeline"}
    # re-organizing back is idempotent and drops the panel groups
    annotate(wf, object_info)
    assert "Controls" not in {g.title for g in wf.groups}


def test_knob_rows_name_the_node_and_skip_foreign_knobs(object_info):
    oi = {**object_info, "Identity": {"input": {"required": {"height": [["Random", "tall"]]}}, "output": ["STRING"]}}
    wf = Workflow.new()
    a = wf.add_node("KSampler", object_info=oi, title="Base")
    b = wf.add_node("KSampler", object_info=oi, title="Face")
    ident = wf.add_node("Identity", object_info=oi)
    rows = knobs.knob_rows([a, b, ident], oi)
    assert any(r.startswith("| seed (Base #") for r in rows)
    assert any(r.startswith("| seed (Face #") for r in rows)
    assert not any("Random" in r for r in rows)
    assert all("(Base" not in r for r in knobs.knob_rows([a], oi))


def test_parked_note_is_named(object_info):
    wf, _ = _txt2img(object_info)
    note = wf.add_node("MarkdownNote", title="My notes")
    note.widgets_values = ["remember the seed"]
    note.pos = [0.0, 0.0]
    note.size = [5000.0, 5000.0]  # covers whatever the layout puts there
    report = annotate(wf, object_info)
    assert report["applied"]["foreign_notes_parked"] == 1
    assert report["parked_notes"][0].startswith(f"#{note.id} 'My notes' -> column left")


def test_big_band_widens_instead_of_one_tall_strip(object_info):
    wf = Workflow.new()
    for _ in range(28):
        node = wf.add_node("FilenameBuilder", object_info=object_info)
        node.widgets_values = ["line\n" * 30]  # tall multiline text boxes
        enc = wf.add_node("CLIPTextEncode", object_info=object_info)
        wf.connect(node.id, 0, enc.id, "text", object_info)
    annotate(wf, object_info)
    group = next(g for g in wf.groups if g.title == "Prompt Building")
    assert group.bounding[3] < 1.5 * group.bounding[2], group.bounding  # was 905x3645
