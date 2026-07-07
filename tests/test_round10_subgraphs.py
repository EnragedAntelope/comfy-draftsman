"""Round-10: subgraph -> API flattening, write-time widget value validation,
compact edit_workflow results, and the list_models metadata digest.

fixtures/subgraph_real_template.json is ComfyUI's bundled
01_get_started_text_to_image template (schema-1.0, subgraph-packaged) -
captured verbatim so the flattener is tested against real boundary/proxyWidget
structure, not just hand-built minimal docs.
"""

import json
from pathlib import Path

import pytest

from comfy_draftsman import server
from comfy_draftsman.comfy.catalog import metadata_digest
from comfy_draftsman.graph.model import MODE_MUTE, Workflow
from comfy_draftsman.graph.subgraph import flatten, has_subgraph_instances
from comfy_draftsman.graph.validate import check_widget_value, validate
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def oi():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture
def real_template():
    return json.loads(
        (FIXTURES / "subgraph_real_template.json").read_text(encoding="utf-8")
    )


# --- flattening the real bundled template ---------------------------------


def test_real_template_flattens_structurally(real_template):
    wf = Workflow.from_ui(real_template)
    assert has_subgraph_instances(wf)
    flat, provenance = flatten(wf, {})
    defs = wf.subgraph_defs()
    # instance replaced by the definition's 9 inner nodes
    assert not any(n.type in defs for n in flat.nodes.values())
    inner_types = {n.type for nid, n in flat.nodes.items() if nid in provenance}
    assert {"KSampler", "VAEDecode", "CLIPTextEncode", "UNETLoader"} <= inner_types
    # provenance uses the frontend's instanceId:innerId convention
    assert all(p["path"].startswith("104:") for p in provenance.values())
    assert all(p["subgraph"] == "Text to Image (Z-Image-Turbo)" for p in provenance.values())
    # the external SaveImage consumer is rewired to the inner VAEDecode
    save = next(n for n in flat.nodes.values() if n.type == "SaveImage")
    link = flat.links[save.inputs[0].link]
    assert flat.nodes[link.origin_id].type == "VAEDecode"
    assert link.origin_id in provenance
    # inner wiring intact: KSampler.latent_image fed by EmptySD3LatentImage
    ks = next(n for n in flat.nodes.values() if n.type == "KSampler")
    latent_link = flat.links[ks.input_by_name("latent_image").link]
    assert flat.nodes[latent_link.origin_id].type == "EmptySD3LatentImage"
    # boundary inputs with no external feed leave widget values in charge
    encode = next(n for n in flat.nodes.values() if n.type == "CLIPTextEncode")
    assert encode.input_by_name("text").link is None
    assert "billboard" in encode.widgets_values[0]


def test_original_workflow_untouched_by_flatten(real_template):
    wf = Workflow.from_ui(real_template)
    before = wf.to_ui()
    flatten(wf, {})
    assert wf.to_ui() == before


# --- boundary + promotion semantics on synthetic docs ---------------------

SG_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def _doc(instance_overrides=None, def_overrides=None, extra_nodes=(), extra_links=()):
    """Top-level graph: [prompt source ->] instance(SG) -> SaveImage.
    Subgraph: CLIPTextEncode-less; KSampler -> VAEDecode -> boundary out, with
    a 'seed' boundary input into KSampler.seed (widget input)."""
    instance = {
        "id": 1,
        "type": SG_ID,
        "pos": [0, 0],
        "size": [200, 100],
        "inputs": [
            {"name": "latent", "type": "LATENT", "link": None},
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
        "widgets_values": [],
        "properties": {},
    }
    instance.update(instance_overrides or {})
    sg = {
        "id": SG_ID,
        "name": "Mini",
        "inputs": [{"name": "latent", "type": "LATENT", "linkIds": [30]}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "linkIds": [16]}],
        "nodes": [
            {
                "id": 3,
                "type": "KSampler",
                "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "positive", "type": "CONDITIONING", "link": None},
                    {"name": "negative", "type": "CONDITIONING", "link": None},
                    {"name": "latent_image", "type": "LATENT", "link": 30},
                ],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9]}],
            },
            {
                "id": 8,
                "type": "VAEDecode",
                "widgets_values": [],
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 9},
                    {"name": "vae", "type": "VAE", "link": None},
                ],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
            },
        ],
        "links": [
            {"id": 9, "origin_id": 3, "origin_slot": 0, "target_id": 8, "target_slot": 0, "type": "LATENT"},
            {"id": 16, "origin_id": 8, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
            {"id": 30, "origin_id": -10, "origin_slot": 0, "target_id": 3, "target_slot": 3, "type": "LATENT"},
        ],
    }
    sg.update(def_overrides or {})
    return {
        "id": "11111111-2222-4333-8444-555555555555",
        "revision": 0,
        "nodes": [
            instance,
            {
                "id": 2,
                "type": "SaveImage",
                "pos": [400, 0],
                "size": [300, 200],
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "widgets_values": ["ComfyUI"],
            },
            *extra_nodes,
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], *extra_links],
        "groups": [],
        "definitions": {"subgraphs": [sg]},
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def test_external_feed_reaches_inner_widget_input(oi):
    doc = _doc(
        instance_overrides={
            "inputs": [{"name": "latent", "type": "LATENT", "link": 40}],
        },
        extra_nodes=[
            {
                "id": 5,
                "type": "EmptyLatentImage",
                "pos": [-300, 0],
                "size": [200, 100],
                "inputs": [],
                "outputs": [{"name": "LATENT", "type": "LATENT", "links": [40]}],
                "widgets_values": [512, 512, 1],
            }
        ],
        extra_links=[[40, 5, 0, 1, 0, "LATENT"]],
    )
    api = Workflow.from_ui(doc).to_api(oi)
    ks = next(e for e in api.values() if e["class_type"] == "KSampler")
    assert ks["inputs"]["latent_image"] == ["5", 0]  # top-level id survives


def test_proxy_widget_values_override_inner_defaults(oi):
    doc = _doc(
        instance_overrides={
            "properties": {"proxyWidgets": [["3", "steps"], ["3", "sampler_name"]]},
            "widgets_values": [33, "heun"],
        }
    )
    api = Workflow.from_ui(doc).to_api(oi)
    ks = next(e for e in api.values() if e["class_type"] == "KSampler")
    assert ks["inputs"]["steps"] == 33
    assert ks["inputs"]["sampler_name"] == "heun"
    assert ks["inputs"]["seed"] == 42  # unproxied widgets keep inner values


def test_muted_instance_is_not_expanded(oi):
    doc = _doc(instance_overrides={"mode": MODE_MUTE})
    wf = Workflow.from_ui(doc)
    assert not has_subgraph_instances(wf)  # muted instances don't count
    api = wf.to_api(oi)
    assert {e["class_type"] for e in api.values()} == {"SaveImage"}


def test_nested_subgraphs_flatten_recursively(oi):
    inner_id = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
    doc = _doc(
        def_overrides={
            # outer def wraps an instance of the inner def
            "nodes": [
                {
                    "id": 7,
                    "type": inner_id,
                    "widgets_values": [],
                    "inputs": [],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                    "properties": {},
                }
            ],
            "links": [
                {"id": 16, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
            ],
        }
    )
    doc["definitions"]["subgraphs"].append(
        {
            "id": inner_id,
            "name": "Innermost",
            "inputs": [],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "linkIds": [16]}],
            "nodes": [
                {
                    "id": 8,
                    "type": "VAEDecode",
                    "widgets_values": [],
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": None},
                        {"name": "vae", "type": "VAE", "link": None},
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                }
            ],
            "links": [
                {"id": 16, "origin_id": 8, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
            ],
        }
    )
    wf = Workflow.from_ui(doc)
    flat, provenance = flatten(wf, oi)
    decode = next(n for n in flat.nodes.values() if n.type == "VAEDecode")
    assert provenance[decode.id]["path"].count(":") == 2  # 1:7:8
    assert provenance[decode.id]["subgraph"] == "Innermost"
    save = next(n for n in flat.nodes.values() if n.type == "SaveImage")
    link = flat.links[save.inputs[0].link]
    assert link.origin_id == decode.id


def test_self_referential_subgraph_hits_depth_cap(oi):
    doc = _doc(
        def_overrides={
            "nodes": [
                {
                    "id": 7,
                    "type": SG_ID,  # instance of ITSELF
                    "widgets_values": [],
                    "inputs": [],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16]}],
                    "properties": {},
                }
            ],
            "links": [
                {"id": 16, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE"},
            ],
        }
    )
    with pytest.raises(ValueError, match="nested deeper"):
        flatten(Workflow.from_ui(doc), oi)


def test_validate_flags_flatten_failure(oi):
    doc = _doc(def_overrides={"nodes": []})  # malformed: no inner nodes
    findings = validate(Workflow.from_ui(doc), oi)
    assert any(f["code"] == "subgraph-flatten-failed" for f in findings)


# --- write-time widget value validation ------------------------------------


def test_check_widget_value_combo_suggestion(oi):
    problem = check_widget_value("KSampler", "sampler_name", "euler_a", oi)
    assert "not an available option" in problem
    assert "euler_ancestral" in problem
    assert check_widget_value("KSampler", "sampler_name", "euler", oi) is None


def test_check_widget_value_range_and_types(oi):
    assert "outside the allowed range" in check_widget_value("KSampler", "steps", 0, oi)
    assert "expects an integer" in check_widget_value("KSampler", "steps", "20", oi)
    assert "expects a number" in check_widget_value("KSampler", "denoise", "1.0", oi)
    assert "cannot be null" in check_widget_value("KSampler", "denoise", None, oi)
    assert check_widget_value("KSampler", "denoise", 0.7, oi) is None
    assert check_widget_value("KSampler", "cfg", 8, oi) is None  # int ok for FLOAT
    # unknown names / classes are someone else's check
    assert check_widget_value("KSampler", "nope", "x", oi) is None
    assert check_widget_value("NotAClass", "steps", 5, oi) is None


@pytest.fixture
def wired(monkeypatch, tmp_path, oi):
    class StubClient:
        async def get_object_info(self, refresh=False):
            return oi

    from comfy_draftsman.config import Config

    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url="http://comfy.test", session_dir=tmp_path)
    )
    monkeypatch.setattr(server._State, "client", StubClient())
    monkeypatch.setattr(server._State, "session", session)
    wf = Workflow.new()
    wf_id = session.create(wf, title="t")
    return wf, wf_id


async def test_set_widget_rejects_invalid_value_at_write_time(wired, oi):
    wf, wf_id = wired
    await server.edit_workflow(wf_id, [{"op": "add_node", "class_type": "KSampler"}])
    (nid,) = wf.nodes
    result = await server.edit_workflow(
        wf_id, [{"op": "set_widget", "node_id": nid, "input": "sampler_name", "value": "dpmpp_sde_fake"}]
    )
    assert "not an available option" in result["error"]
    assert wf.get_widget(nid, "sampler_name", oi) == "euler"  # unchanged
    forced = await server.edit_workflow(
        wf_id,
        [{"op": "set_widget", "node_id": nid, "input": "sampler_name",
          "value": "dpmpp_sde_fake", "force": True}],
    )
    assert "error" not in forced
    assert wf.get_widget(nid, "sampler_name", oi) == "dpmpp_sde_fake"


async def test_add_node_rejects_invalid_widget_value_atomically(wired):
    wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [{"op": "add_node", "class_type": "KSampler", "widgets": {"steps": 0}}],
    )
    assert "outside the allowed range" in result["error"]
    assert wf.nodes == {}  # graph unchanged


# --- compact edit_workflow result ------------------------------------------


async def test_edit_result_is_compact_delta_by_default(wired):
    _wf, wf_id = wired
    result = await server.edit_workflow(
        wf_id,
        [
            {"op": "add_node", "class_type": "KSampler"},
            {"op": "add_node", "class_type": "VAEDecode"},
        ],
    )
    assert "summary" not in result
    assert result["nodes"] == 2 and result["links"] == 0
    assert {c["class_type"] for c in result["changed"]} == {"KSampler", "VAEDecode"}
    full = await server.edit_workflow(
        wf_id, [{"op": "set_widget", "node_id": 1, "input": "steps", "value": 25}], summary=True
    )
    assert "changed" not in full
    assert len(full["summary"]["nodes"]) == 2  # full graph, not just the touched node


# --- list_models metadata digest --------------------------------------------


def test_metadata_digest_trims_to_essentials():
    meta = {
        "ss_base_model_version": "sdxl_base_v1-0",
        "ss_output_name": "capybara_style",
        "ss_tag_frequency": json.dumps(
            {"10_capy": {"capybara": 50, "samurai": 30, "1boy": 5},
             "5_extra": {"capybara": 20}}
        ),
        "ss_bucket_info": "x" * 50_000,  # the huge stuff that must not pass through
    }
    digest = metadata_digest(meta)
    assert digest["ss_base_model_version"] == "sdxl_base_v1-0"
    assert digest["top_training_tags"][0] == "capybara (70)"
    assert "ss_bucket_info" not in digest
    assert len(json.dumps(digest)) < 1000


def test_metadata_digest_handles_unrecognized_metadata():
    assert "no recognizable" in metadata_digest({"weird_key": "1"})["note"]
    assert "weird_key" in metadata_digest({"weird_key": "1"})["note"]


async def test_list_models_metadata_for(monkeypatch, tmp_path, oi):
    from comfy_draftsman.config import Config

    class StubClient:
        async def list_model_folders(self):
            return ["loras"]

        async def get_model_metadata(self, folder, filename):
            assert (folder, filename) == ("loras", "capy.safetensors")
            return {"ss_output_name": "capy", "ss_base_model_version": "sdxl_base_v1-0"}

    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url="http://comfy.test", session_dir=tmp_path)
    )
    monkeypatch.setattr(server._State, "client", StubClient())
    result = await server.list_models(folder="loras", metadata_for="capy.safetensors")
    assert result["metadata"]["ss_output_name"] == "capy"

    class Missing(StubClient):
        async def get_model_metadata(self, folder, filename):
            raise FileNotFoundError(filename)

    monkeypatch.setattr(server._State, "client", Missing())
    result = await server.list_models(folder="loras", metadata_for="capy.safetensors")
    assert "no embedded metadata" in result["error"]
