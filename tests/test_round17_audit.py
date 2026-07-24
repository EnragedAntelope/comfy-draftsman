"""Round 17: repo-audit remediation.

Each test here pins a defect the suite did not previously cover. The two
headline ones are silent: a `set_widget` that destroyed a neighbouring widget's
value on pack nodes with custom JS-widget inputs, and muted branches producing
blocking validation errors for nodes that are never submitted at all.
"""

import json
from pathlib import Path

import httpx
import pytest

from comfy_draftsman import server
from comfy_draftsman.comfy.registry import RegistryClient, RegistryUnavailableError
from comfy_draftsman.config import Config
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.annotate import GREEN, annotate
from comfy_draftsman.graph.model import MODE_BYPASS, MODE_MUTE, Workflow
from comfy_draftsman.graph.validate import check_widget_value, validate
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def object_info():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


# --- #1 custom JS-widget inputs survive the write path -----------------------
#
# A pack can declare an input whose type is a bespoke string that its own
# frontend renders as a widget, not a socket (LoraManager's
# AUTOCOMPLETE_TEXT_LORAS, StyleStringInjector2's ZIPN_STYLE_GALLERY_BUTTON).
# Schema alone can't tell that from a connection type, so it is recognized
# per-instance: an input the node did NOT serialize in its `inputs` socket array
# can only be a widget. to_api/validate already passed that context; the write
# path did not, so the slot walk missed the widget and every value after it
# shifted up one position.

CUSTOM_WIDGET_OI = {
    "LoraManagerLoader": {
        "input": {
            "required": {
                "lora_text": ["AUTOCOMPLETE_TEXT_LORAS", {}],
                "strength": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}],
            }
        },
        "output": ["MODEL"],
        "output_name": ["MODEL"],
        "python_module": "custom_nodes.lora_manager",
    }
}


def _custom_widget_workflow() -> Workflow:
    # `inputs: []` is the point: the node serialized no sockets, so lora_text
    # can only be a JS widget.
    return Workflow.from_ui(
        {
            "nodes": [
                {
                    "id": 1,
                    "type": "LoraManagerLoader",
                    "widgets_values": ["<lora:foo:0.8>", 1.0],
                    "inputs": [],
                    "outputs": [],
                }
            ],
            "links": [],
        }
    )


def test_set_widget_preserves_custom_js_widget_value():
    wf = _custom_widget_workflow()
    wf.set_widget(1, "strength", 2.0, CUSTOM_WIDGET_OI)
    # before the fix this was [2.0]: the lora text was destroyed and the
    # strength landed in its slot
    assert wf.nodes[1].widgets_values == ["<lora:foo:0.8>", 2.0]


def test_to_api_still_sees_both_widgets_after_a_write():
    wf = _custom_widget_workflow()
    wf.set_widget(1, "strength", 2.0, CUSTOM_WIDGET_OI)
    api = wf.to_api(CUSTOM_WIDGET_OI)["1"]["inputs"]
    assert api["lora_text"] == "<lora:foo:0.8>"
    assert api["strength"] == 2.0


def test_set_widget_can_target_the_custom_widget_itself():
    wf = _custom_widget_workflow()
    wf.set_widget(1, "lora_text", "<lora:bar:1.0>", CUSTOM_WIDGET_OI)
    assert wf.nodes[1].widgets_values == ["<lora:bar:1.0>", 1.0]


def test_get_widget_reads_through_the_custom_widget():
    wf = _custom_widget_workflow()
    assert wf.get_widget(1, "strength", CUSTOM_WIDGET_OI) == 1.0
    assert wf.get_widget(1, "lora_text", CUSTOM_WIDGET_OI) == "<lora:foo:0.8>"


def test_check_widget_value_finds_the_spec_behind_a_custom_widget():
    # without socket context the slot walk never reaches `strength`, so its spec
    # is missed and an out-of-range value passes the write-time check
    problem = check_widget_value(
        "LoraManagerLoader",
        "strength",
        99.0,
        CUSTOM_WIDGET_OI,
        ["<lora:foo:0.8>", 1.0],
        socket_names=set(),
    )
    assert problem is not None and "outside the allowed range" in problem


def test_named_to_widgets_without_socket_context_stays_schema_only():
    # the fresh-node path must NOT infer a custom widget (no instance to read
    # sockets from) - that conservatism is deliberate, so pin it
    assert w.widget_slot_names("LoraManagerLoader", CUSTOM_WIDGET_OI) == ["strength"]


# --- #2 muted / bypassed nodes don't block a run -----------------------------


def _disabled_branch(object_info, mode) -> Workflow:
    wf = Workflow.new()
    upscale = wf.add_node("ImageUpscaleWithModel", object_info=object_info)
    save = wf.add_node("SaveImage", object_info=object_info)
    upscale.mode = mode
    save.mode = mode
    return wf


@pytest.mark.parametrize("mode", [MODE_MUTE, MODE_BYPASS])
def test_disabled_branch_does_not_block_run_or_save(object_info, mode):
    wf = _disabled_branch(object_info, mode)
    findings = validate(wf, object_info)
    assert [f for f in findings if f["level"] == "error"] == []
    # ...because those nodes aren't in the submitted prompt at all
    assert wf.to_api(object_info) == {}


@pytest.mark.parametrize("mode", [MODE_MUTE, MODE_BYPASS])
def test_disabled_nodes_are_reported_as_info(object_info, mode):
    findings = validate(_disabled_branch(object_info, mode), object_info)
    disabled = [f for f in findings if f["code"] == "node-disabled"]
    assert len(disabled) == 2
    assert all(f["level"] == "info" for f in disabled)


def test_enabling_the_branch_restores_the_errors(object_info):
    wf = _disabled_branch(object_info, MODE_MUTE)
    for node in wf.nodes.values():
        node.mode = 0
    codes = {f["code"] for f in validate(wf, object_info) if f["level"] == "error"}
    assert "unconnected-input" in codes


def test_bypassed_dead_end_is_caught_on_the_consumer(object_info):
    """Bypass is a passthrough, so a bypassed node with nothing feeding it
    forwards a hole: to_api drops the consumer's input entirely and ComfyUI
    rejects the prompt. Previously invisible to validate."""
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    upscale = wf.add_node("LatentUpscale", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "MODEL", sampler.id, "model", object_info)
    wf.connect(upscale.id, 0, sampler.id, "latent_image", object_info)
    upscale.mode = MODE_BYPASS

    findings = validate(wf, object_info)
    dead = [f for f in findings if f["code"] == "dead-input-source"]
    assert len(dead) == 1
    assert dead[0]["node_id"] == sampler.id
    assert dead[0]["input"] == "latent_image"
    # and the finding is true: the input really is missing from the prompt
    assert "latent_image" not in wf.to_api(object_info)[str(sampler.id)]["inputs"]


# --- #6 import parse failures are actionable ---------------------------------


def test_from_ui_names_a_node_missing_its_type():
    with pytest.raises(ValueError, match="needs an 'id' and a 'type'"):
        Workflow.from_ui({"nodes": [{"id": 1}], "links": []})


def test_from_ui_names_a_non_numeric_node_id():
    with pytest.raises(ValueError, match="not a number"):
        Workflow.from_ui({"nodes": [{"id": "abc", "type": "KSampler"}], "links": []})


def test_from_api_accepts_non_numeric_node_ids():
    """A prompt keyed by arbitrary strings used to crash the import on int()."""
    api = {
        "a1b2": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["a1b2", 1]}},
    }
    wf = Workflow.from_api(api, {})
    types = {n.type for n in wf.nodes.values()}
    assert types == {"CheckpointLoaderSimple", "CLIPTextEncode"}
    # the original key is preserved, and the remapped id doesn't collide with 7
    remapped = next(n for n in wf.nodes.values() if n.type == "CheckpointLoaderSimple")
    assert remapped.properties["api_node_id"] == "a1b2"
    assert remapped.id != 7
    # the connection survived the remap
    assert len(wf.links) == 1


def test_from_api_rejects_a_ui_document_with_a_clear_message():
    with pytest.raises(ValueError, match="not a valid API-format entry"):
        Workflow.from_api({"0": ["not", "an", "entry"]}, {})


@pytest.mark.asyncio
async def test_import_workflow_reports_bad_json(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "session", Session(tmp_path))
    result = await server.import_workflow(workflow_json="{not json")
    assert "error" in result and "not valid JSON" in result["error"]
    assert "hint" in result


@pytest.mark.asyncio
async def test_import_workflow_reports_a_malformed_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "session", Session(tmp_path))
    result = await server.import_workflow(workflow_json='{"nodes": [{"id": 1}]}')
    assert "error" in result and "could not parse" in result["error"]


# --- #5 the registry degrades instead of crashing ----------------------------


class _DeadTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ConnectError("all connection attempts failed", request=request)


def _offline_registry() -> RegistryClient:
    registry = RegistryClient(Config(registry_url="http://registry.test"))
    registry._http = httpx.AsyncClient(
        base_url="http://registry.test", transport=_DeadTransport()
    )
    return registry


@pytest.mark.asyncio
async def test_registry_offline_raises_an_actionable_error():
    with pytest.raises(RegistryUnavailableError, match="works offline"):
        await _offline_registry().resolve_node_classes(["SomePackNode"])


@pytest.mark.asyncio
async def test_diagnose_keeps_its_findings_when_the_registry_is_offline(
    monkeypatch, tmp_path, object_info
):
    """The registry is the one remote dependency; losing it must not throw away
    the local validation work already done."""

    class _Client:
        async def get_object_info(self, refresh: bool = False):
            return object_info

    wf = Workflow.new()
    wf.add_node("NotInstalledNode")
    session = Session(tmp_path)
    wf_id = session.create(wf, title="t")
    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "registry", _offline_registry())

    result = await server.diagnose_workflow(wf_id)
    assert any(f["code"] == "missing-node-class" for f in result["findings"])
    assert "can't reach the Comfy Registry" in result["missing_node_packs"]["error"]


@pytest.mark.asyncio
async def test_resolve_node_classes_runs_lookups_concurrently():
    """A dozen missing classes used to be a dozen serial round trips."""
    in_flight = 0
    peak = 0

    class _CountingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                import asyncio

                await asyncio.sleep(0)
                return httpx.Response(404, request=request)
            finally:
                in_flight -= 1

    registry = RegistryClient(Config(registry_url="http://registry.test"))
    registry._http = httpx.AsyncClient(
        base_url="http://registry.test", transport=_CountingTransport()
    )
    result = await registry.resolve_node_classes([f"Node{i}" for i in range(6)])
    assert result["unresolved"] == [f"Node{i}" for i in range(6)]
    assert peak > 1


# --- #8 session persistence is durable ---------------------------------------


def test_persist_is_atomic_and_leaves_no_temp_files(tmp_path):
    session = Session(tmp_path)
    wf_id = session.create(Workflow.new(), title="t")
    path = session.persist(wf_id)
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["extra"]["draftsman_title"] == "t"


def test_corrupt_session_file_reads_as_a_named_error(tmp_path):
    session = Session(tmp_path)
    (tmp_path / "deadbeef.json").write_text("{truncated", encoding="utf-8")
    with pytest.raises(KeyError, match="could not be read"):
        session.get("deadbeef")


def test_list_ignores_a_corrupt_file(tmp_path):
    session = Session(tmp_path)
    (tmp_path / "deadbeef.json").write_text("{truncated", encoding="utf-8")
    assert session.list() == [{"id": "deadbeef", "title": "untitled"}]


# --- #11 organize clears a stale "touch me" highlight ------------------------


def test_reorganize_clears_green_from_a_node_that_stopped_being_a_knob(object_info):
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    sampler = wf.add_node("KSampler", object_info=object_info)
    wf.connect(ckpt.id, "CLIP", pos.id, "clip", object_info)
    wf.connect(pos.id, "CONDITIONING", sampler.id, "positive", object_info)
    annotate(wf, object_info)
    assert (pos.color, pos.bgcolor) == GREEN  # typeable prompt -> a knob

    # now the prompt text is driven from upstream, so it is no longer typeable
    wildcard = wf.add_node("DPRandomGenerator", object_info=object_info)
    wf.connect(wildcard.id, 0, pos.id, "text", object_info)
    annotate(wf, object_info)
    assert (pos.color, pos.bgcolor) != GREEN
