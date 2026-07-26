"""Round 19: socket typing, PrimitiveNode authoring, and non-file outputs.

From a live session that built a character-cycling workflow and hit three walls:
a STRING wired into a COMBO widget validated clean but was rejected by the live
server (silent partial render), `PrimitiveNode` could not be added at all, and a
custom save node's `filenames`/`path` return values never surfaced.
"""

import json

import pytest

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.annotate import annotate, classify
from comfy_draftsman.graph.model import (
    MODE_MUTE,
    PRIMITIVE_TYPE,
    REROUTE_TYPE,
    Workflow,
    types_compatible,
)
from comfy_draftsman.graph.validate import validate

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"
SDXL = "tests/fixtures/sdxl_simple_example.json"


@pytest.fixture(scope="module")
def oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


def _errors(findings, code=None):
    return [
        f
        for f in findings
        if f["level"] == "error" and (code is None or f["code"] == code)
    ]


# --- types_compatible --------------------------------------------------------


@pytest.mark.parametrize(
    "out_type,in_type,ok",
    [
        ("IMAGE", "IMAGE", True),
        ("image", "IMAGE", True),  # litegraph typing is case-insensitive
        ("IMAGE", "LATENT", False),
        ("*", "COMBO", True),  # a wildcard producer adopts
        ("STRING", "*", True),
        ("STRING", "", True),
        ("ANY", "MODEL", True),  # pack spellings of the same idea
        ("COMBO", "COMBO", True),  # a mirrored primitive feeding a combo
        ("STRING", "COMBO", False),  # THE reported bug
        ("INT", "COMBO", False),
        ("IMAGE,LATENT", "LATENT", True),  # comma-joined union types intersect
        ("IMAGE,MASK", "LATENT", False),
    ],
)
def test_types_compatible(out_type, in_type, ok):
    assert types_compatible(out_type, in_type) is ok


# --- connect refuses a mismatched wire into a COMBO widget -------------------


def _string_into_ckpt_combo(oi):
    """The exact shape of the reported failure: a STRING source wired into a
    combo widget that was converted to an input."""
    wf = Workflow.new()
    gen = wf.add_node("DPRandomGenerator", object_info=oi)
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    return wf, gen, loader


def test_connect_refuses_string_into_a_combo_widget(oi):
    wf, gen, loader = _string_into_ckpt_combo(oi)
    with pytest.raises(ValueError, match="type mismatch"):
        wf.connect(gen.id, 0, loader.id, "ckpt_name", oi)
    # nothing was wired, and the widget was not converted to an input
    assert not wf.links
    assert loader.input_by_name("ckpt_name") is None


def test_a_refused_connect_leaves_no_half_converted_widget(oi):
    """Converting a widget to an input is a visible change - the node grows an
    empty socket and stops taking a typed value - so a refused connect must undo
    it rather than leave a dangling socket behind."""
    wf, gen, loader = _string_into_ckpt_combo(oi)
    before = [s.name for s in loader.inputs]
    with pytest.raises(ValueError):
        wf.connect(gen.id, 0, loader.id, "ckpt_name", oi)
    assert [s.name for s in loader.inputs] == before
    # and the widget value is still settable, i.e. the node is fully intact
    choices = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    wf.set_widget(loader.id, "ckpt_name", choices[1], oi)
    assert loader.widgets_values[0] == choices[1]


def test_connect_error_names_the_real_alternatives(oi):
    wf, gen, loader = _string_into_ckpt_combo(oi)
    with pytest.raises(ValueError) as excinfo:
        wf.connect(gen.id, 0, loader.id, "ckpt_name", oi)
    message = str(excinfo.value)
    assert "return_type_mismatch" in message  # what the live server would answer
    assert "set_widget" in message and "PrimitiveNode" in message


def test_connect_force_still_wires_a_mismatch(oi):
    """The escape hatch: draftsman is not the final judge of a custom pack's types."""
    wf, gen, loader = _string_into_ckpt_combo(oi)
    wf.connect(gen.id, 0, loader.id, "ckpt_name", oi, force=True)
    assert len(wf.links) == 1


def test_connect_still_allows_a_matching_widget_input(oi):
    """The stricter check must not break converting a normal widget to an input."""
    wf = Workflow.new()
    gen = wf.add_node("DPRandomGenerator", object_info=oi)
    encode = wf.add_node("CLIPTextEncode", object_info=oi)
    wf.connect(gen.id, 0, encode.id, "text", oi)  # STRING -> STRING
    assert encode.input_by_name("text").link is not None


# --- validate catches a mismatch that arrived by import ----------------------


def test_validate_reports_a_link_type_mismatch(oi):
    wf, gen, loader = _string_into_ckpt_combo(oi)
    wf.connect(gen.id, 0, loader.id, "ckpt_name", oi, force=True)
    found = _errors(validate(wf, oi), "link-type-mismatch")
    assert len(found) == 1
    assert found[0]["node_id"] == loader.id
    assert found[0]["input"] == "ckpt_name"
    assert "COMBO" in found[0]["message"]


def test_validate_is_quiet_on_a_correctly_typed_graph(oi):
    with open(SDXL, encoding="utf-8") as fh:
        wf = Workflow.from_ui(json.load(fh))
    # a real bundled template must not trip the new check
    assert not _errors(validate(wf, oi), "link-type-mismatch")


def test_validate_skips_a_disabled_mismatch(oi):
    """A muted node never runs, so its wiring cannot break the prompt."""
    wf, gen, loader = _string_into_ckpt_combo(oi)
    wf.connect(gen.id, 0, loader.id, "ckpt_name", oi, force=True)
    loader.mode = MODE_MUTE
    assert not _errors(validate(wf, oi), "link-type-mismatch")


def test_validate_skips_mismatch_through_a_reroute(oi):
    """Reroutes adopt their neighbour's type, so they are not the mismatch."""
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    reroute = wf.add_node(REROUTE_TYPE)
    encode = wf.add_node("CLIPTextEncode", object_info=oi)
    wf.connect(loader.id, 1, reroute.id, "", oi)  # CLIP -> Reroute
    wf.connect(reroute.id, 0, encode.id, "clip", oi)
    assert not _errors(validate(wf, oi), "link-type-mismatch")


# --- PrimitiveNode / Reroute are authorable ----------------------------------


def test_add_primitive_node_is_allowed_and_typeless(oi):
    wf = Workflow.new()
    node = wf.add_node(PRIMITIVE_TYPE, object_info=oi)
    assert node.outputs[0].type == "*"  # no type until it is connected
    assert not node.inputs
    assert node.properties == {"Run widget replace on values": False}


def test_add_reroute_node_has_one_slot_each_way(oi):
    wf = Workflow.new()
    node = wf.add_node(REROUTE_TYPE, object_info=oi)
    assert len(node.inputs) == 1 and len(node.outputs) == 1
    assert node.inputs[0].type == "*" and node.outputs[0].type == "*"
    assert node.size == [75.0, 26.0]


def test_primitive_mirrors_an_int_widget(oi):
    """Shape verified against a real export (sdxl_simple_example.json #45)."""
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    assert prim.outputs[0].type == "INT"
    assert prim.outputs[0].name == "INT"
    assert prim.outputs[0].widget_name == "steps"
    assert prim.title == "steps"
    assert prim.widgets_values == [20, "fixed"]  # schema default + control slot
    assert prim.size == [210.0, 82.0]


def test_primitive_mirrors_a_combo_widget_as_COMBO(oi):
    """The capability that was impossible before: a value that adapts to a
    dropdown and can cycle it."""
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, loader.id, "ckpt_name", oi)
    choices = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    assert prim.outputs[0].type == "COMBO"
    assert prim.widgets_values == [choices[0], "fixed"]
    # and the wire itself is legal, unlike the STRING workaround
    assert not _errors(validate(wf, oi), "link-type-mismatch")


def test_primitive_mirroring_a_string_gets_no_control_slot(oi):
    """The frontend adds control_after_generate only for number/combo widgets."""
    wf = Workflow.new()
    encode = wf.add_node("CLIPTextEncode", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, encode.id, "text", oi)
    assert prim.outputs[0].type == "STRING"
    assert prim.widgets_values == [""]  # value only
    assert prim.size == [300.0, 160.0]  # multiline gets the taller box


def test_a_bound_primitive_is_not_retyped_by_a_second_consumer(oi):
    wf = Workflow.new()
    a = wf.add_node("KSampler", object_info=oi)
    b = wf.add_node("CLIPTextEncode", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, a.id, "steps", oi)
    wf.set_widget(prim.id, "value", 12, oi)
    with pytest.raises(ValueError, match="type mismatch"):
        wf.connect(prim.id, 0, b.id, "text", oi)  # INT primitive -> STRING widget
    assert prim.outputs[0].type == "INT"
    assert prim.widgets_values[0] == 12  # value survived the refused connect


def test_primitive_value_is_inlined_into_the_api_prompt(oi):
    """A primitive is virtual: /prompt never sees it, it sees its value."""
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, loader.id, "ckpt_name", oi)
    choices = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    wf.set_widget(prim.id, "value", choices[2], oi)
    api = wf.to_api(oi)
    assert str(prim.id) not in api
    assert api[str(loader.id)]["inputs"]["ckpt_name"] == choices[2]


# --- primitive widget addressing ---------------------------------------------


def test_primitive_value_addressable_by_alias_and_real_name(oi):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    wf.set_widget(prim.id, "value", 30, oi)
    assert prim.widgets_values[0] == 30
    wf.set_widget(prim.id, "steps", 40, oi)  # the mirrored widget's own name
    assert prim.widgets_values[0] == 40


def test_primitive_control_mode_is_settable_and_checked(oi):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    wf.set_widget(prim.id, "control_after_generate", "increment", oi)
    assert prim.widgets_values == [20, "increment"]
    with pytest.raises(ValueError, match="must be one of"):
        wf.set_widget(prim.id, "control_after_generate", "sideways", oi)


def test_string_primitive_rejects_a_control_mode(oi):
    wf = Workflow.new()
    encode = wf.add_node("CLIPTextEncode", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, encode.id, "text", oi)
    with pytest.raises(ValueError, match="only gives number and combo"):
        wf.set_widget(prim.id, "control_after_generate", "increment", oi)


def test_unknown_primitive_widget_name_is_actionable(oi):
    wf = Workflow.new()
    prim = wf.add_node(PRIMITIVE_TYPE)
    with pytest.raises(ValueError, match="connect it to a widget input first"):
        wf.set_widget(prim.id, "nonsense", 1, oi)


# --- headless cycling: the capability the report said was missing ------------


def _cycling_graph(oi, mode="increment"):
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, loader.id, "ckpt_name", oi)
    wf.set_widget(prim.id, "control_after_generate", mode, oi)
    return wf, prim, oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]


def test_combo_primitive_advances_one_option_per_run(oi):
    wf, prim, choices = _cycling_graph(oi)
    assert prim.widgets_values[0] == choices[0]
    for expected in choices[1:5]:
        assert wf.apply_seed_control(oi) is True
        assert prim.widgets_values[0] == expected


def test_combo_primitive_clamps_at_the_end_like_the_frontend(oi):
    """ComfyUI's addValueControlWidgets clamps the index; it does not wrap.
    Mirroring that matters more than being clever - a wrap would silently differ
    from what the same workflow does in the browser."""
    wf, prim, choices = _cycling_graph(oi)
    wf.set_widget(prim.id, "value", choices[-1], oi)
    assert wf.apply_seed_control(oi) is False
    assert prim.widgets_values[0] == choices[-1]


def test_combo_primitive_decrement_clamps_at_zero(oi):
    wf, prim, choices = _cycling_graph(oi, mode="decrement")
    assert wf.apply_seed_control(oi) is False  # already at index 0
    wf.set_widget(prim.id, "value", choices[1], oi)
    assert wf.apply_seed_control(oi) is True
    assert prim.widgets_values[0] == choices[0]


def test_combo_primitive_randomize_stays_in_the_option_list(oi):
    wf, prim, choices = _cycling_graph(oi, mode="randomize")
    for _ in range(20):
        wf.apply_seed_control(oi)
        assert prim.widgets_values[0] in choices


def test_fixed_primitive_never_moves(oi):
    wf, prim, choices = _cycling_graph(oi, mode="fixed")
    assert wf.apply_seed_control(oi) is False
    assert prim.widgets_values[0] == choices[0]


def test_int_primitive_increments_by_step_and_clamps_to_max(oi):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    wf.set_widget(prim.id, "control_after_generate", "increment", oi)
    wf.apply_seed_control(oi)
    assert prim.widgets_values[0] == 21  # default 20 + step 1
    wf.set_widget(prim.id, "value", 10000, oi)  # schema max
    assert wf.apply_seed_control(oi) is False


def test_an_unbound_primitive_never_rolls(oi):
    wf = Workflow.new()
    prim = wf.add_node(PRIMITIVE_TYPE)
    prim.widgets_values = [5, "increment"]
    assert wf.apply_seed_control(oi) is False


def test_primitive_rolling_survives_a_reroute(oi):
    """The value still has to reach a widget to know what it is."""
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    reroute = wf.add_node(REROUTE_TYPE)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, reroute.id, "", oi)
    reroute.outputs[0].type = "COMBO"
    wf.connect(reroute.id, 0, loader.id, "ckpt_name", oi)
    choices = oi["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    prim.widgets_values = [choices[0], "increment"]
    assert wf.apply_seed_control(oi) is True
    assert prim.widgets_values[0] == choices[1]


# --- primitive validation ----------------------------------------------------


def test_primitive_with_an_invalid_combo_value_is_an_error(oi):
    """Nothing checked this before: the consumer's slot is connected, so it skips
    its own widget check, and virtual nodes were skipped entirely."""
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, loader.id, "ckpt_name", oi)
    prim.widgets_values = ["not_a_real_model.safetensors", "fixed"]
    found = _errors(validate(wf, oi), "primitive-value-invalid")
    assert len(found) == 1
    assert found[0]["node_id"] == prim.id
    assert "ckpt_name" in found[0]["message"]


def test_a_valid_primitive_value_passes(oi):
    wf, _prim, _choices = _cycling_graph(oi)
    assert not _errors(validate(wf, oi), "primitive-value-invalid")


def test_an_unbound_primitive_warns_but_does_not_block(oi):
    """to_api drops it, so it cannot break a run - it is just useless."""
    wf = Workflow.new()
    wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    findings = validate(wf, oi)
    unbound = [f for f in findings if f["code"] == "primitive-unbound"]
    assert len(unbound) == 1
    assert unbound[0]["level"] == "warning"
    assert unbound[0]["node_id"] == prim.id


def test_set_widget_op_rejects_a_bogus_primitive_value(oi, monkeypatch, tmp_path):
    """Write-time rejection, so a made-up dropdown value fails at the edit, not
    at run time."""
    from comfy_draftsman.graph.validate import check_primitive_value

    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, loader.id, "ckpt_name", oi)
    problem = check_primitive_value(wf, prim, "invented_model.safetensors", oi)
    assert problem and "ckpt_name" in problem
    assert check_primitive_value(wf, prim, prim.widgets_values[0], oi) is None


# --- the round-trip data loss ------------------------------------------------


def test_primitive_output_widget_marker_survives_a_round_trip():
    """A real primitive's output records which widget it drives
    ({"widget": {"name": "steps"}}). OutputSlot had no such field, so every
    save/load cycle dropped it - including for the bundled SDXL template."""
    with open(SDXL, encoding="utf-8") as fh:
        source = json.load(fh)
    wf = Workflow.from_ui(source)
    out = wf.to_ui()
    by_id = {n["id"]: n for n in out["nodes"]}
    for node in source["nodes"]:
        if node["type"] != PRIMITIVE_TYPE:
            continue
        original = node["outputs"][0].get("widget")
        assert original is not None, "fixture should exercise the marker"
        assert by_id[node["id"]]["outputs"][0].get("widget") == original


def test_ordinary_outputs_gain_no_widget_key():
    with open(SDXL, encoding="utf-8") as fh:
        wf = Workflow.from_ui(json.load(fh))
    for node in wf.to_ui()["nodes"]:
        if node["type"] == PRIMITIVE_TYPE:
            continue
        assert all("widget" not in o for o in node["outputs"])


# --- layout / annotate placement --------------------------------------------


def test_primitive_classifies_as_a_tweakable_input(oi):
    wf = Workflow.new()
    prim = wf.add_node(PRIMITIVE_TYPE)
    assert classify(prim, oi) == "inputs"


def test_organize_paints_a_primitive_green_and_keeps_its_size(oi):
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    wf.add_node("SaveImage", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    annotate(wf, oi)
    assert (prim.color, prim.bgcolor) == ("#232", "#353")
    # estimate_size knows nothing about virtual classes and would flatten it
    assert prim.size == [210.0, 82.0]


def test_organize_keeps_a_reroute_tiny(oi):
    wf = Workflow.new()
    loader = wf.add_node("CheckpointLoaderSimple", object_info=oi)
    reroute = wf.add_node(REROUTE_TYPE)
    encode = wf.add_node("CLIPTextEncode", object_info=oi)
    wf.connect(loader.id, 1, reroute.id, "", oi)
    wf.connect(reroute.id, 0, encode.id, "clip", oi)
    annotate(wf, oi)
    assert reroute.size == [75.0, 26.0]


def test_summary_shows_primitive_wiring(oi, monkeypatch, tmp_path):
    """The summary used to hide every link touching a virtual node, so an
    authored primitive looked unconnected in the very view used to check it."""
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "session", None)
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    prim = wf.add_node(PRIMITIVE_TYPE)
    wf.connect(prim.id, 0, sampler.id, "steps", oi)
    wf_id = server._session().create(wf)
    links = server._summary(wf_id, wf)["links"]
    assert any(f"#{prim.id}[0]" in link and "steps" in link for link in links)


def test_summary_still_hides_note_links(oi, monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "session", None)
    wf = Workflow.new()
    sampler = wf.add_node("KSampler", object_info=oi)
    note = wf.add_node("Note")
    wf_id = server._session().create(wf)
    summary = server._summary(wf_id, wf)
    # a note has no sockets, so it contributes no links - but it is still listed
    # as a node, flagged virtual
    assert summary["links"] == []
    assert any(n["id"] == note.id and n.get("virtual") for n in summary["nodes"])
    assert any(n["id"] == sampler.id for n in summary["nodes"])


# --- non-file outputs --------------------------------------------------------


def test_data_outputs_surface_text_and_paths():
    """A custom save node's filenames/path and a ShowText node's text were
    dropped: only the four file keys were harvested."""
    history = {
        "outputs": {
            "7": {
                "images": [{"filename": "a.png", "subfolder": "", "type": "output"}],
                "filenames": ["D:/renders/a.png"],
                "path": "D:/renders",
                "saved_count": 1,
                "animated": [False],
            },
            "9": {"text": ["a generated prompt"]},
        }
    }
    data = ComfyClient._collect_data_outputs(history)
    assert data["7"] == {
        "filenames": ["D:/renders/a.png"],
        "path": "D:/renders",
        "saved_count": 1,
    }
    assert data["9"] == {"text": ["a generated prompt"]}
    assert "images" not in data["7"]  # file refs stay in `outputs`
    assert "animated" not in data["7"]  # frontend noise


def test_file_outputs_are_unchanged_so_relocation_still_works():
    history = {
        "outputs": {
            "7": {
                "images": [{"filename": "a.png", "subfolder": "", "type": "output"}],
                "text": ["ignored here"],
            }
        }
    }
    files = ComfyClient._collect_outputs(history)
    assert files == [
        {
            "filename": "a.png",
            "subfolder": "",
            "type": "output",
            "node_id": "7",
            "kind": "images",
        }
    ]


def test_data_outputs_are_empty_when_there_is_nothing_extra():
    """Costs nothing in the common case - the key is omitted entirely."""
    history = {"outputs": {"7": {"images": [{"filename": "a.png"}]}}}
    assert ComfyClient._collect_data_outputs(history) == {}


def test_a_huge_text_output_is_clipped_and_says_so():
    history = {"outputs": {"9": {"text": "x" * 50_000}}}
    data = ComfyClient._collect_data_outputs(history)
    assert len(data["9"]["text"]) <= ComfyClient._DATA_VALUE_CHARS + 1
    assert data["9"]["text"].endswith("…")
    assert "9.text" in data["note"]


def test_data_outputs_respect_a_total_budget():
    history = {
        "outputs": {str(i): {"text": "y" * 1000} for i in range(50)}
    }
    data = ComfyClient._collect_data_outputs(history)
    assert len(json.dumps(data)) < 2 * ComfyClient._DATA_TOTAL_CHARS
    assert "omitted" in data["note"]


@pytest.mark.asyncio
async def test_get_run_status_reports_data_outputs(monkeypatch, tmp_path):
    class _Client:
        _collect_outputs = staticmethod(ComfyClient._collect_outputs)

        async def get_object_info(self, refresh: bool = False):
            return {}  # partial-accept detection consults it

        async def get_history(self, prompt_id):
            return {
                "outputs": {"9": {"text": ["hello"]}},
                "status": {"messages": []},
                "prompt": [0, "p", {}, {}, []],
            }

    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    result = await server.get_run_status(prompt_id="p1")
    assert result["status"] == "success"
    assert result["data_outputs"] == {"9": {"text": ["hello"]}}


# --- the per-session tool-schema budget --------------------------------------


@pytest.mark.asyncio
async def test_tool_schemas_stay_within_the_session_budget():
    """Every tool's description + input schema is re-sent on EVERY session, before
    the agent does anything - the one payload nobody can opt out of. Measured
    19,631 chars (~4.9k tokens) at round 19, down from 19,994 while adding the
    primitive/data-output docs. The ceiling is loose (a regression guard, not a
    golden master), but a new tool or a docstring that doubles will trip it."""
    tools = await server.mcp.list_tools()
    total = sum(
        len(t.description or "") + len(json.dumps(t.inputSchema)) for t in tools
    )
    assert total < 25_000, f"tool schemas grew to {total} chars"
    # no single tool should dominate: the two fattest are the op-schema mirror
    # (edit_workflow) and run_workflow's parameter semantics
    worst = max(len(t.description or "") for t in tools)
    assert worst < 2_000


# --- widgets helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        (["INT", {}], True),
        (["FLOAT", {}], True),
        ([["a", "b"], {}], True),  # legacy combo
        (["COMBO", {"options": ["a"]}], True),
        (["STRING", {}], False),
        (["BOOLEAN", {}], False),
        (["MODEL", {}], False),
    ],
)
def test_primitive_takes_control(spec, expected):
    assert w.primitive_takes_control(spec) is expected


def test_combo_choices_covers_every_flavour():
    assert w.combo_choices([["a", "b"], {}]) == ["a", "b"]
    assert w.combo_choices(["COMBO", {"options": ["x"]}]) == ["x"]
    assert w.combo_choices(["STRING", {}]) is None
    assert w.combo_choices(["INT", {}]) is None
