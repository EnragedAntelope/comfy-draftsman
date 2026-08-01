"""Round 20: ComfyUI V3 meta io types, and telling the truth about subgraph edits.

From a live session that built a Krea-2 + LM Studio workflow and logged what got
in its way. Three of the seven reported issues were real:

- every bundled Krea-2 template was rejected with `link-type-mismatch` on its
  `ComfySwitchNode`s. That node is core, and its `COMFY_MATCHTYPE_V3` slots are
  wildcards the backend short-circuits (`comfy_execution.validation`
  `validate_node_input`, lines 31-34). Draftsman treated the marker as a concrete
  type, which made those templates unrunnable AND unsavable.
- a model path wrong inside a sealed subgraph reported "edit_workflow can't reach
  inside; rebuild flat" - false since the `*_in_definition` ops landed. The
  session hand-rebuilt a 14-node graph to change one string.
- `inspect_workflow` listed all six of a definition's boundary inputs; the
  instance exposes one. `connect` to any of the other five is impossible.

Plus two found while verifying those: V3 meta types were being counted as custom
JS widgets (inventing a widget slot and shifting every later `widgets_values`
entry), and `no-prompt-preview` fired on a workflow whose Show Text was tapped
off the generator rather than sitting inline.
"""

import json

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MATCH_TYPE, Workflow, types_compatible
from comfy_draftsman.graph.subgraph import flatten
from comfy_draftsman.graph.validate import validate
from comfy_draftsman.session import Session

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"
SUBGRAPH_TEMPLATE = "tests/fixtures/subgraph_real_template.json"

# The real ComfyUI schema for the core If/Else Switch (comfy_extras/nodes_logic.py
# SwitchNode): MatchType in, MatchType out, sharing one template.
SWITCH_SCHEMA = {
    "input": {
        "required": {
            "switch": ["BOOLEAN", {"default": True}],
            "on_false": [MATCH_TYPE, {"template": {"template_id": "switch"}}],
            "on_true": [MATCH_TYPE, {"template": {"template_id": "switch"}}],
        }
    },
    "output": [MATCH_TYPE],
    "output_name": ["output"],
    "name": "ComfySwitchNode",
    "display_name": "If/Else Switch",
    "python_module": "comfy_extras.nodes_logic",
}


@pytest.fixture(scope="module")
def oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def oi_switch(oi):
    return {**oi, "ComfySwitchNode": SWITCH_SCHEMA}


@pytest.fixture
def subgraph_doc():
    with open(SUBGRAPH_TEMPLATE, encoding="utf-8") as fh:
        return json.load(fh)


def _errors(findings, code=None):
    return [
        f
        for f in findings
        if f["level"] == "error" and (code is None or f["code"] == code)
    ]


# --- MatchType is a wildcard -------------------------------------------------


@pytest.mark.parametrize(
    "out_type,in_type",
    [
        (MATCH_TYPE, "STRING"),  # switch output -> a text widget
        (MATCH_TYPE, "MODEL"),  # switch output -> a sampler's model input
        ("MODEL", MATCH_TYPE),  # a loader -> switch input
        ("CONDITIONING", MATCH_TYPE),
        (MATCH_TYPE, MATCH_TYPE),  # switch chained into switch
        (MATCH_TYPE, "COMBO"),  # even a converted combo widget
    ],
)
def test_match_type_is_compatible_with_anything(out_type, in_type):
    assert types_compatible(out_type, in_type)


def test_match_type_matches_backend_short_circuit():
    """The literal spelling matters - it is the io_type string ComfyUI emits."""
    assert MATCH_TYPE == "COMFY_MATCHTYPE_V3"
    assert types_compatible(MATCH_TYPE.lower(), "IMAGE")  # comparison is case-folded


def test_concrete_mismatch_still_refused():
    """The MatchType exemption must not become a general amnesty."""
    assert not types_compatible("IMAGE", "LATENT")
    assert not types_compatible("STRING", "COMBO")


def test_switch_node_wiring_validates_clean(oi_switch):
    """The regression that made every bundled Krea-2 template unusable."""
    wf = Workflow.new()
    switch = wf.add_node("ComfySwitchNode", object_info=oi_switch)
    encode = wf.add_node("CLIPTextEncode", object_info=oi_switch)
    loader = wf.add_node("UNETLoader", object_info=oi_switch)
    wf.connect(loader.id, 0, switch.id, "on_true", oi_switch)
    wf.connect(switch.id, 0, encode.id, "text", oi_switch)
    assert _errors(validate(wf, oi_switch), "link-type-mismatch") == []


def test_switch_node_connect_not_refused(oi_switch):
    """`connect` and `validate` share types_compatible - both had to be fixed."""
    wf = Workflow.new()
    switch = wf.add_node("ComfySwitchNode", object_info=oi_switch)
    encode = wf.add_node("CLIPTextEncode", object_info=oi_switch)
    # no force=True: this must be legal on its own merits
    wf.connect(switch.id, 0, encode.id, "text", oi_switch)
    assert len(wf.links) == 1


# --- V3 meta types are not custom JS widgets ---------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "COMFY_MATCHTYPE_V3",
        "COMFY_AUTOGROW_V3",
        "COMFY_DYNAMICSLOT_V3",
        "COMFY_DYNAMICCOMBO_V3",
        "COMFY_MULTITYPED_V3",
    ],
)
def test_v3_meta_types_recognized(kind):
    assert w.is_v3_meta_type(kind)


@pytest.mark.parametrize(
    "kind", ["STRING", "IMAGE", "AUTOCOMPLETE_TEXT_LORAS", "COMBO", "", None, 3]
)
def test_non_meta_types_not_recognized(kind):
    assert not w.is_v3_meta_type(kind)


def test_match_type_input_is_not_a_widget_slot(oi_switch):
    """A ComfySwitchNode with both MatchType inputs unconnected serializes ONE
    widget value (`switch`). Counting the absent MatchType slots as custom JS
    widgets made the walk claim three, so every value after the first shifted."""
    slots = w.widget_slot_names(
        "ComfySwitchNode", oi_switch, [True], socket_names=set()
    )
    assert list(slots) == ["switch"]


def test_autogrow_marker_is_not_a_widget_slot():
    """`Autogrow.Input.get_all()` puts the marker AND the generated value slots in
    the schema; the node serializes only the value sockets it uses."""
    schema = {
        "input": {
            "required": {
                "delimiter": ["STRING", {"default": ", "}],
                "values": ["COMFY_AUTOGROW_V3", {"prefix": "value", "min": 1}],
                "value0": ["STRING", {"forceInput": True}],
                "value1": ["STRING", {"forceInput": True}],
            }
        },
        "output": ["STRING"],
    }
    oi = {"Concat": schema}
    slots = list(w.widget_slot_names("Concat", oi, ["; "], socket_names={"value0"}))
    assert slots == ["delimiter"]


def test_genuine_custom_js_widget_still_detected():
    """The V3 carve-out must not blind the pack-widget heuristic that round 17
    added - a bespoke pack type absent from the socket array is still a widget."""
    spec = ["AUTOCOMPLETE_TEXT_LORAS", {}]
    assert w._is_custom_widget("text", spec, socket_names=set())
    assert not w._is_custom_widget("text", spec, socket_names={"text"})


# --- subgraph provenance names a real remedy ---------------------------------


def test_flatten_provenance_carries_definition_and_inner_id(subgraph_doc, oi):
    wf = Workflow.from_ui(subgraph_doc)
    _flat, provenance, _diag = flatten(wf, oi)
    assert provenance
    entry = next(iter(provenance.values()))
    assert entry["definition"] in wf.subgraph_defs()
    assert entry["depth"] == 1
    assert entry["editable"] is True
    # inner_id is the node's id INSIDE the definition, not its flattened id
    inner_ids = {n["id"] for n in wf.subgraph_defs()[entry["definition"]]["nodes"]}
    assert all(p["inner_id"] in inner_ids for p in provenance.values())


def test_inner_finding_names_the_definition_op(subgraph_doc, oi):
    """An inner node's finding must carry the two ids an *_in_definition op takes,
    and must NOT tell the caller to rebuild flat."""
    wf = Workflow.from_ui(subgraph_doc)
    # break an inner widget: a model file that isn't installed
    def_id = next(iter(wf.subgraph_defs()))
    inner = wf.subgraph_as_workflow(def_id)
    unet = next(n for n in inner.nodes.values() if n.type == "UNETLoader")
    inner.set_widget(unet.id, "unet_name", "nope-not-installed.safetensors", oi)
    wf.update_subgraph(def_id, inner)

    findings = validate(wf, oi)
    bad = [f for f in findings if f.get("code") == "invalid-combo-value"]
    assert bad, "expected the fabricated model path to be caught"
    f = bad[0]
    assert f["definition_id"] == def_id
    assert f["inner_node_id"] == unet.id
    assert "rebuild flat" not in f["message"]


def test_subgraph_edit_hint_stated_once(subgraph_doc, oi):
    """The how-to is a per-RESULT sentence; the per-finding text stays a locator."""
    wf = Workflow.from_ui(subgraph_doc)
    def_id = next(iter(wf.subgraph_defs()))
    inner = wf.subgraph_as_workflow(def_id)
    unet = next(n for n in inner.nodes.values() if n.type == "UNETLoader")
    inner.set_widget(unet.id, "unet_name", "nope.safetensors", oi)
    wf.update_subgraph(def_id, inner)
    findings = validate(wf, oi)

    hint = server._subgraph_edit_hint(findings)
    assert "set_widget_in_definition" in hint["subgraph_edit"]
    # and never duplicated onto the findings themselves
    assert not any("set_widget_in_definition" in f["message"] for f in findings)


def test_no_subgraph_hint_on_a_flat_graph(oi):
    wf = Workflow.new()
    wf.add_node("CLIPTextEncode", object_info=oi)
    assert server._subgraph_edit_hint(validate(wf, oi)) == {}


def test_definition_op_actually_applies(subgraph_doc, oi):
    """The remedy the message now names has to work end to end."""
    wf = Workflow.from_ui(subgraph_doc)
    def_id = next(iter(wf.subgraph_defs()))
    inner = wf.subgraph_as_workflow(def_id)
    unet = next(n for n in inner.nodes.values() if n.type == "UNETLoader")
    installed = w.combo_choices(
        oi["UNETLoader"]["input"]["required"]["unet_name"]
    )[0]
    inner.set_widget(unet.id, "unet_name", installed, oi)
    wf.update_subgraph(def_id, inner)
    assert _errors(validate(wf, oi), "invalid-combo-value") == []


# --- inspect distinguishes exposed from internal boundary inputs -------------


def test_internal_subgraph_inputs_are_marked(subgraph_doc):
    """The definition declares six boundary inputs; the instance exposes `text`.
    Reporting all six unqualified sent a session chasing a socket that isn't
    there."""
    wf = Workflow.from_ui(subgraph_doc)
    sg = next(iter(wf.subgraph_defs().values()))
    summary = server._subgraph_summary(sg, wf)
    assert "text" in summary["inputs"]
    assert "width (internal)" in summary["inputs"]
    internal = [i for i in summary["inputs"] if i.endswith("(internal)")]
    assert len(internal) == len(summary["inputs"]) - 1


def test_subgraph_summary_without_workflow_is_unqualified(subgraph_doc):
    """Omitting the parent means we can't know exposure - so claim nothing."""
    wf = Workflow.from_ui(subgraph_doc)
    sg = next(iter(wf.subgraph_defs().values()))
    assert not any(
        i.endswith("(internal)") for i in server._subgraph_summary(sg)["inputs"]
    )


async def test_inspect_note_no_longer_denies_definition_edits(
    subgraph_doc, tmp_path, config, monkeypatch
):
    """inspect_workflow carried the same false claim as validate's finding tail,
    and it is the tool an agent reads BEFORE deciding how to fix a template."""
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "session", session)
    wf_id = session.create(Workflow.from_ui(subgraph_doc), title="t")

    result = await server.inspect_workflow(wf_id)
    note = result["subgraph_note"]
    assert "set_widget_in_definition" in note
    assert "edit_workflow ops don't reach inside" not in note
    assert "width (internal)" in result["subgraphs"][0]["inputs"]


# --- lint: a tapped Show Text is a preview -----------------------------------


def _generator_schema():
    return {
        "input": {"required": {"prompt": ["STRING", {"multiline": True}]}},
        "output": ["STRING"],
        "output_name": ["response"],
        "name": "TextGen",
    }


def _showtext_schema():
    return {
        "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
        # ShowText|pys passes the text straight through, which is what makes the
        # inline placement (generator -> ShowText -> encoder) possible at all
        "output": ["STRING"],
        "output_name": ["STRING"],
        "output_node": True,
        "name": "ShowText|pys",
    }


@pytest.fixture
def oi_textgen(oi):
    return {**oi, "TextGen": _generator_schema(), "ShowText|pys": _showtext_schema()}


def _generated_prompt_graph(oi_textgen, *, with_display: str | None):
    """positive prompt generated by TextGen; `with_display` places the Show Text
    inline (in the chain) or tapped (a sibling off the generator's output)."""
    wf = Workflow.new()
    gen = wf.add_node("TextGen", object_info=oi_textgen)
    encode = wf.add_node("CLIPTextEncode", object_info=oi_textgen)
    ksampler = wf.add_node("KSampler", object_info=oi_textgen)
    wf.connect(encode.id, 0, ksampler.id, "positive", oi_textgen)
    if with_display == "inline":
        show = wf.add_node("ShowText|pys", object_info=oi_textgen)
        wf.connect(gen.id, 0, show.id, "text", oi_textgen)
        wf.connect(show.id, 0, encode.id, "text", oi_textgen, force=True)
    else:
        wf.connect(gen.id, 0, encode.id, "text", oi_textgen)
        if with_display == "tapped":
            show = wf.add_node("ShowText|pys", object_info=oi_textgen)
            wf.connect(gen.id, 0, show.id, "text", oi_textgen)
    return wf


def _preview_codes(wf, oi_textgen):
    return [f["code"] for f in lint(wf, oi_textgen) if f["code"] == "no-prompt-preview"]


def test_tapped_show_text_counts_as_a_preview(oi_textgen):
    """generator -> ShowText alongside generator -> encoder displays the identical
    string. The chain walk alone never saw it, so a correct graph was told to add
    a node it already had."""
    wf = _generated_prompt_graph(oi_textgen, with_display="tapped")
    assert _preview_codes(wf, oi_textgen) == []


def test_inline_show_text_still_counts(oi_textgen):
    wf = _generated_prompt_graph(oi_textgen, with_display="inline")
    assert _preview_codes(wf, oi_textgen) == []


def test_generated_prompt_with_no_display_still_flagged(oi_textgen):
    """The rule still has to fire - this is the case it exists for."""
    wf = _generated_prompt_graph(oi_textgen, with_display=None)
    assert _preview_codes(wf, oi_textgen) == ["no-prompt-preview"]


# --- list_templates is honestly bounded --------------------------------------


class _FakeTemplateClient:
    def __init__(self, count):
        self._index = [
            {
                "title": "Image",
                "templates": [
                    {
                        "name": f"tpl_{i}",
                        "title": f"Template {i}",
                        "description": "d" * 400,
                        "models": [f"model_{i}.safetensors"],
                    }
                    for i in range(count)
                ],
            }
        ]

    async def get_template_index(self):
        return self._index


@pytest.fixture
def many_templates(monkeypatch):
    client = _FakeTemplateClient(452)
    monkeypatch.setattr(server, "_client", lambda: client)
    return client


async def test_list_templates_reports_true_count(many_templates):
    """It used to return `out[:60]` bare - a caller who saw no match reasonably
    concluded the catalog had none, with ~390 hidden."""
    result = await server.list_templates()
    assert result["count"] == 452
    assert len(result["templates"]) == server._TEMPLATES_CAP
    assert "search=" in result["hint"]


async def test_list_templates_clips_descriptions(many_templates):
    result = await server.list_templates()
    assert all(
        len(t["description"]) <= server._TEMPLATE_DESC_CAP
        for t in result["templates"]
    )


async def test_list_templates_no_hint_when_complete(monkeypatch):
    monkeypatch.setattr(server, "_client", lambda: _FakeTemplateClient(3))
    result = await server.list_templates()
    assert result["count"] == 3
    assert "hint" not in result


async def test_list_templates_search_matches_beyond_the_clip(monkeypatch):
    """Search runs over the FULL record: a model name, or detail past the
    description clip, is exactly what a caller searches for."""
    monkeypatch.setattr(server, "_client", lambda: _FakeTemplateClient(452))
    result = await server.list_templates(search="model_7.safetensors")
    assert result["count"] == 1
    assert result["templates"][0]["name"] == "tpl_7"
