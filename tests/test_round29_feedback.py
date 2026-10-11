"""0.20: positional widget-mapping plausibility (pack JS reorders/inserts
widgets), set_widget by index, organize staging/knobs/sizes, and learned-note
history - from a live Krea2 build session."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from comfy_draftsman import server
from comfy_draftsman.config import Config
from comfy_draftsman.graph import annotate as an
from comfy_draftsman.graph import widgets as w
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.graph.validate import validate
from comfy_draftsman.knowledge import get_guidance, save_learning
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOM = "custom_nodes.some_pack"

# --- synthetic pack schemas ---------------------------------------------------

PACK_OI = {
    # Identity Forge shape: prompt builder filed under conditioning/, STRING out
    "CharBuilder": {
        "input": {"required": {"age": ["INT", {"default": 30}], "style": [["calm", "wild"], {}]}},
        "output": ["STRING"],
        "category": "conditioning/character",
        "python_module": CUSTOM,
    },
    # rgthree Seed shape: buttons serialize as trailing nulls
    "SeedPack": {
        "input": {"required": {"seed": ["INT", {"min": 0, "max": 2**53}]}},
        "output": ["INT"],
        "category": "utils",
        "python_module": CUSTOM,
    },
    # LoRA Manager loader shape: bespoke widget-backed sockets
    "LoraPackLoader": {
        "input": {
            "required": {"model": ["MODEL", {}], "text": ["AUTOCOMPLETE_TEXT_LORAS", {}]},
            "optional": {"loras": ["LORAS", {}]},
        },
        "output": ["MODEL"],
        "category": "Lora Manager/loaders",
        "python_module": CUSTOM,
    },
}


def _node(nid, type_, values, inputs=(), outputs=(), size=None):
    node = {
        "id": nid,
        "type": type_,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "widgets_values": values,
    }
    if size:
        node["size"] = size
    return node


def _wf(*nodes, links=()):
    return Workflow.from_ui({"nodes": list(nodes), "links": list(links)})


# --- P1-B: plausibility + headless split --------------------------------------


def test_misaligned_pack_node_is_one_unmapped_finding():
    # a header widget's null hole pushed every value one slot right
    wf = _wf(_node(1, "CharBuilder", [None, 30, "calm"]))
    findings = validate(wf, PACK_OI)
    codes = [f["code"] for f in findings if f.get("node_id") == 1]
    assert codes.count("widget-layout-unmapped") == 1
    assert not {"null-widget-value", "invalid-combo-value", "widget-count-drift"} & set(codes)
    unmapped = next(f for f in findings if f["code"] == "widget-layout-unmapped")
    assert unmapped["level"] == "error" and unmapped["headless"] is True


def test_trailing_button_nulls_are_not_errors():
    wf = _wf(_node(1, "SeedPack", [5, None, None, None]))
    findings = validate(wf, PACK_OI)
    assert not [f for f in findings if f["level"] == "error"]
    drift = [f for f in findings if f["code"] == "widget-count-drift"]
    assert drift and drift[0]["level"] == "info"


def test_core_null_is_not_called_misalignment():
    oi = {"CoreThing": {**PACK_OI["CharBuilder"], "python_module": "nodes"}}
    wf = _wf(_node(1, "CoreThing", [None, "calm", "extra"]))
    codes = {f["code"] for f in validate(wf, oi)}
    assert "null-widget-value" in codes and "widget-layout-unmapped" not in codes


def test_uninstalled_model_is_not_misalignment():
    oi = {
        "PackLoader": {
            "input": {"required": {"ckpt": [["a.safetensors"], {}]}},
            "output": ["MODEL"],
            "category": "loaders",
            "python_module": CUSTOM,
        }
    }
    assert w.positional_mapping_plausible("PackLoader", ["missing.safetensors", "state"], oi)


def test_js_widget_input_is_headless():
    wf = _wf(
        _node(
            1,
            "LoraPackLoader",
            [{"meta": 1}, "<lora:x:1>", [{"name": "x", "active": True}]],
            inputs=[
                {"name": "model", "type": "MODEL", "link": None},
                {"name": "text", "type": "AUTOCOMPLETE_TEXT_LORAS", "link": None, "widget": {"name": "text"}},
            ],
        )
    )
    js = [f for f in validate(wf, PACK_OI) if f["code"] == "js-widget-input"]
    assert js and js[0]["headless"] is True


@pytest.fixture
def env(monkeypatch, tmp_path):
    saved = {}

    class StubClient:
        async def get_object_info(self, refresh=False):
            return PACK_OI

        async def save_userdata_workflow(self, name, document, overwrite=False):
            saved[name] = document
            return f"{name}.json"

    config = Config(
        comfyui_url="http://comfy.test", session_dir=tmp_path / "s", learned_dir=tmp_path / "learned"
    )
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", StubClient())
    monkeypatch.setattr(server._State, "session", Session(config.session_dir))
    return saved


async def _import(nodes, links=()):
    ui = {"nodes": nodes, "links": list(links)}
    return (await server.import_workflow(workflow_json=json.dumps(ui)))["workflow_id"]


async def test_save_allows_headless_only_errors(env):
    wf_id = await _import([_node(1, "CharBuilder", [None, 30, "calm"])])
    result = await server.save_workflow(wf_id, "forge")
    assert result["saved"] is True, result
    assert "run_workflow will refuse" in result["note"]


# --- P1-A: set_widget by index, safe by-name sets -----------------------------

_LORA_NODE = _node(
    1,
    "LoraPackLoader",
    [{"meta": 1}, "<lora:x:1>", [{"name": "x", "active": True}]],
    inputs=[
        {"name": "model", "type": "MODEL", "link": None},
        {"name": "text", "type": "AUTOCOMPLETE_TEXT_LORAS", "link": None, "widget": {"name": "text"}},
        {"name": "loras", "type": "LORAS", "link": None, "widget": {"name": "loras"}},
    ],
)


async def test_set_widget_by_index_writes_raw(env):
    wf_id = await _import([copy.deepcopy(_LORA_NODE)])
    loras = [{"name": "y", "active": True, "strength": 0.6}]
    result = await server.edit_workflow(
        wf_id, [{"op": "set_widget", "node_id": 1, "index": 2, "value": loras}]
    )
    assert "error" not in result, result
    assert "raw, unchecked" in result["applied"][0]
    assert server._wf(wf_id).nodes[1].widgets_values[2] == loras


async def test_set_widget_by_name_on_js_widget_hints_index(env):
    wf_id = await _import([copy.deepcopy(_LORA_NODE)])
    result = await server.edit_workflow(
        wf_id, [{"op": "set_widget", "node_id": 1, "input": "loras", "value": []}]
    )
    assert '"index"' in result["error"]


@pytest.mark.parametrize(
    "op",
    [
        {"op": "set_widget", "node_id": 1, "input": "text", "index": 1, "value": "x"},
        {"op": "set_widget", "node_id": 1, "value": "x"},
        {"op": "set_widget", "node_id": 1, "index": 9, "value": "x"},
    ],
)
async def test_set_widget_index_misuse_is_refused(env, op):
    wf_id = await _import([copy.deepcopy(_LORA_NODE)])
    assert "error" in await server.edit_workflow(wf_id, [op])


def test_by_name_set_refused_on_misaligned_node():
    wf = _wf(_node(1, "CharBuilder", [None, 30, "calm"]))
    with pytest.raises(ValueError, match="index"):
        wf.set_widget(1, "age", 40, PACK_OI)


def test_by_name_set_keeps_pack_state_past_the_schema():
    wf = _wf(_node(1, "SeedPack", [5, None, None, None]))
    wf.set_widget(1, "seed", 7, PACK_OI)
    assert wf.nodes[1].widgets_values == [7, None, None, None]


# --- P1-C: staging --------------------------------------------------------------


def test_prompt_builder_filed_under_conditioning_is_prompt_build():
    node = _wf(_node(1, "CharBuilder", [30, "calm"])).nodes[1]
    assert an.classify(node, PACK_OI) == "prompt_build"


def test_conditioning_output_decides_when_category_does_not():
    oi = {"Enc": {"input": {"required": {}}, "output": ["CONDITIONING"], "category": "pack/x"}}
    assert an.classify(_wf(_node(1, "Enc", [])).nodes[1], oi) == "conditioning"


def _chain_oi():
    return {
        "Pool": {"input": {"required": {}}, "output": ["POOL"], "category": "pack"},
        "Picker": {"input": {"required": {"pool": ["POOL", {}]}}, "output": ["PICK"], "category": "pack"},
        "Loader": {
            "input": {"required": {"pick": ["PICK", {}], "name": [["a.safetensors"], {}]}},
            "output": ["MODEL"],
            "category": "loaders",
        },
        "Lonely": {"input": {"required": {}}, "output": ["THING"], "category": "pack"},
    }


def _link(lid, src, dst, type_):
    return [lid, src, 0, dst, 0, type_]


def test_undecided_nodes_follow_their_consumers():
    oi = _chain_oi()
    wf = _wf(
        _node(1, "Pool", [], outputs=[{"name": "POOL", "type": "POOL", "links": [1]}]),
        _node(
            2,
            "Picker",
            [],
            inputs=[{"name": "pool", "type": "POOL", "link": 1}],
            outputs=[{"name": "PICK", "type": "PICK", "links": [2]}],
        ),
        _node(3, "Loader", ["a.safetensors"], inputs=[{"name": "pick", "type": "PICK", "link": 2}]),
        _node(4, "Lonely", []),
        links=[_link(1, 1, 2, "POOL"), _link(2, 2, 3, "PICK")],
    )
    stage_of_key, undecided = {}, set()
    for node in wf.nodes.values():
        key = an.classify(node, oi, default="")
        if not key:
            undecided.add(node.id)
        stage_of_key[node.id] = key or "sampling"
    an._restage_by_graph(wf, oi, stage_of_key, undecided)
    assert stage_of_key == {1: "models", 2: "models", 3: "models", 4: "sampling"}
    # the public contract is unchanged
    assert an.classify(wf.nodes[1], oi) == "sampling"


# --- P1-D: sizes ------------------------------------------------------------------


def test_organize_keeps_a_taller_saved_size():
    oi = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    wf = _wf(
        _node(1, "LoadImage", ["a.png", "image"], size=[315, 760]),
        _node(2, "EmptyLatentImage", [1024, 1024, 1]),
    )
    an.annotate(wf, oi)
    assert wf.nodes[1].size[1] >= 760
    a, b = wf.nodes[1], wf.nodes[2]
    overlap_x = a.pos[0] < b.pos[0] + b.size[0] and b.pos[0] < a.pos[0] + a.size[0]
    overlap_y = a.pos[1] < b.pos[1] + b.size[1] and b.pos[1] < a.pos[1] + a.size[1]
    assert not (overlap_x and overlap_y)


# --- P1-E: knobs ------------------------------------------------------------------


@pytest.fixture
def oi():
    return json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))


def _sampler_graph(cfg):
    return _wf(
        _node(
            1,
            "CLIPTextEncode",
            ["blurry"],
            inputs=[{"name": "clip", "type": "CLIP", "link": None}],
            outputs=[{"name": "CONDITIONING", "type": "CONDITIONING", "links": [1]}],
        ),
        _node(
            2,
            "KSampler",
            [1, "fixed", 8, cfg, "euler", "normal", 1.0],
            inputs=[
                {"name": "model", "type": "MODEL", "link": None},
                {"name": "positive", "type": "CONDITIONING", "link": None},
                {"name": "negative", "type": "CONDITIONING", "link": 1},
                {"name": "latent_image", "type": "LATENT", "link": None},
            ],
            outputs=[{"name": "LATENT", "type": "LATENT", "links": []}],
        ),
        links=[[1, 1, 0, 2, 2, "CONDITIONING"]],
    )


def test_negative_at_cfg_one_is_not_a_knob(oi):
    wf = _sampler_graph(1.0)
    an._title_nodes(wf, oi)
    an._paint_knobs(wf, oi, {1: "inputs", 2: "sampling"})
    assert wf.nodes[1].title == an.INERT_NEGATIVE_TITLE
    assert (wf.nodes[1].color, wf.nodes[1].bgcolor) != an.GREEN


def test_negative_with_real_cfg_stays_a_knob(oi):
    wf = _sampler_graph(4.0)
    an._title_nodes(wf, oi)
    an._paint_knobs(wf, oi, {1: "inputs", 2: "sampling"})
    assert wf.nodes[1].title == "🚫 Negative Prompt"
    assert (wf.nodes[1].color, wf.nodes[1].bgcolor) == an.GREEN


def test_orphans_and_wired_canvas_are_not_knobs(oi):
    wf = _wf(
        _node(1, "LoadImage", ["a.png", "image"]),
        _node(
            2,
            "EmptyLatentImage",
            [1024, 1024, 1],
            inputs=[
                {"name": "width", "type": "INT", "link": 7, "widget": {"name": "width"}},
                {"name": "height", "type": "INT", "link": 8, "widget": {"name": "height"}},
            ],
            outputs=[{"name": "LATENT", "type": "LATENT", "links": [9]}],
        ),
    )
    an._paint_knobs(wf, oi, {1: "inputs", 2: "inputs"})
    assert all((n.color, n.bgcolor) != an.GREEN for n in wf.nodes.values())


def test_model_pickers_and_prompt_builders_are_knobs():
    oi = {**PACK_OI, **_chain_oi()}
    wf = _wf(
        _node(1, "Loader", ["a.safetensors"], outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}]),
        _node(2, "CharBuilder", [30, "calm"], outputs=[{"name": "STRING", "type": "STRING", "links": [2]}]),
    )
    an._paint_knobs(wf, oi, {1: "models", 2: "prompt_build"})
    assert all((n.color, n.bgcolor) == an.GREEN for n in wf.nodes.values())


def test_model_patch_without_a_file_is_not_a_knob():
    oi = {
        "Patch": {
            "input": {"required": {"model": ["MODEL", {}], "shift": ["FLOAT", {}]}},
            "output": ["MODEL"],
            "category": "advanced/model",
        }
    }
    wf = _wf(_node(1, "Patch", [1.15], outputs=[{"name": "MODEL", "type": "MODEL", "links": [1]}]))
    an._paint_knobs(wf, oi, {1: "models"})
    assert (wf.nodes[1].color, wf.nodes[1].bgcolor) != an.GREEN


# --- P1-F: the base sampler drives the steps note -----------------------------------


def _two_pass(oi, base_steps):
    refiner = _node(1, "KSampler", [1, "fixed", 4, 1.0, "euler", "normal", 0.3])
    base = _node(2, "KSampler", [1, "fixed", base_steps, 1.0, "euler", "normal", 1.0])
    wf = _wf(refiner, base)
    guidance = {"display_name": "Krea 2", "sampling": {"steps": {"min": 6, "max": 12, "default": 8}}}
    return an._note_text("sampling", wf, oi, guidance, [wf.nodes[1], wf.nodes[2]]) or ""


def test_refiner_steps_do_not_trigger_the_outside_warning(oi):
    assert "outside" not in _two_pass(oi, 10)
    assert "outside" in _two_pass(oi, 30)


# --- P1-G: learned history ------------------------------------------------------------


def test_overwritten_note_is_kept_as_superseded(tmp_path):
    save_learning(tmp_path, "flux", {"sampling": {"notes": "July lesson"}}, source="a")
    save_learning(tmp_path, "flux", {"sampling": {"notes": "July lesson"}}, source="same")
    path = save_learning(tmp_path, "flux", {"sampling": {"notes": "October study"}}, source="b")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["superseded"] == [
        {"date": doc["sources"][-1]["date"], "key": "sampling.notes", "value": "July lesson"}
    ]
    assert "superseded" not in get_guidance("flux", learned_dir=tmp_path)


async def test_record_learning_reports_what_it_replaced(env):
    await server.record_learning("flux", {"sampling": {"notes": "old"}}, source="a")
    result = await server.record_learning("flux", {"sampling": {"notes": "new"}}, source="b")
    assert result["replaced"] == ["sampling.notes"] and "superseded" in result["hint"]
    fresh = await server.record_learning("flux", {"sampling": {"other": 1}}, source="c")
    assert "replaced" not in fresh


def test_unwired_text_box_flagged_output_node_is_still_an_orphan():
    # Chibi's Textbox sets output_node on a STRING-only node for its preview
    oi = {
        "Textbox": {
            "input": {"required": {"text": ["STRING", {"multiline": True}]}},
            "output": ["STRING"],
            "output_node": True,
            "category": "text",
        }
    }
    wf = _wf(_node(1, "Textbox", ["notes"], outputs=[{"name": "text", "type": "STRING", "links": []}]))
    an._paint_knobs(wf, oi, {1: "prompt_build"})
    assert (wf.nodes[1].color, wf.nodes[1].bgcolor) != an.GREEN


def test_port_skips_a_misaligned_pack_node():
    from comfy_draftsman.graph import port

    wf = _wf(_node(1, "CharBuilder", [None, 30, "calm"]))
    changes: list[str] = []
    assert port._set_if_valid(wf, 1, "style", "wild", PACK_OI, changes) is True
    assert wf.nodes[1].widgets_values == [None, 30, "calm"] and not changes
