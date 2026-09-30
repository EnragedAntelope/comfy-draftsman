"""Round 27: fixes from the live Qwen Image 2.1 build (see CHANGELOG 0.17.0)."""

import json
from pathlib import Path

import pytest

from comfy_draftsman import knowledge
from comfy_draftsman.graph.model import Workflow

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def oi():
    return json.loads((FIXTURES / "object_info_qwen21.json").read_text(encoding="utf-8"))


@pytest.fixture()
def qwen_wf():
    data = json.loads((FIXTURES / "qwen21_candid_llm.json").read_text(encoding="utf-8"))
    return Workflow.from_ui(data)


# --- 1.1 qwen_image21 floor family ------------------------------------------------


def _unet_wf(oi, name):
    wf = Workflow.new()
    unet = wf.add_node("UNETLoader", object_info=oi)
    wf.set_widget(unet.id, "unet_name", name, oi)
    return wf


@pytest.mark.parametrize(
    "name",
    ["qwen_image_2.1_int8_convrot.safetensors", "qwenImage21_21_int8.safetensors"],
)
def test_qwen21_detects_its_own_family(oi, name):
    detail = knowledge.detect_family_detail(_unet_wf(oi, name), oi)
    assert detail["family"] == "qwen_image21"


def test_qwen1x_still_detects_as_qwen_image(oi):
    wf = _unet_wf(oi, "qwen_image_fp8_e4m3fn.safetensors")
    assert knowledge.detect_family(wf, oi) == "qwen_image"


def test_qwen21_floor_guidance_is_2x_not_1x():
    g = knowledge.get_guidance("qwen_image21")
    assert g["sampling"]["cfg"]["default"] == 1
    assert "Qwen3-VL" in g["notes"]["loaders"]
    assert "lora" not in g["techniques"]["viggle_turbo_6step"]  # no machine-local path
    assert "hardware" not in g  # never invent VRAM numbers
    assert all(s["url"].startswith("https://huggingface.co/") for s in g["sources"])


# --- 1.2 frontend-exact bypass routing ---------------------------------------------

_S = ["STRING", {"forceInput": True}]
BYPASS_OI = {
    "StrSrc": {"input": {"required": {}}, "output": ["STRING"], "output_name": ["STRING"]},
    "ImgSrc": {"input": {"required": {}}, "output": ["IMAGE"], "output_name": ["IMAGE"]},
    "Enh": {
        "input": {"required": {"img": ["IMAGE"], "prompt": _S, "sys": _S}},
        "output": ["STRING", "STRING"],
        "output_name": ["a", "b"],
    },
    "Pair": {
        "input": {"required": {"x": _S, "y": _S}},
        "output": ["STRING", "STRING"],
        "output_name": ["a", "b"],
    },
    "ImgOnly": {
        "input": {"required": {"img": ["IMAGE"]}},
        "output": ["STRING"],
        "output_name": ["s"],
    },
    "Sink": {
        "input": {"required": {"text": _S}},
        "output": [],
        "output_name": [],
        "output_node": True,
    },
}


def _bypass_graph(kind, wire, out=0, mode=4):
    """StrSrc nodes feed the named inputs of a `kind` node (bypassed), which
    feeds Sink.text from output `out`. Returns (workflow, {input: source_id})."""
    wf = Workflow.new()
    node = wf.add_node(kind, object_info=BYPASS_OI)
    node.mode = mode
    srcs = {}
    for name in wire:
        cls = "ImgSrc" if name == "img" else "StrSrc"
        src = wf.add_node(cls, object_info=BYPASS_OI)
        wf.connect(src.id, 0, node.id, name, object_info=BYPASS_OI)
        srcs[name] = src.id
    sink = wf.add_node("Sink", object_info=BYPASS_OI)
    wf.connect(node.id, out, sink.id, "text", object_info=BYPASS_OI)
    return wf, srcs, sink.id


def _sink_text(wf):
    api = wf.to_api(BYPASS_OI)
    return next(v for v in api.values() if v["class_type"] == "Sink")["inputs"].get("text")


def test_bypass_routes_first_input_of_matching_type():
    wf, srcs, _ = _bypass_graph("Enh", ["img", "prompt", "sys"])
    assert _sink_text(wf) == [str(srcs["prompt"]), 0]


def test_bypass_chosen_input_unlinked_drops_the_input_no_fall_through():
    # the frontend picks `prompt` (first STRING) even though only `sys` is linked
    wf, _, _ = _bypass_graph("Enh", ["img", "sys"])
    assert _sink_text(wf) is None


def test_bypass_same_index_input_wins_for_the_output_slot():
    wf, srcs, _ = _bypass_graph("Pair", ["x", "y"], out=1)
    assert _sink_text(wf) == [str(srcs["y"]), 0]
    wf, srcs, _ = _bypass_graph("Pair", ["x", "y"], out=0)
    assert _sink_text(wf) == [str(srcs["x"]), 0]


def test_bypass_with_no_compatible_input_drops_the_input():
    wf, _, _ = _bypass_graph("ImgOnly", ["img"])
    assert _sink_text(wf) is None


def test_bypass_chain_and_reroute_pass_through():
    wf, srcs, sink_id = _bypass_graph("Pair", ["x"], out=0)
    second = wf.add_node("Pair", object_info=BYPASS_OI)
    second.mode = 4
    first = next(n for n in wf.nodes.values() if n.type == "Pair" and n.id != second.id)
    # rewire: Pair(first) -> Pair(second).x -> Sink
    sink = wf.nodes[sink_id]
    wf.links.pop(sink.inputs[0].link)
    sink.inputs[0].link = None
    wf.connect(first.id, 0, second.id, "x", object_info=BYPASS_OI)
    wf.connect(second.id, 0, sink.id, "text", object_info=BYPASS_OI)
    assert _sink_text(wf) == [str(srcs["x"]), 0]


def test_bypass_routing_matches_frontend_on_the_qwen_fixture(qwen_wf, oi):
    # LoRA (#34) is bypassed on MODEL->MODEL: its consumer must see the UNET side
    api = qwen_wf.to_api(oi)
    assert "34" not in api
    consumer = next(
        v for v in api.values() if v["class_type"] == "ModelSamplingFlux"
    )
    assert consumer["inputs"]["model"] == ["1", 0]


def test_lg_valid_matches_litegraph():
    from comfy_draftsman.graph.model import _lg_valid

    assert _lg_valid("*", "IMAGE") and _lg_valid("", "IMAGE")
    assert _lg_valid("image", "IMAGE")
    assert _lg_valid("IMAGE,LATENT", "LATENT")
    assert not _lg_valid("IMAGE", "STRING")
    # unlike types_compatible, MatchType is an ordinary type here
    assert not _lg_valid("COMFY_MATCHTYPE_V3", "IMAGE")


# --- 1.3 organize staging + band shape ---------------------------------------------


def _group_title_of(wf, nid):
    node = wf.nodes[nid]
    cx, cy = node.pos[0] + node.size[0] / 2, node.pos[1] + node.size[1] / 2
    return next(
        g.title
        for g in wf.groups
        if g.bounding[0] <= cx <= g.bounding[0] + g.bounding[2]
        and g.bounding[1] <= cy <= g.bounding[1] + g.bounding[3]
    )


def test_organize_stages_a_non_trivial_graph_sensibly(qwen_wf, oi):
    from comfy_draftsman.graph.annotate import annotate
    from comfy_draftsman.graph.lint import lint

    report = annotate(qwen_wf, oi)
    assert report["family"] == "qwen_image21"
    of = lambda nid: _group_title_of(qwen_wf, nid)  # noqa: E731
    # core primitives are hand-tweaked knobs
    for primitive in (8, 9, 11, 23, 28, 29, 35, 40, 41):
        assert "Inputs" in of(primitive), primitive
    # MODEL->MODEL patches sit with the loaders they continue
    for patch in (3, 4, 5):
        assert "Models" in of(patch), patch
    # the i2i reference scaler is preprocessing, beside its LoadImage
    assert of(25) == of(24)
    assert "Post" not in of(25)
    # switches follow what they feed, never the schema-less "sampling" default
    assert "Prompt" in of(38) and "Prompt" in of(39)
    assert "Sampling" in of(27) and "Sampling" in of(30)
    xs = [n.pos[0] for n in qwen_wf.nodes.values()]
    ys = [n.pos[1] for n in qwen_wf.nodes.values()]
    width = max(n.pos[0] + n.size[0] for n in qwen_wf.nodes.values()) - min(xs)
    height = max(n.pos[1] + n.size[1] for n in qwen_wf.nodes.values()) - min(ys)
    assert width <= 2.5 * height, f"canvas {width:.0f}x{height:.0f} is a strip"
    assert not [f for f in lint(qwen_wf, oi) if f["code"] == "overlap"]


def test_pure_post_processing_graph_stays_post(oi):
    """No sampling stage -> the upstream-post rule must not fire."""
    from comfy_draftsman.graph.annotate import annotate

    wf = Workflow.new()
    load = wf.add_node("LoadImage", object_info=oi)
    up = wf.add_node("ImageScaleToTotalPixels", object_info=oi)
    save = wf.add_node("SaveImage", object_info=oi)
    wf.connect(load.id, 0, up.id, "image", object_info=oi)
    wf.connect(up.id, 0, save.id, "images", object_info=oi)
    annotate(wf, oi)
    assert "Post" in _group_title_of(wf, up.id)


# --- 1.4 honest sampling note ------------------------------------------------------


def _notes(wf):
    return " ".join(str(n.widgets_values[0]) for n in wf.nodes.values() if n.type == "MarkdownNote")


def test_sampling_note_in_range_is_a_reference_not_a_recommendation(qwen_wf, oi):
    from comfy_draftsman.graph.annotate import annotate

    annotate(qwen_wf, oi)
    text = _notes(qwen_wf)
    assert "Family reference: CFG 1-2, steps 16-40." in text
    assert "Tuned for" not in text and "leave these alone" not in text


def test_sampling_note_omits_base_numbers_when_graph_is_outside_them(qwen_wf, oi):
    from comfy_draftsman.graph.annotate import annotate

    qwen_wf.set_widget(18, "steps", 6, oi)  # a turbo-LoRA style 6-step graph
    annotate(qwen_wf, oi)
    text = _notes(qwen_wf)
    assert "steps=6" in text and "outside" in text and "acceleration" in text
    assert "Family reference" not in text
    assert "20 steps is enough" not in text  # the base-model sampling prose


# --- 1.5 prompt-preview lint scope + save note -------------------------------------


def test_llm_node_with_prompt_input_is_not_flagged_as_an_encoder(qwen_wf, oi):
    from comfy_draftsman.graph.lint import lint

    codes = [(f["code"], f.get("node_id")) for f in lint(qwen_wf, oi)]
    assert not [c for c in codes if c[0] == "no-prompt-preview" and c[1] == 12]


def test_real_encoder_fed_by_an_unpreviewed_generator_is_still_flagged(qwen_wf, oi):
    from comfy_draftsman.graph.lint import lint

    qwen_wf.remove_node(13)  # the Show Text sitting between the LLM and the encoder
    qwen_wf.connect(12, 0, 14, "prompt", object_info=oi)
    flagged = [f for f in lint(qwen_wf, oi) if f["code"] == "no-prompt-preview"]
    assert [f["node_id"] for f in flagged] == [14]


async def test_save_note_only_recommends_organize_for_layout_lint(qwen_wf, oi, tmp_path, config, monkeypatch):
    from comfy_draftsman import server
    from comfy_draftsman.graph.annotate import annotate
    from comfy_draftsman.session import Session

    class Client:
        async def get_object_info(self, refresh=False):
            return oi

        async def save_userdata_workflow(self, name, document, overwrite=False):
            return name if name.endswith(".json") else f"{name}.json"

    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "client", Client())
    monkeypatch.setattr(server._State, "session", session)
    annotate(qwen_wf, oi)
    wf_id = session.create(qwen_wf, title="t")
    clean = await server.save_workflow(wf_id, "a", allow_invalid=True)
    assert clean["saved"] is True and "organize_workflow" not in clean["note"]
    qwen_wf.groups = []  # now the layout lint fires
    dirty = await server.save_workflow(wf_id, "b", allow_invalid=True)
    assert "organize_workflow" in dirty["note"]


# --- 1.6 output node behind a lazy input -------------------------------------------


def test_output_node_feeding_a_lazy_switch_input_is_flagged(oi):
    from comfy_draftsman.graph.lint import lint

    wf = Workflow.new()
    llm_a = wf.add_node("EA_LMStudio", object_info=oi)
    llm_b = wf.add_node("EA_LMStudio", object_info=oi)
    sw = wf.add_node("ComfySwitchNode", object_info=oi)
    show = wf.add_node("ShowText|pysssss", object_info=oi)
    wf.connect(llm_a.id, 0, sw.id, "on_false", object_info=oi)
    wf.connect(llm_b.id, 0, sw.id, "on_true", object_info=oi)
    wf.connect(sw.id, 0, show.id, "text", object_info=oi)
    hits = [f for f in lint(wf, oi) if f["code"] == "output-behind-lazy-input"]
    assert sorted(f["node_id"] for f in hits) == sorted([llm_a.id, llm_b.id])
    assert "every output node" in hits[0]["message"]


def test_non_output_producer_behind_a_switch_is_fine(oi):
    from comfy_draftsman.graph.lint import lint

    wf = Workflow.new()
    a = wf.add_node("PrimitiveString", object_info=oi)
    sw = wf.add_node("ComfySwitchNode", object_info=oi)
    wf.connect(a.id, 0, sw.id, "on_false", object_info=oi)
    assert not [f for f in lint(wf, oi) if f["code"] == "output-behind-lazy-input"]


# --- 2.x friction fixes ------------------------------------------------------------


@pytest.fixture()
def server_state(tmp_path, config, monkeypatch):
    import dataclasses

    from comfy_draftsman import server
    from comfy_draftsman.session import Session

    # learned_dir defaults to the REAL ~/.comfy-draftsman/learned - never write there
    config = dataclasses.replace(config, learned_dir=tmp_path / "learned")
    monkeypatch.setattr(server._State, "config", config)
    monkeypatch.setattr(server._State, "session", Session(tmp_path / "sessions"))
    return server


async def test_model_guidance_detects_family_from_filename_alone(server_state):
    out = await server_state.get_model_guidance(model_filename="qwen_image_2.1_bf16.safetensors")
    assert out["detected_family"] == "qwen_image21"
    assert out["sampling"]["cfg"]["default"] == 1
    miss = await server_state.get_model_guidance(model_filename="totally_unknown_thing.safetensors")
    assert "families" in miss and "totally_unknown_thing" in miss["hint"]


def test_elapsed_s_from_history_messages():
    from comfy_draftsman.comfy.client import ComfyClient

    history = {
        "status": {
            "messages": [
                ["execution_start", {"prompt_id": "p", "timestamp": 1_000}],
                ["execution_cached", {"nodes": [], "timestamp": 1_010}],
                ["execution_success", {"prompt_id": "p", "timestamp": 17_240}],
            ]
        }
    }
    assert ComfyClient._elapsed_s(history) == 16.2
    assert ComfyClient._elapsed_s({}) is None
    assert ComfyClient._elapsed_s({"status": {"messages": [["execution_success", {"timestamp": 5}]]}}) is None


def test_full_output_returns_one_node_unclipped():
    from comfy_draftsman.comfy.client import ComfyClient

    long = "x" * 3000
    history = {"outputs": {"12": {"text": [long]}, "13": {"text": ["short"]}}}
    clipped = ComfyClient._collect_data_outputs(history)
    assert len(clipped["12"]["text"]) < 3000 and "note" in clipped
    full = ComfyClient._collect_data_outputs(history, node_id="12", budget=50_000)
    assert full == {"12": {"text": [long]}}
    capped = ComfyClient._collect_data_outputs(
        {"outputs": {"12": {"text": "y" * 60_000}}}, node_id="12", budget=50_000
    )
    assert len(capped["12"]["text"]) == 50_001 and "note" in capped


async def test_replace_in_widget_edits_one_sentence(oi, server_state):
    from comfy_draftsman import server

    class Client:
        async def get_object_info(self, refresh=False):
            return oi

    server._State.client = Client()
    try:
        wf = Workflow.new()
        node = wf.add_node("PrimitiveStringMultiline", object_info=oi)
        wf.set_widget(node.id, "value", "Be terse. Never use lists. Be terse again.", oi)
        wf_id = server._session().create(wf, title="t")
        out = await server.edit_workflow(
            wf_id, [{"op": "replace_in_widget", "node_id": node.id, "input": "value",
                     "old": "Never use lists.", "new": "Use lists."}]
        )
        assert "Use lists." in str(wf.nodes[node.id].widgets_values)
        assert "Never use lists." not in str(wf.nodes[node.id].widgets_values)
        assert out["applied"]
        for old, needle in (("Be terse", "matches 2 times"), ("nope", "matches 0 times")):
            bad = await server.edit_workflow(
                wf_id, [{"op": "replace_in_widget", "node_id": node.id, "input": "value",
                         "old": old, "new": "x"}]
            )
            assert needle in bad["error"]
    finally:
        server._State.client = None


async def test_set_widget_echo_is_clipped(oi, server_state):
    from comfy_draftsman import server

    class Client:
        async def get_object_info(self, refresh=False):
            return oi

    server._State.client = Client()
    try:
        wf = Workflow.new()
        node = wf.add_node("PrimitiveStringMultiline", object_info=oi)
        wf_id = server._session().create(wf, title="t")
        out = await server.edit_workflow(
            wf_id, [{"op": "set_widget", "node_id": node.id, "input": "value", "value": "z" * 5000}]
        )
        assert sum(len(a) for a in out["applied"]) < 300
    finally:
        server._State.client = None


async def test_record_learning_returns_paths_not_the_merged_guidance(server_state):
    out = await server_state.record_learning(
        "zz_test_family", {"sampling": {"cfg": {"default": 3}}, "notes": {"x": "y"}}, "test"
    )
    assert out["updated"] == ["sampling.cfg.default", "notes.x"]
    assert "guidance_now" not in out and out["saved"].endswith(".yaml")


def test_choices_filter_can_target_one_combo():
    from comfy_draftsman.comfy.catalog import node_summary

    oi = {
        "N": {
            "input": {
                "required": {
                    "sampler_name": [["euler", "res_2s", "dpmpp"], {}],
                    "scheduler": [["simple", "beta", "karras"], {}],
                }
            },
            "output": [],
        }
    }
    by = {i["name"]: i for i in node_summary(oi, "N", choices_filter="sampler_name:res")["inputs"]}
    assert by["sampler_name"]["choices"] == ["res_2s"]
    assert by["scheduler"]["choices"] == ["simple", "beta", "karras"]
    both = {i["name"]: i for i in node_summary(oi, "N", choices_filter="e")["inputs"]}
    assert both["scheduler"]["choices"] == ["simple", "beta"]  # old behavior: every combo


# --- 2.8 elicitation honesty -------------------------------------------------------


class _Ctx:
    def __init__(self, action=None, can=True, confirm=True):
        self.calls = 0
        self._action, self._confirm = action, confirm
        session = type("S", (), {})()
        session.check_client_capability = lambda cap: can
        self.session = session

    async def elicit(self, message, schema):
        self.calls += 1
        data = type("D", (), {"confirm": self._confirm})() if self._action == "accept" else None
        return type("R", (), {"action": self._action, "data": data})()


async def test_confirm_maps_each_client_answer_honestly(server_state):
    c = server_state._confirm
    assert await c(_Ctx("accept"), "m", "hint", "x") is None
    declined = await c(_Ctx("decline"), "m", "hint", "x")
    assert declined["status"] == "x_declined" and declined["client_action"] == "decline"
    cancelled = await c(_Ctx("cancel"), "m", "hint", "x")
    assert cancelled["status"] == "x_not_confirmed" and "NOT refused" in cancelled["hint"]


async def test_confirm_never_elicits_without_the_capability_or_when_off(server_state, monkeypatch):
    import dataclasses

    c = server_state._confirm
    no_cap = _Ctx("accept", can=False)
    assert (await c(no_cap, "m", "hint", "x"))["status"] == "x_confirmation_required"
    assert no_cap.calls == 0
    assert await c(_Ctx("accept", can=False), "m", None, "x") is None  # explicit-flag callers proceed
    cfg = dataclasses.replace(server_state._State.config, elicitation=False)
    monkeypatch.setattr(server_state._State, "config", cfg)
    off = _Ctx("accept")
    assert (await c(off, "m", "hint", "x"))["status"] == "x_confirmation_required"
    assert off.calls == 0


# --- 4 sweep mode ------------------------------------------------------------------

SWEEP_OI = {
    "Gen": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0, "min": 0, "max": 2**32}],
                "steps": ["INT", {"default": 10, "min": 1, "max": 100}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "output_node": True,
    }
}


def _png(color):
    import io

    from PIL import Image

    img = Image.new("RGB", (1024, 768))
    px = img.load()
    for x in range(1024):
        for y in range(768):
            px[x, y] = ((x + color) % 256, y % 256, color % 256)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class SweepClient:
    def __init__(self):
        self.apis = []

    async def get_object_info(self, refresh=False):
        return SWEEP_OI

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        self.apis.append(api)
        n = len(self.apis)
        return {
            "status": "success", "prompt_id": f"p{n}", "elapsed_s": 1.5,
            "outputs": [{"filename": f"{n}.png", "subfolder": "", "type": "output", "kind": "images"}],
        }

    async def fetch_output(self, item):
        return _png(int(item["filename"].split(".")[0]) * 40)


@pytest.fixture()
def sweep_env(server_state, tmp_path):
    client = SweepClient()
    server_state._State.client = client
    wf = Workflow.new()
    node = wf.add_node("Gen", object_info=SWEEP_OI)
    wf_id = server_state._session().create(wf, title="t")
    yield server_state, client, wf, node, wf_id, tmp_path / "out"
    server_state._State.client = None


def _variants():
    return [
        {"label": "base", "ops": []},
        {"label": "fast", "ops": [{"op": "set_widget", "node_id": 1, "input": "steps", "value": 4}]},
    ]


async def test_sweep_runs_variants_x_seeds_and_builds_sheets(sweep_env):
    import io

    from PIL import Image

    server, client, wf, node, wf_id, out = sweep_env
    result = await server.run_workflow(
        wf_id, front=True, save_dir=str(out),
        sweep={"variants": _variants(), "seeds": [7, 8], "crops": [[10, 20, 110, 120]]},
    )
    body, _thumb, *rest = result
    assert [(r["label"], r["seed"], r["status"], r["elapsed_s"]) for r in body["runs"]] == [
        ("base", 7, "success", 1.5), ("base", 8, "success", 1.5),
        ("fast", 7, "success", 1.5), ("fast", 8, "success", 1.5),
    ]
    assert [a["1"]["inputs"]["seed"] for a in client.apis] == [7, 8, 7, 8]
    assert [a["1"]["inputs"]["steps"] for a in client.apis] == [10, 10, 4, 4]
    assert 10 in wf.nodes[node.id].widgets_values and 4 not in wf.nodes[node.id].widgets_values  # session wf untouched
    assert Path(body["contact_sheet"]).is_file() and Path(body["crop_sheet"]).is_file()
    # the first crop tile is the source region, pixel for pixel (1:1, never scaled)
    source = Image.open(io.BytesIO(_png(40))).convert("RGB")
    sheet = Image.open(body["crop_sheet"]).convert("RGB")
    assert sheet.crop((0, 0, 100, 100)).tobytes() == source.crop((10, 20, 110, 120)).tobytes()
    assert len(rest) == 1  # the 100px-wide crop sheet is small enough to inline


async def test_sweep_refuses_more_than_24_runs_and_bad_specs(sweep_env):
    server, client, _wf, _node, wf_id, _ = sweep_env
    big = await server.run_workflow(wf_id, front=True, sweep={"variants": _variants(), "seeds": list(range(13))})
    assert big["status"] == "invalid" and "24" in big["error"]
    wide = await server.run_workflow(wf_id, front=True, sweep={"crops": [[0, 0, 600, 100]]})
    assert wide["status"] == "invalid" and "512" in wide["error"]
    bg = await server.run_workflow(wf_id, front=True, wait=False, sweep={})
    assert bg["status"] == "invalid" and "wait=True" in bg["error"]
    assert client.apis == []


async def test_sweep_skips_an_invalid_variant_and_still_runs_the_rest(sweep_env):
    server, client, _wf, _node, wf_id, out = sweep_env
    variants = [
        {"label": "broken", "ops": [{"op": "set_widget", "node_id": 99, "input": "steps", "value": 4}]},
        {"label": "ok", "ops": []},
    ]
    result = await server.run_workflow(wf_id, front=True, save_dir=str(out), sweep={"variants": variants})
    body = result[0] if isinstance(result, list) else result
    assert [(r["label"], r["status"]) for r in body["runs"]] == [("broken", "invalid"), ("ok", "success")]
    assert len(client.apis) == 1


async def test_oversized_crop_sheet_is_a_file_not_an_inline_image(sweep_env):
    server, _client, _wf, _node, wf_id, out = sweep_env
    box = [0, 0, 512, 512]
    result = await server.run_workflow(
        wf_id, front=True, save_dir=str(out), sweep={"crops": [box, box, box, box]}
    )
    body, *images = result
    assert len(images) == 1  # contact sheet only
    assert "crop_hint" in body and Path(body["crop_sheet"]).is_file()


def test_crop_tiles_outside_the_image_are_none_not_bad_tiles():
    from PIL import Image

    from comfy_draftsman.imaging import crop_tiles

    tiles = crop_tiles(Image.new("RGB", (100, 100)), [[10, 10, 50, 50], [200, 200, 300, 300]])
    assert tiles[0].size == (40, 40) and tiles[1] is None
