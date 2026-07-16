"""Round-14: layout companions, canvas-node placement, and queue etiquette.

1. Display nodes (Show Text, PreviewImage) are glued next to the node they
   display instead of being swept into a far-away Output group - previously a
   reader had to trace wires across the canvas to pair previews with samplers.
2. Empty-latent canvas nodes (the resolution knob) live with the user-facing
   Inputs on the far left, not inside Sampling.
3. run_workflow checks the queue first: with >=2 prompts already pending it
   queues NOTHING and returns queue_busy so the user can choose front=True
   (run next; pending jobs untouched) or front=False (wait in line).
4. get_run_status detects queue-time partial accepts from the stored history
   entry (submitted prompt vs outputs_to_execute).
"""

import json
from pathlib import Path

import httpx
import pytest
import respx

from comfy_draftsman import server
from comfy_draftsman.comfy.client import ComfyClient
from comfy_draftsman.config import Config
from comfy_draftsman.graph.annotate import annotate, classify
from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "http://comfy.test"

_SHOWTEXT_SCHEMA = {
    "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
    "output": ["STRING"],
    "output_node": True,
    "python_module": "custom_nodes.pysssss",
}


@pytest.fixture(scope="module")
def object_info():
    info = json.loads((FIXTURES / "object_info_trimmed.json").read_text(encoding="utf-8"))
    info["ShowText|pysssss"] = _SHOWTEXT_SCHEMA
    return info


def _dual_sampler_wf(object_info):
    """Two independent sampler->decode->preview chains off one checkpoint."""
    wf = Workflow.new()
    ckpt = wf.add_node("CheckpointLoaderSimple", object_info=object_info)
    pos = wf.add_node("CLIPTextEncode", object_info=object_info)
    neg = wf.add_node("CLIPTextEncode", object_info=object_info)
    latent = wf.add_node("EmptyLatentImage", object_info=object_info)
    wildcard = wf.add_node("DPRandomGenerator", object_info=object_info)
    show = wf.add_node("ShowText|pysssss", object_info=object_info)
    chains = {}
    for tag in ("a", "b"):
        sampler = wf.add_node("KSampler", object_info=object_info)
        decode = wf.add_node("VAEDecode", object_info=object_info)
        preview = wf.add_node("PreviewImage", object_info=object_info)
        wf.connect(ckpt.id, "MODEL", sampler.id, "model")
        wf.connect(pos.id, "CONDITIONING", sampler.id, "positive")
        wf.connect(neg.id, "CONDITIONING", sampler.id, "negative")
        wf.connect(latent.id, "LATENT", sampler.id, "latent_image")
        wf.connect(sampler.id, "LATENT", decode.id, "samples")
        wf.connect(ckpt.id, "VAE", decode.id, "vae")
        wf.connect(decode.id, "IMAGE", preview.id, "images")
        chains[tag] = (decode.id, preview.id)
    wf.connect(ckpt.id, "CLIP", pos.id, "clip")
    wf.connect(ckpt.id, "CLIP", neg.id, "clip")
    # wildcard source feeds the encoder AND a Show Text beside it (the pattern
    # the no-prompt-preview lint asks for)
    wf.connect(wildcard.id, "STRING", pos.id, "text", object_info=object_info)
    wf.connect(wildcard.id, "STRING", show.id, "text", object_info=object_info)
    return wf, chains, {
        "pos": pos.id, "show": show.id, "latent": latent.id, "wildcard": wildcard.id,
    }


# --- 1. display companions ----------------------------------------------------


def test_previews_sit_beside_their_decode(object_info):
    wf, chains, _ = _dual_sampler_wf(object_info)
    annotate(wf, object_info)
    for decode_id, preview_id in chains.values():
        decode, preview = wf.nodes[decode_id], wf.nodes[preview_id]
        assert preview.pos[0] == decode.pos[0], "preview not in its source's column"
        assert 0 < preview.pos[1] - decode.pos[1] <= decode.size[1] + 80, (
            "preview not glued directly beneath its decode node"
        )


def test_show_text_follows_prompt_not_output_group(object_info):
    wf, _, ids = _dual_sampler_wf(object_info)
    annotate(wf, object_info)
    wildcard, show = wf.nodes[ids["wildcard"]], wf.nodes[ids["show"]]
    assert show.pos[0] == wildcard.pos[0]
    assert show.pos[1] > wildcard.pos[1]
    # and no Output group exists at all: previews moved in with their sources
    assert not any("output" in g.title.lower() for g in wf.groups)


def test_unwired_preview_stays_in_output_stage(object_info):
    wf = Workflow.new()
    node = wf.add_node("PreviewImage", object_info=object_info)
    assert classify(node, object_info) == "output"
    annotate(wf, object_info)  # no upstream -> no companion source; must not crash


def test_companion_layout_has_no_overlaps(object_info):
    wf, _, _ = _dual_sampler_wf(object_info)
    annotate(wf, object_info)

    def box(n):
        return (n.pos[0], n.pos[1], n.pos[0] + n.size[0], n.pos[1] + n.size[1])

    items = [box(n) for n in wf.nodes.values()]
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


# --- 2. canvas nodes are inputs -----------------------------------------------


def test_empty_latent_classified_as_inputs(object_info):
    wf = Workflow.new()
    for class_type in ("EmptyLatentImage", "EmptySD3LatentImage"):
        node = wf.add_node(class_type, object_info=object_info)
        assert classify(node, object_info) == "inputs"


def test_canvas_node_leftmost_and_green(object_info):
    wf, chains, ids = _dual_sampler_wf(object_info)
    annotate(wf, object_info)
    latent = wf.nodes[ids["latent"]]
    assert latent.color == "#232"  # touch-me green
    for decode_id, _ in chains.values():
        assert latent.pos[0] < wf.nodes[decode_id].pos[0]
    size_groups = [g for g in wf.groups if "size" in g.title.lower()]
    assert size_groups, f"no image-size group: {[g.title for g in wf.groups]}"


# --- 3. run_workflow queue etiquette -------------------------------------------


class QueueClient:
    def __init__(self, pending=0):
        self.pending = pending
        self.queued_front = None

    async def get_object_info(self, refresh=False):
        return {}

    async def get_queue(self):
        return {
            "queue_running": [[0, "r1"]],
            "queue_pending": [[i, f"q{i}"] for i in range(self.pending)],
        }

    async def queue_prompt(self, api, extra_data=None, client_id=None, front=False):
        self.queued_front = front
        return {"prompt_id": "p1"}

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        self.queued_front = front
        return {"status": "success", "prompt_id": "p1", "outputs": []}


class StubTracker:
    client_id = "tracker-client"

    def ensure_running(self):
        pass

    def snapshot(self, prompt_id):
        return {}


@pytest.fixture
def wired_queue(monkeypatch, tmp_path):
    def wire(client):
        session = Session(tmp_path / "sessions")
        monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
        monkeypatch.setattr(server._State, "client", client)
        monkeypatch.setattr(server._State, "session", session)
        monkeypatch.setattr(server._State, "tracker", StubTracker())
        return session.create(Workflow.new(), title="t")

    return wire


async def test_busy_queue_blocks_and_offers(wired_queue):
    client = QueueClient(pending=3)
    wf_id = wired_queue(client)
    result = await server.run_workflow(wf_id)
    assert result["status"] == "queue_busy"
    assert result["queue_pending"] == 3
    assert "front=True" in result["hint"]
    assert client.queued_front is None, "a prompt was queued despite queue_busy"


async def test_front_true_jumps_line_without_deleting(wired_queue):
    client = QueueClient(pending=3)
    wf_id = wired_queue(client)
    result = await server.run_workflow(wf_id, front=True)
    assert result["status"] == "success"
    assert client.queued_front is True


async def test_front_false_waits_in_line(wired_queue):
    client = QueueClient(pending=3)
    wf_id = wired_queue(client)
    result = await server.run_workflow(wf_id, front=False)
    assert result["status"] == "success"
    assert client.queued_front is False


async def test_short_queue_runs_without_prompting(wired_queue):
    client = QueueClient(pending=1)
    wf_id = wired_queue(client)
    result = await server.run_workflow(wf_id)
    assert result["status"] == "success"
    assert client.queued_front is False


async def test_wait_false_front_reaches_queue_prompt(wired_queue):
    client = QueueClient(pending=3)
    wf_id = wired_queue(client)
    result = await server.run_workflow(wf_id, wait=False, front=True)
    assert result["status"] == "queued"
    assert client.queued_front is True


@respx.mock
async def test_client_sends_front_flag():
    captured = {}

    def record(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": "p1"})

    respx.post(f"{BASE}/prompt").mock(side_effect=record)
    client = ComfyClient(Config(comfyui_url=BASE))
    try:
        await client.queue_prompt({"1": {"class_type": "X", "inputs": {}}}, front=True)
        assert captured["front"] is True
        captured.clear()
        await client.queue_prompt({"1": {"class_type": "X", "inputs": {}}})
        assert "front" not in captured
    finally:
        await client.close()


# --- 4. get_run_status partial detection ---------------------------------------

_RUN_OI = {
    "SaveImage": {"input": {}, "output": [], "output_node": True},
    "PreviewImage": {"input": {}, "output": [], "output_node": True},
    "KSampler": {"input": {}, "output": ["LATENT"]},
}


class HistoryClient:
    def __init__(self, history):
        self.history = history

    async def get_object_info(self, refresh=False):
        return _RUN_OI

    async def get_history(self, prompt_id):
        return self.history

    @staticmethod
    def _collect_outputs(history):
        return ComfyClient._collect_outputs(history)


def _history(outputs_to_execute):
    prompt_graph = {
        "3": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {}},
        "12": {"class_type": "PreviewImage", "inputs": {}},
    }
    return {
        "prompt": [0, "p1", prompt_graph, {}, outputs_to_execute],
        "outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}},
        "status": {"messages": []},
    }


async def test_run_status_flags_dropped_output_nodes(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
    monkeypatch.setattr(server._State, "client", HistoryClient(_history(["9"])))
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    result = await server.get_run_status("p1")
    assert result["status"] == "partial"
    assert result["dropped_output_nodes"] == ["12"]
    assert "warning" in result


async def test_run_status_clean_when_all_outputs_executed(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", Config(comfyui_url=BASE, session_dir=tmp_path))
    monkeypatch.setattr(server._State, "client", HistoryClient(_history(["9", "12"])))
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    result = await server.get_run_status("p1")
    assert result["status"] == "success"
    assert "dropped_output_nodes" not in result
