"""Partner/API spend gate and the precise queue-destruction gate.

Queueing a partner node is a purchase, not a render, and there is no undo. The
three degrade paths below all have to work, because elicitation support varies
by client and the no-capability path is the common one on some of them - it is
also the one that silently regresses, since it needs no user interaction to
look fine.
"""

import json

import pytest

from comfy_draftsman import server
from comfy_draftsman.config import Config
from comfy_draftsman.graph.model import MODE_BYPASS, MODE_MUTE, Workflow
from comfy_draftsman.graph.spend import api_nodes, is_api_node
from comfy_draftsman.session import Session

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"


@pytest.fixture(scope="module")
def oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


class _Answer:
    def __init__(self, action, confirm=None):
        self.action = action
        self.data = type("D", (), {"confirm": confirm})() if confirm is not None else None


class FakeCtx:
    """A client that supports elicitation. `accept` decides the answer."""

    def __init__(self, accept: bool):
        self.accept = accept
        self.messages: list[str] = []

    async def elicit(self, message, schema):
        self.messages.append(message)
        return _Answer("accept", True) if self.accept else _Answer("decline")


class DeafCtx:
    """A client with no elicitation capability - elicit raises."""

    def __init__(self):
        self.calls = 0

    async def elicit(self, message, schema):
        self.calls += 1
        raise RuntimeError("Method not found: elicitation/create")


# --- detection -------------------------------------------------------------


def test_api_node_flag_is_the_primary_signal(oi):
    assert is_api_node(oi["LumaImageNode"]) is True
    assert is_api_node(oi["KSampler"]) is False


def test_category_fallback_covers_instances_without_the_flag():
    """The api_node boolean is comparatively recent; a partner pack on an older
    instance still files itself under an 'api node/...' category."""
    assert is_api_node({"category": "api node/video/Kling"}) is True
    assert is_api_node({"category": "API node/image/Runway"}) is True
    assert is_api_node({"category": "conditioning"}) is False


def test_missing_field_is_not_billable():
    """Unknown is not paid. Flagging everything an old instance can't classify
    would train users to click through the one prompt that matters."""
    assert is_api_node({}) is False
    assert is_api_node(None) is False
    assert is_api_node({"api_node": False}) is False
    # a string "true" is not the boolean the schema promises
    assert is_api_node({"api_node": "true"}) is False


def test_disabled_partner_nodes_are_not_billable(oi):
    """A muted or bypassed node never reaches the executor, so confirming a
    spend for it would be a prompt for something that cannot happen."""
    wf = Workflow.new()
    muted = wf.add_node("LumaImageNode", object_info=oi)
    muted.mode = MODE_MUTE
    bypassed = wf.add_node("LumaImageNode", object_info=oi)
    bypassed.mode = MODE_BYPASS
    assert api_nodes(wf, oi) == []
    live = wf.add_node("LumaImageNode", object_info=oi)
    assert [n["node_id"] for n in api_nodes(wf, oi)] == [live.id]


def test_titles_ride_along_only_when_they_say_something(oi):
    wf = Workflow.new()
    node = wf.add_node("LumaImageNode", object_info=oi)
    assert "title" not in api_nodes(wf, oi)[0]
    node.title = "hero shot"
    assert api_nodes(wf, oi)[0]["title"] == "hero shot"


# --- run_workflow's gate ---------------------------------------------------


class SpendClient:
    def __init__(self, oi):
        self.oi = oi
        self.queued = 0
        self.queue: dict = {"queue_running": [], "queue_pending": []}

    async def get_object_info(self, refresh: bool = False):
        return self.oi

    async def get_system_stats(self):
        return {"devices": []}

    async def get_queue(self):
        return self.queue

    async def queue_prompt(self, api, extra_data=None, client_id=None, front=False):
        self.queued += 1
        return {"prompt_id": "spend-1"}

    async def run_and_wait(self, api, timeout=600.0, extra_data=None, front=False):
        self.queued += 1
        return {"status": "success", "prompt_id": "spend-1", "outputs": []}


class StubTracker:
    client_id = "t"

    def ensure_running(self):
        pass


@pytest.fixture
def billable(monkeypatch, tmp_path, oi):
    client = SpendClient(oi)
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(
        server._State,
        "config",
        Config(comfyui_url="http://comfy.test", session_dir=tmp_path, comfy_api_key="k-123"),
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    wf = Workflow.new()
    wf.add_node("LumaImageNode", object_info=oi)
    return client, session.create(wf, title="paid")


async def test_no_capability_returns_the_instructions_and_queues_nothing(billable):
    client, wf_id = billable
    result = await server.run_workflow(wf_id, wait=False)
    assert result["status"] == "spend_confirmation_required"
    assert result["api_nodes"] == [{"node_id": 1, "class_type": "LumaImageNode"}]
    assert "confirm_spend=True" in result["hint"]
    assert client.queued == 0


async def test_declining_queues_nothing(billable):
    client, wf_id = billable
    ctx = FakeCtx(accept=False)
    result = await server.run_workflow(wf_id, wait=False, ctx=ctx)
    assert result["status"] == "spend_declined"
    assert client.queued == 0
    assert "LumaImageNode" in ctx.messages[0]


async def test_accepting_queues_the_run(billable):
    client, wf_id = billable
    result = await server.run_workflow(wf_id, wait=False, ctx=FakeCtx(accept=True))
    assert result["status"] == "queued"
    assert client.queued == 1


async def test_confirm_spend_skips_the_prompt_entirely(billable):
    client, wf_id = billable
    ctx = DeafCtx()
    result = await server.run_workflow(wf_id, wait=False, confirm_spend=True, ctx=ctx)
    assert result["status"] == "queued"
    assert client.queued == 1
    assert ctx.calls == 0  # already authorized; no reason to ask again


async def test_an_elicitation_error_degrades_rather_than_raising(billable):
    client, wf_id = billable
    result = await server.run_workflow(wf_id, wait=False, ctx=DeafCtx())
    assert result["status"] == "spend_confirmation_required"
    assert client.queued == 0


async def test_missing_api_key_fails_early_with_a_name(billable, monkeypatch, tmp_path):
    """Today this surfaces as an opaque queue-time 'Unauthorized'."""
    client, wf_id = billable
    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url="http://comfy.test", session_dir=tmp_path)
    )
    result = await server.run_workflow(wf_id, wait=False, confirm_spend=True, ctx=FakeCtx(True))
    assert result["status"] == "missing_api_key"
    assert "COMFY_API_KEY" in result["hint"]
    assert client.queued == 0


async def test_an_ordinary_graph_is_never_gated(monkeypatch, tmp_path, oi):
    client = SpendClient(oi)
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(
        server._State, "config", Config(comfyui_url="http://comfy.test", session_dir=tmp_path)
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    wf = Workflow.new()
    wf.add_node("KSampler", object_info=oi)
    wf_id = session.create(wf, title="free")
    result = await server.run_workflow(wf_id, wait=False, allow_invalid=True, ctx=FakeCtx(False))
    assert result["status"] == "queued"
    # the happy path carries no spend keys at all
    assert not any(k.startswith("api_nodes") or k == "capacity" for k in result)


async def test_the_api_nodes_list_is_capped(monkeypatch, tmp_path, oi):
    client = SpendClient(oi)
    session = Session(tmp_path / "sessions")
    monkeypatch.setattr(
        server._State,
        "config",
        Config(comfyui_url="http://comfy.test", session_dir=tmp_path, comfy_api_key="k"),
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "session", session)
    monkeypatch.setattr(server._State, "tracker", StubTracker())
    wf = Workflow.new()
    for _ in range(30):
        wf.add_node("LumaImageNode", object_info=oi)
    wf_id = session.create(wf, title="many")
    result = await server.run_workflow(wf_id, wait=False)
    assert len(result["api_nodes"]) == server._API_NODES_CAP
    assert result["api_nodes_truncated"] == 30 - server._API_NODES_CAP
    assert len(json.dumps(result)) < 2_000


# --- manage_queue's precise gate -------------------------------------------


class QueueClient:
    def __init__(self, queue):
        self.queue = queue
        self.done: list = []

    async def get_queue(self):
        return self.queue

    async def interrupt(self):
        self.done.append("interrupt")

    async def clear_queue(self):
        self.done.append("clear")

    async def delete_queue_items(self, ids):
        self.done.append(("delete", tuple(ids)))

    async def free(self, unload_models=False):
        self.done.append("free")


def _queue(monkeypatch, tmp_path, running=(), pending=()):
    client = QueueClient(
        {
            "queue_running": [[0, pid, {}] for pid in running],
            "queue_pending": [[i, pid, {}] for i, pid in enumerate(pending)],
        }
    )
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path))
    return client


async def test_clearing_only_our_own_jobs_never_prompts(monkeypatch, tmp_path):
    """The precision that makes this gate worth having: draftsman cleaning up
    after itself stays silent."""
    client = _queue(monkeypatch, tmp_path, pending=["mine-1", "mine-2"])
    monkeypatch.setattr(server._State, "submitted", {"mine-1": "w", "mine-2": "w"})
    ctx = FakeCtx(accept=False)
    result = await server.manage_queue("clear", ctx=ctx)
    assert result["done"] == "pending queue cleared"
    assert ctx.messages == []
    assert client.done == ["clear"]


async def test_clearing_someone_elses_job_asks_first(monkeypatch, tmp_path):
    client = _queue(monkeypatch, tmp_path, pending=["mine-1", "theirs"])
    monkeypatch.setattr(server._State, "submitted", {"mine-1": "w"})
    result = await server.manage_queue("clear", ctx=FakeCtx(accept=False))
    assert result["status"] == "queue_declined"
    assert client.done == []


async def test_accepting_the_queue_prompt_proceeds(monkeypatch, tmp_path):
    client = _queue(monkeypatch, tmp_path, pending=["theirs"])
    result = await server.manage_queue("clear", ctx=FakeCtx(accept=True))
    assert result["done"] == "pending queue cleared"
    assert client.done == ["clear"]


async def test_no_capability_refuses_to_discard_foreign_work(monkeypatch, tmp_path):
    client = _queue(monkeypatch, tmp_path, running=["theirs"])
    result = await server.manage_queue("interrupt", ctx=DeafCtx())
    assert result["status"] == "queue_confirmation_required"
    assert "manage_queue(action='status')" in result["hint"]
    assert client.done == []


async def test_deleting_ids_that_are_ours_stays_silent(monkeypatch, tmp_path):
    client = _queue(monkeypatch, tmp_path, pending=["mine-1", "theirs"])
    monkeypatch.setattr(server._State, "submitted", {"mine-1": "w"})
    result = await server.manage_queue("delete", prompt_ids=["mine-1"], ctx=FakeCtx(False))
    assert result["done"] == "deleted 1 pending prompt(s)"
    assert client.done == [("delete", ("mine-1",))]


async def test_deleting_an_unqueued_id_is_not_a_destructive_act(monkeypatch, tmp_path):
    """An id that isn't in the queue can't destroy anything, so it must not
    trigger a prompt."""
    client = _queue(monkeypatch, tmp_path, pending=["theirs"])
    result = await server.manage_queue("delete", prompt_ids=["ghost"], ctx=FakeCtx(False))
    assert "done" in result
    assert client.done == [("delete", ("ghost",))]


async def test_free_and_status_are_never_gated(monkeypatch, tmp_path):
    client = _queue(monkeypatch, tmp_path, running=["theirs"], pending=["also-theirs"])
    ctx = FakeCtx(accept=False)
    assert "running" in await server.manage_queue("status", ctx=ctx)
    assert "freed memory" in (await server.manage_queue("free", ctx=ctx))["done"]
    assert ctx.messages == []
    assert client.done == ["free"]


async def test_an_unreachable_queue_never_blocks_cleanup(monkeypatch, tmp_path):
    """Same best-effort posture as run_workflow's queue etiquette check."""

    class Broken(QueueClient):
        async def get_queue(self):
            raise OSError("connection refused")

    client = Broken({})
    monkeypatch.setattr(server._State, "client", client)
    monkeypatch.setattr(server._State, "config", Config(session_dir=tmp_path))
    result = await server.manage_queue("clear", ctx=FakeCtx(accept=False))
    assert result["done"] == "pending queue cleared"
