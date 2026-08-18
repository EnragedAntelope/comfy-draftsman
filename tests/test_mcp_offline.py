"""End-to-end through the MCP protocol layer with NO live ComfyUI.

tests/test_mcp_e2e.py already drives the real protocol surface, but it is
integration-marked and needs a running instance, so it never runs in CI or in a
sandboxed session. This module mounts the same FastMCP server against a
respx-mocked ComfyUI, which means the protocol layer - tool registration,
schema generation, annotations, elicitation round-trips, the JSON that actually
reaches a client - is covered on every commit.

The negative assertions matter as much as the positive ones. Every gate added
in 0.15.0 is silent on the happy path by design, and "silent" is precisely the
property that regresses without anyone noticing.
"""

import json

import httpx
import pytest
import respx
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

BASE = "http://comfy.offline.test"

with open("tests/fixtures/object_info_trimmed.json", encoding="utf-8") as _fh:
    OBJECT_INFO = json.load(_fh)


def _stats(vram_gb: float | None, free_gb: float | None = None) -> dict:
    if vram_gb is None:
        devices: list[dict] = []
    else:
        free = vram_gb if free_gb is None else free_gb
        devices = [
            {
                "name": "cuda:0 NVIDIA Test",
                "type": "cuda",
                "vram_total": int(vram_gb * 1024**3),
                "vram_free": int(free * 1024**3),
            }
        ]
    return {
        "system": {"comfyui_version": "0.29.0", "os": "posix", "python_version": "3.11"},
        "devices": devices,
    }


class _StubTracker:
    """The websocket is out of scope here: respx mocks HTTP only, and a
    background reconnect loop would just add noise."""

    client_id = "offline-tracker"

    def ensure_running(self) -> None:
        pass

    def snapshot(self, prompt_id):
        return {"ws_connected": False}


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """A fully mocked instance + a cold server state. Returns a setter for the
    /system_stats payload so a test can choose the GPU it is talking to."""
    monkeypatch.setenv("COMFYUI_URL", BASE)
    monkeypatch.setenv("DRAFTSMAN_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("DRAFTSMAN_LEARNED_DIR", str(tmp_path / "learned"))
    from comfy_draftsman import server

    for attr in ("config", "client", "registry", "session", "devices"):
        monkeypatch.setattr(server._State, attr, None)
    monkeypatch.setattr(server._State, "tracker", _StubTracker())

    state = {"stats": _stats(48), "queued": []}

    with respx.mock(assert_all_called=False) as mock:
        mock.get(f"{BASE}/system_stats").mock(
            side_effect=lambda request: httpx.Response(200, json=state["stats"])
        )
        mock.get(f"{BASE}/object_info").mock(return_value=httpx.Response(200, json=OBJECT_INFO))
        mock.get(f"{BASE}/queue").mock(
            return_value=httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        )

        def _prompt(request):
            state["queued"].append(json.loads(request.content))
            return httpx.Response(200, json={"prompt_id": "offline-1"})

        mock.post(f"{BASE}/prompt").mock(side_effect=_prompt)
        state["mock"] = mock
        yield server, state


def _json(result):
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


async def _session(server, **kwargs):
    return create_connected_server_and_client_session(server.mcp._mcp_server, **kwargs)


# --- orientation -----------------------------------------------------------


async def test_instance_info_normalizes_vram_to_gb(offline):
    server, state = offline
    state["stats"] = _stats(16, free_gb=9)
    async with await _session(server) as mcp_session:
        info = _json(await mcp_session.call_tool("get_instance_info", {}))
    device = info["devices"][0]
    assert device["vram_total_gb"] == 16.0
    assert device["vram_free_gb"] == 9.0
    assert device["vram_total"] == 16 * 1024**3  # raw bytes still reported


# --- the fit verdict, and its silence --------------------------------------


async def test_guidance_warns_on_a_gpu_too_small_for_the_family(offline):
    server, state = offline
    state["stats"] = _stats(8)
    async with await _session(server) as mcp_session:
        guidance = _json(
            await mcp_session.call_tool("get_model_guidance", {"family": "flux"})
        )
    assert guidance["fit"]["verdict"] == "insufficient"
    assert guidance["fit"]["required_gb"] == 12
    assert guidance["fit"]["source"].startswith("https://")
    assert guidance["sampling"], "the guidance itself is still the point of the call"


async def test_guidance_says_nothing_on_a_big_gpu(offline):
    server, state = offline
    state["stats"] = _stats(48)
    async with await _session(server) as mcp_session:
        guidance = _json(
            await mcp_session.call_tool("get_model_guidance", {"family": "flux"})
        )
    assert "fit" not in guidance


async def test_guidance_says_nothing_for_a_family_with_no_vram_data(offline):
    """`unknown` is deliberately silent: a nag on every call is one an agent
    learns to skip, and it costs tokens forever."""
    server, state = offline
    state["stats"] = _stats(4)
    async with await _session(server) as mcp_session:
        guidance = _json(await mcp_session.call_tool("get_model_guidance", {"family": "ltx"}))
    assert "fit" not in guidance


async def test_guidance_never_returns_the_raw_hardware_block(offline):
    server, state = offline
    state["stats"] = _stats(8)
    async with await _session(server) as mcp_session:
        for family in ("flux", "sdxl", "wan"):
            guidance = _json(await mcp_session.call_tool("get_model_guidance", {"family": family}))
            assert "hardware" not in guidance, family


async def test_guidance_survives_an_unreachable_instance(offline):
    """The verdict is a bonus; a knowledge lookup must not become an error
    because ComfyUI happens to be down."""
    server, state = offline
    state["mock"].get(f"{BASE}/system_stats").mock(side_effect=httpx.ConnectError("down"))
    async with await _session(server) as mcp_session:
        guidance = _json(await mcp_session.call_tool("get_model_guidance", {"family": "flux"}))
    assert guidance["sampling"]
    assert "fit" not in guidance


# --- authoring flow --------------------------------------------------------


SDXL_OPS = [
    {"op": "add_node", "class_type": "CheckpointLoaderSimple",
     "widgets": {"ckpt_name": "SDXL\\spaSplashedAfter_v20.safetensors"}},
    {"op": "add_node", "class_type": "CLIPTextEncode", "widgets": {"text": "a lighthouse"}},
    {"op": "add_node", "class_type": "CLIPTextEncode", "widgets": {"text": "blurry"}},
    {"op": "add_node", "class_type": "EmptyLatentImage", "widgets": {"width": 1024, "height": 1024}},
    {"op": "add_node", "class_type": "KSampler", "widgets": {"steps": 20, "cfg": 6.0, "seed": 1}},
    {"op": "add_node", "class_type": "VAEDecode"},
    {"op": "add_node", "class_type": "SaveImage", "widgets": {"filename_prefix": "offline"}},
    {"op": "connect", "from_node": 1, "from_output": "MODEL", "to_node": 5, "to_input": "model"},
    {"op": "connect", "from_node": 1, "from_output": "CLIP", "to_node": 2, "to_input": "clip"},
    {"op": "connect", "from_node": 1, "from_output": "CLIP", "to_node": 3, "to_input": "clip"},
    {"op": "connect", "from_node": 2, "from_output": "CONDITIONING", "to_node": 5, "to_input": "positive"},
    {"op": "connect", "from_node": 3, "from_output": "CONDITIONING", "to_node": 5, "to_input": "negative"},
    {"op": "connect", "from_node": 4, "from_output": "LATENT", "to_node": 5, "to_input": "latent_image"},
    {"op": "connect", "from_node": 5, "from_output": "LATENT", "to_node": 6, "to_input": "samples"},
    {"op": "connect", "from_node": 1, "from_output": "VAE", "to_node": 6, "to_input": "vae"},
    {"op": "connect", "from_node": 6, "from_output": "IMAGE", "to_node": 7, "to_input": "images"},
]


async def _build_sdxl(mcp_session) -> str:
    wf_id = _json(await mcp_session.call_tool("create_workflow", {"title": "offline"}))[
        "workflow_id"
    ]
    edited = _json(
        await mcp_session.call_tool(
            "edit_workflow", {"workflow_id": wf_id, "operations": SDXL_OPS}
        )
    )
    assert "error" not in edited, edited
    return wf_id


async def test_the_whole_authoring_loop_runs_offline(offline):
    server, _state = offline
    async with await _session(server) as mcp_session:
        wf_id = await _build_sdxl(mcp_session)
        valid = _json(await mcp_session.call_tool("validate_workflow", {"workflow_id": wf_id}))
        assert valid["ok"], valid
        organized = _json(await mcp_session.call_tool("organize_workflow", {"workflow_id": wf_id}))
        assert organized["family"] == "sdxl"
        assert organized["lint"] == []
        exported = _json(
            await mcp_session.call_tool(
                "export_workflow_json", {"workflow_id": wf_id, "format": "api"}
            )
        )
        assert any(n["class_type"] == "KSampler" for n in exported.values())


async def test_an_ordinary_run_carries_no_gate_keys(offline):
    """The happy path is the one that has to stay free."""
    server, state = offline
    state["stats"] = _stats(48)
    async with await _session(server) as mcp_session:
        wf_id = await _build_sdxl(mcp_session)
        result = _json(
            await mcp_session.call_tool(
                "run_workflow", {"workflow_id": wf_id, "wait": False}
            )
        )
    assert result["status"] == "queued"
    assert "capacity" not in result
    assert "api_nodes" not in result
    assert len(state["queued"]) == 1


async def test_a_small_gpu_gets_an_advisory_capacity_block_but_still_runs(offline):
    """The verdict warns; it never refuses. A curated floor is not authoritative
    enough to stand between a user and their own hardware."""
    server, state = offline
    state["stats"] = _stats(4)
    async with await _session(server) as mcp_session:
        wf_id = await _build_sdxl(mcp_session)
        result = _json(
            await mcp_session.call_tool(
                "run_workflow", {"workflow_id": wf_id, "wait": False}
            )
        )
    assert result["status"] == "queued"
    assert result["capacity"]["verdict"] == "insufficient"
    assert result["capacity"]["family"] == "sdxl"
    assert len(state["queued"]) == 1, "warned, not blocked"


# --- the spend gate, all three degrade paths -------------------------------


async def _build_paid(mcp_session) -> str:
    wf_id = _json(await mcp_session.call_tool("create_workflow", {"title": "paid"}))[
        "workflow_id"
    ]
    edited = _json(
        await mcp_session.call_tool(
            "edit_workflow",
            {
                "workflow_id": wf_id,
                "operations": [
                    {"op": "add_node", "class_type": "LumaImageNode",
                     "widgets": {"prompt": "a lighthouse"}},
                    {"op": "add_node", "class_type": "SaveImage",
                     "widgets": {"filename_prefix": "paid"}},
                    {"op": "connect", "from_node": 1, "from_output": "IMAGE",
                     "to_node": 2, "to_input": "images"},
                ],
            },
        )
    )
    assert "error" not in edited, edited
    return wf_id


def _elicit(action: str, content: dict | None = None):
    async def callback(context, params):
        return types.ElicitResult(action=action, content=content)

    return callback


@pytest.fixture
def with_key(offline, monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "offline-key")
    return offline


async def test_spend_gate_accept_queues_the_paid_run(with_key):
    server, state = with_key
    async with await _session(
        server, elicitation_callback=_elicit("accept", {"confirm": True})
    ) as mcp_session:
        wf_id = await _build_paid(mcp_session)
        result = _json(
            await mcp_session.call_tool("run_workflow", {"workflow_id": wf_id, "wait": False})
        )
    assert result["status"] == "queued"
    assert len(state["queued"]) == 1


async def test_spend_gate_decline_queues_nothing(with_key):
    server, state = with_key
    async with await _session(
        server, elicitation_callback=_elicit("decline")
    ) as mcp_session:
        wf_id = await _build_paid(mcp_session)
        result = _json(
            await mcp_session.call_tool("run_workflow", {"workflow_id": wf_id, "wait": False})
        )
    assert result["status"] == "spend_declined"
    assert result["api_nodes"][0]["class_type"] == "LumaImageNode"
    assert state["queued"] == []


async def test_spend_gate_without_elicitation_capability_explains_the_way_out(with_key):
    """No elicitation_callback = a client that cannot ask. This is the path that
    silently regresses, because nothing in the transcript looks wrong."""
    server, state = with_key
    async with await _session(server) as mcp_session:
        wf_id = await _build_paid(mcp_session)
        result = _json(
            await mcp_session.call_tool("run_workflow", {"workflow_id": wf_id, "wait": False})
        )
        assert result["status"] == "spend_confirmation_required"
        assert "confirm_spend=True" in result["hint"]
        assert state["queued"] == []
        # ...and the escape hatch actually works
        confirmed = _json(
            await mcp_session.call_tool(
                "run_workflow",
                {"workflow_id": wf_id, "wait": False, "confirm_spend": True},
            )
        )
        assert confirmed["status"] == "queued"
        assert len(state["queued"]) == 1


async def test_an_accept_that_says_no_is_a_decline(with_key):
    """A form returned with the box unticked is not consent."""
    server, state = with_key
    async with await _session(
        server, elicitation_callback=_elicit("accept", {"confirm": False})
    ) as mcp_session:
        wf_id = await _build_paid(mcp_session)
        result = _json(
            await mcp_session.call_tool("run_workflow", {"workflow_id": wf_id, "wait": False})
        )
    assert result["status"] == "spend_declined"
    assert state["queued"] == []


async def test_partner_nodes_without_an_api_key_fail_by_name(offline):
    server, state = offline
    async with await _session(
        server, elicitation_callback=_elicit("accept", {"confirm": True})
    ) as mcp_session:
        wf_id = await _build_paid(mcp_session)
        result = _json(
            await mcp_session.call_tool("run_workflow", {"workflow_id": wf_id, "wait": False})
        )
    assert result["status"] == "missing_api_key"
    assert state["queued"] == []


# --- manage_queue's precise gate, over the wire ----------------------------


async def test_manage_queue_clear_asks_before_dropping_a_foreign_job(offline):
    server, state = offline
    state["mock"].get(f"{BASE}/queue").mock(
        return_value=httpx.Response(
            200, json={"queue_running": [], "queue_pending": [[0, "someone-else", {}]]}
        )
    )
    deleted = state["mock"].post(f"{BASE}/queue").mock(return_value=httpx.Response(200))
    async with await _session(
        server, elicitation_callback=_elicit("decline")
    ) as mcp_session:
        result = _json(await mcp_session.call_tool("manage_queue", {"action": "clear"}))
    assert result["status"] == "queue_declined"
    assert not deleted.called


async def test_manage_queue_clear_proceeds_when_accepted(offline):
    server, state = offline
    state["mock"].get(f"{BASE}/queue").mock(
        return_value=httpx.Response(
            200, json={"queue_running": [], "queue_pending": [[0, "someone-else", {}]]}
        )
    )
    cleared = state["mock"].post(f"{BASE}/queue").mock(return_value=httpx.Response(200))
    async with await _session(
        server, elicitation_callback=_elicit("accept", {"confirm": True})
    ) as mcp_session:
        result = _json(await mcp_session.call_tool("manage_queue", {"action": "clear"}))
    assert result["done"] == "pending queue cleared"
    assert cleared.called


async def test_manage_queue_without_elicitation_refuses_to_drop_foreign_work(offline):
    server, state = offline
    state["mock"].get(f"{BASE}/queue").mock(
        return_value=httpx.Response(
            200, json={"queue_running": [], "queue_pending": [[0, "someone-else", {}]]}
        )
    )
    cleared = state["mock"].post(f"{BASE}/queue").mock(return_value=httpx.Response(200))
    async with await _session(server) as mcp_session:
        result = _json(await mcp_session.call_tool("manage_queue", {"action": "clear"}))
    assert result["status"] == "queue_confirmation_required"
    assert not cleared.called


async def test_manage_queue_stays_silent_on_an_empty_queue(offline):
    server, state = offline
    cleared = state["mock"].post(f"{BASE}/queue").mock(return_value=httpx.Response(200))
    async with await _session(
        server, elicitation_callback=_elicit("decline")
    ) as mcp_session:
        result = _json(await mcp_session.call_tool("manage_queue", {"action": "clear"}))
    assert result["done"] == "pending queue cleared"
    assert cleared.called


# --- the protocol surface itself -------------------------------------------


async def test_read_only_annotations_match_what_the_tools_actually_do(offline):
    """A tool marked readOnlyHint is one a client may auto-approve. Getting this
    wrong hands away a mutation the user meant to see."""
    server, _state = offline
    mutating = {
        "create_workflow", "import_workflow", "edit_workflow", "organize_workflow",
        "port_workflow", "run_workflow", "save_workflow", "save_output",
        "upload_image", "manage_queue", "record_learning",
    }
    async with await _session(server) as mcp_session:
        tools = (await mcp_session.list_tools()).tools
    by_name = {t.name: t for t in tools}
    assert set(mutating) <= set(by_name)
    for name, tool in by_name.items():
        annotations = tool.annotations
        assert annotations is not None, name
        assert annotations.readOnlyHint is (name not in mutating), name


async def test_the_injected_context_parameter_is_not_on_the_wire(offline):
    """ctx is FastMCP-injected; if it ever leaked into a schema it would cost
    tokens on every request and invite a client to pass nonsense."""
    server, _state = offline
    async with await _session(server) as mcp_session:
        tools = (await mcp_session.list_tools()).tools
    for tool in tools:
        assert "ctx" not in tool.inputSchema.get("properties", {}), tool.name


async def test_confirm_spend_is_the_only_new_parameter(offline):
    server, _state = offline
    async with await _session(server) as mcp_session:
        tools = {t.name: t for t in (await mcp_session.list_tools()).tools}
    assert "confirm_spend" in tools["run_workflow"].inputSchema["properties"]
    # ...and it is not required, so every existing call site still type-checks
    assert "confirm_spend" not in tools["run_workflow"].inputSchema.get("required", [])
