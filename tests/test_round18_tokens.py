"""Round 18: token-efficiency ceilings.

Everything this server returns is paid for in the caller's context window, and
the expensive failures are silent - a response does not get slower or throw when
it quietly grows twenty-fold. These tests assert measured CEILINGS on the
payloads most likely to blow up, so a future change that reintroduces an
unbounded list fails here instead of in someone's conversation.

Ceilings are deliberately loose (roughly 2x the current measured size): they are
regression guards, not golden-master assertions, and should not need editing for
ordinary wording changes.

The one exception is the tool-surface ceiling at the bottom of this file. That
budget is paid on every request rather than only when a tool is called, so it is
held tight on purpose and raising it is a deliberate decision, not a fix.
"""

import json

import pytest

from comfy_draftsman import server
from comfy_draftsman.graph.lint import lint
from comfy_draftsman.graph.model import MODE_MUTE, VIRTUAL_TYPES, Workflow
from comfy_draftsman.graph.validate import validate

FIXTURE_OI = "tests/fixtures/object_info_trimmed.json"


def _chars(payload) -> int:
    return len(json.dumps(payload))


@pytest.fixture(scope="module")
def oi():
    with open(FIXTURE_OI, encoding="utf-8") as fh:
        return json.load(fh)


def _mid_build_graph(oi, count=20):
    """What a build loop actually produces: add_node leaves every node at the
    default position, so the whole graph is mutually overlapping until
    organize_workflow runs."""
    wf = Workflow.new()
    for _ in range(count):
        wf.add_node("KSampler", object_info=oi)
    return wf


# --- lint: the overlap report must not be quadratic ---------------------------


def test_overlap_is_one_finding_not_one_per_pair(oi):
    wf = _mid_build_graph(oi, count=20)
    overlaps = [f for f in lint(wf, oi) if f["code"] == "overlap"]
    # 20 co-located nodes = 190 pairs; that must not be 190 findings
    assert len(overlaps) == 1
    assert overlaps[0]["node_ids"] == sorted(n.id for n in wf.nodes.values())
    assert "190 overlapping pair(s)" in overlaps[0]["message"]


def test_overlap_report_does_not_grow_with_node_count(oi):
    """The whole point: 3x the nodes must not mean ~9x the overlap output."""
    small = [f for f in lint(_mid_build_graph(oi, 10), oi) if f["code"] == "overlap"]
    large = [f for f in lint(_mid_build_graph(oi, 60), oi) if f["code"] == "overlap"]
    assert len(small) == len(large) == 1
    # the id list grows linearly, but the message itself stays bounded
    assert len(large[0]["message"]) < 2 * len(small[0]["message"])


def test_lint_of_a_mid_build_graph_stays_bounded(oi):
    """Was 292 findings / 27,108 chars before the overlap collapse."""
    capped = server._cap_lint(lint(_mid_build_graph(oi, 20), oi))
    assert len(capped) <= server._FINDINGS_CAP + 1
    assert _chars(capped) < 10_000


def test_lint_is_bounded_even_on_a_large_messy_graph(oi):
    """Was 1,428 findings / 122,754 chars (~30,688 tokens) in one response."""
    wf = Workflow.new()
    for i in range(40):
        wf.add_node(f"SomeOldCustomNode{i}")
    for _ in range(12):
        wf.add_node("KSampler", object_info=oi)
    capped = server._cap_lint(lint(wf, oi))
    assert len(capped) <= server._FINDINGS_CAP + 1
    assert _chars(capped) < 10_000


# --- validate: one note for a disabled branch, not one per node --------------


def test_disabled_nodes_collapse_to_a_single_finding(oi):
    with open("tests/fixtures/sdxl_simple_example.json", encoding="utf-8") as fh:
        wf = Workflow.from_ui(json.load(fh))
    for node in wf.nodes.values():
        node.mode = MODE_MUTE
    disabled = [f for f in validate(wf, oi) if f["code"] == "node-disabled"]
    assert len(disabled) == 1
    # an uninstalled class is reported as missing-node-class instead, which is
    # the more useful finding - so only the known, non-virtual nodes land here
    expected = [
        n.id for n in wf.nodes.values() if n.type in oi and n.type not in VIRTUAL_TYPES
    ]
    assert disabled[0]["node_ids"] == sorted(expected)
    # flat cost regardless of how many nodes are disabled
    assert _chars(disabled[0]) < 500


def test_disabled_finding_names_a_few_ids_and_counts_the_rest(oi):
    wf = Workflow.new()
    for _ in range(40):
        node = wf.add_node("KSampler", object_info=oi)
        node.mode = MODE_MUTE
    finding = next(f for f in validate(wf, oi) if f["code"] == "node-disabled")
    assert "more)" in finding["message"]  # the tail is counted, not listed
    assert len(finding["node_ids"]) == 40  # ...but the full list is available
    assert _chars(finding) < 800


# --- _cap_findings must never make a payload bigger --------------------------


def test_cap_never_returns_more_than_it_received():
    errors = [{"level": "error", "code": "x", "message": "m"} for _ in range(88)]
    out = server._cap_findings(errors)
    assert len(out) <= len(errors)
    assert not any(f.get("code") == "findings-truncated" for f in out)


def test_cap_still_marks_a_real_truncation():
    mixed = [{"level": "error", "code": "e", "message": "m"} for _ in range(3)]
    mixed += [{"level": "info", "code": "i", "message": "m"} for _ in range(100)]
    out = server._cap_findings(mixed)
    marker = out[-1]
    assert marker["code"] == "findings-truncated"
    assert "63 more" in marker["message"]
    # every error survived the trim
    assert sum(1 for f in out if f.get("level") == "error") == 3


def test_cap_lint_marks_truncation():
    warnings = [{"code": "c", "message": "m"} for _ in range(100)]
    out = server._cap_lint(warnings)
    assert len(out) == server._FINDINGS_CAP + 1
    assert out[-1]["code"] == "lint-truncated"


# --- discovery payloads are capped -------------------------------------------


@pytest.mark.asyncio
async def test_list_models_caps_a_huge_folder(monkeypatch, tmp_path):
    """A real instance can hold hundreds of LoRAs; object_info combos of the
    same filenames are capped at 24, so this must be capped too."""
    files = [f"pack_v{i}_style_concept_{i:03d}.safetensors" for i in range(400)]

    class _Client:
        async def list_model_folders(self):
            return ["loras"]

        async def list_models(self, folder):
            return files

    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))

    result = await server.list_models(folder="loras")
    assert result["count"] == 400  # the true total is still reported
    assert len(result["files"]) == server._MODEL_FILES_CAP
    assert result["truncated"] == 400 - server._MODEL_FILES_CAP
    assert "search=" in result["hint"]
    assert _chars(result) < 6_000  # was ~21,134


@pytest.mark.asyncio
async def test_list_models_does_not_cap_a_normal_folder(monkeypatch, tmp_path):
    class _Client:
        async def list_model_folders(self):
            return ["checkpoints"]

        async def list_models(self, folder):
            return [f"model_{i}.safetensors" for i in range(10)]

    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    result = await server.list_models(folder="checkpoints")
    assert len(result["files"]) == 10
    assert "truncated" not in result and "hint" not in result


@pytest.mark.asyncio
async def test_search_nodes_detail_folds_only_the_top_hits(monkeypatch, oi, tmp_path):
    """detail=True at the default limit was ~13,068 chars on the trimmed
    fixture alone."""

    class _Client:
        async def get_object_info(self, refresh: bool = False):
            return oi

    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))

    results = await server.search_nodes(query="", limit=25, detail=True)
    with_schema = [r for r in results if "schema" in r]
    assert len(with_schema) <= server._DETAIL_SCHEMA_CAP
    assert any("note" in r for r in results)  # the truncation is announced


@pytest.mark.asyncio
async def test_search_nodes_detail_folds_everything_when_it_fits(
    monkeypatch, oi, tmp_path
):
    class _Client:
        async def get_object_info(self, refresh: bool = False):
            return oi

    monkeypatch.setattr(server._State, "client", _Client())
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    results = await server.search_nodes(query="KSampler", limit=3, detail=True)
    assert results and all("schema" in r for r in results)
    assert not any("note" in r for r in results)


# --- the always-loaded surface: the ceiling that governs every new feature ---
#
# Tool descriptions and input schemas are re-sent on EVERY request for the life
# of the session, whether or not a single tool is called. That makes them the
# one budget where a "small addition" compounds without limit, and the reason
# 0.15.0 added four features with one new parameter between them. Measured
# 21,910 chars / 29 tools at the time of writing; the ceiling leaves ~1% of
# headroom and no more. If this fails, the fix is to trim an existing docstring,
# not to raise the number - move the explanation into the response that needs
# it, where only the caller who hit that path pays for it.


async def _surface():
    tools = await server.mcp.list_tools()
    return tools, sum(
        len(t.description or "") + len(json.dumps(t.inputSchema)) for t in tools
    )


@pytest.mark.asyncio
async def test_total_tool_surface_stays_under_the_ceiling():
    tools, total = await _surface()
    assert total <= 22_200, (
        f"tool descriptions + schemas grew to {total} chars. Pay for new text by "
        "trimming existing text, or teach it in a response instead."
    )
    assert len(tools) == 29, (
        "a new tool costs its full description on every request, forever - "
        "extend an existing tool instead"
    )


@pytest.mark.asyncio
async def test_the_handshake_block_stays_under_its_ceiling():
    """Paid once per session, by every session - including the ones that never
    queue a paid render or touch a small GPU. Gates teach through their own
    responses precisely so this does not have to grow."""
    assert len(server.mcp.instructions or "") <= 900


@pytest.mark.asyncio
async def test_the_injected_context_parameter_costs_nothing():
    """FastMCP hides Context from the JSON schema. The spend gate's whole
    token story depends on that being true rather than assumed."""
    tools, _ = await _surface()
    for tool in tools:
        assert "ctx" not in (tool.inputSchema.get("properties") or {}), tool.name


# --- gate payloads ------------------------------------------------------------


@pytest.mark.asyncio
async def test_guidance_costs_nothing_extra_on_a_comfortable_gpu(monkeypatch, tmp_path):
    """The overwhelmingly common call must return exactly what it always did."""
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "devices", [
        {"name": "big", "vram_total": 48 * 1024**3, "vram_free": 48 * 1024**3}
    ])
    result = await server.get_model_guidance(family="flux")
    assert "fit" not in result
    assert "hardware" not in result  # the raw block never rides along


@pytest.mark.asyncio
async def test_a_fit_verdict_is_a_few_hundred_chars_not_a_lecture(monkeypatch, tmp_path):
    monkeypatch.setattr(server._State, "config", server.Config(session_dir=tmp_path))
    monkeypatch.setattr(server._State, "devices", [
        {"name": "small", "vram_total": 6 * 1024**3, "vram_free": 6 * 1024**3}
    ])
    result = await server.get_model_guidance(family="flux")
    assert result["fit"]["verdict"] == "insufficient"
    assert _chars(result["fit"]) < 800


def test_the_spend_payload_is_capped_no_matter_how_paid_the_graph_is():
    nodes = [{"node_id": i, "class_type": "SomePartnerNode", "title": "t" * 40} for i in range(60)]
    payload = server._spend_payload(nodes)
    assert len(payload["api_nodes"]) == server._API_NODES_CAP
    assert payload["api_nodes_truncated"] == 60 - server._API_NODES_CAP
    assert _chars(payload) < 2_000
