"""Round 22: from a live Chroma-HD-Flash troubleshooting session's bug report.

- Muting a node feeding another node's OPTIONAL input validated clean and then
  crashed ComfyUI's own /prompt validation with a raw KeyError - the
  muted/dead-source check only ever walked schema-*required* inputs. Covered in
  tests/test_validate.py (test_muted_producer_feeding_optional_socket_is_flagged
  and the autogrow variant); this file covers the smaller UX finding from the
  same session.
- get_node_info's "not installed" error for a real class typed slightly wrong
  ("SigmasRescale" vs the installed "Sigmas Rescale") gave no path back to the
  right name - a plain substring search doesn't bridge a name that's the same
  minus a space either, since neither string contains the other.
"""

from __future__ import annotations

from comfy_draftsman import server


def test_similar_class_hint_bridges_a_missing_space():
    object_info = {"Sigmas Rescale": {}, "KSampler": {}}
    hint = server._similar_class_hint("SigmasRescale", object_info)
    assert "Sigmas Rescale" in hint


def test_similar_class_hint_empty_when_nothing_close():
    object_info = {"KSampler": {}, "VAEDecode": {}}
    assert server._similar_class_hint("TotallyUnrelatedXyz123", object_info) == ""


# --- prompt_id -> workflow_id attribution (queue etiquette gap) --------------
#
# A live session's run_workflow(wait=False) timed out repeatedly while the
# user's own jobs shared the same queue; manage_queue/get_run_status had only
# the raw ComfyUI prompt_id to go on, so a hung job and "queued behind the
# user's work" looked identical. _State.submitted + _workflow_tag close that
# gap; manage_queue's own attribution is covered in test_round8_execution.py
# (test_manage_queue_attributes_prompts_this_session_queued).


def test_workflow_tag_present_when_submitted():
    server._State.submitted["p1"] = "wf-abc"
    try:
        assert server._workflow_tag("p1") == {"workflow_id": "wf-abc"}
    finally:
        server._State.submitted.clear()


def test_workflow_tag_empty_when_unknown():
    assert server._workflow_tag("never-seen") == {}


def test_record_submission_evicts_oldest_past_cap(monkeypatch):
    monkeypatch.setattr(server, "_SUBMITTED_CAP", 2)
    monkeypatch.setattr(server._State, "submitted", {})
    server._record_submission("a", "wf-a")
    server._record_submission("b", "wf-b")
    server._record_submission("c", "wf-c")
    assert server._State.submitted == {"b": "wf-b", "c": "wf-c"}
