"""Session store: workflows held by id, persisted as JSON, reloadable."""

from comfy_draftsman.graph.model import Workflow
from comfy_draftsman.session import Session


def test_create_get_roundtrip(tmp_path):
    session = Session(tmp_path)
    wf = Workflow.new()
    wf.add_node("Note", raw_widgets=["hello"])
    wf_id = session.create(wf, title="my workflow")
    assert session.get(wf_id) is wf
    assert session.title(wf_id) == "my workflow"


def test_persists_to_disk_and_reloads(tmp_path):
    session = Session(tmp_path)
    wf = Workflow.new()
    wf.add_node("Note", raw_widgets=["hello"])
    wf_id = session.create(wf, title="persisted")
    session.persist(wf_id)

    fresh = Session(tmp_path)
    loaded = fresh.get(wf_id)
    assert loaded.nodes[1].widgets_values == ["hello"]
    assert fresh.title(wf_id) == "persisted"


def test_unknown_id_raises(tmp_path):
    session = Session(tmp_path)
    try:
        session.get("nope")
        raise AssertionError("expected KeyError")
    except KeyError as e:
        assert "nope" in str(e)


def test_list_workflows(tmp_path):
    session = Session(tmp_path)
    a = session.create(Workflow.new(), title="a")
    b = session.create(Workflow.new(), title="b")
    listed = session.list()
    assert {x["id"] for x in listed} == {a, b}
