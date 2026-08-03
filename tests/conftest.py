import pytest

from comfy_draftsman.config import Config


@pytest.fixture
def config(tmp_path):
    return Config(
        comfyui_url="http://comfy.test",
        registry_url="http://registry.test",
        session_dir=tmp_path / "sessions",
        request_timeout=5.0,
    )


@pytest.fixture(autouse=True)
def _isolated_submitted_prompts(monkeypatch):
    """server._State.submitted (prompt_id -> workflow_id, for manage_queue/
    get_run_status attribution) is a class-level dict shared across the whole
    test process. Without a reset, a fake prompt_id like "r1" recorded by one
    test's run_workflow call would leak into an unrelated test's manage_queue
    assertion run later in the same session."""
    from comfy_draftsman import server

    monkeypatch.setattr(server._State, "submitted", {})
