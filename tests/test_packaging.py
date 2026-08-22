"""Guard the single source of truth for the package version.

The version lives in comfy_draftsman.__version__ and pyproject sources it from
there dynamically. This used to be hand-duplicated and drifted (pyproject 0.5.0
vs __init__ 0.4.2); these tests fail if a static version is reintroduced.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_pyproject_version_is_dynamic():
    project = _pyproject()["project"]
    assert "version" in project.get("dynamic", []), "version must be declared dynamic"
    assert "version" not in project, "no hardcoded version in [project] - it drifts"


def test_hatch_version_sources_from_init():
    data = _pyproject()
    assert data["tool"]["hatch"]["version"]["path"] == "src/comfy_draftsman/__init__.py"


def test_project_urls_cover_the_pypi_sidebar():
    """PyPI renders these as the project's sidebar links; a package with no
    Documentation/Changelog link is a package nobody can navigate from."""
    urls = _pyproject()["project"]["urls"]
    for key in ("Repository", "Issues", "Documentation", "Changelog"):
        assert key in urls, f"[project.urls] is missing {key}"
        assert urls[key].startswith("https://")


def test_console_script_is_declared():
    """`uv tool install comfy-draftsman` is only useful if it puts the server
    on PATH - the MCP client config in the README depends on this entry point."""
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["comfy-draftsman"] == "comfy_draftsman.server:main"


def test_release_workflow_guards_tag_against_version():
    """A tag that disagrees with __version__ publishes a surprise release, and
    PyPI never lets a version number be reused. The guard must stay wired up."""
    workflow = ROOT / ".github" / "workflows" / "release.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "id-token: write" in text, "Trusted Publishing needs OIDC permission"
    assert "password:" not in text, "no stored token - Trusted Publishing only"
    # the guard extracts __version__ with this regex; keep them in lockstep
    pattern = r'(?<=^__version__ = ")[^"]+'
    assert pattern in text
    init = (ROOT / "src" / "comfy_draftsman" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(pattern, init, re.MULTILINE), "regex no longer matches __version__"


def test_release_workflow_creates_a_github_release():
    """A tag is not a Release - GitHub creates one only when asked. Without this
    job, "Watch -> Releases" notifies nobody, which is the only out-of-band
    mechanism users have for learning that a new version exists."""
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "github-release:" in text
    assert "gh release create" in text
    assert "contents: write" in text, "creating a release needs write permission"
    assert "--notes-file notes.md" in text, "release notes come from the CHANGELOG"


def test_github_release_only_fires_on_a_real_tag_push():
    """`workflow_dispatch` can be launched from a tag ref, so a ref check alone
    would let a TestPyPI dry run cut a real Release - notifying every watcher
    about a version that never reached PyPI, and blocking the real tag later."""
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    guard = "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert guard in text


def test_changelog_has_an_entry_for_the_current_version():
    """The release job pulls its notes from the CHANGELOG section matching the
    tag. A missing section still releases, but with a placeholder nobody wants."""
    import comfy_draftsman

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(
        rf"^## {re.escape(comfy_draftsman.__version__)}([^0-9.]|$)",
        changelog,
        re.MULTILINE,
    ), f"CHANGELOG.md has no '## {comfy_draftsman.__version__}' section"
