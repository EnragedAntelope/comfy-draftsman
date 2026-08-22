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


def test_release_workflow_is_least_privilege_by_default():
    """Without a top-level `permissions` block, every job inherits the
    repository default - read/write on all scopes on an older repo - and hands
    it to third-party actions. Each job that needs more must widen it itself."""
    import yaml

    data = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))
    assert data["permissions"] == {"contents": "read"}
    assert data["jobs"]["publish"]["permissions"] == {"id-token": "write"}
    assert data["jobs"]["github-release"]["permissions"] == {"contents": "write"}
    assert "permissions" not in data["jobs"]["build"], "build needs no token at all"


def test_runtime_dependencies_are_upper_bounded():
    """An unbounded pin is invisible to every locked job - uv.lock keeps dev and
    CI on a working version while a fresh `pip install` resolves the new major
    and fails on import. That is exactly how 0.15.0 shipped unstartable: mcp
    2.0.0 removed `mcp.server.fastmcp`, which server.py imports at module scope,
    and `mcp>=1.10` happily resolved it.

    Only `mcp` is required to carry a ceiling, and deliberately so. It is the
    one dependency with a *demonstrated* major-version break in an import
    server.py performs at module scope. The rest stay open on purpose - this
    repo does not synthesize a requirement it cannot point at, and an invented
    ceiling causes resolution conflicts for users to prevent a break nobody has
    observed. pydantic in particular needs no direct cap: mcp already requires
    `pydantic<3.0.0`, so a second copy here would be redundant and could drift
    out of step with it."""
    deps = _pyproject()["project"]["dependencies"]
    pins = {d.split(">=")[0].split("[")[0].strip(): d for d in deps}
    assert "mcp" in pins, "mcp is no longer a declared dependency"
    assert "<" in pins["mcp"], (
        "mcp has no upper bound - a new major can break a fresh install while "
        f"uv.lock hides it from dev and CI. Got: {pins['mcp']!r}"
    )


def test_the_wheel_check_imports_the_server_module():
    """Both wheel checks resolve deps against the index rather than uv.lock, so
    they are the only place a bad pin can surface. Importing only the knowledge
    package is what let mcp 2.0.0 through - the server import is the assertion
    that matters, so it must stay wired up in both workflows."""
    for name in ("ci.yml", "release.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "import comfy_draftsman.server" in text, f"{name} lost the server import"
