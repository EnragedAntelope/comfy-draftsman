"""Guard the single source of truth for the package version.

The version lives in comfy_draftsman.__version__ and pyproject sources it from
there dynamically. This used to be hand-duplicated and drifted (pyproject 0.5.0
vs __init__ 0.4.2); these tests fail if a static version is reintroduced.
"""

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
