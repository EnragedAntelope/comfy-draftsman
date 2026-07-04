"""In-memory workflow store with JSON persistence.

Agents edit workflows incrementally by id instead of resending full JSON on
every tool call. ``persist``ed workflows survive server restarts.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

from .graph.model import Workflow


class Session:
    def __init__(self, directory: Path | str):
        self._dir = Path(directory)
        self._workflows: dict[str, Workflow] = {}
        self._titles: dict[str, str] = {}

    def create(self, wf: Workflow, title: str = "untitled") -> str:
        wf_id = secrets.token_hex(4)
        self._workflows[wf_id] = wf
        self._titles[wf_id] = title
        return wf_id

    def get(self, wf_id: str) -> Workflow:
        if wf_id not in self._workflows and not self._load(wf_id):
            raise KeyError(f"no workflow with id '{wf_id}'")
        return self._workflows[wf_id]

    def title(self, wf_id: str) -> str:
        self.get(wf_id)
        return self._titles.get(wf_id, "untitled")

    def set_title(self, wf_id: str, title: str) -> None:
        self._titles[wf_id] = title

    def list(self) -> list[dict[str, Any]]:
        entries = [
            {"id": wf_id, "title": self._titles.get(wf_id, "untitled")}
            for wf_id in self._workflows
        ]
        if self._dir.is_dir():
            for path in self._dir.glob("*.json"):
                wf_id = path.stem
                if wf_id not in self._workflows:
                    entries.append({"id": wf_id, "title": self._peek_title(path)})
        return entries

    def persist(self, wf_id: str) -> Path:
        wf = self.get(wf_id)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{wf_id}.json"
        document = wf.to_ui()
        document.setdefault("extra", {})["draftsman_title"] = self._titles.get(wf_id, "untitled")
        path.write_text(json.dumps(document, indent=1), encoding="utf-8")
        return path

    def _load(self, wf_id: str) -> bool:
        path = self._dir / f"{wf_id}.json"
        if not path.is_file():
            return False
        document = json.loads(path.read_text(encoding="utf-8"))
        self._workflows[wf_id] = Workflow.from_ui(document)
        self._titles[wf_id] = document.get("extra", {}).get("draftsman_title", "untitled")
        return True

    def _peek_title(self, path: Path) -> str:
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get(
                "draftsman_title", "untitled"
            )
        except (OSError, json.JSONDecodeError):
            return "untitled"
