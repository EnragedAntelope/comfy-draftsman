"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
REGISTRY_URL = "https://api.comfy.org"


def _default_session_dir() -> Path:
    # Never derive from cwd: MCP hosts (Claude Desktop) launch servers with
    # cwd set to a system directory (e.g. C:\Windows\System32) where writes
    # need admin rights.
    return Path(
        os.environ.get("DRAFTSMAN_SESSION_DIR", Path.home() / ".comfy-draftsman" / "sessions")
    )


def _default_learned_dir() -> Path:
    return Path(
        os.environ.get("DRAFTSMAN_LEARNED_DIR", Path.home() / ".comfy-draftsman" / "learned")
    )


def _default_mount_dir() -> Path | None:
    # A folder the caller (e.g. a Claude Desktop / Cowork sandbox) can reach,
    # where save_output relocates finished renders out of ComfyUI's output tree.
    # Unset -> no default; save_output then needs an explicit dest_dir.
    value = os.environ.get("COMFYUI_MOUNT_DIR")
    return Path(value) if value else None


@dataclass(frozen=True)
class Config:
    """Runtime configuration, resolved once at server start."""

    comfyui_url: str = field(default_factory=lambda: os.environ.get("COMFYUI_URL", DEFAULT_COMFYUI_URL).rstrip("/"))
    registry_url: str = field(default_factory=lambda: os.environ.get("COMFY_REGISTRY_URL", REGISTRY_URL).rstrip("/"))
    session_dir: Path = field(default_factory=_default_session_dir)
    learned_dir: Path = field(default_factory=_default_learned_dir)
    mount_dir: Path | None = field(default_factory=_default_mount_dir)
    request_timeout: float = field(default_factory=lambda: float(os.environ.get("DRAFTSMAN_TIMEOUT", "30")))


def load_config() -> Config:
    return Config()
