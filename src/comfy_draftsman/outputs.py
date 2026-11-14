"""Moving finished renders out of ComfyUI onto the local filesystem.

Pure file handling (no MCP, no session state): server.py passes in the client
and the configured mount dir.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from pathlib import Path
from typing import Any

from .comfy.client import ComfyClient

SWEEP_SHEET_MAX_AGE_S = 30 * 86400


def resolve_dest(dest_dir: str, mount_dir: Path | None) -> tuple[Path | None, str | None]:
    """Resolve+create the relocation directory. dest_dir empty -> mount_dir
    (COMFYUI_MOUNT_DIR). A relative path is refused: this server's cwd is NOT the
    caller's (MCP hosts often launch it from a system dir like System32), so a
    relative path would resolve somewhere invisible. Returns (path, None) or
    (None, error)."""
    root = Path(dest_dir) if dest_dir else mount_dir
    if root is None:
        return None, (
            "no destination: pass save_dir/dest_dir, or set COMFYUI_MOUNT_DIR so "
            "outputs relocate to a folder the caller can reach"
        )
    root = root.expanduser()  # ~ expands to an absolute path; ./foo does not
    if not root.is_absolute():
        return None, (
            f"destination must be an absolute path (got {str(root)!r}): the server's "
            "working directory is not the agent's, so a relative path would resolve "
            "somewhere invisible - pass an absolute save_dir/dest_dir (or set "
            "COMFYUI_MOUNT_DIR to an absolute folder both sides can reach)"
        )
    try:
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return None, f"destination directory unusable: {e}"
    return root, None


def dedupe_path(path: Path) -> Path:
    """`path`, or `path` with a numeric suffix if it already exists."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for i in range(1, 10000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    # every numbered name taken: a random suffix still never overwrites
    return parent / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def safe_dest(root: Path, name: str) -> Path | None:
    """Join a filename under root, refusing anything that escapes it."""
    target = (root / Path(name).name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


async def relocate_outputs(
    client: ComfyClient,
    items: list[dict[str, Any]],
    dest_root: Path,
    dest_filename: str | None = None,
    overwrite: bool = False,
) -> tuple[list[str], list[str]]:
    """Fetch each output image's bytes and write them under dest_root. Returns
    (saved_paths, errors)."""
    saved: list[str] = []
    errors: list[str] = []
    for item in items:
        filename = item.get("filename", "")
        try:
            data = await client.fetch_output(item)
        except Exception as e:  # surface the fetch failure per file, keep going
            errors.append(f"fetch {filename!r}: {e}")
            continue
        target = safe_dest(dest_root, dest_filename or filename)
        if target is None:
            errors.append(f"unsafe destination name: {dest_filename or filename!r}")
            continue
        if not overwrite:
            target = dedupe_path(target)
        try:
            target.write_bytes(data)
        except OSError as e:
            errors.append(f"write {target}: {e}")
            continue
        saved.append(str(target))
    return saved, errors


def prune_sweep_sheets(root: Path) -> None:
    """Delete draftsman's own sweep sheets older than 30 days from the default
    sweeps dir (never a caller's save_dir) - nothing else ever removes them."""
    cutoff = time.time() - SWEEP_SHEET_MAX_AGE_S
    for prefix in ("contact-", "crops-", "crop-guide-"):
        for sheet in root.glob(f"{prefix}*.png"):
            with contextlib.suppress(OSError):
                if sheet.stat().st_mtime < cutoff:
                    sheet.unlink()
