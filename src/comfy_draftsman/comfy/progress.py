"""Background progress tracking for non-blocking runs (run_workflow wait=False).

ComfyUI only reports step-level progress over the /ws event stream, and it
routes per-prompt events to the socket whose clientId queued the prompt. The
tracker therefore owns its OWN client id: non-blocking prompts are queued
under the tracker's id so its long-lived socket receives their events, while
run_and_wait's per-call socket (the server's main client id) stays untouched
- two sockets sharing one clientId would displace each other in ComfyUI's
socket table.

Progress-tool concept credit: KerbalTheGathering/ComfyUI_MCP (independently
implemented; see README acknowledgments).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import websockets

_MAX_ENTRIES = 20
_RECONNECT_DELAY = 2.0


class ProgressTracker:
    """Listens on /ws and keeps the latest execution state per prompt_id.

    Best-effort by design: if the socket is down, snapshots say so and the
    caller falls back to /queue + /history, which remain authoritative.
    """

    def __init__(self, ws_url_factory: Callable[[str], str]):
        self._ws_url_factory = ws_url_factory
        self.client_id = uuid.uuid4().hex
        self._states: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._task: asyncio.Task[None] | None = None
        self.connected = False

    def ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._listen())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _listen(self) -> None:
        while True:
            try:
                url = self._ws_url_factory(self.client_id)
                async with websockets.connect(url, max_size=32 * 1024 * 1024) as ws:
                    self.connected = True
                    async for frame in ws:
                        if isinstance(frame, bytes):  # preview image frames
                            continue
                        self.handle_event(json.loads(frame))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(
                    f"[draftsman] progress ws disconnected ({type(e).__name__}): {e}",
                    file=sys.stderr,
                )
                # reconnect below; /history remains the source of truth
            finally:
                self.connected = False
            await asyncio.sleep(_RECONNECT_DELAY)

    def _state(self, prompt_id: str) -> dict[str, Any]:
        state = self._states.get(prompt_id)
        if state is None:
            state = {"status": "pending"}
            self._states[prompt_id] = state
            while len(self._states) > _MAX_ENTRIES:
                self._states.popitem(last=False)
        else:
            self._states.move_to_end(prompt_id)
        return state

    def handle_event(self, event: dict[str, Any]) -> None:
        data = event.get("data") or {}
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return
        kind = event.get("type")
        state = self._state(prompt_id)
        if kind == "execution_start":
            state["status"] = "running"
        elif kind == "executing":
            if data.get("node") is None:
                state["status"] = "finished"
            else:
                state["status"] = "running"
                state["node"] = data.get("node")
        elif kind == "progress":
            total = data.get("max") or 0
            state.update(
                status="running",
                node=data.get("node", state.get("node")),
                step=data.get("value"),
                total=total,
                percent=round(100.0 * data.get("value", 0) / total, 1) if total else None,
            )
        elif kind == "execution_success":
            state["status"] = "finished"
        elif kind == "execution_error":
            state["status"] = "error"
            state["error"] = {
                "node_id": data.get("node_id"),
                "node_type": data.get("node_type"),
                "message": data.get("exception_message"),
                "type": data.get("exception_type"),
            }
        elif kind == "execution_interrupted":
            state["status"] = "error"
            state["error"] = {"message": "interrupted"}

    def snapshot(self, prompt_id: str) -> dict[str, Any]:
        state = dict(self._states.get(prompt_id) or {})
        state["ws_connected"] = self.connected
        return state
