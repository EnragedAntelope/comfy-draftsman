"""Async HTTP client for a local ComfyUI instance.

Endpoint shapes verified against ComfyUI 0.27.0 (July 2026):
- GET  /system_stats            -> {"system": {...}, "devices": [...]}
- GET  /object_info             -> {class_type: schema, ...}  (~2.5 MB; cached here)
- GET  /models                  -> [folder, ...]
- GET  /models/{folder}         -> [filename, ...]
- GET  /templates/index.json    -> [{"moduleName", "templates": [...]}, ...]
- GET  /templates/{name}.json   -> UI-format workflow JSON (schema 0.4)
- POST /prompt                  -> {"prompt_id", "number"} | 400 {"error", "node_errors"}
- GET  /history/{prompt_id}     -> {prompt_id: {...}}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from urllib.parse import quote

import httpx
import websockets

from ..config import Config


class ComfyValidationError(Exception):
    """ComfyUI rejected a prompt at validation time."""

    def __init__(self, error: dict[str, Any], node_errors: dict[str, Any]):
        self.error = error
        self.node_errors = node_errors
        message = error.get("message", "prompt validation failed")
        super().__init__(f"{message}; node_errors={list(node_errors)}")


class ComfyConnectionError(Exception):
    """Couldn't reach the ComfyUI instance at all - down, wrong URL, or blocked.

    Distinct from ComfyValidationError (reached it, prompt rejected): a raw httpx
    ConnectError/timeout is opaque to an agent, so we turn it into an actionable
    message naming the URL and the likely fixes."""


class ComfyClient:
    def __init__(self, config: Config):
        self._config = config
        self.base_url = config.comfyui_url
        self.client_id = uuid.uuid4().hex
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=config.request_timeout)
        self._object_info: dict[str, Any] | None = None

    async def close(self) -> None:
        await self._http.aclose()

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue a request, converting a failure-to-connect into a clear error.

        Only transport-level failures (can't connect, DNS, connect timeout) become
        ComfyConnectionError; an HTTP response - even a 4xx/5xx - is returned for
        the caller's own raise_for_status/status handling."""
        try:
            return await self._http.request(method, path, **kwargs)
        except httpx.RequestError as e:
            raise ComfyConnectionError(
                f"can't reach ComfyUI at {self.base_url}: {type(e).__name__}. "
                "Is ComfyUI running and is COMFYUI_URL correct? A remote instance "
                "must be started with --listen and reachable from this host."
            ) from e

    async def _get_json(self, path: str) -> Any:
        response = await self._send("GET", path)
        response.raise_for_status()
        return response.json()

    async def get_system_stats(self) -> dict[str, Any]:
        return await self._get_json("/system_stats")

    async def get_object_info(self, refresh: bool = False) -> dict[str, Any]:
        if self._object_info is None or refresh:
            self._object_info = await self._get_json("/object_info")
        return self._object_info

    async def list_model_folders(self) -> list[str]:
        return await self._get_json("/models")

    async def list_models(self, folder: str) -> list[str]:
        return await self._get_json(f"/models/{quote(folder, safe='')}")

    async def get_model_metadata(self, folder: str, filename: str) -> dict[str, Any]:
        """Embedded safetensors __metadata__ via /view_metadata. ComfyUI 404s
        for missing files, non-.safetensors, and headers without metadata."""
        clean = filename.replace("\\", "/")
        if clean.startswith("/") or ".." in clean.split("/"):
            raise ValueError(f"invalid filename: {filename!r}")
        response = await self._send(
            "GET", f"/view_metadata/{quote(folder, safe='')}", params={"filename": filename}
        )
        if response.status_code == 404:
            raise FileNotFoundError(filename)
        response.raise_for_status()
        return response.json()

    async def get_template_index(self) -> list[dict[str, Any]]:
        return await self._get_json("/templates/index.json")

    async def get_template_workflow(self, name: str) -> dict[str, Any]:
        clean = name.replace("\\", "/")
        if clean.startswith("/") or ".." in clean.split("/"):
            raise ValueError(f"invalid template name: {name!r}")
        return await self._get_json(f"/templates/{name}.json")

    async def queue_prompt(
        self,
        api_prompt: dict[str, Any],
        extra_data: dict[str, Any] | None = None,
        client_id: str | None = None,
        front: bool = False,
    ) -> dict[str, Any]:
        # client_id override: non-blocking runs queue under the ProgressTracker's
        # id so ITS socket receives the per-prompt events (ComfyUI routes them
        # to the queuing client's socket only).
        payload: dict[str, Any] = {"prompt": api_prompt, "client_id": client_id or self.client_id}
        if front:
            # run next after the current job finishes; pending jobs stay queued
            payload["front"] = True
        if extra_data:
            payload["extra_data"] = extra_data
        response = await self._send("POST", "/prompt", json=payload)
        if response.status_code == 400:
            body = response.json()
            raise ComfyValidationError(body.get("error", {}), body.get("node_errors", {}))
        response.raise_for_status()
        return response.json()

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        history = await self._get_json(f"/history/{prompt_id}")
        return history.get(prompt_id, {})

    async def get_queue(self) -> dict[str, Any]:
        return await self._get_json("/queue")

    async def interrupt(self) -> None:
        response = await self._send("POST", "/interrupt")
        response.raise_for_status()

    async def clear_queue(self) -> None:
        response = await self._send("POST", "/queue", json={"clear": True})
        response.raise_for_status()

    async def delete_queue_items(self, prompt_ids: list[str]) -> None:
        response = await self._send("POST", "/queue", json={"delete": prompt_ids})
        response.raise_for_status()

    async def free(self, unload_models: bool = False, free_memory: bool = True) -> None:
        response = await self._send(
            "POST", "/free", json={"unload_models": unload_models, "free_memory": free_memory}
        )
        response.raise_for_status()

    async def upload_image(
        self,
        data: bytes,
        name: str,
        subfolder: str = "",
        overwrite: bool = False,
        image_type: str = "input",
    ) -> dict[str, Any]:
        """POST /upload/image -> {"name", "subfolder", "type"}."""
        form: dict[str, str] = {"type": image_type, "overwrite": "true" if overwrite else "false"}
        if subfolder:
            form["subfolder"] = subfolder
        response = await self._send(
            "POST", "/upload/image", files={"image": (name, data)}, data=form
        )
        response.raise_for_status()
        return response.json()

    async def upload_mask(
        self,
        data: bytes,
        name: str,
        original_ref: dict[str, Any],
        subfolder: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """POST /upload/mask; original_ref names the image the mask belongs to."""
        form: dict[str, str] = {
            "type": "input",
            "overwrite": "true" if overwrite else "false",
            "original_ref": json.dumps(original_ref),
        }
        if subfolder:
            form["subfolder"] = subfolder
        response = await self._send(
            "POST", "/upload/mask", files={"image": (name, data)}, data=form
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _workflow_userdata_path(name: str) -> str:
        """Relative userdata path for a workflow browser file, refusing traversal."""
        clean = name.replace("\\", "/").strip("/")
        if not clean or ".." in clean.split("/"):
            raise ValueError(f"invalid workflow name: {name!r}")
        filename = clean if clean.endswith(".json") else f"{clean}.json"
        return f"workflows/{filename}"

    async def list_userdata_workflows(self) -> list[str]:
        """Workflow files in ComfyUI's workflow browser (userdata), incl. subdirs."""
        response = await self._send(
            "GET", "/api/userdata", params={"dir": "workflows", "recurse": "true", "split": "false"}
        )
        if response.status_code == 404:  # no workflows dir yet on a fresh instance
            return []
        response.raise_for_status()
        return [p.replace("\\", "/") for p in response.json()]

    async def get_userdata_workflow(self, name: str) -> dict[str, Any]:
        """Load one workflow browser file by name (with or without .json).

        Raises FileNotFoundError if it doesn't exist, ValueError on a name that
        would escape the workflows directory.
        """
        path = self._workflow_userdata_path(name)
        response = await self._send("GET", f"/api/userdata/{quote(path, safe='')}")
        if response.status_code == 404:
            raise FileNotFoundError(name)
        response.raise_for_status()
        return response.json()

    async def save_userdata_workflow(
        self, name: str, document: dict[str, Any], overwrite: bool = False
    ) -> str:
        """Save a UI-format workflow into ComfyUI's workflow browser (userdata API).

        With overwrite=False (default) ComfyUI answers 409 if the file exists,
        surfaced here as FileExistsError so callers can pick another name.
        """
        filename = name if name.endswith(".json") else f"{name}.json"
        response = await self._send(
            "POST",
            f"/api/userdata/{quote(f'workflows/{filename}', safe='')}",
            params={"overwrite": "true" if overwrite else "false"},
            json=document,
        )
        if response.status_code == 409:
            raise FileExistsError(filename)
        response.raise_for_status()
        return filename

    #: output keys holding {filename, subfolder, type} FILE refs - the only ones
    #: /view can fetch and _relocate_outputs can move.
    FILE_OUTPUT_KEYS = ("images", "gifs", "videos", "audio")

    #: non-file keys that carry no information for the caller: `animated` is a
    #: per-image bool list the frontend uses to pick a player.
    _DATA_KEYS_IGNORED = frozenset({"animated"})

    # A node can return anything JSON-serializable, and some return a lot (an
    # entire generated prompt, a batch's worth of paths). These are explicitly
    # requested run results rather than a recurring summary, so the budget is
    # generous - but it is still a budget.
    _DATA_VALUE_CHARS = 1200
    _DATA_TOTAL_CHARS = 5000

    @staticmethod
    def _collect_outputs(history: dict[str, Any]) -> list[dict[str, Any]]:
        """FILE outputs only, flattened to relocatable {filename, subfolder, type}
        refs. Non-file return values go through _collect_data_outputs."""
        outputs: list[dict[str, Any]] = []
        for node_id, node_output in (history.get("outputs") or {}).items():
            for key in ComfyClient.FILE_OUTPUT_KEYS:
                for item in node_output.get(key, []) or []:
                    outputs.append({**item, "node_id": node_id, "kind": key})
        return outputs

    @classmethod
    def _collect_data_outputs(cls, history: dict[str, Any]) -> dict[str, Any]:
        """Every NON-file value an output node returned, keyed by node id.

        Output nodes are free to return anything in their `ui` dict, and plenty
        do: ShowText-style nodes return `text`, and save nodes commonly return the
        paths they wrote (`filenames`, `path`, `saved_count`). Harvesting only the
        four file keys meant those results were silently dropped - a workflow whose
        product was a generated string, or whose save node reported where it wrote,
        looked like it produced nothing at all.

        Values are truncated per item and the whole payload is budgeted; a `note`
        records anything cut so a short value is never mistaken for the full one.
        """
        collected: dict[str, Any] = {}
        used = 0
        truncated: list[str] = []
        for node_id, node_output in (history.get("outputs") or {}).items():
            if not isinstance(node_output, dict):
                continue
            for key, value in node_output.items():
                if key in cls.FILE_OUTPUT_KEYS or key in cls._DATA_KEYS_IGNORED:
                    continue
                rendered = value if isinstance(value, str) else json.dumps(value, default=str)
                # over-long values degrade to a clipped STRING; anything within
                # budget keeps its original shape (list/dict/number)
                kept: Any = value
                if len(rendered) > cls._DATA_VALUE_CHARS:
                    kept = rendered[: cls._DATA_VALUE_CHARS] + "…"
                    truncated.append(f"{node_id}.{key}")
                cost = len(kept) if isinstance(kept, str) else len(rendered)
                if used + cost > cls._DATA_TOTAL_CHARS:
                    truncated.append(f"{node_id}.{key} (omitted)")
                    continue
                used += cost
                collected.setdefault(str(node_id), {})[key] = kept
        if truncated:
            collected["note"] = (
                "clipped to keep the response bounded: " + ", ".join(truncated[:10])
            )
        return collected

    def _ws_url(self, client_id: str | None = None) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[1]
        return f"{scheme}://{host}/ws?clientId={client_id or self.client_id}"

    async def run_and_wait(
        self,
        api_prompt: dict[str, Any],
        timeout: float = 600.0,
        extra_data: dict[str, Any] | None = None,
        front: bool = False,
    ) -> dict[str, Any]:
        """Queue a prompt and wait for completion via the /ws event stream.

        Returns {"status", "prompt_id", "outputs" (image/video/audio files from
        history), "error"?}.
        """
        async with websockets.connect(self._ws_url(), max_size=32 * 1024 * 1024) as ws:
            queued = await self.queue_prompt(api_prompt, extra_data=extra_data, front=front)
            prompt_id = queued["prompt_id"]
            # ComfyUI accepts a prompt (HTTP 200 + prompt_id) yet still returns
            # node_errors for nodes it rejected at validation, executing only the
            # rest of the graph. Capture them so a partial run isn't reported as a
            # clean success with mysteriously-empty outputs.
            partial_node_errors = queued.get("node_errors") or {}
            error: dict[str, Any] | None = None
            async with asyncio.timeout(timeout):
                while True:
                    frame = await ws.recv()
                    if isinstance(frame, bytes):  # preview image frames
                        continue
                    event = json.loads(frame)
                    data = event.get("data", {})
                    if data.get("prompt_id") not in (None, prompt_id):
                        continue
                    kind = event.get("type")
                    if kind == "execution_error":
                        error = data
                        break
                    if kind == "execution_interrupted":
                        error = {"exception_message": "interrupted"}
                        break
                    finished = kind == "execution_success" or (
                        kind == "executing" and data.get("node") is None
                    )
                    if finished and data.get("prompt_id") == prompt_id:
                        break
        # /history can lag the execution_success event by a beat; on a clean run
        # with no outputs yet, re-poll briefly before trusting an empty list.
        outputs: list[dict[str, Any]] = []
        data_outputs: dict[str, Any] = {}
        for attempt in range(5):
            history = await self.get_history(prompt_id)
            outputs = self._collect_outputs(history)
            data_outputs = self._collect_data_outputs(history)
            if (
                outputs
                or data_outputs
                or error
                or history.get("status", {}).get("completed") is False
            ):
                break
            if attempt < 4:
                await asyncio.sleep(0.2 * (attempt + 1))
        result: dict[str, Any] = {
            "status": "error" if error else "success",
            "prompt_id": prompt_id,
            "outputs": outputs,
        }
        if data_outputs:
            # non-file return values (generated text, written paths, counts) -
            # omitted entirely when empty so the common case costs nothing
            result["data_outputs"] = data_outputs
        if error:
            result["error"] = {
                "node_id": error.get("node_id"),
                "node_type": error.get("node_type"),
                "message": error.get("exception_message"),
                "type": error.get("exception_type"),
            }
        if partial_node_errors:
            result["node_errors"] = partial_node_errors
        return result

    async def fetch_output(self, item: dict[str, Any]) -> bytes:
        response = await self._send("GET", "/view", params={
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
        response.raise_for_status()
        return response.content
