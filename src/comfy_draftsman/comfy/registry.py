"""Comfy Registry (api.comfy.org) client.

Resolves ComfyUI node class names to the node packs that provide them
(GET /comfy-nodes/{name}/node) and searches packs (GET /nodes/search).
Read-only: installation is left to the user / ComfyUI-Manager, because
custom node packs execute arbitrary code.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from ..config import Config


class RegistryClient:
    def __init__(self, config: Config):
        self._http = httpx.AsyncClient(
            base_url=config.registry_url, timeout=config.request_timeout
        )

    async def close(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _pack_info(data: dict[str, Any]) -> dict[str, Any]:
        pack_id = data.get("id", "")
        return {
            "pack_id": pack_id,
            "name": data.get("name", pack_id),
            "description": (data.get("description") or "")[:300],
            "repository": data.get("repository", ""),
            "downloads": data.get("downloads"),
            "latest_version": (data.get("latest_version") or {}).get("version"),
            "registry_url": f"https://registry.comfy.org/nodes/{pack_id}",
            "install_hint": (
                f"comfy node install {pack_id} (comfy-cli), or search '{pack_id}' in "
                "ComfyUI-Manager. Custom nodes run arbitrary code - review the repo first."
            ),
        }

    async def resolve_node_class(self, class_type: str) -> dict[str, Any] | None:
        response = await self._http.get(f"/comfy-nodes/{quote(class_type, safe='')}/node")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._pack_info(response.json())

    async def resolve_node_classes(self, class_types: list[str]) -> dict[str, Any]:
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        packs: dict[str, dict[str, Any]] = {}
        for class_type in class_types:
            info = await self.resolve_node_class(class_type)
            if info is None:
                unresolved.append(class_type)
            else:
                resolved[class_type] = info["pack_id"]
                packs[info["pack_id"]] = info
        return {"resolved": resolved, "unresolved": unresolved, "packs": list(packs.values())}

    async def search_packs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        response = await self._http.get("/nodes/search", params={"search": query, "limit": limit})
        response.raise_for_status()
        return [self._pack_info(n) for n in response.json().get("nodes", [])]
