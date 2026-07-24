"""Comfy Registry (api.comfy.org) client.

Resolves ComfyUI node class names to the node packs that provide them
(GET /comfy-nodes/{name}/node) and searches packs (GET /nodes/search).
Read-only: installation is left to the user / ComfyUI-Manager, because
custom node packs execute arbitrary code.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from ..config import Config

# The registry is the one REMOTE dependency of an otherwise local-first tool, so
# it is also the one that is routinely unreachable (offline workstation, egress
# firewall, air-gapped render box). Resolving a handful of node classes must
# never take that many serial round trips either.
_RESOLVE_CONCURRENCY = 6


class RegistryUnavailableError(Exception):
    """Couldn't reach the Comfy Registry - offline, blocked, or DNS failure.

    Distinct from "the registry answered, the pack isn't there" (which is a
    None/empty result): callers degrade gracefully on this instead of losing
    the local work they had already computed."""


class RegistryClient:
    def __init__(self, config: Config):
        self._url = config.registry_url
        self._http = httpx.AsyncClient(
            base_url=config.registry_url, timeout=config.request_timeout
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        """GET, converting a failure-to-connect into an actionable error."""
        try:
            return await self._http.get(path, **kwargs)
        except httpx.RequestError as e:
            raise RegistryUnavailableError(
                f"can't reach the Comfy Registry at {self._url}: {type(e).__name__}. "
                "It needs outbound internet access; everything else in draftsman "
                "works offline against your local instance."
            ) from e

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
        response = await self._get(f"/comfy-nodes/{quote(class_type, safe='')}/node")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._pack_info(response.json())

    async def resolve_node_classes(self, class_types: list[str]) -> dict[str, Any]:
        """Resolve many class names at once. Lookups run concurrently (bounded):
        a diagnose on a workflow with a dozen missing classes was a dozen serial
        round trips to a remote host."""
        semaphore = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

        async def lookup(class_type: str) -> tuple[str, dict[str, Any] | None]:
            async with semaphore:
                return class_type, await self.resolve_node_class(class_type)

        results = await asyncio.gather(*(lookup(c) for c in class_types))
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        packs: dict[str, dict[str, Any]] = {}
        for class_type, info in results:
            if info is None:
                unresolved.append(class_type)
            else:
                resolved[class_type] = info["pack_id"]
                packs[info["pack_id"]] = info
        return {"resolved": resolved, "unresolved": unresolved, "packs": list(packs.values())}

    async def search_packs(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        response = await self._get("/nodes/search", params={"search": query, "limit": limit})
        response.raise_for_status()
        return [self._pack_info(n) for n in response.json().get("nodes", [])]
