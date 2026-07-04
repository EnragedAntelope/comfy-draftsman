# comfy-draftsman

**The MCP server that drafts ComfyUI workflows a human can actually read.**

Local-first [Model Context Protocol](https://modelcontextprotocol.io) server that lets coding agents (Claude Code, Claude Desktop, Cursor, ...) build, repair, port, validate, and run ComfyUI workflows against **your own ComfyUI instance** — and save them as clean, organized, fully-labeled **UI-format** workflows: computed layout, colored semantic groups, titled nodes, and markdown guidance notes that tell a regular person exactly which knobs to touch and which tuned settings to leave alone.

> Most agent tooling for ComfyUI emits raw API-format JSON — a working but unreadable pile of unpositioned nodes. Draftsman's whole reason to exist is the finished drawing: a workflow that opens in ComfyUI looking hand-crafted and self-documenting.

## What it does

- **Draft** — seed from ComfyUI's bundled templates (always current with the latest models) or from scratch, then edit with semantic graph operations.
- **Organize** — the differentiator: layered auto-layout, colored groups by pipeline stage, human titles on every node, and generated notes in two registers: "👇 knobs you're meant to touch" vs "⚙️ tuned settings — leave alone."
- **Diagnose & modernize** — hand it an old broken workflow; it reports every incompatibility against your live instance (renamed nodes, changed widgets, missing custom nodes, missing models) with concrete fixes, resolved via the Comfy Registry.
- **Port** — retarget a workflow across model families (SDXL → Flux/Krea, etc.) with family-correct loaders, CFG, samplers, and resolutions.
- **Validate & prove** — structural + live validation, then an actual render on your instance before you ever open it.
- **Stay current** — ground truth comes from your running ComfyUI (`/object_info`, live templates, live model lists), not from a bundled snapshot.

## Status

Alpha — under active initial development. Not yet on PyPI.

## Requirements

- Python ≥ 3.11
- A running ComfyUI instance (default `http://127.0.0.1:8188`)

## Install (Claude Code example)

```bash
claude mcp add comfy-draftsman -e COMFYUI_URL=http://127.0.0.1:8188 -- uvx --from git+https://github.com/EnragedAntelope/comfy-draftsman comfy-draftsman
```

Full client setup docs, tool reference, and examples coming as the alpha lands.

## Security notes

- Runs over stdio only; the server opens no listening port.
- Talks only to the ComfyUI URL you configure (and, for missing-node resolution, the official Comfy Registry API).
- Installing custom node packs executes third-party code; the corresponding tool is opt-in, requires ComfyUI-Manager, and is flagged destructive so agents must ask first.

## License

MIT
