# Reducing permission prompts

Drafting or modernizing a workflow makes **many** tool calls — schema lookups,
validation, layout, registry resolution. If your agent client prompts for approval
on each one, that gets old fast. This page explains what draftsman does to help and
how to grant a durable "approve once" for the safe tools.

## What the server already does

- **Read-only annotations.** Every tool that only *reads* (queries your instance or
  inspects a session workflow) is marked with the MCP `readOnlyHint` annotation.
  Clients that understand these hints can auto-approve reads.
- **Batched schema lookups.** `get_node_info` accepts a **list**:
  `get_node_info(class_types=["KSampler", "CLIPTextEncode", "VAEDecode"])` returns all
  of them in one call instead of one prompt per node. `search_nodes(detail=True)` folds
  each hit's full schema in-line so you often don't need a follow-up call at all.

**But** how often you're prompted is ultimately your *client's* permission policy —
the MCP server cannot approve itself. The rest is a one-time client setting.

## Claude Code: allowlist the read-only tools

How often Claude Code prompts is controlled by `permissions.allow` in your settings
(`~/.claude/settings.json` for user scope, or `.claude/settings.json` in a project).
Add the read-only draftsman tools so they never prompt, while the tools that change
something (queue a render, write to your workflow browser, write learned knowledge)
still ask:

```jsonc
{
  "permissions": {
    "allow": [
      "mcp__comfy-draftsman__get_instance_info",
      "mcp__comfy-draftsman__search_nodes",
      "mcp__comfy-draftsman__get_node_info",
      "mcp__comfy-draftsman__list_models",
      "mcp__comfy-draftsman__list_templates",
      "mcp__comfy-draftsman__inspect_workflow",
      "mcp__comfy-draftsman__lint_workflow",
      "mcp__comfy-draftsman__validate_workflow",
      "mcp__comfy-draftsman__diagnose_workflow",
      "mcp__comfy-draftsman__get_model_guidance",
      "mcp__comfy-draftsman__search_node_packs",
      "mcp__comfy-draftsman__resolve_missing_nodes",
      "mcp__comfy-draftsman__export_workflow_json"
    ]
  }
}
```

The quickest way to create these entries is to pick **"Yes, and don't ask again for
this tool"** the next time Claude Code prompts for one — it writes the same rule.

### Tiers — decide how much to pre-approve

| Tools | What they do | Recommendation |
|---|---|---|
| The 13 read-only tools above | Query the instance / inspect the session workflow. No side effects. | **Allow** — safe to pre-approve. |
| `create_workflow`, `import_workflow`, `edit_workflow`, `organize_workflow`, `port_workflow` | Modify the **in-memory** session workflow only (nothing on disk or the instance yet). | Optional — allow if you don't want to confirm every edit batch. |
| `run_workflow` | **Queues a render** on your ComfyUI (uses the GPU). | Leave prompting, or allow if you're iterating fast. |
| `save_workflow` | **Writes** a file into ComfyUI's workflow browser. | Leave prompting. |
| `record_learning` | **Writes** to your learned-knowledge dir. | Leave prompting. |

### Broadest option

To approve the **entire** server in one rule (all tools, including the mutating ones):

```jsonc
{ "permissions": { "allow": ["mcp__comfy-draftsman"] } }
```

Only do this if you're comfortable with `run_workflow`/`save_workflow` never asking.

> Rule syntax note: entries are `mcp__<server>__<tool>` for a single tool, or
> `mcp__<server>` for the whole server. Confirm against your installed Claude Code
> version if a rule doesn't take effect.
