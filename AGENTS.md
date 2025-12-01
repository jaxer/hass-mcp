# AI Agent Guide for `hass-mcp`

This document is a quick-start for AI coding agents working on the Hass-MCP Home Assistant integration. It summarizes the repo layout, common workflows, and expectations so future iterations of Codex (or other agents) can stay productive.

## Repository Basics
- **Purpose:** A Model Context Protocol server that lets LLMs interact with a Home Assistant instance (query states, run services, manage labels, etc.).
- **Key modules:**
  - `app/hass.py` – HTTP/WebSocket helpers for talking to Home Assistant.
  - `app/server.py` – MCP tool definitions and prompts wired into FastMCP.
  - `tests/` – Pytest coverage for config, hass helpers, and the server layer.
- **Tooling:** Python 3.13+, [`uv`](https://github.com/astral-sh/uv) for dependency management, and `pytest` for tests (already listed under the `test` extra).

## Working in the Codex CLI
1. **Enter the repo:** `cd /config/hass-mcp`.
2. **Install deps (first run only):** `uv sync --all-extras`. The sandbox already has uv, so no venv activation is needed.
3. **Environment variables:** When running Hass-connected commands, the CLI already injects `HA_URL` and `HA_TOKEN` via `codex-config.toml`. Avoid printing the token.
4. **Preferred file editing:** Use `apply_patch` for manual edits. Only fall back to other techniques for bulk generation or formatting tools.
5. **Encoding:** Keep files ASCII unless the file already uses UTF‑8 characters and there is a clear reason to add more.

## Testing Instructions
Always run the full pytest suite before handing work back unless the change is obviously doc-only.

```bash
cd /config/hass-mcp
uv run pytest
```

The tests cover:
- `tests/test_hass.py` – API helpers (e.g., service calls, label operations, reload helper).
- `tests/test_server.py` – MCP tool wiring, docstrings, prompt availability, search behavior.
- `tests/test_config.py` – Environment variable handling and header generation.

For targeted checks you can run subsets, for example:

```bash
uv run pytest tests/test_hass.py::TestHassAPI::test_reload_home_assistant
```

Document the exact command and summarize results in the final response.

### Testing via the Codex CLI (MCP round-trip)
Whenever you add or modify MCP tools/commands, validate them end-to-end with the Codex CLI so you know clients can actually call them.

1. From the repo root (`/config/hass-mcp`), run Codex against the existing configuration:

   ```bash
   codex exec "list labels in hass"
   ```

   The CLI automatically loads `codex-config.toml`, which already wires this repo up as an MCP server (via `uv run hass-mcp`) and injects `HA_URL` / `HA_TOKEN`.

2. To exercise a new command, replace the quoted prompt with the tool invocation you expect the agent to make, e.g.:

   ```bash
   codex exec "call the new foo_bar tool with ..."
   ```

   Watch the command log: you should see `mcp: home_assistant ready` followed by the specific tool call (e.g., `home_assistant.foo_bar`).

3. If a tool returns structured data, verify Codex renders it (not an "Unexpected response type"). When responses look wrong, re-run with `--json` for raw events:

   ```bash
   codex exec --json "list labels in hass" | jq '.type? // empty'
   ```

4. Capture any CLI failures in your final summary and note whether they are due to HA state (e.g., missing entities) or actual bugs.

Following this flow ensures MCP changes work through the same path the user will take.

## Coding & Review Expectations
- **Do not revert user changes** that already exist in the worktree unless explicitly asked.
- **No destructive git commands** (`reset --hard`, `checkout -- <file>`, etc.).
- **Logging & comments:** Only add concise comments when the code is not self-explanatory. Avoid noisy logging changes unless required for debugging.
- **Doc updates:** Keep README/AGENTS in sync whenever new MCP tools or workflows are added.
- **Pull requests/commits:** If asked to commit, use `git commit -am "<message>"`. Otherwise leave the tree dirty for the user to review.

## Useful Commands & References
- **Inspect HA labels/entities:** use the MCP tools (`list_labels`, `list_entities`, etc.) through the Codex MCP integration; this avoids hand-rolling HTTP calls.
- **Automation traces:** `list_automation_traces` surfaces recent runs (with `run_id`s) and `get_automation_trace` returns the detailed flow graph—perfect for debugging why an automation behaved a certain way.
- **Search the codebase:** `rg 'pattern'` (preferred over `grep` for speed).
- **Format JSON output for docs:** `python -m json.tool` or `jq` if needed (both available in the sandbox).
- **Home Assistant token:** stored in `codex-config.toml`; never echo it in logs or responses.

## When Implementing New Tools
1. Add the helper in `app/hass.py`, guarded with `@handle_api_errors`.
2. Import and expose it via `app/server.py` as an MCP tool (`@mcp.tool()` plus docstring).
3. Update documentation (`README.md` and, if relevant, this `AGENTS.md`).
4. Add or extend tests in `tests/test_hass.py` (API behavior) and `tests/test_server.py` (tool registration/docstring).
5. Run `uv run pytest` and report results.

## Finishing Up
- Summarize changes clearly in the final Codex response, referencing updated files using the `path:line` format when possible.
- Suggest next actions only when there is a natural follow-up (e.g., “deploy container,” “update HA token,” etc.).
- If you create new helper docs/processes, append them here so the next agent starts with better context.

Happy automating!
