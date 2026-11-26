"""Phase 6B: read-only capability probe, started only by `supervisor_5b.py --status` inside the agent's 5C sandbox.

It lists SamToolkit's MCP tools (list_tools only: no tool is ever called), asks the loopback Ollama port which
models are installed (no inference, nothing loaded), and reports package versions. Prints one JSON line.
"""
import harness_config as cfg  # noqa: I001  (must be first)

import asyncio
import json
import re
from importlib import metadata
from urllib.request import ProxyHandler, Request, build_opener

from langchain.mcp import MCPAdapter

SEARCH_TOOLS = cfg.SAMTOOLKIT_ROOT / "src" / "sam_toolkit" / "search_tools.py"


async def mcp_tools() -> dict:
    try:
        async with MCPAdapter(cfg.make_mcp_transport()) as adapter:
            names = sorted(t.name for t in await asyncio.wait_for(adapter.list_tools(), 45))
        return {"ok": set(names) == cfg.ALLOWED_TOOLS, "server": "SamToolkit (stdio, read-only)", "tools": names}
    except Exception as exc:  # reported, never retried
        return {"ok": False, "server": "SamToolkit (stdio, read-only)", "tools": [], "error": type(exc).__name__}


def model_installed() -> dict:
    try:
        req = Request(cfg.OLLAMA_BASE_URL + "/api/tags", method="GET")
        with build_opener(ProxyHandler({})).open(req, timeout=5) as resp:
            names = [m.get("name") for m in json.load(resp).get("models", []) if isinstance(m, dict)]
        return {"ok": cfg.MODEL in names, "model": cfg.MODEL}
    except Exception as exc:
        return {"ok": False, "model": cfg.MODEL, "error": type(exc).__name__}


def samtoolkit_web_search() -> dict:
    """SamToolkit's own Ollama web search: whether its code lock is still closed (read, never called)."""
    try:
        locked = re.search(r"^LIVE_SEARCH_RELEASED = False$", SEARCH_TOOLS.read_text(), re.M) is not None
    except OSError:
        locked = None
    return {"exposed_to_agent": False, "code_lock_closed": locked}


def main() -> None:
    versions = {}
    for package in ("langchain", "langchain-core", "langgraph", "langchain-ollama", "fastmcp"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    out = {"ok": True, "versions": versions, "model": model_installed(), "mcp": asyncio.run(mcp_tools()),
           "samtoolkit_web_search": samtoolkit_web_search()}
    out["ok"] = out["model"]["ok"] and out["mcp"]["ok"]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
