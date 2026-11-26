"""Step 4: connect SamToolkit's stdio MCP server through langchain.mcp.MCPAdapter.

Discovers the tools, prints their schemas, proves that secrets/flags in this
process do not reach the server, and makes a few EXPLICIT, harmless,
harness-initiated tool calls (never model-proposed).
"""
import harness_config as cfg  # noqa: I001  (must be first: scrubs environment)

import asyncio
import ctypes
import json
import os
import subprocess

from langchain.mcp import MCPAdapter
from langchain_core.utils.function_calling import convert_to_openai_tool

# Canaries: fake values placed in THIS process only. None may reach the server.
CANARIES = {
    "OLLAMA_API_KEY": "canary-NOT-A-REAL-KEY-4b",
    "SAM_TOOLKIT_ENABLE_WEB_SEARCH": "1",
    "SAM_TOOLKIT_ALLOW_LIVE_SEARCH": "1",
}


def _procargs(pid: int) -> str:
    """Read a process's argv + environment via sysctl KERN_PROCARGS2 (what `ps eww` uses).

    `ps` itself is setuid and cannot be launched inside the sandbox, but a process
    may read this for its own user's processes.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
    size = ctypes.c_size_t(1 << 18)
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "sysctl KERN_PROCARGS2 failed")
    return buf.raw[4:size.value].replace(b"\0", b" ").decode(errors="replace")


def child_environment() -> str:
    """Return argv + environment of our single MCP server child process."""
    pids = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout.split()
    server = [s for s in (_procargs(int(p)) for p in pids) if "server.py" in s]
    assert len(server) == 1, f"expected one server child among pids {pids}"
    return server[0]


async def main() -> None:
    os.environ.update(CANARIES)
    adapter = MCPAdapter(cfg.make_mcp_transport())
    async with adapter:
        tools = cfg.require_allowed_tools(await adapter.list_tools())
        child_env = child_environment()
        leaked = [k for k in CANARIES if f"{k}=" in child_env]
        print("server child env has SAM_TOOLKIT_WORKSPACE:", "SAM_TOOLKIT_WORKSPACE=" in child_env)
        print("canaries leaked into server env:", leaked or "none")
        assert not leaked

        print("discovered tools:", sorted(t.name for t in tools))
        for t in tools:
            print(f"\n== {t.name}  ({type(t).__name__}, response_format={t.response_format})")
            print("description:", t.description)
            print("args_schema:", json.dumps(t.args_schema))
            print("as sent to the model:", json.dumps(convert_to_openai_tool(t)))
            print("metadata:", json.dumps(t.metadata))

        by_name = {t.name: t for t in tools}
        # Explicit, harness-initiated connection tests (NOT model-proposed).
        probes = [("inspect_python", "demo.py"), ("inspect_csv", "sales.csv"),
                  ("inspect_python", "missing_module.py"), ("inspect_python", "../server.py")]
        print("\n== explicit connection-test calls")
        for i, (name, path) in enumerate(probes, 1):
            msg = await by_name[name].ainvoke(
                {"type": "tool_call", "id": f"conn-{i}", "name": name, "args": {"path": path}})
            text = " ".join(b.get("text", "") for b in msg.content) if isinstance(msg.content, list) else msg.content
            print(f"{name}({path!r}) -> status={msg.status} artifact={json.dumps(msg.artifact)} text={text[:160]!r}")


if __name__ == "__main__":
    asyncio.run(main())
