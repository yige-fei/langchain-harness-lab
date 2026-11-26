"""Shared, locked-down settings for the Phase 4B LangChain experiment.

Import this module FIRST: it scrubs risky environment variables before
LangChain, LangSmith, FastMCP or the Ollama client read them.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# --- 1. Environment hardening (must run before the libraries are imported) -----
# The `ollama` Python client turns OLLAMA_API_KEY into an "authorization: Bearer"
# header on EVERY request, including localhost (ollama/_client.py), and
# OLLAMA_HOST could redirect it. Remove both from this process.
for _name in ("OLLAMA_API_KEY", "OLLAMA_HOST",
              "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
    os.environ.pop(_name, None)
os.environ["LANGSMITH_TRACING"] = "false"       # LangSmith tracing off
os.environ["LANGCHAIN_TRACING_V2"] = "false"    # legacy name, also off
os.environ["FASTMCP_CHECK_FOR_UPDATES"] = "off" # FastMCP's PyPI update check (CLI banner only)

from langchain_core._api import LangChainBetaWarning  # noqa: E402

warnings.filterwarnings("ignore", category=LangChainBetaWarning)  # langchain.mcp is beta; noted in README

from fastmcp.client.transports import StdioTransport  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

# --- 2. Fixed experiment constants --------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  # explicit loopback; never read from env
MODEL = "hf.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M"  # the only installed model
NUM_CTX = 4096
NUM_PREDICT = 256
SEED = 7

SAMTOOLKIT_ROOT = Path.home() / "MCP-Toolbox" / "sam-toolkit-mcp"
SAMTOOLKIT_PYTHON = SAMTOOLKIT_ROOT / ".venv" / "bin" / "python"   # server runs in ITS OWN venv
SAMTOOLKIT_SERVER = SAMTOOLKIT_ROOT / "server.py"
WORKSPACE = SAMTOOLKIT_ROOT / "examples"

ALLOWED_TOOLS = frozenset({"inspect_python", "inspect_csv"})
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SYSTEM_PROMPT = (
    "You help inspect files in a read-only workspace. "
    "File paths are relative to the workspace root, for example 'demo.py' or 'sales.csv'. "
    "Use a tool only when the user names a specific file. "
    "If the user does not name a file, ask which file they mean. Never invent file names. "
    "Answer general questions directly without tools."
)


def make_llm() -> ChatOllama:
    """Local-only chat model: fixed loopback URL, no redirects, no proxies, bounded output."""
    return ChatOllama(
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        seed=SEED,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        validate_model_on_init=True,  # fails fast if the model is not installed locally
        client_kwargs={"follow_redirects": False, "trust_env": False, "timeout": 120},
    )


def make_mcp_transport() -> StdioTransport:
    """Launch SamToolkit's unchanged server.py over stdio with an explicit, minimal env.

    The MCP SDK gives the child only a safe default set (PATH, HOME, ...) plus `env`
    below, so no API key or web-search flag is inherited.
    """
    return StdioTransport(
        command=str(SAMTOOLKIT_PYTHON),
        args=[str(SAMTOOLKIT_SERVER)],
        env={"SAM_TOOLKIT_WORKSPACE": str(WORKSPACE)},
        cwd=str(SAMTOOLKIT_ROOT),
        keep_alive=False,
    )


def require_allowed_tools(tools: list) -> list:
    """Refuse to continue unless discovery returned exactly the two allowed tools."""
    names = {t.name for t in tools}
    if names != ALLOWED_TOOLS:
        raise RuntimeError(f"Unexpected MCP tool set {sorted(names)}; expected {sorted(ALLOWED_TOOLS)}")
    return tools
