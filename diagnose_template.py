"""Phase 4B.1 template diagnosis: is the missing native tool call caused by the
Ollama template, Ollama's tool-call parser, or the model itself?

R1  raw /api/generate, prompt rendered by hand with Qwen's OFFICIAL chat template
    (tokenizer_config.json), our system prompt. Bypasses Ollama's template AND parser.
R2  same, but Qwen's official default system message ("You are Qwen ...").
C   normal /api/chat with tools (Ollama template + parser), same messages as R1.

Outputs are recorded only. Nothing is executed.
"""
import harness_config as cfg  # noqa: I001  (must be first: scrubs environment)

import asyncio
import json
import urllib.request
from datetime import datetime

from langchain.mcp import MCPAdapter
from langchain_core.utils.function_calling import convert_to_openai_tool

import resource_guard

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
OPTIONS = {"temperature": 0, "seed": cfg.SEED, "num_ctx": cfg.NUM_CTX, "num_predict": cfg.NUM_PREDICT}
CASES = [  # the five Phase 4B requests
    ("demo", "What functions are defined in demo.py?"),
    ("sales", "How many rows are in sales.csv?"),
    ("no_tool", "What is the capital of France?"),
    ("missing", "List the functions in missing_module.py."),
    ("unspecified", "How many rows are in my CSV file?"),
]
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(endpoint: str, body: dict) -> dict:
    resource_guard.check()
    req = urllib.request.Request(f"{cfg.OLLAMA_BASE_URL}/api/{endpoint}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with _opener.open(req, timeout=180) as resp:
        return json.load(resp)


def render_official(system: str, user: str, tools: list[dict]) -> str:
    """Qwen2.5 official chat_template, tools branch, single user turn, add_generation_prompt."""
    tool_lines = "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        f"<|im_start|>system\n{system}"
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        f"{tool_lines}"
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


async def discover() -> list[dict]:
    async with MCPAdapter(cfg.make_mcp_transport()) as adapter:
        tools = cfg.require_allowed_tools(await adapter.list_tools())
    return [convert_to_openai_tool(t) for t in tools]


def main() -> None:
    tools = asyncio.run(discover())
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out = cfg.RESULTS_DIR / f"template_diagnosis_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with out.open("w") as log:
        for cid, user in CASES:
            for label, system in (("R1-raw-official-oursys", cfg.SYSTEM_PROMPT),
                                  ("R2-raw-official-qwensys", QWEN_DEFAULT_SYSTEM)):
                data = post("generate", {"model": cfg.MODEL, "prompt": render_official(system, user, tools),
                                         "raw": True, "stream": False, "options": OPTIONS})
                text = data.get("response", "")
                rec = {"run": label, "case": cid, "has_tool_call_tag": "<tool_call>" in text, "output": text}
                log.write(json.dumps(rec) + "\n")
                print(f"[{label}] {cid:11} tag={rec['has_tool_call_tag']!s:5} output={text!r}")
            data = post("chat", {"model": cfg.MODEL, "stream": False, "options": OPTIONS, "tools": tools,
                                 "messages": [{"role": "system", "content": cfg.SYSTEM_PROMPT},
                                              {"role": "user", "content": user}]})
            msg = data.get("message", {})
            rec = {"run": "C-api-chat", "case": cid, "tool_calls": msg.get("tool_calls"), "content": msg.get("content")}
            log.write(json.dumps(rec) + "\n")
            print(f"[C-api-chat           ] {cid:11} tool_calls={json.dumps(msg.get('tool_calls'))} content={msg.get('content')!r}")
    print("log:", out)


if __name__ == "__main__":
    main()
