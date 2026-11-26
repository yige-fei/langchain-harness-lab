"""Step 3: verify ordinary local chat through ChatOllama (no tools)."""
import harness_config as cfg  # noqa: I001  (must be first: scrubs environment)

import json
import os


def main() -> None:
    llm = cfg.make_llm()
    http = llm._client._client  # the underlying httpx.Client, inspected read-only
    checks = {
        "base_url": str(http.base_url),
        "authorization_header_present": "authorization" in {k.lower() for k in http.headers},
        "follow_redirects": http.follow_redirects,
        "trust_env (proxy env vars honoured)": http._trust_env,
        "OLLAMA_API_KEY in env": "OLLAMA_API_KEY" in os.environ,
        "LANGSMITH_TRACING": os.environ.get("LANGSMITH_TRACING"),
    }
    print("client checks:", json.dumps(checks))
    assert checks["base_url"].startswith("http://127.0.0.1:11434")
    assert not checks["authorization_header_present"] and not checks["follow_redirects"]

    reply = llm.invoke([("user", "Reply with exactly these words: local chat ok")])
    meta = reply.response_metadata
    print("reply content:", repr(reply.content))
    print("tool_calls:", reply.tool_calls)
    print("metadata:", json.dumps({k: meta.get(k) for k in (
        "model_name", "done_reason", "total_duration", "load_duration",
        "prompt_eval_count", "eval_count", "eval_duration")}))


if __name__ == "__main__":
    main()
