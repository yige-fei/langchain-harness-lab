#!/bin/zsh
# Run a harness script under a macOS sandbox that blocks ALL outbound network
# traffic except the local Ollama port. Child processes (the MCP server) inherit it.
set -eu
cd "${0:A:h}"
PROFILE='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:11434"))(allow network-outbound (remote unix-socket))'
exec sandbox-exec -p "$PROFILE" .venv/bin/python "$@"
