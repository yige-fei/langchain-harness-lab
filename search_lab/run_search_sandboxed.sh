#!/bin/zsh
# Run the search probe in a macOS sandbox: outbound TCP only to port 443 (HTTPS) plus the
# system DNS resolver socket. Everything else outbound (including localhost/Ollama) is denied.
set -eu
cd "${0:A:h}/.."
PROFILE='(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote tcp "*:443"))(allow network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))'
exec sandbox-exec -p "$PROFILE" .venv/bin/python -I search_lab/ddgs_probe.py "$@"
