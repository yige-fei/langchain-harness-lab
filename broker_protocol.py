"""Phase 5B: tiny message protocol between the sandboxed agent and the supervisor.

One JSON object per line over two inherited pipes (no sockets, no files, nothing another process
can connect to). Standard library only. Every message is bounded in size and must be an object
with a string "type".

agent -> supervisor                         supervisor -> agent
  {"type": "approval_request", "card": {...}}   {"type": "approval_response", "decision", "proposal_id",
                                                 "digest", "via", "note"}
  {"type": "search", "thread_id", "proposal_id",  {"type": "search_result", "record": {...}}
   "digest", "query"}
  {"type": "done", ...}                         (no reply)
"""
from __future__ import annotations

import json
import os
import threading

MAX_MESSAGE_BYTES = 262_144
AGENT_TYPES = frozenset({"approval_request", "search", "done"})
SUPERVISOR_TYPES = frozenset({"approval_response", "search_result"})


class ProtocolError(Exception):
    pass


def send(stream, message: dict) -> None:
    line = json.dumps(message, ensure_ascii=True, sort_keys=True) + "\n"
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    stream.write(line)
    stream.flush()


def recv(stream, allowed_types: frozenset[str]) -> dict:
    """Read one message. Raises EOFError when the other side closed the pipe."""
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        raise EOFError("peer closed the pipe")
    if len(line) > MAX_MESSAGE_BYTES or not line.endswith("\n"):
        raise ProtocolError("message too large or truncated")
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        raise ProtocolError("message is not JSON") from None
    if not isinstance(message, dict) or message.get("type") not in allowed_types:
        raise ProtocolError("unexpected message type")
    return message


class BrokerClient:
    """Agent side. Strictly request/response; one call at a time."""

    def __init__(self, fd_in: int, fd_out: int):
        self._in = os.fdopen(fd_in, "r", encoding="ascii", newline="\n")
        self._out = os.fdopen(fd_out, "w", encoding="ascii", newline="\n")
        self._lock = threading.Lock()

    def call(self, message: dict, expect: str) -> dict:
        with self._lock:
            send(self._out, message)
            reply = recv(self._in, SUPERVISOR_TYPES)
        if reply["type"] != expect:
            raise ProtocolError("unexpected reply type")
        return reply

    def notify(self, message: dict) -> None:
        with self._lock:
            send(self._out, message)

    def close(self) -> None:
        for stream in (self._out, self._in):
            try:
                stream.close()
            except OSError:
                pass
