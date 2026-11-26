"""Phase 5B supervisor: the only UNSANDBOXED process. It never opens a network connection itself.

Why it exists (measured in Phase 5B, see search_lab/isolation_check.py): a process inside a macOS
sandbox passes that sandbox to every child, and it cannot apply a new sandbox at all
(`sandbox_apply: Operation not permitted`), not even a stricter one. So a search worker started by
the localhost-only agent could never reach Wikipedia, and giving the agent HTTPS would weaken it.
Instead this supervisor starts two SIBLING sandboxes:

    supervisor_5b.py (no sandbox; standard library only; owns the terminal and the approvals)
      |- agent:  sandbox-exec AGENT_PROFILE  .venv/bin/python agent_5b.py      (localhost:11434 only)
      |            '- SamToolkit MCP server (inherits the same localhost-only sandbox)
      '- worker: sandbox-exec SEARCH_PROFILE .venv/bin/python -I search_lab/wiki_worker.py
                   (HTTPS port 443 + the system DNS resolver socket; started once per approved search)

The agent talks to the supervisor only through two inherited pipes (broker_protocol.py).

Approval: the supervisor itself displays every approval card and reads the human's decision. For a
wiki_search it independently re-checks the card (same request it was started with, fixed search spec,
v3 validator, digest), and later runs a search ONLY if it matches an approval it collected itself,
exactly once. A file-tool approval never authorizes a search. Local rate limit: one search per
session, at least MIN_INTERVAL_S between searches and at most MAX_PER_HOUR per hour.

  .venv/bin/python supervisor_5b.py "Search Wikipedia for Python programming language." --approval tty

Phase 6B (LocalCodePilot bridge), all additive; the tty, fifo and deny modes are unchanged:
- `--approval pipe --control-fd N`: the human's decision arrives as ONE line on an inherited pipe whose
  write end only the launcher holds (no file path another process could open). The supervisor still
  decides: it approves only if that line is exactly "approve <CODE>" for the card it checked and showed.
- `--events-fd M`: bounded JSON-lines events (started, card, approval, search, agent_report, result) for
  the launcher. Both fds are made non-inheritable, so neither the agent nor the worker ever receives them.
- Process lifecycle: SIGTERM/SIGHUP/SIGINT end the run through the same cleanup; an overall deadline
  kills a hung agent; the agent's whole process tree (including the SamToolkit MCP server, which the MCP
  SDK starts in its own session) and any search worker are stopped and killed on every exit path.
- `--status`: a read-only capability report (no model call, no tool call, no search).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

LAB = Path(__file__).resolve().parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import broker_protocol as bp  # noqa: E402
import decision_validator_v3 as v3  # noqa: E402
import wiki_capability as wc  # noqa: E402

PYTHON = LAB / ".venv" / "bin" / "python"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
AGENT_ENTRY = LAB / "agent_5b.py"
WORKER = LAB / "search_lab" / "wiki_worker.py"
WORKSPACE_LABEL = str(Path.home() / "MCP-Toolbox" / "sam-toolkit-mcp" / "examples")  # = harness_config.WORKSPACE
# Identical to run_sandboxed.sh (agent) and search_lab/run_wiki_sandboxed.sh (search); tests compare them.
AGENT_PROFILE = ('(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip '
                 '"localhost:11434"))(allow network-outbound (remote unix-socket))')
SEARCH_PROFILE = ('(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote tcp "*:443"))'
                  '(allow network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))')
# Phase 5C: the profiles actually used. Each is the 5B profile above with ONLY extra deny rules appended
# (tests check the prefix), verified by search_lab/isolation_check.py:
#   agent : no DNS-resolver socket (the 5B "(remote unix-socket)" allowance is overridden by a later deny),
#           file writes only to its own run log, and only the Python interpreter may be executed;
#   worker: no file writes at all, and only the Python interpreter may be executed.
SAMTOOLKIT_PYTHON = Path.home() / "MCP-Toolbox" / "sam-toolkit-mcp" / ".venv" / "bin" / "python"
PYTHON_FRAMEWORK = "/Library/Frameworks/Python.framework/Versions/3.12"
AGENT_RUN_LOG = LAB / "results" / "agent_runs_5b.jsonl"
_ONLY_PYTHON = (f'(deny process-exec)(allow process-exec (subpath "{PYTHON_FRAMEWORK}") (literal "{PYTHON}")'
                f' (literal "{SAMTOOLKIT_PYTHON}"))')
AGENT_PROFILE_5C = (AGENT_PROFILE + "(deny network-outbound (remote unix-socket))"
                    f'(deny file-write*)(allow file-write* (literal "{AGENT_RUN_LOG}") (literal "/dev/null")'
                    ' (literal "/dev/dtracehelper"))' + _ONLY_PYTHON)
SEARCH_PROFILE_5C = (SEARCH_PROFILE + '(deny file-write*)(allow file-write* (literal "/dev/null")'
                     ' (literal "/dev/dtracehelper"))'
                     f'(deny process-exec)(allow process-exec (subpath "{PYTHON_FRAMEWORK}") (literal "{PYTHON}"))')
AGENT_ENV = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "en_US.UTF-8"}  # no inherited secrets
WORKER_ENV = {"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"}  # nothing secret reaches the worker
MAX_AGENT_MESSAGES = 16                     # approval + search + done needs 3; more is abuse
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
WORKER_TIMEOUT_S = 25
MAX_WORKER_OUTPUT = 131_072
AGENT_TIMEOUT_S = 900
RATE_STATE = LAB / "results" / "wiki_rate_state.json"
MIN_INTERVAL_S = 10
MAX_PER_HOUR = 30
MAX_SEARCHES_PER_SESSION = 1
MAX_APPROVAL_REQUESTS = 1
FILE_TOOLS = frozenset({"inspect_python", "inspect_csv"})
CARD_KEYS = frozenset({"type", "tool", "arguments", "workspace", "user_reference", "request", "proposal_id",
                       "thread_id", "digest", "approval_code"})
# Phase 6B
APPROVAL_TIMEOUT_S = 300          # pipe mode: no decision within this time is a denial
CONTROL_LINE_MAX = 200
EVENT_MAX_BYTES = 131_072         # one event line, including the newline
MAX_EVENTS = 64
MONITOR_INTERVAL_S = 0.5          # how often the agent's process tree is recorded
KILL_ROUNDS = 8
PS = "/bin/ps"
PROBE = LAB / "agent_probe_5c.py"
PROBE_TIMEOUT_S = 90
REPORT_KEYS = frozenset({"thread_id", "request", "answer_mode", "decision", "validation", "proposal", "approval",
                         "tool_result", "events", "final", "tool_calls_made", "mcp_executions", "search_handoffs",
                         "seconds_to_decision", "seconds_after_approval", "disk_free_gib", "memory_pressure_level"})


class Interrupted(BaseException):
    """Raised by the signal handler so every exit path runs the same cleanup."""


# --- process trees (Phase 6B) ------------------------------------------------------------------------
class _BsdInfo(ctypes.Structure):
    """<sys/proc_info.h> struct proc_bsdinfo (PROC_PIDTBSDINFO); 136 bytes."""
    _fields_ = [("flags", ctypes.c_uint32), ("status", ctypes.c_uint32), ("xstatus", ctypes.c_uint32),
                ("pid", ctypes.c_uint32), ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32), ("rgid", ctypes.c_uint32),
                ("svuid", ctypes.c_uint32), ("svgid", ctypes.c_uint32), ("rfu1", ctypes.c_uint32),
                ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32), ("nfiles", ctypes.c_uint32),
                ("pgid", ctypes.c_uint32), ("pjobc", ctypes.c_uint32), ("tdev", ctypes.c_uint32),
                ("tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32), ("start_sec", ctypes.c_uint64),
                ("start_usec", ctypes.c_uint64)]


def _libproc_table() -> list[tuple[int, int, int, str]] | None:
    """The process table through libproc (no exec; also works inside a sandbox, unlike the setuid /bin/ps)."""
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        count = lib.proc_listallpids(None, 0)
        if count <= 0 or ctypes.sizeof(_BsdInfo) != 136:
            return None
        buf = (ctypes.c_int * (count + 256))()
        count = lib.proc_listallpids(buf, ctypes.sizeof(buf))
    except (OSError, AttributeError):
        return None
    uid, rows, info = os.getuid(), [], _BsdInfo()
    for pid in buf[:max(count, 0)]:
        if pid > 0 and lib.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) == ctypes.sizeof(info) \
                and info.uid == uid:
            rows.append((pid, int(info.ppid), int(info.pgid), f"{info.start_sec}.{info.start_usec:06d}"))
    return rows or None


def _process_table() -> list[tuple[int, int, int, str]]:
    """(pid, ppid, pgid, start time) of this user's processes; libproc first, /bin/ps as a fallback.
    Empty if neither is available."""
    rows = _libproc_table()
    if rows is not None:
        return rows
    try:
        out = subprocess.run([PS, "-A", "-o", "pid=,ppid=,pgid=,uid=,lstart="], capture_output=True, text=True,
                             timeout=10, env={"PATH": "/usr/bin:/bin", "LANG": "C"}).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    uid, rows = os.getuid(), []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and all(p.isdigit() for p in parts[:4]) and int(parts[3]) == uid:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[4].strip()))
    return rows


def process_tree(roots: set[int], table: list | None = None) -> dict[int, str]:
    """Every live process descended from `roots` or in a root's process group -> its start time.
    Never includes this process, its parent, its process group or pid 1."""
    table = _process_table() if table is None else table
    present = {pid for pid, *_ in table}
    found = {p for p in roots if p in present}
    groups = {pgid for pid, _, pgid, _ in table if pid in found} - {os.getpgrp(), 0, 1}
    changed = True
    while changed:
        changed = False
        for pid, ppid, pgid, _ in table:
            if pid not in found and (ppid in found or pgid in groups):
                found.add(pid)
                changed = True
    found -= {os.getpid(), os.getppid(), 1, 0}
    return {pid: start for pid, _, _, start in table if pid in found}


def kill_tree(roots: set[int], known: dict[int, str] | None = None) -> dict:
    """Stop (so nothing can fork) and then kill every process in the trees of `roots`, plus `known`
    processes recorded earlier (pid -> start time; a pid is only touched if its start time still matches,
    so a reused pid is never hit). Returns counts for the audit."""
    stopped: dict[int, str] = {}
    ps_ok = True
    for _ in range(KILL_ROUNDS):
        table = _process_table()
        if not table:
            ps_ok = False
            break
        current = {pid: start for pid, _, _, start in table}
        tree = process_tree(set(roots) | set(stopped), table)
        tree.update({p: s for p, s in (known or {}).items() if current.get(p) == s and p not in (os.getpid(), 1)})
        new = {p: s for p, s in tree.items() if p not in stopped}
        if not new:
            break
        for pid in new:
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        stopped.update(new)
    for pid in stopped:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for pid in ([] if ps_ok else roots):  # fallback only when ps is unavailable: the roots and their groups
        for fn, target in ((os.killpg, pid), (os.kill, pid)):
            try:
                if fn is os.killpg and target == os.getpgrp():
                    continue
                fn(target, signal.SIGKILL)
            except OSError:
                pass
    return {"processes_killed": len(stopped)}


class TreeMonitor:
    """Records the agent's descendants while it runs, so a child left behind by a crashed agent (it would be
    re-parented to launchd and lose its link to the agent) is still found and killed at cleanup."""

    def __init__(self, root: int | None, interval: float = MONITOR_INTERVAL_S):
        self.root, self.interval, self.seen = root, interval, {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "TreeMonitor":
        if self.root is not None:
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.seen.update(process_tree({self.root}))
            self._stop.wait(self.interval)

    def stop(self) -> dict[int, str]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)
        return dict(self.seen)


_LIVE_WORKERS: set[int] = set()     # search workers currently running (killed by cleanup)


# --- events for the launcher (Phase 6B) --------------------------------------------------------------
class EventSink:
    """Bounded JSON-lines events on an inherited fd. Never raises; a broken pipe just stops the events."""

    def __init__(self, fd: int | None = None):
        self.fd, self.count, self.broken = fd, 0, False
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields) -> None:
        with self._lock:
            if self.fd is None or self.broken or self.count >= MAX_EVENTS:
                return
            self.count += 1
            last = self.count == MAX_EVENTS and kind != "result"
            event = {"type": "events_truncated" if last else kind, "seq": self.count, **({} if last else fields)}
            line = json.dumps(event, ensure_ascii=True, sort_keys=True, default=str) + "\n"
            if len(line) > EVENT_MAX_BYTES:
                line = json.dumps({"type": kind, "seq": self.count, "oversize": True}) + "\n"
            data = line.encode("ascii")
            try:
                while data:
                    data = data[os.write(self.fd, data):]
            except OSError:
                self.broken = True


def _bounded(value: object, depth: int = 0) -> object:
    """A JSON-safe copy with strings, lists and nesting bounded (agent-reported data)."""
    if depth > 6:
        return None
    if isinstance(value, str):
        return value[:8_000]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_bounded(v, depth + 1) for v in value[:60]]
    if isinstance(value, dict):
        return {str(k)[:80]: _bounded(v, depth + 1) for k, v in list(value.items())[:60]}
    return str(value)[:200]


def search_summary(record: dict) -> dict:
    """What the worker actually returned, as observed by the supervisor (bounded; the text is untrusted)."""
    results = record.get("results") if isinstance(record.get("results"), list) else []
    rows = [{"title": str(r.get("title", ""))[:300], "url": str(r.get("url", ""))[:500],
             "snippet": str(r.get("snippet", ""))[:1_000]} for r in results[:wc.MAX_RESULTS] if isinstance(r, dict)]
    return {**{k: record.get(k) for k in ("status", "http_status", "requests", "bytes", "seconds", "worker_exit",
                                          "error")}, "results": rows}


class RateLimiter:
    """Conservative local limit, persisted across runs in results/wiki_rate_state.json."""

    def __init__(self, path: Path = RATE_STATE, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock

    def acquire(self) -> str | None:
        now = self.clock()
        try:
            stamps = json.loads(self.path.read_text()) if self.path.exists() else []
        except (OSError, ValueError):
            return "rate_state_unreadable"
        if not isinstance(stamps, list) or not all(isinstance(t, (int, float)) for t in stamps):
            return "rate_state_unreadable"
        recent = [t for t in stamps if now - 3600 < t <= now + 5]
        if len(recent) >= MAX_PER_HOUR:
            return "hourly_limit"
        if recent and now - max(recent) < MIN_INTERVAL_S:
            return "min_interval"
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(recent + [now]))
        return None

    def status(self) -> dict:
        """Read-only view of the limit (never writes the state file)."""
        now = self.clock()
        try:
            stamps = json.loads(self.path.read_text()) if self.path.exists() else []
            recent = [t for t in stamps if isinstance(t, (int, float)) and now - 3600 < t <= now + 5]
        except (OSError, ValueError, TypeError):
            return {"readable": False}
        wait = max(0.0, MIN_INTERVAL_S - (now - max(recent))) if recent else 0.0
        return {"readable": True, "searches_last_hour": len(recent), "max_per_hour": MAX_PER_HOUR,
                "min_interval_s": MIN_INTERVAL_S, "seconds_until_allowed": round(wait, 1),
                "per_session": MAX_SEARCHES_PER_SESSION}


def spawn_worker(plan: dict, code: str, on_start: Callable[[int], None] | None = None) -> dict:
    """Run ONE search in the separately sandboxed worker. No retries. The worker is registered while it
    runs, so every supervisor exit path can kill it (and anything it started)."""
    payload = json.dumps({"plan": plan, "approval_code": code})
    try:
        proc = subprocess.Popen([SANDBOX_EXEC, "-p", SEARCH_PROFILE_5C, str(PYTHON), "-I", str(WORKER)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=WORKER_ENV, cwd=str(LAB), start_new_session=True)
    except OSError:
        return {"status": "worker_error", "requests": 0}
    _LIVE_WORKERS.add(proc.pid)
    monitor = TreeMonitor(proc.pid).start()
    try:
        if on_start:
            on_start(proc.pid)
        try:
            stdout, _ = proc.communicate(payload, timeout=WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            kill_tree({proc.pid}, monitor.stop())  # not yet reaped, so the pid is still the worker's
            proc.wait()
            return {"status": "worker_timeout", "requests": None}
    finally:
        known = monitor.stop()
        if proc.poll() is None:  # interrupted (signal, error): never leave the worker running
            kill_tree({proc.pid}, known)
            proc.wait()
        else:  # finished: sweep anything it left behind (recorded pids, start times re-checked)
            kill_tree(set(), known)
        _LIVE_WORKERS.discard(proc.pid)
    lines = stdout.strip().splitlines()
    if len(stdout) > MAX_WORKER_OUTPUT or not lines:
        return {"status": "worker_error", "requests": None, "worker_exit": proc.returncode}
    try:
        record = json.loads(lines[-1])
    except ValueError:
        return {"status": "worker_error", "requests": None, "worker_exit": proc.returncode}
    if not isinstance(record, dict):
        return {"status": "worker_error", "requests": None, "worker_exit": proc.returncode}
    record["worker_exit"] = proc.returncode
    return record


def printable(text: object) -> str:
    """Remove terminal control characters (escape sequences could redraw or hide the approval card)."""
    return _CONTROL.sub("?", str(text))


def render_card(card: dict) -> str:
    rows = ["", "=================== APPROVAL REQUIRED (shown by the supervisor) ===================",
            f"tool          : {card['tool']}"]
    if card["tool"] == wc.SEARCH_TOOL:
        s = card["search_spec"]
        rows += [f"provider      : {s['provider']}",
                 f"hostname      : {s['hostname']}  (HTTPS {s['method']} {s['endpoint']})",
                 f"search query  : {json.dumps(card['arguments']['query'], ensure_ascii=False)}  (sent exactly as shown)",
                 f"User-Agent    : {s['user_agent']}",
                 f"max results   : {s['max_results']}",
                 f"timeout       : {s['timeout_s']} s; retries: {s['retries']}; redirects: {s['redirects']}",
                 f"fixed params  : {json.dumps(s['params'], sort_keys=True)}",
                 f"sent outside  : {s['transmitted']}"]
    else:
        rows += [f"arguments     : {json.dumps(card['arguments'])}  (read-only SamToolkit MCP tool, local only)",
                 f"workspace     : {card['workspace']}"]
    rows += [f"user wrote    : {card['user_reference']}",
             f"request       : {card['request']}",
             f"proposal id   : {card['proposal_id']}",
             f"APPROVAL CODE : {card['approval_code']}",
             "=" * 84]
    return printable("\n".join(rows))


class Supervisor:
    def __init__(self, request: str, user_agent: str, approver: Callable[[dict], bool], via: str,
                 spawn: Callable[[dict, str], dict] = spawn_worker, limiter: RateLimiter | None = None,
                 workspace_label: str = WORKSPACE_LABEL, out: Callable[[str], None] = print,
                 events: EventSink | None = None):
        self.request, self.spec, self.approver, self.via = request, wc.search_spec_for(user_agent), approver, via
        self.spawn, self.limiter, self.workspace_label, self.out = spawn, limiter or RateLimiter(), workspace_label, out
        self.events = events or EventSink(None)
        self.approval_requests, self.searches = 0, 0
        self.approved_searches: dict[str, dict] = {}
        self.audit: list[dict] = []
        self.agent_report: dict | None = None

    # -- messages from the agent ------------------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        kind = message.get("type")
        if kind == "approval_request":
            return self._approval(message.get("card"))
        if kind == "search":
            return {"type": "search_result", "record": self._search(message)}
        if kind == "done" and self.agent_report is None and isinstance(message.get("report"), dict):
            # AGENT-REPORTED data (decision, validation, file-tool result, final text): shown, never trusted.
            self.agent_report = _bounded({k: v for k, v in message["report"].items() if k in REPORT_KEYS})
            self.events.emit("agent_report", report=self.agent_report)
        self.audit.append({"event": "agent_done"} if kind == "done" else {"event": "ignored_message"})
        return None

    def check_card(self, card: object) -> str | None:
        """Reason to refuse the card, or None. Independent of the agent's own checks."""
        if not isinstance(card, dict):
            return "card_not_object"
        tool = card.get("tool")
        expected_keys = CARD_KEYS | ({"search_spec"} if tool == wc.SEARCH_TOOL else set())
        if set(card) != expected_keys or card.get("type") != "tool_approval_request":
            return "card_fields"
        if card["request"] != self.request:
            return "request_mismatch"
        if card["workspace"] != self.workspace_label:
            return "workspace_mismatch"
        if not all(isinstance(card[k], str) and card[k] for k in ("thread_id", "proposal_id", "digest")):
            return "card_fields"
        args = card["arguments"]
        if tool == wc.SEARCH_TOOL:
            if card["search_spec"] != self.spec:
                return "search_spec_mismatch"
            if not isinstance(args, dict) or set(args) != {"query"} or not isinstance(args["query"], str):
                return "bad_arguments"
            r = v3.validate_decision(self.request, {"action": wc.SEARCH_TOOL, "path": "", "query": args["query"],
                                                    "message": ""})
            if not r.ready_for_human_review or r.query != args["query"]:
                return "validator_disagrees"
        elif tool in FILE_TOOLS:
            if not isinstance(args, dict) or set(args) != {"path"} or not isinstance(args["path"], str):
                return "bad_arguments"
        else:
            return "unknown_tool"
        proposal = {k: card[k] for k in wc.DIGEST_KEYS}
        if tool == wc.SEARCH_TOOL:
            proposal["search_spec"] = card["search_spec"]
        digest = wc.proposal_digest(card["thread_id"], proposal)
        if card["digest"] != digest or card["approval_code"] != wc.approval_code(digest):
            return "digest_mismatch"
        return None

    def _approval(self, card: object) -> dict:
        self.approval_requests += 1
        ids = card if isinstance(card, dict) else {}
        deny = {"type": "approval_response", "decision": "deny", "proposal_id": ids.get("proposal_id"),
                "digest": ids.get("digest"), "via": self.via}
        if self.approval_requests > MAX_APPROVAL_REQUESTS:
            self.audit.append({"event": "approval_refused", "reason": "too_many_approval_requests"})
            self.events.emit("card_refused", reason="too_many_approval_requests")
            return {**deny, "note": "too_many_approval_requests"}
        reason = self.check_card(card)
        if reason:
            self.out(f"SUPERVISOR: approval card refused ({reason}); denied without asking.")
            self.audit.append({"event": "approval_refused", "reason": reason})
            self.events.emit("card_refused", reason=reason)
            return {**deny, "note": reason}
        self.out(render_card(card))
        self.events.emit("card", card=card, approval_timeout_s=APPROVAL_TIMEOUT_S)  # only a card that passed checks
        approved = self.approver(card) is True
        self.audit.append({"event": "approval_decision", "tool": card["tool"], "approved": approved,
                           "approval_code": card["approval_code"], "arguments": card["arguments"]})
        self.events.emit("approval", decision="approve" if approved else "deny", via=self.via, tool=card["tool"],
                         digest=card["digest"], proposal_id=card["proposal_id"])
        if approved and card["tool"] == wc.SEARCH_TOOL:
            self.approved_searches[card["digest"]] = {"query": card["arguments"]["query"], "used": False,
                                                      "proposal_id": card["proposal_id"],
                                                      "thread_id": card["thread_id"]}
        return {**deny, "decision": "approve" if approved else "deny"}

    def _search(self, msg: dict) -> dict:
        def refuse(reason: str) -> dict:
            self.out(f"SUPERVISOR: search refused ({reason}); nothing was sent.")
            self.audit.append({"event": "search_refused", "reason": reason})
            self.events.emit("search", status="broker_refused", requests=0, reason=reason, results=[])
            return {"status": "broker_refused", "requests": 0, "error": reason, "results": []}
        if self.searches >= MAX_SEARCHES_PER_SESSION:
            return refuse("session_search_limit")
        entry = self.approved_searches.get(msg.get("digest"))
        if entry is None:
            return refuse("no_matching_human_approval")
        if entry["used"]:
            return refuse("approval_already_used")
        if (msg.get("query"), msg.get("proposal_id"), msg.get("thread_id")) != \
                (entry["query"], entry["proposal_id"], entry["thread_id"]):
            return refuse("does_not_match_approval")
        entry["used"], self.searches = True, self.searches + 1  # consumed BEFORE anything is sent
        limited = self.limiter.acquire()
        if limited:
            self.audit.append({"event": "search_rate_limited", "reason": limited})
            self.events.emit("search", status="local_rate_limit", requests=0, reason=limited, results=[])
            return {"status": "local_rate_limit", "requests": 0, "error": limited, "results": []}
        plan = wc.plan_for(self.spec, entry["query"])
        started = time.perf_counter()
        if self.spawn is spawn_worker:
            record = spawn_worker(plan, wc.wiki.approval_code(plan),
                                  on_start=lambda pid: self.events.emit("worker_started", pid=pid))
        else:
            record = self.spawn(plan, wc.wiki.approval_code(plan))
        record = record if isinstance(record, dict) else {"status": "worker_error", "requests": None}
        summary = {k: record.get(k) for k in ("status", "http_status", "requests", "bytes", "seconds", "worker_exit",
                                               "retry_after", "error")}
        summary["results"] = len(record.get("results") or [])
        self.audit.append({"event": "search_executed", "query": entry["query"], "host": wc.wiki.HOST,
                           "wall_seconds": round(time.perf_counter() - started, 2), **summary})
        self.events.emit("search", query=entry["query"], host=wc.wiki.HOST, **search_summary(record))
        return record


# --- approvers (the human decides; none of these can approve by itself) -----------------------------
def tty_approver(card: dict) -> bool:
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as tty:
            tty.write(f"Type {card['approval_code']} to approve this exact call (anything else denies): ")
            tty.flush()
            return tty.readline().strip() == card["approval_code"]
    except OSError:
        print("no interactive terminal: denied.")
        return False


def fifo_approver(path: str) -> Callable[[dict], bool]:
    def approve(card: dict) -> bool:
        try:
            st = os.stat(path)
        except OSError:
            print("approval FIFO is missing: denied.")
            return False
        if not stat.S_ISFIFO(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            print("approval FIFO must be a FIFO owned by you with mode 600: denied.")
            return False
        print(f"Waiting for a human decision relayed through {path} ...", flush=True)
        with open(path, encoding="utf-8") as f:
            line = f.readline(200).strip()
        ok = line == f"approve {card['approval_code']}"
        print(f"relayed decision received: {'APPROVE' if ok else 'DENY'} ({line[:40]!r})", flush=True)
        return ok
    return approve


def deny_approver(card: dict) -> bool:
    print("approval mode 'deny': denied automatically; nothing will run.")
    return False


def pipe_approver(fd: int, timeout_s: float = APPROVAL_TIMEOUT_S) -> Callable[[dict], bool]:
    """Phase 6B: ONE human decision relayed by the launcher over an inherited pipe (the launcher alone holds
    the write end; the agent never receives it). Approves only the exact line "approve <CODE>\\n" for the card
    this supervisor checked and showed. Anything else, EOF, an over-long line or no decision within
    `timeout_s` is a denial. The pipe is closed after the first decision, so it can never be used twice."""
    used: list[bool] = []

    def approve(card: dict) -> bool:
        if used:
            return False
        used.append(True)
        deadline, buf = time.monotonic() + timeout_s, b""
        try:
            while b"\n" not in buf and len(buf) <= CONTROL_LINE_MAX:
                left = deadline - time.monotonic()
                if left <= 0:
                    print("no decision before the approval timeout: denied.", flush=True)
                    return False
                ready, _, _ = select.select([fd], [], [], left)
                if not ready:
                    continue
                chunk = os.read(fd, CONTROL_LINE_MAX + 1 - len(buf))
                if not chunk:
                    break
                buf += chunk
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        line = buf.split(b"\n", 1)[0] if b"\n" in buf else b""
        ok = line == f"approve {card['approval_code']}".encode("ascii")
        print(f"relayed decision received: {'APPROVE' if ok else 'DENY'}", flush=True)
        return ok
    return approve


# --- process plumbing ------------------------------------------------------------------------------
def _relay(stream, out: Callable[[str], None]) -> None:
    """Show the agent's output with control characters removed and every line marked as agent output."""
    for line in iter(lambda: stream.readline(8_192), ""):
        out("agent| " + printable(line.rstrip("\n")))


_CLEANING = False


def run(request: str, approver: Callable[[dict], bool], via: str, agent_cmd: list[str] | None = None,
        supervisor: Supervisor | None = None, answer_mode: str = "evidence",
        out: Callable[[str], None] = lambda line: print(line, flush=True), events: EventSink | None = None,
        timeout_s: float = AGENT_TIMEOUT_S) -> dict:
    global _CLEANING
    sup = supervisor or Supervisor(request, wc.load_user_agent(), approver, via, events=events)
    if events is not None:
        sup.events = events
    ev = sup.events
    to_agent_r, to_agent_w = os.pipe()
    from_agent_r, from_agent_w = os.pipe()
    base = agent_cmd or [SANDBOX_EXEC, "-p", AGENT_PROFILE_5C, str(PYTHON), str(AGENT_ENTRY)]
    cmd = [*base, "--broker-in", str(to_agent_r), "--broker-out", str(from_agent_w),
           "--answer-mode", answer_mode, "--", request]
    # New session: the agent has no controlling terminal (no TIOCSTI keystroke injection), and its
    # stdout/stderr are a pipe that the supervisor relays, so it holds no terminal handle at all.
    # close_fds (the default) means the agent receives ONLY the two broker fds: never the control or events fd.
    proc = subprocess.Popen(cmd, pass_fds=(to_agent_r, from_agent_w), stdin=subprocess.DEVNULL, cwd=str(LAB),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            start_new_session=True, env=AGENT_ENV if agent_cmd is None else None)
    os.close(to_agent_r)
    os.close(from_agent_w)
    pid = getattr(proc, "pid", None)
    ev.emit("started", agent_pid=pid, answer_mode=answer_mode, via=via, timeout_s=timeout_s)
    monitor = TreeMonitor(pid).start()
    timed_out = threading.Event()

    def on_deadline() -> None:  # a hung agent is killed; its pipe closes and the loop below ends
        timed_out.set()
        if pid is not None and proc.returncode is None:  # never signal a pid that was already reaped
            kill_tree({pid})
    timer = threading.Timer(timeout_s, on_deadline)
    timer.daemon = True
    timer.start()
    relay = threading.Thread(target=_relay, args=(proc.stdout, out), daemon=True)
    relay.start()
    messages, code, ended, cleanup = 0, None, "normal", {}
    started = time.monotonic()
    try:
        with os.fdopen(from_agent_r, "r", encoding="ascii", newline="\n") as fin, \
                os.fdopen(to_agent_w, "w", encoding="ascii", newline="\n") as fout:
            while True:
                try:
                    message = bp.recv(fin, bp.AGENT_TYPES)
                except EOFError:
                    break
                except bp.ProtocolError as exc:
                    sup.audit.append({"event": "protocol_error", "reason": str(exc)})
                    ended = "protocol_error"
                    proc.kill()
                    break
                messages += 1
                if messages > MAX_AGENT_MESSAGES:
                    sup.audit.append({"event": "protocol_error", "reason": "too many messages"})
                    ended = "protocol_error"
                    proc.kill()
                    break
                reply = sup.handle(message)
                if reply is not None:
                    try:
                        bp.send(fout, reply)
                    except BrokenPipeError:
                        break
        try:
            code = proc.wait(timeout=max(1.0, timeout_s - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out.set()
    except Interrupted as exc:  # SIGTERM / SIGHUP / SIGINT: same cleanup as every other exit
        ended = f"interrupted:{exc}"
    finally:
        _CLEANING = True
        try:
            timer.cancel()
            known = monitor.stop()
            if pid is not None:  # the agent's whole tree, recorded descendants, and any live search worker
                # A reaped agent pid is never used as a root (it could already belong to another process);
                # its leftovers are then found through the monitor's records, whose start times are re-checked.
                roots = set(_LIVE_WORKERS) | ({pid} if proc.returncode is None else set())
                cleanup = kill_tree(roots, known)
            if proc.poll() is None:
                proc.kill()
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            relay.join(timeout=5)
        finally:
            _CLEANING = False
    if timed_out.is_set() and ended == "normal":
        ended = "timeout"
    result = {"agent_exit": code, "approval_requests": sup.approval_requests, "searches": sup.searches,
              "audit": sup.audit, "ended": ended, "cleanup": cleanup}
    ev.emit("result", **result)
    return result


# --- read-only status (Phase 6B) -----------------------------------------------------------------
def status_report() -> dict:
    """What this harness can do right now. No model call, no tool call, no search, no file written."""
    import resource_guard
    rep: dict = {"harness": "LangChain + LangGraph agent (Phase 5C), supervised", "workspace": WORKSPACE_LABEL,
                 "sandbox_exec": os.access(SANDBOX_EXEC, os.X_OK), "lab_python": PYTHON.exists(),
                 "agent_sandbox": "localhost:11434 only; no DNS; writes only its run log; executes only Python",
                 "approval": "every tool call needs a human decision; at most one per request",
                 "answer_mode": "evidence"}
    try:
        spec = wc.search_spec_for(wc.load_user_agent())
        rep["wikipedia"] = {"available": True, "provider": spec["provider"], "hostname": spec["hostname"],
                            "endpoint": spec["endpoint"], "user_agent": spec["user_agent"],
                            "max_results": spec["max_results"], "timeout_s": spec["timeout_s"], "retries": 0,
                            "transmitted": spec["transmitted"], "rate_limit": RateLimiter().status()}
    except (OSError, ValueError) as exc:
        rep["wikipedia"] = {"available": False, "error": type(exc).__name__}
    try:
        rep["resources"] = {"free_gib": round(resource_guard.free_gib(), 2),
                            "memory_pressure_level": resource_guard.memory_pressure_level(),
                            "min_free_gib": resource_guard.MIN_FREE_GIB,
                            "critical_level": resource_guard.CRITICAL_PRESSURE}
    except OSError:
        rep["resources"] = {"error": "unreadable"}
    probe = None
    try:
        probe = subprocess.Popen([SANDBOX_EXEC, "-p", AGENT_PROFILE_5C, str(PYTHON), str(PROBE)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, env=AGENT_ENV, cwd=str(LAB), start_new_session=True)
        stdout, _ = probe.communicate(timeout=PROBE_TIMEOUT_S)
        lines = stdout.strip().splitlines()
        rep["agent_probe"] = _bounded(json.loads(lines[-1])) if lines and len(stdout) < 65_536 else \
            {"ok": False, "error": "no_output"}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        rep["agent_probe"] = {"ok": False, "error": type(exc).__name__}
    finally:
        if probe is not None:
            kill_tree({probe.pid})
            probe.wait()
    return rep


def _fd_arg(value: str) -> int:
    fd = int(value)
    if fd < 3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise argparse.ArgumentTypeError("must be an inherited pipe fd >= 3")
    return fd


def _on_signal(signum, frame) -> None:
    if not _CLEANING:  # during cleanup the signal is ignored: cleanup always finishes
        raise Interrupted(signal.Signals(signum).name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("request", nargs="?")
    p.add_argument("--approval", choices=("tty", "fifo", "deny", "pipe"), default="tty")
    p.add_argument("--fifo")
    p.add_argument("--control-fd", type=_fd_arg, help="pipe mode: inherited read end carrying one decision line")
    p.add_argument("--events-fd", type=_fd_arg, help="inherited write end for JSON-lines events")
    p.add_argument("--timeout", type=int, default=AGENT_TIMEOUT_S, help="overall agent deadline, 30-900 s")
    p.add_argument("--answer-mode", choices=("evidence", "model"), default="evidence",
                   help="evidence (default): quote the snippets; model: Qwen answer, shown only if verified")
    p.add_argument("--status", action="store_true", help="print a read-only capability report and exit")
    args = p.parse_args(argv)
    for fd in (args.control_fd, args.events_fd):
        if fd is not None:
            os.set_inheritable(fd, False)  # never passed on to the agent or the worker
    if args.status:
        print(json.dumps(status_report()), flush=True)
        return 0
    if not args.request:
        p.error("a request is required")
    if args.approval == "fifo" and not args.fifo:
        p.error("--fifo is required with --approval fifo")
    if (args.approval == "pipe") != (args.control_fd is not None):
        p.error("--control-fd is required with, and only with, --approval pipe")
    if args.control_fd is not None and args.control_fd == args.events_fd:
        p.error("--control-fd and --events-fd must differ")
    if not 30 <= args.timeout <= AGENT_TIMEOUT_S:
        p.error(f"--timeout must be between 30 and {AGENT_TIMEOUT_S}")
    approver = {"tty": tty_approver, "deny": deny_approver}.get(args.approval) or (
        pipe_approver(args.control_fd) if args.approval == "pipe" else fifo_approver(args.fifo))
    via = {"tty": "supervisor-tty", "deny": "auto-deny", "fifo": "relayed-human-dialog",
           "pipe": "relayed-human-ui"}[args.approval]
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)
    started = time.perf_counter()
    events = EventSink(args.events_fd)
    try:
        outcome = run(args.request, approver, via, answer_mode=args.answer_mode, events=events,
                      timeout_s=args.timeout)
    except Interrupted as exc:  # before the agent started: nothing to clean up
        outcome = {"agent_exit": None, "approval_requests": 0, "searches": 0, "audit": [],
                   "ended": f"interrupted:{exc}", "cleanup": {}}
        events.emit("result", **outcome)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request": args.request,
              "approval_mode": args.approval, "answer_mode": args.answer_mode,
              "seconds": round(time.perf_counter() - started, 2), **outcome}
    print("supervisor audit:", json.dumps({k: record[k] for k in ("agent_exit", "approval_requests", "searches",
                                                                   "audit", "seconds", "ended")}))
    (LAB / "results").mkdir(exist_ok=True)
    with (LAB / "results" / "supervisor_5b.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    if args.control_fd is not None:
        try:
            os.close(args.control_fd)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
