"""Phase 5B/5C: demonstrate (OS-enforced, not code-level) the permissions of the agent and search worker.

Local only: loopback ports, the local DNS-resolver socket, files and a FIFO in a private temporary
directory, /usr/bin/true, and a throwaway pseudo-terminal from `script`. Never contacts an outside host,
never launches apps. Run it UNSANDBOXED (it starts the sandboxes itself):

    .venv/bin/python search_lab/isolation_check.py

TCP results: "DENIED" = the sandbox refused the connect (EPERM); "permitted" = the sandbox let it
through (ECONNREFUSED just means nobody listens there, which is expected for 443).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
import supervisor_5b as sv  # noqa: E402  (the profiles actually used by the supervisor)

PY = str(LAB / ".venv" / "bin" / "python")
LISTEN_PORT = 18080
PROBE = r"""
import json, os, socket, subprocess, sys
lab, tmp = sys.argv[1], sys.argv[2]
def tcp(port):
    s = socket.socket(); s.settimeout(2)
    try: s.connect(("127.0.0.1", port)); return "permitted(connected)"
    except PermissionError: return "DENIED"
    except ConnectionRefusedError: return "permitted(refused:no listener)"
    except OSError as e: return f"error:{e.errno}"
    finally: s.close()
def unix(path):
    s = socket.socket(socket.AF_UNIX); s.settimeout(2)
    try: s.connect(path); return "permitted(connected)"
    except PermissionError: return "DENIED"
    except OSError as e: return f"error:{e.errno}"
    finally: s.close()
def write(path, mode="a"):
    try:
        with open(path, mode): pass
        return "permitted"
    except PermissionError: return "DENIED"
    except OSError as e: return f"error:{e.errno}"
def fifo(path):
    try: os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK)); return "permitted"
    except PermissionError: return "DENIED"
    except OSError as e: return f"error:{e.errno}"
def run(cmd):
    try: return "permitted" if subprocess.run(cmd, capture_output=True, timeout=20).returncode == 0 else "failed"
    except PermissionError: return "DENIED"
    except OSError as e: return f"error:{e.errno}"
print(json.dumps({
    "ollama_11434": tcp(11434), "https_443": tcp(443), "other_%d": tcp(%d),
    "dns_resolver_socket": unix("/private/var/run/mDNSResponder"),
    "write_lab_source": write(f"{lab}/supervisor_5b.py"),
    "write_results_rate_state": write(f"{lab}/results/wiki_rate_state.json"),
    "write_agent_run_log": write(f"{lab}/results/agent_runs_5b.jsonl"),
    "write_tmp": write(f"{tmp}/new_file.txt", "w"),
    "write_approval_fifo": fifo(f"{tmp}/approval.fifo"),
    "exec_non_python": run(["/usr/bin/true"]),
    "exec_python": run([sys.executable, "-c", "pass"]),
}))
""".replace("%d", str(LISTEN_PORT))
TTY_CHILD = r"""
import fcntl, os, termios, json
r = {}
try: os.close(os.open("/dev/tty", os.O_RDWR)); r["open_dev_tty"] = "permitted"
except OSError as e: r["open_dev_tty"] = f"blocked:{e.errno}"
try: fcntl.ioctl(1, termios.TIOCSTI, b" "); r["tiocsti_inject"] = "permitted"
except OSError as e: r["tiocsti_inject"] = f"blocked:{e.errno}"
print("TTYRESULT" + json.dumps(r), flush=True)
"""
TTY_DRIVER = r"""
import subprocess, sys
profile, child, new_session = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, sys.executable, "-I", "-c", child],
               stdin=subprocess.DEVNULL, start_new_session=new_session)
"""


def probe(profile: str, tmp: str) -> dict:
    p = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, PY, "-I", "-c", PROBE, str(LAB), tmp],
                       capture_output=True, text=True, timeout=60)
    return json.loads(p.stdout) if p.returncode == 0 else {"exit": p.returncode, "stderr": p.stderr[-200:]}


def nested(outer: str, inner: str) -> dict:
    """From INSIDE `outer`, try to start a child under `inner` (the escalation path)."""
    code = ("import subprocess,sys;p=subprocess.run(sys.argv[1:],capture_output=True,text=True);"
            "print(p.returncode);print(p.stderr.strip()[-200:])")
    p = subprocess.run(["/usr/bin/sandbox-exec", "-p", outer, PY, "-I", "-c", code,
                        "/usr/bin/sandbox-exec", "-p", inner, PY, "-I", "-c", "pass"],
                       capture_output=True, text=True, timeout=30)
    rc, _, err = p.stdout.partition("\n")
    return {"inner_exit": int(rc) if rc.strip().lstrip("-").isdigit() else None, "stderr": err.strip(),
            "outer_exit": p.returncode}


def tty(profile: str, new_session: bool) -> dict:
    """Keystroke injection into the terminal the agent's stdout points at (a throwaway pty from `script`)."""
    p = subprocess.run(["/usr/bin/script", "-q", "/dev/null", PY, "-I", "-c", TTY_DRIVER, profile, TTY_CHILD,
                        "1" if new_session else "0"], capture_output=True, text=True, timeout=60,
                       stdin=subprocess.DEVNULL)
    line = next((x for x in p.stdout.replace("\r", "").splitlines() if "TTYRESULT" in x), "")
    return json.loads(line.split("TTYRESULT", 1)[1]) if line else {"error": p.stdout[-200:]}


def pipe_roundtrip(profile: str) -> bool:
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()
    code = ("import os,sys;i,o=int(sys.argv[1]),int(sys.argv[2]);fi=os.fdopen(i);fo=os.fdopen(o,'w');"
            "fo.write('ping\\n');fo.flush();fo.write(fi.readline());fo.flush()")
    p = subprocess.Popen(["/usr/bin/sandbox-exec", "-p", profile, PY, "-I", "-c", code, str(r1), str(w2)],
                         pass_fds=(r1, w2), stdin=subprocess.DEVNULL, start_new_session=True)
    os.close(r1)
    os.close(w2)
    with os.fdopen(r2) as fi, os.fdopen(w1, "w") as fo:
        first = fi.readline().strip()
        fo.write("pong\n")
        fo.flush()
        second = fi.readline().strip()
    return p.wait(timeout=10) == 0 and (first, second) == ("ping", "pong")


def main() -> int:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LISTEN_PORT))
    srv.listen(16)
    threading.Thread(target=lambda: [srv.accept()[0].close() for _ in iter(int, 1)], daemon=True).start()
    other = f"other_{LISTEN_PORT}"
    with tempfile.TemporaryDirectory(prefix="isocheck_") as tmp:
        os.mkfifo(f"{tmp}/approval.fifo", 0o600)
        keep_reader = os.open(f"{tmp}/approval.fifo", os.O_RDONLY | os.O_NONBLOCK)  # so a permitted open succeeds
        report = {
            "agent_5c": probe(sv.AGENT_PROFILE_5C, tmp),
            "worker_5c": probe(sv.SEARCH_PROFILE_5C, tmp),
            "agent_5c_child_inherits": None,
            "agent_5c_starts_search_sandbox": nested(sv.AGENT_PROFILE_5C, sv.SEARCH_PROFILE_5C),
            "agent_5c_starts_unrestricted_sandbox": nested(sv.AGENT_PROFILE_5C, "(version 1)(allow default)"),
            "agent_5c_tty_same_session": tty(sv.AGENT_PROFILE_5C, False),
            "agent_5c_tty_new_session": tty(sv.AGENT_PROFILE_5C, True),
            "broker_pipe_roundtrip_5c": pipe_roundtrip(sv.AGENT_PROFILE_5C),
        }
        inherit = subprocess.run(["/usr/bin/sandbox-exec", "-p", sv.AGENT_PROFILE_5C, PY, "-I", "-c",
                                  "import subprocess,sys;subprocess.run(sys.argv[1:])", PY, "-I", "-c", PROBE,
                                  str(LAB), tmp], capture_output=True, text=True, timeout=60)
        report["agent_5c_child_inherits"] = json.loads(inherit.stdout) if inherit.returncode == 0 else {}
        leftovers = sorted(p.name for p in Path(tmp).iterdir())  # after the 5C probes only
        report["agent_5b_reference"] = probe(sv.AGENT_PROFILE, tmp)  # last: it is ALLOWED to write here
        os.close(keep_reader)
    a, w, c, ref = (report["agent_5c"], report["worker_5c"], report["agent_5c_child_inherits"],
                    report["agent_5b_reference"])
    checks = {
        "agent: Ollama allowed": a["ollama_11434"].startswith("permitted"),
        "agent: HTTPS 443 denied": a["https_443"] == "DENIED",
        "agent: other TCP denied": a[other] == "DENIED",
        "agent: DNS resolver socket denied (5C)": a["dns_resolver_socket"] == "DENIED",
        "agent: cannot write lab source": a["write_lab_source"] == "DENIED",
        "agent: cannot write rate-limit state": a["write_results_rate_state"] == "DENIED",
        "agent: can write its own run log": a["write_agent_run_log"] == "permitted",
        "agent: cannot write /tmp": a["write_tmp"] == "DENIED",
        "agent: cannot write the approval FIFO": a["write_approval_fifo"] == "DENIED",
        "agent: cannot exec non-Python programs": a["exec_non_python"] == "DENIED",
        "agent: Python (MCP server) still executable": a["exec_python"] == "permitted",
        "agent child inherits all of it": c == a,
        "agent cannot start the search sandbox": report["agent_5c_starts_search_sandbox"]["inner_exit"] != 0,
        "agent cannot start an unrestricted sandbox": report["agent_5c_starts_unrestricted_sandbox"]["inner_exit"] != 0,
        "agent cannot open /dev/tty": report["agent_5c_tty_new_session"].get("open_dev_tty", "").startswith("blocked"),
        "agent in new session cannot inject keystrokes": report["agent_5c_tty_new_session"].get(
            "tiocsti_inject", "").startswith("blocked"),
        "broker pipes cross the sandbox boundary": report["broker_pipe_roundtrip_5c"] is True,
        "worker: HTTPS 443 permitted": w["https_443"].startswith("permitted"),
        "worker: DNS resolver socket permitted": w["dns_resolver_socket"].startswith("permitted"),
        "worker: Ollama denied": w["ollama_11434"] == "DENIED",
        "worker: other TCP denied": w[other] == "DENIED",
        "worker: no file writes (lab, results, tmp, FIFO)": all(w[k] == "DENIED" for k in (
            "write_lab_source", "write_results_rate_state", "write_agent_run_log", "write_tmp", "write_approval_fifo")),
        "worker: cannot exec non-Python programs": w["exec_non_python"] == "DENIED",
        "5C probes left no files behind": leftovers == ["approval.fifo"],
    }
    report["checks"] = checks
    report["5b_reference_gaps_closed_in_5c"] = {k: f"{ref[k]} -> {a[k]}" for k in a if ref[k] != a[k]}
    report["kept_on_purpose"] = {"same-session TIOCSTI (why the agent gets a new session)":
                                 report["agent_5c_tty_same_session"].get("tiocsti_inject")}
    print(json.dumps(report, indent=2))
    out = LAB / "results" / f"isolation_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{sum(checks.values())}/{len(checks)} checks passed; log: {out.name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
