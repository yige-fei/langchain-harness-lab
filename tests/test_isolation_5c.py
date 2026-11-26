"""Phase 5C: OS-enforced isolation (not code-level). Runs search_lab/isolation_check.py, which starts the real
agent and worker sandbox profiles used by the supervisor and probes them locally (loopback, local DNS socket,
temporary files/FIFO, /usr/bin/true, a throwaway pty). Skipped when the suite itself runs inside a sandbox,
because a sandboxed process cannot start another sandbox (that impossibility is itself one of the checks)."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def inside_sandbox() -> bool:
    p = subprocess.run(["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
                       capture_output=True)
    return p.returncode != 0


@unittest.skipIf(inside_sandbox(), "needs to start sandboxes; run the suite unsandboxed for this test")
class OsEnforcedIsolation(unittest.TestCase):
    def test_all_isolation_checks_pass(self):
        p = subprocess.run([str(LAB / ".venv" / "bin" / "python"), str(LAB / "search_lab" / "isolation_check.py")],
                           capture_output=True, text=True, timeout=300)
        report = json.loads(p.stdout[:p.stdout.rindex("}") + 1])
        failed = [k for k, ok in report["checks"].items() if not ok]
        self.assertEqual(failed, [])
        self.assertEqual(p.returncode, 0)
        for key in ("agent: HTTPS 443 denied", "agent: DNS resolver socket denied (5C)",
                    "agent: cannot write the approval FIFO", "agent in new session cannot inject keystrokes",
                    "agent cannot start the search sandbox"):
            self.assertIn(key, report["checks"])


if __name__ == "__main__":
    unittest.main()
