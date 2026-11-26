"""Phase 5C: independent tests for the Phase 4C resume() fix (empty / interrupt-id-keyed dicts).
Fake model and tools only; sockets blocked."""
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

import agent_graph as ag  # noqa: E402

PY = {"status": "success", "text": "", "structured": {"file": "demo.py", "functions": [], "classes": []}}
REQ = "What functions are in demo.py?"


class ResumeFix(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls, self.net = [], []

        def blocked(*a, **k):
            self.net.append(a)
            raise OSError("network disabled in tests")
        for p in (mock.patch.object(socket.socket, "connect", blocked),
                  mock.patch.object(socket, "create_connection", blocked)):
            p.start()
            self.addCleanup(p.stop)

        async def decide(r):
            return {"action": "inspect_python", "path": "demo.py", "message": ""}

        async def execute(t, a):
            self.calls.append((t, a))
            return PY

        async def answer(r, res):
            return "demo.py defines nothing."
        self.graph = ag.build_graph(decide, execute, answer, "/fake/examples")

    def tearDown(self):
        self.assertEqual(self.net, [])

    async def paused(self, tid):
        out = await ag.start(self.graph, REQ, tid)
        self.assertEqual(out["status"], "awaiting_approval")
        return out

    async def test_empty_dict_is_now_an_explicit_denial(self):
        await self.paused("e1")
        done = await ag.resume(self.graph, "e1", {})
        self.assertEqual((done["status"], done["final"]), ("done", ag.DENIED_TEXT))
        self.assertIn("approval_denied", done["events"])
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):  # the thread is finished, not left paused
            await ag.resume(self.graph, "e1", {})

    async def test_interrupt_id_keyed_map_is_denied_even_with_a_valid_approval_inside(self):
        out = await self.paused("e2")
        snap = await self.graph.aget_state(ag.config_for("e2"))
        keyed = {snap.interrupts[0].id: ag.human_response(out["card"], True, via="id-keyed")}
        done = await ag.resume(self.graph, "e2", keyed)
        self.assertEqual(done["final"], ag.DENIED_TEXT)
        self.assertEqual(self.calls, [])

    async def test_dicts_without_decision_are_denied(self):
        for i, response in enumerate(({"via": "x"}, {"approve": True}, {"proposal_id": "p", "digest": "d"})):
            with self.subTest(response=response):
                await self.paused(f"e3-{i}")
                done = await ag.resume(self.graph, f"e3-{i}", response)
                self.assertEqual(done["final"], ag.DENIED_TEXT)
        self.assertEqual(self.calls, [])

    async def test_backward_compatible_for_real_approvals_and_denials(self):
        out = await self.paused("ok1")
        done = await ag.resume(self.graph, "ok1", ag.human_response(out["card"], True, via="test-human"))
        self.assertEqual((done["tool_calls_made"], len(self.calls)), (1, 1))
        out = await self.paused("ok2")
        done = await ag.resume(self.graph, "ok2", ag.human_response(out["card"], False, via="test-human"))
        self.assertEqual(done["final"], ag.DENIED_TEXT)
        self.assertEqual(len(self.calls), 1)
        out = await self.paused("ok3")  # non-dict values were already denied before the fix
        done = await ag.resume(self.graph, "ok3", None)
        self.assertEqual(done["final"], ag.DENIED_TEXT)


if __name__ == "__main__":
    unittest.main()
