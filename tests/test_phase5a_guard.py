"""Phase 5A Part 1: resource-limit reasons in the per-request audit trail, and labelled
fallback answers. Offline: fake model/tools, sockets blocked."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_graph as ag  # noqa: E402
import resource_guard as rg  # noqa: E402
from test_agent_graph import FakeAnswer, FakeModel, FakeTool, GraphTestCase, dec  # noqa: E402


class GuardExceptionTests(unittest.TestCase):
    def test_threshold_is_unchanged(self):
        self.assertEqual(rg.MIN_FREE_GIB, 8.0)
        self.assertEqual(rg.CRITICAL_PRESSURE, 4)

    def test_low_disk_reason(self):
        with mock.patch.object(rg, "free_gib", return_value=7.88), \
             mock.patch.object(rg, "memory_pressure_level", return_value=2):
            with self.assertRaises(rg.ResourceLimitReached) as cm:
                rg.check()
        e = cm.exception
        self.assertEqual((e.reason, e.audit_detail, e.user_hint),
                         ("low_disk", "low_disk:free_gib=7.88<min_gib=8.00", "low free disk space"))

    def test_memory_pressure_reason(self):
        with mock.patch.object(rg, "free_gib", return_value=12.0), \
             mock.patch.object(rg, "memory_pressure_level", return_value=4):
            with self.assertRaises(rg.ResourceLimitReached) as cm:
                rg.check()
        self.assertEqual(cm.exception.audit_detail, "memory_pressure:level=4>=critical=4")

    def test_disk_checked_first_and_boundary(self):
        with mock.patch.object(rg, "free_gib", return_value=8.0), \
             mock.patch.object(rg, "memory_pressure_level", return_value=2):
            rg.check()  # exactly 8.0 GiB is allowed (limit is "below 8")
        with mock.patch.object(rg, "free_gib", return_value=7.0), \
             mock.patch.object(rg, "memory_pressure_level", return_value=4):
            with self.assertRaises(rg.ResourceLimitReached) as cm:
                rg.check()
        self.assertEqual(cm.exception.reason, "low_disk")

    def test_details_contain_no_paths(self):
        for e in (rg.ResourceLimitReached("low_disk", 1.5, 8), rg.ResourceLimitReached("memory_pressure", 4, 4)):
            for text in (e.audit_detail, e.user_hint, str(e)):
                self.assertNotRegex(text, r"[/\\~]")

    def test_unknown_reason_rejected(self):
        with self.assertRaises(ValueError):
            rg.ResourceLimitReached("anything", 1, 2)


class AuditAndFallbackTests(GraphTestCase):
    async def _approved(self, answer, request="What functions are defined in demo.py?", thread=None):
        self.make(dec("inspect_python", "demo.py"), answer=answer)
        t = thread or self.tid()
        card = (await ag.start(self.graph, request, t))["card"]
        return await ag.resume(self.graph, t, ag.human_response(card, True, "t"))

    async def test_guard_in_answer_step_is_recorded_and_labelled(self):
        out = await self._approved(FakeAnswer(exc=rg.ResourceLimitReached("low_disk", 7.88, 8.0)))
        self.assertIn("answered:fallback:ResourceLimitReached", out["events"])
        self.assertIn("guard:low_disk:free_gib=7.88<min_gib=8.00", out["events"])
        self.assertTrue(out["final"].startswith(ag.FALLBACK_PREFIX))
        self.assertIn("low free disk space", out["final"])
        self.assertIn("greet (line 3)", out["final"])
        self.assertEqual(out["final"].count("greet (line 3)"), 1)  # no duplicated summary
        self.assertEqual(out["tool_calls_made"], 1)

    async def test_guard_in_decide_step_is_recorded_and_nothing_runs(self):
        self.make(model_exc=rg.ResourceLimitReached("memory_pressure", 4, 4))
        out = await ag.start(self.graph, "What functions are defined in demo.py?", self.tid())
        self.assertEqual(self.tool.calls, [])
        self.assertIn("guard:memory_pressure:level=4>=critical=4", out["events"])
        self.assertTrue(out["final"].startswith(ag.FAILURE_TEXT))
        self.assertIn("critical memory pressure", out["final"])

    async def test_other_exceptions_record_type_only(self):
        out = await self._approved(FakeAnswer(exc=RuntimeError("disk at /Users/alice/private is full")))
        self.assertIn("answered:fallback:RuntimeError", out["events"])
        self.assertFalse(any(e.startswith("guard:") for e in out["events"]))
        self.assertNotIn("/Users/alice", out["final"])
        self.assertIn("unavailable because of an error", out["final"])

    async def test_unsafe_opt_in_detail_is_ignored(self):
        class Sneaky(Exception):
            audit_detail = "low_disk at /Users/alice/secret"
            user_hint = "see ~/notes"
        out = await self._approved(FakeAnswer(exc=Sneaky()))
        self.assertFalse(any(e.startswith("guard:") for e in out["events"]))
        self.assertNotIn("alice", out["final"])
        self.assertNotIn("~/notes", out["final"])

    async def test_tool_call_like_reply_is_labelled_fallback(self):
        out = await self._approved(FakeAnswer('{"name": "inspect_csv", "arguments": {"path": "x.csv"}}'))
        self.assertTrue(out["final"].startswith(ag.FALLBACK_PREFIX))
        self.assertIn("discarded", out["final"])

    async def test_model_answer_is_not_labelled(self):
        out = await self._approved(FakeAnswer("demo.py defines greet()."))
        self.assertFalse(out["final"].startswith(ag.FALLBACK_PREFIX))
        self.assertIn("answered:model", out["events"])

    async def test_concurrent_requests_do_not_mix_audit_details(self):
        class PerRequestAnswer:
            async def __call__(self, request, tool_result):
                await asyncio.sleep(0.01)
                value = 7.5 if "first" in request else 6.25
                raise rg.ResourceLimitReached("low_disk", value, 8.0)
        self.make(dec("inspect_python", "demo.py"), answer=PerRequestAnswer())
        cards = {}
        for name in ("first", "second"):
            t = f"concurrent-{name}"
            cards[t] = (await ag.start(self.graph, f"List the functions in demo.py ({name})", t))["card"]
        outs = await asyncio.gather(*(ag.resume(self.graph, t, ag.human_response(c, True, "t"))
                                      for t, c in cards.items()))
        first, second = outs
        self.assertIn("guard:low_disk:free_gib=7.50<min_gib=8.00", first["events"])
        self.assertNotIn("guard:low_disk:free_gib=6.25<min_gib=8.00", first["events"])
        self.assertIn("guard:low_disk:free_gib=6.25<min_gib=8.00", second["events"])
        self.assertNotIn("guard:low_disk:free_gib=7.50<min_gib=8.00", second["events"])


if __name__ == "__main__":
    unittest.main()
