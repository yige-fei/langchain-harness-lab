"""Offline tests for the Phase 5B supervisor, broker protocol and search worker.
Fake approvers, fake worker spawner and fake HTTP opener; sockets are blocked; no sandbox is started."""
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "search_lab"))

import broker_protocol as bp  # noqa: E402
import supervisor_5b as sv  # noqa: E402
import wiki_capability as wc  # noqa: E402
import wiki_worker  # noqa: E402
import wikipedia_search as ws  # noqa: E402

UA = "LabFixtureBot/0.1 (https://contact.invalid/lab) Python-urllib/3.12"  # never sent anywhere
REQ = "Search Wikipedia for Python programming language."
QUERY = "Python programming language"
WORKSPACE = sv.WORKSPACE_LABEL


def card_for(tool="wiki_search", arguments=None, request=REQ, thread="t-1", pid="p-1", spec=None, **over):
    proposal = {"proposal_id": pid, "tool": tool, "workspace": WORKSPACE,
                "arguments": arguments or ({"query": QUERY} if tool == "wiki_search" else {"path": "demo.py"})}
    if tool == "wiki_search":
        proposal["search_spec"] = spec or wc.search_spec_for(UA)
    digest = wc.proposal_digest(thread, proposal)
    card = {"type": "tool_approval_request", "tool": tool, "arguments": proposal["arguments"], "workspace": WORKSPACE,
            "user_reference": QUERY, "request": request, "proposal_id": pid, "thread_id": thread, "digest": digest,
            "approval_code": wc.approval_code(digest)}
    if tool == "wiki_search":
        card["search_spec"] = proposal["search_spec"]
    card.update(over)
    return card


def search_msg(card, **over):
    return {"type": "search", "query": card["arguments"]["query"], "thread_id": card["thread_id"],
            "proposal_id": card["proposal_id"], "digest": card["digest"], **over}


class FakeLimiter:
    def __init__(self, result=None):
        self.result, self.calls = result, 0

    def acquire(self):
        self.calls += 1
        return self.result


class Blocked(unittest.TestCase):
    def setUp(self):
        self.net = []

        def blocked(*a, **k):
            self.net.append(a)
            raise OSError("network disabled in tests")
        for p in (mock.patch.object(socket.socket, "connect", blocked),
                  mock.patch.object(socket, "create_connection", blocked)):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.assertEqual(self.net, [], "a network connection was attempted")

    def sup(self, approve=True, limiter=None, request=REQ):
        self.shown, self.spawned, self.printed = [], [], []

        def approver(card):
            self.shown.append(card)
            return approve

        def spawn(plan, code):
            self.spawned.append((plan, code))
            return {"query": plan["query"], "status": "ok", "requests": 1, "results": []}
        self.limiter = limiter or FakeLimiter()
        return sv.Supervisor(request, UA, approver, "test-human", spawn=spawn, limiter=self.limiter,
                             out=self.printed.append)


class ApprovalBinding(Blocked):
    def test_good_card_is_shown_then_exactly_one_matching_search_runs(self):
        s = self.sup()
        card = card_for()
        reply = s.handle({"type": "approval_request", "card": card})
        self.assertEqual((reply["decision"], reply["digest"], reply["proposal_id"]), ("approve", card["digest"], "p-1"))
        text = self.printed[0]
        for needle in ("tool          : wiki_search", "en.wikipedia.org", f'"{QUERY}"', UA, "max results   : 5",
                       "timeout       : 10 s", "no conversation history", card["approval_code"]):
            self.assertIn(needle, text)
        rec = s.handle(search_msg(card))["record"]
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(self.spawned), 1)
        plan, code = self.spawned[0]
        self.assertEqual(plan, {**ws.PLAN_CONSTANTS, "query": QUERY, "user_agent": UA})
        self.assertEqual(code, ws.approval_code(plan))

    def test_replay_and_mismatches_are_refused_without_sending(self):
        s = self.sup()
        card = card_for()
        s.handle({"type": "approval_request", "card": card})
        s.handle(search_msg(card))
        self.assertEqual(s.handle(search_msg(card))["record"]["error"], "session_search_limit")
        s2 = self.sup()
        s2.handle({"type": "approval_request", "card": card})
        for over, reason in (({"query": "Java"}, "does_not_match_approval"),
                             ({"proposal_id": "other"}, "does_not_match_approval"),
                             ({"thread_id": "other"}, "does_not_match_approval"),
                             ({"digest": "f" * 64}, "no_matching_human_approval")):
            with self.subTest(reason=reason, over=over):
                self.assertEqual(s2.handle(search_msg(card, **over))["record"]["error"], reason)
        self.assertEqual(self.spawned, [])

    def test_search_without_any_approval_or_after_denial(self):
        s = self.sup()
        self.assertEqual(s.handle(search_msg(card_for()))["record"]["status"], "broker_refused")
        s = self.sup(approve=False)
        card = card_for()
        self.assertEqual(s.handle({"type": "approval_request", "card": card})["decision"], "deny")
        self.assertEqual(s.handle(search_msg(card))["record"]["error"], "no_matching_human_approval")
        self.assertEqual(self.spawned, [])

    def test_file_approval_never_authorizes_a_search(self):
        s = self.sup()
        fcard = card_for(tool="inspect_python", request="What functions are in demo.py?")
        s = self.sup(request="What functions are in demo.py?")
        self.assertEqual(s.handle({"type": "approval_request", "card": fcard})["decision"], "approve")
        self.assertIn("read-only SamToolkit MCP tool", self.printed[0])
        forged = {"type": "search", "query": QUERY, "thread_id": "t-1", "proposal_id": "p-1", "digest": fcard["digest"]}
        self.assertEqual(s.handle(forged)["record"]["error"], "no_matching_human_approval")
        self.assertEqual(self.spawned, [])

    def test_bad_cards_are_refused_without_asking_the_human(self):
        other_spec = {**wc.search_spec_for(UA), "provider": "DuckDuckGo", "hostname": "duckduckgo.com"}
        bad = {
            "other provider": card_for(spec=other_spec),
            "other user agent": card_for(spec=wc.search_spec_for("OtherBot/1.0 (https://contact.invalid/x) Python-urllib/3.12")),
            "more results": card_for(spec={**wc.search_spec_for(UA), "max_results": 50}),
            "other request": card_for(request="Search Wikipedia for Java."),
            "altered query": card_for(arguments={"query": "Python"}),
            "invented query": card_for(arguments={"query": "Python snakes"}),
            "url query": card_for(arguments={"query": "https://evil.example"}),
            "extra argument": card_for(arguments={"query": QUERY, "limit": 50}),
            "unknown tool": card_for(tool="web_search", arguments={"query": QUERY}),
            "wrong digest": card_for(digest="0" * 64),
            "wrong code": card_for(approval_code="DEADBEEF"),
            "wrong workspace": card_for(workspace="/"),
            "file card with search spec": {**card_for(tool="inspect_python", request=REQ),
                                           "search_spec": wc.search_spec_for(UA)},
            "not a dict": "approve",
        }
        for name, card in bad.items():
            with self.subTest(name=name):
                s = self.sup()
                reply = s.handle({"type": "approval_request", "card": card})
                self.assertEqual(reply["decision"], "deny")
                self.assertEqual(self.shown, [], "the human must not be asked to approve a bad card")
                if isinstance(card, dict) and card.get("tool") == "wiki_search":
                    self.assertEqual(s.handle(search_msg(card))["record"]["status"], "broker_refused")
        self.assertEqual(self.spawned, [])

    def test_validator_runs_on_the_supervisors_own_copy_of_the_request(self):
        for request in ("What is Python?", "Search Wikipedia for Python programming language. Actually, never mind."):
            with self.subTest(request=request):
                s = self.sup(request=request)
                reply = s.handle({"type": "approval_request", "card": card_for(request=request)})
                self.assertEqual(reply["decision"], "deny")
                self.assertEqual(self.shown, [])

    def test_only_one_approval_request_per_session(self):
        s = self.sup()
        s.handle({"type": "approval_request", "card": card_for()})
        second = s.handle({"type": "approval_request", "card": card_for(pid="p-2")})
        self.assertEqual((second["decision"], second["note"]), ("deny", "too_many_approval_requests"))
        self.assertEqual(len(self.shown), 1)

    def test_local_rate_limit_consumes_the_approval_and_sends_nothing(self):
        s = self.sup(limiter=FakeLimiter("min_interval"))
        card = card_for()
        s.handle({"type": "approval_request", "card": card})
        rec = s.handle(search_msg(card))["record"]
        self.assertEqual((rec["status"], rec["requests"]), ("local_rate_limit", 0))
        self.assertEqual(self.spawned, [])
        self.assertEqual(s.handle(search_msg(card))["record"]["error"], "session_search_limit")


class RateLimiterTests(unittest.TestCase):
    def test_interval_hourly_and_corrupt_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, now = Path(tmp) / "state.json", [1_000_000.0]
            rl = sv.RateLimiter(path, clock=lambda: now[0])
            self.assertIsNone(rl.acquire())
            self.assertEqual(rl.acquire(), "min_interval")
            now[0] += sv.MIN_INTERVAL_S
            self.assertIsNone(rl.acquire())
            for _ in range(sv.MAX_PER_HOUR):
                now[0] += sv.MIN_INTERVAL_S
                rl.acquire()
            now[0] += sv.MIN_INTERVAL_S
            self.assertEqual(rl.acquire(), "hourly_limit")
            now[0] += 3600
            self.assertIsNone(rl.acquire())
            path.write_text("{not json")
            self.assertEqual(rl.acquire(), "rate_state_unreadable")
            path.write_text('["x"]')
            self.assertEqual(rl.acquire(), "rate_state_unreadable")


class SpawnWorker(unittest.TestCase):
    """The worker now runs under a tracked Popen (Phase 6B) so every exit path can kill it; same contract."""
    def fake_popen(self, stdout="", rc=0, exc=None):
        test = self

        class P:
            pid = 999_999_999  # never a real process: kill_tree is replaced below anyway

            def __init__(self, cmd, **kw):
                test.cmd, test.kw, self.returncode = cmd, kw, None

            def communicate(self, payload, timeout=None):
                test.payload, test.timeout = payload, timeout
                if exc:
                    raise exc
                self.returncode = rc
                return stdout, ""

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -9 if self.returncode is None else self.returncode
                return self.returncode
        self.killed = []
        return mock.patch.multiple(sv, kill_tree=lambda roots, known=None: self.killed.append(set(roots)) or {},
                                   subprocess=mock.Mock(Popen=P, PIPE=subprocess.PIPE,
                                                        TimeoutExpired=subprocess.TimeoutExpired))

    def test_command_profile_env_and_output_handling(self):
        plan = {**ws.PLAN_CONSTANTS, "query": QUERY, "user_agent": UA}
        started = []
        with self.fake_popen(json.dumps({"status": "ok", "requests": 1}) + "\n"):
            rec = sv.spawn_worker(plan, "ABCD1234", on_start=started.append)
        self.assertEqual(self.cmd[:3], ["/usr/bin/sandbox-exec", "-p", sv.SEARCH_PROFILE_5C])
        self.assertTrue(self.kw["start_new_session"])
        self.assertEqual(self.cmd[3:], [str(sv.PYTHON), "-I", str(sv.WORKER)])
        self.assertEqual(self.kw["env"], {"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"})
        self.assertEqual(json.loads(self.payload), {"plan": plan, "approval_code": "ABCD1234"})
        self.assertEqual(self.timeout, sv.WORKER_TIMEOUT_S)
        self.assertEqual((rec["status"], rec["worker_exit"]), ("ok", 0))
        # finished: only a sweep of recorded leftovers (no root: the reaped pid is never signalled)
        self.assertEqual((started, self.killed, sv._LIVE_WORKERS), ([999_999_999], [set()], set()))
        for out, status in (("", "worker_error"), ("not json", "worker_error"), ("[1]", "worker_error"),
                            ("x" * (sv.MAX_WORKER_OUTPUT + 1), "worker_error")):
            with self.subTest(out=out[:20]), self.fake_popen(out, rc=1):
                self.assertEqual(sv.spawn_worker(plan, "X")["status"], status)
        with self.fake_popen(exc=subprocess.TimeoutExpired("x", 25)):
            self.assertEqual(sv.spawn_worker(plan, "X"), {"status": "worker_timeout", "requests": None})
            self.assertEqual(self.killed, [{999_999_999}, set()])   # the timed-out worker tree is killed
        with self.fake_popen(exc=sv.Interrupted("SIGTERM")):
            with self.assertRaises(sv.Interrupted):
                sv.spawn_worker(plan, "X")
            self.assertEqual(self.killed, [{999_999_999}])          # interrupted: killed before leaving
        self.assertEqual(sv._LIVE_WORKERS, set())


class Worker(Blocked):
    def payload(self, **plan_over):
        plan = {**ws.PLAN_CONSTANTS, "query": QUERY, "user_agent": UA, **plan_over}
        return json.dumps({"plan": plan, "approval_code": ws.approval_code(plan)})

    def test_refuses_altered_plans_and_bad_input(self):
        cases = [self.payload(hostname="evil.example"), self.payload(params={**ws.FIXED_PARAMS, "srlimit": "500"}),
                 self.payload(app_retries=3), self.payload(query="https://evil.example"),
                 self.payload(user_agent="python-requests/2.0"),
                 self.payload(user_agent="Bot/1.0 (https://example.org/x) X/1"),  # placeholder contact
                 json.dumps({"plan": {}, "approval_code": "x", "extra": 1}), "not json", "x" * 20_000]
        for raw in cases:
            with self.subTest(raw=raw[:50]):
                rec = wiki_worker.run(raw, opener=mock.Mock(side_effect=AssertionError("must not open")))
                self.assertEqual((rec["status"], rec["requests"]), ("worker_refused", 0))

    def test_wrong_code_refused(self):
        data = json.loads(self.payload())
        data["approval_code"] = "00000000"
        rec = wiki_worker.run(json.dumps(data), opener=mock.Mock(side_effect=AssertionError("must not open")))
        self.assertEqual(rec["status"], "worker_refused")

    def test_valid_plan_sends_one_request_through_the_adapter(self):
        body = json.dumps({"query": {"search": [{"ns": 0, "title": "Python (programming language)", "pageid": 23862,
                                                 "snippet": "<span>Python</span> is a language"}]}}).encode()

        class Resp(io.BytesIO):
            status = 200
            headers = {"Content-Type": "application/json; charset=utf-8"}

            def geturl(self):
                return ws.ENDPOINT + "?x"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        opener = mock.Mock()
        opener.open.return_value = Resp(body)
        # The fixture contact (.invalid) is rightly refused as a placeholder; relax ONLY that rule here.
        with mock.patch.object(ws, "_PLACEHOLDER", re.compile(r"(?!x)x")):
            rec = wiki_worker.run(self.payload(), opener=opener)
        self.assertEqual((rec["status"], rec["requests"], rec["query"]), ("ok", 1, QUERY))
        self.assertEqual(opener.open.call_count, 1)
        req = opener.open.call_args[0][0]
        self.assertEqual(req.get_header("User-agent"), UA)
        self.assertEqual(wc.normalize_search_record(rec, QUERY)["status"], "success")


class ProtocolAndPlumbing(unittest.TestCase):
    def test_framing_limits(self):
        buf = io.StringIO()
        bp.send(buf, {"type": "done"})
        buf.seek(0)
        self.assertEqual(bp.recv(buf, bp.AGENT_TYPES), {"type": "done"})
        for raw in ('{"type": "evil"}\n', "[1]\n", "nope\n", '{"type": "done"}', "x" * (bp.MAX_MESSAGE_BYTES + 5)):
            with self.subTest(raw=raw[:20]), self.assertRaises(bp.ProtocolError):
                bp.recv(io.StringIO(raw), bp.AGENT_TYPES)
        with self.assertRaises(EOFError):
            bp.recv(io.StringIO(""), bp.AGENT_TYPES)
        with self.assertRaises(bp.ProtocolError):
            bp.send(io.StringIO(), {"type": "search", "query": "x" * bp.MAX_MESSAGE_BYTES})

    def test_real_pipes_with_a_fake_agent_process(self):
        """The supervisor loop over inherited pipes; the fake agent is a plain Python child (no sandbox)."""
        card = card_for()
        agent = textwrap.dedent(f"""
            import json, sys
            sys.path.insert(0, {str(LAB)!r})
            import broker_protocol as bp
            args = sys.argv[1:]
            b = bp.BrokerClient(int(args[args.index('--broker-in') + 1]), int(args[args.index('--broker-out') + 1]))
            card = json.loads({json.dumps(json.dumps(card))})
            r1 = b.call({{"type": "approval_request", "card": card}}, "approval_response")
            r2 = b.call({{"type": "search", "query": card["arguments"]["query"], "thread_id": card["thread_id"],
                         "proposal_id": card["proposal_id"], "digest": card["digest"]}}, "search_result")
            r3 = b.call({{"type": "search", "query": card["arguments"]["query"], "thread_id": card["thread_id"],
                         "proposal_id": card["proposal_id"], "digest": card["digest"]}}, "search_result")
            b.notify({{"type": "done"}})
            print(json.dumps([r1["decision"], r2["record"]["status"], r3["record"]["error"], sys.argv[-1]]))
        """)
        spawned = []
        s = sv.Supervisor(REQ, UA, lambda c: True, "test-human", limiter=FakeLimiter(),
                          spawn=lambda p, c: spawned.append(p) or {"query": p["query"], "status": "ok", "requests": 1,
                                                                   "results": []}, out=lambda *_: None)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_agent.py"
            script.write_text(agent)
            relayed = []
            result = sv.run(REQ, None, "test-human", agent_cmd=[sys.executable, str(script)], supervisor=s,
                            out=relayed.append)
            self.assertEqual(result["agent_exit"], 0)
            agent_lines = [x for x in relayed if x.startswith("agent| [")]
            self.assertEqual(json.loads(agent_lines[-1][len("agent| "):]),
                             ["approve", "ok", "session_search_limit", REQ])
        self.assertEqual(len(spawned), 1)
        self.assertEqual([e["event"] for e in s.audit],
                         ["approval_decision", "search_executed", "search_refused", "agent_done"])

    def test_protocol_violation_kills_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "bad_agent.py"
            script.write_text("import os,sys,time\nargs=sys.argv[1:]\nfd=int(args[args.index('--broker-out')+1])\n"
                              "os.write(fd, b'{\"type\": \"exec\"}\\n')\ntime.sleep(30)\n")
            s = sv.Supervisor(REQ, UA, lambda c: True, "t", limiter=FakeLimiter(), spawn=lambda p, c: {},
                              out=lambda *_: None)
            result = sv.run(REQ, None, "t", agent_cmd=[sys.executable, str(script)], supervisor=s)
        self.assertNotEqual(result["agent_exit"], 0)
        self.assertEqual(result["audit"][0]["event"], "protocol_error")
        self.assertEqual(result["searches"], 0)


class Phase5CHardening(unittest.TestCase):
    def test_5c_profiles_only_add_restrictions_to_the_5b_profiles(self):
        for old, new in ((sv.AGENT_PROFILE, sv.AGENT_PROFILE_5C), (sv.SEARCH_PROFILE, sv.SEARCH_PROFILE_5C)):
            self.assertTrue(new.startswith(old))
            added = new[len(old):]
            self.assertNotIn("allow network", added)
            self.assertNotIn("allow default", added)
            for rule in re.findall(r"\(allow ([\w*-]+)", added):  # every new allow is inside a fresh deny
                self.assertIn(f"(deny {rule})", added)
        self.assertIn("(deny network-outbound (remote unix-socket))", sv.AGENT_PROFILE_5C)
        self.assertNotIn("sam-toolkit-mcp", sv.SEARCH_PROFILE_5C)

    def spawn_with_fake_popen(self, agent_out=b""):
        seen = {}

        class P:
            def __init__(self, cmd, **kw):
                seen.update(cmd=cmd, kw=kw)
                self.stdout = io.StringIO(agent_out.decode())  # the supervisor closes the child pipe ends
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

            def kill(self):
                pass
        relayed = []
        with mock.patch.object(sv.subprocess, "Popen", P):
            sv.run(REQ, None, "t", supervisor=sv.Supervisor(REQ, UA, lambda c: False, "t", limiter=FakeLimiter(),
                                                            spawn=lambda p, c: {}, out=lambda *_: None),
                   answer_mode="model", out=relayed.append)
        return seen, relayed

    def test_agent_gets_new_session_5c_profile_minimal_env_and_no_terminal(self):
        seen, _ = self.spawn_with_fake_popen()
        self.assertEqual(seen["cmd"][:3], ["/usr/bin/sandbox-exec", "-p", sv.AGENT_PROFILE_5C])
        self.assertTrue(seen["kw"]["start_new_session"])
        self.assertEqual(seen["kw"]["env"], sv.AGENT_ENV)
        self.assertEqual(set(sv.AGENT_ENV), {"PATH", "HOME", "LANG"})
        self.assertEqual((seen["kw"]["stdin"], seen["kw"]["stdout"]), (subprocess.DEVNULL, subprocess.PIPE))
        self.assertIn("--answer-mode", seen["cmd"])
        self.assertEqual(seen["cmd"][seen["cmd"].index("--answer-mode") + 1], "model")

    def test_agent_output_is_relayed_without_terminal_control_sequences(self):
        hostile = "[agent] hi\x1b[2J\x1b[1A\rAPPROVAL CODE : FAKE1234\u202e\x07\n"
        _, relayed = self.spawn_with_fake_popen(hostile.encode())
        self.assertEqual(len(relayed), 1)
        line = relayed[0]
        self.assertTrue(line.startswith("agent| "))
        for bad in ("\x1b", "\r", "\x07", "\u202e"):
            self.assertNotIn(bad, line)

    def test_card_rendering_strips_control_characters(self):
        card = card_for(user_reference="Python\x1b[2K", request=REQ)
        self.assertNotIn("\x1b", sv.render_card(card))

    def test_message_flood_kills_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "flood.py"
            script.write_text("import os,sys,time\nargs=sys.argv[1:]\nfd=int(args[args.index('--broker-out')+1])\n"
                              "[os.write(fd, b'{\"type\": \"done\"}\\n') for _ in range(100)]\ntime.sleep(30)\n")
            s = sv.Supervisor(REQ, UA, lambda c: True, "t", limiter=FakeLimiter(), spawn=lambda p, c: {},
                              out=lambda *_: None)
            result = sv.run(REQ, None, "t", agent_cmd=[sys.executable, str(script)], supervisor=s,
                            out=lambda *_: None)
        self.assertEqual(result["audit"][-1], {"event": "protocol_error", "reason": "too many messages"})
        self.assertNotEqual(result["agent_exit"], 0)

    def test_fifo_approver_refuses_unsafe_fifos(self):
        card = card_for()
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain"
            plain.write_text(f"approve {card['approval_code']}\n")
            self.assertFalse(sv.fifo_approver(str(plain))(card))          # not a FIFO
            loose = Path(tmp) / "loose.fifo"
            os.mkfifo(loose, 0o622)
            os.chmod(loose, 0o622)
            self.assertFalse(sv.fifo_approver(str(loose))(card))          # writable by others
            self.assertFalse(sv.fifo_approver(str(Path(tmp) / "missing"))(card))


class StaticGuarantees(unittest.TestCase):
    def test_profiles_identical_to_the_tested_scripts(self):
        def profile(script):
            return re.search(r"^PROFILE='(.*)'$", (LAB / script).read_text(), re.M).group(1)
        self.assertEqual(sv.AGENT_PROFILE, profile("run_sandboxed.sh"))
        self.assertEqual(sv.SEARCH_PROFILE, profile("search_lab/run_wiki_sandboxed.sh"))
        self.assertIn('(remote ip "localhost:11434")', sv.AGENT_PROFILE)
        self.assertNotIn("443", sv.AGENT_PROFILE)
        self.assertNotIn("11434", sv.SEARCH_PROFILE)

    def test_supervisor_imports_no_model_mcp_or_graph_code_and_never_calls_http(self):
        code = ("import sys; sys.path.insert(0, %r); import supervisor_5b; bad = [m for m in ('langchain', "
                "'langchain_core', 'langgraph', 'ollama', 'fastmcp', 'mcp', 'httpx', 'requests', 'ddgs', 'primp', "
                "'agent_graph_5b', 'agent_runtime_5b') if m in sys.modules]; print(bad)") % str(LAB)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "[]", out.stderr[-500:])
        # urllib is imported (via the adapter's constants) but the supervisor code never uses HTTP or sockets:
        for name in ("supervisor_5b.py", "wiki_capability.py", "broker_protocol.py"):
            src = (LAB / name).read_text()
            for call in ("search_once", "run_plan", "build_opener", "urlopen", "socket.", "http.client"):
                self.assertNotIn(call, src, f"{name} must not call {call}")

    def test_workspace_label_matches_harness_config(self):
        code = "import sys; sys.path.insert(0, %r); import harness_config as c; print(c.WORKSPACE)" % str(LAB)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.stdout.strip(), sv.WORKSPACE_LABEL, out.stderr[-300:])

    def test_configured_user_agent_is_the_approved_one(self):
        self.assertEqual(wc.load_user_agent(),
                         "SamToolkitSearchLab/0.1 (https://github.com/yige-fei) Python-urllib/3.12")


if __name__ == "__main__":
    unittest.main()
