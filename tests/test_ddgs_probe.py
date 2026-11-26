"""Offline tests for search_lab/ddgs_probe.py (Phase 5A.2). No real search request is made:
fake clients are used, and the real engine is only constructed (never sent) or pointed at a
reserved `.invalid` name inside the all-deny sandbox."""
import json
import re
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "search_lab"))

import ddgs_probe as probe  # noqa: E402

SANDBOX_SCRIPT = LAB / "search_lab" / "run_search_sandboxed.sh"
EXPECTED_PROFILE = ('(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote tcp "*:443"))'
                    '(allow network-outbound (remote unix-socket (path-literal "/private/var/run/mDNSResponder")))')
GOOD = [{"title": "csv — CSV File Reading", "href": "https://docs.python.org/3/library/csv.html", "body": "The csv module"}]
CLEAN_SCUTIL = "<dictionary> {\n  ExceptionsList : <array> {\n    0 : *.local\n  }\n  FTPPassive : 1\n}\n"


def resolver_socket_blocked() -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect("/private/var/run/mDNSResponder")
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    finally:
        s.close()


def plan(**changes):
    p = {**probe.PLAN_CONSTANTS, "queries": ["Python csv module documentation"]}
    p.update(changes)
    return p


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.requests = list(outcomes), [], []

    def text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class NoNetwork(unittest.TestCase):
    def setUp(self):
        self.net = []

        def blocked(*a, **k):
            self.net.append(a)
            raise OSError("blocked in tests")
        for target, name in ((socket.socket, "connect"), (socket, "create_connection")):
            p = mock.patch.object(target, name, blocked)
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.assertEqual(self.net, [], "a network connection was attempted")


# 1. DDGS uses the injected system-resolver client ---------------------------------------
class InjectedClient(NoNetwork):
    def test_engine_uses_injected_system_resolver_client(self):
        import primp
        engine = probe.build_engine()
        self.assertEqual(type(engine).__name__, "Duckduckgo")
        self.assertIsInstance(engine.http_client.client, primp.Client)
        self.assertEqual(engine.search_url, "https://html.duckduckgo.com/html/")

    @unittest.skipUnless(resolver_socket_blocked(), "only runs inside the all-deny sandbox")
    def test_engine_request_goes_through_getaddrinfo(self):
        client = probe.make_client()
        client.engine.search_url = "http://dns-check.invalid:9/"  # instance-only, reserved name
        with self.assertRaises(Exception) as cm:
            client.text("offline check", region="us-en", max_results=5, backend="duckduckgo")
        self.assertIn("failed to lookup address information", str(cm.exception))  # system resolver
        self.assertEqual([r["url"] for r in client.requests], ["http://dns-check.invalid:9/"])

    def test_structure_change_fails_closed(self):
        from ddgs.engines.duckduckgo import Duckduckgo
        with mock.patch.object(Duckduckgo, "search_url", "https://elsewhere.example/html/"):
            with self.assertRaises(probe.IncompatibleLibrary):
                probe.build_engine()
        with mock.patch.object(Duckduckgo, "search_method", "GET"):
            with self.assertRaises(probe.IncompatibleLibrary):
                probe.build_engine()
        with mock.patch("ddgs.engines.duckduckgo.Duckduckgo.__init__",
                        lambda self, **k: setattr(self, "http_client", object())):
            with self.assertRaises(probe.IncompatibleLibrary):
                probe.build_engine()


# 2. Incompatible versions are rejected ----------------------------------------------------
class Versions(NoNetwork):
    def test_exact_versions_installed(self):
        probe.check_versions()

    def test_other_versions_rejected(self):
        for found in ({"ddgs": "9.16.2", "primp": "2.0.1"}, {"ddgs": "9.16.1", "primp": "2.0.0"},
                      {"ddgs": None, "primp": "2.0.1"}):
            with self.subTest(found=found):
                with self.assertRaises(probe.IncompatibleLibrary):
                    probe.check_versions(found.get)
                self.assertTrue(probe.preflight(found.get, lambda: {}, lambda: CLEAN_SCUTIL))

    def test_build_engine_checks_versions_first(self):
        with mock.patch.object(probe, "_installed_version", lambda n: "0.0"):
            with self.assertRaises(probe.IncompatibleLibrary):
                probe.build_engine()


# 3. No search request during initialization or approval preparation -----------------------
class NoRequestBeforeApproval(NoNetwork):
    def test_construction_and_preparation_send_nothing(self):
        p = probe.load_plan()
        probe.approval_code(p)
        self.assertEqual(probe.preflight(scutil=lambda: CLEAN_SCUTIL), [])
        client = probe.make_client()
        self.assertEqual(client.requests, [])

    def test_dry_run_never_builds_a_client(self):
        with mock.patch.object(probe, "make_client", side_effect=AssertionError("built a client")), \
             mock.patch.object(probe, "preflight", return_value=[]), mock.patch("builtins.print"):
            self.assertEqual(probe.main([]), 0)

    def test_wrong_approval_sends_nothing(self):
        built = []
        for code in ("", "WRONG123", probe.approval_code(plan(queries=["another query"]))):
            with self.assertRaises(PermissionError):
                probe.run_plan(plan(), code, client_factory=lambda: built.append(1))
        self.assertEqual(built, [])

    def test_preflight_failure_blocks_run(self):
        with mock.patch.object(probe, "preflight", return_value=["proxy on"]), \
             mock.patch.object(probe, "run_plan", side_effect=AssertionError("ran")), mock.patch("builtins.print"):
            self.assertEqual(probe.main(["--approved", probe.approval_code(probe.load_plan())]), 2)


# 4 + 5. Sandbox unchanged; no direct port-53 permission ------------------------------------
class SandboxProfile(unittest.TestCase):
    def profile(self):
        m = re.search(r"PROFILE='([^']*)'", SANDBOX_SCRIPT.read_text())
        self.assertIsNotNone(m)
        return m.group(1)

    def test_profile_is_exactly_the_phase5a_profile(self):
        self.assertEqual(self.profile(), EXPECTED_PROFILE)

    def test_no_dns_port_or_udp_or_extra_allowances(self):
        prof = self.profile()
        self.assertNotIn(":53", prof)
        self.assertNotIn("udp", prof)
        self.assertEqual(len(re.findall(r"\(allow network", prof)), 2)
        self.assertIn("(deny network-outbound)", prof)


# 6. Only the pinned backend ---------------------------------------------------------------
class PinnedBackend(NoNetwork):
    def test_text_refuses_other_backends(self):
        client = probe.PinnedDuckDuckGo(SimpleNamespace(http_client=SimpleNamespace(request=lambda *a, **k: None)))
        for backend in ("auto", "bing", "google", "duckduckgo,bing"):
            with self.assertRaises(ValueError):
                client.text("q", region="us-en", max_results=5, backend=backend)

    def test_probe_never_uses_the_multi_provider_orchestrator(self):
        src = (LAB / "search_lab" / "ddgs_probe.py").read_text()
        self.assertNotRegex(src, r"\bDDGS\(")
        self.assertNotIn("from ddgs import DDGS", src)

    def test_fake_run_passes_pinned_backend(self):
        client = FakeClient([GOOD])
        probe.run_plan(plan(), probe.approval_code(plan()), client_factory=lambda: client)
        self.assertEqual(client.calls[0][1], {"region": "us-en", "max_results": 5, "backend": "duckduckgo"})


# 7. Query text and configuration are bound to the approval code ---------------------------
class ApprovalBinding(NoNetwork):
    def test_every_field_changes_the_code(self):
        base = probe.approval_code(plan())
        for change in ({"queries": ["python csv module documentation"]}, {"region": "uk-en"},
                       {"max_results": 4}, {"timeout_s": 11}, {"dns_mode": "hickory"}, {"port": 53},
                       {"hostname": "duckduckgo.com"}, {"app_retries": 1}, {"other_providers": "enabled"},
                       {"backend": "auto"}, {"paid_apis": "some"}, {"provider": "Bing"}):
            with self.subTest(change=change):
                self.assertNotEqual(probe.approval_code(plan(**change)), base)

    def test_plan_file_rejects_changed_configuration(self):
        for change in ({"dns_mode": "hickory"}, {"port": 53}, {"app_retries": 1}, {"backend": "auto"},
                       {"other_providers": "enabled"}, {"queries": ["a", "b"]}, {"queries": ["see ~/notes"]},
                       {"extra": True}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                f = Path(d) / "plan.json"
                f.write_text(json.dumps(plan(**change)))
                with self.assertRaises(ValueError):
                    probe.load_plan(f)

    def test_shipped_plan_matches_the_requested_plan(self):
        self.assertEqual(probe.load_plan(), plan())


# 8. Failures stop; no application-level retries --------------------------------------------
class FailuresStop(NoNetwork):
    def test_failures_classified_with_one_call(self):
        cases = [(type("TimeoutException", (Exception,), {})("timed out"), "timeout"),
                 (probe.RatelimitException("HTTP 202"), "rate_limited"),
                 (probe.DDGSException("No results found."), "empty_or_blocked"),
                 (probe.DDGSException("HTTP 403"), "provider_error"),
                 (RuntimeError("boom in /Users/alice/x"), "unexpected_error")]
        for exc, status in cases:
            with self.subTest(status=status):
                client = FakeClient([exc, GOOD])
                records = probe.run_plan(plan(), probe.approval_code(plan()), client_factory=lambda: client)
                self.assertEqual((len(client.calls), records[0]["status"]), (1, status))
                self.assertNotIn("/Users/alice", records[0]["error"])

    def test_non_200_becomes_error_without_second_request(self):
        sent = []

        class FakeEngine:
            def __init__(self, status):
                self.status = status
                self.http_client = SimpleNamespace(request=self.raw_request)

            def raw_request(self, method, url, *a, **k):
                sent.append(url)
                return SimpleNamespace(status_code=self.status, content=b"", _resp=SimpleNamespace(url=url))

            def search(self, query, **kw):
                resp = self.http_client.request("POST", "https://html.duckduckgo.com/html/", data={"q": query})
                return None if resp.status_code != 200 else []
        for status, kind in ((202, probe.RatelimitException), (429, probe.RatelimitException), (403, probe.DDGSException)):
            with self.subTest(status=status):
                sent.clear()
                client = probe.PinnedDuckDuckGo(FakeEngine(status))
                with self.assertRaises(kind):
                    client.text("q", region="us-en", max_results=5, backend="duckduckgo")
                self.assertEqual(len(sent), 1)
                self.assertEqual(client.requests[0]["status"], status)

    def test_client_construction_failure_is_recorded_not_retried(self):
        calls = []

        def broken():
            calls.append(1)
            raise probe.IncompatibleLibrary("changed")
        records = probe.run_plan(plan(), probe.approval_code(plan()), client_factory=broken)
        self.assertEqual((len(calls), records[0]["status"]), (1, "incompatible_library"))

    def test_results_bounded_and_unsafe_urls_dropped(self):
        raw = [{"title": "T" * 999, "href": "https://ok.example/a", "body": "b‮" * 999},
               {"title": "x", "href": "javascript:alert(1)", "body": ""},
               {"title": "x", "href": "file:///etc/passwd", "body": ""}]
        client = FakeClient([raw])
        r = probe.run_plan(plan(), probe.approval_code(plan()), client_factory=lambda: client)[0]
        self.assertEqual((len(r["results"]), r["dropped"]), (1, 2))
        self.assertEqual(len(r["results"][0]["title"]), probe.MAX_TITLE_CHARS)
        self.assertNotIn("‮", r["results"][0]["snippet"])


class ProxyChecks(unittest.TestCase):
    def test_clean_system(self):
        self.assertEqual(probe.proxy_problems(lambda: {}, lambda: CLEAN_SCUTIL), [])

    def test_detects_proxies(self):
        self.assertTrue(probe.proxy_problems(lambda: {"https": "http://proxy:3128"}, lambda: CLEAN_SCUTIL))
        for flag in ("HTTPSEnable", "SOCKSEnable", "ProxyAutoConfigEnable", "ProxyAutoDiscoveryEnable"):
            self.assertTrue(probe.proxy_problems(lambda: {}, lambda: f"<dictionary> {{\n  {flag} : 1\n}}"))
        with mock.patch.dict("os.environ", {"PRIMP_PROXY": "socks5://x:1"}):
            self.assertTrue(probe.proxy_problems(lambda: {}, lambda: CLEAN_SCUTIL))

    def test_unreadable_settings_fail_closed(self):
        def boom():
            raise OSError("no scutil")
        self.assertTrue(probe.proxy_problems(lambda: {}, boom))


if __name__ == "__main__":
    unittest.main()
