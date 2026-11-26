"""Phase 5A.1 offline checks of primp/ddgs DNS behaviour. No real lookups or requests.

The resolution test runs only when the macOS resolver socket is blocked (i.e. inside the
all-deny sandbox), so it can never trigger a real DNS query.
"""
import os
import socket
import unittest

import ddgs
import primp
from ddgs.engines.duckduckgo import Duckduckgo

MDNS_SOCKET = "/private/var/run/mDNSResponder"
INVALID_URL = "http://dns-check.invalid:9/"  # RFC 6761 reserved name


def resolver_socket_blocked() -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(MDNS_SOCKET)
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    finally:
        s.close()


class InstalledVersions(unittest.TestCase):
    def test_versions_are_the_investigated_ones(self):
        self.assertEqual(ddgs.__version__, "9.16.1")
        from importlib.metadata import version
        self.assertEqual(version("primp"), "2.0.1")


class ResolverOption(unittest.TestCase):
    def test_system_resolver_is_accepted(self):
        primp.Client(dns_resolver="system")
        primp.Client(dns_resolver=["system"])
        primp.Client(dns_resolver=("system",))

    def test_invalid_resolver_values_are_rejected(self):
        for bad in (5, ["system", 5], "dot://", "dns://"):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                primp.Client(dns_resolver=bad)

    def test_ddgs_client_can_be_replaced_without_editing_the_package(self):
        engine = Duckduckgo(proxy=None, timeout=3, verify=True)
        self.assertIsInstance(engine.http_client.client, primp.Client)
        replacement = primp.Client(timeout=3, dns_resolver="system")
        engine.http_client.client = replacement
        self.assertIs(engine.http_client.client, replacement)
        self.assertEqual(Duckduckgo.search_url, "https://html.duckduckgo.com/html/")  # class default untouched


@unittest.skipUnless(resolver_socket_blocked(), "only runs inside the all-deny sandbox")
class ResolutionPathInsideSandbox(unittest.TestCase):
    def setUp(self):
        for n in ("PRIMP_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ.pop(n, None)

    def test_default_uses_hickory(self):
        with self.assertRaises(primp.DNSError) as cm:
            primp.Client(proxy=None, timeout=3).get(INVALID_URL)
        self.assertIn("no records found for Query", str(cm.exception))  # hickory's message

    def test_system_option_uses_getaddrinfo(self):
        with self.assertRaises(primp.DNSError) as cm:
            primp.Client(proxy=None, timeout=3, dns_resolver="system").get(INVALID_URL)
        self.assertIn("failed to lookup address information", str(cm.exception))  # getaddrinfo's message

    def test_ddgs_engine_with_injected_system_client(self):
        engine = Duckduckgo(proxy=None, timeout=3, verify=True)
        engine.http_client.client = primp.Client(proxy=None, timeout=3, dns_resolver="system")
        engine.search_url = INVALID_URL
        with self.assertRaises(ddgs.exceptions.DDGSException) as cm:
            engine.search("offline dns check")
        self.assertIn("failed to lookup address information", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
