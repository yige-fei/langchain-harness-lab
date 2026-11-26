"""Phase 5A.1 OFFLINE check of which DNS path primp/ddgs use. Run inside a sandbox that
denies ALL networking: every lookup is refused by the kernel before a packet is sent, and
macOS records the refused destination, which shows the mechanism used.

Only reserved `.invalid` names (RFC 6761) are used, so even the attempted lookups refer
to names that cannot exist. Nothing is executed against a real search provider.

    usage: python dns_mode_check.py {hickory-default|system|ddgs-system}
"""
import os
import sys

for _name in ("PRIMP_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_name, None)

import primp  # noqa: E402

TEST_URL = "http://dns-check.invalid:9/"  # reserved TLD, discard port, plain HTTP


def system_client():
    """The ddgs HttpClient settings, plus the documented primp option dns_resolver="system"."""
    return primp.Client(proxy=None, timeout=3, impersonate="random", impersonate_os="random",
                        verify=True, dns_resolver="system")


def main(mode: str) -> None:
    print(f"pid={os.getpid()} mode={mode}", flush=True)
    if mode == "hickory-default":
        client = primp.Client(proxy=None, timeout=3)  # what ddgs builds today (no dns_resolver)
        target = TEST_URL
        call = lambda: client.get(target)  # noqa: E731
    elif mode == "system":
        client = system_client()
        call = lambda: client.get(TEST_URL)  # noqa: E731
    elif mode == "ddgs-system":
        import ddgs
        from ddgs.engines.duckduckgo import Duckduckgo
        assert ddgs.__version__ == "9.16.1" and primp.__name__ == "primp"
        engine = Duckduckgo(proxy=None, timeout=3, verify=True)
        engine.http_client.client = system_client()      # inject the resolver choice (no package edits)
        engine.http_client.client.headers_update(engine.headers_update)
        engine.search_url = TEST_URL                      # instance override: never the real endpoint
        call = lambda: engine.search("offline dns check")  # noqa: E731
    else:
        raise SystemExit("unknown mode")
    try:
        result = call()
        print(f"RESULT: unexpected success {result!r}"[:200])
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT: {type(exc).__module__}.{type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    main(sys.argv[1])
