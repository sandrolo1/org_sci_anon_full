"""Semantic Scholar as the third open-access source: asked last, only with a key, one request at a time, the key never leaving its API."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from research0 import fulltext as ft
from test_fulltext import PDF, HTML, Response, FetchError, NOT_IN_OPENALEX, oa, unpaywall, openalex, scripted

KEY = "s2-test-key"
CLOSED = unpaywall(is_oa=False, best_oa_location=None)
NOT_IN_S2 = Response(404, "s", "application/json", b'{"error": "Paper with id DOI:10.1/x not found"}')


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(ft, "SEMANTIC_SCHOLAR_INTERVAL", 0)
    monkeypatch.setattr(ft, "_semantic_scholar_next", 0.0)


def s2(url="https://repo.example/s2.pdf", status="GREEN", license="CCBY"):
    pdf = {"url": url, "status": status, "license": license, "disclaimer": "..."} if url is not None else None
    return Response(200, "s", "application/json", json.dumps({"paperId": "abc", "isOpenAccess": url is not None, "openAccessPdf": pdf}).encode())


def recording(table):
    """scripted() that also keeps the headers of every request."""
    get, headers = scripted(table), {}
    def wrapped(url, **kwargs):
        headers[url] = kwargs.get("headers") or {}
        return get(url, **kwargs)
    wrapped.calls, wrapped.headers = get.calls, headers
    return wrapped


def test_a_copy_only_semantic_scholar_knows_is_downloaded_with_its_licence():
    get = recording({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": s2(), "s2.pdf": Response(200, "u", "application/pdf", PDF)})
    result = ft.acquire("10.1/X", "me@example.org", get, semantic_scholar_key=KEY)
    assert (result["outcome"], result["data"], result["found_by"], result["source"]) == ("downloaded", PDF, "semantic_scholar", ft.RESOLVER_VERSION)
    assert (result["pdf_url"], result["license"], result["oa_status"], result["version"]) == ("https://repo.example/s2.pdf", "CCBY", "green", None)
    asked = get.calls[2]
    assert asked.startswith("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/x?") and "fields=isOpenAccess%2CopenAccessPdf" in asked
    assert get.headers[asked] == {"x-api-key": KEY}
    # The key goes to Semantic Scholar only, never to the other APIs or to the host of the PDF.
    assert all(not h for url, h in get.headers.items() if url != asked)


def test_without_a_key_semantic_scholar_is_never_asked():
    get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX})
    assert ft.acquire("10.1/x", "me@example.org", get)["outcome"] == "not_open_access"
    assert not any("semanticscholar" in call for call in get.calls)


def test_a_file_from_an_earlier_source_never_reaches_semantic_scholar():
    get = scripted({"api.unpaywall.org": oa("https://pub.example/p.pdf"), "p.pdf": Response(200, "u", "application/pdf", PDF)})
    assert ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)["found_by"] == "unpaywall"
    get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": openalex(("https://repo.example/a.pdf", "acceptedVersion")),
                    "a.pdf": Response(200, "u", "application/pdf", PDF)})
    assert ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)["found_by"] == "openalex"
    assert not any("semanticscholar" in call for call in get.calls)


def test_nothing_new_from_semantic_scholar_leaves_the_earlier_answer():
    for answer in (NOT_IN_S2, s2(url=None), s2(url=""), s2(status="CLOSED"), s2(url="javascript:alert(1)")):
        get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": answer})
        result = ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)
        assert (result["outcome"], result["found_by"], result["data"]) == ("not_open_access", "unpaywall", None)
        assert sum("semanticscholar" in call for call in get.calls) == 1 and len(get.calls) == 3


def test_a_link_another_source_already_tried_is_not_repeated():
    get = scripted({"api.unpaywall.org": oa("https://one.example/a.pdf"), "one.example": Response(403, "u", "", b""),
                    "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": s2(url="https://one.example/a.pdf")})
    result = ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)
    assert result["outcome"] == "download_failed" and result["found_by"] == "unpaywall"
    assert sum("one.example" in call for call in get.calls) == 1


def test_a_semantic_scholar_link_that_fails_is_reported_like_any_other():
    get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": s2(), "s2.pdf": Response(200, "u", "text/html", HTML)})
    assert ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)["outcome"] == "not_a_pdf"
    get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": s2(),
                    "s2.pdf": Response(403, "u", "text/html", b"<title>Just a moment...</title>", challenge=True)})
    assert ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)["outcome"] == "blocked_by_site"


def test_semantic_scholar_failing_is_not_a_final_answer_so_the_paper_is_asked_again():
    for answer, words in ((FetchError("timed out"), "unreachable"), (Response(500, "s", "", b"oops"), "HTTP 500"),
                          (Response(429, "s", "application/json", b'{"message": "Too Many Requests"}'), "HTTP 429"),
                          (Response(403, "s", "application/json", b'{"message": "Forbidden"}'), "check SEMANTIC_SCHOLAR_API_KEY"),
                          (Response(200, "s", "application/json", b"not json"), "HTTP 200")):
        result = ft.acquire("10.1/x", "me@example.org", scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": NOT_IN_OPENALEX,
                                                                   "api.semanticscholar.org": answer}), semantic_scholar_key=KEY)
        assert result["outcome"] == "lookup_failed" and result["outcome"] in ft.RETRY and "Semantic Scholar" in result["detail"] and words in result["detail"]


def test_an_earlier_outage_stays_retryable_even_when_semantic_scholar_gives_a_final_answer():
    get = scripted({"api.unpaywall.org": CLOSED, "api.openalex.org": FetchError("down"), "api.semanticscholar.org": s2(),
                    "s2.pdf": Response(200, "u", "text/html", HTML)})
    result = ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)
    assert result["outcome"] == "lookup_failed" and "OpenAlex" in result["detail"]
    get = scripted({"api.unpaywall.org": FetchError("down"), "api.openalex.org": NOT_IN_OPENALEX, "api.semanticscholar.org": FetchError("down too")})
    detail = ft.acquire("10.1/x", "me@example.org", get, semantic_scholar_key=KEY)["detail"]
    assert "Unpaywall" in detail and "Semantic Scholar" in detail


def test_requests_from_parallel_workers_go_one_at_a_time_and_spaced(monkeypatch):
    monkeypatch.setattr(ft, "SEMANTIC_SCHOLAR_INTERVAL", 0.05)
    spans, active, lock = [], [0], threading.Lock()

    def get(url, **kwargs):
        with lock:
            active[0] += 1
            assert active[0] == 1, "two Semantic Scholar requests at once"
        start = time.monotonic()
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        spans.append((start, time.monotonic()))
        return NOT_IN_S2

    workers = [threading.Thread(target=ft._semantic_scholar_get, args=(get, "https://api.semanticscholar.org/x", KEY, "test")) for _ in range(6)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    spans.sort()
    assert len(spans) == 6
    assert all(later[0] - earlier[1] >= 0.045 for earlier, later in zip(spans, spans[1:]))


def test_the_process_reads_the_key_from_the_environment(monkeypatch):
    seen = []
    monkeypatch.setattr(ft, "acquire", lambda doi, email, get, openalex_key, s2_key: seen.append(s2_key) or {"outcome": "no_doi", "data": None})
    stop = threading.Event()
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", f"  {KEY} ")
    ft._work(("id", "10.1/x"), "me@example.org", None, None, stop)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")
    ft._work(("id", "10.1/x"), "me@example.org", None, None, stop)
    assert seen == [KEY, None]


class Redirecting(BaseHTTPRequestHandler):
    seen = {}

    def do_GET(self):
        Redirecting.seen[self.path] = self.headers.get("x-api-key")
        if self.path == "/api":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def test_the_key_header_is_not_carried_over_a_redirect(monkeypatch):
    # Loopback is refused in production; allowed here only to watch what a real redirect carries.
    monkeypatch.setattr(ft, "_public_socket", lambda host, port, timeout: socket.create_connection((host, port), timeout))
    server = HTTPServer(("127.0.0.1", 0), Redirecting)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reply = ft.http_get(f"http://127.0.0.1:{server.server_address[1]}/api", max_bytes=1000, user_agent="test", timeout=3, headers={"x-api-key": KEY})
    finally:
        server.shutdown()
    assert reply.status == 200 and Redirecting.seen == {"/api": KEY, "/elsewhere": None}
