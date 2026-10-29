"""Regression cases for lost OA copies, HTML discovery, retry policy and cohort coverage."""
import json
import time
from email.utils import formatdate

import pytest

from research0 import fulltext as ft
from test_fulltext import PDF, HTML, Response, FetchError, TooLarge, oa, unpaywall, openalex, scripted, NOT_IN_OPENALEX


def test_unpaywall_outage_still_downloads_an_openalex_repository_copy():
    get = scripted({"unpaywall": FetchError("outage"), "openalex": openalex(("https://repo.example/p.pdf", "acceptedVersion")),
                    "p.pdf": Response(200, "https://cdn.example/p.pdf", "application/pdf", PDF)})
    r = ft.acquire("https://doi.org/10.1234/Test", "me@example.org", get)
    assert (r["outcome"], r["found_by"], r["version"], r["doi"]) == ("downloaded", "openalex", "accepted", "10.1234/test")
    assert r["pdf_url"] == "https://cdn.example/p.pdf"


def test_fourth_copy_is_tried_after_three_refusals():
    get = scripted({"unpaywall": oa(*(f"https://repo.example/{i}.pdf" for i in range(4))),
                    **{f"/{i}.pdf": Response(403, "u", "", b"") for i in range(3)},
                    "/3.pdf": Response(200, "u", "application/pdf", PDF)})
    assert ft.acquire("10.1/x", "me@example.org", get)["outcome"] == "downloaded"


@pytest.mark.parametrize("provider", ["unpaywall", "openalex"])
def test_page_only_copy_exposes_relative_citation_pdf_with_version(provider):
    loc = {"url_for_landing_page": "https://repo.example/item/1", "version": "acceptedVersion", "license": "cc-by"}
    table = {"unpaywall": unpaywall(best_oa_location=loc),
             "/item/1": Response(200, "https://repo.example/record/1", "text/html",
                                 b'<meta name="citation_doi" content="10.1/x"><meta name="citation_pdf_url" content="../files/main.pdf?download=1&amp;v=2">'),
             "/files/main.pdf?download=1&v=2": Response(200, "u", "application/octet-stream", PDF)}
    if provider == "openalex":
        table["unpaywall"] = unpaywall(is_oa=False)
        table["openalex"] = openalex((None, "acceptedVersion"), landing="https://repo.example/item/1")
    r = ft.acquire("10.1/x", "me@example.org", scripted(table))
    assert (r["outcome"], r["version"], r["found_by"]) == ("downloaded", "accepted", provider)
    assert r["landing_url"] == "https://repo.example/record/1"


def test_html_returned_by_a_pdf_endpoint_is_resolved():
    get = scripted({"unpaywall": oa("https://repo.example/view.pdf"),
                    "view.pdf": Response(200, "https://repo.example/view.pdf", "text/html", b'<iframe src="/viewer?file=%2Ffiles%2Fmain.pdf"></iframe>'),
                    "/files/main.pdf": Response(200, "u", "application/pdf", PDF)})
    assert ft.acquire("10.1/x", "me@example.org", get)["data"] == PDF


def test_repository_bitstream_and_supplements_are_distinguished():
    page = b'''<a href="/bitstream/id/supplement.pdf">Download</a>
               <a href="https://other.example/cited.pdf">PDF</a>
               <a href="/bitstream/id/article.pdf">manuscript</a>
               <a href="/files/privacy.pdf">Privacy policy</a>'''
    assert ft.page_pdf_links(page, "https://repo.example/item/1", "10.1/x") == ["https://repo.example/bitstream/id/article.pdf"]


def test_different_article_metadata_is_not_downloaded():
    page = b'<meta name="citation_doi" content="10.1/other"><meta name="citation_pdf_url" content="/wrong.pdf">'
    assert ft.page_pdf_links(page, "https://repo.example/1", "10.1/x") == []


@pytest.mark.parametrize("link", ["javascript:alert(1)", "file:///secret", "ftp://repo.example/p.pdf", "https://user:pass@repo.example/p.pdf"])
def test_page_links_cannot_introduce_unsupported_schemes_or_credentials(link):
    assert ft.page_pdf_links(f'<meta name="citation_pdf_url" content="{link}">'.encode(), "https://repo.example/1") == []


def test_discovered_link_still_passes_through_public_address_guard():
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if "unpaywall" in url:
            return oa("https://repo.example/p.pdf")
        if url == "https://repo.example/p.pdf":
            return Response(200, url, "text/html", b'<meta name="citation_pdf_url" content="http://127.0.0.1/private.pdf">')
        if url == "http://127.0.0.1/private.pdf":
            return ft._http_get_once(url, **kwargs)
        if "openalex" in url:
            return NOT_IN_OPENALEX
        return Response(200, url, "text/html", HTML)
    r = ft.acquire("10.1/x", "me@example.org", get)
    assert r["data"] is None and "non-public" in r["detail"]


def test_unknown_license_does_not_inherit_a_different_copys_license():
    get = scripted({"unpaywall": unpaywall(best_oa_location={"url_for_pdf": "https://repo.example/a.pdf", "license": "cc-by"},
                                           oa_locations=[{"url_for_pdf": "https://repo.example/b.pdf", "version": "submittedVersion"}]),
                    "/a.pdf": Response(403, "u", "", b""), "/b.pdf": Response(200, "u", "application/pdf", PDF)})
    r = ft.acquire("10.1/x", "me@example.org", get)
    assert r["license"] is None and r["version"] == "submitted"


def test_no_closed_locations_are_scraped_even_if_work_is_open():
    record = {"open_access": {"is_oa": True}, "best_oa_location": None,
              "locations": [{"is_oa": False, "pdf_url": "https://closed.example/x.pdf", "landing_page_url": "https://closed.example/1"}]}
    get = scripted({"unpaywall": unpaywall(is_oa=False), "openalex": Response(200, "o", "application/json", json.dumps(record).encode())})
    assert ft.acquire("10.1/x", "me@example.org", get)["outcome"] == "no_pdf_link"
    assert len(get.calls) == 2


def test_html_cycles_and_request_count_are_bounded(monkeypatch):
    monkeypatch.setattr(ft, "MAX_DOCUMENT_REQUESTS", 3)
    get = scripted({"unpaywall": oa("https://repo.example/one.pdf"),
                    "one.pdf": Response(200, "u", "text/html", b'<meta name="citation_pdf_url" content="https://repo.example/two.pdf">'),
                    "two.pdf": Response(200, "u", "text/html", b'<meta name="citation_pdf_url" content="https://repo.example/one.pdf">'),
                    "openalex": openalex(("https://repo.example/one.pdf", "acceptedVersion"))})
    assert ft.acquire("10.1/x", "me@example.org", get)["data"] is None
    assert len([u for u in get.calls if not any(p in u for p in ("unpaywall", "openalex"))]) <= 3
    assert get.calls.count("https://repo.example/one.pdf") == 1


@pytest.mark.parametrize("first,delays", [(Response(429, "u", "", b"", "3"), [3]),
                                          (Response(503, "u", "", b""), [1]), (FetchError("reset"), [1])])
def test_transient_network_failure_recovers_with_backoff(monkeypatch, first, delays):
    replies = iter([first, Response(200, "u", "application/pdf", PDF)])
    def once(*args, **kwargs):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    slept = []
    monkeypatch.setattr(ft, "_http_get_once", once)
    monkeypatch.setattr(ft.time, "sleep", slept.append)
    assert ft.http_get("https://repo.example/1", max_bytes=1000, user_agent="test").body == PDF
    assert slept == delays


@pytest.mark.parametrize("reply", [Response(403, "u", "", b""), Response(404, "u", "", b""), Response(429, "u", "", b"", "3600"),
                                   Response(503, "u", "", b"", formatdate(time.time() + 3600, usegmt=True))])
def test_permanent_refusals_and_long_cooldowns_are_not_retried(monkeypatch, reply):
    calls = []
    monkeypatch.setattr(ft, "_http_get_once", lambda *a, **k: calls.append(1) or reply)
    monkeypatch.setattr(ft.time, "sleep", lambda s: pytest.fail("must not sleep"))
    assert ft.http_get("https://repo.example/1", max_bytes=1000, user_agent="test") == reply
    assert len(calls) == 1


def test_size_limit_never_retried(monkeypatch):
    def once(*args, **kwargs):
        raise TooLarge("oversized")
    monkeypatch.setattr(ft, "_http_get_once", once)
    monkeypatch.setattr(ft.time, "sleep", lambda s: pytest.fail("must not sleep"))
    with pytest.raises(TooLarge):
        ft.http_get("https://repo.example/1", max_bytes=1000, user_agent="test")


def test_old_two_provider_failures_rechecked_once_and_existing_files_skipped(con, world):
    ids = [r[0] for r in ft.select_papers(con, world.run_id, "eligible", include_finished=True)]
    # Mock world already contains files; pick two that are genuinely missing.
    missing = [r[0] for r in con.execute("SELECT canonical_id FROM canonical_paper c WHERE canonical_id=ANY(%s) AND NOT EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id=c.canonical_id) LIMIT 2", (ids,))]
    for cid, source in zip(missing, ["unpaywall+openalex", ft.RESOLVER_VERSION]):
        con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,source,outcome) VALUES (%s,%s,%s,'no_pdf_link')", (world.run_id, cid, source))
    chosen = {r[0] for r in ft.select_papers(con, world.run_id, "eligible")}
    assert missing[0] in chosen and missing[1] not in chosen
    stored = {r[0] for r in con.execute("SELECT canonical_id FROM fulltext_file WHERE canonical_id=ANY(%s)", (ids,))}
    assert not chosen & stored
    report = ft.coverage(con, world.run_id, "eligible")
    assert report["papers"] == len(ids) and report["with_file"] == len(stored)
    assert report["coverage_percent"] == round(100 * len(stored) / len(ids), 2)


def test_cli_keeps_completed_attempts_if_a_later_download_aborts(world, monkeypatch):
    import sys
    from research0 import cli
    recorded = []
    def fail_after_one(con, run_id, papers, email):
        cid = papers[0][0]
        ft._record(con, run_id, cid, ft.acquire(None, email))
        recorded.append(cid)
        raise RuntimeError("later storage failure")
    monkeypatch.setattr(cli, "database", world.connect)
    monkeypatch.setattr(ft, "process", fail_after_one)
    monkeypatch.setenv("UNPAYWALL_EMAIL", "test@example.org")
    monkeypatch.setattr(sys, "argv", ["research0", "fulltext-fetch", "--run-id", str(world.run_id), "--scope", "eligible", "--limit", "1"])
    with pytest.raises(RuntimeError, match="later storage"):
        cli.main()
    with world.connect() as con:
        assert con.execute("SELECT outcome FROM fulltext_attempt WHERE canonical_id=%s", (recorded[0],)).fetchone() == ("no_doi",)
