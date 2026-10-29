import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from research0 import fulltext
from research0.fulltext import FetchError, Response, TooLarge

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
HTML = b"<html><body>Please sign in</body></html>"


def unpaywall(**fields):
    return Response(200, "u", "application/json", json.dumps({"is_oa": True, "oa_status": "gold", **fields}).encode())


def oa(*pdf_urls, landing="https://publisher.example/landing"):
    locations = [{"url_for_pdf": u, "url": landing, "license": "cc-by"} for u in pdf_urls]
    return unpaywall(best_oa_location=locations[0] if locations else {"url": landing, "license": None}, oa_locations=locations)


NOT_IN_OPENALEX = Response(404, "o", "application/json", b"{}")


def openalex(*links, is_oa=True, landing="https://repo.example/landing"):
    """An OpenAlex work; each link is a (pdf_url, version) pair, the first one is also best_oa_location."""
    locations = [{"is_oa": True, "pdf_url": url, "landing_page_url": landing, "license": "cc-by-nc", "version": version} for url, version in links]
    return Response(200, "o", "application/json", json.dumps({"id": "W1", "open_access": {"is_oa": is_oa, "oa_status": "green" if is_oa else "closed"},
                                                              "best_oa_location": locations[0] if locations else ({"landing_page_url": landing, "license": None} if is_oa else None),
                                                              "locations": locations}).encode())


def scripted(table):
    """A fake HTTP getter: `table` maps a URL fragment to a Response or an exception to raise."""
    calls = []
    # OA landing pages in the legacy fixtures contain no download links unless scripted explicitly.
    pages = set()
    for answer in table.values():
        if isinstance(answer, Response) and answer.content_type == "application/json":
            try:
                record = json.loads(answer.body)
            except ValueError:
                continue
            for loc in [record.get("best_oa_location"), *(record.get("oa_locations") or []), *(record.get("locations") or [])]:
                for field in ("url", "url_for_landing_page", "landing_page_url"):
                    if (loc or {}).get(field):
                        pages.add(loc[field])

    def get(url, **kwargs):
        calls.append(url)
        for fragment, answer in table.items():
            if fragment in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        if url in pages:
            return Response(200, url, "text/html", HTML)
        raise AssertionError(f"unexpected request {url}")
    get.calls = calls
    return get


class FakeStore:
    location = "s3://fake/fulltext/"

    def __init__(self):
        self.objects = {}

    def put(self, key, data, sha256):
        self.objects[key] = data
        return f"s3://fake/{key}"

    def delete(self, uri):
        self.objects.pop(uri.removeprefix("s3://fake/"))


def test_pdf_check_accepts_pdfs_and_refuses_everything_else(monkeypatch):
    fulltext.check_pdf(PDF)
    fulltext.check_pdf(b"\xef\xbb\xbf  \n" + PDF)  # a few bytes before the header are legal
    for bad in (b"", HTML, b"PK\x03\x04zip"):
        with pytest.raises(ValueError):
            fulltext.check_pdf(bad)
    monkeypatch.setattr(fulltext, "MAX_PDF_BYTES", 10)
    with pytest.raises(ValueError, match="larger"):
        fulltext.check_pdf(PDF)


def test_candidates_are_direct_pdf_links_best_first_without_repeats():
    record = {"best_oa_location": {"url_for_pdf": "https://a/1.pdf"},
              "oa_locations": [{"url_for_pdf": "https://a/1.pdf"}, {"url_for_pdf": None, "url": "https://landing"},
                               {"url_for_pdf": "https://b/2.pdf"}, {"url_for_pdf": "https://c/3.pdf"}, {"url_for_pdf": "https://d/4.pdf"}]}
    assert fulltext.pdf_candidates(record) == ["https://a/1.pdf", "https://b/2.pdf", "https://c/3.pdf", "https://d/4.pdf"]
    assert fulltext.pdf_candidates({"best_oa_location": None, "oa_locations": None}) == []


def test_open_access_pdf_is_downloaded_and_the_source_is_recorded():
    get = scripted({"api.unpaywall.org": oa("https://pub.example/paper.pdf"), "paper.pdf": Response(200, "u", "application/pdf", PDF)})
    result = fulltext.acquire("10.1000/ok", "me@example.org", get)
    assert (result["outcome"], result["data"], result["pdf_url"], result["license"], result["oa_status"]) == ("downloaded", PDF, "https://pub.example/paper.pdf", "cc-by", "gold")
    assert "email=me%40example.org" in get.calls[0] and "10.1000/ok" in get.calls[0]


def test_a_login_page_is_never_kept_and_the_next_link_is_tried():
    get = scripted({"api.unpaywall.org": oa("https://one.example/a.pdf", "https://two.example/b.pdf"),
                    "one.example": Response(200, "u", "text/html", HTML), "two.example": Response(200, "u", "application/pdf", PDF)})
    assert fulltext.acquire("10.1/x", "me@example.org", get)["outcome"] == "downloaded"
    get = scripted({"api.unpaywall.org": oa("https://one.example/a.pdf"), "one.example": Response(200, "u", "text/html", HTML), "api.openalex.org": NOT_IN_OPENALEX})
    result = fulltext.acquire("10.1/x", "me@example.org", get)
    assert (result["outcome"], result["data"]) == ("not_a_pdf", None)


@pytest.mark.parametrize("doi,table,outcome", [
    (None, {}, "no_doi"),
    ("10.1/x", {"unpaywall": Response(404, "u", "", b"{}")}, "not_open_access"),
    ("10.1/x", {"unpaywall": unpaywall(is_oa=False, best_oa_location=None)}, "not_open_access"),
    ("10.1/x", {"unpaywall": oa(landing="https://publisher.example/landing")}, "no_pdf_link"),
    ("10.1/x", {"unpaywall": Response(429, "u", "", b"slow down")}, "lookup_failed"),
    ("10.1/x", {"unpaywall": FetchError("timed out")}, "lookup_failed"),
    ("10.1/x", {"unpaywall": oa("https://one.example/a.pdf"), "one.example": Response(403, "u", "", b"")}, "download_failed"),
    ("10.1/x", {"unpaywall": oa("https://one.example/a.pdf"), "one.example": FetchError("reset")}, "download_failed"),
    ("10.1/x", {"unpaywall": oa("https://one.example/a.pdf"), "one.example": TooLarge("big")}, "too_large"),
])
def test_every_failure_has_its_own_outcome_and_no_bytes(doi, table, outcome):
    result = fulltext.acquire(doi, "me@example.org", scripted({"api.openalex.org": NOT_IN_OPENALEX, **table}))  # OpenAlex knows nothing in these cases
    assert result["outcome"] == outcome and result["data"] is None


def test_no_pdf_link_keeps_the_open_access_page_for_the_person():
    result = fulltext.acquire("10.1/x", "me@example.org", scripted({"unpaywall": oa(landing="https://publisher.example/landing"), "api.openalex.org": NOT_IN_OPENALEX}))
    assert result["landing_url"] == "https://publisher.example/landing"


# --- OpenAlex: asked only when Unpaywall gave no file --------------------------------------------------------------------

def test_openalex_candidates_are_open_pdf_links_with_their_version_and_licence():
    record = {"best_oa_location": {"is_oa": True, "pdf_url": "https://a/1.pdf", "version": "acceptedVersion", "license": "cc-by"},
              "locations": [{"is_oa": True, "pdf_url": "https://a/1.pdf", "version": "acceptedVersion"}, {"is_oa": True, "pdf_url": None, "landing_page_url": "https://l"},
                            {"is_oa": False, "pdf_url": "https://closed/2.pdf"}, {"is_oa": True, "pdf_url": "https://b/3.pdf", "version": "submittedVersion"},
                            {"is_oa": True, "pdf_url": "https://c/4.pdf", "version": "weird"}, {"is_oa": True, "pdf_url": "https://d/5.pdf"}]}
    assert fulltext.openalex_candidates(record) == [("https://a/1.pdf", "accepted", "cc-by"), ("https://b/3.pdf", "submitted", None), ("https://c/4.pdf", None, None), ("https://d/5.pdf", None, None)]
    assert fulltext.openalex_candidates(record, skip=["https://a/1.pdf"])[0][0] == "https://b/3.pdf"
    assert fulltext.openalex_candidates({"best_oa_location": None, "locations": None}) == []


def test_a_repository_copy_found_by_openalex_is_downloaded_with_its_version():
    get = scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None),
                    "api.openalex.org": openalex(("https://repo.example/accepted.pdf", "acceptedVersion")), "accepted.pdf": Response(200, "u", "application/pdf", PDF)})
    result = fulltext.acquire("10.1/closed", "me@example.org", get, openalex_key="k3y")
    assert (result["outcome"], result["data"], result["found_by"], result["source"]) == ("downloaded", PDF, "openalex", fulltext.RESOLVER_VERSION)
    assert (result["version"], result["license"], result["pdf_url"], result["oa_status"]) == ("accepted", "cc-by-nc", "https://repo.example/accepted.pdf", "green")
    asked = get.calls[1]
    assert asked.startswith("https://api.openalex.org/works/doi:10.1/closed?") and "api_key=k3y" in asked and "select=" in asked


def test_the_key_is_sent_only_when_there_is_one():
    get = scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": NOT_IN_OPENALEX})
    fulltext.acquire("10.1/x", "me@example.org", get)
    assert "api_key" not in get.calls[1]


def test_unpaywall_success_never_reaches_openalex_and_keeps_its_version():
    location = {"url_for_pdf": "https://pub.example/p.pdf", "url": "https://pub.example/landing", "license": "cc-by", "version": "publishedVersion"}
    get = scripted({"api.unpaywall.org": unpaywall(best_oa_location=location, oa_locations=[location]), "p.pdf": Response(200, "u", "application/pdf", PDF)})
    result = fulltext.acquire("10.1/ok", "me@example.org", get)
    assert (result["outcome"], result["version"], result["found_by"], result["source"]) == ("downloaded", "published", "unpaywall", fulltext.RESOLVER_VERSION)
    assert not any("openalex" in call for call in get.calls)


def test_openalex_does_not_repeat_a_link_unpaywall_already_tried():
    get = scripted({"api.unpaywall.org": oa("https://one.example/a.pdf"), "one.example": Response(403, "u", "", b""),
                    "api.openalex.org": openalex(("https://one.example/a.pdf", "publishedVersion"))})
    result = fulltext.acquire("10.1/x", "me@example.org", get)
    assert result["outcome"] == "download_failed" and result["source"] == fulltext.RESOLVER_VERSION
    assert sum("one.example" in call for call in get.calls) == 1


def test_openalex_failing_is_not_a_final_answer_so_the_paper_is_asked_again():
    closed = unpaywall(is_oa=False, best_oa_location=None)
    for answer in (FetchError("timed out"), Response(500, "o", "", b"oops"), Response(200, "o", "application/json", b"not json")):
        result = fulltext.acquire("10.1/x", "me@example.org", scripted({"api.unpaywall.org": closed, "api.openalex.org": answer}))
        assert result["outcome"] == "lookup_failed" and result["outcome"] in fulltext.RETRY and "OpenAlex" in result["detail"] and result["data"] is None


def test_openalex_that_knows_nothing_leaves_the_unpaywall_answer_but_records_that_it_was_asked():
    result = fulltext.acquire("10.1/x", "me@example.org", scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": NOT_IN_OPENALEX}))
    assert (result["outcome"], result["source"]) == ("not_open_access", fulltext.RESOLVER_VERSION)
    closed = fulltext.acquire("10.1/x", "me@example.org", scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": openalex(is_oa=False)}))
    assert closed["outcome"] == "not_open_access"


def test_an_open_copy_without_a_pdf_link_sends_the_person_to_its_page():
    result = fulltext.acquire("10.1/x", "me@example.org", scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None),
                                                                      "api.openalex.org": openalex(landing="https://repo.example/handle/1")}))
    assert (result["outcome"], result["landing_url"], result["oa_status"], result["data"]) == ("no_pdf_link", "https://repo.example/handle/1", "green", None)


def test_a_login_page_from_openalex_is_never_kept():
    get = scripted({"api.unpaywall.org": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": openalex(("https://repo.example/x.pdf", "acceptedVersion")),
                    "x.pdf": Response(200, "u", "text/html", HTML)})
    result = fulltext.acquire("10.1/x", "me@example.org", get)
    assert (result["outcome"], result["data"]) == ("not_a_pdf", None)


def test_an_unpaywall_outage_remains_retryable_when_openalex_has_no_copy():
    get = scripted({"api.unpaywall.org": FetchError("down"), "api.openalex.org": NOT_IN_OPENALEX})
    assert fulltext.acquire("10.1/x", "me@example.org", get)["outcome"] == "lookup_failed" and len(get.calls) == 2


def test_spreadsheet_formulas_in_titles_are_defused():
    assert fulltext._cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert fulltext._cell(None) == "" and fulltext._cell(2019) == "2019" and fulltext._cell("Normal title") == "Normal title"


# --- The downloader must never be steered to this machine or its network -----------------------------------------------

class Local(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(PDF)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("url", ["http://127.0.0.1:{port}/a.pdf", "http://localhost:{port}/a.pdf", "http://[::1]:{port}/a.pdf",
                                 "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/a.pdf", "ftp://example.org/a.pdf", "file:///etc/passwd"])
def test_downloader_refuses_private_addresses_and_other_schemes(url):
    server = HTTPServer(("127.0.0.1", 0), Local)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(FetchError):
            fulltext.http_get(url.format(port=server.server_address[1]), max_bytes=1000, user_agent="test", timeout=3)
    finally:
        server.shutdown()


# --- Database ----------------------------------------------------------------------------------------------------------

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="Set local PostgreSQL environment to run integration test")
MIGRATIONS = ("003_phase1_catalog", "004_validation_decisions", "005_community_validation", "007_fulltext", "017_fulltext_version", "018_fulltext_blocked", "019_fulltext_exclusion")


def apply_schema(con):
    for name in MIGRATIONS:
        con.execute(open(f"migrations/{name}.sql", encoding="utf-8").read())


@pytest.fixture
def db():
    """A transaction on the real catalog that is rolled back: tests read real paper IDs and keep nothing."""
    from research0.cli import database
    con = database()
    apply_schema(con)
    run = con.execute("SELECT run_id FROM catalog_run ORDER BY canonical_count DESC, created_at DESC LIMIT 1").fetchone()
    ids = [r[0] for r in con.execute(
        # Papers never downloaded for real: the expected counts below must not depend on what the user has already fetched.
        "SELECT canonical_id FROM canonical_paper c WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0 "
        "AND NOT EXISTS (SELECT 1 FROM fulltext_attempt a WHERE a.canonical_id=c.canonical_id) "
        "AND NOT EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id=c.canonical_id) ORDER BY canonical_id LIMIT 30", (run[0],))] if run else []
    if len(ids) < 30:
        con.close()
        pytest.skip("no catalog run in the database")
    yield con, run[0], ids
    con.rollback()
    con.close()


@needs_db
def test_process_saves_files_and_every_outcome_and_lists_what_is_left_for_hand(db):
    con, run_id, ids = db
    before = fulltext.totals(con, run_id)
    get = scripted({"v2/10.1/ok?": oa("https://pub.example/ok.pdf"), "pub.example/ok.pdf": Response(200, "u", "application/pdf", PDF),
                    "v2/10.1/closed?": unpaywall(is_oa=False, best_oa_location=None), "v2/10.1/nolink?": oa(landing="https://publisher.example/p"),
                    "v2/10.1/flaky?": Response(503, "u", "", b""), "api.openalex.org": NOT_IN_OPENALEX})
    papers = [(ids[0], "10.1/ok"), (ids[1], "10.1/closed"), (ids[2], None), (ids[3], "10.1/nolink"), (ids[4], "10.1/flaky")]
    store = FakeStore()
    counts = fulltext.process(con, run_id, papers, "me@example.org", get, store, workers=2)
    assert counts == {"downloaded": 1, "not_open_access": 1, "no_doi": 1, "no_pdf_link": 1, "lookup_failed": 1}
    (key, data), = store.objects.items()
    assert key.startswith(f"fulltext/{ids[0]}/") and key.endswith(".pdf") and data == PDF
    row = con.execute("SELECT origin,s3_uri,sha256,size_bytes,source_url,acquired_by,validation_status FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()
    assert row[0] == "open_access_download" and row[1] == f"s3://fake/{key}" and row[3] == len(PDF)
    assert (row[4], row[5], row[6]) == ("https://pub.example/ok.pdf", "unpaywall", "unvalidated") and key.endswith(row[2] + ".pdf")

    after = fulltext.totals(con, run_id)
    assert after["with_open_access_file"] - before["with_open_access_file"] == 1
    assert after["to_get_by_hand"] - before["to_get_by_hand"] == 3      # closed, no DOI, no direct link
    assert after["to_retry"] - before["to_retry"] == 1                  # a lookup failure is not a final answer
    assert after["not_attempted"] - before["not_attempted"] == -5

    wanted = {str(ids[1]): "not_open_access", str(ids[2]): "no_doi", str(ids[3]): "no_pdf_link"}
    listed = {p["id"]: p for p in fulltext.missing(con, run_id, 0, 100000)}
    assert {k: listed[k]["reason_code"] for k in wanted} == wanted
    assert str(ids[0]) not in listed and str(ids[4]) not in listed
    assert listed[str(ids[3])]["open_access_page"] == "https://publisher.example/p"
    assert listed[str(ids[3])]["suggested_filename"] == f"{ids[3]}.pdf"
    assert str(ids[3]) in fulltext.missing_csv(con, run_id)


@needs_db
def test_excluded_papers_leave_the_lists_and_their_pdfs_can_be_purged(db):
    con, run_id, ids = db
    get = scripted({"v2/10.1/ok?": oa("https://pub.example/ok.pdf"), "pub.example/ok.pdf": Response(200, "u", "application/pdf", PDF),
                    "v2/10.1/closed?": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": NOT_IN_OPENALEX})
    store = FakeStore()
    fulltext.process(con, run_id, [(ids[0], "10.1/ok"), (ids[1], "10.1/closed")], "me@example.org", get, store, workers=1)
    before = fulltext.totals(con, run_id)
    assert str(ids[1]) in {p["id"] for p in fulltext.missing(con, run_id, 0, 100000)}

    with pytest.raises(ValueError):
        fulltext.exclude_papers(con, run_id, [ids[0]], "  ", "me")
    fulltext.exclude_papers(con, run_id, [ids[0], ids[1], ids[2]], "not assigned to CTO or TIM by the Batch", "me")
    after = fulltext.totals(con, run_id)
    assert before["eligible"] - after["eligible"] == 3 and before["with_file"] - after["with_file"] == 1
    assert before["to_get_by_hand"] - after["to_get_by_hand"] == 1 and before["not_attempted"] - after["not_attempted"] == 1
    assert str(ids[1]) not in {p["id"] for p in fulltext.missing(con, run_id, 0, 100000)}
    assert str(ids[1]) not in fulltext.missing_csv(con, run_id)
    assert ids[2] not in {cid for cid, _ in fulltext.select_papers(con, run_id, "eligible", 40)}
    assert con.execute("SELECT count(*) FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 1  # excluding deletes nothing

    out = fulltext.purge_files(con, store, run_id, [ids[0], ids[1], ids[2]])
    assert [f["canonical_id"] for f in out["files"]] == [str(ids[0])] and out["attempts_deleted"] == 1 and out["storage_failures"] == []
    assert store.objects == {}
    assert con.execute("SELECT count(*) FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM fulltext_attempt WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM fulltext_attempt WHERE canonical_id=%s", (ids[1],)).fetchone()[0] == 1  # other attempts stay as history

    fulltext.exclude_papers(con, run_id, [ids[1]], "lifted", "me", excluded=False)  # the latest row decides
    assert fulltext.totals(con, run_id)["eligible"] == after["eligible"] + 1


@needs_db
def test_a_new_run_skips_finished_papers_and_retries_failures(db):
    con, run_id, ids = db
    get = scripted({"v2/10.1/ok?": oa("https://pub.example/ok.pdf"), "pub.example": Response(200, "u", "application/pdf", PDF), "v2/10.1/flaky?": Response(503, "u", "", b""), "api.openalex.org": NOT_IN_OPENALEX})
    fulltext.process(con, run_id, [(ids[0], "10.1/ok"), (ids[1], "10.1/flaky"), (ids[2], None)], "me@example.org", get, FakeStore(), workers=1)
    chosen = {cid for cid, _ in fulltext.select_papers(con, run_id, "eligible", 40)}
    assert ids[1] in chosen and ids[0] not in chosen and ids[2] not in chosen
    with pytest.raises(ValueError):
        fulltext.select_papers(con, run_id, "everything")


@needs_db
def test_openalex_file_is_stored_with_its_version_and_source_and_old_unpaywall_answers_are_asked_again(db):
    con, run_id, ids = db
    closed = unpaywall(is_oa=False, best_oa_location=None)
    # A paper that an earlier run (Unpaywall only) closed as not open access.
    con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,doi,source,outcome) VALUES (%s,%s,'10.1/old','unpaywall','not_open_access')", (run_id, ids[0]))
    assert ids[0] in {cid for cid, _ in fulltext.select_papers(con, run_id, "eligible", 100000)}
    get = scripted({"v2/10.1/old?": closed, "works/doi:10.1/old?": openalex(("https://repo.example/old.pdf", "acceptedVersion")),
                    "repo.example/old.pdf": Response(200, "u", "application/pdf", PDF)})
    store = FakeStore()
    assert fulltext.process(con, run_id, [(ids[0], "10.1/old")], "me@example.org", get, store, workers=1) == {"downloaded": 1}
    row = con.execute("SELECT origin,acquired_by,version_label,source_url FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()
    assert row == ("open_access_download", "openalex", "accepted", "https://repo.example/old.pdf")
    # (by outcome, not by time: inside one test transaction now() is the same for the seeded attempt and the new one)
    assert con.execute("SELECT source,outcome,license FROM fulltext_attempt WHERE canonical_id=%s AND outcome='downloaded'", (ids[0],)).fetchall() == \
        [(fulltext.RESOLVER_VERSION, "downloaded", "cc-by-nc")]
    # A final answer from the current resolver is skipped; a service outage is retried.
    con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,doi,source,outcome) VALUES (%s,%s,'10.1/n',%s,'not_open_access')", (run_id, ids[1], fulltext.RESOLVER_VERSION))
    con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,doi,source,outcome) VALUES (%s,%s,'10.1/o','unpaywall+openalex','lookup_failed')", (run_id, ids[2]))
    chosen = {cid for cid, _ in fulltext.select_papers(con, run_id, "eligible", 100000)}
    assert ids[1] not in chosen and ids[2] in chosen and ids[0] not in chosen
    fulltext.save_upload(con, store, run_id, ids[3], PDF, "x.pdf", "Anna")
    assert con.execute("SELECT version_label FROM fulltext_file WHERE canonical_id=%s", (ids[3],)).fetchone()[0] is None  # a hand upload has no known version
    with pytest.raises(Exception):
        con.execute("UPDATE fulltext_file SET version_label='draft' WHERE canonical_id=%s", (ids[0],))


@needs_db
def test_attempts_are_appended_never_overwritten(db):
    con, run_id, ids = db
    get = scripted({"v2/10.1/closed?": unpaywall(is_oa=False, best_oa_location=None), "api.openalex.org": NOT_IN_OPENALEX})
    fulltext.process(con, run_id, [(ids[0], "10.1/closed")], "me@example.org", get, FakeStore(), workers=1)
    fulltext.process(con, run_id, [(ids[0], "10.1/closed")], "me@example.org", get, FakeStore(), workers=1)
    assert con.execute("SELECT count(*) FROM fulltext_attempt WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 2


@needs_db
def test_the_run_stops_when_unpaywall_keeps_failing(db):
    con, run_id, ids = db
    get = scripted({"unpaywall": FetchError("refused"), "api.openalex.org": NOT_IN_OPENALEX})
    with pytest.raises(RuntimeError, match="in a row"):
        fulltext.process(con, run_id, [(cid, f"10.1/{n}") for n, cid in enumerate(ids)], "me@example.org", get, FakeStore(), workers=1)


@needs_db
def test_a_storage_failure_stops_the_run_without_recording_a_false_outcome(db):
    con, run_id, ids = db

    class Broken(FakeStore):
        def put(self, *args):
            raise OSError("bucket unreachable")
    get = scripted({"api.unpaywall.org": oa("https://pub.example/ok.pdf"), "pub.example": Response(200, "u", "application/pdf", PDF)})
    with pytest.raises(OSError):
        fulltext.process(con, run_id, [(ids[0], "10.1/ok")], "me@example.org", get, Broken(), workers=1)
    assert con.execute("SELECT count(*) FROM fulltext_attempt WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 0


@needs_db
def test_hand_uploaded_pdf_is_stored_once_and_attributed(db):
    con, run_id, ids = db
    store = FakeStore()
    first = fulltext.save_upload(con, store, run_id, ids[0], PDF, "../../Elsevier download (1).pdf", "  Anna  ")
    again = fulltext.save_upload(con, store, run_id, ids[0], PDF, "other.pdf", "Anna")
    assert not first["already_stored"] and again["already_stored"] and first["file_id"] == again["file_id"]
    assert len(store.objects) == 1 and next(iter(store.objects)) == f"fulltext/{ids[0]}/{first['sha256']}.pdf"
    row = con.execute("SELECT origin,acquired_by,original_filename FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()
    assert row == ("manual_upload", "Anna", "Elsevier download (1).pdf")
    # The same bytes may legitimately belong to a different paper; a changed file is a new version of the same paper.
    other = fulltext.save_upload(con, store, run_id, ids[1], PDF, "x.pdf", "Anna")
    changed = fulltext.save_upload(con, store, run_id, ids[0], PDF + b"\n% revised", "x.pdf", "Anna")
    assert not other["already_stored"] and not changed["already_stored"]
    assert con.execute("SELECT count(*) FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 2


@needs_db
def test_upload_refuses_non_pdf_unknown_paper_and_missing_name(db):
    con, run_id, ids = db
    store = FakeStore()
    for args, message in (((run_id, ids[0], HTML, "a.pdf", "Anna"), "not a PDF"), ((run_id, uuid.uuid4(), PDF, "a.pdf", "Anna"), "Unknown paper"),
                          ((run_id, ids[0], PDF, "a.pdf", "  "), "name"), ((run_id, ids[0], b"", "a.pdf", "Anna"), "empty")):
        with pytest.raises(ValueError, match=message):
            fulltext.save_upload(con, store, *args)
    assert not store.objects


@needs_db
def test_folder_import_matches_files_by_paper_id_and_reports_the_rest(db, tmp_path):
    con, run_id, ids = db
    (tmp_path / f"{ids[0]}.pdf").write_bytes(PDF)
    (tmp_path / f"{ids[1]}.pdf").write_bytes(HTML)
    (tmp_path / "Smith 2019.pdf").write_bytes(PDF)
    store = FakeStore()
    report = fulltext.import_folder(con, store, run_id, tmp_path, "Anna")
    assert (report["imported"], report["already_stored"]) == (1, 0)
    assert {r["file"] for r in report["rejected"]} == {f"{ids[1]}.pdf", "Smith 2019.pdf"}
    assert fulltext.import_folder(con, store, run_id, tmp_path, "Anna")["already_stored"] == 1


def docx_bytes(*paragraphs):
    import io
    import docx
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64


def test_document_check_recognises_pdf_docx_and_doc_by_content_and_refuses_the_rest():
    import io
    import zipfile
    assert fulltext.check_document(PDF) == "pdf"
    assert fulltext.check_document(docx_bytes("Hello")) == "docx"
    assert fulltext.check_document(OLE2) == "doc"
    for bad in (b"", HTML, b"PK\x03\x04not really a zip"):
        with pytest.raises(ValueError):
            fulltext.check_document(bad)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "a zip that is not a Word document")
    with pytest.raises(ValueError, match="not a PDF or a Word"):
        fulltext.check_document(buffer.getvalue())


@needs_db
def test_word_files_are_stored_only_when_asked_with_their_own_extension_and_type(db):
    con, run_id, ids = db
    store = FakeStore()
    data = docx_bytes("Intro", "A paragraph of the paper.")
    with pytest.raises(ValueError, match="not a PDF"):
        fulltext.save_upload(con, store, run_id, ids[0], data, "p.docx", "Anna")  # the dashboard path stays PDF-only
    out = fulltext.save_upload(con, store, run_id, ids[0], data, "p.docx", "Anna", source_url="https://repo.example/p", version="accepted", allow_word=True)
    assert next(iter(store.objects)) == f"fulltext/{ids[0]}/{out['sha256']}.docx"
    row = con.execute("SELECT origin,source_url,version_label,acquired_by FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()
    assert row == ("manual_upload", "https://repo.example/p", "accepted", "Anna")
    assert fulltext.CONTENT_TYPES["docx"].endswith("wordprocessingml.document")


def test_index_versions_map_to_the_three_accepted_values_and_nothing_else():
    from research0.fulltext_collection import version_label
    for text, label in (("Accepted author manuscript (repository declaration)", "accepted"), ("acceptedVersion", "accepted"), ("publishedVersion", "published"),
                        ("Published version (metadata declaration)", "published"), ("Submitted manuscript / preprint (metadata declaration)", "submitted"),
                        ("submittedVersion", "submitted"), ("Preprint: arXiv 2404.00029v2", "submitted"), ("Unverified version", None),
                        ("Working paper (2024); earlier than journal publication", None), ("repository / publisher instance; OPEN", None), ("", None), (None, None)):
        assert version_label(text) == label, text


@needs_db
def test_collection_import_imports_holds_back_and_rejects_and_is_repeatable(db, tmp_path):
    import csv
    import hashlib
    from research0 import fulltext_collection as fc
    con, run_id, ids = db
    store = FakeStore()
    folder = tmp_path / "x"
    contents = {"new": PDF + b"1", "word": docx_bytes("Body text"), "same": PDF + b"2", "other": PDF + b"3", "twin_a": PDF + b"4", "twin_b": PDF + b"5", "bad": PDF + b"6"}
    fulltext.save_upload(con, store, run_id, ids[2], contents["same"], "s.pdf", "Anna")  # identical bytes already stored
    fulltext.save_upload(con, store, run_id, ids[3], PDF + b"older", "o.pdf", "Anna")    # a different file already stored
    rows = [("new", ids[0], "PDF", "Reuse_license_reported", "publishedVersion"), ("word", ids[1], "DOCX", "Sharing_permission_review", "Unverified version"),
            ("same", ids[2], "PDF", "Reuse_license_reported", ""), ("other", ids[3], "PDF", "Reuse_license_reported", ""),
            ("twin_a", ids[4], "PDF", "Reuse_license_reported", ""), ("twin_b", ids[4], "PDF", "Reuse_license_reported", ""),
            ("bad", ids[5], "PDF", "Reuse_license_reported", "")]
    index = tmp_path / "index.csv"
    with open(index, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "path", "format", "sha256", "url", "version"])
        writer.writeheader()
        for name, cid, fmt, bucket, version in rows:
            relative = f"Papers/{bucket}/2020/{name}.{fmt.lower()}"
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[name])
            writer.writerow({"record_id": cid, "path": relative, "format": fmt, "sha256": hashlib.sha256(contents[name]).hexdigest() if name != "bad" else "0" * 64,
                             "url": f"https://src.example/{name}", "version": version})
    dry = fc.import_collection(con, None, run_id, folder, index, "Anna", execute=False)
    assert dry["by_action"] == {"import": 2, "already_stored": 1, "held": 3, "rejected": 1} and dry["mode"] == "dry-run"
    assert con.execute("SELECT count(*) FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone()[0] == 0  # a dry run writes nothing
    done = fc.import_collection(con, store, run_id, folder, index, "Anna", execute=True)
    assert done["by_action"] == {"imported": 2, "already_stored": 1, "held": 3, "rejected": 1}
    reasons = {e["record_id"]: e["reason"] for e in done["entries"] if e["reason"]}
    assert "different file" in reasons[str(ids[3])] and "several" in reasons[str(ids[4])] and "SHA-256" in reasons[str(ids[5])]
    assert con.execute("SELECT origin,acquired_by,version_label,source_url FROM fulltext_file WHERE canonical_id=%s", (ids[0],)).fetchone() == (
        "manual_upload", "Anna (Drive OrgSci/Reuse_license_reported)", "published", "https://src.example/new")
    assert con.execute("SELECT s3_uri FROM fulltext_file WHERE canonical_id=%s", (ids[1],)).fetchone()[0].endswith(".docx")
    assert con.execute("SELECT count(*) FROM fulltext_file WHERE canonical_id=ANY(%s)", ([ids[3], ids[4], ids[5]],)).fetchone()[0] == 1  # only the older file of ids[3]
    again = fc.import_collection(con, store, run_id, folder, index, "Anna", execute=True)
    assert again["by_action"] == {"already_stored": 3, "held": 3, "rejected": 1}


# --- The dashboard server ----------------------------------------------------------------------------------------------

def request(port, method, path, body=None, content_type="application/json", headers=None):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=data, headers={"Content-Type": content_type, **(headers or {})})
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def as_json(answer):
    return answer[0], json.loads(answer[2])


@pytest.fixture
def server():
    from research0.cli import database
    from research0.dashboard_server import serve
    with database() as con:
        apply_schema(con)  # committed on exit, like the other server tests
        run = con.execute("SELECT run_id FROM catalog_run ORDER BY canonical_count DESC, created_at DESC LIMIT 1").fetchone()
        paper = con.execute("SELECT canonical_id FROM canonical_paper WHERE run_id=%s ORDER BY canonical_id LIMIT 1", (run[0],)).fetchone() if run else None
    if not paper:
        pytest.skip("no catalog run in the database")
    httpd, run_id = serve(str(run[0]), 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], str(run_id), paper[0]
    httpd.shutdown()


@needs_db
def test_status_missing_list_and_csv_are_served(server):
    port, run_id, _ = server
    status, body = as_json(request(port, "GET", "/api/fulltext/status"))
    assert status == 200 and body["job"]["state"] == "idle" and set(body["scopes"]) == {"sample", "audit", "candidates", "assigned", "eligible"}
    assert body["totals"]["eligible"] > 0 and body["storage"].endswith("/fulltext/")
    status, body = as_json(request(port, "GET", "/api/fulltext/missing?offset=0&limit=5"))
    assert status == 200 and isinstance(body["papers"], list)
    assert request(port, "GET", "/api/fulltext/missing?offset=-1")[0] == 400
    assert request(port, "GET", "/api/fulltext/missing?limit=abc")[0] == 400
    status, headers, text = request(port, "GET", "/api/fulltext/missing.csv")
    assert status == 200 and headers["Content-Type"].startswith("text/csv") and "attachment" in headers["Content-Disposition"]
    assert text.decode("utf-8-sig").startswith("id,suggested_filename,doi,doi_url,title")


@needs_db
def test_download_start_validates_before_anything_runs(server, monkeypatch):
    port, run_id, _ = server
    monkeypatch.setattr("research0.dashboard_server.FULLTEXT_DOWNLOAD_ADMIN_MODE", True)    # the download is locked by default (tests/test_dashboard_deploy.py)
    start = lambda **kw: as_json(request(port, "POST", "/api/fulltext/start", {"run_id": run_id, "scope": "sample", **kw}))
    monkeypatch.setenv("UNPAYWALL_EMAIL", "")
    status, body = start()
    assert status == 400 and "UNPAYWALL_EMAIL" in body["error"]
    monkeypatch.setenv("UNPAYWALL_EMAIL", "me@example.org")
    assert start(scope="everything")[0] == 400
    assert start(limit=0)[0] == 400 and start(limit="5")[0] == 400
    assert as_json(request(port, "POST", "/api/fulltext/start", {"run_id": str(uuid.uuid4()), "scope": "sample"}))[0] == 409
    assert request(port, "POST", "/api/fulltext/start", {}, headers={"Origin": "http://evil.example"})[0] == 403
    assert as_json(request(port, "GET", "/api/fulltext/status"))[1]["job"]["state"] == "idle"


@needs_db
def test_upload_endpoint_refuses_bad_requests(server, monkeypatch):
    port, run_id, paper = server
    up = lambda body=PDF, ctype="application/pdf", **kw: as_json(request(
        port, "POST", "/api/fulltext/upload?" + urllib.parse.urlencode({"run_id": run_id, "paper_id": paper, "uploader": "Anna", "filename": "a.pdf", **kw}),
        body, ctype))
    assert up(ctype="application/json")[0] == 415
    assert up(body=b"")[0] == 413
    monkeypatch.setattr(fulltext, "MAX_PDF_BYTES", 10)
    assert up()[0] == 413
    monkeypatch.undo()
    assert up(run_id=str(uuid.uuid4()))[0] == 409
    assert up(body=HTML)[0] == 400
    assert up(uploader=" ")[0] == 400
    assert up(paper_id="not-a-uuid")[0] == 400
    assert up(paper_id=str(uuid.uuid4()))[0] == 400
    assert request(port, "POST", "/api/fulltext/upload", PDF, "application/pdf", {"Origin": "http://evil.example"})[0] == 403
    assert request(port, "POST", "/api/fulltext/upload", PDF, "application/pdf", {"Host": "evil.example"})[0] == 403


@needs_db
@pytest.mark.skipif(not os.getenv("S3_BUCKET"), reason="Set the local S3 environment to run the storage test")
def test_uploaded_pdf_reaches_s3_and_the_database(server):
    from research0.cli import database
    port, run_id, paper = server
    marker = b"\n% pytest " + uuid.uuid4().hex.encode()
    query = urllib.parse.urlencode({"run_id": run_id, "paper_id": paper, "uploader": "pytest", "filename": "t.pdf"})
    status, body = as_json(request(port, "POST", "/api/fulltext/upload?" + query, PDF + marker, "application/pdf"))
    assert status == 200 and not body["already_stored"] and body["s3_uri"].endswith(f"/fulltext/{paper}/{body['sha256']}.pdf")
    store = fulltext.S3Store()
    key = body["s3_uri"].split(f"s3://{store.bucket}/", 1)[1]
    try:
        assert store.client.get_object(Bucket=store.bucket, Key=key)["Body"].read() == PDF + marker
        assert as_json(request(port, "POST", "/api/fulltext/upload?" + query, PDF + marker, "application/pdf"))[1]["already_stored"]
    finally:
        with database() as con:
            con.execute("DELETE FROM fulltext_file WHERE canonical_id=%s AND acquired_by='pytest'", (paper,))  # remove test rows only
        store.client.delete_object(Bucket=store.bucket, Key=key)
