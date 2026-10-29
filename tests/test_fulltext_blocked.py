"""Bot-protection walls (Cloudflare, AWS WAF, "Human Verification") are told apart from failures; figshare is read through its public API."""
import json

import pytest

from research0 import fulltext as ft
from test_fulltext import PDF, Response, FetchError, oa, scripted, NOT_IN_OPENALEX, needs_db, apply_schema

CLOUDFLARE = b'<!DOCTYPE html><html lang="en-US"><head><title>Just a moment...</title></head></html>'
EMAIL = "me@example.org"


def wall(url="https://pure.example/files/1/p.pdf"):
    return Response(403, url, "text/html", CLOUDFLARE, None, True)


@pytest.mark.parametrize("status,headers,body,expected", [
    (403, {}, CLOUDFLARE, True),
    (405, {}, b"<title>Human Verification</title>", True),
    (403, {}, b"<title>Site Protection: blocked</title>", True),
    (200, {"cf-mitigated": "challenge"}, b"", True),
    (202, {"x-amzn-waf-action": "challenge"}, b"", True),
    (403, {}, b"Forbidden", False),                 # a plain refusal says nothing about bots
    (404, {}, b"Not found", False),
    (200, {}, CLOUDFLARE, False),                   # a normal page that merely quotes the text is not a refusal
])
def test_challenge_detection(status, headers, body, expected):
    assert ft.is_challenge(status, headers, body) is expected


def test_wall_is_not_retried_and_is_its_own_outcome():
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if "unpaywall" in url:
            return oa("https://pure.example/files/1/p.pdf")
        return wall(url) if "pure.example" in url else NOT_IN_OPENALEX
    result = ft.acquire("10.1/x", EMAIL, get)
    assert (result["outcome"], result["data"]) == ("blocked_by_site", None)
    assert "pure.example" in result["detail"] and result["http_status"] == 403
    assert sum("p.pdf" in c for c in calls) == 1


def test_http_get_does_not_repeat_a_wall(monkeypatch):
    attempts = []
    monkeypatch.setattr(ft, "_http_get_once", lambda *a, **k: attempts.append(1) or Response(503, "u", "", CLOUDFLARE, None, True))
    monkeypatch.setattr(ft.time, "sleep", lambda s: pytest.fail("must not wait"))
    assert ft.http_get("https://x.example/a.pdf", max_bytes=10, user_agent="t").challenge and len(attempts) == 1


def test_plain_403_is_still_a_download_failure():
    get = scripted({"unpaywall": oa("https://one.example/a.pdf"), "one.example": Response(403, "u", "", b"Forbidden"), "openalex": NOT_IN_OPENALEX})
    assert ft.acquire("10.1/x", EMAIL, get)["outcome"] == "download_failed"


def test_another_copy_wins_over_a_wall():
    get = scripted({"unpaywall": oa("https://pure.example/a.pdf", "https://repo.example/b.pdf"), "pure.example": wall(),
                    "repo.example": Response(200, "u", "application/pdf", PDF)})
    assert ft.acquire("10.1/x", EMAIL, get)["outcome"] == "downloaded"


def test_a_later_plain_failure_does_not_hide_the_wall():
    get = scripted({"unpaywall": oa("https://pure.example/a.pdf", "https://repo.example/b.pdf"), "pure.example": wall(),
                    "repo.example": Response(500, "u", "", b"oops"), "openalex": NOT_IN_OPENALEX})
    assert ft.acquire("10.1/x", EMAIL, get)["outcome"] == "blocked_by_site"


def test_wall_is_listed_for_hand_download_with_its_reason():
    assert "blocked_by_site" in ft.REASONS and "blocked_by_site" not in ft.RETRY


# --- figshare ---------------------------------------------------------------------------------------------------------

PAGE = "https://tandf.figshare.com/articles/journal_contribution/Some_title/22210657"


def article(**fields):
    files = [{"name": "Paper.pdf", "download_url": "https://ndownloader.figshare.com/files/1", "is_link_only": False},
             {"name": "Supplementary data.pdf", "download_url": "https://ndownloader.figshare.com/files/2"},
             {"name": "data.csv", "download_url": "https://ndownloader.figshare.com/files/3"}]
    return Response(200, "a", "application/json", json.dumps({"is_embargoed": False, "is_confidential": False, "files": files, **fields}).encode())


@pytest.mark.parametrize("url,expected", [
    (PAGE, "22210657"),
    (PAGE + "/2", "22210657"),                      # a version suffix is not the article
    ("https://figshare.com/articles/dataset/T/9502481", "9502481"),
    ("https://api.figshare.com/v2/articles/9502481", None),
    ("https://figshare.com/collections/x/123", None),
    ("https://notfigshare.com/articles/a/b/123", None),
])
def test_figshare_article_number(url, expected):
    assert ft._figshare_article(url) == expected


def test_figshare_page_is_replaced_by_its_api_and_only_the_paper_is_taken():
    get = scripted({"unpaywall": oa(PAGE), "api.figshare.com/v2/articles/22210657": article(),
                    "ndownloader.figshare.com/files/1": Response(200, "u", "application/pdf", PDF)})
    result = ft.acquire("10.1/x", EMAIL, get)
    assert (result["outcome"], result["pdf_url"]) == ("downloaded", "https://ndownloader.figshare.com/files/1")
    assert not any("files/2" in c or "files/3" in c for c in get.calls) and PAGE not in get.calls


@pytest.mark.parametrize("fields,reason", [({"is_embargoed": True}, "embargoed"), ({"is_confidential": True}, "embargoed or confidential"),
                                           ({"files": []}, "no PDF file")])
def test_figshare_embargo_and_missing_files_are_not_forced(fields, reason):
    get = scripted({"unpaywall": oa(PAGE), "api.figshare.com": article(**fields), "openalex": NOT_IN_OPENALEX})
    result = ft.acquire("10.1/x", EMAIL, get)
    assert (result["outcome"], result["data"]) == ("no_pdf_link", None) and reason in result["detail"]
    assert PAGE not in get.calls and not any("ndownloader" in c for c in get.calls)


def test_figshare_api_outage_falls_back_to_the_page():
    get = scripted({"unpaywall": oa(PAGE), "api.figshare.com": FetchError("down"),
                    "tandf.figshare.com": Response(202, PAGE, "text/html", b"", None, True), "openalex": NOT_IN_OPENALEX})
    assert ft.acquire("10.1/x", EMAIL, get)["outcome"] == "blocked_by_site"


# --- Database ---------------------------------------------------------------------------------------------------------

@needs_db
def test_migration_allows_the_new_outcome_and_is_repeatable():
    import psycopg
    from research0.cli import database
    con = database()
    try:
        apply_schema(con)
        for _ in range(2):
            con.execute(open("migrations/018_fulltext_blocked.sql", encoding="utf-8").read())
        definition = con.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname='fulltext_attempt_outcome_check'").fetchone()[0]
        assert "blocked_by_site" in definition and "download_failed" in definition and "lookup_failed" in definition
        with pytest.raises(psycopg.errors.CheckViolation):
            con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,source,outcome) "
                        "SELECT run_id,canonical_id,'t','nonsense' FROM canonical_paper LIMIT 1")
    finally:
        con.rollback()
        con.close()
