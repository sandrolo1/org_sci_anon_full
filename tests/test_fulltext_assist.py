"""Hand-assisted full text: the list for the person and the folder importer. Matching needs strong evidence or the file stays untouched."""
import os
import uuid

import pytest

from research0 import fulltext_assist as fa
from research0.mock_world import make_pdf, MemoryStore

TITLE = "Digital platforms and the reconfiguration of organizational boundaries"


def paper(doi, title=TITLE, pid=None, reason="blocked_by_site"):
    return {"id": str(pid or uuid.uuid4()), "doi": doi, "doi_key": doi.lower() if doi else None, "title": title, "title_key": fa._norm(title),
            "reason_code": reason, "reason": "r", "year": 2021, "source": "Journal", "doi_url": f"https://doi.org/{doi}" if doi else None,
            "open_access_page": "https://repo.example/item/1"}


def test_doi_on_the_first_page_identifies_the_paper():
    a, b = paper("10.1111/isj.12057"), paper("10.1016/j.respol.2019.103916", "Another study of something entirely different from it")
    found, how = fa.identify("download (3).pdf", make_pdf([["Information Systems Journal", "https://doi.org/10.1111/ISJ.12057."]]), [a, b])
    assert found is a and "first page" in how


def test_title_alone_identifies_the_paper_when_it_is_long_and_unique():
    a, b = paper("10.1/a"), paper("10.1/b", "Quantum leaps or baby steps expertise distance and construal level")
    found, how = fa.identify("x.pdf", make_pdf([[TITLE.upper()], ["Abstract text"]]), [a, b])
    assert found is a and how == "full title on the first pages"


def test_file_name_that_is_the_paper_id_wins_without_reading_the_file():
    a = paper("10.1/a")
    found, how = fa.identify(f"{a['id']}.pdf", b"%PDF-1.4 not readable", [a, paper("10.1/b")])
    assert found is a and "paper ID" in how


def test_doi_in_the_file_name_with_underscore():
    a = paper("10.1111/isj.12057")
    found, how = fa.identify("10.1111_isj.12057.pdf", make_pdf([["no identifiers here at all"]]), [a])
    assert found is a and "file name" in how


@pytest.mark.parametrize("pages,why", [
    ([["A completely unrelated paper about birds"]], "no DOI or full title"),
    ([["Short title"], ["10.1000/other"]], "no DOI or full title"),         # a DOI that is not on the list proves nothing
    ([["see 10.1000/a and also 10.1000/b"]], "2 missing papers' DOIs"),     # a reference list is not evidence
])
def test_weak_or_ambiguous_evidence_is_never_guessed(pages, why):
    a, b = paper("10.1000/a", "Short title"), paper("10.1000/b", "Other")
    found, reason = fa.identify("f.pdf", make_pdf(pages), [a, b])
    assert found is None and why in reason


def test_short_titles_are_not_enough():
    a = paper("10.1/a", "Digital platforms")
    assert fa.identify("f.pdf", make_pdf([["Digital platforms are discussed here"]]), [a])[0] is None


def test_doi_must_not_match_a_longer_doi():
    a = paper("10.1111/isj.1205")
    assert fa.identify("f.pdf", make_pdf([["10.1111/isj.12057"]]), [a])[0] is None


def test_unreadable_pdf_is_just_unmatched():
    assert fa.identify("f.pdf", b"%PDF-1.4 garbage", [paper("10.1/a")])[0] is None


def test_page_escapes_titles_and_only_links_web_urls():
    evil = paper("10.1/a", '<script>alert(1)</script> "x"')
    evil["open_access_page"] = "javascript:alert(1)"
    page = fa.assist_page([evil, paper("10.1/b")], "C:/pdfs")
    assert "<script>alert" not in page and "javascript:" not in page and "&lt;script&gt;" in page
    assert page.count('href="https://doi.org/10.1/') == 2 and 'rel="noopener noreferrer"' in page


# --- Folder and database ----------------------------------------------------------------------------------------------

def _age(path, seconds=60):
    stamp = os.path.getmtime(path) - seconds
    os.utime(path, (stamp, stamp))


def test_scan_folder_imports_matches_moves_them_and_leaves_the_rest(con, world, tmp_path):
    from research0 import fulltext
    ids = [r[0] for r in con.execute("SELECT canonical_id FROM canonical_paper c WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0 "
                                     "AND doi IS NOT NULL AND NOT EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id=c.canonical_id) ORDER BY canonical_id LIMIT 2",
                                     (world.run_id,))]
    assert len(ids) == 2
    for cid in ids:
        con.execute("INSERT INTO fulltext_attempt(run_id,canonical_id,source,outcome,detail) VALUES (%s,%s,%s,'blocked_by_site','Bot protection')",
                    (world.run_id, cid, fulltext.RESOLVER_VERSION))
    papers = fa.papers_to_fetch(con, world.run_id)
    mine = [p for p in papers if p["id"] in {str(i) for i in ids}]
    assert len(mine) == 2 and papers[0]["reason_code"] == "blocked_by_site"
    (tmp_path / "found.pdf").write_bytes(make_pdf([["Article", f"https://doi.org/{mine[0]['doi']}"]]))
    (tmp_path / "unknown.pdf").write_bytes(make_pdf([["nothing identifies this file"]]))
    (tmp_path / "notes.txt").write_text("ignored")
    (tmp_path / "fresh.pdf").write_bytes(make_pdf([[f"doi {mine[1]['doi']}"]]))
    for name in ("found.pdf", "unknown.pdf"):
        _age(tmp_path / name)                                                  # fresh.pdf may still be downloading: skipped
    store, seen = MemoryStore(), {}
    report = {r["file"]: r for r in fa.scan_folder(con, store, world.run_id, tmp_path, "Anna", papers, seen)}
    assert report["found.pdf"]["status"] == "imported" and report["found.pdf"]["paper"] == mine[0]["id"]
    assert report["unknown.pdf"]["status"] == "unmatched" and "fresh.pdf" not in report
    assert (tmp_path / "imported" / "found.pdf").exists() and not (tmp_path / "found.pdf").exists() and (tmp_path / "unknown.pdf").exists()
    row = con.execute("SELECT origin,acquired_by FROM fulltext_file WHERE canonical_id=%s", (uuid.UUID(mine[0]["id"]),)).fetchone()
    assert row == ("manual_upload", "Anna") and len(store.objects) == 1
    assert mine[0]["id"] not in {p["id"] for p in papers} and mine[1]["id"] in {p["id"] for p in papers}
    # An unchanged unmatched file is not judged again; once the second one has settled it is imported.
    _age(tmp_path / "fresh.pdf")
    again = fa.scan_folder(con, store, world.run_id, tmp_path, "Anna", papers, seen)
    assert [(r["file"], r["status"]) for r in again] == [("fresh.pdf", "imported")]


def test_scan_folder_rejects_a_file_that_is_not_a_pdf(con, world, tmp_path):
    (tmp_path / "page.pdf").write_bytes(b"<html>Just a moment...</html>")
    _age(tmp_path / "page.pdf")
    report = fa.scan_folder(con, MemoryStore(), world.run_id, tmp_path, "Anna", [], {})
    assert report[0]["status"] == "rejected" and "not a PDF" in report[0]["reason"]


def test_watch_once_summarises(con, world, tmp_path):
    summary = fa.watch(con, MemoryStore(), world.run_id, tmp_path / "new-folder", "Anna", once=True, emit=lambda line: None)
    assert summary["imported"] == 0 and (tmp_path / "new-folder").is_dir() and "still_missing" in summary
