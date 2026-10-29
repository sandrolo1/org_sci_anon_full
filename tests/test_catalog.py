import csv
import os
import uuid

import pytest

from research0.catalog import build_clusters, cluster_records
from research0.core import norm_title

B1, B2 = uuid.uuid4(), uuid.uuid4()
TITLE = "Digital transformation and the organization of work in practice"


def rec(n, eid=None, doi=None, title=TITLE, year=2020, authors="Smith J.", source="Journal A", batch=B1, abstract=0, doctype="Article"):
    return {"record_id": uuid.UUID(int=n), "batch_id": batch, "row_number": n, "doi": doi, "eid": eid, "title": title,
            "title_normalized": norm_title(title), "authors": authors, "publication_year": year, "source_title": source,
            "document_type": doctype, "abstract_length": abstract}


def run(records):
    cl, log, flagged, stats = cluster_records(records)
    return build_clusters(cl, flagged), log, stats


def test_same_eid_across_batches_merges_and_keeps_every_record():
    records = [rec(1, eid="e1", doi="10.1/a", abstract=5), rec(2, eid="e1", doi="10.1/a", batch=B2, abstract=50), rec(3, eid="e2", doi="10.1/b", title="Other")]
    papers, log, stats = run(records)
    assert len(papers) == 2 and sum(len(p["members"]) for p in papers) == 3
    merged = papers[0]
    assert merged["representative"]["row_number"] == 2 and merged["batch_count"] == 2
    assert [e["method"] for e in log] == ["eid"] and stats["merged_eid_cross_batch"] == 1


def test_doi_merges_record_without_eid_but_not_two_different_eids():
    papers, log, _ = run([rec(1, eid="e1", doi="10.1/a"), rec(2, doi="10.1/a", batch=B2)])
    assert len(papers) == 1 and log[0]["method"] == "doi"
    papers, log, _ = run([rec(1, eid="e1", doi="10.1/a"), rec(2, eid="e2", doi="10.1/a", batch=B2)])
    assert len(papers) == 2 and {p["status"] for p in papers} == {"review_needed"}
    assert [(e["decision"], e["method"]) for e in log] == [("flagged_for_review", "doi_multiple_eids")]


def test_same_eid_with_different_doi_merges_but_flags():
    papers, log, _ = run([rec(1, eid="e1", doi="10.1/a"), rec(2, eid="e1", doi="10.1/b", batch=B2)])
    assert len(papers) == 1 and papers[0]["status"] == "review_needed"
    assert {e["decision"] for e in log} == {"merged", "flagged_for_review"}


def test_bibliographic_fallback_merges_only_without_doi_conflict():
    papers, log, _ = run([rec(1, eid="e1"), rec(2, eid="e2", batch=B2)])
    assert len(papers) == 1 and log[0]["method"] == "bibliographic_fallback"
    papers, log, _ = run([rec(1, eid="e1", doi="10.1/a"), rec(2, eid="e2", doi="10.1/b", batch=B2)])
    assert len(papers) == 2 and log[0]["method"] == "bibliographic_doi_conflict"
    # Any bibliographic field that differs prevents the fallback.
    assert len(run([rec(1, eid="e1"), rec(2, eid="e2", year=2021)])[0]) == 2
    assert len(run([rec(1, eid="e1"), rec(2, eid="e2", authors="Doe A.")])[0]) == 2
    assert len(run([rec(1, eid="e1"), rec(2, eid="e2", source="Journal B")])[0]) == 2


def test_short_titles_are_never_merged_by_fallback():
    papers, log, stats = run([rec(1, eid="e1", title="Editorial"), rec(2, eid="e2", title="Editorial")])
    assert len(papers) == 2 and not log and stats["fallback_skipped_short_title"] == 1


def test_perimeter_flags_retain_the_paper():
    papers, _, _ = run([rec(1, eid="e1", year=2014), rec(2, eid="e2", title="Another long enough title here", doctype="Editorial")])
    assert [p["perimeter_flags"] for p in papers] == [["year_outside_2015_2025"], ["document_type_outside_article_review"]]


def test_clustering_is_deterministic():
    records = [rec(1, eid="e1"), rec(2, eid="e2", batch=B2), rec(3, eid="e3", doi="10.1/c", title="Unrelated study of firms")]
    first, second = run(records), run(records)
    assert [[m["row_number"] for m in p["members"]] for p in first[0]] == [[m["row_number"] for m in p["members"]] for p in second[0]]
    assert first[1] == second[1]


@pytest.mark.skipif(not os.getenv("DATABASE_URL") or not os.getenv("S3_BUCKET"), reason="Set local PostgreSQL and S3 environment to run integration test")
def test_catalog_end_to_end_is_idempotent(tmp_path):
    from pathlib import Path

    from research0.catalog import build_catalog
    from research0.catalog_report import write_reports
    from research0.cli import database, import_scopus, register_query

    migration = Path(__file__).resolve().parents[1] / "migrations"
    label = "test-" + uuid.uuid4().hex[:12]
    header = ["Authors", "Title", "Year", "Source title", "Document Type", "Abstract", "DOI", "EID"]
    files = []
    for name, rows in (("a.csv", [["Smith J.", TITLE, 2020, "Journal A", "Article", "short", "10.9/x", "2-s2.0-1"],
                                   ["Doe A.", "A second distinct paper about innovation", 2021, "Journal A", "Review", "", "10.9/y", "2-s2.0-2"]]),
                       ("b.csv", [["Smith J.", TITLE, 2020, "Journal A", "Article", "a much longer abstract", "10.9/x", "2-s2.0-1"],
                                   ["Lee K.", "A third distinct paper about technology", 2022, "Journal B", "Article", "", "", "2-s2.0-3"]])):
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        files.append(path)
    seed_header = "seed_id,community,version,doi,eid,title,authors,publication_year,journal,top_rank,expert_rationale,status,retrieval_note,validated_by,validated_on\n"
    cto = tmp_path / "cto.csv"
    tim = tmp_path / "tim.csv"
    cto.write_text(seed_header + f"S1,CTO,v,10.9/x,,{TITLE},Smith J.,2020,Journal A,1,fixture,eligible,fixture,Fixture,2026-01-01\n", encoding="utf-8")
    tim.write_text(seed_header + "S2,TIM,v,,,Nothing like this in the catalog at all,Nobody,2020,Journal A,1,fixture,eligible,fixture,Fixture,2026-01-01\n", encoding="utf-8")
    with database() as con:
        for sql in sorted(migration.glob("*.sql")):
            con.execute(sql.read_text(encoding="utf-8"))
        qid, _ = register_query(con, label, None)
        for path in files:
            import_scopus(con, path, qid, "2026-01-01", "synthetic fixture")
        run_id, summary, already = build_catalog(con, qid)
        assert not already and (summary["input_records"], summary["canonical_papers"]) == (4, 3)
        again = build_catalog(con, qid)
        assert again[0] == run_id and again[2]
        assert con.execute("SELECT count(*) FROM canonical_paper WHERE run_id=%s", (run_id,)).fetchone()[0] == 3
        assert con.execute("SELECT count(*) FROM catalog_member WHERE run_id=%s", (run_id,)).fetchone()[0] == 4
        assert con.execute("SELECT count(*) FROM dedup_log WHERE run_id=%s AND decision='merged' AND method='eid'", (run_id,)).fetchone()[0] == 1
        from research0.dashboard import build_snapshot
        snapshot = build_snapshot(con, run_id)
        kinds = [i["type"] for i in snapshot["items"]]
        assert kinds.count("query_unrecorded") == 1 and kinds.count("seed_not_recovered") == 0
        write_reports(con, run_id, tmp_path / "report", [(cto, "CTO"), (tim, "TIM")])
        text = (tmp_path / "report" / "catalog_report.md").read_text(encoding="utf-8")
        assert [i["type"] for i in build_snapshot(con, run_id)["items"]].count("seed_not_recovered") == 1
        assert "| CTO | 1 / 1 | 100.0% |" in text and "| TIM | 0 / 1 | 0.0% |" in text and "**no** — 1 unresolved" in text
