import os
import uuid
from pathlib import Path

import pytest

from research0.cli import database, import_scopus, import_seeds, import_sources, make_query, report


@pytest.mark.skipif(not os.getenv("DATABASE_URL") or not os.getenv("S3_BUCKET"), reason="Set local PostgreSQL and S3 environment to run integration test")
def test_phase0_end_to_end(tmp_path):
    from openpyxl import Workbook

    version = "test-" + uuid.uuid4().hex[:12]
    source = tmp_path / "sources.csv"
    source.write_text(f"source_id,journal_name,inclusion_rationale,active,version,validated_on,validated_by\n999999999,TEST ONLY Journal,synthetic fixture,true,{version},2026-01-01,Fixture Human\n", encoding="utf-8")
    seed = tmp_path / "seeds.csv"
    seed.write_text(f"seed_id,community,version,doi,eid,title,authors,publication_year,journal,top_rank,expert_rationale,status,retrieval_note,validated_by,validated_on\n,CTO,{version},10.9999/example,,Synthetic example,A Tester,2020,TEST ONLY Journal,fixture,synthetic test,eligible,fixture source,Fixture Human,2026-01-01\n,TIM,{version},,,Out of range,B Tester,2020,TEST ONLY Journal,fixture,synthetic test,out_of_scope,fixture excluded,Fixture Human,2026-01-01\n", encoding="utf-8")
    book = Workbook(); ws = book.active
    ws.append(["DOI", "EID", "Title", "Authors", "Year", "Source title", "Source ID", "Document Type", "Source Type", "Extra"])
    ws.append(["10.9999/example", "2-s2.0-TEST", "Synthetic example", "A Tester", 2020, "TEST ONLY Journal", "999999999", "Article", "Journal", "retained"])
    xlsx = tmp_path / "synthetic.xlsx"; book.save(xlsx)
    with database() as con:
        con.execute(Path(__file__).resolve().parents[1].joinpath("migrations/001_phase0.sql").read_text(encoding="utf-8"))
        import_sources(con, source); import_seeds(con, seed)
        qid, _ = make_query(con, version)
        bid, reconciliation, already = import_scopus(con, xlsx, qid, "2026-01-01", "synthetic fixture")
        assert not already and reconciliation["record_count"] == 1
        assert import_scopus(con, xlsx, qid, "2026-01-01", "synthetic fixture")[2]
        run, lines = report(con, bid, version, tmp_path / "report")
        assert any("CTO | 1 / 1 | 100.0%" in line for line in lines)
        assert any("TIM | 0 / 0 | N/A" in line for line in lines)
        assert con.execute("SELECT raw_fields->>'Extra' FROM export_record WHERE batch_id=%s", (bid,)).fetchone()[0] == "retained"
