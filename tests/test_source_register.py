import csv

import pytest

from research0 import source_register as sr

QUERY = "( SRCID(111) OR SRCID(222) OR SRCID(333) OR SRCID(111) ) AND PUBYEAR AFT 2014"


def good(**changes):
    row = {"source_id": "111", "version": "reg-v1", "exact_title": "Journal of Mock Studies", "issn": "1234-5678", "eissn": "", "ajg_grade": "4*", "utd24": "",
           "ais_supplement": "", "community": "CTO", "verified_doi": "https://doi.org/10.1/ABC", "verified_eid": "", "verified_on": "2026-10-10", "verified_by": "Ana",
           "title_variants": "J. Mock Stud.; Mock Studies Journal", "notes": ""}
    row.update(changes)
    return row


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sr.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_query_sources_are_read_in_order_without_repeats():
    assert sr.query_srcids(QUERY) == ["111", "222", "333"]


def test_the_template_has_every_source_and_nothing_filled_in(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text(QUERY, encoding="utf-8")
    assert sr.write_template(q, tmp_path / "t.csv") == 3
    rows = list(csv.DictReader((tmp_path / "t.csv").open(encoding="utf-8")))
    assert [r["source_id"] for r in rows] == ["111", "222", "333"] and all(not v for r in rows for k, v in r.items() if k != "source_id")
    real = list(csv.DictReader(open("templates/source_register_from_query_v1.csv", encoding="utf-8"))) if __import__("os").path.exists("templates/source_register_from_query_v1.csv") else []
    assert len(real) in (0, 34)


def test_a_valid_row_is_normalised():
    r = sr.parse_row(good())
    assert r["verified_doi"] == "10.1/abc" and r["title_variants"] == ["J. Mock Stud.", "Mock Studies Journal"] and r["utd24"] is False and r["eissn"] is None
    assert sr.parse_row(good(issn="", eissn="2345-678X", verified_doi="", verified_eid="2-S2.0-1"))["verified_eid"] == "2-s2.0-1"


@pytest.mark.parametrize("changes,fragment", [
    ({"source_id": "abc"}, "numeric"), ({"exact_title": ""}, "exact_title"), ({"version": ""}, "version"), ({"issn": "12345678"}, "issn"), ({"issn": "", "eissn": ""}, "ISSN"),
    ({"ajg_grade": "5"}, "ajg_grade"), ({"community": "both,"}, "community"), ({"community": ""}, "community"), ({"verified_doi": "", "verified_eid": ""}, "verified CTO/TIM paper"),
    ({"verified_on": "yesterday"}, "date"), ({"verified_by": ""}, "verified_by"), ({"utd24": "maybe"}, "true or false")])
def test_every_missing_or_malformed_field_is_refused_with_its_source(changes, fragment):
    with pytest.raises(ValueError, match=fragment):
        sr.parse_row(good(**changes))


def test_eligibility_follows_the_proposal_rule():
    p = lambda **c: sr.parse_row(good(**c))
    assert sr.eligibility(p(ajg_grade="4")) == "eligible" and sr.eligibility(p(ajg_grade="", utd24="true")) == "eligible"
    assert sr.eligibility(p(ajg_grade="3", utd24="false")) == "not_eligible"                         # never reported as meeting the criterion
    assert sr.eligibility(p(ajg_grade="3", ais_supplement="true")) == "ais_supplement"              # separately flagged
    assert sr.eligibility(p(ajg_grade="", utd24="", ais_supplement="")) == "not_eligible"


def test_import_is_idempotent_all_or_nothing_and_refuses_changes(world, con, tmp_path):
    path = write(tmp_path / "r.csv", [good(), good(source_id="222", ajg_grade="", utd24="true", community="both")])
    assert sr.import_register(con, path) == {"rows": 2, "inserted": 2}
    assert sr.import_register(con, path) == {"rows": 2, "inserted": 0}
    changed = write(tmp_path / "c.csv", [good(exact_title="Another Title")])
    with pytest.raises(ValueError, match="new version"):
        sr.import_register(con, changed)
    bad = write(tmp_path / "b.csv", [good(source_id="333"), good(source_id="444", community="nobody")])
    with pytest.raises(ValueError, match="community"):
        sr.import_register(con, bad)
    assert con.execute("SELECT count(*) FROM source_register").fetchone()[0] == 2                    # the good row of the bad file was not kept
    with pytest.raises(ValueError, match="twice"):
        sr.import_register(con, write(tmp_path / "d.csv", [good(source_id="555"), good(source_id="555")]))
    with pytest.raises(ValueError, match="Required columns"):
        (tmp_path / "e.csv").write_text("source_id\n1\n", encoding="utf-8")
        sr.import_register(con, tmp_path / "e.csv")
    # a new version of the same source is a different record
    assert sr.import_register(con, write(tmp_path / "v2.csv", [good(version="reg-v2", exact_title="Another Title")]))["inserted"] == 1


def test_the_check_compares_the_register_with_the_query(world, con, tmp_path):
    sr.import_register(con, write(tmp_path / "r.csv", [good(), good(source_id="222", ajg_grade="3", ais_supplement="true"), good(source_id="999", ajg_grade="2")]))
    report = sr.check(con, "reg-v1", QUERY)
    assert report["missing_in_register"] == ["333"] and report["not_in_query"] == ["999"] and report["not_eligible"] == ["999"] and report["ais_supplements"] == ["222"]
    assert report["status"] == {"eligible": 1, "ais_supplement": 1, "not_eligible": 1} and not report["ready"] and len(report["problems"]) == 3
    con.execute("DELETE FROM source_register WHERE source_id='999'")
    sr.import_register(con, write(tmp_path / "r2.csv", [good(source_id="333", utd24="true", community="TIM")]))
    done = sr.check(con, "reg-v1", QUERY)
    assert done["ready"] and done["problems"] == [] and done["by_community"] == {"CTO": 2, "TIM": 1}
    assert not sr.check(con, "reg-v1", "no sources here")["ready"]


def test_the_database_refuses_a_row_without_evidence(world, con):
    import psycopg
    for sql in ("INSERT INTO source_register(version,source_id,exact_title,issn,community,verified_on,verified_by) VALUES ('v','1','T','1234-5678','CTO','2026-01-01','Ana')",   # no verified paper
                "INSERT INTO source_register(version,source_id,exact_title,community,verified_doi,verified_on,verified_by) VALUES ('v','1','T','CTO','10.1/x','2026-01-01','Ana')"):  # no ISSN
        with pytest.raises(psycopg.errors.CheckViolation):
            con.execute(sql)
        con.rollback()
