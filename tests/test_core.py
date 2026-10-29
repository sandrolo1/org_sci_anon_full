import csv

from research0.core import coverage, mapped, match, norm_title, query_text, read_scopus, reconcile


def test_query_only_source_and_fixed_filters():
    q = query_text(["20", "10", "10"])
    assert q == "(SRCID(10) OR SRCID(20))\nAND PUBYEAR AFT 2014\nAND PUBYEAR BEF 2026\nAND (DOCTYPE(ar) OR DOCTYPE(re))\nAND SRCTYPE(j)"


def test_identifier_conflict_is_ambiguous():
    seed = dict(status="eligible", doi="a", eid="2", title="A title", publication_year=2020)
    records = [dict(record_id="r1", doi="a", eid="1", title_normalized="a title", publication_year=2020), dict(record_id="r2", doi="b", eid="2", title_normalized="other", publication_year=2020)]
    assert match(seed, records)[0] == "ambiguous"


def test_same_identifier_across_two_exports_counts_once_for_seed():
    seed = dict(status="eligible", doi="10.1/a", eid=None, title="A title", publication_year=2020)
    records = [dict(record_id="file1-row2", doi="10.1/a", eid="scopus-1", title_normalized="a title", publication_year=2020),
               dict(record_id="file2-row7", doi="10.1/a", eid="scopus-1", title_normalized="a title", publication_year=2020)]
    outcome, method, ids, _ = match(seed, records)
    assert (outcome, method, len(ids)) == ("found", "doi", 2)


def test_title_fallback_and_out_of_scope_denominator():
    rec = dict(record_id="r1", doi=None, eid=None, title_normalized=norm_title("Café & Work"), publication_year=2019)
    seed = dict(status="eligible", doi=None, eid=None, title="Cafe and Work", publication_year=2019)
    assert match(seed, [rec])[0] == "missing"
    seed["title"] = "Cafe & Work"
    assert match(seed, [rec])[0] == "found"
    result = coverage([dict(seed_status="eligible", outcome="found"), dict(seed_status="out_of_scope", outcome="out_of_scope")])
    assert (result["found"], result["eligible"], result["coverage"]) == (1, 1, 1)


def test_mapping_preserves_unknown_field_and_reconciles():
    raw = {"DOI": "https://doi.org/10.1/X", "EID": "", "Title": "Example", "Year": "2020", "Unmapped": "retained"}
    r = mapped(raw)
    assert r["doi"] == "10.1/x" and raw["Unmapped"] == "retained"
    result = reconcile([(2, r), (3, r)])
    assert result["missing_eid_rows"] == [2, 3]
    assert any(x["method"] == "doi" for x in result["potential_duplicates"])


def test_wrapped_semicolon_csv(tmp_path):
    path = tmp_path / "wrapped.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["Title,Year,DOI", "", ""])
        writer.writerow(['"Research; work",2020,10.9999/test', "", ""])
    rows = list(read_scopus(path))
    assert rows == [(2, {"Title": "Research; work", "Year": "2020", "DOI": "10.9999/test"})]
