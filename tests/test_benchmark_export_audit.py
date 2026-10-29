"""Checks for the exported benchmark audit; uses fabricated papers only."""

import csv
from datetime import date
import json

import pytest

from research0.benchmark_export_audit import _leave_one_journal_out, audit


def test_leave_one_journal_out_exposes_single_journal_dependence():
    rows = [
        {"journal": "Four Star A", "grade": "4*", "impact": 9},
        {"journal": "Four Star B", "grade": "4*", "impact": 3},
        {"journal": "Other A", "grade": "3", "impact": 5},
        {"journal": "Other B", "grade": "2", "impact": 5},
    ]
    result = _leave_one_journal_out(rows, "impact")
    assert result == {"journals_tested": 4, "minimum_gap": -2,
                      "maximum_gap": 4, "all_gaps_positive": False}


def files(tmp_path):
    csv_path = tmp_path / "evaluation.csv"
    json_path = tmp_path / "evaluation.json"
    base = {
        "run_id": "run-1", "model": "model-a", "knowledge_cutoff": "2025-08-31",
        "title": "A paper", "doi": "10.1/test", "status": "ok",
        "novelty_first": "7", "novelty_hybrid": "8", "methodology": "6",
        "impact": "8", "mean_of_three": "7", "judged_twice": "no",
        "gap_between_copies": "", "lexical_only_candidates": "5", "not_scored_reasons": "",
        "closest_abstract_similarity": "0.5", "closest_abstract_percentile": "0.5",
        "mean_similarity_topk": "0.5", "run_label": "", "test_data": "no",
    }
    rows = [
        {**base, "paper_id": "one", "journal": "Work Organisation, Labour &amp; Globalisation", "abs_grade_2024": "4*", "online_date": "2026-10-01"},
        {**base, "paper_id": "two", "journal": "Work Organisation Labour and Globalisation", "abs_grade_2024": "3", "online_date": "2026-11-01", "novelty_first": "5", "novelty_hybrid": "6", "methodology": "6", "impact": "4", "mean_of_three": "5"},
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    details = [{
        "paper_id": r["paper_id"], "journal": r["journal"], "ajg2024": r["abs_grade_2024"],
        "status": "ok", "novelty": float(r["novelty_first"]),
        "hybrid": float(r["novelty_hybrid"]), "methodology": float(r["methodology"]),
        "impact": float(r["impact"]), "mean": float(r["mean_of_three"]),
    } for r in rows]
    export = {"run": {"run_id": "run-1", "model": "model-a", "knowledge_cutoff": "2025-08-31", "parameters": {}, "usage": {}}, "rows": details}
    json_path.write_text(json.dumps(export), encoding="utf-8")
    return csv_path, json_path, export


def test_audit_checks_scores_and_future_dates(tmp_path):
    csv_path, json_path, export = files(tmp_path)
    result = audit(csv_path, json_path, as_of=date(2026, 10, 21), iterations=100)
    assert result["data_issues"] == []
    assert result["future_online_dates"] == ["two"]
    assert result["four_star_vs_other"]["impact"]["difference"] == 4
    export["rows"][0]["impact"] = 3
    json_path.write_text(json.dumps(export), encoding="utf-8")
    assert "impact differs" in " ".join(audit(csv_path, json_path, as_of=date(2026, 10, 21), iterations=1)["data_issues"])


def test_audit_refuses_missing_or_duplicate_ids(tmp_path):
    csv_path, json_path, export = files(tmp_path)
    export["rows"][1]["paper_id"] = "one"
    json_path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="IDs"):
        audit(csv_path, json_path, as_of=date(2026, 10, 21))


def test_audit_accepts_matching_missing_scores_but_detects_one_sided_missingness(tmp_path):
    csv_path, json_path, export = files(tmp_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    rows[0]["novelty_hybrid"] = ""
    rows[0]["methodology"] = ""
    rows[0]["mean_of_three"] = ""
    rows[0]["status"] = "partial"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    export["rows"][0].update(hybrid=None, methodology=None, mean=None, status="partial")
    json_path.write_text(json.dumps(export), encoding="utf-8")
    assert audit(csv_path, json_path, as_of=date(2026, 10, 21), iterations=10)["data_issues"] == []
    export["rows"][0]["hybrid"] = 7
    json_path.write_text(json.dumps(export), encoding="utf-8")
    assert "novelty_hybrid differs" in " ".join(audit(csv_path, json_path, as_of=date(2026, 10, 21), iterations=10)["data_issues"])
