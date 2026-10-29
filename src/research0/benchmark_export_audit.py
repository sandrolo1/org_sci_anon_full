"""Audit a private 2026 paper-evaluation export without model or database calls.

Example:
  PYTHONPATH=src python3 -m research0.benchmark_export_audit \
    --csv /private/path/evaluation.csv --json /private/path/evaluation.json \
    --as-of 2026-10-21 --output reports/benchmark_audit.json

The input files and row-level output contain research material. Keep them out of Git.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import html
import json
import math
from pathlib import Path
import random
import re
import statistics


SCORES = {
    "novelty_first": ("novelty_first", "novelty"),
    "novelty_hybrid": ("novelty_hybrid", "hybrid"),
    "methodology": ("methodology", "methodology"),
    "impact": ("impact", "impact"),
    "mean_of_three": ("mean_of_three", "mean"),
}
GRADES = ("1", "2", "3", "4", "4*")


def _number(value):
    return None if value in (None, "") else float(value)


def _journal(value):
    normalized = html.unescape(value or "").casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def _mean(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return statistics.mean(values) if values else None


def _rounded(value):
    return None if value is None else round(value, 3)


def _cluster_interval(high, other, field, *, seed, iterations):
    """Paper-weighted contrast, resampling journals with observed scores."""
    rng = random.Random(seed)
    groups = []
    for rows in (high, other):
        by_journal = {}
        for row in rows:
            by_journal.setdefault(_journal(row["journal"]), []).append(row[field])
        groups.append([[value for value in values if value is not None]
                       for values in by_journal.values() if any(value is not None for value in values)])
    if not all(groups) or iterations < 1:
        return None
    contrasts = []
    for _ in range(iterations):
        means = []
        for journals in groups:
            draw = [rng.choice(journals) for _ in journals]
            values = [value for journal in draw for value in journal if value is not None]
            means.append(statistics.mean(values))
        contrasts.append(means[0] - means[1])
    contrasts.sort()
    lo = contrasts[int(0.025 * (iterations - 1))]
    hi = contrasts[int(0.975 * (iterations - 1))]
    return [_rounded(lo), _rounded(hi)]


def _leave_one_journal_out(rows, field):
    """Check whether one sampled journal determines the four-star contrast."""
    gaps = []
    for omitted in sorted({_journal(row["journal"]) for row in rows}):
        kept = [row for row in rows if _journal(row["journal"]) != omitted]
        high = _mean([row for row in kept if row["grade"] == "4*"], field)
        other = _mean([row for row in kept if row["grade"] != "4*"], field)
        if high is not None and other is not None:
            gaps.append(high - other)
    if not gaps:
        return None
    return {"journals_tested": len(gaps), "minimum_gap": _rounded(min(gaps)),
            "maximum_gap": _rounded(max(gaps)), "all_gaps_positive": all(gap > 0 for gap in gaps)}


def audit(csv_path, json_path, *, as_of, seed=20261020, iterations=10000):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    export = json.loads(Path(json_path).read_text(encoding="utf-8"))
    detail_rows = export["rows"]
    if not source_rows or len(source_rows) != len(detail_rows):
        raise ValueError("CSV and JSON row counts differ or are empty")
    by_id = {row["paper_id"]: row for row in detail_rows}
    if len(by_id) != len(detail_rows) or {r["paper_id"] for r in source_rows} != set(by_id):
        raise ValueError("CSV and JSON paper IDs are missing or duplicated")

    rows = []
    issues = []
    run = export["run"]
    for csv_row in source_rows:
        paper_id = csv_row["paper_id"]
        detail = by_id[paper_id]
        for key in ("run_id", "model", "knowledge_cutoff"):
            if csv_row[key] != run[key]:
                issues.append(f"{paper_id}: CSV {key} differs from JSON run")
        for csv_key, json_key in SCORES.values():
            a, b = _number(csv_row[csv_key]), _number(detail[json_key])
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=0.011)):
                issues.append(f"{paper_id}: {csv_key} differs between CSV and JSON")
        if csv_row["abs_grade_2024"] != detail["ajg2024"]:
            issues.append(f"{paper_id}: grade differs between CSV and JSON")
        if _journal(csv_row["journal"]) != _journal(detail["journal"]):
            issues.append(f"{paper_id}: journal differs between CSV and JSON")
        if csv_row["status"] != detail["status"]:
            issues.append(f"{paper_id}: status differs between CSV and JSON")
        grade = csv_row["abs_grade_2024"]
        if grade not in GRADES:
            issues.append(f"{paper_id}: unsupported grade {grade!r}")
        values = {name: _number(csv_row[name]) for name in SCORES}
        if all(values[k] is not None for k in ("novelty_first", "methodology", "impact", "mean_of_three")):
            expected = statistics.mean(values[k] for k in ("novelty_first", "methodology", "impact"))
            if not math.isclose(values["mean_of_three"], expected, abs_tol=0.011):
                issues.append(f"{paper_id}: mean_of_three is not the first novelty + methodology + impact mean")
        online = date.fromisoformat(csv_row["online_date"])
        rows.append({
            "paper_id": paper_id,
            "journal": csv_row["journal"],
            "grade": grade,
            "online_date": online.isoformat(),
            "status": csv_row["status"],
            **values,
        })
    high = [r for r in rows if r["grade"] == "4*"]
    other = [r for r in rows if r["grade"] != "4*"]
    by_grade = {
        grade: {
            "papers": sum(r["grade"] == grade for r in rows),
            "journals": len({_journal(r["journal"]) for r in rows if r["grade"] == grade}),
            "means": {field: _rounded(_mean([r for r in rows if r["grade"] == grade], field)) for field in SCORES},
        }
        for grade in GRADES
    }
    contrasts = {}
    for index, field in enumerate(SCORES):
        a, b = _mean(high, field), _mean(other, field)
        contrasts[field] = {
            "four_star_mean": _rounded(a),
            "other_mean": _rounded(b),
            "difference": _rounded(a - b) if a is not None and b is not None else None,
            "journal_cluster_bootstrap_95pct": _cluster_interval(
                high, other, field, seed=seed + index, iterations=iterations
            ),
            "leave_one_journal_out": _leave_one_journal_out(rows, field),
        }
    future = [r["paper_id"] for r in rows if r["online_date"] > as_of.isoformat()]
    completed = [r for r in rows if r["status"] == "ok"]
    return {
        "run_id": run["run_id"],
        "model": run["model"],
        "knowledge_cutoff": run["knowledge_cutoff"],
        "as_of": as_of.isoformat(),
        "sampling": run["parameters"],
        "usage": run["usage"],
        "papers": len(rows),
        "completed": len(completed),
        "four_star_papers": len(high),
        "four_star_journals": len({_journal(r["journal"]) for r in high}),
        "other_papers": len(other),
        "other_journals": len({_journal(r["journal"]) for r in other}),
        "by_grade": by_grade,
        "four_star_vs_other": contrasts,
        "future_online_dates": future,
        "data_issues": issues,
        "interpretation": "Exploratory descriptive association; journal grade is not an article-level truth label.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args(argv)
    result = audit(args.csv, args.json, as_of=args.as_of, iterations=args.iterations)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if result["data_issues"]:
        raise SystemExit("Audit found data inconsistencies; inspect data_issues")


if __name__ == "__main__":
    main()
