"""Build a private, read-only 2026 paper benchmark view from the evaluator's export."""

import argparse
import csv
from datetime import date
import hashlib
from html import unescape
import json
from pathlib import Path
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research0.benchmark_export_audit import audit  # noqa: E402


def _exports(folder):
    pairs = []
    for path in folder.glob("evaluation_*.json"):
        csv_path = path.with_suffix(".csv")
        if not csv_path.is_file():
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        pairs.append((source.get("exported_at", ""), csv_path, path))
    return sorted(pairs)


def _archive_context(folder, run_id):
    short_id = run_id.split("-")[0]
    for path in sorted(folder.glob(f"*{short_id}*.zip")):
        with ZipFile(path) as archive:
            name = "inputs/evaluation_2026/fulltext/fulltext_index.csv"
            if name not in archive.namelist():
                continue
            from io import StringIO
            rows = list(csv.DictReader(StringIO(archive.read(name).decode("utf-8-sig"))))
            versions = {kind: sum(r["version"] == kind for r in rows) for kind in ("submitted", "accepted", "published")}
            origins = {kind: sum(r["origin"] == kind for r in rows) for kind in ("project_archive", "open_access_download")}
            return {"papers": len(rows), "versions": versions, "origins": origins}
    return None


def prepare(csv_path=None, json_path=None, as_of=None):
    folder = ROOT / "inputs/evaluation_2026/exports"
    target = ROOT / "web/public/data/benchmark.json"
    if csv_path or json_path:
        csv_path = Path(csv_path) if csv_path else Path(json_path).with_suffix(".csv")
        json_path = Path(json_path) if json_path else Path(csv_path).with_suffix(".json")
    else:
        pairs = _exports(folder)
        csv_path, json_path = pairs[-1][1:] if pairs else (None, None)
    if csv_path is None or not csv_path.is_file() or not json_path.is_file():
        target.unlink(missing_ok=True)
        return {"available": False, "reason": "Place the private benchmark CSV and JSON in inputs/evaluation_2026/exports."}
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    as_of = as_of or date.fromisoformat(raw["exported_at"][:10])
    summary = audit(csv_path, json_path, as_of=as_of)
    if summary["data_issues"]:
        raise ValueError("Benchmark CSV/JSON disagree; inspect the audit before building the app")
    summary["source_versions"] = _archive_context(folder.parent, summary["run_id"])
    earlier = [(stamp, c, j) for stamp, c, j in _exports(folder) if j != json_path and stamp < raw["exported_at"]]
    if earlier:
        with earlier[-1][1].open(encoding="utf-8-sig", newline="") as handle:
            previous_ids = {row["paper_id"] for row in csv.DictReader(handle)}
        current_ids = {row["paper_id"] for row in raw["rows"]}
        summary["previous_export"] = {"papers": len(previous_ids), "reused": len(current_ids & previous_ids), "new": len(current_ids - previous_ids)}
    items = [{
        "id": row["paper_id"], "title": unescape(row["title"]), "journal": unescape(row["journal"]),
        "grade": row["ajg2024"], "doi": row["doi"], "online_date": row["online_date"],
        "status": row["status"], "novelty_first": row["novelty"],
        "novelty_hybrid": row["hybrid"], "methodology": row["methodology"],
        "impact": row["impact"], "mean_of_three": row["mean"],
        "judged_twice": row["double"], "gap_between_copies": row["gap"],
        "closest_similarity": row["top1"], "closest_percentile": row["percentile"],
    } for row in raw["rows"]]
    payload = {
        "summary": summary,
        "source": {
            "exported_at": raw["exported_at"],
            "csv_name": csv_path.name,
            "json_name": json_path.name,
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        },
        "items": items,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"available": True, "papers": len(items), "output": str(target)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    args = parser.parse_args()
    print(json.dumps(prepare(args.csv, args.json, args.as_of)))
