"""Prepare an aggregate-only private retrieval diagnostic for the Vite workbench.

The input report is generated offline by research0.novelty_known_answer. Build a
fresh, explicitly allowlisted payload so paper text or identifiers cannot be
copied into the browser data if the report format later grows.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports/evaluator_export_2026-10-20/known_answer_retrieval.json"
DEFAULT_ARCHIVE = ROOT / "inputs/evaluator_export_2026-10-20/raw/proposals_evaluations_export_2026-10-20.zip"
DEFAULT_OUTPUT = ROOT / "web/public/data/retrieval.json"
CONTROLS = (
    "full_abstract", "title_only", "lead_40_words", "middle_20_words",
    "lead_20_words", "lead_20_drop_rarest_quarter",
    "lead_20_drop_rarest_half", "lead_20_drop_rarest_three_quarters",
)


def _integer(value, label, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid aggregate count: {label}")
    return value


def prepare(report=DEFAULT_REPORT, archive=DEFAULT_ARCHIVE, output=DEFAULT_OUTPUT):
    report, archive, output = Path(report), Path(archive), Path(output)
    output.unlink(missing_ok=True)
    if not report.is_file():
        return {"available": False, "reason": f"Aggregate report unavailable: {report.name}"}
    raw_bytes = report.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("method") != "known-source-bm25-controls-v1":
        raise ValueError("Unsupported known-source retrieval report")
    seed = raw.get("sample_seed")
    if not isinstance(seed, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", seed):
        raise ValueError("Invalid sample seed")
    n = _integer(raw.get("sample_size"), "sample_size")
    if not n:
        raise ValueError("Empty retrieval sample")
    source_hash = raw.get("source_zip_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("Missing source archive checksum")
    if archive.is_file() and hashlib.sha256(archive.read_bytes()).hexdigest() != source_hash:
        return {"available": False, "reason": "Aggregate report and local source archive have different checksums"}
    coverage = raw.get("coverage") or {}
    clean_coverage = {key: _integer(coverage.get(key), key) for key in (
        "catalog_rows", "eligible_unique_abstracts", "excluded_short_missing_or_duplicate")}
    if clean_coverage["eligible_unique_abstracts"] + clean_coverage["excluded_short_missing_or_duplicate"] != clean_coverage["catalog_rows"]:
        raise ValueError("Catalog coverage counts disagree")
    clean_metrics = {}
    for name in CONTROLS:
        values = (raw.get("metrics") or {}).get(name) or {}
        count = _integer(values.get("n"), f"{name}.n")
        if count != n:
            raise ValueError(f"Sample size disagrees for {name}")
        metric = {"n": count}
        for field in ("positive_source_score", "source_at_1", "source_at_5", "source_at_20"):
            metric[field] = _integer(values.get(field), f"{name}.{field}", count)
        if not (metric["source_at_1"] <= metric["source_at_5"] <= metric["source_at_20"] <= metric["positive_source_score"]):
            raise ValueError(f"Recovery counts disagree for {name}")
        median = values.get("median_source_rank_when_scored")
        if median is not None and (isinstance(median, bool) or not isinstance(median, (int, float)) or not math.isfinite(median) or median < 1):
            raise ValueError(f"Invalid median rank for {name}")
        metric["median_source_rank_when_scored"] = median
        clean_metrics[name] = metric
    payload = {
        "method": raw["method"], "sample_seed": seed, "sample_size": n,
        "coverage": clean_coverage, "metrics": clean_metrics,
        "source": {"report_name": report.name,
                   "report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                   "archive_sha256": source_hash,
                   "local_archive_verified": archive.is_file()},
    }
    vector = raw.get("vector_self_check")
    if vector is not None:
        if vector.get("method") != "frozen-vector-self-alignment-v1" or vector.get("sample_seed") != seed:
            raise ValueError("Unsupported vector self-check")
        vector_n = _integer(vector.get("n"), "vector_self_check.n", n)
        if not vector_n:
            raise ValueError("Empty vector self-check")
        checks = {field: _integer(vector.get(field), f"vector_self_check.{field}", vector_n)
                  for field in ("source_at_1", "source_at_5", "source_at_20")}
        if not checks["source_at_1"] <= checks["source_at_5"] <= checks["source_at_20"]:
            raise ValueError("Vector self-check counts disagree")
        payload["vector_self_check"] = {"method": vector["method"], "n": vector_n, **checks}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"available": True, "sample_size": n, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.report, args.archive, args.output)))
