"""Build private local app data from the evaluator's frozen evaluation export."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from research0.evaluation_integrity import audit_export  # noqa: E402
from research0.evaluation_snapshot import build_followup_csv, build_json, load_payload  # noqa: E402


def prepare(export=None):
    export = Path(export) if export else ROOT / "inputs/evaluator_export_2026-10-20/raw/proposals_evaluations_export_2026-10-20.zip"
    if not export.is_file():
        raise FileNotFoundError(f"Evaluation export not found: {export}")
    audit = audit_export(export, ROOT / "data/paper_cards_2026-10-19/method_cards.jsonl",
                         ROOT / "data/paper_cards_2026-10-19/impact_cards.jsonl")
    report_dir = ROOT / "reports/evaluator_export_2026-10-20"
    report_dir.mkdir(parents=True, exist_ok=True)
    aggregate = report_dir / "integrity_aggregate.json"
    cases = report_dir / "integrity_cases.json"
    aggregate.write_text(json.dumps(audit["aggregate"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cases.write_text(json.dumps(audit["case_flags"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = build_json(export, ROOT / "web/public/data/evaluations.json",
                        integrity_aggregate=aggregate, integrity_cases=cases)
    payload = load_payload(export, integrity_aggregate=aggregate, integrity_cases=cases)
    queue = ROOT / "web/public/data/followup_queue.csv"
    result["followup_queue"] = build_followup_csv(payload, queue)["cases"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Override the dated local export ZIP")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export)))
