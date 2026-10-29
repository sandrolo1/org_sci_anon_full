import csv
import json

import pytest

from research0.phase2 import build_review_packet


def test_packet_uses_latest_validator_decision_and_rejects_other_run(tmp_path):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"run": {"run_id": "run-1", "query_text": "SRCID(123) OR SRCID(456)"},
                                    "items": [{"id": "issue-1", "type": "dedup_conflict", "severity": "medium",
                                               "title": "Conflict", "decisions": ["same_paper", "distinct_papers", "needs_discussion"],
                                               "evidence": []}], "gates": []}), encoding="utf-8")
    decisions = tmp_path / "decisions.csv"
    with decisions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run_id", "item_id", "decision", "validator", "decided_at"])
        writer.writerow(["run-1", "issue-1", "same_paper", "Alice", "2026-10-09T10:00:00Z"])
        writer.writerow(["run-1", "issue-1", "withdrawn", "Alice", "2026-10-09T11:00:00Z"])
        writer.writerow(["run-1", "issue-1", "distinct_papers", "Bob", "2026-10-09T11:00:00Z"])
    result = build_review_packet(snapshot, tmp_path / "out", decisions)
    assert result["statuses"] == {"recorded": 1}
    with (tmp_path / "out" / "review_queue.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["decision"] == "distinct_papers" and row["validators"] == "Bob"
    with (tmp_path / "out" / "source_validation_worksheet.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [r["source_id"] for r in csv.DictReader(stream)] == ["123", "456"]
    decisions.write_text(decisions.read_text(encoding="utf-8").replace("run-1,issue-1", "run-2,issue-1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="different catalog run"):
        build_review_packet(snapshot, tmp_path / "out", decisions)
