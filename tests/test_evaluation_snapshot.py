import json
import csv
import tempfile
import unittest
from pathlib import Path

from research0.evaluation_integrity import audit_export
from research0.evaluation_snapshot import build, build_followup_csv, build_json, load_payload
from tests.test_evaluation_integrity import fixture


class EvaluationSnapshotTests(unittest.TestCase):
    def test_audit_flags_are_embedded_only_for_the_matching_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export, method, impact = fixture(root)
            audit = audit_export(export, method, impact)
            aggregate, cases, html = root / "aggregate.json", root / "cases.json", root / "view.html"
            aggregate.write_text(json.dumps(audit["aggregate"]))
            cases.write_text(json.dumps(audit["case_flags"]))
            build(export, html, integrity_aggregate=aggregate, integrity_cases=cases)
            self.assertIn("hybrid:shift_at_least_two", html.read_text(encoding="utf-8"))
            audit["aggregate"]["source_sha256"] = "wrong"
            aggregate.write_text(json.dumps(audit["aggregate"]))
            with self.assertRaisesRegex(ValueError, "another export"):
                build(export, html, integrity_aggregate=aggregate, integrity_cases=cases)

    def test_json_uses_same_export_and_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export, method, impact = fixture(root)
            audit = audit_export(export, method, impact)
            aggregate, cases, output = root / "aggregate.json", root / "cases.json", root / "app.json"
            aggregate.write_text(json.dumps(audit["aggregate"]))
            cases.write_text(json.dumps(audit["case_flags"]))
            result = build_json(export, output, integrity_aggregate=aggregate, integrity_cases=cases)
            payload = json.loads(output.read_text())
            self.assertEqual(result["proposals"], len(payload["items"]))
            self.assertEqual(payload["source_sha256"], audit["aggregate"]["source_sha256"])
            self.assertEqual(payload["items"][0]["integrity_flags"], audit["case_flags"][0]["flags"])
            queue = root / "queue.csv"
            self.assertEqual(build_followup_csv(payload, queue)["cases"], 1)
            with queue.open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["issue"], "Novelty score shifted")
            self.assertEqual(row["delta"], "2")
            self.assertEqual(row["primary_papers_retained"], "1")
            self.assertEqual(row["added_papers_compared"], "1")
            self.assertEqual(row["source_sha256"], payload["source_sha256"])

    def test_failed_hybrid_becomes_rerun_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            export, method, impact = fixture(root, hybrid_failed=True)
            audit = audit_export(export, method, impact)
            aggregate, cases = root / "aggregate.json", root / "cases.json"
            aggregate.write_text(json.dumps(audit["aggregate"]))
            cases.write_text(json.dumps(audit["case_flags"]))
            payload = load_payload(export, integrity_aggregate=aggregate, integrity_cases=cases)
            followup = payload["items"][0]["followup"]
            self.assertEqual(followup["kind"], "rerun")
            self.assertIn("quote mismatch", followup["detail"])


if __name__ == "__main__":
    unittest.main()
