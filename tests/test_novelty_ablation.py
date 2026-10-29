import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from research0.novelty_ablation import prepare


class NoveltyAblationTests(unittest.TestCase):
    def test_same_rubric_and_proposal_with_a_year_matched_placebo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = io.StringIO()
            writer = csv.DictWriter(catalog, fieldnames=["eid", "title", "abstract", "year", "doi", "journal"])
            writer.writeheader()
            for i in range(20):
                writer.writerow({"eid": f"p{i}", "title": f"Research {i}",
                                 "abstract": "This paper studies organizational mechanisms and predictions. " * (i + 1),
                                 "year": 2024 if i % 2 else 2025, "doi": f"10.1000/{i}", "journal": "Hidden journal"})
            eval_doc = {"proposal_id": "proposal-one", "evaluation": {"criteria": {"novelty": {
                "proposal_text": "TITLE: A new idea\nABSTRACT: We test a mechanism.",
                "neighbours": [{"id": f"p{i}"} for i in range(5)],
                "hybrid": {"candidates": [{"id": f"p{i}"} for i in range(10)]}}}}}
            archive = root / "export.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("proposals.json", json.dumps([{"proposal_id": "proposal-one"}]))
                output.writestr("evaluations.json", json.dumps([eval_doc]))
                output.writestr("catalog_novelty_corpus.csv", catalog.getvalue())
            manifest = prepare(archive, root / "run-one")
            prepare(archive, root / "run-two")
            self.assertEqual(manifest["packets"], 3)
            self.assertEqual(manifest["packets_sha256"], hashlib.sha256((root / "run-one/packets.jsonl").read_bytes()).hexdigest())
            self.assertEqual(manifest["private_key_sha256"], hashlib.sha256((root / "run-one/private_key.json").read_bytes()).hexdigest())
            self.assertEqual(manifest["placebo_exact_year_matches"], 5)
            self.assertEqual((root / "run-one/packets.jsonl").read_bytes(),
                             (root / "run-two/packets.jsonl").read_bytes())
            packets = [json.loads(line) for line in (root / "run-one/packets.jsonl").read_text().splitlines()]
            self.assertEqual({len(p["papers"]) for p in packets}, {5, 10})
            self.assertEqual(len({p["rubric"] for p in packets}), 1)
            self.assertEqual(len({p["proposal_text"] for p in packets}), 1)
            self.assertTrue(all("journal" not in paper and "doi" not in paper
                                for packet in packets for paper in packet["papers"]))
            key = json.loads((root / "run-one/private_key.json").read_text())
            arms = {row["arm"]: set(row["paper_ids"].values()) for row in key}
            self.assertEqual(len(arms["semantic_five"]), 5)
            self.assertTrue(arms["semantic_five"].issubset(arms["semantic_plus_lexical"]))
            self.assertTrue(arms["semantic_five"].issubset(arms["semantic_plus_placebo"]))
            self.assertEqual(len(arms["semantic_plus_lexical"] & arms["semantic_plus_placebo"]), 5)
            ten_keys = {row["arm"]: row["paper_ids"] for row in key}
            semantic_ids = arms["semantic_five"]
            self.assertEqual({ref: eid for ref, eid in ten_keys["semantic_plus_lexical"].items() if eid in semantic_ids},
                             {ref: eid for ref, eid in ten_keys["semantic_plus_placebo"].items() if eid in semantic_ids})


if __name__ == "__main__":
    unittest.main()
