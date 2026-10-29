import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research0.novelty_ablation_analysis import analyze, check_bundle


PROPOSAL = "The proposal tests whether organizational trust changes how teams use AI advice."
PAPER = "The published study examines organizational trust and the adoption of AI advice."
PROMPT_HASH = "a" * 64
CONFIG_HASH = "b" * 64
RUBRIC_VERSION = "novelty-three-dimensions-ablation-v1"


def fixture(root, *, bad_quote=False, incomplete=False):
    bundle = root / "bundle"
    bundle.mkdir()
    packets, keys, results = [], [], []
    arms = {"semantic_five": list(range(1, 6)),
            "semantic_plus_lexical": list(range(1, 11)),
            "semantic_plus_placebo": list(range(1, 6)) + list(range(11, 16))}
    for arm, ids in arms.items():
        packet_id = f"packet-{arm}"
        paper_ids = {f"V{index}": f"paper-{number}" for index, number in enumerate(ids, 1)}
        packets.append({"packet_id": packet_id, "rubric_version": RUBRIC_VERSION,
                        "rubric": "Compare question, mechanism, and predictions with quoted passages.",
                        "proposal_text": PROPOSAL,
                        "papers": [{"ref": ref, "title": eid, "year": 2025, "abstract": PAPER} for ref, eid in paper_ids.items()]})
        keys.append({"packet_id": packet_id, "proposal_id": "proposal-1", "arm": arm,
                     "paper_ids": paper_ids, "source_sha256": "source-hash"})
        dimension = {"assessment": "overlap", "explanation": "These claims concern the same process.",
                     "proposal_passage": PROPOSAL, "neighbour_passage": "This passage is absent" if bad_quote and arm == "semantic_plus_lexical" else PAPER}
        if not (incomplete and arm == "semantic_plus_placebo"):
            results.append({"packet_id": packet_id, "repeat": 1, "model_id": "test-model",
                            "prompt_sha256": PROMPT_HASH, "rubric_version": RUBRIC_VERSION,
                            "status": "scored", "vote": {"semantic_five": 5, "semantic_plus_lexical": 7,
                                                         "semantic_plus_placebo": 6}[arm],
                            "comparisons": [{"neighbour_id": "V6" if arm != "semantic_five" else "V1",
                                             **{name: dimension for name in ("question", "mechanism", "predictions")}}]})
    packet_path, key_path = bundle / "packets.jsonl", bundle / "private_key.json"
    packet_path.write_text("".join(json.dumps(packet) + "\n" for packet in packets))
    key_path.write_text(json.dumps(keys))
    (bundle / "manifest.json").write_text(json.dumps({"source_sha256": "source-hash",
        "packet_layout_version": "fixed-semantic-slots-v2",
        "proposals": 1, "packets": 3, "packets_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "private_key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest()}))
    packet_hash = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    for row in results:
        row["packets_sha256"] = packet_hash
        row["config_sha256"] = CONFIG_HASH
    result_path = root / "results.jsonl"
    result_path.write_text("".join(json.dumps(result) + "\n" for result in results))
    return bundle, result_path


class NoveltyAblationAnalysisTests(unittest.TestCase):
    def test_paired_contrasts_and_added_paper_use(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root)
            self.assertEqual(check_bundle(bundle)["status"], "verified_prepared_only_no_model_calls")
            report = analyze(bundle, results, root / "report")
            self.assertEqual(report["contrasts"]["lexical_vs_placebo"]["mean_difference"], 1)
            self.assertEqual(report["contrasts"]["lexical_vs_semantic"]["mean_difference"], 2)
            self.assertEqual(report["valid_quote_pairs"], 9)
            self.assertEqual(report["scored_ten_paper_runs_quoting_added_papers"],
                             {"semantic_plus_lexical": 1, "semantic_plus_placebo": 1})
            self.assertTrue((root / "report/private_cases.csv").is_file())

    def test_invalid_quote_does_not_become_a_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root, bad_quote=True)
            report = analyze(bundle, results, root / "report")
            self.assertEqual(report["arm_status"]["semantic_plus_lexical"], {"invalid": 1})
            self.assertEqual(report["contrasts"]["lexical_vs_placebo"]["proposals_paired"], 0)

    def test_insufficient_evidence_is_counted_without_a_vote(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            for row in rows:
                if row["packet_id"] == "packet-semantic_plus_lexical":
                    row.update(status="insufficient_evidence", vote=None,
                               reason="The proposed mechanism is too underspecified.", comparisons=[])
            results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = analyze(bundle, results, root / "report")
            self.assertEqual(report["arm_status"]["semantic_plus_lexical"], {"insufficient_evidence": 1})
            self.assertIsNone(report["contrasts"]["lexical_vs_placebo"]["mean_difference"])

    def test_mention_without_paired_quote_is_not_added_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            lexical = next(row for row in rows if row["packet_id"] == "packet-semantic_plus_lexical")
            lexical["comparisons"][0]["neighbour_id"] = "V1"
            lexical["comparisons"].append({"neighbour_id": "V6", **{
                name: {"assessment": "unclear", "explanation": "This abstract does not address it.",
                       "proposal_passage": None, "neighbour_passage": None}
                for name in ("question", "mechanism", "predictions")}})
            results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = analyze(bundle, results, root / "report")
            self.assertEqual(report["arm_status"]["semantic_plus_lexical"], {"scored": 1})
            self.assertEqual(report["scored_ten_paper_runs_quoting_added_papers"],
                             {"semantic_plus_lexical": 0, "semantic_plus_placebo": 1})

    def test_semantic_order_must_match_between_ten_paper_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, _ = fixture(root)
            key_path = bundle / "private_key.json"
            keys = json.loads(key_path.read_text())
            placebo = next(row for row in keys if row["arm"] == "semantic_plus_placebo")
            placebo["paper_ids"]["V1"], placebo["paper_ids"]["V2"] = (
                placebo["paper_ids"]["V2"], placebo["paper_ids"]["V1"])
            key_path.write_text(json.dumps(keys))
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["private_key_sha256"] = hashlib.sha256(key_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "semantic papers moved"):
                check_bundle(bundle)

    def test_incomplete_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root, incomplete=True)
            with self.assertRaisesRegex(ValueError, "incomplete or unbalanced"):
                analyze(bundle, results, root / "report")

    def test_result_from_other_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, results = fixture(root)
            rows = [json.loads(line) for line in results.read_text().splitlines()]
            rows[0]["packets_sha256"] = "c" * 64
            results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "different packet bundle"):
                analyze(bundle, results, root / "report")

    def test_tampered_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle, _ = fixture(root)
            with (bundle / "packets.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
