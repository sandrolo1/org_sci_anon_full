import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from research0.evaluation_integrity import audit_export


PROPOSAL_QUOTE = "A specific mechanism connects uncertainty to employee decisions."
PAPER_QUOTE = "A published study tests a comparable mechanism in organizations."
CARD_QUOTE = "An experiment compares responses across two groups of employees."


def fixture(root, *, bad_quote=False, hybrid_failed=False):
    root = Path(root)
    method_path, impact_path = root / "method.jsonl", root / "impact.jsonl"
    for path, identifier in ((method_path, "method-paper"), (impact_path, "impact-paper")):
        path.write_text(json.dumps({"canonical_id": identifier, "card": {"design": {
            "text": CARD_QUOTE, "passages": [CARD_QUOTE], "status": "extracted"}}}) + "\n")
    proposal = {"proposal_id": "p1", "title": "A proposal", "abstract": PROPOSAL_QUOTE,
                "references": ["First source. doi:10.1000/one", "Second source. doi:10.1000/two"],
                "cards": {"method": {"card": {"design": {"status": "inferred"}}}},
                "evaluation": {"votes": {"novelty": 6, "methodology": 7, "impact": 5}, "mean_of_three": 6}}
    def comparison(neighbor, paper=PAPER_QUOTE):
        return {"neighbour_id": neighbor, "proposal_passage": PROPOSAL_QUOTE,
                "neighbour_passage": paper}
    def criterion(kind, paper_id, paper=PAPER_QUOTE):
        return {"status": "scored", "vote": {"novelty": 6, "methodology": 7, "impact": 5}[kind],
                "proposal_text": PROPOSAL_QUOTE, "neighbours": [{"id": paper_id, "label": "V1"}],
                "judges": {"judge_1": {"vote": 6, "comparisons": [comparison("V1", paper)]}}}
    novelty = criterion("novelty", "paper-one")
    if hybrid_failed:
        novelty["hybrid"] = {"status": "failed", "reason": "quote mismatch"}
    else:
        hybrid_comparison = {"neighbour_id": "V2"}
        for dimension in ("question", "mechanism", "predictions"):
            hybrid_comparison[dimension] = comparison("V2")
        novelty["hybrid"] = {"status": "scored", "vote": 8,
                             "candidates": [{"id": "paper-one", "label": "V1", "found_by": ["semantic"]},
                                            {"id": "paper-two", "label": "V2", "found_by": ["lexical"]}],
                             "judges": {"judge_1": {"vote": 8, "comparisons": [hybrid_comparison]}}}
    criteria = {"novelty": novelty, "methodology": criterion("methodology", "method-paper", CARD_QUOTE),
                "impact": criterion("impact", "impact-paper", "Wrong paper passage" if bad_quote else CARD_QUOTE)}
    evaluation = {"proposal_id": "p1", "evaluation": {"proposal_id": "p1", "status": "ok", "criteria": criteria}}
    catalog = io.StringIO()
    writer = csv.DictWriter(catalog, fieldnames=["eid", "canonical_id", "doi", "abstract"])
    writer.writeheader()
    writer.writerow({"eid": "paper-one", "canonical_id": "c1", "doi": "10.1000/one", "abstract": PAPER_QUOTE})
    writer.writerow({"eid": "paper-two", "canonical_id": "c2", "doi": "10.1000/two", "abstract": PAPER_QUOTE})
    archive = root / "export.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("proposals.json", json.dumps([proposal]))
        output.writestr("evaluations.json", json.dumps([evaluation]))
        output.writestr("catalog_novelty_corpus.csv", catalog.getvalue())
    return archive, method_path, impact_path


class EvaluationIntegrityTests(unittest.TestCase):
    def test_all_criteria_and_hybrid_passages_are_checked_against_correct_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_export(*fixture(folder))
        counts = result["aggregate"]["counts"]
        self.assertEqual(counts["novelty_quote_pairs"], 1)
        self.assertEqual(counts["methodology_quote_pairs"], 1)
        self.assertEqual(counts["impact_quote_pairs"], 1)
        self.assertEqual(counts["hybrid_quote_pairs"], 3)
        self.assertEqual(counts["hybrid_lexical_comparisons"], 1)
        self.assertEqual(counts["hybrid_cases_with_added_paper_compared"], 1)
        self.assertEqual(counts["hybrid_changed_with_added_paper_compared"], 1)
        self.assertEqual(counts["paired_reference_dois_in_corpus"], 2)
        self.assertEqual(counts["paired_reference_dois_in_semantic_five"], 1)
        self.assertEqual(counts["paired_reference_dois_in_hybrid_ten"], 2)
        self.assertEqual(counts["flag_hybrid:shift_at_least_two"], 1)
        self.assertNotIn("impact_paper_quote_mismatch", counts)

    def test_mismatched_paper_quote_and_failed_hybrid_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            result = audit_export(*fixture(folder, bad_quote=True, hybrid_failed=True))
        counts = result["aggregate"]["counts"]
        self.assertEqual(counts["impact_paper_quote_mismatch"], 1)
        self.assertEqual(counts["hybrid_failed"], 1)
        self.assertEqual(result["case_flags"][0]["flags"],
                         ["hybrid:failed", "impact:paper_quote_mismatch"])


if __name__ == "__main__":
    unittest.main()
