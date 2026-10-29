import json
import tempfile
import unittest
import datetime as dt
from pathlib import Path

from research0.novelty_offline import build_dossier, main, validate_claims
from research0.novelty_export import export_snapshot


CATALOG = {"catalog_run_id": "frozen-1", "exported_at": "2026-10-18T00:00:00Z",
           "literature_cutoff_year": 2025, "papers": [
               {"id": "p1", "title": "Distributed decision rights", "year": 2024,
                "abstract": "Distributed decision rights change coordination in teams.", "eligible": True},
               {"id": "p2", "title": "Authority in teams", "year": 2025,
                "abstract": None, "eligible": True},
               {"id": "p3", "title": "A later study", "year": 2026,
                "abstract": "Distributed decision rights", "eligible": True},
               {"id": "p4", "title": "Out of scope", "year": 2024,
                "abstract": "Distributed decision rights", "eligible": False},
           ]}
QUESTIONS = [{"id": "q1", "content": {"rq": "How do teams share decision rights?",
              "puzzle": "Teams differ.", "mechanism": "Distributed authority changes coordination.",
              "predictions": [{"statement": "Coordination improves."}]},
              "author_stated_aliases": [{"text": "decision rights", "paper_id": "p1",
                                         "quote": "Distributed decision rights change coordination in teams."}]}]


class OfflineNoveltyTests(unittest.TestCase):
    def test_dossier_filters_scope_and_preserves_coverage(self):
        report = build_dossier(CATALOG, QUESTIONS)
        self.assertEqual(report["coverage"], {"exported": 4, "eligible": 2,
                                                "abstract_available": 1, "title_only": 1,
                                                "excluded": 1, "dated_after_cutoff": 1})
        self.assertEqual(report["questions"][0]["status"], "retrieval_only")
        self.assertEqual(report["questions"][0]["candidates"][0]["id"], "p1")
        self.assertNotIn("p3", [h["id"] for h in report["questions"][0]["candidates"]])
        self.assertNotIn("p4", [h["id"] for h in report["questions"][0]["candidates"]])

    def test_missing_question_abstains_and_unsupported_alias_fails(self):
        missing = [{"id": "q2", "content": {"rq": "Only a question"}}]
        self.assertEqual(build_dossier(CATALOG, missing)["questions"][0]["status"], "abstain")
        altered = json.loads(json.dumps(QUESTIONS))
        altered[0]["author_stated_aliases"][0]["quote"] = "Invented evidence"
        with self.assertRaisesRegex(ValueError, "exact abstract quote"):
            build_dossier(CATALOG, altered)

    def test_claim_pointer_requires_exact_abstract_quote(self):
        claims = [{"question_id": "q1", "paper_id": "p1", "label": "partial_overlap",
                   "quote": "Distributed decision rights change coordination in teams."},
                  {"question_id": "q1", "paper_id": "p2", "label": "apparent_prior_answer",
                   "quote": "Authority in teams"},
                  {"question_id": "q1", "paper_id": "p1", "label": "partial_overlap",
                   "quote": "Distributed decision rights prove novelty."}]
        checks = validate_claims(claims, CATALOG["papers"][:2], {"q1"})
        self.assertEqual([c["valid_pointer"] for c in checks], [True, False, False])
        self.assertEqual(checks[1]["reason"], "quote_not_in_abstract")

    def test_cli_writes_reproducible_input_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "catalog.json").write_text(json.dumps(CATALOG), encoding="utf-8")
            (root / "questions.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
            main(["--catalog", str(root / "catalog.json"), "--questions", str(root / "questions.json"),
                  "--output", str(root / "report.json")])
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["input_sha256"]["catalog"]), 64)
            self.assertEqual(report["catalog_run_id"], "frozen-1")

    def test_export_uses_read_only_snapshot_and_preserves_excluded_rows(self):
        class Cursor:
            def __init__(self, rows):
                self.rows = rows

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return self.rows

        class Connection:
            def __init__(self):
                self.queries = []

            def execute(self, sql, params=None):
                self.queries.append((sql, params))
                if sql.startswith("SET TRANSACTION"):
                    return Cursor([])
                if "FROM catalog_run" in sql:
                    return Cursor([("run-1", "query-1", ["batch-1"],
                                    dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc))])
                if "FROM canonical_paper" in sql:
                    return Cursor([("p1", "10.1/a", None, "A paper", "A. Author", 2025,
                                    "Journal", "retained", [], "Abstract"),
                                   ("p2", None, None, "Out of scope", None, 2024,
                                    "Journal", "review_needed", ["outside_source"], None)])
                if "FROM forum_episode" in sql:
                    return Cursor([(1,)])
                if "FROM question" in sql:
                    return Cursor([("q1", QUESTIONS[0]["content"])])
                raise AssertionError(sql)

        con = Connection()
        catalog, questions = export_snapshot(con, "run-1", 2025, "episode-1")
        self.assertIn("READ ONLY", con.queries[0][0])
        self.assertEqual([p["eligible"] for p in catalog["papers"]], [True, False])
        self.assertEqual(questions[0]["id"], "q1")


if __name__ == "__main__":
    unittest.main()
