import unittest

from research0.novelty_export import export_scored_proposals
from research0.novelty_retrieval_audit import audit, doi_key


CATALOG = {"catalog_run_id": "r1", "literature_cutoff_year": 2025, "papers": [
    {"id": "p1", "title": "Distributed decision rights", "abstract": "Distributed decision rights change coordination in teams.",
     "doi": "10.1000/a", "eid": "2-s2.0-1", "year": 2024, "eligible": True},
    {"id": "p2", "title": "Authority in teams", "abstract": "[No abstract available]",
     "doi": "10.1000/b", "year": 2025, "eligible": True},
    {"id": "p3", "title": "Future decisions", "abstract": "Distributed decision rights change coordination in teams.",
     "doi": "10.1000/c", "year": 2026, "eligible": True},
    {"id": "p4", "title": "Excluded decisions", "abstract": "Distributed decision rights change coordination in teams.",
     "doi": "10.1000/d", "year": 2024, "eligible": False},
]}


class RetrievalAuditTests(unittest.TestCase):
    def test_compares_full_catalog_lexical_hits_with_mapped_semantic_neighbors(self):
        score = {"corpora": {"novelty": {"corpus_id": "cor_scopus"}},
                 "criteria": {"novelty": {"neighbours": [
                     {"id": "scopus-1", "doi": "https://doi.org/10.1000/A"},
                     {"id": "scopus-missing", "doi": "10.1000/missing"}]}}}
        rows = [{"id": "proposal-1", "title": "Decision rights", "abstract": "How do distributed decision rights affect team coordination?", "score": score}]
        report = audit(CATALOG, rows)
        item = report["proposals"][0]
        self.assertEqual(report["coverage"]["eligible_within_cutoff"], 2)
        self.assertEqual(item["status"], "comparison_ready")
        self.assertEqual(item["lexical_candidates"][0]["id"], "p1")
        self.assertEqual(item["semantic_mapped"][0]["id"], "p1")
        self.assertEqual(item["semantic_mapped"][0]["mapped_by"], "doi")
        self.assertEqual(item["semantic_unmapped"][0]["neighbor_id"], "scopus-missing")
        self.assertEqual(item["overlap_at_5"], 1)
        self.assertEqual(item["semantic_corpus_id"], "cor_scopus")
        self.assertEqual(doi_key("DOI:10.1000/A"), "10.1000/a")

    def test_scopus_eid_maps_when_doi_is_missing(self):
        score = {"criteria": {"novelty": {"neighbours": [{"id": "2-s2.0-1", "doi": None}]}}}
        report = audit(CATALOG, [{"id": "proposal-1", "title": "Decision rights",
                                  "abstract": "Distributed decision rights affect coordination", "score": score}])
        item = report["proposals"][0]
        self.assertEqual(item["status"], "comparison_ready")
        self.assertEqual(item["semantic_mapped"][0]["id"], "p1")
        self.assertEqual(item["semantic_mapped"][0]["mapped_by"], "eid")

    def test_missing_abstract_or_new_score_stays_visible(self):
        report = audit(CATALOG, [{"id": "old", "title": "Old", "abstract": None, "score": None},
                                 {"id": "new", "title": "Decision rights", "abstract": "Coordination and decision rights", "score": None}])
        self.assertEqual(report["summary"]["statuses"], {"missing_proposal_abstract": 1, "lexical_only_no_new_score": 1})

    def test_export_keeps_older_proposals_without_score(self):
        class Result:
            def fetchall(self):
                return [("p1", {"title": "T", "abstract": "A"}, {"criteria": {}}),
                        ("p2", {"title": "Old"}, None)]

        class Connection:
            def execute(self, query, params):
                self.query, self.params = query, params
                return Result()

        con = Connection()
        rows = export_scored_proposals(con, "episode-1")
        self.assertEqual(con.params, ("episode-1",))
        self.assertIn("synthetic=false", con.query)
        self.assertEqual(rows[0]["score"], {"criteria": {}})
        self.assertIsNone(rows[1]["abstract"])


if __name__ == "__main__":
    unittest.main()
