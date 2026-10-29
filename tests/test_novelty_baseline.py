import unittest
from unittest.mock import patch

from research0.novelty_baseline import compare_abstracts, compare_catalog, fuse_candidates, lexical_candidates, question_card


class NoveltyBaselineTests(unittest.TestCase):
    def test_question_card_copies_structured_fields_without_design_text(self):
        content = {"rq": "Why do teams share authority?", "puzzle": "Existing accounts diverge.",
                   "mechanism": "Shared routines change who decides.",
                   "predictions": [{"statement": "Authority disperses."}],
                   "empirical_design": {"measures": "A proprietary survey"}}
        card = question_card(content)
        self.assertIn("Shared routines change who decides.", card)
        self.assertIn("Prediction 1: Authority disperses.", card)
        self.assertNotIn("proprietary survey", card)
        with self.assertRaisesRegex(ValueError, "required"):
            question_card({"rq": "A question", "puzzle": "A puzzle", "predictions": []})

    def test_searches_globally_and_keeps_centroid_separate(self):
        papers = [
            {"id": "a", "title": "Nearest", "doi": "10.1/a", "year": 2025,
             "abstract": "A real abstract", "vector": [1, 0], "cluster": 2},
            {"id": "b", "title": "Other topic", "doi": "10.1/b", "year": 2024,
             "abstract": "Another abstract", "vector": [0, 1], "cluster": 2},
            {"id": "c", "title": "Cluster mate", "doi": "10.1/c", "year": 2023,
             "abstract": "A third abstract", "vector": [0.8, 0.6], "cluster": 1},
        ]
        result = compare_abstracts([1, 0], papers, top_k=2)
        self.assertEqual([p["id"] for p in result["nearest_papers"]], ["a", "c"])
        self.assertEqual(result["counts"]["compared"], 3)
        self.assertEqual(result["nearest_clusters"][0]["cluster"], 1)

    def test_exclusions_are_visible(self):
        papers = [
            {"id": "a", "year": 2025, "abstract": "[No abstract available]", "vector": [1, 0]},
            {"id": "b", "year": 2024, "abstract": "Has abstract", "vector": None},
            {"id": "c", "year": 2027, "abstract": "Future paper", "vector": [1, 0]},
        ]
        result = compare_abstracts([1, 0], papers, max_publication_year=2025)
        self.assertEqual(result["counts"], {"input": 3, "after_year_filter": 2,
                                             "without_abstract": 1, "without_vector": 1,
                                             "without_cluster": 0, "compared": 0})
        self.assertEqual(result["nearest_papers"], [])
        self.assertEqual(result["nearest_clusters"], [])

    def test_mismatched_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            compare_abstracts([1, 0], [{"id": "a", "abstract": "Present", "vector": [1, 0, 0]}])

    def test_catalog_packet_uses_matching_cluster_and_keeps_provenance(self):
        class Cursor:
            def __init__(self, rows):
                self.rows = rows

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return self.rows

            def __iter__(self):
                return iter(self.rows)

        class Connection:
            def execute(self, sql, _params):
                if "FROM cluster_run" in sql:
                    return Cursor([("run-1", "model-1")])
                if "FROM canonical_paper" in sql:
                    return Cursor([("paper-1", "A paper", "10.1/a", 2025, "A journal", "An abstract"),
                                   ("paper-2", "Hidden match", None, 2024, "Another journal", None)])
                if "FROM cluster_assignment" in sql:
                    return Cursor([("paper-1", 3)])
                raise AssertionError(sql)

        class Embedder:
            model_id = "model-1"

            def embed(self, texts):
                self.texts = texts
                return [[1, 0]]

        embedder = Embedder()
        with patch("research0.explore.load_vectors", return_value=(["paper-1"], [[1, 0]], ["paper text"], [True])):
            result = compare_catalog(Connection(), "run-1", "cluster-1", embedder, "Proposal abstract",
                                     lexical_queries=["Hidden match"])
        self.assertEqual(result["counts"]["compared"], 1)
        self.assertEqual(result["coverage"]["papers_without_current_embedding"], 1)
        self.assertEqual(result["nearest_papers"][0]["abstract"], "An abstract")
        self.assertEqual(result["nearest_papers"][0]["source"], "A journal")
        self.assertEqual(result["lexical_papers"][0]["id"], "paper-2")
        self.assertEqual({hit["id"] for hit in result["combined_candidates"]}, {"paper-1", "paper-2"})
        self.assertEqual(result["provenance"]["embedding_model"], "model-1")
        self.assertEqual(embedder.texts, ["Proposal abstract"])

    def test_lexical_variants_find_renamed_and_title_only_papers(self):
        papers = [
            {"id": "a", "title": "Technology absorption and decision rights", "abstract": "An organizational study"},
            {"id": "b", "title": "Distributed authority in teams", "abstract": None},
            {"id": "c", "title": "Unrelated finance study", "abstract": "Market volatility"},
        ]
        hits = lexical_candidates(["decision rights", "distributed authority"], papers)
        self.assertEqual({hit["id"] for hit in hits}, {"a", "b"})
        renamed = next(hit for hit in hits if hit["id"] == "b")
        self.assertEqual(renamed["evidence_available"], "title_only")
        self.assertEqual(renamed["matched_queries"], ["distributed authority"])

    def test_fusion_preserves_both_retrieval_paths(self):
        semantic = [{"id": "a", "title": "A", "abstract": "Text"}, {"id": "b", "title": "B"}]
        lexical = [{"id": "b", "title": "B", "matched_queries": ["alias"]},
                   {"id": "c", "title": "C"}]
        hits = fuse_candidates(semantic, lexical)
        self.assertEqual(hits[0]["id"], "b")
        self.assertEqual(hits[0]["ranks"], {"vector": 2, "lexical": 1})
        self.assertEqual(hits[0]["matched_queries"], ["alias"])


if __name__ == "__main__":
    unittest.main()
