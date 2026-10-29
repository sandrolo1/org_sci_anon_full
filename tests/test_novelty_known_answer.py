import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research0.novelty_known_answer import benchmark, compare_methods, controls, frozen_vector_self_check, select_sources, _query_vectors
from research0.scoring_lexical import Bm25
import numpy as np


def _rows():
    return [
        {"eid": "alpha", "title": "Distributed authority in creative teams",
         "abstract": " ".join(["Creative teams distribute authority through rotating leadership and reciprocal critique."] * 12)},
        {"eid": "beta", "title": "Supplier finance and factory resilience",
         "abstract": " ".join(["Factories use supplier finance to buffer inventory shocks and preserve production continuity."] * 12)},
        {"eid": "gamma", "title": "Market entry and ecosystem governance",
         "abstract": " ".join(["Ecosystem governance shapes complementor entry through access rules and platform boundaries."] * 12)},
    ]


class KnownAnswerTests(unittest.TestCase):
    def test_recovery_is_deterministic_and_aggregate_only(self):
        rows = _rows()
        result = benchmark(rows, sample_size=3, seed="fixed")
        self.assertEqual(result["coverage"], {"catalog_rows": 3, "eligible_unique_abstracts": 3,
                                              "excluded_short_missing_or_duplicate": 0})
        for metric in result["metrics"].values():
            self.assertEqual(metric["n"], 3)
            self.assertEqual(metric["source_at_1"], 3)
        self.assertEqual(result, benchmark(list(reversed(rows)), sample_size=3, seed="fixed"))
        serialized = json.dumps(result)
        self.assertTrue(all(word not in serialized for word in ("alpha", "beta", "gamma", "Creative teams")))
        self.assertIn("not a paraphrase or substantive novelty accuracy", result["interpretation"])

    def test_erasure_removes_rarest_terms_and_keeps_query_nonempty(self):
        rows = _rows()
        index = Bm25([row["title"] + " " + row["abstract"] for row in rows])
        queries = controls(rows[0], np.diff(index.weights.indptr), index.vectorizer.vocabulary_, len(rows))
        self.assertEqual(len(queries["lead_40_words"].split()), 40)
        self.assertEqual(len(queries["middle_20_words"].split()), 20)
        self.assertTrue(queries["lead_20_drop_rarest_three_quarters"])
        self.assertLess(len(queries["lead_20_drop_rarest_three_quarters"].split()), 20)

    def test_rejects_ambiguous_or_insufficient_source_sets(self):
        rows = _rows()
        duplicate = dict(rows[0], eid="delta")
        sample, coverage = select_sources(rows + [duplicate], sample_size=2)
        self.assertEqual(len(sample), 2)
        self.assertEqual(coverage["excluded_short_missing_or_duplicate"], 2)
        with self.assertRaisesRegex(ValueError, "only 2"):
            select_sources(rows + [duplicate], sample_size=3)
        with self.assertRaisesRegex(ValueError, "unique and nonempty"):
            benchmark(rows + [dict(rows[0])], sample_size=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            select_sources(rows, sample_size=0)

    def test_hybrid_adds_lexical_source_when_semantic_top_five_misses(self):
        rows = [{"eid": f"p{i}", "title": f"Distinctive{i} mechanism",
                 "abstract": " ".join([f"Distinctive{i} mechanism shapes a unique organizational pathway for research teams."] * 12)}
                for i in range(6)]
        source, _ = select_sources(rows, sample_size=1, seed="fixed")
        target = rows.index(source[0])
        query_vector = np.full(6, 0.5)
        query_vector[target] = -1.0
        metrics = compare_methods(rows, [{"id": row["eid"], "title": row["title"]} for row in rows],
                                  np.eye(6), np.vstack([query_vector] * 3), sample_size=1, seed="fixed")
        direct = metrics["lead_20_words"]
        self.assertEqual(direct["semantic_at_5"], 0)
        self.assertEqual(direct["lexical_at_5"], 1)
        self.assertEqual(direct["hybrid_5_plus_5"], 1)
        self.assertEqual(direct["lexical_added_to_hybrid"], 1)

    def test_missing_cache_requires_explicit_api_opt_in(self):
        with self.assertRaisesRegex(ValueError, "--allow-api"):
            _query_vectors([("p1", "lead_20_words", "some abstract words")],
                           cache_path="/tmp/nonexistent_novelty_known_answer_test_cache.npz",
                           source_hash="source", allow_api=False)
        with self.assertRaisesRegex(ValueError, "6,000 words total"):
            _query_vectors([("p1", "lead_20_words", "long " * 41)],
                           cache_path="/tmp/nonexistent_novelty_known_answer_test_cache.npz",
                           source_hash="source", allow_api=True)

    def test_frozen_vector_self_check_verifies_source_hashes(self):
        rows = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "corpus.json").write_text(json.dumps({"embedder": "openai:text-embedding-3-small",
                                                          "corpus_id": "test"}), encoding="utf-8")
            entries = [{"id": row["eid"], "text_sha256": hashlib.sha256(row["abstract"].encode()).hexdigest()}
                       for row in rows]
            (root / "entries.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            vectors = np.zeros((3, 1536), dtype=np.float32)
            vectors[np.arange(3), np.arange(3)] = 1
            np.save(root / "vectors.npy", vectors)
            result = frozen_vector_self_check(rows, root, sample_size=3)
            self.assertEqual((result["source_at_1"], result["n"]), (3, 3))
            self.assertNotIn("alpha", json.dumps(result))
            entries[0]["text_sha256"] = "wrong"
            (root / "entries.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                frozen_vector_self_check(rows, root, sample_size=3)


if __name__ == "__main__":
    unittest.main()
