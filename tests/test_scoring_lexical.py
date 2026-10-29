"""The lexical half of the novelty retrieval (scoring_lexical.py and AbstractCorpus.candidates): BM25 over the abstracts, and the union with the embedding's closest papers."""
import numpy as np

from research0 import scoring_corpora, scoring_io, scoring_lexical

ABSTRACTS = [
    "Platform owners govern complementor entry through boundary resources and access rules.",
    "Open innovation contests attract solvers from distant fields and raise solution diversity.",
    "Remote work changes how teams share tacit knowledge across offices.",
    "Algorithmic management reallocates tasks between workers and artificial intelligence.",
    "Venture capital investors shape how startups scale and when they pivot.",
    "Boundary resources let platform owners open the platform to complementors without losing control.",
]


def test_words_are_lowercased_stop_words_dropped_and_plurals_folded():
    assert scoring_lexical.tokens("The Platforms of strategies: process, analysis and AI") == ["platform", "strategy", "process", "analysis", "ai"]
    assert scoring_lexical.tokens("") == [] and scoring_lexical.tokens(None) == []


def test_bm25_ranks_the_documents_that_share_the_rare_terms_first():
    index = scoring_lexical.Bm25(ABSTRACTS)
    scores = index.scores("How do boundary resources open a platform to complementors?")
    order = list(np.argsort(-scores))
    assert set(order[:2]) == {0, 5} and scores[0] > 0 and scores[5] > 0
    assert scores[2] == 0 and scores[4] == 0                                    # nothing in common with the query
    assert index.scores("zzzz qqqq")[0] == 0 and len(index.scores("zzzz qqqq")) == len(ABSTRACTS)
    assert index.scores("platforms")[0] == index.scores("platform")[0]          # the plural is the same term


def test_a_term_repeated_in_the_query_counts_once_and_a_rarer_term_weighs_more():
    index = scoring_lexical.Bm25(ABSTRACTS)
    assert index.scores("platform platform platform")[0] == index.scores("platform")[0]
    tiny = scoring_lexical.Bm25(["alpha beta", "alpha gamma", "alpha delta"])
    assert tiny.scores("beta")[0] > tiny.scores("alpha")[0] > 0                       # 'beta' is in one document, 'alpha' in all three: the rarer term weighs more


def write_corpus(root, vectors):
    root.mkdir(parents=True)
    np.save(root / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    for i, text in enumerate(ABSTRACTS):
        scoring_io.append_jsonl(root / "entries.jsonl", {"id": f"P{i}", "title": f"Paper {i}", "journal": "J", "year": 2020, "ajg2024": "4"})
        scoring_io.append_jsonl(root / "texts.jsonl", {"id": f"P{i}", "text": scoring_corpora.ABSTRACT_CARD + text})
    scoring_io.write_json_once(root / "corpus.json", {"corpus_id": "cor_lex", "embedder": "x", "entries": len(ABSTRACTS)})
    return scoring_corpora.AbstractCorpus(root)


def test_the_candidates_are_the_closest_by_embedding_then_what_only_the_lexical_search_brings(tmp_path):
    vectors = np.eye(6)                                                           # paper i is the unit vector i
    corpus = write_corpus(tmp_path / "cor", vectors)
    proposal = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0])                           # closest by embedding: P0, P1, P2 ...
    query = "Boundary resources let platform owners reach complementors"         # ... but the words are those of P5 and P0
    plain = corpus.neighbours(proposal, k=3)
    found = corpus.candidates(proposal, query, k=3, lexical_k=2)
    assert [c["entry"]["id"] for c in found[:3]] == [n["entry"]["id"] for n in plain]            # the embedding's own order is untouched
    assert [c["entry"]["id"] for c in found] == ["P0", "P1", "P2", "P5"]                          # P0 is found by both and listed once; P5 is the lexical-only one
    by_id = {c["entry"]["id"]: c for c in found}
    assert by_id["P0"]["found_by"] == ["semantic", "lexical"] and by_id["P1"]["found_by"] == ["semantic"] and by_id["P5"]["found_by"] == ["lexical"]
    assert by_id["P5"]["semantic_rank"] == 6 and by_id["P5"]["lexical_rank"] in (1, 2) and by_id["P5"]["bm25"] > 0 and by_id["P1"]["lexical_rank"] is None
    assert by_id["P5"]["sim"] == 0.0                                                # its cosine is shown too: the embedding sees nothing there
    assert [c["entry"]["id"] for c in corpus.candidates(proposal, query, k=3, lexical_k=0)] == ["P0", "P1", "P2"]      # no lexical search, no extra candidate
    assert [c["entry"]["id"] for c in corpus.candidates(proposal, "zzzz", k=3, lexical_k=4)] == ["P0", "P1", "P2"]     # no shared term, nothing to add
