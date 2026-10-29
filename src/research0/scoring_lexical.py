"""The lexical half of the retrieval of the novelty criterion (forum_scoring.py): BM25 over the abstracts of the CTO/TIM corpus. The embedding finds papers that mean what the proposal means;
this finds papers that say what the proposal says, in its own technical words, which an embedding can rank low. The union of the two is what the hybrid judge reads.

Offline and derived: the index is rebuilt in memory from the corpus texts (about a second for 10,000 abstracts), nothing is stored and no model is called. Words are lower-cased, stop words
(scikit-learn's English list) dropped and plurals folded ('platforms' -> 'platform'); there is no further stemming, so 'innovate' and 'innovation' stay different terms. A document is
scored by the sum, over the DISTINCT terms of the query, of the usual BM25 weight (k1 1.5, b 0.75).
"""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

K1, B = 1.5, 0.75
_WORD = re.compile(r"[a-z][a-z0-9]+")
_KEEP_S = ("ss", "us", "is")          # 'process', 'focus', 'analysis' are not plurals


def fold(word):
    """A word without its plural ending: 'strategies' -> 'strategy', 'platforms' -> 'platform'; 'process' and 'analysis' stay."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(_KEEP_S):
        return word[:-1]
    return word


def tokens(text):
    return [fold(w) for w in _WORD.findall(str(text or "").casefold()) if w not in ENGLISH_STOP_WORDS]


class Bm25:
    def __init__(self, texts):
        self.vectorizer = CountVectorizer(analyzer=tokens)
        counts = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        n = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        length = np.asarray(counts.sum(axis=1)).ravel()
        mean_length = float(length.mean()) if n and length.mean() > 0 else 1.0
        rows = np.repeat(np.arange(n), np.diff(counts.indptr))
        tf = counts.data
        counts.data = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * length[rows] / mean_length)) * idf[counts.indices]
        self.weights = sparse.csc_matrix(counts)
        self.size = n

    def scores(self, query):
        """The BM25 score of every document for `query`, in the order of the texts the index was built from (0 = no term in common)."""
        known = self.vectorizer.vocabulary_
        columns = sorted({known[t] for t in tokens(query) if t in known})
        if not columns:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self.weights[:, columns].sum(axis=1), dtype=np.float32).ravel()
