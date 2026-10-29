"""Embedders. An embedding is a derived, rebuildable representation used for retrieval and exploration only.

No embedding score decides membership or equivalence of constructs (the internal project notes, principle 3).
Embeddings are computed on what the proposal says they are for: the textual evidence of constructs (definitions, items, measures) for S_sem-op, and the
chunks of the full texts for retrieval. The default is OpenAI `text-embedding-3-small` (EMBEDDING_MODEL=openai:<model> to change it; it sends the
text to OpenAI, costs money and needs OPENAI_API_KEY). `hash-rp-v1` is an offline, deterministic and LEXICAL stand-in (hashed word 1-2 grams, random
projection): free and private, used by the tests, the demo and the optional abstract-level exploration, but it does not know that two different words
mean the same thing.
"""
import os

import numpy as np

DEFAULT_MODEL = "hash-rp-v1"          # the offline stand-in
OPENAI_DEFAULT = "openai:text-embedding-3-small"


class Embedder:
    model_id = ""
    kind = "lexical"
    dimensions = 0
    description = ""

    def embed(self, texts):
        raise NotImplementedError

    parameters = {}

    def register(self, con):
        import json
        con.execute("INSERT INTO embedding_model(model_id,kind,dimensions,description,normalized,parameters) VALUES (%s,%s,%s,%s,true,%s::jsonb) ON CONFLICT DO NOTHING",
                    (self.model_id, self.kind, self.dimensions, self.description, json.dumps(self.parameters)))


class HashEmbedder(Embedder):
    model_id = DEFAULT_MODEL
    kind = "lexical"
    dimensions = 256
    description = "Hashed word 1-2 grams (sublinear tf, English stop words removed) projected to 256 dimensions; offline and lexical."

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.random_projection import SparseRandomProjection
        self.vectorizer = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), stop_words="english", alternate_sign=False,
                                            norm=None, lowercase=True, token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
        self.projection = SparseRandomProjection(n_components=self.dimensions, dense_output=True, random_state=20261010)
        self.projection.fit(self.vectorizer.transform(["warm up"]))  # depends only on the number of features, so it is reproducible

    def embed(self, texts):
        matrix = self.vectorizer.transform(list(texts))
        matrix.data = np.log1p(matrix.data)
        dense = np.asarray(self.projection.transform(matrix), dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


class OpenAIEmbedder(Embedder):
    kind = "semantic"

    def __init__(self, model, batch=256):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for openai embeddings")
        from openai import OpenAI
        self.client, self.batch = OpenAI(max_retries=6), batch
        self.model_id = "openai:" + model
        self.model = model
        self.description = f"OpenAI {model} embeddings (text is sent to OpenAI)"
        self.parameters = {"provider": "openai", "model": model, "input_truncation_chars": 30000}
        self.dimensions = len(self._call(["dimension probe"])[0])

    def _call(self, texts):
        response = self.client.embeddings.create(model=self.model, input=[t[:30000] or " " for t in texts])
        return [item.embedding for item in response.data]

    def embed(self, texts):
        texts, out = list(texts), []
        for i in range(0, len(texts), self.batch):
            out.extend(self._call(texts[i:i + self.batch]))
        dense = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms


def get_embedder(name=None, offline_default=False):
    """EMBEDDING_MODEL (or `name`) wins; otherwise OpenAI text-embedding-3-small, or the offline stand-in when offline_default is set."""
    name = name or os.environ.get("EMBEDDING_MODEL") or (DEFAULT_MODEL if offline_default else OPENAI_DEFAULT)
    if name == DEFAULT_MODEL:
        return HashEmbedder()
    if name.startswith("openai:") and len(name) > 7:
        return OpenAIEmbedder(name[7:])
    raise ValueError(f"Unknown embedding model {name!r}; use {OPENAI_DEFAULT}, another openai:<model>, or {DEFAULT_MODEL}")


def to_vector_literal(row):
    return "[" + ",".join(f"{x:.7g}" for x in row) + "]"


def cosine(a, b):
    """Cosine of two L2-normalised vectors, clipped to [0, 1] (negative similarity carries no meaning for text embeddings here)."""
    return float(max(0.0, min(1.0, np.dot(a, b))))
