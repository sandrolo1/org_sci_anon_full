"""Offline known-source retrieval controls for the deployed novelty BM25 index.

Each query is mechanically extracted from an article already in the index, so
the article is a known relevant source. This tests retrieval plumbing and
robustness to shorter/erased wording. It cannot test substantive novelty,
paraphrase understanding, or the quality of a judge's comparison. No paper
text, identifiers, or row-level results are written to the aggregate output.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluation_integrity import _member
from .scoring_lexical import Bm25, tokens


CONTROL_NAMES = ("full_abstract", "title_only", "lead_40_words", "middle_20_words",
                 "lead_20_words", "lead_20_drop_rarest_quarter", "lead_20_drop_rarest_half",
                 "lead_20_drop_rarest_three_quarters")
SEMANTIC_CONTROLS = ("lead_20_words", "lead_20_drop_rarest_half", "lead_20_drop_rarest_three_quarters")
SEMANTIC_MODEL = "text-embedding-3-small"
SEMANTIC_MAX_INPUTS = 300


def load_catalog(export):
    """Read exactly the novelty corpus member of a frozen private export."""
    with zipfile.ZipFile(export) as archive:
        content = _member(archive, "catalog_novelty_corpus.csv")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    required = {"eid", "title", "abstract"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("novelty corpus needs eid, title, and abstract columns")
    ids = [row["eid"] for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("novelty corpus IDs must be unique and nonempty")
    return rows


def _abstract_key(abstract):
    return re.sub(r"\W+", "", str(abstract or "").casefold())


def select_sources(rows, *, sample_size=1000, seed="novelty-known-answer-v1"):
    """Hash-order a declared eligible set, independent of CSV row order."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    counts = Counter(_abstract_key(row.get("abstract")) for row in rows)
    eligible = [row for row in rows if row.get("title") and len(str(row.get("abstract") or "").split()) >= 80
                and counts[_abstract_key(row.get("abstract"))] == 1]
    if len(eligible) < sample_size:
        raise ValueError(f"only {len(eligible)} unique abstracts have at least 80 words")
    ordered = sorted(eligible, key=lambda row: (
        hashlib.sha256(f"{seed}\x1f{row['eid']}".encode()).hexdigest(), row["eid"]))
    return ordered[:sample_size], {"catalog_rows": len(rows), "eligible_unique_abstracts": len(eligible),
                                   "excluded_short_missing_or_duplicate": len(rows) - len(eligible)}


def controls(row, document_frequency, vocabulary, total_docs):
    """Generate extractive queries; erasure deliberately removes lexical clues."""
    abstract = str(row["abstract"])
    words = abstract.split()
    lead = " ".join(words[:40])
    middle = " ".join(words[max(0, (len(words) - 20) // 2):][:20])
    lead_terms = tokens(" ".join(words[:20]))
    unique_terms = set(lead_terms)
    # Smaller document frequency means a rarer, more discriminating term.
    rarest = sorted(unique_terms, key=lambda term: (
        document_frequency[vocabulary[term]] if term in vocabulary else total_docs + 1, term))
    queries = {"full_abstract": abstract, "title_only": str(row["title"]),
               "lead_40_words": lead, "middle_20_words": middle,
               "lead_20_words": " ".join(words[:20])}
    for label, fraction in (("quarter", 0.25), ("half", 0.5), ("three_quarters", 0.75)):
        remove_count = min(len(unique_terms) - 1, math.ceil(fraction * len(unique_terms)))
        removed = set(rarest[:remove_count])
        queries[f"lead_20_drop_rarest_{label}"] = " ".join(term for term in lead_terms if term not in removed)
    return queries


def benchmark(rows, *, sample_size=1000, seed="novelty-known-answer-v1"):
    """Return only aggregate retrieval results, with no private row content."""
    ids = [str(row.get("eid") or "") for row in rows]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("catalog IDs must be unique and nonempty")
    samples, coverage = select_sources(rows, sample_size=sample_size, seed=seed)
    # This is the same title + abstract recipe and BM25 implementation that
    # the deployed hybrid novelty retrieval uses.
    index = Bm25([f"{row.get('title') or ''} {row.get('abstract') or ''}" for row in rows])
    frequency = np.diff(index.weights.indptr)
    vocabulary = index.vectorizer.vocabulary_
    id_to_index = {value: i for i, value in enumerate(ids)}
    ids_for_ties = np.asarray(ids)
    outcomes = {name: [] for name in CONTROL_NAMES}
    for row in samples:
        target = id_to_index[row["eid"]]
        for name, query in controls(row, frequency, vocabulary, len(rows)).items():
            scores = index.scores(query)
            if scores[target] <= 0:
                outcomes[name].append(None)
                continue
            # The secondary ID sort prevents CSV row order from deciding ties.
            ordered = np.lexsort((ids_for_ties, -scores))
            outcomes[name].append(int(np.flatnonzero(ordered == target)[0]) + 1)
    metrics = {}
    for name, ranks in outcomes.items():
        found = [rank for rank in ranks if rank is not None]
        metrics[name] = {"n": len(ranks), "positive_source_score": len(found),
                         "source_at_1": sum(rank == 1 for rank in found),
                         "source_at_5": sum(rank <= 5 for rank in found),
                         "source_at_20": sum(rank <= 20 for rank in found),
                         "median_source_rank_when_scored": statistics.median(found) if found else None}
    return {"method": "known-source-bm25-controls-v1", "sample_seed": seed,
            "sample_size": sample_size, "coverage": coverage, "metrics": metrics,
            "interpretation": "Known-source recovery from mechanically extracted text in the indexed article. This is an in-corpus retrieval smoke test, not a paraphrase or substantive novelty accuracy benchmark."}


def _semantic_queries(rows, index, *, sample_size, seed):
    samples, _ = select_sources(rows, sample_size=sample_size, seed=seed)
    frequency = np.diff(index.weights.indptr)
    vocabulary = index.vectorizer.vocabulary_
    return [(row["eid"], name, controls(row, frequency, vocabulary, len(rows))[name])
            for row in samples for name in SEMANTIC_CONTROLS]


def _validate_corpus(rows, corpus_dir):
    """Refuse a vector index from a different model or text snapshot."""
    directory = Path(corpus_dir)
    manifest = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    if manifest.get("embedder") != f"openai:{SEMANTIC_MODEL}":
        raise ValueError("vector corpus model does not match the frozen query model")
    entries = [json.loads(line) for line in (directory / "entries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    if len(entries) != len(rows) or vectors.shape != (len(rows), 1536):
        raise ValueError("vector corpus shape does not match the exported novelty corpus")
    by_id = {row["eid"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("export IDs must be unique")
    for entry in entries:
        row = by_id.get(entry.get("id"))
        if row is None or entry.get("text_sha256") != hashlib.sha256(row["abstract"].encode()).hexdigest():
            raise ValueError("vector corpus does not match the exported abstracts")
    return entries, np.asarray(vectors, dtype=np.float32), manifest


def _unit_rows(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("query embeddings must be a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("query embeddings must be nonzero")
    return values / norms


def _query_vectors(queries, *, cache_path, source_hash, allow_api):
    """Read a private cache or make exactly one capped embedding request."""
    cache_path = Path(cache_path)
    if not queries or len(queries) > SEMANTIC_MAX_INPUTS:
        raise ValueError(f"comparison requires 1 to {SEMANTIC_MAX_INPUTS} query inputs")
    words = [len(text.split()) for _, _, text in queries]
    if any(count < 1 or count > 40 for count in words) or sum(words) > 6000:
        raise ValueError("query payload exceeds 40 words per input or 6,000 words total")
    query_hash = hashlib.sha256(json.dumps([text for _, _, text in queries], ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["source_sha256"]) != source_hash or str(cache["query_sha256"]) != query_hash:
                raise ValueError("embedding cache belongs to another source or query set")
            vectors = _unit_rows(cache["vectors"])
            usage = int(cache["input_tokens"])
        if vectors.shape != (len(queries), 1536):
            raise ValueError("cached embedding matrix has the wrong shape")
        return vectors, {"source": "verified_private_cache", "input_tokens": usage,
                         "query_sha256": query_hash, "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()}
    if not allow_api:
        raise ValueError("private embedding cache absent; pass --allow-api for one capped OpenAI request")
    from openai import OpenAI
    response = OpenAI(max_retries=0, timeout=60).embeddings.create(
        model=SEMANTIC_MODEL, input=[text for _, _, text in queries])
    ordered = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(len(queries))):
        raise ValueError("provider returned missing or duplicate embedding indexes")
    vectors = _unit_rows([item.embedding for item in ordered])
    if vectors.shape != (len(queries), 1536):
        raise ValueError("provider embedding dimension differs from frozen corpus")
    usage = int(response.usage.total_tokens) if response.usage else -1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, vectors=vectors, source_sha256=source_hash,
                        query_sha256=query_hash, input_tokens=usage)
    return vectors, {"source": "one_capped_api_request_then_private_cache", "input_tokens": usage,
                     "query_sha256": query_hash, "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest()}


def compare_methods(rows, entries, corpus_vectors, query_vectors, *, sample_size=100,
                    seed="novelty-known-answer-v1"):
    """Evaluate semantic top five, lexical top five, and deployed five-plus-five union."""
    if sample_size * len(SEMANTIC_CONTROLS) > SEMANTIC_MAX_INPUTS:
        raise ValueError(f"comparison permits at most {SEMANTIC_MAX_INPUTS} query vectors")
    ids = [entry["id"] for entry in entries]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise ValueError("vector entries must have the same number of unique IDs as catalog rows")
    by_id = {row["eid"]: row for row in rows}
    if set(ids) != set(by_id):
        raise ValueError("vector IDs differ from the catalog IDs")
    vectors = _unit_rows(corpus_vectors)
    queries_matrix = _unit_rows(query_vectors)
    if vectors.shape[0] != len(entries) or queries_matrix.shape != (sample_size * len(SEMANTIC_CONTROLS), vectors.shape[1]):
        raise ValueError("query/corpus vector shape mismatch")
    index = Bm25([f"{entry['title']}. {by_id[entry['id']]['abstract']}" for entry in entries])
    queries = _semantic_queries(rows, index, sample_size=sample_size, seed=seed)
    positions = {value: i for i, value in enumerate(ids)}
    outcomes = {name: Counter() for name in SEMANTIC_CONTROLS}
    semantic_matrix = queries_matrix @ vectors.T
    for (source_id, name, query), semantic_scores in zip(queries, semantic_matrix):
        target = positions[source_id]
        lexical_scores = index.scores(query)
        lexical_order = np.argsort(-lexical_scores, kind="stable")
        semantic_order = np.argsort(-semantic_scores, kind="stable")
        sem_five = {int(i) for i in semantic_order[:5]}
        lex_five = {int(i) for i in lexical_order[:5] if lexical_scores[i] > 0}
        extras = []
        for i in lexical_order:
            if lexical_scores[i] <= 0 or len(extras) >= 5:
                break
            if int(i) not in sem_five:
                extras.append(int(i))
        found = outcomes[name]
        found["n"] += 1
        found["lexical_at_5"] += target in lex_five
        found["semantic_at_5"] += target in sem_five
        found["hybrid_5_plus_5"] += target in sem_five or target in extras
        found["lexical_at_20"] += target in set(int(i) for i in lexical_order[:20] if lexical_scores[i] > 0)
        found["semantic_at_20"] += target in set(int(i) for i in semantic_order[:20])
        found["semantic_only_at_5"] += target in sem_five and target not in lex_five
        found["lexical_added_to_hybrid"] += target in extras
    return {name: dict(counts) for name, counts in outcomes.items()}


def semantic_comparison(export, corpus_dir, cache_path, *, sample_size=100,
                        seed="novelty-known-answer-v1", allow_api=False):
    """Same-query comparison; only aggregate values leave this function."""
    rows = load_catalog(export)
    entries, corpus_vectors, manifest = _validate_corpus(rows, corpus_dir)
    source_hash = hashlib.sha256(Path(export).read_bytes()).hexdigest()
    by_id = {row["eid"]: row for row in rows}
    index = Bm25([f"{entry['title']}. {by_id[entry['id']]['abstract']}" for entry in entries])
    queries = _semantic_queries(rows, index, sample_size=sample_size, seed=seed)
    if len(queries) > SEMANTIC_MAX_INPUTS:
        raise ValueError(f"at most {SEMANTIC_MAX_INPUTS} query embeddings are allowed")
    query_vectors, cache = _query_vectors(queries, cache_path=cache_path, source_hash=source_hash, allow_api=allow_api)
    metrics = compare_methods(rows, entries, corpus_vectors, query_vectors, sample_size=sample_size, seed=seed)
    return {"method": "known-source-semantic-lexical-hybrid-v1", "source_zip_sha256": source_hash,
            "corpus_id": manifest["corpus_id"], "corpus_model": manifest["embedder"],
            "sample_seed": seed, "sample_size": sample_size, "query_inputs": len(queries),
            "query_embedding": cache, "metrics": metrics,
            "interpretation": "Same in-corpus extractive queries across frozen semantic and deployed lexical search. Hybrid is semantic top five plus up to five new lexical papers. Known-source recovery is not paraphrase or substantive novelty accuracy."}


def frozen_vector_self_check(rows, corpus_dir, *, sample_size=1000,
                             seed="novelty-known-answer-v1"):
    """Offline alignment check: retrieve each sampled paper with its own stored vector.

    This deliberately uses the original paper embedding. It cannot estimate
    semantic recovery for truncated or reworded queries.
    """
    entries, corpus_vectors, manifest = _validate_corpus(rows, corpus_dir)
    samples, _ = select_sources(rows, sample_size=sample_size, seed=seed)
    vectors = _unit_rows(corpus_vectors)
    positions = {entry["id"]: i for i, entry in enumerate(entries)}
    source_positions = [positions[row["eid"]] for row in samples]
    ranks = []
    for start in range(0, len(source_positions), 100):
        part = source_positions[start:start + 100]
        scores = vectors[part] @ vectors.T
        for source, row in zip(part, scores):
            order = np.argsort(-row, kind="stable")
            ranks.append(int(np.flatnonzero(order == source)[0]) + 1)
    return {"method": "frozen-vector-self-alignment-v1", "corpus_id": manifest["corpus_id"],
            "corpus_model": manifest["embedder"], "sample_seed": seed, "n": len(ranks),
            "source_at_1": sum(rank == 1 for rank in ranks),
            "source_at_5": sum(rank <= 5 for rank in ranks),
            "source_at_20": sum(rank <= 20 for rank in ranks),
            "interpretation": "Offline index alignment using each paper's own stored title-plus-abstract vector. This is not a query-embedding, truncation, paraphrase, or novelty test."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", default="novelty-known-answer-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-corpus", type=Path, help="Optional frozen 10,050-vector corpus directory")
    parser.add_argument("--semantic-cache", type=Path, help="Private ignored NPZ cache for query embeddings")
    parser.add_argument("--semantic-sample-size", type=int, default=100)
    parser.add_argument("--allow-api", action="store_true", help="Permit one capped paid query-embedding request when cache is absent")
    args = parser.parse_args(argv)
    report = benchmark(load_catalog(args.export), sample_size=args.sample_size, seed=args.seed)
    report["source_zip_sha256"] = hashlib.sha256(args.export.read_bytes()).hexdigest()
    if args.semantic_cache and not args.semantic_corpus:
        parser.error("--semantic-cache requires --semantic-corpus")
    if args.allow_api and not args.semantic_cache:
        parser.error("--allow-api requires --semantic-cache")
    if args.semantic_corpus:
        report["vector_self_check"] = frozen_vector_self_check(
            load_catalog(args.export), args.semantic_corpus, sample_size=args.sample_size, seed=args.seed)
        if args.semantic_cache:
            report["semantic_comparison"] = semantic_comparison(
                args.export, args.semantic_corpus, args.semantic_cache,
                sample_size=args.semantic_sample_size, seed=args.seed, allow_api=args.allow_api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
