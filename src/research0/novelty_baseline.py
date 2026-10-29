"""Read-only abstract comparison for a proposal novelty audit.

This module ranks possible prior art; it does not decide that a question is
novel or redundant. The caller supplies a proposal abstract and paper vectors
from the same embedding model and text recipe, plus provenance for each paper.
"""

import hashlib
import re
from collections import Counter, defaultdict
from math import isfinite, log, sqrt


def _tokens(text):
    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


QUERY_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in into is it of on or our the their then there "
    "these this to under via was were what when where which who why with would "
    "research question puzzle mechanism prediction study paper evidence".split()
)


def question_card(content):
    """Make a reproducible search text from an existing structured question.

    The forum already stores these fields. Copying them verbatim avoids an
    extra generative abstract and makes the input auditable without a person
    rewriting it. Missing core fields cause an abstention by the caller.
    """
    if not isinstance(content, dict):
        raise ValueError("question content must be an object")
    fields = [("Research question", content.get("rq")),
              ("Puzzle", content.get("puzzle")),
              ("Mechanism", content.get("mechanism"))]
    if any(not isinstance(value, str) or not value.strip() for _, value in fields):
        raise ValueError("question, puzzle, and mechanism are required")
    predictions = content.get("predictions")
    if not isinstance(predictions, list) or not predictions or any(
            not isinstance(item, dict) or not isinstance(item.get("statement"), str) or
            not item["statement"].strip() for item in predictions):
        raise ValueError("at least one prediction statement is required")
    return "\n".join([f"{label}: {value.strip()}" for label, value in fields] +
                     [f"Prediction {i}: {item['statement'].strip()}" for i, item in enumerate(predictions, 1)])


def _unit(vector):
    values = tuple(float(value) for value in vector)
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("embedding must contain finite values")
    length = sqrt(sum(value * value for value in values))
    if length == 0:
        raise ValueError("embedding must not be zero")
    return tuple(value / length for value in values)


def compare_abstracts(proposal_vector, papers, *, top_k=20, max_publication_year=None):
    """Rank the whole corpus and its cluster centroids against one proposal.

    ``papers`` contains dictionaries with id, title, doi, year, abstract,
    source, vector, and cluster. All vectors must come from the same model and recipe.
    Missing abstracts and vectors are counted but excluded from similarity.
    Centroids average unit paper vectors and are then normalized. The nearest
    paper is searched globally, not only inside the closest cluster.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    query = _unit(proposal_vector)
    ranked = []
    clusters = defaultdict(list)
    seen = set()
    counts = {"input": 0, "after_year_filter": 0, "without_abstract": 0,
              "without_vector": 0, "without_cluster": 0, "compared": 0}
    for paper in papers:
        counts["input"] += 1
        year = paper.get("year")
        if max_publication_year is not None and (year is None or year > max_publication_year):
            continue
        counts["after_year_filter"] += 1
        paper_id = str(paper["id"])
        if paper_id in seen:
            raise ValueError(f"duplicate canonical paper id: {paper_id}")
        seen.add(paper_id)
        abstract = str(paper.get("abstract") or "").strip()
        if not abstract or abstract.lower() == "[no abstract available]":
            counts["without_abstract"] += 1
            continue
        if paper.get("vector") is None:
            counts["without_vector"] += 1
            continue
        vector = _unit(paper["vector"])
        if len(vector) != len(query):
            raise ValueError(f"embedding dimensions differ for paper {paper_id}")
        similarity = sum(a * b for a, b in zip(query, vector))
        cluster = paper.get("cluster")
        hit = {"id": paper_id, "title": paper.get("title"), "doi": paper.get("doi"),
               "year": year, "source": paper.get("source"), "abstract": abstract,
               "cluster": cluster, "cosine_similarity": round(similarity, 6),
               "_sort_similarity": similarity}
        ranked.append(hit)
        if cluster is not None:
            clusters[cluster].append(vector)
        else:
            counts["without_cluster"] += 1
        counts["compared"] += 1
    ranked.sort(key=lambda hit: (-hit["_sort_similarity"], hit["id"]))
    for hit in ranked:
        del hit["_sort_similarity"]
    centroid_hits = []
    for label, vectors in clusters.items():
        mean = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(query)))
        try:
            centroid = _unit(mean)
        except ValueError:
            continue
        similarity = sum(a * b for a, b in zip(query, centroid))
        centroid_hits.append({"cluster": label, "papers": len(vectors),
                              "cosine_similarity": round(similarity, 6)})
    centroid_hits.sort(key=lambda hit: (-hit["cosine_similarity"], str(hit["cluster"])))
    return {"method": "abstract-neighbor-centroid-v1", "counts": counts,
            "nearest_papers": ranked[:top_k], "nearest_clusters": centroid_hits,
            "max_publication_year": max_publication_year,
            "interpretation": "Retrieval candidates for prior-art review; similarity is not a novelty score."}


def compare_catalog(con, run_id, cluster_run_id, embedder, proposal_text, *, lexical_queries=None, top_k=20):
    """Read the current catalog embeddings and return a provenance-tagged packet.

    The caller prepares ``proposal_text`` with the same documented recipe as
    the paper embeddings. This function only reads the database; it never
    generates an abstract, embeds papers, or writes an evaluation.
    """
    from . import explore

    row = con.execute("SELECT run_id,model_id FROM cluster_run WHERE cluster_run_id=%s",
                      (cluster_run_id,)).fetchone()
    if not row or str(row[0]) != str(run_id) or row[1] != embedder.model_id:
        raise ValueError("cluster run must belong to this catalog run and embedding model")
    text = proposal_text.strip()
    if not text:
        raise ValueError("proposal text must not be empty")
    ids, vectors, _, has_abstract = explore.load_vectors(con, run_id, embedder.model_id)
    metadata = {str(cid): (title, doi, year, source, abstract) for cid, title, doi, year, source, abstract in con.execute(
        "SELECT c.canonical_id,c.title,c.doi,c.publication_year,c.source_title,r.raw_fields->>'Abstract' "
        "FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE c.run_id=%s AND c.status='retained' AND cardinality(c.perimeter_flags)=0", (run_id,)).fetchall()}
    assignments = {str(cid): label for cid, label in con.execute(
        "SELECT canonical_id,cluster_label FROM cluster_assignment WHERE cluster_run_id=%s AND canonical_id = ANY(%s)",
        (cluster_run_id, ids)).fetchall()}
    papers = []
    for cid, vector, has in zip(ids, vectors, has_abstract):
        title, doi, year, source, abstract = metadata[str(cid)]
        papers.append({"id": str(cid), "title": title, "doi": doi, "year": year,
                       "source": source, "abstract": abstract if has else None, "vector": vector,
                       "cluster": assignments.get(str(cid))})
    result = compare_abstracts(embedder.embed([text])[0], papers, top_k=top_k)
    result["coverage"] = {"eligible_papers": len(metadata), "papers_with_current_embedding": len(ids),
                          "papers_without_current_embedding": len(metadata) - len(ids)}
    if lexical_queries is not None:
        lexical_papers = [{"id": cid, "title": title, "doi": doi, "year": year,
                           "source": source, "abstract": abstract}
                          for cid, (title, doi, year, source, abstract) in metadata.items()]
        result["lexical_papers"] = lexical_candidates(lexical_queries, lexical_papers, top_k=top_k)
        result["combined_candidates"] = fuse_candidates(result["nearest_papers"], result["lexical_papers"], top_k=top_k)
    result["provenance"] = {"catalog_run_id": str(run_id), "cluster_run_id": str(cluster_run_id),
                            "embedding_model": embedder.model_id,
                            "embedding_kind": getattr(embedder, "kind", "unknown"),
                            "proposal_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return result


def lexical_candidates(queries, papers, *, top_k=20):
    """Rank title/abstract matches for explicit query variants with BM25.

    ``queries`` should contain the original question and separately recorded
    variants made from *adjudicated* equivalent construct labels. This search
    includes title-only papers, but marks them as such. It is a retrieval aid,
    never evidence that a question is answered.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    variants = [(str(query).strip(), set(_tokens(query)) - QUERY_STOPWORDS) for query in queries]
    variants = [(text, tokens) for text, tokens in variants if tokens]
    if not variants:
        raise ValueError("at least one nonempty query is required")
    documents = []
    seen = set()
    for paper in papers:
        paper_id = str(paper["id"])
        if paper_id in seen:
            raise ValueError(f"duplicate canonical paper id: {paper_id}")
        seen.add(paper_id)
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if abstract.lower() == "[no abstract available]":
            abstract = ""
        tokens = _tokens(title + " " + abstract)
        if tokens:
            documents.append((paper, abstract, tokens, Counter(tokens)))
    if not documents:
        return []
    n = len(documents)
    avg_length = sum(len(tokens) for _, _, tokens, _ in documents) / n
    frequency = Counter(token for _, _, tokens, _ in documents for token in set(tokens))
    results = []
    for paper, abstract, tokens, term_counts in documents:
        variant_scores = []
        for query_text, query_terms in variants:
            score = 0.0
            for term in query_terms:
                count = term_counts[term]
                if not count:
                    continue
                idf = log(1 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * len(tokens) / avg_length))
            if score > 0:
                variant_scores.append((query_text, score))
        if variant_scores:
            score = max(value for _, value in variant_scores)
            results.append({"id": str(paper["id"]), "title": paper.get("title"),
                            "doi": paper.get("doi"), "year": paper.get("year"),
                            "source": paper.get("source"), "abstract": abstract or None,
                            "evidence_available": "abstract" if abstract else "title_only",
                            "bm25_score": round(score, 6),
                            "matched_queries": [text for text, _ in variant_scores],
                            "_sort_score": score})
    results.sort(key=lambda hit: (-hit["_sort_score"], hit["id"]))
    for hit in results:
        del hit["_sort_score"]
    return results[:top_k]


def fuse_candidates(semantic_hits, lexical_hits, *, top_k=20, rrf_k=60):
    """Merge two ranked candidate lists, retaining each paper's source rank."""
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k and rrf_k must be positive")
    merged = {}
    for method, hits in (("vector", semantic_hits), ("lexical", lexical_hits)):
        for rank, hit in enumerate(hits, 1):
            paper_id = str(hit["id"])
            entry = merged.setdefault(paper_id, {"id": paper_id, "title": hit.get("title"),
                                                 "doi": hit.get("doi"), "year": hit.get("year"),
                                                 "source": hit.get("source"), "abstract": hit.get("abstract"),
                                                 "ranks": {}, "matched_queries": []})
            if method in entry["ranks"]:
                raise ValueError(f"duplicate {method} hit: {paper_id}")
            entry["ranks"][method] = rank
            if method == "lexical":
                entry["matched_queries"] = hit.get("matched_queries", [])
            if not entry["abstract"] and hit.get("abstract"):
                entry["abstract"] = hit["abstract"]
    for entry in merged.values():
        score = sum(1 / (rrf_k + rank) for rank in entry["ranks"].values())
        entry["rrf_score"] = round(score, 8)
        entry["_sort_score"] = score
    result = sorted(merged.values(), key=lambda hit: (-hit["_sort_score"], hit["id"]))[:top_k]
    for entry in result:
        del entry["_sort_score"]
    return result
