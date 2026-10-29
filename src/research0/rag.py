"""Phase 8: hybrid retrieval over a validated release, read-only for discovery agents.

A discovery agent never writes to the research database. It connects as the PostgreSQL role `research_agent` (SELECT everywhere,
INSERT only on its own retrieval log) through `reader_connection`, which also opens every transaction read-only. Retrieval is limited to
the frozen scope of one VALIDATED release: the chunks of that release's ingestion run. Each hit carries what a reader needs to check
it: paper, section, pages, character offsets and the chunk text. Every query is logged (retrieval_run, retrieval_hit) so that an answer
can be traced back to the passages that produced it.

Order of operations, as in the proposal: authorisation and community filters first (an agent profile fixes the community it may read), then full-text
rank and pgvector cosine distance fused by reciprocal rank (k = 60), then an optional reranker, then a diversity rule that keeps one paper from filling
the context (at most `max_per_paper` chunks per paper). Neither score means "relevant" by itself; with a lexical embedding (the offline stand-in) the
vector side only matches shared words.
"""
import json
import logging
import os
import uuid

import psycopg

from . import release as release_module
from .embeddings import to_vector_literal

LOG = logging.getLogger("research0")
RRF_K = 60
CANDIDATES = 50


def reader_connection(url=None):
    """A connection whose transactions are read-only. Use the research_agent role in the URL for a real guarantee."""
    con = psycopg.connect(url or os.environ["AGENT_DATABASE_URL"])
    con.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    con.commit()
    return con


def release_scope(con, release_id, allow_unvalidated=False):
    """The frozen scope of a release. Discovery agents read only a validated one, unless the caller explicitly accepts a release that is not (the forum does,
    on the user's decision, and tags every such episode)."""
    if not allow_unvalidated and release_module.status(con, release_id) != "validated":
        raise ValueError("Only a validated release can be read by discovery agents (all gates pass and a person signed it off)")
    manifest = con.execute("SELECT manifest FROM release WHERE release_id=%s", (release_id,)).fetchone()[0]
    ingestion = manifest["runs"].get("ingestion_run_id")
    if not ingestion:
        raise ValueError("The release has no document ingestion run to search")
    return uuid.UUID(ingestion), manifest


def _filters(sections, papers):
    return ("AND c.section = ANY(%(sections)s)" if sections else "") + (" AND v.canonical_id = ANY(%(papers)s)" if papers is not None else "")


def _lexical(con, query, ingestion_id, sections, n, papers=None):
    return con.execute(
        "SELECT c.chunk_id,ts_rank_cd(c.tsv, q) AS score FROM document_chunk c JOIN paper_version v USING (version_id), websearch_to_tsquery('english', %(q)s) q "
        f"WHERE v.ingestion_run_id=%(ing)s AND c.tsv @@ q {_filters(sections, papers)} ORDER BY score DESC, c.chunk_id LIMIT %(n)s",
        {"q": query, "ing": ingestion_id, "sections": sections, "n": n, "papers": papers}).fetchall()


def _vector(con, vector, model_id, ingestion_id, sections, n, papers=None):
    # The column has no fixed dimension; casting to the model's dimension lets the planner use the per-model index made by create_vector_index.
    dims = con.execute("SELECT dimensions FROM embedding_model WHERE model_id=%s", (model_id,)).fetchone()[0]
    return con.execute(
        f"SELECT c.chunk_id,(e.embedding::vector({int(dims)})) <=> %(v)s::vector({int(dims)}) AS distance FROM chunk_embedding e JOIN document_chunk c USING (chunk_id) JOIN paper_version v USING (version_id) "
        f"WHERE v.ingestion_run_id=%(ing)s AND e.model_id=%(m)s {_filters(sections, papers)} ORDER BY distance, c.chunk_id LIMIT %(n)s",
        {"v": to_vector_literal(vector), "m": model_id, "ing": ingestion_id, "sections": sections, "n": n, "papers": papers}).fetchall()


def create_vector_index(con, model_id, kind="hnsw"):
    """An approximate-nearest-neighbour index for one embedding model (HNSW or IVFFlat). The proposal says to benchmark the choice on the real corpus: build
    one, compare latency and recall against the exact scan, keep the better. pgvector indexes vectors of up to 2000 dimensions."""
    import hashlib
    from psycopg import sql
    if kind not in ("hnsw", "ivfflat"):
        raise ValueError("kind must be hnsw or ivfflat")
    row = con.execute("SELECT dimensions FROM embedding_model WHERE model_id=%s", (model_id,)).fetchone()
    if not row:
        raise ValueError("Unknown embedding model")
    if row[0] > 2000:
        raise ValueError(f"pgvector cannot index {row[0]} dimensions with {kind}; use a smaller model or reduced dimensions")
    name = f"chunk_embedding_{kind}_{hashlib.md5(model_id.encode()).hexdigest()[:8]}"
    con.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON chunk_embedding USING " + kind + " ((embedding::vector({})) vector_cosine_ops) WHERE model_id = {}").format(
        sql.Identifier(name), sql.SQL(str(int(row[0]))), sql.Literal(model_id)))
    return name


def community_papers(con, manifest, community, include_v1=False):
    """Papers whose pre-assignment in the release is 'yes' for the community (the view an agent of that community reads)."""
    if not manifest["runs"].get("preassignment_run_id"):
        raise ValueError("The release has no pre-assignment run, so a community view cannot be built")
    from .confirmation import memberships
    return [cid for (cid, c), m in memberships(con, manifest["catalog_run_id"], include_v1).items() if c == community and m["label"] == "yes"]


def agent_profile(con, agent):
    row = con.execute("SELECT community,retrieval_policy FROM agent_profile WHERE agent_id=%s", (agent,)).fetchone()
    return {"community": row[0], **(row[1] or {})} if row else {}


def diversify(fused, owner, k, max_per_paper):
    """Keep the best-ranked chunks but at most max_per_paper per paper."""
    taken, per_paper = [], {}
    for item in fused:
        paper = owner[item[0]]
        if per_paper.get(paper, 0) < max_per_paper:
            taken.append(item)
            per_paper[paper] = per_paper.get(paper, 0) + 1
        if len(taken) == k:
            break
    return taken


def fuse(lexical, vector, k=10, rrf_k=RRF_K):
    """Reciprocal rank fusion. lexical: [(id, score)] best first; vector: [(id, distance)] best first. Deterministic (ties by id)."""
    parts = {}
    for rank, (cid, score) in enumerate(lexical, 1):
        parts.setdefault(cid, {})["lexical_rank"], parts[cid]["lexical_score"] = rank, float(score)
    for rank, (cid, distance) in enumerate(vector, 1):
        parts.setdefault(cid, {})["vector_rank"], parts[cid]["vector_distance"] = rank, float(distance)
    scored = []
    for cid, p in parts.items():
        total = sum(1 / (rrf_k + p[r]) for r in ("lexical_rank", "vector_rank") if r in p)
        scored.append((-total, str(cid), cid, total, p))
    scored.sort(key=lambda x: x[:2])
    return [(cid, total, p) for _, _, cid, total, p in scored[:k]]


def search(con, release_id, query, embedder, k=10, sections=None, agent="unnamed-agent", log=True, community=None, max_per_paper=2, reranker=None, allow_unvalidated=False, include_v1=False):
    """Hybrid retrieval over a validated release (or, with allow_unvalidated, over any release). `community` restricts the search to the papers pre-assigned 'yes' to it (a list such as ["CTO", "TIM"]
    reads the union of their views); if the agent has a profile its community and policy (max_per_paper) apply.
    `reranker(query, hits) -> {chunk_id: score}` reorders the fused candidates (none by default)."""
    query = (query or "").strip()
    if not query or len(query) > 1000 or not 1 <= k <= 50 or not 1 <= max_per_paper <= 50:
        raise ValueError("A query of up to 1000 characters, k between 1 and 50 and max_per_paper between 1 and 50 are required")
    ingestion_id, manifest = release_scope(con, release_id, allow_unvalidated)
    profile = agent_profile(con, agent)
    community = profile.get("community") or community
    max_per_paper = int(profile.get("max_per_paper", max_per_paper))
    papers = sorted({p for c in ([community] if isinstance(community, str) else community) for p in community_papers(con, manifest, c, include_v1)}) if community else None
    model = con.execute("SELECT 1 FROM embedding_model WHERE model_id=%s", (embedder.model_id,)).fetchone()
    lexical = _lexical(con, query, ingestion_id, sections, CANDIDATES, papers)
    vector = _vector(con, embedder.embed([query])[0], embedder.model_id, ingestion_id, sections, CANDIDATES, papers) if model else []
    fused = fuse(lexical, vector, CANDIDATES)
    owner = {cid: pid for cid, pid in con.execute("SELECT c.chunk_id,v.canonical_id FROM document_chunk c JOIN paper_version v USING (version_id) WHERE c.chunk_id = ANY(%s)", ([f[0] for f in fused],))}
    if reranker:
        scores = reranker(query, hydrate(con, fused))
        fused = sorted(((cid, total, {**parts, "rerank_score": float(scores.get(str(cid), 0.0))}) for cid, total, parts in fused), key=lambda x: (-x[2]["rerank_score"], str(x[0])))
    fused = diversify(fused, owner, k, max_per_paper)
    hits = hydrate(con, fused)
    con.commit()  # ends the read transaction so that the log can be written in a new one (an agent's connection holds nothing else)
    if log:
        log_retrieval(con, release_id, query, agent, {"k": k, "sections": sections, "embedding_model": embedder.model_id, "rrf_k": RRF_K, "vector_side": bool(model),
                                                       "max_per_paper": max_per_paper, "reranker": bool(reranker), "strategy": "filters -> fts+vector (rrf) -> rerank -> diversity"},
                      fused, {"community": community, "sections": sections})
    return hits


def hydrate(con, fused):
    ids = [cid for cid, _, _ in fused]
    info = {cid: (title, year, doi, authors, journal, section, ps, pe, cs, ce, text, canonical, version)
            for cid, title, year, doi, authors, journal, section, ps, pe, cs, ce, text, canonical, version in con.execute(
        "SELECT c.chunk_id,p.title,p.publication_year,p.doi,p.authors,p.source_title,c.section,c.page_start,c.page_end,c.char_start,c.char_end,c.text,v.canonical_id,v.version_id "
        "FROM document_chunk c JOIN paper_version v USING (version_id) JOIN canonical_paper p ON p.canonical_id=v.canonical_id WHERE c.chunk_id = ANY(%s)", (ids,))}
    hits = []
    for rank, (cid, score, components) in enumerate(fused, 1):
        title, year, doi, authors, journal, section, ps, pe, cs, ce, text, canonical, version = info[cid]
        hits.append({"rank": rank, "score": score, "components": components, "chunk_id": str(cid), "canonical_id": str(canonical), "version_id": str(version),
                     "title": title, "year": year, "doi": doi, "authors": authors, "journal": journal, "section": section, "pages": [ps, pe], "char_span": [cs, ce], "text": text})
    return hits


def log_retrieval(con, release_id, query, agent, params, fused, scope=None):
    """The only write an agent performs: its own retrieval log (allowed by the research_agent role)."""
    run_id = uuid.uuid4()
    con.execute("SET TRANSACTION READ WRITE")
    con.execute("INSERT INTO retrieval_run(retrieval_run_id,release_id,query_text,method,params,agent,scope) VALUES (%s,%s,%s,'hybrid-rrf',%s::jsonb,%s,%s::jsonb)",
                (run_id, release_id, query, json.dumps(params), agent[:80], json.dumps(scope or {})))
    with con.cursor() as cur:
        cur.executemany("INSERT INTO retrieval_hit(retrieval_run_id,rank,chunk_id,score,components) VALUES (%s,%s,%s,%s,%s::jsonb)",
                        [(run_id, rank, cid, score, json.dumps(parts)) for rank, (cid, score, parts) in enumerate(fused, 1)])
    con.commit()
    return run_id


def grounded_context(hits):
    """Numbered passages for a language model prompt, each with the citation a reader can follow back to the page."""
    blocks = [f"[{h['rank']}] {h['title']} ({h['year']}), {h['section']}, pp. {h['pages'][0]}-{h['pages'][1]}, chunk {h['chunk_id']}\n{h['text']}" for h in hits]
    return ("Answer only from the numbered passages below and cite them by number. If they do not contain the answer, say so.\n\n" + "\n\n".join(blocks))


# --- Other read-only views an agent may use ---------------------------------------------------------------------------------------

def settled_pairs(con, release_id, allow_unvalidated=False, provisional=False):
    """Settled pairs of the release. `provisional=True` (only the forum asks for it, explicitly) adds the pairs decided by a single reviewer with no final decision;
    each item says `provisional`. Contested pairs are never included."""
    _, manifest = release_scope(con, release_id, allow_unvalidated)
    signal_run_id = manifest["runs"].get("signal_run_id")
    if not signal_run_id:
        return []
    from . import pairs
    decided, single = {}, set()
    for pair_id, per in pairs.current(con, uuid.UUID(signal_run_id)).items():      # the current codebook only: decisions of the earlier one are history
        state, decision = pairs.state_of(per)
        if state == "settled":
            decided[pair_id] = decision
        elif provisional and state == "pending" and len(per) == 1:
            decided[pair_id] = next(iter(per.values()))[0]
            single.add(pair_id)
    info = pairs.details(con, list(decided))
    rows = con.execute("SELECT pair_id,content FROM pair_dossier WHERE pair_id = ANY(%s)", (list(decided),)).fetchall()
    con.commit()
    out = []
    for pid, content in rows:
        agreeing = [d for d in info.get(pid, []) if d["class"] == decided[pid]]          # the people whose decision is the one that counts
        first = (sorted(agreeing, key=lambda d: not d["final"]) or [{}])[0]
        out.append({"pair_id": str(pid), "class": decided[pid], "dossier": content, "provisional": pid in single,
                    "rationale": [d["rationale"] for d in agreeing if d["rationale"]],
                    "checklist": {k: first.get(k) for k in (*pairs.CHECKLIST, "theoretical_consequence", "counterexample")}})
    return out
