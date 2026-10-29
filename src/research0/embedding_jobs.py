"""Embedding jobs for the corpus the discovery agents read: the chunks of the PDFs (chunk_embedding) and the records of the construct database
(record_embedding). Both go to PostgreSQL + pgvector, are derived and rebuildable, and are never a source of record.

The jobs are made for a long, paid run: they commit after every batch (an interruption loses at most one batch and a rerun continues where it
stopped), they size batches by real tokens (tiktoken, the tokenizer of the text-embedding-3 models) so that no request nears the API limits, and
`dry_run` counts everything without calling the model. OpenAI embeddings SEND the texts to OpenAI and cost money.
"""
import json
import logging
import time

from .core import digest
from .embeddings import to_vector_literal

LOG = logging.getLogger("research0")
RECIPE = "records-v1"
MAX_ITEMS = 128                 # texts per request
MAX_BATCH_TOKENS = 120_000      # the API accepts up to 300k tokens per request; stay well below
MAX_TEXT_TOKENS = 8000          # the model's input limit is 8191


def _encoder():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception:                                  # no tokenizer file reachable: a cautious estimate, flagged in the report
        return None


def _clip(text, count):
    """Texts longer than the model accepts are cut (the stored chunks are at most ~900 estimated tokens, so this is a safety net)."""
    if count and count(text) > MAX_TEXT_TOKENS:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(enc.encode(text, disallowed_special=())[:MAX_TEXT_TOKENS])
    return text


# --- What is embedded for each kind of record --------------------------------------------------------------------------------------

def _sql_records(kind):
    """(SQL selecting record_id, extraction_run_id, canonical_id and the text parts, for one extraction run)"""
    if kind == "mention":
        return ("SELECT m.mention_id,m.extraction_run_id,m.canonical_id,m.label_normalized,m.role,m.definition_text,m.quote FROM construct_mention m WHERE m.extraction_run_id=%(run)s")
    if kind == "measure":
        return ("SELECT x.measure_id,m.extraction_run_id,m.canonical_id,m.label_normalized,x.name,x.items_text,x.scale,x.quote FROM construct_measure x JOIN construct_mention m USING (mention_id) "
                "WHERE m.extraction_run_id=%(run)s")
    if kind == "claim":
        return ("SELECT c.claim_id,c.extraction_run_id,c.canonical_id,s.label_normalized,c.relation,o.label_normalized,c.direction,c.evidence_status,c.quote FROM nomological_claim c "
                "JOIN construct_mention s ON s.mention_id=c.subject_mention_id JOIN construct_mention o ON o.mention_id=c.object_mention_id WHERE c.extraction_run_id=%(run)s")
    return ("SELECT a.attribute_id,m.extraction_run_id,m.canonical_id,m.label_normalized,a.kind,a.basis,a.value,a.quote FROM extraction_attribute a JOIN construct_mention m USING (mention_id) "
            "WHERE m.extraction_run_id=%(run)s AND a.basis <> 'missing' AND a.value IS NOT NULL AND a.quote IS NOT NULL")


def _text(kind, row):
    p = [("" if v is None else str(v)).strip() for v in row[3:]]
    if kind == "mention":
        label, role, definition, quote = p
        return f"Construct: {label} (role: {role}). Definition: {definition or 'not given'}. Evidence: {quote}"
    if kind == "measure":
        label, name, items, scale, quote = p
        return f"Measure of {label}: {name}. Items: {items or 'not given'}. Scale: {scale or 'not given'}. Evidence: {quote}"
    if kind == "claim":
        subject, relation, obj, direction, status, quote = p
        return f"Claim: {subject} {relation.replace('_', ' ')} {obj} (direction: {direction}; evidence status: {status}). Evidence: {quote}"
    label, attribute, basis, value, quote = p
    return f"Attribute of {label}: {attribute.replace('_', ' ')} = {value} ({basis}). Evidence: {quote}"


KINDS = ("mention", "measure", "claim", "attribute")


def record_texts(con, extraction_run_id, kind, model_id=None, only_pending=True):
    """[(record_id, extraction_run_id, canonical_id, text)] for one kind of record, skipping those already embedded with `model_id`."""
    done = set()
    if only_pending and model_id:
        done = {r[0] for r in con.execute("SELECT record_id FROM record_embedding WHERE record_kind=%s AND model_id=%s AND extraction_run_id=%s", (kind, model_id, extraction_run_id))}
    rows = con.execute(_sql_records(kind) + " ORDER BY 1", {"run": extraction_run_id}).fetchall()
    return [(r[0], r[1], r[2], _text(kind, r)) for r in rows if r[0] not in done]


# --- Counting without calling the model --------------------------------------------------------------------------------------------

def dry_run(con, ingestion_run_id, extraction_run_id, model_id):
    """What a run would send: counts, characters and tokens per source. Calls nothing."""
    count = _encoder()
    out = {"tokenizer": "tiktoken cl100k_base" if count else "unavailable: tokens estimated as characters / 3", "model": model_id, "sources": {}}

    def add(name, texts):
        chars = sum(len(t) for t in texts)
        tokens = sum(count(t) for t in texts) if count else chars // 3
        out["sources"][name] = {"texts": len(texts), "characters": chars, "tokens": tokens, "max_tokens": max((count(t) for t in texts), default=0) if count else None}
    chunks = [t for _, t in con.execute("SELECT c.chunk_id,c.text FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s "
                                        "AND NOT EXISTS (SELECT 1 FROM chunk_embedding e WHERE e.chunk_id=c.chunk_id AND e.model_id=%s)", (ingestion_run_id, model_id))]
    add("chunks (PDF text)", chunks)
    for kind in KINDS:
        add(f"records: {kind}", [t[3] for t in record_texts(con, extraction_run_id, kind, model_id)])
    out["total_tokens"] = sum(s["tokens"] for s in out["sources"].values())
    out["total_texts"] = sum(s["texts"] for s in out["sources"].values())
    return out


# --- The runs ----------------------------------------------------------------------------------------------------------------------

def _batches(items, count):
    batch, tokens = [], 0
    for item in items:
        n = count(item[-1]) if count else len(item[-1]) // 3
        if batch and (len(batch) >= MAX_ITEMS or tokens + n > MAX_BATCH_TOKENS):
            yield batch, tokens
            batch, tokens = [], 0
        batch.append(item)
        tokens += n
    if batch:
        yield batch, tokens


def _progress(name, done, total, tokens, started, log):
    event = {"event": "embedding_progress", "source": name, "done": done, "total": total, "tokens_sent": tokens, "seconds": int(time.monotonic() - started)}
    LOG.info(json.dumps(event))
    if log:
        log(event)


def embed_chunks(con, ingestion_run_id, embedder, log=None):
    """Chunks of the PDFs -> chunk_embedding, committing after every batch. Returns {"embedded", "tokens"}."""
    count = _encoder()
    embedder.register(con)
    con.commit()
    rows = con.execute("SELECT c.chunk_id,c.text FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s "
                       "AND NOT EXISTS (SELECT 1 FROM chunk_embedding e WHERE e.chunk_id=c.chunk_id AND e.model_id=%s) ORDER BY c.chunk_id", (ingestion_run_id, embedder.model_id)).fetchall()
    con.commit()
    done = tokens = 0
    started = time.monotonic()
    for batch, n in _batches([(cid, _clip(text, count)) for cid, text in rows], count):
        vectors = embedder.embed([t for _, t in batch])
        with con.cursor() as cur:
            cur.executemany("INSERT INTO chunk_embedding(chunk_id,model_id,embedding) VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                            [(cid, embedder.model_id, to_vector_literal(v)) for (cid, _), v in zip(batch, vectors)])
        con.commit()
        done += len(batch)
        tokens += n
        if (done // MAX_ITEMS) % 20 == 0 or done == len(rows):
            _progress("chunks", done, len(rows), tokens, started, log)
    return {"embedded": done, "tokens": tokens}


def embed_records(con, extraction_run_id, embedder, kinds=KINDS, log=None):
    """Records of the construct database -> record_embedding, committing after every batch."""
    count = _encoder()
    embedder.register(con)
    con.commit()
    result = {}
    for kind in kinds:
        items = record_texts(con, extraction_run_id, kind, embedder.model_id)
        con.commit()
        done = tokens = 0
        started = time.monotonic()
        for batch, n in _batches([(rid, run, cid, _clip(text, count)) for rid, run, cid, text in items], count):
            vectors = embedder.embed([b[3] for b in batch])
            with con.cursor() as cur:
                cur.executemany("INSERT INTO record_embedding(record_kind,record_id,model_id,extraction_run_id,canonical_id,text_sha256,recipe,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector) "
                                "ON CONFLICT DO NOTHING",
                                [(kind, rid, embedder.model_id, run, cid, digest(text.encode()), RECIPE, to_vector_literal(v)) for (rid, run, cid, text), v in zip(batch, vectors)])
            con.commit()
            done += len(batch)
            tokens += n
            if (done // MAX_ITEMS) % 10 == 0 or done == len(items):
                _progress(f"records:{kind}", done, len(items), tokens, started, log)
        result[kind] = {"embedded": done, "tokens": tokens}
    return result


# --- Searching the records (used by the discovery forum) -------------------------------------------------------------------------------

_ID_COLUMN = {"mention": "m.mention_id", "measure": "x.measure_id", "claim": "c.claim_id", "attribute": "a.attribute_id"}
_PAGE_SQL = {"mention": "SELECT mention_id,page FROM construct_mention WHERE mention_id = ANY(%s)", "measure": "SELECT measure_id,page FROM construct_measure WHERE measure_id = ANY(%s)",
             "claim": "SELECT claim_id,page FROM nomological_claim WHERE claim_id = ANY(%s)", "attribute": "SELECT attribute_id,page FROM extraction_attribute WHERE attribute_id = ANY(%s)"}


def search_records(con, extraction_run_id, model_id, vector, k=6, kinds=None, papers=None):
    """Nearest records (cosine) of one extraction run, optionally limited to some kinds and to a list of papers. Returns dicts with the record's text, its paper,
    page and distance. Vector search only: a record is a short, dense statement, so words matter less than meaning here."""
    dims = con.execute("SELECT dimensions FROM embedding_model WHERE model_id=%s", (model_id,)).fetchone()[0]
    where, params = "", {"v": to_vector_literal(vector), "m": model_id, "run": extraction_run_id, "k": k}
    if kinds:
        where += " AND record_kind = ANY(%(kinds)s)"
        params["kinds"] = list(kinds)
    if papers is not None:
        where += " AND canonical_id = ANY(%(papers)s)"
        params["papers"] = list(papers)
    hits = con.execute(f"SELECT record_kind,record_id,canonical_id,(embedding::vector({int(dims)})) <=> %(v)s::vector({int(dims)}) AS distance FROM record_embedding "
                       f"WHERE model_id=%(m)s AND extraction_run_id=%(run)s{where} ORDER BY distance, record_id LIMIT %(k)s", params).fetchall()
    return _hydrate(con, extraction_run_id, hits)


def _hydrate(con, extraction_run_id, hits):
    """hits: [(kind, record_id, canonical_id, distance)] -> dicts with the record's text, paper and page."""
    by_kind = {}
    for kind, rid, cid, dist in hits:
        by_kind.setdefault(kind, []).append(rid)
    texts, pages = {}, {}
    for kind, ids in by_kind.items():
        for row in con.execute(_sql_records(kind) + f" AND {_ID_COLUMN[kind]} = ANY(%(ids)s)", {"run": extraction_run_id, "ids": ids}).fetchall():
            texts[(kind, row[0])] = _text(kind, row)
        pages.update({(kind, rid): page for rid, page in con.execute(_PAGE_SQL[kind], (ids,))})
    info = {r[0]: r[1:] for r in con.execute("SELECT canonical_id,title,publication_year,doi,authors,source_title FROM canonical_paper WHERE canonical_id = ANY(%s)", ([h[2] for h in hits],))}
    out = []
    for kind, rid, cid, dist in hits:
        title, year, doi, authors, journal = info.get(cid, ("", None, None, None, None))
        out.append({"record_kind": kind, "record_id": str(rid), "canonical_id": str(cid), "title": title, "year": year, "doi": doi, "authors": authors, "journal": journal, "page": pages.get((kind, rid)),
                    "text": texts.get((kind, rid), ""), "distance": float(dist)})
    return out


def records(con, extraction_run_id, refs):
    """refs: [(kind, record_id, canonical_id)] -> the same dicts as search_records (distance 0), in the order given. Reads the extraction tables, not the embeddings."""
    return _hydrate(con, extraction_run_id, [(kind, rid, cid, 0.0) for kind, rid, cid in refs])


def random_records(con, extraction_run_id, model_id, papers, n, rng, kinds=("mention", "measure", "claim")):
    """n records drawn at random (with `rng`, so a seed makes the draw repeatable) from the papers given; a mix of the kinds. Same shape as search_records."""
    pool = {}
    for kind in kinds:
        pool[kind] = [(kind, rid, cid, 0.0) for rid, cid in con.execute(
            "SELECT record_id,canonical_id FROM record_embedding WHERE record_kind=%s AND model_id=%s AND extraction_run_id=%s AND canonical_id = ANY(%s) ORDER BY record_id",
            (kind, model_id, extraction_run_id, list(papers)))]
    chosen = []
    for _ in range(n):
        kind = rng.choice([k for k in kinds if pool[k]] or [None])
        if kind is None:
            break
        chosen.append(pool[kind].pop(rng.randrange(len(pool[kind]))))
    return _hydrate(con, extraction_run_id, chosen) if chosen else []
