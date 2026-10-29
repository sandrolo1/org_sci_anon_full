"""Phase 2: exploratory clustering of the eligible papers.

The result is a map for people to look at (which topics exist, where the benchmark seeds fall, which regions the pre-assignment
treats differently). It does not decide membership of CTO or TIM and it is never an input to a decision (the internal project notes, principles 2-3).
Embeddings and cluster assignments are derived: rerunning with the same inputs returns the same run, new parameters make a new run.
"""
import csv
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from .core import digest, stable_id
from .embeddings import to_vector_literal

LOG = logging.getLogger("research0")
ELIGIBLE = "c.status='retained' AND cardinality(c.perimeter_flags)=0"
NO_ABSTRACT = ("", "[no abstract available]")
VERSION = "explore-v1"


def paper_text(title, abstract, author_keywords, index_keywords):
    abstract = "" if (abstract or "").strip().lower() in NO_ABSTRACT else abstract.strip()
    return "\n".join(x.strip() for x in (title, abstract, author_keywords, index_keywords) if x and x.strip())


def eligible_texts(con, run_id, limit=None):
    """[(canonical_id, text, has_abstract, text_sha256)] for eligible papers, in a stable order."""
    rows = con.execute(
        f"SELECT c.canonical_id,c.title,r.raw_fields->>'Abstract',r.raw_fields->>'Author Keywords',r.raw_fields->>'Index Keywords' "
        f"FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id WHERE c.run_id=%s AND {ELIGIBLE} "
        "ORDER BY c.canonical_id LIMIT %s", (run_id, limit)).fetchall()
    out = []
    for cid, title, abstract, ak, ik in rows:
        text = paper_text(title or "", abstract or "", ak or "", ik or "")
        out.append((cid, text, (abstract or "").strip().lower() not in NO_ABSTRACT, digest(text.encode())))
    return out


def embed_papers(con, run_id, embedder, limit=None, batch=2000):
    """Compute and store the embedding of every eligible paper that has none for its current text. Returns (new, existing)."""
    embedder.register(con)
    papers = eligible_texts(con, run_id, limit)
    have = {(cid, sha) for cid, sha in con.execute(
        "SELECT canonical_id,text_sha256 FROM paper_embedding WHERE model_id=%s AND canonical_id = ANY(%s)", (embedder.model_id, [p[0] for p in papers]))}
    todo = [p for p in papers if (p[0], p[3]) not in have]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vectors = embedder.embed([p[1] for p in part])
        with con.cursor() as cur:
            cur.executemany("INSERT INTO paper_embedding(canonical_id,model_id,text_sha256,embedding) VALUES (%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                            [(p[0], embedder.model_id, p[3], to_vector_literal(v)) for p, v in zip(part, vectors)])
        LOG.info(json.dumps({"event": "embedded", "done": min(i + batch, len(todo)), "total": len(todo)}))
    return len(todo), len(papers) - len(todo)


def load_vectors(con, run_id, model_id, limit=None):
    papers = eligible_texts(con, run_id, limit)
    wanted = {(cid, sha): (text, has) for cid, text, has, sha in papers}
    ids, vectors, texts, has_abstract = [], [], [], []
    for cid, sha, vec in con.execute("SELECT canonical_id,text_sha256,embedding::text FROM paper_embedding WHERE model_id=%s AND canonical_id = ANY(%s) ORDER BY canonical_id",
                                     (model_id, [p[0] for p in papers])):
        if (cid, sha) in wanted:
            ids.append(cid)
            vectors.append(np.asarray(json.loads(vec), dtype=np.float32))
            texts.append(wanted[(cid, sha)][0])
            has_abstract.append(wanted[(cid, sha)][1])
    if not ids:
        raise ValueError("No embeddings for this model; run the embedding step first")
    return ids, np.vstack(vectors), texts, has_abstract


def top_terms(texts, labels, k, per_cluster=8):
    """Class-based TF-IDF: terms that characterise a cluster against the others. A reading aid, not a topic model."""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5, max_features=60000,
                                 token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    try:
        counts = vectorizer.fit_transform(texts)
    except ValueError:  # empty vocabulary
        return {c: [] for c in range(k)}
    vocabulary = np.array(vectorizer.get_feature_names_out())
    labels = np.asarray(labels)
    per_class = np.vstack([np.asarray(counts[labels == c].sum(axis=0)).ravel() if (labels == c).any() else np.zeros(counts.shape[1]) for c in range(k)])
    tf = per_class / np.maximum(per_class.sum(axis=1, keepdims=True), 1)
    idf = np.log(1 + per_class.sum() / np.maximum(per_class.sum(axis=0), 1))
    score = tf * idf
    result = {}
    for c in range(k):
        order = np.argsort(-score[c])[:per_cluster]
        result[c] = [str(vocabulary[i]) for i in order if score[c][i] > 0]
    return result


def cluster(con, run_id, embedder, k=40, seed=7, limit=None):
    """KMeans over the stored embeddings. Idempotent: same run, model, parameters and texts give the same cluster_run."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    ids, vectors, texts, has_abstract = load_vectors(con, run_id, embedder.model_id, limit)
    if not 2 <= k <= len(ids):
        raise ValueError(f"k must be between 2 and the number of papers ({len(ids)})")
    params = {"k": k, "n_init": 3, "version": VERSION}
    cluster_run_id = stable_id("cluster", run_id, embedder.model_id, json.dumps(params, sort_keys=True), seed, digest("".join(map(str, ids)).encode()))
    existing = con.execute("SELECT summary FROM cluster_run WHERE cluster_run_id=%s", (cluster_run_id,)).fetchone()
    if existing:
        return cluster_run_id, existing[0], True
    model = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(vectors)
    labels = model.labels_
    distances = np.linalg.norm(vectors - model.cluster_centers_[labels], axis=1)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(ids), size=min(2000, len(ids)), replace=False)
    silhouette = float(silhouette_score(vectors[sample], labels[sample])) if len(set(labels[sample])) > 1 else None
    seeds = {"CTO": set(), "TIM": set()}
    for community, found in con.execute("SELECT community,canonical_ids FROM catalog_seed_result WHERE run_id=%s AND outcome='found' AND seed_status='eligible'", (run_id,)):
        seeds[community].update(found)
    terms = top_terms(texts, labels, k)
    clusters = []
    for c in range(k):
        members = np.where(labels == c)[0]
        nearest = members[np.argsort(distances[members])[:3]]
        clusters.append({"label": c, "size": int(len(members)), "top_terms": terms[c], "without_abstract": int(sum(not has_abstract[i] for i in members)),
                         "cto_seeds": sum(ids[i] in seeds["CTO"] for i in members), "tim_seeds": sum(ids[i] in seeds["TIM"] for i in members),
                         "nearest_papers": [str(ids[i]) for i in nearest]})
    summary = {"version": VERSION, "model_id": embedder.model_id, "embedding_kind": embedder.kind, "papers": len(ids), "k": k, "seed": seed,
               "silhouette_sample": silhouette, "inertia": float(model.inertia_), "clusters": clusters,
               "warning": "Exploratory map. It does not decide CTO/TIM membership or construct equivalence."
                          + (" The embedding is lexical: clusters follow shared words, not meaning." if embedder.kind == "lexical" else "")}
    with con.transaction():
        con.execute("INSERT INTO cluster_run(cluster_run_id,run_id,model_id,algorithm,params,seed,paper_count,cluster_count,summary) VALUES (%s,%s,%s,'kmeans',%s::jsonb,%s,%s,%s,%s::jsonb)",
                    (cluster_run_id, run_id, embedder.model_id, json.dumps(params), seed, len(ids), k, json.dumps(summary)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO cluster_assignment(cluster_run_id,canonical_id,cluster_label,distance) VALUES (%s,%s,%s,%s)",
                            [(cluster_run_id, cid, int(l), float(d)) for cid, l, d in zip(ids, labels, distances)])
    return cluster_run_id, summary, False


def cross_tab(con, cluster_run_id, preassignment_run_id):
    """Pre-assignment labels per cluster, to see where the model and the map disagree. Descriptive only."""
    rows = con.execute(
        "SELECT a.cluster_label,p.community,p.label,count(*) FROM cluster_assignment a JOIN community_preassignment p USING (canonical_id) "
        "WHERE a.cluster_run_id=%s AND p.preassignment_run_id=%s GROUP BY 1,2,3 ORDER BY 1,2,3", (cluster_run_id, preassignment_run_id)).fetchall()
    table = {}
    for label, community, answer, n in rows:
        table.setdefault(label, {}).setdefault(community, Counter())[answer] += n
    return table


def write_report(con, cluster_run_id, output, preassignment_run_id=None):
    row = con.execute("SELECT summary,run_id FROM cluster_run WHERE cluster_run_id=%s", (cluster_run_id,)).fetchone()
    if not row:
        raise ValueError("Unknown cluster run")
    summary = row[0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "clusters.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["canonical_id", "cluster_label", "distance"])
        writer.writerows(con.execute("SELECT canonical_id,cluster_label,distance FROM cluster_assignment WHERE cluster_run_id=%s ORDER BY cluster_label,distance", (cluster_run_id,)))
    table = cross_tab(con, cluster_run_id, preassignment_run_id) if preassignment_run_id else {}
    titles = {str(cid): title for cid, title in con.execute("SELECT canonical_id,title FROM canonical_paper WHERE canonical_id = ANY(%s)",
                                                         ([uuid_ for c in summary["clusters"] for uuid_ in c["nearest_papers"]],))}
    lines = [f"# Exploratory clusters {cluster_run_id}", "", f"**{summary['warning']}**", "",
             f"Papers: {summary['papers']} | k = {summary['k']} | embedding `{summary['model_id']}` ({summary['embedding_kind']}) | "
             f"silhouette (2000-paper sample): {summary['silhouette_sample'] if summary['silhouette_sample'] is None else round(summary['silhouette_sample'], 3)}", "",
             "| Cluster | Papers | Top terms | CTO seeds | TIM seeds | No abstract |", "|---:|---:|---|---:|---:|---:|"]
    for c in sorted(summary["clusters"], key=lambda c: -c["size"]):
        lines.append(f"| {c['label']} | {c['size']} | {', '.join(c['top_terms'])} | {c['cto_seeds']} | {c['tim_seeds']} | {c['without_abstract']} |")
    lines += ["", "## Reading aid: the three papers closest to each centre", ""]
    for c in sorted(summary["clusters"], key=lambda c: -c["size"]):
        lines.append(f"- **{c['label']}**: " + "; ".join(titles.get(p, p) for p in c["nearest_papers"]))
    if table:
        lines += ["", "## Pre-assignment labels per cluster (descriptive)", "", "| Cluster | CTO yes/no/uncertain | TIM yes/no/uncertain |", "|---:|---|---|"]
        for label in sorted(table):
            cell = lambda comm: "/".join(str(table[label].get(comm, Counter())[a]) for a in ("yes", "no", "uncertain"))
            lines.append(f"| {label} | {cell('CTO')} | {cell('TIM')} |")
    (output / "cluster_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
