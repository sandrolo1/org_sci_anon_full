"""Phase 1: canonical catalog from export records, with a logged, reversible deduplication.

Export records are never deleted. A duplicate is linked to a canonical paper, and every
link or unresolved conflict is written to the deduplication log.
"""
import json
import logging
from collections import Counter, defaultdict

from psycopg.types.json import Jsonb

from .core import norm_title, stable_id

LOG = logging.getLogger("research0")
DEDUP_VERSION = "dedup-v1"
PERIMETER_YEARS = (2015, 2025)
PERIMETER_DOCTYPES = {"article", "review"}
MIN_FALLBACK_TITLE_TOKENS = 4


def _tokens(value):
    return len((value or "").split())


class _Clusters:
    """Clusters keyed by the earliest member; members keep their input order."""

    def __init__(self, records):
        self.order = {r["record_id"]: i for i, r in enumerate(records)}
        self.by_id = {r["record_id"]: r for r in records}
        self.where = {r["record_id"]: r["record_id"] for r in records}
        self.members = {r["record_id"]: [r["record_id"]] for r in records}
        self.link = {}

    def eids(self, cid):
        return {self.by_id[m]["eid"] for m in self.members[cid] if self.by_id[m]["eid"]}

    def dois(self, cid):
        return {self.by_id[m]["doi"] for m in self.members[cid] if self.by_id[m]["doi"]}

    def merge(self, a, b, method):
        """Merge the clusters of records a and b; return (kept, absorbed) cluster ids."""
        ca, cb = self.where[a], self.where[b]
        if ca == cb:
            return None
        keep, drop = (ca, cb) if self.order[ca] < self.order[cb] else (cb, ca)
        self.link[drop] = (method, keep)
        moved = self.members.pop(drop)
        for m in moved:
            self.where[m] = keep
        self.members[keep] = sorted(self.members[keep] + moved, key=self.order.get)
        return keep, drop


def cluster_records(records):
    """Deduplicate records (already in deterministic order) with a prudent policy.

    1. Same EID: merge (strong). A DOI disagreement is merged but flagged.
    2. Same DOI: merge only if at most one side carries a Scopus EID; two clusters with
       different EIDs and the same DOI stay separate and are flagged.
    3. Fallback on normalized title + year + authors + source, only for titles of at least
       four words and never across two different DOIs; those are flagged instead.
    """
    cl = _Clusters(records)
    log = []
    flagged = set()
    stats = Counter()

    def merged(method, rec, anchor, reason):
        if cl.merge(rec["record_id"], anchor["record_id"], method) is None:
            return
        same_batch = rec["batch_id"] == anchor["batch_id"]
        stats[f"merged_{method}_{'same_batch' if same_batch else 'cross_batch'}"] += 1
        log.append({"decision": "merged", "method": method, "record_id": rec["record_id"],
                    "other_record_id": anchor["record_id"], "reason": reason,
                    "details": {"same_batch": same_batch}})

    def flag(method, a, b, reason, **details):
        flagged.update((a["record_id"], b["record_id"]))
        stats[f"flagged_{method}"] += 1
        log.append({"decision": "flagged_for_review", "method": method, "record_id": a["record_id"],
                    "other_record_id": b["record_id"], "reason": reason, "details": details})

    # Step 1: EID.
    first_eid = {}
    for rec in records:
        eid = rec["eid"]
        if not eid:
            continue
        if eid not in first_eid:
            first_eid[eid] = rec
            continue
        anchor = first_eid[eid]
        merged("eid", rec, anchor, "Identical Scopus EID")
        if rec["doi"] and anchor["doi"] and rec["doi"] != anchor["doi"]:
            flag("eid_doi_conflict", rec, anchor, "Same EID but different DOI", dois=[rec["doi"], anchor["doi"]])

    # Step 2: DOI.
    by_doi = defaultdict(list)
    for rec in records:
        if rec["doi"]:
            by_doi[rec["doi"]].append(rec)
    for doi, group in by_doi.items():
        cids = list(dict.fromkeys(cl.where[r["record_id"]] for r in group))
        if len(cids) < 2:
            continue
        with_eid = [c for c in cids if cl.eids(c)]
        if len(with_eid) >= 2:
            head = cl.by_id[cids[0]]
            for other in cids[1:]:
                flag("doi_multiple_eids", head, cl.by_id[other], "Same DOI on different Scopus EIDs; kept separate",
                     doi=doi, eids=sorted(cl.eids(cids[0]) | cl.eids(other)))
            continue
        head = cl.by_id[cids[0]]
        for other in cids[1:]:
            merged("doi", cl.by_id[other], head, "Identical DOI")

    # Step 3: bibliographic fallback.
    by_key = defaultdict(list)
    for rec in records:
        key = (rec["title_normalized"], rec["publication_year"], norm_title(rec["authors"]), norm_title(rec["source_title"]))
        if all(key):
            by_key[key].append(rec)
    for key, group in by_key.items():
        cids = list(dict.fromkeys(cl.where[r["record_id"]] for r in group))
        if len(cids) < 2:
            continue
        if _tokens(key[0]) < MIN_FALLBACK_TITLE_TOKENS:
            stats["fallback_skipped_short_title"] += 1
            continue
        head_id = cids[0]
        for other in cids[1:]:
            head, absorbed = cl.by_id[head_id], cl.by_id[other]
            if any(m in flagged for c in (cl.where[head_id], other) for m in cl.members[c]):
                stats["fallback_skipped_already_flagged"] += 1  # an open conflict needs a human, not a weaker rule
                continue
            head_dois, other_dois = cl.dois(cl.where[head_id]), cl.dois(other)
            if head_dois and other_dois and head_dois.isdisjoint(other_dois):
                flag("bibliographic_doi_conflict", head, absorbed,
                     "Title, year, authors and source agree but DOIs differ; kept separate",
                     dois=sorted(head_dois | other_dois))
                continue
            merged("bibliographic_fallback", absorbed, head,
                   "Normalized title, year, authors and source agree; no conflicting DOI")
    return cl, log, flagged, stats


def build_clusters(cl, flagged):
    """Turn clusters into canonical paper dicts, sorted by their first member."""
    papers = []
    for cid in sorted(cl.members, key=cl.order.get):
        members = [cl.by_id[m] for m in sorted(cl.members[cid], key=cl.order.get)]
        rep = max(members, key=lambda r: (r["abstract_length"], -cl.order[r["record_id"]]))
        all_eids = sorted({r["eid"] for r in members if r["eid"]})
        all_dois = sorted({r["doi"] for r in members if r["doi"]})
        flags = []
        years = {r["publication_year"] for r in members}
        if any(y is None or not PERIMETER_YEARS[0] <= y <= PERIMETER_YEARS[1] for y in years):
            flags.append("year_outside_2015_2025")
        if any((r["document_type"] or "").strip().lower() not in PERIMETER_DOCTYPES for r in members):
            flags.append("document_type_outside_article_review")
        papers.append({
            "anchor": cid, "members": members, "representative": rep,
            "doi": rep["doi"] or (all_dois[0] if all_dois else None),
            "eid": rep["eid"] or (all_eids[0] if all_eids else None),
            "all_eids": all_eids, "all_dois": all_dois,
            "batch_count": len({r["batch_id"] for r in members}),
            "status": "review_needed" if any(r["record_id"] in flagged for r in members) else "retained",
            "perimeter_flags": flags,
        })
    return papers


def load_records(con, batch_ids):
    rows = con.execute(
        "SELECT r.record_id,r.batch_id,r.row_number,r.doi,r.eid,r.title,r.title_normalized,r.authors,"
        "r.publication_year,r.source_title,r.document_type,length(coalesce(r.raw_fields->>'Abstract','')) "
        "FROM export_record r JOIN export_batch b USING (batch_id) WHERE r.batch_id = ANY(%s) "
        "ORDER BY b.original_sha256, r.row_number", (list(batch_ids),)).fetchall()
    keys = ("record_id", "batch_id", "row_number", "doi", "eid", "title", "title_normalized", "authors",
            "publication_year", "source_title", "document_type", "abstract_length")
    return [dict(zip(keys, row)) for row in rows]


def build_catalog(con, query_id):
    """Build the catalog for all batches of one query. Re-running with the same inputs is a no-op."""
    batch_ids = sorted(r[0] for r in con.execute("SELECT batch_id FROM export_batch WHERE query_id=%s", (query_id,)))
    if not batch_ids:
        raise ValueError("No export batches for this query")
    run_id = stable_id("catalog", DEDUP_VERSION, query_id, *batch_ids)
    existing = con.execute("SELECT canonical_count,summary FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone()
    if existing:
        LOG.info(json.dumps({"event": "catalog_exists", "run_id": str(run_id)}))
        return run_id, existing[1], True
    records = load_records(con, batch_ids)
    cl, log, flagged, stats = cluster_records(records)
    papers = build_clusters(cl, flagged)
    canonical_of = {}
    for p in papers:
        p["canonical_id"] = stable_id("canonical", run_id, p["anchor"])
        for m in p["members"]:
            canonical_of[m["record_id"]] = p["canonical_id"]
    summary = {
        "dedup_version": DEDUP_VERSION, "input_records": len(records), "canonical_papers": len(papers),
        "records_linked_to_existing_paper": len(records) - len(papers), "stats": dict(stats),
        "review_needed_papers": sum(p["status"] == "review_needed" for p in papers),
        "perimeter_flagged_papers": sum(bool(p["perimeter_flags"]) for p in papers),
        "batches": {str(b): sum(r["batch_id"] == b for r in records) for b in batch_ids},
    }
    with con.transaction():
        con.execute("INSERT INTO catalog_run(run_id,query_id,dedup_version,batch_ids,input_record_count,canonical_count,summary) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)", (run_id, query_id, DEDUP_VERSION, batch_ids, len(records), len(papers), Jsonb(summary)))
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO canonical_paper(canonical_id,run_id,representative_record_id,doi,eid,all_dois,all_eids,title,authors,"
                "publication_year,source_title,document_type,member_count,batch_count,status,perimeter_flags) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(p["canonical_id"], run_id, p["representative"]["record_id"], p["doi"], p["eid"], p["all_dois"], p["all_eids"],
                  p["representative"]["title"], p["representative"]["authors"], p["representative"]["publication_year"],
                  p["representative"]["source_title"], p["representative"]["document_type"], len(p["members"]),
                  p["batch_count"], p["status"], p["perimeter_flags"]) for p in papers])
            rows = []
            for p in papers:
                for m in p["members"]:
                    method = "representative" if m["record_id"] == p["anchor"] else cl.link[m["record_id"]][0]
                    rows.append((run_id, m["record_id"], p["canonical_id"], method))
            cur.executemany("INSERT INTO catalog_member(run_id,record_id,canonical_id,link_method) VALUES (%s,%s,%s,%s)", rows)
            cur.executemany(
                "INSERT INTO dedup_log(log_id,run_id,decision,method,record_id,other_record_id,canonical_id,reason,details) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(stable_id("dedup-log", run_id, i), run_id, e["decision"], e["method"], e["record_id"], e["other_record_id"],
                  canonical_of[e["record_id"]], e["reason"], Jsonb(e["details"])) for i, e in enumerate(log)])
    LOG.info(json.dumps({"event": "catalog_built", "run_id": str(run_id), **{k: summary[k] for k in ("input_records", "canonical_papers")}}))
    return run_id, summary, False
