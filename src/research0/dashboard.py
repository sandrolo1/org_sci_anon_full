"""Static validation dashboard: a read-only snapshot of one catalog run, rendered as one HTML file.

The snapshot holds statistics plus the items that need a human decision. It contains no full
catalog, only the records involved in an open question.
"""
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .core import norm_title

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
SEED_DECISIONS = ["not_indexed_in_scopus", "metadata_differs_fix_seed", "journal_missing_from_source_list", "remove_from_benchmark", "needs_discussion"]
DEDUP_DECISIONS = ["same_paper", "distinct_papers", "needs_discussion"]


def _record(row):
    rid, title, authors, year, source, doctype, doi, eid, link, filename = row
    return {"record_id": str(rid), "title": title, "authors": authors, "year": year, "source": source, "document_type": doctype,
            "doi": doi, "eid": eid, "link": link, "export_file": filename}


RECORD_SQL = ("SELECT r.record_id,r.title,r.authors,r.publication_year,r.source_title,r.document_type,r.doi,r.eid,"
              "r.raw_fields->>'Link',b.original_filename FROM export_record r JOIN export_batch b USING (batch_id) WHERE r.record_id=%s")


def current_decisions(con, run_id):
    """Latest decision per item and validator; withdrawn ones are kept so clients can tell."""
    out = defaultdict(list)
    for item_id, validator, decision, note, at in con.execute(
            "SELECT DISTINCT ON (item_id,validator) item_id,validator,decision,note,decided_at FROM validation_decision "
            "WHERE run_id=%s ORDER BY item_id,validator,decided_at DESC", (run_id,)):
        out[item_id].append({"validator": validator, "decision": decision, "note": note, "at": at.isoformat(timespec="seconds")})
    return dict(out)


def build_snapshot(con, run_id):
    query_id, summary, created = con.execute("SELECT query_id,summary,created_at FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone()
    label, query_text = con.execute("SELECT source_version,query_text FROM query_version WHERE query_id=%s", (query_id,)).fetchone()
    srcids = re.findall(r"SRCID\((\d+)\)", query_text)
    batches = [{"file": name, "sha256": sha, "exported_on": str(day), "provenance": prov, "records": n, "batch_id": str(bid)} for bid, name, sha, day, prov, n in con.execute(
        "SELECT batch_id,original_filename,original_sha256,exported_on,provenance,record_count FROM export_batch WHERE query_id=%s ORDER BY original_filename", (query_id,))]
    by_year = [[y, n] for y, n in con.execute("SELECT publication_year,count(*) FROM canonical_paper WHERE run_id=%s GROUP BY 1 ORDER BY 1", (run_id,))]
    journals = [{"source": s, "papers": n, "records": r} for s, n, r in con.execute(
        "SELECT c.source_title,count(*),sum(c.member_count) FROM canonical_paper c WHERE c.run_id=%s GROUP BY 1 ORDER BY 2 DESC", (run_id,))]
    no_doi = con.execute("SELECT count(*) FROM canonical_paper WHERE run_id=%s AND doi IS NULL", (run_id,)).fetchone()[0]
    no_abstract = con.execute(
        "SELECT count(*) FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE c.run_id=%s AND (coalesce(r.raw_fields->>'Abstract','')='' OR r.raw_fields->>'Abstract' ILIKE '[No abstract available]%%')", (run_id,)).fetchone()[0]
    validated_sources = con.execute("SELECT count(*) FROM source_list WHERE source_id = ANY(%s)", (srcids,)).fetchone()[0]

    items = []
    # 1. Deduplication conflicts: pairs the rules refused to decide.
    for log_id, method, reason, a, b, details in con.execute(
            "SELECT log_id,method,reason,record_id,other_record_id,details FROM dedup_log WHERE run_id=%s AND decision='flagged_for_review' ORDER BY method,log_id", (run_id,)):
        items.append({"id": f"dedup:{log_id}", "type": "dedup_conflict", "severity": "medium",
                      "title": f"Same paper or distinct? ({method.replace('_', ' ')})", "description": reason + ". Kept as separate canonical papers until a human decides.",
                      "evidence": [_record(con.execute(RECORD_SQL, (a,)).fetchone()), _record(con.execute(RECORD_SQL, (b,)).fetchone())],
                      "details": details, "decisions": DEDUP_DECISIONS})
    # 2. Eligible benchmark seeds not recovered.
    seeds = []
    for community, key, status, outcome, cause, title, authors, year, journal, doi, eid, rationale in con.execute(
            "SELECT r.community,r.seed_key,r.seed_status,r.outcome,r.possible_cause,s.title,s.authors,s.publication_year,s.journal,s.doi,s.eid,s.expert_rationale "
            "FROM catalog_seed_result r LEFT JOIN benchmark_seed s ON s.community=r.community AND s.version=r.benchmark_version AND s.external_seed_id=r.seed_key "
            "WHERE r.run_id=%s ORDER BY r.community,r.seed_key", (run_id,)):
        seeds.append({"community": community, "status": status, "outcome": outcome})
        if status == "eligible" and outcome != "found":
            items.append({"id": f"seed:{community}:{key}", "type": "seed_not_recovered", "severity": "high",
                          "title": f"{community} {key} not recovered: {title}", "description": f"The benchmark paper is {outcome} in the catalog. {cause or ''}".strip(),
                          "evidence": [{"title": title, "authors": authors, "year": year, "source": journal, "doi": doi, "eid": eid, "note": rationale}],
                          "details": {}, "decisions": SEED_DECISIONS})
    gates = []
    for community in ("CTO", "TIM"):
        sub = [s for s in seeds if s["community"] == community and s["status"] == "eligible"]
        found = sum(s["outcome"] == "found" for s in sub)
        gates.append({"community": community, "found": found, "eligible": len(sub), "coverage": found / len(sub) if sub else None})
    # 3. Journal titles that look like variants of one source (one SRCID, two strings).
    groups = defaultdict(list)
    for j in journals:
        groups[norm_title(j["source"])].append(j)
    for key, group in groups.items():
        if len(group) > 1:
            items.append({"id": f"journal-variants:{key}", "type": "journal_name_variants", "severity": "low",
                          "title": "One journal under several titles: " + " / ".join(g["source"] for g in group),
                          "description": "Probably one Scopus source that was renamed. Confirm the SRCID so the source list has one entry.",
                          "evidence": [{"title": g["source"], "note": f"{g['papers']} canonical papers"} for g in group], "details": {},
                          "decisions": ["same_source", "different_sources", "needs_discussion"]})
    # 4. Source list: the query is registered, but no SRCID has a human validation record.
    if not srcids:
        items.append({"id": "query:unrecorded", "type": "query_unrecorded", "severity": "high", "title": "Scopus query text was not recorded",
                      "description": "Without the query text the perimeter cannot be audited. Register the exact query that produced these exports.",
                      "evidence": [{"title": label, "note": query_text}], "details": {}, "decisions": ["query_supplied", "needs_discussion"]})
    elif validated_sources < len(srcids):
        items.append({"id": "source-list:unvalidated", "type": "source_list_unvalidated", "severity": "high",
                      "title": f"Source list not validated: {len(srcids) - validated_sources} of {len(srcids)} SRCIDs have no validated entry",
                      "description": "The exports have no Source ID column, so journal titles cannot be tied to the SRCIDs in the query automatically. A human must confirm each SRCID, journal name and inclusion rationale (template: templates/source_list.csv).",
                      "evidence": [{"title": ", ".join(srcids), "note": f"{len(journals)} distinct Source title strings found in the exports"}], "details": {},
                      "decisions": ["validated_and_submitted", "needs_discussion"]})
    # 5. Papers without DOI: matching relies on EID and title for them.
    rows = con.execute("SELECT c.canonical_id,c.title,c.authors,c.publication_year,c.source_title,c.document_type,c.doi,c.eid,r.raw_fields->>'Link','' "
                       "FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id WHERE c.run_id=%s AND c.doi IS NULL ORDER BY c.publication_year,c.title LIMIT 40", (run_id,)).fetchall()
    if no_doi:
        items.append({"id": "missing-doi", "type": "missing_doi", "severity": "low", "title": f"{no_doi} canonical papers have no DOI",
                      "description": "They are identified by EID only. Check that none is a duplicate of a paper with a DOI.",
                      "evidence": [_record(r) for r in rows], "details": {"shown": len(rows), "total": no_doi}, "decisions": ["checked_no_duplicates", "duplicates_found", "needs_discussion"]})
    order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda i: (order[i["severity"]], i["type"], i["id"]))
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "phase": "Phase 1: canonical catalog",
        "run": {"run_id": str(run_id), "created_at": created.isoformat(timespec="seconds"), "dedup_version": summary["dedup_version"], "query_id": str(query_id),
                "query_label": label, "query_text": query_text, "srcid_count": len(srcids)},
        "totals": {"records": summary["input_records"], "canonical": summary["canonical_papers"], "linked": summary["records_linked_to_existing_paper"],
                   "review_needed": summary["review_needed_papers"], "perimeter_flagged": summary["perimeter_flagged_papers"],
                   "missing_doi": no_doi, "missing_abstract": no_abstract, "journals": len(journals)},
        "dedup_stats": summary["stats"], "batches": batches, "by_year": by_year, "journals": journals, "gates": gates,
        "items": items, "decisions": current_decisions(con, run_id), "live": False,
        "limits": ["Decisions are stored in the validator's browser and exported as CSV; they are not yet imported into the database.",
                   "The export has no Source ID or Source type column, so SRCID and SRCTYPE(j) cannot be checked from the file.",
                   "Seed recovery measures retrieval of the human benchmark, not completeness of the query."],
    }


def render_html(snapshot):
    data = json.dumps(snapshot, ensure_ascii=False, default=str).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__SNAPSHOT__*/null", data)


def render(snapshot, output):
    output.mkdir(parents=True, exist_ok=True)
    html = render_html(snapshot)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return output / "index.html"
