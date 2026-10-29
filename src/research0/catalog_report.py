"""Phase 1 reports: catalog, deduplication log, counts, PRISMA-style flow and seed verification."""
import csv
import difflib
import json
from collections import Counter, defaultdict
from uuid import UUID

from .core import coverage, match, norm_title
from .file_report import load_seeds


def _write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_seeds(con, run_id, seed_files):
    """Match human-validated seeds against the catalog. A title/year ambiguity is resolved only
    when every candidate row belongs to one retained canonical paper."""
    seeds = []
    for path, community in seed_files:
        seeds.extend(load_seeds(path, community))
    records = [dict(zip(("record_id", "doi", "eid", "title_normalized", "publication_year"), row)) for row in con.execute(
        "SELECT m.record_id,r.doi,r.eid,r.title_normalized,r.publication_year FROM catalog_member m "
        "JOIN export_record r USING (record_id) WHERE m.run_id=%s", (run_id,))]
    for r in records:
        r["record_id"] = str(r["record_id"])
    canonical = {str(rid): (cid, status) for rid, cid, status in con.execute(
        "SELECT m.record_id,m.canonical_id,c.status FROM catalog_member m JOIN canonical_paper c USING (canonical_id) WHERE m.run_id=%s", (run_id,))}
    titles = [(r["title_normalized"], r["publication_year"]) for r in records if r["title_normalized"]]
    results = []
    for seed in seeds:
        outcome, method, ids, cause = match(seed, records)
        canon_ids = sorted({canonical[i][0] for i in ids}, key=str)
        if outcome == "ambiguous" and cause == "Multiple title/year candidates" and len(canon_ids) == 1 \
                and canonical[ids[0]][1] == "retained":
            outcome, cause = "found", None
        hint = ""
        if outcome == "missing":
            wanted = norm_title(seed["title"]) or ""
            pool = sorted({t for t, y in titles if y in (seed["publication_year"], seed["publication_year"] + 1)})
            close = difflib.get_close_matches(wanted, pool, n=1, cutoff=.6)
            hint = f"closest catalog title (review aid, not a match): {close[0]}" if close else "no similar title in the catalog for that year"
        results.append({"community": seed["community"], "seed_id": seed["seed_id"], "benchmark_version": seed["version"],
                        "title": seed["title"], "seed_status": seed["status"], "outcome": outcome, "match_method": method or "",
                        "canonical_ids": ";".join(map(str, canon_ids)), "possible_cause": " | ".join(x for x in (cause, hint) if x)})
    with con.transaction():
        con.execute("DELETE FROM catalog_seed_result WHERE run_id=%s", (run_id,))
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO catalog_seed_result(run_id,seed_key,community,benchmark_version,seed_status,outcome,match_method,canonical_ids,possible_cause) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(run_id, r["seed_id"], r["community"], r["benchmark_version"], r["seed_status"], r["outcome"], r["match_method"] or None,
                  [UUID(x) for x in r["canonical_ids"].split(";") if x], r["possible_cause"] or None) for r in results])
    return results


def write_reports(con, run_id, output, seed_files):
    output.mkdir(parents=True, exist_ok=True)
    query_id, summary, created = con.execute("SELECT query_id,summary,created_at FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone()
    batch_label = {bid: f"{name} ({sha[:8]})" for bid, name, sha in con.execute(
        "SELECT batch_id,original_filename,original_sha256 FROM export_batch WHERE batch_id = ANY(%s)", ([UUID(b) for b in summary["batches"]],))}
    catalog_rows, papers_in_batch = [], defaultdict(set)
    for row in con.execute(
            "SELECT c.canonical_id,c.doi,c.eid,c.all_eids,c.all_dois,c.title,c.authors,c.publication_year,c.source_title,c.document_type,"
            "c.member_count,c.batch_count,c.status,c.perimeter_flags,"
            "(SELECT array_agg(DISTINCT e.batch_id) FROM catalog_member m JOIN export_record e USING (record_id) WHERE m.canonical_id=c.canonical_id) "
            "FROM canonical_paper c WHERE c.run_id=%s ORDER BY c.publication_year, c.title, c.canonical_id", (run_id,)):
        (cid, doi, eid, eids, dois, title, authors, year, source, doctype, members, nbatch, status, flags, bids) = row
        catalog_rows.append({"canonical_id": cid, "doi": doi or "", "eid": eid or "", "all_eids": ";".join(eids), "all_dois": ";".join(dois),
                             "title": title, "authors": authors, "year": year, "source_title": source, "document_type": doctype,
                             "member_count": members, "batch_count": nbatch, "status": status, "perimeter_flags": ";".join(flags),
                             "export_batches": ";".join(sorted(batch_label[b] for b in bids))})
        for b in bids:
            papers_in_batch[b].add(cid)
    _write_csv(output / "catalog.csv", catalog_rows, list(catalog_rows[0]))
    log_rows = [{"log_id": a, "decision": b, "method": c, "reason": d, "record_id": e, "other_record_id": f, "canonical_id": g,
                 "details": json.dumps(h, ensure_ascii=False, sort_keys=True)}
                for a, b, c, d, e, f, g, h in con.execute(
                    "SELECT log_id,decision,method,reason,record_id,other_record_id,canonical_id,details FROM dedup_log WHERE run_id=%s ORDER BY decision,method,log_id", (run_id,))]
    _write_csv(output / "dedup_log.csv", log_rows, ["log_id", "decision", "method", "reason", "record_id", "other_record_id", "canonical_id", "details"])
    by_journal = Counter(r["source_title"] or "(missing)" for r in catalog_rows)
    records_by_journal = Counter(src or "(missing)" for (src,) in con.execute(
        "SELECT e.source_title FROM catalog_member m JOIN export_record e USING (record_id) WHERE m.run_id=%s", (run_id,)))
    _write_csv(output / "counts_by_journal.csv", [{"source_title": j, "canonical_papers": n, "export_records": records_by_journal[j]} for j, n in sorted(by_journal.items())],
               ["source_title", "canonical_papers", "export_records"])
    by_year = Counter(r["year"] for r in catalog_rows)
    _write_csv(output / "counts_by_year.csv", [{"year": y, "canonical_papers": n} for y, n in sorted(by_year.items(), key=lambda x: (x[0] is None, x[0]))], ["year", "canonical_papers"])
    multi = {r["canonical_id"] for r in catalog_rows if r["batch_count"] > 1}
    batch_rows = []
    for bid, label in sorted(batch_label.items(), key=lambda x: x[1]):
        papers = papers_in_batch[bid]
        shared = len(papers & multi)
        batch_rows.append({"export_batch": label, "batch_id": bid, "records": summary["batches"][str(bid)], "canonical_papers": len(papers),
                           "papers_also_in_other_batch": shared, "papers_only_in_this_batch": len(papers) - shared})
    _write_csv(output / "counts_by_batch.csv", batch_rows, list(batch_rows[0]))
    seeds = verify_seeds(con, run_id, seed_files) if seed_files else []
    if seeds:
        _write_csv(output / "seed_check.csv", seeds, list(seeds[0]))

    stats = summary["stats"]
    by_method = {m: sum(v for k, v in stats.items() if k.startswith(f"merged_{m}_")) for m in ("eid", "doi", "bibliographic_fallback")}
    flagged_pairs = sum(v for k, v in stats.items() if k.startswith("flagged_"))
    lines = [f"# Phase 1 canonical catalog — run {run_id}", "",
             f"Query: {query_id}  |  Dedup version: {summary['dedup_version']}  |  Created: {created:%Y-%m-%d %H:%M} UTC", "",
             "## PRISMA-style flow", "", "```mermaid", "flowchart TD"]
    lines += [f'  B{i}["Export {r["export_batch"]}<br/>{r["records"]} records"]' for i, r in enumerate(batch_rows)]
    lines += [f'  ID["Records identified: {summary["input_records"]}"]'] + [f"  B{i} --> ID" for i in range(len(batch_rows))]
    lines += [f'  DUP["Linked to an existing paper (kept, not deleted): {summary["records_linked_to_existing_paper"]}<br/>'
              f'EID {by_method["eid"]} · DOI {by_method["doi"]} · bibliographic fallback {by_method["bibliographic_fallback"]}"]',
              f'  CAT["Canonical papers: {summary["canonical_papers"]}"]',
              f'  REV["Flagged for manual identity review: {summary["review_needed_papers"]}"]',
              f'  PER["With a perimeter flag (year / document type): {summary["perimeter_flagged_papers"]}"]',
              "  ID --> DUP --> CAT", "  CAT --> REV", "  CAT --> PER", "```", "",
             "| Step | Records |", "|---|---:|", f"| Records identified across exports | {summary['input_records']} |",
             f"| Linked to an existing paper by EID | {by_method['eid']} |", f"| Linked by DOI | {by_method['doi']} |",
             f"| Linked by bibliographic fallback | {by_method['bibliographic_fallback']} |",
             f"| **Canonical papers** | **{summary['canonical_papers']}** |",
             f"| Unresolved conflicts logged (kept separate) | {flagged_pairs} |",
             f"| Canonical papers needing manual review | {summary['review_needed_papers']} |",
             f"| Canonical papers with a perimeter flag | {summary['perimeter_flagged_papers']} |", ""]
    if stats.get("fallback_skipped_short_title"):
        lines += [f"The bibliographic fallback skipped {stats['fallback_skipped_short_title']} candidate group(s) with titles shorter than 4 words.", ""]
    lines += ["## Counts", "", "| Export batch | Records | Canonical papers | Also in another batch | Only in this batch |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['export_batch']} | {r['records']} | {r['canonical_papers']} | {r['papers_also_in_other_batch']} | {r['papers_only_in_this_batch']} |" for r in batch_rows]
    lines += ["", f"Also written: `counts_by_year.csv` ({len(by_year)} values), `counts_by_journal.csv` ({len(by_journal)} journals), `catalog.csv`, `dedup_log.csv`.", ""]
    if seeds:
        lines += ["## Benchmark seed verification", "", "| Community | Found / eligible | Coverage | 95% target | Ambiguous | Missing |", "|---|---:|---:|---|---:|---:|"]
        for community in ("CTO", "TIM"):
            sub = [s for s in seeds if s["community"] == community]
            c = coverage(sub)
            missing = sum(s["seed_status"] == "eligible" and s["outcome"] == "missing" for s in sub)
            pct = "N/A" if c["coverage"] is None else f"{c['coverage']:.1%}"
            target = "N/A" if c["target_met"] is None else ("met" if c["target_met"] else "not met")
            lines.append(f"| {community} | {c['found']} / {c['eligible']} | {pct} | {target} | {c['ambiguous']} | {missing} |")
        unresolved = [s for s in seeds if s["seed_status"] == "eligible" and s["outcome"] != "found"]
        lines += ["", "All eligible seeds recovered: " + ("yes" if not unresolved else f"**no** — {len(unresolved)} unresolved"), ""]
        lines += [f"- {s['community']} {s['seed_id']}: {s['title']} — {s['outcome']}. {s['possible_cause']}" for s in unresolved]
        lines.append("")
    lines += ["## Limits", "",
              "- Deduplication links records; it never deletes them. A conflict is kept separate and logged, not guessed.",
              "- The export has no Source ID or Source type column, so the SRCID and SRCTYPE(j) parts of the query cannot be verified from the file; only year and document type are checked.",
              "- Counts per journal use the Scopus Source title string, not validated source IDs.",
              "- Seed recovery measures retrieval of the human benchmark, not completeness of the query."]
    (output / "catalog_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, seeds
