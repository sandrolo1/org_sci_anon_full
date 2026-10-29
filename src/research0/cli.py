import argparse
import csv
import datetime as dt
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import boto3
import psycopg
from psycopg.types.json import Jsonb

from .core import SEED_COLUMNS, SOURCE_COLUMNS, coverage, digest, file_digest, mapped, match, norm_doi, norm_eid, query_text, read_scopus, reconcile, stable_id


LOG = logging.getLogger("research0")
MATCHER_VERSION = "phase0-v1"


def database():
    return psycopg.connect(os.environ["DATABASE_URL"])


def store_original(path, sha):
    bucket = os.environ["S3_BUCKET"]
    client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"), region_name=os.getenv("S3_REGION", "us-east-1"))
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError as e:
        if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
            raise
        client.create_bucket(Bucket=bucket)
    key = f"scopus/originals/{sha}/{path.name}"
    client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
    return f"s3://{bucket}/{key}"


def csv_rows(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(x not in reader.fieldnames for x in columns):
            raise ValueError(f"Required columns: {', '.join(columns)}")
        for row in reader:
            if any((v or "").strip() for v in row.values()):
                yield {k: (v or "").strip() for k, v in row.items()}


def require(row, fields):
    missing = [f for f in fields if not row.get(f)]
    if missing:
        raise ValueError("Missing required values: " + ", ".join(missing))


def import_sources(con, path):
    rows = list(csv_rows(path, SOURCE_COLUMNS))
    with con.transaction():
        for row in rows:
            require(row, SOURCE_COLUMNS)
            if row["active"].lower() not in ("true", "false"):
                raise ValueError("active must be true or false")
            values = (row["source_id"], row["journal_name"], row["inclusion_rationale"], row["active"].lower() == "true", row["version"], row["validated_on"], row["validated_by"])
            existing = con.execute("SELECT journal_name,inclusion_rationale,active,validated_on,validated_by FROM source_list WHERE version=%s AND source_id=%s", (row["version"], row["source_id"])).fetchone()
            if existing:
                if tuple(str(x).lower() if isinstance(x, bool) else str(x) for x in existing) != tuple(str(x).lower() if isinstance(x, bool) else str(x) for x in (values[1], values[2], values[3], values[5], values[6])):
                    raise ValueError("Source version already exists with different content; create a new version")
            else:
                con.execute("INSERT INTO source_list(source_id,journal_name,inclusion_rationale,active,version,validated_on,validated_by) VALUES (%s,%s,%s,%s,%s,%s,%s)", values)
    return len(rows)


def import_seeds(con, path):
    rows = list(csv_rows(path, SEED_COLUMNS))
    with con.transaction():
        for row in rows:
            require(row, ["community", "version", "title", "publication_year", "journal", "top_rank", "expert_rationale", "status", "retrieval_note", "validated_by", "validated_on"])
            if row["community"] not in ("CTO", "TIM") or row["status"] not in ("eligible", "out_of_scope", "duplicate", "to_verify"):
                raise ValueError("Invalid community or status")
            if not row["doi"] and not row["eid"] and not row["title"]:
                raise ValueError("Seed needs an identifier or title")
            external_id = row["seed_id"] or None
            try:
                sid = uuid.UUID(external_id) if external_id else stable_id("seed", row["community"], row["version"], norm_doi(row["doi"]) or norm_eid(row["eid"]) or row["title"], row["publication_year"])
            except ValueError:
                sid = stable_id("seed-external", row["community"], row["version"], external_id)
            vals = (sid, external_id, row["community"], row["version"], norm_doi(row["doi"]), norm_eid(row["eid"]), row["title"], row["authors"], int(row["publication_year"]), row["journal"], row["top_rank"], row["expert_rationale"], row["status"], row["retrieval_note"], row["validated_by"], row["validated_on"])
            existing = con.execute("SELECT external_seed_id,community,version,doi,eid,title,authors,publication_year,journal,top_rank,expert_rationale,status,retrieval_note,validated_by,validated_on FROM benchmark_seed WHERE seed_id=%s", (sid,)).fetchone()
            if existing:
                if tuple(str(x or "") for x in existing) != tuple(str(x or "") for x in vals[1:]):
                    raise ValueError("Seed ID already exists with different content; create a new version and seed ID")
            else:
                con.execute("INSERT INTO benchmark_seed(seed_id,external_seed_id,community,version,doi,eid,title,authors,publication_year,journal,top_rank,expert_rationale,status,retrieval_note,validated_by,validated_on) VALUES (" + ",".join(["%s"] * 16) + ")", vals)
    return len(rows)


def make_query(con, version):
    ids = [r[0] for r in con.execute("SELECT source_id FROM source_list WHERE version=%s AND active=true", (version,))]
    query = query_text(ids)
    sha = digest(query.encode())
    qid = stable_id("query", version, sha)
    with con.transaction():
        con.execute("INSERT INTO query_version(query_id,source_version,query_text,query_sha256) VALUES (%s,%s,%s,%s) ON CONFLICT (source_version,query_sha256) DO NOTHING", (qid, version, query, sha))
    return qid, query


def register_query(con, label, text_file):
    """Record the Scopus query behind a set of exports without inventing its text."""
    text = text_file.read_text(encoding="utf-8").strip() if text_file else "UNRECORDED: query text not supplied by the user; perimeter not verified"
    sha = digest(text.encode())
    qid = stable_id("query", label, sha)
    with con.transaction():
        con.execute("INSERT INTO query_version(query_id,source_version,query_text,query_sha256) VALUES (%s,%s,%s,%s) ON CONFLICT (source_version,query_sha256) DO NOTHING", (qid, label, text, sha))
    return qid, text


def import_scopus(con, path, query_id, export_date, provenance):
    sha = file_digest(path)
    bid = stable_id("batch", query_id, sha)
    existing = con.execute("SELECT batch_id FROM export_batch WHERE query_id=%s AND original_sha256=%s", (query_id, sha)).fetchone()
    if existing:
        recs = [(n, {"doi": doi, "eid": eid, "title_normalized": title, "publication_year": year}) for n, doi, eid, title, year in con.execute("SELECT row_number,doi,eid,title_normalized,publication_year FROM export_record WHERE batch_id=%s ORDER BY row_number", (existing[0],))]
        return existing[0], reconcile(recs), True
    # Validate and count before uploading the original; avoid holding a large export in RAM.
    rows = [(n, mapped(raw)) for n, raw in read_scopus(path)]
    uri = store_original(path, sha)
    with con.transaction():
        con.execute("INSERT INTO export_batch(batch_id,query_id,original_uri,original_sha256,original_filename,exported_on,provenance,record_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", (bid, query_id, uri, sha, path.name, export_date, provenance, len(rows)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO export_record(record_id,batch_id,row_number,doi,eid,title,title_normalized,authors,publication_year,source_title,source_id,document_type,source_type,raw_fields) VALUES (" + ",".join(["%s"] * 14) + ")",
                            ((stable_id("record", bid, n), bid, n, rec["doi"], rec["eid"], rec["title"], rec["title_normalized"], rec["authors"], rec["publication_year"], rec["source_title"], rec["source_id"], rec["document_type"], rec["source_type"], Jsonb(raw))
                             for n, raw in read_scopus(path) for rec in (mapped(raw),)))
    return bid, reconcile(rows), False


def report(con, batch_id, version, output):
    seeds = list(con.execute("SELECT seed_id,community,version,doi,eid,title,publication_year,status FROM benchmark_seed WHERE version=%s ORDER BY community,title", (version,)))
    if not seeds:
        raise ValueError("No benchmark seeds for this version")
    records = [dict(zip(("record_id", "doi", "eid", "title_normalized", "publication_year"), r)) for r in con.execute("SELECT record_id,doi,eid,title_normalized,publication_year FROM export_record WHERE batch_id=%s", (batch_id,))]
    run_id = stable_id("matching", batch_id, version, MATCHER_VERSION)
    results = []
    with con.transaction():
        con.execute("INSERT INTO matching_run(run_id,batch_id,benchmark_version,matcher_version) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", (run_id, batch_id, version, MATCHER_VERSION))
        for sid, community, _, doi, eid, title, year, status in seeds:
            outcome, method, candidates, cause = match({"doi": doi, "eid": eid, "title": title, "publication_year": year, "status": status}, records)
            con.execute("INSERT INTO matching_result(run_id,seed_id,outcome,match_method,candidate_record_ids,possible_cause) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id,seed_id) DO NOTHING", (run_id, sid, outcome, method, candidates, cause))
            results.append({"community": community, "seed_id": str(sid), "title": title, "seed_status": status, "outcome": outcome, "match_method": method or "", "candidate_record_ids": ";".join(map(str, candidates)), "possible_cause": cause or ""})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = [f"# Query coverage run {run_id}", "", f"Batch: {batch_id}  |  Benchmark version: {version}", "", "| Community | Found / eligible | Coverage | >=95% target | Out of scope | Ambiguous eligible |", "|---|---:|---:|---|---:|---:|"]
    for community in ("CTO", "TIM"):
        stats = coverage([r for r in results if r["community"] == community])
        pct = "N/A" if stats["coverage"] is None else f"{stats['coverage']:.1%}"
        target = "N/A" if stats["target_met"] is None else ("yes" if stats["target_met"] else "no")
        lines.append(f"| {community} | {stats['found']} / {stats['eligible']} | {pct} | {target} | {stats['out_of_scope']} | {stats['ambiguous']} |")
    lines.extend(["", "## Missing or ambiguous eligible seeds", ""])
    for r in results:
        if r["seed_status"] == "eligible" and r["outcome"] != "found":
            lines.append(f"- **{r['community']} — {r['title']}**: {r['outcome']}; {r['possible_cause']}")
    lines.extend(["", "Coverage counts only human eligible seeds. Ambiguous records do not count as found. Causes are hypotheses requiring review."])
    (output / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_id, lines


def main():
    from .local_env import load_local_env
    load_local_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="research0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p = sub.add_parser("sources-import"); p.add_argument("csv", type=Path)
    p = sub.add_parser("benchmark-import"); p.add_argument("csv", type=Path)
    p = sub.add_parser("query"); p.add_argument("source_version")
    p = sub.add_parser("scopus-import"); p.add_argument("file", type=Path); p.add_argument("--query-id", required=True); p.add_argument("--export-date", required=True); p.add_argument("--provenance", required=True)
    p = sub.add_parser("scopus-inspect"); p.add_argument("file", type=Path)
    p = sub.add_parser("report-files"); p.add_argument("--benchmark-cto", type=Path, required=True); p.add_argument("--benchmark-tim", type=Path, required=True); p.add_argument("--exports", type=Path, nargs="+", required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report"); p.add_argument("--batch-id", required=True); p.add_argument("--benchmark-version", required=True); p.add_argument("--output", type=Path, default=Path("reports"))
    p = sub.add_parser("query-register"); p.add_argument("--label", required=True); p.add_argument("--text-file", type=Path)
    p = sub.add_parser("catalog-build"); p.add_argument("--query-id", required=True)
    p = sub.add_parser("catalog-report"); p.add_argument("--run-id", required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--benchmark-cto", type=Path); p.add_argument("--benchmark-tim", type=Path)
    p = sub.add_parser("dashboard-build"); p.add_argument("--run-id", required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("dashboard-demo"); p.add_argument("--snapshot", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--pilot-run", type=Path); p.add_argument("--assignments", type=Path)
    p = sub.add_parser("dashboard-serve"); p.add_argument("--run-id", required=True); p.add_argument("--port", type=int, default=None)
    p.add_argument("--staged", action="store_true", help="a parallel, read-only dashboard that shows the extraction run waiting in staging instead of the live one")
    p = sub.add_parser("phase2-report"); p.add_argument("--snapshot", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--decisions", type=Path)
    p = sub.add_parser("migrate"); p.add_argument("--run-id", action="append", required=True); p.add_argument("--execute", action="store_true"); p.add_argument("--skip-forum", action="store_true", help="do not copy forum episodes, messages, questions, analyses or proposals"); p.add_argument("--skip-table", action="append", default=[], help="leave this table out of the copy and its verification (repeatable)"); p.add_argument("--output", type=Path, default=Path("reports/migrations"))
    p = sub.add_parser("community-prepare"); p.add_argument("--exports", type=Path, nargs="+", required=True); p.add_argument("--scope", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--batch-size", type=int, default=10); p.add_argument("--limit", type=int)
    p = sub.add_parser("community-submit"); p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("community-collect"); p.add_argument("--run", type=Path, required=True); p.add_argument("--scope", type=Path, required=True)
    p = sub.add_parser("community-retry-prepare"); p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("fulltext-fetch"); p.add_argument("--run-id", required=True); p.add_argument("--scope", choices=("sample", "audit", "candidates", "assigned", "eligible"), default="sample"); p.add_argument("--limit", type=int); p.add_argument("--years", help="publication years, for example 2023-2025")
    p = sub.add_parser("fulltext-import"); p.add_argument("--run-id", required=True); p.add_argument("--folder", type=Path, required=True); p.add_argument("--uploader", required=True)
    p = sub.add_parser("fulltext-import-collection"); p.add_argument("--run-id", required=True); p.add_argument("--folder", type=Path, required=True, help="folder the collection was extracted to (the index paths are relative to it)"); p.add_argument("--index", type=Path, required=True, help="Downloaded_files_index.csv"); p.add_argument("--uploader", required=True); p.add_argument("--output", type=Path, required=True, help="JSON report with one entry per index row"); p.add_argument("--execute", action="store_true", help="without it, only report what would change")
    p = sub.add_parser("fulltext-missing"); p.add_argument("--run-id", required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("fulltext-exclude"); p.add_argument("--run-id", required=True); p.add_argument("--ids-file", type=Path, required=True, help="one paper ID per line"); p.add_argument("--reason", required=True); p.add_argument("--by", required=True); p.add_argument("--delete-files", action="store_true", help="also delete the stored PDFs (irreversible)"); p.add_argument("--execute", action="store_true", help="without it, only report what would change")
    p = sub.add_parser("fulltext-assist"); p.add_argument("--run-id", required=True); p.add_argument("--folder", type=Path, required=True); p.add_argument("--output", type=Path, default=Path("reports/fulltext_assist.html"))
    p = sub.add_parser("fulltext-watch"); p.add_argument("--run-id", required=True); p.add_argument("--folder", type=Path, required=True); p.add_argument("--uploader", required=True); p.add_argument("--once", action="store_true", help="import what is in the folder now and stop")
    from . import pipeline_cli
    pipeline_cli.register(sub)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.command in pipeline_cli.COMMANDS:
        print(json.dumps(pipeline_cli.run(args, database), ensure_ascii=False, default=str))
        return
    if args.command == "scopus-inspect":
        errors = []
        rows = [(n, mapped(raw)) for n, raw in read_scopus(args.file, errors=errors)]
        result = reconcile(rows)
        print(json.dumps({"file": str(args.file), "valid_record_count": result["record_count"], "malformed_row_count": len(errors), "malformed_examples": errors[:10], "missing_doi_count": len(result["missing_doi_rows"]), "missing_eid_count": len(result["missing_eid_rows"]), "potential_duplicate_groups": len(result["potential_duplicates"]), "columns": list(next(read_scopus(args.file))[1])}))
        if errors:
            raise SystemExit(2)
        return
    if args.command == "report-files":
        from .file_report import build_report
        summary, _ = build_report(args.benchmark_cto, args.benchmark_tim, args.exports, args.output)
        print(json.dumps(summary, ensure_ascii=False, default=str))
        return
    if args.command == "phase2-report":
        from .phase2 import build_review_packet
        print(json.dumps(build_review_packet(args.snapshot, args.output, args.decisions), ensure_ascii=False))
        return
    if args.command == "dashboard-demo":
        from .workflow import build_demo
        if args.assignments and not args.pilot_run:
            raise ValueError("--assignments requires --pilot-run")
        print(json.dumps(build_demo(args.snapshot, args.output, args.pilot_run, args.assignments), ensure_ascii=False))
        return
    if args.command == "dashboard-serve":
        from .dashboard_server import serve
        if args.staged:             # the views of the staged run, a search_path that reads them first, and no writes at all
            from . import staging_view
            with database() as con:
                base = con.execute("SELECT current_schema()").fetchone()[0]
                view = staging_view.create(con)
                con.commit()
            os.environ["DATABASE_URL"] = staging_view.connection_url(os.environ["DATABASE_URL"], view, base)
            os.environ["DASHBOARD_STAGING"] = "1"
            server, run_id = serve(args.run_id, args.port or 8766, read_only=True)
        else:
            server, run_id = serve(args.run_id, args.port or 8765)
        print(json.dumps({"event": "dashboard_serving", "run_id": str(run_id), "staged": args.staged, "url": f"http://127.0.0.1:{server.server_address[1]}/"}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
        return
    if args.command == "migrate":
        from .migrate import Target, migrate
        report = migrate([uuid.UUID(r) for r in args.run_id], Target.from_env(), args.execute, args.output, args.skip_forum, args.skip_table)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["ok"] else 1)
    if args.command == "community-prepare":
        from .community_phase1 import prepare
        manifest = prepare(args.exports, args.scope, args.output, args.batch_size, args.limit)
        print(json.dumps({key: manifest[key] for key in ("model", "scope_version", "all_unique_records",
                                                            "ready_records", "selected_records", "request_count",
                                                            "identity_conflicts")}, ensure_ascii=False))
        return
    if args.command == "community-submit":
        from .community_phase1 import submit
        print(json.dumps(submit(args.run), ensure_ascii=False))
        return
    if args.command == "community-collect":
        from .community_phase1 import collect_remote
        print(json.dumps(collect_remote(args.run, args.scope), ensure_ascii=False))
        return
    if args.command == "community-retry-prepare":
        from .community_recovery import prepare_retry
        print(json.dumps(prepare_retry(args.run), ensure_ascii=False))
        return
    with database() as con:
        if args.command == "init":
            from .migrate import apply_migrations
            for version in apply_migrations(con):
                print(json.dumps({"event": "migration_applied", "version": version}))
        elif args.command == "sources-import":
            print(json.dumps({"event": "sources_imported", "count": import_sources(con, args.csv)}))
        elif args.command == "benchmark-import":
            print(json.dumps({"event": "benchmark_imported", "count": import_seeds(con, args.csv)}))
        elif args.command == "query":
            qid, query = make_query(con, args.source_version)
            print(json.dumps({"query_id": str(qid), "source_version": args.source_version, "query": query}))
        elif args.command == "query-register":
            qid, text = register_query(con, args.label, args.text_file)
            print(json.dumps({"query_id": str(qid), "label": args.label, "query": text}))
        elif args.command == "catalog-build":
            from .catalog import build_catalog
            run_id, summary, already = build_catalog(con, uuid.UUID(args.query_id))
            print(json.dumps({"event": "catalog_built", "run_id": str(run_id), "already_built": already, "summary": summary}))
        elif args.command == "fulltext-fetch":
            from . import fulltext
            email = os.environ.get("UNPAYWALL_EMAIL", "").strip()
            if "@" not in email:
                raise ValueError("Set UNPAYWALL_EMAIL in .env (a real address; Unpaywall asks for it)")
            papers = fulltext.select_papers(con, uuid.UUID(args.run_id), args.scope, args.limit, fulltext.parse_years(args.years))
            con.commit()  # end the selection transaction so each recorded attempt commits independently
            outcomes = fulltext.process(con, uuid.UUID(args.run_id), papers, email)
            print(json.dumps({"event": "fulltext_fetched", "run_id": args.run_id, "scope": args.scope, "papers": len(papers), "outcomes": outcomes,
                              "coverage": fulltext.coverage(con, uuid.UUID(args.run_id), args.scope, fulltext.parse_years(args.years))}))
        elif args.command == "fulltext-import":
            from . import fulltext
            report = fulltext.import_folder(con, fulltext.S3Store(), uuid.UUID(args.run_id), args.folder, args.uploader)
            print(json.dumps({"event": "fulltext_imported", "run_id": args.run_id, **report}, ensure_ascii=False))
        elif args.command == "fulltext-import-collection":
            from . import fulltext, fulltext_collection
            report = fulltext_collection.import_collection(con, fulltext.S3Store() if args.execute else None, uuid.UUID(args.run_id), args.folder, args.index, args.uploader, args.execute)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps({"event": "fulltext_collection", **{k: v for k, v in report.items() if k != "entries"}, "output": str(args.output)}, ensure_ascii=False))
        elif args.command == "fulltext-missing":
            from . import fulltext
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(fulltext.missing_csv(con, uuid.UUID(args.run_id)), encoding="utf-8", newline="")
            print(json.dumps({"event": "fulltext_missing_written", "run_id": args.run_id, "output": str(args.output)}))
        elif args.command == "fulltext-exclude":
            from . import fulltext
            run_id = uuid.UUID(args.run_id)
            ids = sorted({uuid.UUID(line.strip()) for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()})
            held = con.execute("SELECT count(*), count(DISTINCT canonical_id) FROM fulltext_file WHERE run_id=%s AND canonical_id=ANY(%s)", (run_id, ids)).fetchone()
            report = {"event": "fulltext_exclude_dry_run", "papers": len(ids), "pdf_files": held[0], "papers_with_pdf": held[1], "delete_files": args.delete_files}
            if args.execute:
                fulltext.exclude_papers(con, run_id, ids, args.reason, args.by)
                report = {**report, "event": "fulltext_excluded"}
                if args.delete_files:
                    report["purged"] = fulltext.purge_files(con, fulltext.S3Store(), run_id, ids)
            print(json.dumps(report, ensure_ascii=False))
        elif args.command == "fulltext-assist":
            from . import fulltext_assist
            args.output.parent.mkdir(parents=True, exist_ok=True)
            count = fulltext_assist.write_page(con, uuid.UUID(args.run_id), args.folder, args.output)
            print(json.dumps({"event": "fulltext_assist_page_written", "run_id": args.run_id, "papers": count, "output": str(args.output), "folder": str(args.folder)}))
        elif args.command == "fulltext-watch":
            from . import fulltext, fulltext_assist
            print(json.dumps({"event": "fulltext_watch_started", "run_id": args.run_id, "folder": str(args.folder), "stop": "Ctrl+C"}), flush=True)
            summary = fulltext_assist.watch(con, fulltext.S3Store(), uuid.UUID(args.run_id), args.folder, args.uploader, once=args.once,
                                            emit=lambda line: print(line, flush=True))
            print(json.dumps({"event": "fulltext_watch_finished", "run_id": args.run_id, **summary}))
        elif args.command == "dashboard-build":
            from .dashboard import build_snapshot, render
            snapshot = build_snapshot(con, uuid.UUID(args.run_id))
            page = render(snapshot, args.output)
            print(json.dumps({"event": "dashboard_built", "run_id": args.run_id, "page": str(page), "items": len(snapshot["items"])}))
        elif args.command == "catalog-report":
            from .catalog_report import write_reports
            if bool(args.benchmark_cto) != bool(args.benchmark_tim):
                raise ValueError("Provide both --benchmark-cto and --benchmark-tim, or neither")
            seed_files = [(args.benchmark_cto, "CTO"), (args.benchmark_tim, "TIM")] if args.benchmark_cto else []
            summary, seeds = write_reports(con, uuid.UUID(args.run_id), args.output, seed_files)
            print(json.dumps({"event": "catalog_report_written", "run_id": args.run_id, "output": str(args.output), "seeds_checked": len(seeds)}))
        elif args.command == "scopus-import":
            bid, reconciliation, already = import_scopus(con, args.file, args.query_id, dt.date.fromisoformat(args.export_date), args.provenance)
            print(json.dumps({"event": "scopus_imported", "batch_id": str(bid), "already_imported": already, "reconciliation": reconciliation}, default=str))
        elif args.command == "report":
            run_id, lines = report(con, args.batch_id, args.benchmark_version, args.output)
            print("\n".join(lines))


if __name__ == "__main__":
    main()
