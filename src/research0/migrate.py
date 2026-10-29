"""Copy one catalog run, with everything it depends on, from the current database and bucket to a target.

The target is any PostgreSQL with pgvector plus any S3-compatible bucket, typically AWS RDS and S3. The copy
is idempotent (rerunning inserts nothing new), append-only (nothing in the target is updated or deleted), and
verified: row counts and content hashes per table, and size plus SHA-256 per original export file.
Without execute=True it only reports what would happen.
"""
import datetime as dt
import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import psycopg
from botocore.exceptions import ClientError

from .core import file_digest

LOG = logging.getLogger("research0")
MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
BATCH_ROWS = 5000   # rows per committed batch of a table copy

BATCHES = "SELECT batch_id FROM export_batch WHERE query_id = ANY(%(queries)s)"
RUNS = "SELECT run_id FROM catalog_run WHERE query_id = ANY(%(queries)s)"
PAPERS = f"SELECT canonical_id FROM canonical_paper WHERE run_id IN ({RUNS})"
INGESTIONS = f"SELECT ingestion_run_id FROM ingestion_run WHERE run_id IN ({RUNS})"
VERSIONS = f"SELECT version_id FROM paper_version WHERE ingestion_run_id IN ({INGESTIONS})"
EXTRACTIONS = f"SELECT extraction_run_id FROM extraction_run WHERE ingestion_run_id IN ({INGESTIONS})"
SIGNALS = f"SELECT signal_run_id FROM signal_run WHERE extraction_run_id IN ({EXTRACTIONS})"
STAGED = f"SELECT extraction_run_id FROM staged_extraction_run WHERE ingestion_run_id IN ({INGESTIONS})"     # runs waiting for promotion (migration 026)
CLUSTERS = f"SELECT cluster_run_id FROM cluster_run WHERE run_id IN ({RUNS})"
PREASSIGNMENTS = f"SELECT preassignment_run_id FROM preassignment_run WHERE run_id IN ({RUNS})"
AUDITS = f"SELECT audit_id FROM audit_sample WHERE run_id IN ({RUNS})"
RELEASES = f"SELECT release_id FROM release WHERE run_id IN ({RUNS})"
EMBEDDING_MODELS = (f"SELECT model_id FROM paper_embedding WHERE canonical_id IN ({PAPERS}) UNION SELECT model_id FROM cluster_run WHERE run_id IN ({RUNS}) "
                    f"UNION SELECT model_id FROM chunk_embedding WHERE chunk_id IN (SELECT chunk_id FROM document_chunk WHERE version_id IN ({VERSIONS})) "
                    f"UNION SELECT model_id FROM signal_run WHERE extraction_run_id IN ({EXTRACTIONS}) "
                    f"UNION SELECT model_id FROM record_embedding WHERE extraction_run_id IN ({EXTRACTIONS})")
DECISION_SUBJECTS = (f"SELECT mention_id FROM construct_mention WHERE extraction_run_id IN ({EXTRACTIONS}) UNION SELECT claim_id FROM nomological_claim WHERE extraction_run_id IN ({EXTRACTIONS}) "
                     f"UNION SELECT measure_id FROM construct_measure WHERE mention_id IN (SELECT mention_id FROM construct_mention WHERE extraction_run_id IN ({EXTRACTIONS})) "
                     f"UNION SELECT version_id FROM paper_version WHERE ingestion_run_id IN ({INGESTIONS})")
GOLD = f"SELECT e.v::uuid FROM release r, jsonb_each_text(r.manifest->'runs') AS e(k,v) WHERE r.run_id IN ({RUNS}) AND e.k LIKE 'gold\\_%%' AND e.v IS NOT NULL"
# Parents before children. (table, scope, columns hashed for verification or None for the whole row)
TABLES = [
    ("query_version", "query_id = ANY(%(queries)s)", None),
    ("export_batch", "query_id = ANY(%(queries)s)",
     "t.batch_id,t.query_id,t.original_sha256,t.original_filename,t.exported_on,t.provenance,t.record_count"),  # URI is rewritten
    ("export_record", f"batch_id IN ({BATCHES})", None),
    ("catalog_run", "query_id = ANY(%(queries)s)", None),
    ("canonical_paper", f"run_id IN ({RUNS})", None),
    ("catalog_member", f"run_id IN ({RUNS})", None),
    ("dedup_log", f"run_id IN ({RUNS})", None),
    ("catalog_seed_result", f"run_id IN ({RUNS})", None),
    ("validation_decision", f"run_id IN ({RUNS})", None),
    ("source_list", "version = ANY(%(source_versions)s)", None),
    ("source_register", "version = ANY(%(source_versions)s)", None),
    ("benchmark_seed", "version = ANY(%(benchmark_versions)s)", None),
    ("matching_run", f"batch_id IN ({BATCHES})", None),
    ("matching_result", f"run_id IN (SELECT run_id FROM matching_run WHERE batch_id IN ({BATCHES})) "
                        "AND seed_id IN (SELECT seed_id FROM benchmark_seed WHERE version = ANY(%(benchmark_versions)s))", None),
    # Community validation (blind sample, ratings, adjudication)
    ("blind_sample", f"run_id IN ({RUNS})", None),
    ("blind_sample_paper", f"run_id IN ({RUNS})", None),
    ("blind_rating", f"run_id IN ({RUNS})", None),
    ("community_adjudication", f"run_id IN ({RUNS})", None),
    # Full text: the rows here, the PDF bytes below (s3_uri is rewritten to the target bucket, so it is not part of the hash)
    ("fulltext_file", f"run_id IN ({RUNS})",
     "t.file_id,t.run_id,t.canonical_id,t.origin,t.sha256,t.size_bytes,t.source_url,t.original_filename,t.acquired_by,t.validation_status,t.created_at,t.version_label"),
    ("fulltext_attempt", f"run_id IN ({RUNS})", None),
    ("fulltext_exclusion", f"run_id IN ({RUNS})", None),
    # Phase 2
    ("embedding_model", f"model_id IN ({EMBEDDING_MODELS})", None),
    ("paper_embedding", f"canonical_id IN ({PAPERS})", None),
    ("cluster_run", f"run_id IN ({RUNS})", None),
    ("cluster_assignment", f"cluster_run_id IN ({CLUSTERS})", None),
    # Phase 5
    ("ingestion_run", f"run_id IN ({RUNS})", None),
    ("paper_version", f"ingestion_run_id IN ({INGESTIONS})", None),
    ("document_chunk", f"version_id IN ({VERSIONS})", None),
    ("chunk_embedding", f"chunk_id IN (SELECT chunk_id FROM document_chunk WHERE version_id IN ({VERSIONS}))", None),
    # Migration 029: the batches sent by the dashboard click and the methodology / contribution cards that came back. Both hang off paper_version.
    ("processing_item", f"version_id IN ({VERSIONS})", None),
    ("paper_card", f"version_id IN ({VERSIONS})", None),
    # Phases 3-4
    ("preassignment_run", f"run_id IN ({RUNS})", None),
    ("community_preassignment", f"preassignment_run_id IN ({PREASSIGNMENTS})", None),
    ("audit_sample", f"run_id IN ({RUNS})", None),
    ("audit_sample_paper", f"audit_id IN ({AUDITS})", None),
    ("audit_rating", f"audit_id IN ({AUDITS})", None),
    ("audit_adjudication", f"audit_id IN ({AUDITS})", None),
    ("extraction_run", f"ingestion_run_id IN ({INGESTIONS})", None),
    ("construct_mention", f"extraction_run_id IN ({EXTRACTIONS})", None),
    ("extraction_attribute", f"mention_id IN (SELECT mention_id FROM construct_mention WHERE extraction_run_id IN ({EXTRACTIONS}))", None),
    ("construct_measure", f"mention_id IN (SELECT mention_id FROM construct_mention WHERE extraction_run_id IN ({EXTRACTIONS}))", None),
    ("nomological_claim", f"extraction_run_id IN ({EXTRACTIONS})", None),
    ("claim_modifier", f"claim_id IN (SELECT claim_id FROM nomological_claim WHERE extraction_run_id IN ({EXTRACTIONS}))", None),
    ("decision_event", f"subject_id IN ({DECISION_SUBJECTS})", None),
    # Extraction runs in staging travel with the data and stay in staging on the target: only a person's promotion moves them into the live tables.
    ("staged_extraction_run", f"ingestion_run_id IN ({INGESTIONS})", None),
    ("staged_construct_mention", f"extraction_run_id IN ({STAGED})", None),
    ("staged_extraction_attribute", f"mention_id IN (SELECT mention_id FROM staged_construct_mention WHERE extraction_run_id IN ({STAGED}))", None),
    ("staged_construct_measure", f"mention_id IN (SELECT mention_id FROM staged_construct_mention WHERE extraction_run_id IN ({STAGED}))", None),
    ("staged_nomological_claim", f"extraction_run_id IN ({STAGED})", None),
    ("staged_claim_modifier", f"claim_id IN (SELECT claim_id FROM staged_nomological_claim WHERE extraction_run_id IN ({STAGED}))", None),
    ("extraction_promotion", f"extraction_run_id IN ({STAGED})", None),
    # Phases 6-8
    ("signal_run", f"extraction_run_id IN ({EXTRACTIONS})", None),
    ("construct_node", f"signal_run_id IN ({SIGNALS})", None),
    ("candidate_pair", f"signal_run_id IN ({SIGNALS})", None),
    ("pair_dossier", f"pair_id IN (SELECT pair_id FROM candidate_pair WHERE signal_run_id IN ({SIGNALS}))", None),
    ("pair_proposal", f"pair_id IN (SELECT pair_id FROM candidate_pair WHERE signal_run_id IN ({SIGNALS}))", None),
    ("pair_sample", f"signal_run_id IN ({SIGNALS})", None),
    ("pair_sample_item", f"pair_sample_id IN (SELECT pair_sample_id FROM pair_sample WHERE signal_run_id IN ({SIGNALS}))", None),
    ("pair_decision", f"pair_id IN (SELECT pair_id FROM candidate_pair WHERE signal_run_id IN ({SIGNALS}))", None),
    ("gold_set", f"gold_id IN ({GOLD})", None),
    ("agent_profile", "true", None),
    ("release", f"run_id IN ({RUNS})", None),
    ("release_signoff", f"release_id IN ({RELEASES})", None),
    ("retrieval_run", f"release_id IN ({RELEASES})", None),
    ("retrieval_hit", f"retrieval_run_id IN (SELECT retrieval_run_id FROM retrieval_run WHERE release_id IN ({RELEASES}))", None),
    # Discovery forum (migration 020). forum_episode.status is a lifecycle column: an episode copied while running will differ from its source later.
    ("forum_episode", f"release_id IN ({RELEASES})", None),
    ("forum_agent", f"episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES}))", None),
    ("forum_message", f"episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES}))", None),
    ("question", f"release_id IN ({RELEASES})", None),
    # Migration 022: code the agents ran (and what it produced) and the proposals written from the questions.
    ("forum_analysis", f"episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES}))", None),
    ("forum_analysis_file", f"episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES}))", None),
    ("forum_proposal", f"episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES}))", None),
    ("forum_proposal_evaluation", f"proposal_id IN (SELECT proposal_id FROM forum_proposal WHERE episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES})))", None),   # migration 023
    ("record_embedding", f"extraction_run_id IN ({EXTRACTIONS})", None),     # migration 021: embeddings of the construct records
    # Migration 027, publishability evaluator. The benchmark is not tied to a catalog run, so it travels whole (parents before children). It holds introductions of published
    # papers: take these tables out of the plan if the texts must not leave this machine.
    ("pub_journal_list", "TRUE", None),
    ("pub_dataset", "TRUE", None),
    ("pub_paper", "TRUE", None),
    ("pub_rejection", "TRUE", None),
    ("pub_split", "TRUE", None),
    ("pub_split_item", "TRUE", None),
    ("pub_config", "TRUE", None),
    ("pub_experiment", "TRUE", None),
    ("pub_experiment_event", "TRUE", None),
    ("pub_candidate", "TRUE", None),
    ("pub_result", "TRUE", None),
    ("pub_call", "TRUE", None),
    ("pub_freeze", "TRUE", None),
    ("pub_holdout_access", "TRUE", None),
    ("pub_test_use", "TRUE", None),
    ("pub_activation", "TRUE", None),
    ("pub_forum_prediction", f"proposal_id IN (SELECT proposal_id FROM forum_proposal WHERE episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES})))", None),
    ("forum_baseline_evaluation", f"proposal_id IN (SELECT proposal_id FROM forum_proposal WHERE episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES})))", None),   # migration 028
    ("forum_proposal_score", f"proposal_id IN (SELECT proposal_id FROM forum_proposal WHERE episode_id IN (SELECT episode_id FROM forum_episode WHERE release_id IN ({RELEASES})))", None),   # migration 030
    ("evaluation_paper", "TRUE", None),     # migration 031: the evaluation of published papers is not tied to a catalog run (title, journal and ABS grade; never the abstract)
    ("evaluation_run", "TRUE", None),
    ("evaluation_score", "TRUE", None),
    ("evaluation_fulltext", "TRUE", None),  # migration 032: what happened to each paper's full text (the PDF and the parsed text stay in the local folder)
    ("evaluation_card", "TRUE", None),      # migration 032: the two cards of each paper, as the corpus cards are made
]
# What the discovery forum generates (episodes, messages, archived questions, analysis code, proposals and their evaluation). `agent_profile` is configuration, not generated, so it still travels.
FORUM_TABLES = {"forum_episode", "forum_agent", "forum_message", "question", "forum_analysis", "forum_analysis_file", "forum_proposal", "forum_proposal_evaluation",
               "pub_forum_prediction", "forum_baseline_evaluation", "forum_proposal_score"}
# Every agent of a forum episode gets an `agent_profile` row and logs its searches in `retrieval_run`/`retrieval_hit` under the agent name `forum:<episode>:<role>`: byproducts of the
# episode, so with skip_forum they stay behind too. The tables themselves still travel (searches made by anyone else, none today, would be copied).
FORUM_BYPRODUCT_SCOPES = {
    "agent_profile": "agent_id NOT LIKE 'forum:%%'",
    "retrieval_run": f"release_id IN ({RELEASES}) AND agent NOT LIKE 'forum:%%'",
    "retrieval_hit": f"retrieval_run_id IN (SELECT retrieval_run_id FROM retrieval_run WHERE release_id IN ({RELEASES}) AND agent NOT LIKE 'forum:%%')",
}


@dataclass
class Target:
    database_url: str
    bucket: str
    region: str | None = None
    endpoint_url: str | None = None
    profile: str | None = None
    access_key: str | None = None
    secret_key: str | None = None

    @classmethod
    def from_env(cls):
        missing = [k for k in ("TARGET_DATABASE_URL", "TARGET_S3_BUCKET") if not os.getenv(k)]
        if missing:
            raise ValueError("Set " + " and ".join(missing) + " (see .env.example)")
        return cls(os.environ["TARGET_DATABASE_URL"], os.environ["TARGET_S3_BUCKET"], os.getenv("TARGET_S3_REGION"),
                   os.getenv("TARGET_S3_ENDPOINT_URL"), os.getenv("TARGET_AWS_PROFILE"),
                   os.getenv("TARGET_AWS_ACCESS_KEY_ID"), os.getenv("TARGET_AWS_SECRET_ACCESS_KEY"))

    def s3(self):
        # An explicit profile makes boto3 ignore the AWS_* variables, which hold local-only keys.
        if self.access_key:
            session = boto3.Session(aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        else:
            session = boto3.Session(profile_name=self.profile or "default")
        region = self.region or session.region_name
        if not region:
            raise ValueError("Set TARGET_S3_REGION or a Region in the AWS profile")
        return session.client("s3", region_name=region, endpoint_url=self.endpoint_url or None), region


def source_s3():
    return boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"), region_name=os.getenv("S3_REGION", "us-east-1"))


def apply_migrations(con):
    applied = []
    for migration in sorted(MIGRATIONS.glob("*.sql")):
        con.execute(migration.read_text(encoding="utf-8"))
        applied.append(migration.stem)
    return applied


def connect(url):
    con = psycopg.connect(url)
    con.execute("SET TIME ZONE 'UTC'")
    return con


def select_scope(con, run_ids):
    rows = con.execute("SELECT run_id,query_id FROM catalog_run WHERE run_id = ANY(%s)", (run_ids,)).fetchall()
    if len(rows) != len(set(run_ids)):
        raise ValueError("Unknown catalog run id")
    queries = sorted({q for _, q in rows}, key=str)
    return {"queries": queries,
            "source_versions": [r[0] for r in con.execute("SELECT DISTINCT source_version FROM query_version WHERE query_id = ANY(%s)", (queries,))],
            "benchmark_versions": [r[0] for r in con.execute(
                "SELECT DISTINCT benchmark_version FROM catalog_seed_result WHERE run_id = ANY(%s)", ([r for r, _ in rows],))]}


def columns(con, table):
    # Generated columns (document_chunk.tsv) are rebuilt by the target itself and cannot be copied.
    return [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s AND is_generated='NEVER' ORDER BY ordinal_position", (table,))]


def primary_key(con, table):
    return [r[0] for r in con.execute(
        "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid=%s::regclass AND i.indisprimary ORDER BY array_position(i.indkey::int2[], a.attnum)", (table,))]


def table_stats(con, table, scope, hashed, params):
    # Streamed: one md5 per row in primary-key order, combined here, so neither server builds the whole
    # table as one string in memory (string_agg over export_record can get Postgres OOM-killed on a 2 GB instance).
    order = ",".join(f"t.{c}" for c in primary_key(con, table))
    expr = f"({hashed})::text" if hashed else "t::text"
    digest, count = hashlib.md5(), 0
    with con.cursor(name=f"stats_{table}") as cur:
        cur.itersize = 10000
        cur.execute(f"SELECT md5({expr}) FROM {table} t WHERE {scope} ORDER BY {order}", params)
        for (row_hash,) in cur:
            digest.update(row_hash.encode())
            count += 1
    return count, digest.hexdigest()


def parse_uri(uri):
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def ensure_target_bucket(client, bucket, region, compatible):
    try:
        client.head_bucket(Bucket=bucket)
        return "exists"
    except ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
            raise
    args = {"Bucket": bucket}
    if region != "us-east-1" and not compatible:
        args["CreateBucketConfiguration"] = {"LocationConstraint": region}
    client.create_bucket(**args)
    if not compatible:  # AWS S3: private, versioned, TLS only
        client.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        client.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
        client.put_bucket_policy(Bucket=bucket, Policy=json.dumps({"Version": "2012-10-17", "Statement": [{
            "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "s3:*",
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"], "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}))
    return "created"


def head(client, bucket, key):
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 or exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def copy_table(src, dst, table, scope, params, cols, target_bucket, migration_id=None, batch_rows=None):
    """Copy in batches of batch_rows in primary-key order, one commit per batch, so that a big table never sits in one huge INSERT on a small target and an
    interrupted copy keeps what it committed. Batch bounds are read from the source only: the two servers may sort text differently."""
    collist = ",".join(cols)
    batch_rows = batch_rows or BATCH_ROWS
    order = ",".join(f"t.{c}" for c in primary_key(src, table))
    after, inserted, batches = None, 0, 0
    while True:
        where, bound = scope, dict(params)
        if after is not None:
            where = f"({scope}) AND ({order}) > ({','.join(f'%(_after{i})s' for i in range(len(after)))})"
            bound.update({f"_after{i}": v for i, v in enumerate(after)})
        last = src.execute(f"SELECT {order} FROM {table} t WHERE {where} ORDER BY {order} OFFSET {batch_rows - 1} LIMIT 1", bound).fetchone()
        upto = where
        if last is not None:
            upto = f"({where}) AND ({order}) <= ({','.join(f'%(_last{i})s' for i in range(len(last)))})"
            bound.update({f"_last{i}": v for i, v in enumerate(last)})
        dst.execute(f"CREATE TEMP TABLE stage (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
        with src.cursor() as out_cur, dst.cursor() as in_cur:
            with out_cur.copy(f"COPY (SELECT {collist} FROM {table} t WHERE {upto}) TO STDOUT", bound) as out, in_cur.copy(f"COPY stage ({collist}) FROM STDIN") as inn:
                for chunk in out:
                    inn.write(chunk)
        if table == "export_batch":
            dst.execute("UPDATE stage SET original_uri = regexp_replace(original_uri, '^s3://[^/]+/', %s)", (f"s3://{target_bucket}/",))
        if table == "fulltext_file":
            dst.execute("UPDATE stage SET s3_uri = regexp_replace(s3_uri, '^s3://[^/]+/', %s)", (f"s3://{target_bucket}/",))
        inserted += dst.execute(f"INSERT INTO {table} ({collist}) SELECT {collist} FROM stage ON CONFLICT DO NOTHING").rowcount
        dst.commit()
        batches += 1
        if last is None:
            return inserted
        LOG.info(json.dumps({"event": "table_batch", "migration_id": migration_id, "table": table, "batches": batches, "inserted": inserted}))
        after = last


def migrate(run_ids, target, execute=False, output=None, skip_forum=False, skip_tables=()):
    """Return a report dict; ok is true only when every table and file verified. With skip_forum the tables in FORUM_TABLES are neither copied nor verified;
    the tables named in skip_tables are left out the same way (the report keeps their names)."""
    unknown = sorted(set(skip_tables) - {name for name, _, _ in TABLES})
    if unknown:
        raise ValueError("Unknown table(s) to skip: " + ", ".join(unknown))
    started = dt.datetime.now(dt.timezone.utc)
    migration_id = str(uuid.uuid4())
    tables = [(name, FORUM_BYPRODUCT_SCOPES.get(name, scope) if skip_forum else scope, hashed) for name, scope, hashed in TABLES
              if not (skip_forum and name in FORUM_TABLES) and name not in skip_tables]
    report = {"migration_id": migration_id, "started_at": started.isoformat(timespec="seconds"), "mode": "execute" if execute else "dry-run",
              "run_ids": [str(r) for r in run_ids], "skip_forum": skip_forum, "skip_tables": sorted(skip_tables), "tables": {}, "files": [], "problems": []}
    src = connect(os.environ["DATABASE_URL"])
    scope_params = select_scope(src, run_ids)
    params = {k: v for k, v in scope_params.items()}
    report["scope"] = {k: [str(x) for x in v] for k, v in scope_params.items()}
    batches = src.execute("SELECT original_uri,original_sha256,original_filename FROM export_batch WHERE query_id = ANY(%(queries)s) ORDER BY original_filename", params).fetchall()
    source_client = source_s3()
    pdfs = src.execute(f"SELECT s3_uri,sha256,canonical_id::text || '.pdf' FROM fulltext_file WHERE run_id IN ({RUNS}) ORDER BY canonical_id,created_at", params).fetchall()
    for kind, rows in (("export", batches), ("pdf", pdfs)):
        for uri, sha, name in rows:
            bucket, key = parse_uri(uri)
            meta = head(source_client, bucket, key)
            report["files"].append({"kind": kind, "file": name, "sha256": sha, "source_uri": uri, "source_size": meta["ContentLength"] if meta else None})
            if meta is None:
                report["problems"].append(f"{'Original file' if kind == 'export' else 'PDF'} missing in the source bucket: {uri}")
    for table, scope, hashed in tables:
        count = src.execute(f"SELECT count(*) FROM {table} t WHERE {scope}", params).fetchone()[0]
        report["tables"][table] = {"source_rows": count}
    LOG.info(json.dumps({"event": "migration_plan", "migration_id": migration_id, "mode": report["mode"], "rows": sum(t["source_rows"] for t in report["tables"].values()), "files": len(batches) + len(pdfs)}))

    dst = None
    try:
        dst = connect(target.database_url)
        report["target_reachable"] = True
    except psycopg.Error as exc:
        report["target_reachable"] = False
        report["problems"].append(f"Target database not reachable: {str(exc).splitlines()[0][:200]}")
    s3_client = region = None
    try:
        s3_client, region = target.s3()
        exists = True
        try:
            s3_client.head_bucket(Bucket=target.bucket)
        except ClientError as exc:
            exists = False if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404 else None
            if exists is None:
                raise
        report["target_bucket"] = {"name": target.bucket, "region": region, "exists": exists}
    except Exception as exc:  # reported, not hidden: credentials, Region and permissions are the usual causes
        report["problems"].append(f"Target bucket not usable: {type(exc).__name__}: {str(exc)[:200]}")
    if not execute or report["problems"] or dst is None or s3_client is None:
        if execute and report["problems"]:
            report["mode"] = "aborted"
        report["ok"] = execute is False and not report["problems"]
        return _finish(report, output, src, dst)

    report["migrations_applied"] = apply_migrations(dst)
    dst.commit()
    report["target_bucket"]["action"] = ensure_target_bucket(s3_client, target.bucket, region, bool(target.endpoint_url))
    for table, scope, hashed in tables:
        cols = columns(src, table)
        if cols != columns(dst, table):
            report["problems"].append(f"Schema differs for {table}; apply the same migrations to both sides")
            continue
        # A table the target already holds in full (an earlier run that stopped later on) is not sent again; the verification below still compares its content.
        if dst.execute(f"SELECT count(*) FROM {table} t WHERE {scope}", params).fetchone()[0] == report["tables"][table]["source_rows"]:
            report["tables"][table].update(inserted=0, skipped="already complete")
            LOG.info(json.dumps({"event": "table_skipped", "migration_id": migration_id, "table": table, **report["tables"][table]}))
            continue
        report["tables"][table]["inserted"] = copy_table(src, dst, table, scope, params, cols, target.bucket, migration_id)
        LOG.info(json.dumps({"event": "table_copied", "migration_id": migration_id, "table": table, **report["tables"][table]}))
    # Original export files: the database stores URI, checksum and provenance; the bytes follow.
    for entry, (uri, sha, name) in zip(report["files"], [*batches, *pdfs]):
        bucket, key = parse_uri(uri)
        want = head(source_client, bucket, key)
        if want is None:
            continue
        have = head(s3_client, target.bucket, key)
        if have and have["ContentLength"] == want["ContentLength"] and have.get("Metadata", {}).get("sha256") == sha:
            entry["action"] = "already present"
        else:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "original"
                source_client.download_file(bucket, key, str(path))
                if file_digest(path) != sha:
                    report["problems"].append(f"Checksum mismatch in the source file {uri}; not copied")
                    continue
                s3_client.upload_file(str(path), target.bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
            entry["action"] = "copied"
        landed = head(s3_client, target.bucket, key)
        entry["target_uri"] = f"s3://{target.bucket}/{key}"
        entry["verified"] = bool(landed and landed["ContentLength"] == want["ContentLength"] and landed.get("Metadata", {}).get("sha256") == sha)
        if not entry["verified"]:
            report["problems"].append(f"File not verified in the target: {key}")
    for table, scope, hashed in tables:
        a, b = table_stats(src, table, scope, hashed, params), table_stats(dst, table, scope, hashed, params)
        report["tables"][table].update(target_rows=b[0], verified=a == b)
        if a != b:
            report["problems"].append(f"{table}: source and target differ (rows {a[0]} vs {b[0]}, or content hash)")
    report["ok"] = not report["problems"]
    return _finish(report, output, src, dst)


def _finish(report, output, *connections):
    for con in connections:
        if con is not None:
            con.close()
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / f"migration_{report['migration_id'][:8]}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info(json.dumps({"event": "migration_finished", "migration_id": report["migration_id"], "mode": report["mode"], "ok": report["ok"], "problems": len(report["problems"])}))
    return report
