"""Migration of everything Phases 2-8 add: rows (including pgvector and generated columns) and the PDFs. Source: a MOCK world."""
import os
import uuid
from urllib.parse import urlparse, urlunparse

import boto3
import psycopg
import pytest

from research0 import community_v2, documents, explore, extraction, mock_world, pairs, quality, release, signals
from research0.embeddings import HashEmbedder
from research0.migrate import TABLES, Target, migrate

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL") or not os.getenv("S3_BUCKET"), reason="Set local PostgreSQL and S3 environment to run integration test")


def client():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="us-east-1")


def empty_and_delete(bucket):
    s3 = client()
    for obj in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
        s3.delete_object(Bucket=bucket, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket)


def test_every_table_with_data_is_in_the_migration_plan():
    planned = {t for t, _, _ in TABLES}
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        existing = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'")}
    assert existing <= planned, f"tables missing from the migration plan: {existing - planned}"


def test_pipeline_data_and_pdfs_move_verified_and_idempotent(tmp_path, monkeypatch):
    tag = uuid.uuid4().hex[:10]
    source_bucket, target_bucket, database = f"migsrc-{tag}", f"migdst-{tag}", f"migtest_{tag}"
    world = mock_world.create_world(n=120, seed=31)
    try:
        s3 = client()
        s3.create_bucket(Bucket=source_bucket)
        with world.connect() as con:
            embedder = HashEmbedder()
            explore.embed_papers(con, world.run_id, embedder)
            explore.cluster(con, world.run_id, embedder, k=3)
            pre, audit_id = mock_world.mock_audit(con, world, n_total=60)
            ingestion, _ = documents.ingest(con, world.run_id, world.store)
            documents.embed_chunks(con, ingestion, embedder)
            directory = tmp_path / "x"
            extraction.prepare(con, ingestion, directory)
            owner = {str(a): b for a, b in con.execute("SELECT version_id,canonical_id FROM paper_version")}
            lines = (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (directory / "batch_output.jsonl").write_text("\n".join(mock_world.scripted_extraction_output(world, lines, owner)) + "\n", encoding="utf-8")
            extracted = extraction.collect_file(con, directory)
            mention = con.execute("SELECT mention_id FROM construct_mention LIMIT 1").fetchone()[0]
            extraction.record_decision(con, "construct_mention", mention, "validated", "Ana")
            signal_run, _, _ = signals.build(con, uuid.UUID(extracted["extraction_run_id"]), embedder)
            signals.build_dossiers(con, signal_run, limit=3)
            pair = con.execute("SELECT pair_id FROM pair_dossier LIMIT 1").fetchone()[0]
            pairs.decide(con, pair, "Ana", "distinct", "r", {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"})
            gold = quality.store_gold(con, "pairs", "g", {"items": [{"canonical_id_a": str(uuid.uuid4()), "label_a": "a", "canonical_id_b": str(uuid.uuid4()), "label_b": "b", "class": "conceptual_jangle"}]}, "Ana")
            release_id, *_ = release.build(con, world.run_id, "mig", pre, audit_id, ingestion, uuid.UUID(extracted["extraction_run_id"]), signal_run, gold_pairs=gold)
            # The mock world keeps its PDFs in memory and has no export file: put real objects in a real bucket and point the rows at them.
            export = b"Title\nMock\n"
            import hashlib
            s3.put_object(Bucket=source_bucket, Key="scopus/originals/mock/mock.csv", Body=export, Metadata={"sha256": hashlib.sha256(export).hexdigest()})
            con.execute("UPDATE export_batch SET original_uri=%s, original_sha256=%s", (f"s3://{source_bucket}/scopus/originals/mock/mock.csv", hashlib.sha256(export).hexdigest()))
            for file_id, uri, sha in con.execute("SELECT file_id,s3_uri,sha256 FROM fulltext_file").fetchall():
                key = uri.removeprefix("mock://")
                s3.put_object(Bucket=source_bucket, Key=key, Body=world.store.objects[key], Metadata={"sha256": sha})
                con.execute("UPDATE fulltext_file SET s3_uri=%s WHERE file_id=%s", (f"s3://{source_bucket}/{key}", file_id))
            con.commit()
        monkeypatch.setenv("DATABASE_URL", world.url)
        parsed = urlparse(os.environ["DATABASE_URL"].split("?")[0])
        with psycopg.connect(os.environ["DATABASE_URL"].split("?")[0], autocommit=True) as admin:
            admin.execute(f"CREATE DATABASE {database}")
        target = Target(urlunparse(parsed._replace(path="/" + database)), target_bucket, "us-east-1", os.environ["S3_ENDPOINT_URL"],
                        access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"])

        dry = migrate([world.run_id], target, execute=False, output=tmp_path)
        assert dry["ok"], dry["problems"]
        planned_rows = {t: v["source_rows"] for t, v in dry["tables"].items()}
        for table in ("fulltext_file", "paper_embedding", "document_chunk", "chunk_embedding", "construct_mention", "nomological_claim", "candidate_pair", "pair_dossier",
                      "pair_decision", "decision_event", "audit_rating", "community_preassignment", "cluster_assignment", "gold_set", "release", "embedding_model"):
            assert planned_rows[table] > 0, table
        pdfs = [f for f in dry["files"] if f["kind"] == "pdf"]
        assert len(pdfs) == planned_rows["fulltext_file"] == sum(1 for p in world.papers if p.pdf)

        monkeypatch.setattr("research0.migrate.BATCH_ROWS", 7)                 # many small batches, so the keyset paging (composite keys too) is exercised
        first = migrate([world.run_id], target, execute=True, output=tmp_path)
        assert first["ok"], first["problems"]
        assert all(t["verified"] and t["target_rows"] == t["source_rows"] for t in first["tables"].values())
        assert all(f["verified"] and f["action"] == "copied" for f in first["files"]) and len(first["files"]) == 1 + len(pdfs)
        second = migrate([world.run_id], target, execute=True, output=tmp_path)
        assert second["ok"] and all(t.get("inserted", 0) == 0 for t in second["tables"].values()) and all(f["action"] == "already present" for f in second["files"])
        assert all(t.get("skipped") == "already complete" for t in second["tables"].values())   # nothing is sent again

        with psycopg.connect(target.database_url) as con:                      # a copy interrupted halfway: the next run sends only that table, and the missing rows
            lost = con.execute("DELETE FROM chunk_embedding WHERE chunk_id IN (SELECT chunk_id FROM chunk_embedding ORDER BY chunk_id LIMIT (SELECT count(*) / 2 FROM chunk_embedding))").rowcount
            con.commit()
        assert lost > 0
        resumed = migrate([world.run_id], target, execute=True, output=tmp_path)
        assert resumed["ok"], resumed["problems"]
        assert resumed["tables"]["chunk_embedding"]["inserted"] == lost and "skipped" not in resumed["tables"]["chunk_embedding"]
        assert all(t.get("skipped") for name, t in resumed["tables"].items() if name != "chunk_embedding")

        with psycopg.connect(target.database_url) as con:
            assert con.execute("SELECT count(*) FROM fulltext_file WHERE s3_uri LIKE %s", (f"s3://{target_bucket}/fulltext/%",)).fetchone()[0] == len(pdfs)
            assert con.execute("SELECT count(*) FROM document_chunk WHERE tsv @@ plainto_tsquery('english','predicted')").fetchone()[0] > 0   # generated column rebuilt
            assert con.execute("SELECT count(*) FROM paper_embedding").fetchone()[0] == planned_rows["paper_embedding"]
            assert con.execute("SELECT count(*) FROM release").fetchone()[0] == 1
        s3_target = client()
        key = pdfs[0]["source_uri"].split(f"{source_bucket}/", 1)[1]
        body = s3_target.get_object(Bucket=target_bucket, Key=key)["Body"].read()
        assert hashlib.sha256(body).hexdigest() == pdfs[0]["sha256"]

        with psycopg.connect(target.database_url) as con:                      # a tampered target row is detected
            con.execute("UPDATE construct_mention SET quote = quote || ' x' WHERE mention_id=%s", (mention,))
            con.commit()
        assert not migrate([world.run_id], target, execute=True, output=tmp_path)["ok"]
    finally:
        monkeypatch.undo()
        with psycopg.connect(os.environ["DATABASE_URL"].split("?")[0], autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        for bucket in (source_bucket, target_bucket):
            try:
                empty_and_delete(bucket)
            except Exception:
                pass
        world.drop()
