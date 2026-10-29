import csv
import os
import uuid
from urllib.parse import urlparse, urlunparse

import boto3
import psycopg
import pytest

from research0.migrate import Target, migrate

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL") or not os.getenv("S3_BUCKET"), reason="Set local PostgreSQL and S3 environment to run integration test")


def test_migration_is_scoped_idempotent_and_verified(tmp_path):
    from research0.catalog import build_catalog
    from research0.cli import database, import_scopus, register_query

    tag = uuid.uuid4().hex[:10]
    # Source data for one run, plus a second run that must NOT be copied.
    with database() as con:
        from research0.migrate import apply_migrations
        apply_migrations(con)
        runs = []
        for n in (1, 2):
            path = tmp_path / f"export{n}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Authors", "Title", "Year", "Source title", "Document Type", "Abstract", "DOI", "EID"])
                writer.writerow(["Smith J.", f"Paper {tag} {n}", 2020, "Journal A", "Article", "abstract", f"10.9/{tag}{n}", f"2-s2.0-{tag}{n}"])
            qid, _ = register_query(con, f"test-migrate-{tag}-{n}", None)
            import_scopus(con, path, qid, "2026-01-01", "synthetic fixture")
            run_id, _, _ = build_catalog(con, qid)
            runs.append(run_id)
        con.execute("INSERT INTO validation_decision(run_id,item_id,item_type,decision,note,validator) VALUES (%s,'x','t','needs_discussion','n','pytest')", (runs[0],))
    # Target: a fresh database and bucket on the same local servers.
    parsed = urlparse(os.environ["DATABASE_URL"])
    name = f"migtest_{tag}"
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    target = Target(urlunparse(parsed._replace(path="/" + name)), f"migtest-{tag}", "us-east-1", os.environ["S3_ENDPOINT_URL"],
                    access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    try:
        dry = migrate([runs[0]], target, execute=False, output=tmp_path)
        assert dry["ok"] and dry["mode"] == "dry-run" and dry["tables"]["export_record"]["source_rows"] == 1
        no_forum = migrate([runs[0]], target, execute=False, output=tmp_path, skip_forum=True)   # forum generations stay behind; everything else is still planned
        assert no_forum["ok"] and not {"forum_episode", "forum_message", "question", "forum_proposal"} & set(no_forum["tables"]) and "export_record" in no_forum["tables"]
        assert {"forum_episode", "question"} <= set(dry["tables"])
        with psycopg.connect(target.database_url) as con:  # nothing was written
            assert con.execute("SELECT to_regclass('export_record')").fetchone()[0] is None
        first = migrate([runs[0]], target, execute=True, output=tmp_path)
        assert first["ok"], first["problems"]
        assert first["tables"]["export_record"]["inserted"] == 1 and first["tables"]["validation_decision"]["inserted"] == 1
        assert first["files"][0]["verified"] and first["files"][0]["action"] == "copied"
        second = migrate([runs[0]], target, execute=True, output=tmp_path)
        assert second["ok"] and all(t.get("inserted", 0) == 0 for t in second["tables"].values())
        assert second["files"][0]["action"] == "already present"
        with psycopg.connect(target.database_url) as con:
            assert con.execute("SELECT count(*) FROM export_batch").fetchone()[0] == 1  # the second run stayed behind
            assert con.execute("SELECT original_uri FROM export_batch").fetchone()[0].startswith(f"s3://{target.bucket}/")
            assert con.execute("SELECT count(*) FROM catalog_run").fetchone()[0] == 1
        # A tampered target row is detected by verification.
        with psycopg.connect(target.database_url) as con:
            con.execute("UPDATE export_record SET title = title || ' edited'")
            con.commit()
        assert not migrate([runs[0]], target, execute=True, output=tmp_path)["ok"]
    finally:
        client = boto3.client("s3", endpoint_url=target.endpoint_url, region_name="us-east-1", aws_access_key_id=target.access_key, aws_secret_access_key=target.secret_key)
        for obj in client.list_objects_v2(Bucket=target.bucket).get("Contents", []):
            client.delete_object(Bucket=target.bucket, Key=obj["Key"])
        client.delete_bucket(Bucket=target.bucket)
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {name} WITH (FORCE)")
