"""Shared fixtures. The mock world is a throw-away PostgreSQL schema (mock_<hex>) that is dropped when the session ends;
the real schema is never written by the tests that use it."""
import os

import pytest

import forum_fake
from research0.local_env import load_local_env

load_local_env()      # the integration tests need DATABASE_URL and S3_*: read .env as the command line does (variables already set win)
forum_fake.register()      # the forum's offline stand-in for a model, for this test session only (the application has no simulation since 13 Oct 2026)


@pytest.fixture(autouse=True)
def no_semantic_scholar_key(monkeypatch):
    """Tests never call Semantic Scholar with the real key: a CLI test loads .env into the process, and the key would then reach every later download test."""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")


@pytest.fixture(scope="session")
def world():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set local PostgreSQL environment to run integration tests")
    from research0 import mock_world
    w = mock_world.create_world()
    yield w
    w.drop()


@pytest.fixture
def con(world):
    """A connection on the mock schema whose work is rolled back at the end of each test."""
    connection = world.connect()
    connection.execute("SELECT 1")   # open the transaction: a later `with connection.transaction()` is then a savepoint, not a commit
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture(scope="session")
def piped(tmp_path_factory):
    """A second mock world where parsing, chunking and (scripted) extraction have already run and are committed. Read-only for tests."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set local PostgreSQL environment to run integration tests")
    import json
    from types import SimpleNamespace

    from research0 import documents, extraction, mock_world
    w = mock_world.create_world(n=200, seed=23)
    try:
        directory = tmp_path_factory.mktemp("piped") / "extraction"
        with w.connect() as c:
            ingestion_id, _ = documents.ingest(c, w.run_id, w.store)
            extraction.prepare(c, ingestion_id, directory)
            owner = {str(a): b for a, b in c.execute("SELECT version_id,canonical_id FROM paper_version")}
            lines = (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (directory / "batch_output.jsonl").write_text("\n".join(mock_world.scripted_extraction_output(w, lines, owner)) + "\n", encoding="utf-8")
            result = extraction.collect_file(c, directory)
            c.commit()
        yield SimpleNamespace(world=w, ingestion_id=ingestion_id, extraction_run_id=result["extraction_run_id"], directory=directory)
    finally:
        w.drop()


@pytest.fixture
def pcon(piped):
    connection = piped.world.connect()
    connection.execute("SELECT 1")
    yield connection
    connection.rollback()
    connection.close()
