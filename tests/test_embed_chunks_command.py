"""`research0 embed-chunks` must survive a chunk longer than the embedding model accepts (the PDFs of 18 Oct 2026 had some, and the old function failed with "maximum input length is
8192 tokens" after hours of work, losing everything because it saved at the end). Needs the local PostgreSQL (a throw-away mock schema, dropped at the end)."""
import argparse
import os
from types import SimpleNamespace

import pytest

from research0 import documents, pipeline_cli
from research0.embeddings import HashEmbedder

OVERSIZED = "字" * 40000                       # about 3 tokens per character: far above 8,191 tokens even after the embedder's own 30,000-character cut


class StrictApiEmbedder(HashEmbedder):
    """Embeds like the lexical stand-in, but refuses any text that the real API would refuse: more than 8,191 tokens (after the same cut at 30,000 characters)."""
    calls = 0

    def embed(self, texts):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        for text in texts:
            if len(enc.encode(text[:30000] or " ", disallowed_special=())) > 8191:
                raise ValueError("Invalid 'input': maximum input length is 8192 tokens.")
        type(self).calls += 1
        return super().embed(texts)


@pytest.fixture(scope="module")
def eworld():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set local PostgreSQL environment to run integration tests")
    from research0 import mock_world
    w = mock_world.create_world(n=40, seed=3)
    try:
        with w.connect() as c:
            ingestion_id, _ = documents.ingest(c, w.run_id, w.store)
            first = c.execute("SELECT chunk_id FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s ORDER BY c.chunk_id LIMIT 1", (ingestion_id,)).fetchone()[0]
            c.execute("UPDATE document_chunk SET text=%s WHERE chunk_id=%s", (OVERSIZED, first))     # the oversized chunk is the FIRST one: the old function died on its first batch
            c.commit()
        yield SimpleNamespace(world=w, ingestion_id=ingestion_id, oversized=first)
    finally:
        w.drop()


def test_the_command_embeds_every_chunk_including_the_oversized_one_and_a_second_run_has_nothing_to_do(eworld, monkeypatch, capsys):
    monkeypatch.setattr("research0.embeddings.get_embedder", lambda name=None, offline_default=False: StrictApiEmbedder())
    args = argparse.Namespace(command="embed-chunks", ingestion_run_id=str(eworld.ingestion_id), model=None)
    with eworld.world.connect() as con:
        total = con.execute("SELECT count(*) FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s", (eworld.ingestion_id,)).fetchone()[0]
        out = pipeline_cli._with_database(args, con)
        assert out["event"] == "chunks_embedded" and out["new"] == total > 10 and out["tokens"] > 0
        assert con.execute("SELECT count(*) FROM chunk_embedding").fetchone()[0] == total
        assert con.execute("SELECT count(*) FROM chunk_embedding WHERE chunk_id=%s", (eworld.oversized,)).fetchone()[0] == 1
        again = pipeline_cli._with_database(args, con)
        assert again["new"] == 0
    assert '"event": "embedding_progress"' in capsys.readouterr().out                # progress is printed as the other long commands do


def test_the_stand_in_really_refuses_what_the_api_refuses():
    with pytest.raises(ValueError, match="maximum input length"):
        StrictApiEmbedder().embed([OVERSIZED])
    assert StrictApiEmbedder().embed(["a short chunk"]).shape[0] == 1
