import uuid

import psycopg
import pytest

from research0 import quality, rag, release, signals
from research0.embeddings import HashEmbedder
from research0.mock_world import url_for


def test_fusion_orders_by_reciprocal_rank_and_is_deterministic():
    a, b, c = (uuid.UUID(int=i) for i in (1, 2, 3))
    fused = rag.fuse([(a, 0.9), (b, 0.5)], [(b, 0.1), (c, 0.2)], k=3)
    assert [x[0] for x in fused] == [b, a, c]                        # b is in both lists, so it wins
    assert fused[0][2] == {"lexical_rank": 2, "lexical_score": 0.5, "vector_rank": 1, "vector_distance": 0.1}
    assert rag.fuse([(a, 1)], [(a, 0)], k=1)[0][1] == pytest.approx(2 / 61)
    assert rag.fuse([], [], k=5) == []
    assert rag.fuse([(a, 1), (b, 1)], [], k=1) == rag.fuse([(a, 1), (b, 1)], [], k=1)


@pytest.fixture
def validated(piped, pcon, monkeypatch):
    """A validated release over the piped world (gates forced to pass: the point here is retrieval, gates are tested elsewhere)."""
    from research0 import documents
    documents.embed_chunks(pcon, piped.ingestion_id, HashEmbedder())
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    real = quality.evaluate
    monkeypatch.setattr(quality, "evaluate", lambda *a, **k: [quality.verdict(g["id"], quality.passing_value(g)) for g in quality.GATES])
    release_id, *_ = release.build(pcon, piped.world.run_id, "rag", ingestion_run_id=piped.ingestion_id, extraction_run_id=piped.extraction_run_id, signal_run_id=run_id)
    release.sign_off(pcon, release_id, "Ana")
    pcon.commit()
    yield release_id
    pcon.rollback()
    pcon.execute("DELETE FROM retrieval_hit"); pcon.execute("DELETE FROM retrieval_run"); pcon.execute("DELETE FROM pair_decision"); pcon.execute("DELETE FROM release_signoff"); pcon.execute("DELETE FROM release")
    pcon.execute("DELETE FROM pair_dossier"); pcon.execute("DELETE FROM candidate_pair"); pcon.execute("DELETE FROM construct_node"); pcon.execute("DELETE FROM signal_run")
    pcon.execute("DELETE FROM chunk_embedding"); pcon.commit()
    monkeypatch.setattr(quality, "evaluate", real)


def test_search_finds_the_passage_that_defines_a_planted_construct(piped, pcon, validated):
    hits = rag.search(pcon, validated, "organizational capacity to adopt and use digital technologies skills infrastructure leadership", HashEmbedder(), k=5, agent="test-agent")
    assert hits and ("Technology preparedness" in hits[0]["text"] or "Digital readiness" in hits[0]["text"])
    top = hits[0]
    paper = piped.world.paper(uuid.UUID(top["canonical_id"]))
    full = pcon.execute("SELECT full_text FROM paper_version WHERE version_id=%s", (top["version_id"],)).fetchone()[0]
    assert full[top["char_span"][0]:top["char_span"][1]] == top["text"]          # a hit can be checked against the stored document
    assert paper.title == top["title"] and 1 <= top["pages"][0] <= top["pages"][1]
    assert {"lexical_rank", "vector_rank"} & set(top["components"])
    runs = pcon.execute("SELECT agent,method,params->>'k' FROM retrieval_run").fetchall()
    assert runs == [("test-agent", "hybrid-rrf", "5")]
    assert pcon.execute("SELECT count(*) FROM retrieval_hit").fetchone()[0] == len(hits)
    assert "[1]" in rag.grounded_context(hits) and "cite them by number" in rag.grounded_context(hits)


def test_section_filter_and_input_checks(piped, pcon, validated):
    hits = rag.search(pcon, validated, "measured scale items", HashEmbedder(), k=8, sections=["method"], log=False)
    assert hits and {h["section"] for h in hits} == {"method"}
    for bad in ("", "x" * 1001):
        with pytest.raises(ValueError):
            rag.search(pcon, validated, bad, HashEmbedder())
    with pytest.raises(ValueError):
        rag.search(pcon, validated, "q", HashEmbedder(), k=0)


def test_a_release_that_is_not_validated_cannot_be_read(piped, pcon):
    blocked, *_ = release.build(pcon, piped.world.run_id, "blocked-for-rag", ingestion_run_id=piped.ingestion_id)
    with pytest.raises(ValueError, match="validated release"):
        rag.search(pcon, blocked, "anything", HashEmbedder())
    with pytest.raises(ValueError, match="validated release"):
        rag.release_scope(pcon, blocked)


def test_the_agent_role_can_read_and_log_but_not_write(piped, pcon, validated):
    """The real guarantee: a login in the research_agent role. Created for the test and removed after."""
    admin = psycopg.connect(__import__("os").environ["DATABASE_URL"], autocommit=True)
    name = "mock_agent_" + uuid.uuid4().hex[:8]
    admin.execute(f"CREATE ROLE {name} LOGIN PASSWORD 'pw' IN ROLE research_agent")
    try:
        base = __import__("os").environ["DATABASE_URL"].replace("research:local_only", f"{name}:pw")
        agent_url = url_for(piped.world.schema, base)
        with rag.reader_connection(agent_url) as agent:
            hits = rag.search(agent, validated, "algorithmic control predicted", HashEmbedder(), k=3, agent="role-test")
            assert hits
            assert agent.execute("SELECT count(*) FROM retrieval_run WHERE agent='role-test'").fetchone()[0] == 1
            for sql in ("INSERT INTO preassignment_run(preassignment_run_id,run_id,source,model,prompt_version,scope_version,codebook_sha256) VALUES (gen_random_uuid(),gen_random_uuid(),'batch_v2','m','p','s','0')",
                        "UPDATE canonical_paper SET title='hacked'", "DELETE FROM document_chunk", "DELETE FROM retrieval_hit", "DROP TABLE release"):
                with pytest.raises(psycopg.Error):
                    agent.execute(sql)
                agent.rollback()
            agent.execute("SELECT 1")
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                agent.execute("CREATE TEMP TABLE x(a int)")
            agent.rollback()
    finally:
        admin.execute(f"DROP OWNED BY {name}")
        admin.execute(f"DROP ROLE {name}")
        admin.close()


def test_settled_pairs_are_visible_to_agents(piped, pcon, validated):
    from research0 import pairs
    run_id = pcon.execute("SELECT signal_run_id FROM signal_run").fetchone()[0]
    signals.build_dossiers(pcon, run_id, limit=2)
    first = pcon.execute("SELECT pair_id FROM pair_dossier LIMIT 1").fetchone()[0]
    pairs.decide(pcon, first, "Ana", "distinct", "first reason", {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"})
    pairs.decide(pcon, first, "Ben", "distinct", "second reason", {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"})
    got = rag.settled_pairs(pcon, validated)
    assert [(g["pair_id"], g["class"]) for g in got] == [(str(first), "distinct")] and sorted(got[0]["rationale"]) == ["first reason", "second reason"]
    assert got[0]["checklist"]["same_referent"] == "unclear"
    pcon.rollback()
