import numpy as np
import pytest

from research0 import explore
from research0.embeddings import HashEmbedder, cosine, get_embedder


def test_hash_embedder_is_deterministic_normalised_and_lexical():
    e = HashEmbedder()
    a, b = e.embed(["algorithmic management of gig workers", "algorithmic management of gig workers"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5 and len(a) == 256
    same_topic = cosine(e.embed(["gig workers and algorithmic management"])[0], a)
    other = cosine(e.embed(["nursing staff and patient safety"])[0], a)
    assert same_topic > other + 0.2 and e.kind == "lexical"
    assert np.allclose(HashEmbedder().embed(["x y z open innovation"]), e.embed(["x y z open innovation"]))


def test_unknown_embedding_model_is_refused():
    with pytest.raises(ValueError):
        get_embedder("whatever")


def test_missing_abstract_placeholder_is_not_used_as_text():
    assert explore.paper_text("Title", "[No abstract available]", "kw", "") == "Title\nkw"


def test_clustering_recovers_the_planted_topics_and_is_idempotent(world, con):
    embedder = HashEmbedder()
    new, existing = explore.embed_papers(con, world.run_id, embedder)
    assert new == len(world.eligible()) and existing == 0
    assert explore.embed_papers(con, world.run_id, embedder) == (0, new)
    run_id, summary, already = explore.cluster(con, world.run_id, embedder, k=4, seed=3)
    assert not already and summary["papers"] == new and "does not decide" in summary["warning"]
    again, _, already = explore.cluster(con, world.run_id, embedder, k=4, seed=3)
    assert already and again == run_id
    labels = dict(con.execute("SELECT canonical_id,cluster_label FROM cluster_assignment WHERE cluster_run_id=%s", (run_id,)).fetchall())
    # Every planted topic should end up mostly in one cluster (lexical vectors, four clearly different vocabularies).
    for topic in ("platform_work", "open_innovation", "nursing_burnout"):
        counts = np.bincount([labels[p.canonical_id] for p in world.eligible() if p.topic == topic and p.abstract != "[No abstract available]"], minlength=4)
        assert counts.max() / counts.sum() > 0.8, topic
    assert sum(c["cto_seeds"] for c in summary["clusters"]) == 6 and sum(c["tim_seeds"] for c in summary["clusters"]) == 6
    assert all(c["top_terms"] for c in summary["clusters"] if c["size"] > 10)


def test_ineligible_papers_are_never_clustered(world, con):
    explore.embed_papers(con, world.run_id, HashEmbedder())
    ids = {r[0] for r in con.execute("SELECT canonical_id FROM paper_embedding")}
    assert world.papers[3].canonical_id not in ids and world.papers[7].canonical_id not in ids


def test_k_out_of_range_is_refused(world, con):
    e = HashEmbedder()
    explore.embed_papers(con, world.run_id, e)
    with pytest.raises(ValueError):
        explore.cluster(con, world.run_id, e, k=1)


def test_report_files(world, con, tmp_path):
    e = HashEmbedder()
    explore.embed_papers(con, world.run_id, e)
    run_id, _, _ = explore.cluster(con, world.run_id, e, k=4)
    out = explore.write_report(con, run_id, tmp_path / "r")
    text = (out / "cluster_report.md").read_text(encoding="utf-8")
    assert "Exploratory" in text and "lexical" in text
    assert len((out / "clusters.csv").read_text().splitlines()) == len(world.eligible()) + 1
