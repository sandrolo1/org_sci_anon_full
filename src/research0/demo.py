"""`research0 demo`: every phase from 2 to 8 end to end on the MOCK world, then the mock world is deleted.

Everything is synthetic (see mock_world.py): the PDFs are generated, the "model" is a script that answers from planted facts, and the
"people" are scripts that answer from the planted truth. It proves that the plumbing works and shows what each output looks like; it
says nothing about real papers. Nothing is written to the real schema. With --keep the mock schema stays for inspection
(`research0 mock-clean` removes it).
"""
import json
import tempfile
import uuid
from pathlib import Path

from . import audit, community_v2, confirmation, documents, explore, extraction, mock_world, pairs, pairsample, proposals, quality, rag, release, signals
from .embeddings import HashEmbedder


def _step(log, name, **values):
    log.append({"step": name, **values})


def run_demo(papers=240, keep=False):
    world = mock_world.create_world(papers)
    log = [{"step": "mock_world", "schema": world.schema, "papers": len(world.papers), "with_pdf": sum(1 for p in world.papers if p.pdf), "note": "SYNTHETIC DATA"}]
    scratch = tempfile.TemporaryDirectory(prefix="research0-demo-")
    work = Path(scratch.name)
    try:
        with world.connect() as con:
            embedder = HashEmbedder()
            # Phase 2
            explore.embed_papers(con, world.run_id, embedder)
            cluster_run, summary, _ = explore.cluster(con, world.run_id, embedder, k=4)
            _step(log, "phase2_clusters", cluster_run_id=str(cluster_run), sizes=sorted((c["size"] for c in summary["clusters"]), reverse=True))
            # Phase 3
            manifest = community_v2.prepare(con, world.run_id, work / "preassign", "config/community_scope_v2.json")
            lines = (work / "preassign" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (work / "preassign" / "batch_output.jsonl").write_text("\n".join(mock_world.scripted_preassignment_output(world, lines)) + "\n", encoding="utf-8")
            collected = community_v2.collect_file(con, work / "preassign", "config/community_scope_v2.json")
            pre = uuid.UUID(collected["preassignment_run_id"])
            _step(log, "phase3_preassignment", counts=community_v2.summary(con, pre), not_sent_because_no_usable_abstract=len(manifest["insufficient"]))
            # Phase 4
            pre_for_audit, audit_id = mock_world.mock_audit(con, world)
            agreement, estimate = audit.agreement(con, audit_id), audit.estimates(con, audit_id)
            _step(log, "phase4_audit", kappa_cto=agreement["CTO"]["three_class"]["kappa"], disagreements=agreement["disagreements"],
                  macro_f1_cto=estimate["CTO"]["strict"]["macro_f1"], warnings=estimate["CTO"]["design_warnings"])
            # Phase 5
            ingestion_id, counts = documents.ingest(con, world.run_id, world.store)
            documents.embed_chunks(con, ingestion_id, embedder)
            confirm_dir = work / "confirm"
            confirmation.prepare(con, world.run_id, ingestion_id, confirm_dir, "config/community_scope_v2.json")
            confirm_lines = (confirm_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (confirm_dir / "batch_output.jsonl").write_text(chr(10).join(mock_world.scripted_confirmation_output(world, confirm_lines)) + chr(10), encoding="utf-8")
            confirmed = confirmation.collect_file(con, confirm_dir, "config/community_scope_v2.json")
            statuses = {}
            for m in confirmation.memberships(con, world.run_id).values():
                statuses[m["status"]] = statuses.get(m["status"], 0) + 1
            _step(log, "phase5_fulltext_confirmation", papers_answered=confirmed["papers_answered"], membership_status=statuses)
            directory = work / "extraction"
            extraction.prepare(con, ingestion_id, directory)
            owner = {str(a): b for a, b in con.execute("SELECT version_id,canonical_id FROM paper_version")}
            request_lines = (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (directory / "batch_output.jsonl").write_text("\n".join(mock_world.scripted_extraction_output(world, request_lines, owner)) + "\n", encoding="utf-8")
            extracted = extraction.collect_file(con, directory)
            _step(log, "phase5_documents_and_extraction", parsed=counts, **{k: v for k, v in extracted.items() if k != "extraction_run_id"})
            # Phase 6
            signal_run, sig, _ = signals.build(con, uuid.UUID(extracted["extraction_run_id"]), embedder)
            created = signals.build_dossiers(con, signal_run)
            by_hint = dict(con.execute("SELECT hint,count(*) FROM candidate_pair WHERE signal_run_id=%s GROUP BY 1", (signal_run,)).fetchall())
            _step(log, "phase6_signals", nodes=sig["nodes"], candidate_pairs=sig["pairs"], dossiers=created, by_triage_hint=by_hint)
            prop_dir = work / "proposals"
            proposals.prepare(con, signal_run, prop_dir, hints=["same_label_divergent", "different_label_close"], limit=10)
            prop_lines = (prop_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (prop_dir / "batch_output.jsonl").write_text(chr(10).join(mock_world.scripted_proposal_output(prop_lines)) + chr(10), encoding="utf-8")
            _step(log, "phase7_llm_proposals", **proposals.collect_file(con, prop_dir))
            sample_id, _, _ = pairsample.create(con, signal_run, n_probabilistic=60, n_challenge=20, min_per_stratum=3)
            # Phase 7
            planted = _planted_gold(con, signal_run)
            for pair_id, kind in planted["decisions"].items():
                for who in ("Ana", "Ben"):
                    mock_world.decide_truth(con, pair_id, who, kind)
            gold = quality.store_gold(con, "pairs", "planted truth of the mock world", {"items": planted["gold"]}, "mock")
            built = release.build(con, world.run_id, "demo", pre, audit_id, ingestion_id, uuid.UUID(extracted["extraction_run_id"]), signal_run, gold_pairs=gold)
            release_id, _, passed, gates, _ = built
            _step(log, "phase7_release", status=release.status(con, release_id), gates=[{k: g[k] for k in ("id", "value", "status")} for g in gates],
                  note="Blocked is the honest outcome here: no gold extraction, no gold ratings, and agreement is below its gate on purpose.")
            # Phase 8 (gates forced to pass ONLY to exercise retrieval on mock data)
            real = quality.evaluate
            quality.evaluate = lambda *a, **k: [quality.verdict(g["id"], quality.passing_value(g), "DEMO ONLY: forced") for g in quality.GATES]
            try:
                forced, *_ = release.build(con, world.run_id, "demo-forced", pre, audit_id, ingestion_id, uuid.UUID(extracted["extraction_run_id"]), signal_run)
            finally:
                quality.evaluate = real
            release.sign_off(con, forced, "demo")
            hits = rag.search(con, forced, "organizational capacity to adopt digital tools skills infrastructure", embedder, k=3, agent="demo")
            _step(log, "phase8_retrieval", note="DEMO ONLY: a release whose gates were forced to pass, to exercise read-only retrieval",
                  top_hit={"title": hits[0]["title"], "section": hits[0]["section"], "pages": hits[0]["pages"]} if hits else None)
            con.rollback()
    finally:
        scratch.cleanup()
        if not keep:
            world.drop()
    log.append({"step": "cleanup", "mock_schema_dropped": not keep, "kept": world.schema if keep else None,
                "remove_with": "research0 mock-clean" if keep else None, "temporary_files_deleted": True})
    return {"event": "demo_finished", "steps": log}


def _planted_gold(con, signal_run_id):
    decisions, gold = {}, []
    for pair_id, ca, la, cb, lb in con.execute(
            "SELECT p.pair_id,a.canonical_id,a.label_normalized,b.canonical_id,b.label_normalized FROM candidate_pair p "
            "JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b "
            "WHERE p.signal_run_id=%s", (signal_run_id,)):
        for pair, relation in mock_world.TRUTH_PAIRS.items():
            if {la, lb} == {key[0] for key in pair}:
                decisions[pair_id] = relation          # the kind of the planted truth (mock_world.TRUTH_DECISIONS)
                gold.append({"canonical_id_a": str(ca), "label_a": la, "canonical_id_b": str(cb), "label_b": lb, "class": mock_world.TRUTH_DECISIONS[relation][0]})
    return {"decisions": decisions, "gold": gold}
