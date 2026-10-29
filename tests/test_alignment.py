"""Behaviour added to match the research proposal (Proposta_ORG_SI_Extended_CTO_TIM.docx), on the MOCK world only."""
import csv
import json
import uuid
from pathlib import Path

import pytest

from research0 import audit, confirmation, fulltext, pairs, pairsample, proposals, quality, rag, release, signals
from research0.embeddings import HashEmbedder
from research0.mock_world import _batch_line, make_preassignment, mock_audit, planted_gold_items, scripted_confirmation_output, scripted_proposal_output

SCOPE = "config/community_scope_v2.json"
UNCLEAR = {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}


# --- Full-text confirmation of the membership (step 05) --------------------------------------------------------------------------------------

def test_the_full_text_confirms_modifies_or_rejects_the_preliminary_membership(piped, pcon, tmp_path):
    world = piped.world
    make_preassignment(pcon, world, miss_rate=0.5, seed=3)                       # the metadata stage misses half of the true members
    manifest = confirmation.prepare(pcon, world.run_id, piped.ingestion_id, tmp_path / "c", SCOPE)
    assert manifest["request_count"] == sum(1 for p in world.papers if p.pdf)
    lines = (tmp_path / "c" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(json.loads(lines[0])["body"]["input"])[0]
    assert {e["section"] for e in first["excerpts"]} <= set(confirmation.SECTIONS) and "uncertain, never no" in json.loads(lines[0])["body"]["instructions"]
    victim = json.loads(lines[1])["custom_id"]
    (tmp_path / "c" / "batch_output.jsonl").write_text("\n".join(scripted_confirmation_output(world, lines, fabricate_for=victim)) + "\n", encoding="utf-8")
    result = confirmation.collect_file(pcon, tmp_path / "c", SCOPE)
    assert result["rejected"] == 1 and result["papers_answered"] == manifest["request_count"] - 1      # the invented quote is refused, the paper stays preliminary
    rows = pcon.execute("SELECT canonical_id,community,label,stage,version_id,evidence_page,evidence_char_start,evidence_char_end,evidence_quote FROM community_preassignment "
                        "WHERE stage='full_text' AND label='yes'").fetchall()
    assert rows
    for cid, community, label, stage, version, page, start, end, quote in rows:
        full = pcon.execute("SELECT full_text FROM paper_version WHERE version_id=%s", (version,)).fetchone()[0]
        assert full[start:end] == quote and page >= 1                                  # the span is checkable against the stored document
    effective = confirmation.memberships(pcon, world.run_id)
    statuses = {m["status"] for k, m in effective.items() if m["stage"] == "full_text"}
    assert "confirmed" in statuses and "modified" in statuses
    assert all(m["status"] == "preliminary" for k, m in effective.items() if k[0] == uuid.UUID(victim))
    truth = {p.canonical_id: p.truth for p in world.papers}
    assert all((m["label"] == "yes") == truth[k[0]][k[1]] for k, m in effective.items() if m["stage"] == "full_text")


def test_a_changed_codebook_blocks_the_confirmation(piped, pcon, tmp_path):
    confirmation.prepare(pcon, piped.world.run_id, piped.ingestion_id, tmp_path / "c", SCOPE)
    other = tmp_path / "scope.json"
    other.write_text(open(SCOPE, encoding="utf-8").read().replace("aom-scope-v2", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="codebook changed"):
        confirmation.collect_file(pcon, tmp_path / "c", other)


# --- The model's proposal on each pair (step 08a) -----------------------------------------------------------------------------------------

def test_proposals_are_stored_apart_and_shown_only_after_the_reviewers_own_decision(piped, pcon, tmp_path):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id)
    manifest = proposals.prepare(pcon, run_id, tmp_path / "p", hints=["same_label_divergent", "different_label_close"], limit=6)
    lines = (tmp_path / "p" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    request = json.loads(lines[0])
    assert "Never invent" in request["body"]["instructions"] and request["body"]["text"]["format"]["strict"] and manifest["prompt_version"] == "pair-proposal-v2"
    assert "substitution test" in request["body"]["instructions"] and request["body"]["text"]["format"]["schema"]["properties"]["decision_class"]["enum"] == list(pairs.CLASSES)
    assert not any(k in json.dumps(proposals.schema()) for k in ("c_cit", "n_nom", "number", "merge"))      # nothing numeric to invent
    broken = json.loads(lines[1])["custom_id"]
    (tmp_path / "p" / "batch_output.jsonl").write_text("\n".join(scripted_proposal_output(lines, break_for=broken)) + "\n", encoding="utf-8")
    result = proposals.collect_file(pcon, tmp_path / "p")
    assert result["stored"] == manifest["request_count"] - 1 and result["rejected"] == 1
    pair = pcon.execute("SELECT pair_id FROM pair_proposal LIMIT 1").fetchone()[0]
    assert proposals.visible_to(pcon, pair, "Ana") is None                                        # no decision yet: the proposal stays hidden
    pcon.execute("INSERT INTO pair_decision(pair_id,relation,flag,rationale,validator) VALUES (%s,'distinct','jingle','old scale','Ana')", (pair,))
    assert proposals.visible_to(pcon, pair, "Ana") is None                                        # a decision of the earlier codebook does not unlock it
    pairs.decide(pcon, pair, "Ana", "unresolved", "my own reading", UNCLEAR)
    seen = proposals.visible_to(pcon, pair, "Ana")
    assert seen["decision_class"] in pairs.CLASSES and seen["substitution_test"] and seen["flag"] is None and proposals.visible_to(pcon, pair, "Ben") is None


def _positive(con, pair_id, validator):
    """A conceptual jingle for the same label, a conceptual jangle otherwise, with the checklist the codebook asks for."""
    if pairs.labels_equal(con, pair_id):
        pairs.decide(con, pair_id, validator, "conceptual_jingle", "r", {"same_referent": "no", "compatible_properties": "no", "substitution_changes": "yes"}, consequence="c")
    else:
        pairs.decide(con, pair_id, validator, "conceptual_jangle", "r", {"same_referent": "yes", "compatible_properties": "yes", "substitution_changes": "no"}, consequence="c")


def test_the_cascade_hands_people_what_the_model_promotes_and_audits_what_it_rejects(piped, pcon, tmp_path):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    hints = {str(p): h for p, h in pcon.execute("SELECT pair_id,hint FROM candidate_pair WHERE signal_run_id=%s", (run_id,))}
    targeted = {p for p, h in hints.items() if h in proposals.TARGETED_HINTS}
    assert len(targeted) >= 4 and len(hints) - len(targeted) >= 4
    signals.build_dossiers(pcon, run_id, hints=proposals.TARGETED_HINTS)
    manifest = proposals.prepare(pcon, run_id, tmp_path / "p", hints=proposals.TARGETED_HINTS)
    assert manifest["request_count"] == len(targeted) and manifest["estimated_tokens"]["input"] > 0                # only the targeted pairs reach the model
    with pytest.raises(ValueError, match="no model proposal"):
        pairsample.create_cascade(pcon, run_id, n_audit=4)                                                     # nothing collected: the model has filtered nothing yet
    ids = sorted(manifest["requests"])
    said = {pid: ("distinct" if i % 2 else "measurement_divergence") for i, pid in enumerate(ids)}
    body = lambda cls: dict(decision_class=cls, substitution_test="s", why_not_mere_variation="", rationale="r", contrary_evidence="", evidence_needed="")
    (tmp_path / "p" / "batch_output.jsonl").write_text("\n".join(_batch_line(pid, body(cls)) for pid, cls in said.items()) + "\n", encoding="utf-8")
    assert proposals.collect_file(pcon, tmp_path / "p")["stored"] == len(ids)
    with pytest.raises(ValueError, match="at least one pair"):
        pairsample.create_cascade(pcon, run_id, n_audit=0)                                                     # no audit = the model would exclude pairs alone
    sample, design, existed = pairsample.create_cascade(pcon, run_id, n_audit=4, seed=3)
    assert not existed and pairsample.create_cascade(pcon, run_id, n_audit=4, seed=3)[2]
    items = {str(p): s for p, s in pcon.execute("SELECT pair_id,stratum FROM pair_sample_item WHERE pair_sample_id=%s", (sample,))}
    promoted, rejected = {p for p, c in said.items() if c != "distinct"}, {p for p, c in said.items() if c == "distinct"}
    assert promoted <= set(items) and all(items[p] == "llm_promoted|measurement_divergence" for p in promoted)      # promoted pairs go to people whole
    assert design["groups"]["llm_rejected"] == {"population": len(rejected), "sample": min(4, len(rejected))}
    assert design["groups"]["not_targeted"] == {"population": len(hints) - len(targeted), "sample": 4} and design["groups"]["llm_missing"]["population"] == 0
    assert all(hints[p] not in proposals.TARGETED_HINTS for p, s in items.items() if s.startswith("not_targeted|"))
    # people find a conceptual fallacy among the pairs the model rejected: the cascade missed it, and the report says so
    signals.build_dossiers(pcon, run_id, pair_ids=list(items))
    missed = next(p for p in sorted(items) if items[p].startswith("llm_rejected|"))
    for pid in items:
        for validator in ("Ana", "Ben"):
            if pid == missed or pid in promoted:
                _positive(pcon, pid, validator)
            else:
                pairs.decide(pcon, pid, validator, "distinct", "r", UNCLEAR)
    cls = "conceptual_jingle" if pairs.labels_equal(pcon, missed) else "conceptual_jangle"
    recall = pairsample.report(pcon, sample)["cascade"]["recall_by_class"][cls]
    assert recall["estimated_in_audit_groups"] > 0 and recall["recall"] < 1 and recall["range"][0] <= recall["recall"] <= recall["range"][1]
    # the jingle/jangle gate now reads the model's class on the pairs it filtered, not the hint
    index = quality.candidate_index(pcon, run_id)
    flags, source = quality.pipeline_flags(pcon, run_id, index)
    assert source == "model" and not any(flags.values())                                                       # the model flagged no conceptual fallacy here


def test_the_submitted_proposal_batch_is_fetched_and_stored(piped, pcon, tmp_path, monkeypatch):
    import sys
    import types
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id, hints=proposals.TARGETED_HINTS)
    manifest = proposals.prepare(pcon, run_id, tmp_path / "p", hints=proposals.TARGETED_HINTS)
    lines = (tmp_path / "p" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    output = ("\n".join(scripted_proposal_output(lines)) + "\n").encode()
    (tmp_path / "p" / "batch.json").write_text(json.dumps({"input_file_id": "file-in", "batch_id": "batch-1"}), encoding="utf-8")
    state = {"status": "in_progress", "output_file_id": None}
    counts = types.SimpleNamespace(total=len(lines), completed=0, failed=0)

    class Streamed:                              # client.files.with_streaming_response.content(id) as response: response.stream_to_file(path)
        def __enter__(self):
            return types.SimpleNamespace(stream_to_file=lambda path: Path(path).write_bytes(output))

        def __exit__(self, *exc):
            return False

    class Client:
        batches = types.SimpleNamespace(retrieve=lambda batch_id: types.SimpleNamespace(error_file_id=None, request_counts=counts, **state))
        files = types.SimpleNamespace(with_streaming_response=types.SimpleNamespace(content=lambda file_id: Streamed()))

        def with_options(self, **options):
            return self
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    assert proposals.collect_remote(pcon, tmp_path / "p") == {"status": "in_progress", "request_counts": {"total": len(lines), "completed": 0, "failed": 0}, "stored": 0}
    state.update(status="completed", output_file_id="file-out")
    result = proposals.collect_remote(pcon, tmp_path / "p")
    assert result["status"] == "completed" and result["stored"] == manifest["request_count"] and result["missing_requests"] == 0
    assert json.loads((tmp_path / "p" / "batch.json").read_text(encoding="utf-8"))["output_file_id"] == "file-out"


# --- Pair sampling ------------------------------------------------------------------------------------------------------------------------

def test_probabilistic_and_challenge_samples_are_separate_and_reported_so(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id)
    sample_id, design, existed = pairsample.create(pcon, run_id, n_probabilistic=60, n_challenge=20, seed=5, min_per_stratum=3)
    assert not existed and pairsample.create(pcon, run_id, n_probabilistic=60, n_challenge=20, seed=5, min_per_stratum=3)[2]
    items = pcon.execute("SELECT pair_id,sample_kind,stratum,population_size,sample_size FROM pair_sample_item WHERE pair_sample_id=%s", (sample_id,)).fetchall()
    kinds = {k for _, k, *_ in items}
    assert kinds == {"probabilistic", "challenge"} and len({p for p, *_ in items}) == len(items)    # a pair is in one sample only
    probabilistic = [i for i in items if i[1] == "probabilistic"]
    assert len(probabilistic) <= 60 and all(1 <= size <= population for _, _, _, population, size in probabilistic)
    assert design["version"] == "pair-sample-v3" and set(design["challenge_kinds"]) == {"same_label_meaning_different", "different_label_meaning_similar",
                                                                                       "same_measure_different_definition", "context_shifted", "near_threshold", "hard_negative", "cross_community"}
    # decide every sampled pair the same way: measurement divergence (not a conceptual fallacy)
    for pair_id, kind, *_ in items:
        pairs.decide(pcon, pair_id, "Ana", "measurement_divergence", "r", UNCLEAR)
        pairs.decide(pcon, pair_id, "Ben", "measurement_divergence", "r", UNCLEAR)
    report = pairsample.report(pcon, sample_id)
    assert report["probabilistic"]["pairs"] == len(probabilistic) and report["challenge"]["pairs"] == len(items) - len(probabilistic)
    assert report["adjudicated_per_positive_class"] == {"conceptual_jingle": 0, "conceptual_jangle": 0} and report["challenge"]["by_class"]["measurement_divergence"] > 0
    assert "not prevalence" in report["warning"] and "at least 50" in report["warning"]
    low, high = report["estimated_candidate_pairs_by_class"]["conceptual_jingle"]["range"]
    assert low == 0 <= high


# --- Gates added from the proposal's tables -------------------------------------------------------------------------------------------

def test_audit_gates_report_screening_multilabel_calibration_and_the_agreement_action(world, con):
    pre, audit_id = mock_audit(con, world, n_total=120, annotator_error_every=7)
    gates = {g["id"]: g for g in quality.membership_gates(con, audit_id)}
    assert set(gates) == set(quality.AUDIT_GATES) and all(g["value"] is not None for g in gates.values())
    assert gates["screening_recall"]["status"] in ("pass", "fail") and "95% interval" in gates["screening_recall"]["detail"]
    assert gates["human_agreement"]["action"] in ("proceed", "revise the codebook and recode", "block the variable")
    assert gates["calibration_error"]["status"] in ("pass", "fail") and 0 <= gates["calibration_error"]["value"] <= 1
    matrices = audit.confusion_matrices(con, audit_id)
    assert set(matrices) == {"CTO", "TIM"} and sum(matrices["CTO"]["audited"].values()) == 120 and matrices["CTO"]["estimated_population"]
    br = audit.breakdowns(con, audit_id)
    assert sum(v["audited"] for v in br["CTO"]["by_period"].values()) == 120


def test_agreement_action_bands():
    pick = lambda v: next(t for floor, t in quality.AGREEMENT_ACTIONS if v >= floor)
    assert pick(0.80) == "proceed" and pick(0.75) == "revise the codebook and recode" and pick(0.667) == "revise the codebook and recode" and pick(0.60) == "block the variable"


def test_definition_precision_comes_from_what_people_judged(piped, pcon):
    from research0 import extraction
    ids = [r[0] for r in pcon.execute("SELECT mention_id FROM construct_mention WHERE extraction_run_id=%s AND definition_text IS NOT NULL ORDER BY mention_id LIMIT 10", (piped.extraction_run_id,))]
    assert quality.definition_precision_gate(pcon, piped.extraction_run_id)["status"] == "not_evaluable"
    for i, mention in enumerate(ids):
        extraction.record_decision(pcon, "construct_mention", mention, "validated", "Ana")
    extraction.record_decision(pcon, "construct_mention", ids[0], "rejected", "Ben")           # one rejection by anybody counts against it
    gate = quality.definition_precision_gate(pcon, piped.extraction_run_id)
    assert gate["value"] == pytest.approx(9 / 10) and gate["status"] == "fail"


def test_pair_gates_are_reported_per_class_with_the_fifty_case_warning(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    gold = quality.store_gold(pcon, "pairs", "planted", {"items": planted_gold_items(pcon, run_id)}, "Ana")
    gate = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, signal_run_id=run_id, gold_pairs=gold)}["jj_recall"]
    assert set(gate["per_class"]) == {"conceptual_jingle", "conceptual_jangle"} and all("at least 50" in w for w in gate["warnings"])
    counts = {k: v["gold_positives"] for k, v in gate["per_class"].items()}
    assert len(gate["warnings"]) == sum(1 for n in counts.values() if n < 50) and gate["per_class"]["conceptual_jingle"]["warning"]


# --- Counts, memberships and the release package -------------------------------------------------------------------------------------------

def test_release_package_has_counts_registry_memberships_and_model_card(piped, pcon, tmp_path):
    make_preassignment(pcon, piped.world)
    pre = pcon.execute("SELECT preassignment_run_id FROM preassignment_run").fetchone()[0]
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    release_id, _, _, _, manifest = release.build(pcon, piped.world.run_id, "pkg", pre, None, piped.ingestion_id, piped.extraction_run_id, run_id, operational_window=(2023, 2025))
    counts = manifest["counts"]
    assert counts["distinct_papers_in_any_community"] <= counts["papers_with_a_membership_label"]
    assert counts["in_both_communities"] > 0
    both = counts["in_both_communities"]
    total_fractional = sum(counts["fractional_1_over_k"].values())
    assert total_fractional == pytest.approx(counts["distinct_papers_in_any_community"])           # 1/k weighting counts every paper exactly once
    assert sum(counts["by_community"]["CTO"].values()) == counts["papers_with_a_membership_label"] and counts["by_community"]["CTO"]["yes"] + both >= both
    assert manifest["time_perimeter"]["operational_window_for_full_text"] == [2023, 2025] and manifest["time_perimeter"]["papers_in_window"] > 0
    assert manifest["signals"]["availability"]["s_def"] > 0 and manifest["signals"]["warning"].startswith("S_def, S_op and S_sem-op were computed with a lexical")
    assert manifest["model_card"]["components"]["signals"]["embedding_model"] == "hash-rp-v1" and manifest["model_card"]["known_limits"]
    assert sum(manifest["full_text_status"].values()) == manifest["catalog"]["eligible_papers"]
    out = release.write_report(pcon, release_id, tmp_path / "rel")
    names = {p.name for p in out.iterdir()}
    assert {"manifest.json", "release_report.md", "memberships.csv", "construct_registry.csv", "construct_relations.csv", "model_card.json", "counts.json"} <= names
    registry = list(csv.DictReader((out / "construct_registry.csv").open(encoding="utf-8")))
    assert registry and all(r["status"] == "provisional" for r in registry) and any(r["definition"] for r in registry) and any("level=" in r["meaning_profile"] for r in registry)
    assert any("referent=" in r["meaning_profile"] for r in registry) and "operationalization" in registry[0] and any(r["measures"] for r in registry)


def test_extraction_validation_sample_takes_ten_percent_or_all_and_counts_shared_papers_once(piped, pcon):
    make_preassignment(pcon, piped.world)
    pre = pcon.execute("SELECT preassignment_run_id FROM preassignment_run").fetchone()[0]
    ids, design = audit.extraction_validation_sample(pcon, piped.world.run_id, pre)
    assert len(ids) == len(set(ids)) and all(design[c]["complete"] for c in ("CTO", "TIM"))            # the mock corpus is small: everything is validated, and it says so
    assert "lower statistical precision" in design["CTO"]["note"]
    again = audit.extraction_validation_sample(pcon, piped.world.run_id, pre)[0]
    assert again == ids


# --- Retrieval: community filter, diversity, index ----------------------------------------------------------------------------------------

@pytest.fixture
def validated(piped, pcon, monkeypatch):
    from research0 import documents
    make_preassignment(pcon, piped.world)
    pre = pcon.execute("SELECT preassignment_run_id FROM preassignment_run").fetchone()[0]
    documents.embed_chunks(pcon, piped.ingestion_id, HashEmbedder())
    monkeypatch.setattr(quality, "evaluate", lambda *a, **k: [quality.verdict(g["id"], quality.passing_value(g)) for g in quality.GATES])
    release_id, *_ = release.build(pcon, piped.world.run_id, "al", pre, None, piped.ingestion_id, piped.extraction_run_id)
    release.sign_off(pcon, release_id, "Ana")
    pcon.commit()
    yield release_id
    pcon.rollback()
    for table in ("retrieval_hit", "retrieval_run", "agent_profile", "release_signoff", "release", "chunk_embedding", "community_preassignment", "preassignment_run"):
        pcon.execute(f"DELETE FROM {table}")
    pcon.commit()


def test_community_view_and_diversity_limit_what_an_agent_reads(piped, pcon, validated):
    world = piped.world
    truth = {p.canonical_id: p.truth for p in world.papers}
    query = "positively predicted performance measured scale items organizational"
    both = rag.search(pcon, validated, query, HashEmbedder(), k=30, log=False, max_per_paper=50)
    only_cto = rag.search(pcon, validated, query, HashEmbedder(), k=30, log=False, community="CTO", max_per_paper=50)
    assert only_cto and all(truth[uuid.UUID(h["canonical_id"])]["CTO"] for h in only_cto)             # the CTO view holds only CTO members
    assert any(not truth[uuid.UUID(h["canonical_id"])]["CTO"] for h in both)
    capped = rag.search(pcon, validated, query, HashEmbedder(), k=30, log=False, max_per_paper=1)
    papers = [h["canonical_id"] for h in capped]
    assert len(papers) == len(set(papers)) and len(capped) > 1                                         # one chunk per paper
    # an agent profile fixes its community and policy
    pcon.execute("INSERT INTO agent_profile(agent_id,community,retrieval_policy) VALUES ('tim-agent','TIM','{\"max_per_paper\": 1}')")
    pcon.commit()
    hits = rag.search(pcon, validated, query, HashEmbedder(), k=20, agent="tim-agent")
    assert hits and all(truth[uuid.UUID(h["canonical_id"])]["TIM"] for h in hits) and len({h["canonical_id"] for h in hits}) == len(hits)
    assert pcon.execute("SELECT scope->>'community', params->>'max_per_paper' FROM retrieval_run WHERE agent='tim-agent'").fetchone() == ("TIM", "1")


def test_a_reranker_reorders_and_the_vector_index_does_not_change_the_answer(piped, pcon, validated):
    query = "digital capacity skills infrastructure leadership"
    plain = rag.search(pcon, validated, query, HashEmbedder(), k=5, log=False)
    reranked = rag.search(pcon, validated, query, HashEmbedder(), k=5, log=False, reranker=lambda q, hits: {h["chunk_id"]: 1.0 if h["section"] == "method" else 0.0 for h in hits})
    assert reranked[0]["section"] == "method" and "rerank_score" in reranked[0]["components"]
    assert {h["chunk_id"] for h in plain} != set() and plain[0]["components"].get("rerank_score") is None
    name = rag.create_vector_index(pcon, "hash-rp-v1", "hnsw")
    assert name.startswith("chunk_embedding_hnsw_") and rag.create_vector_index(pcon, "hash-rp-v1", "hnsw") == name
    assert [h["chunk_id"] for h in rag.search(pcon, validated, query, HashEmbedder(), k=5, log=False)] == [h["chunk_id"] for h in plain]
    pcon.execute("INSERT INTO embedding_model(model_id,kind,dimensions,description) VALUES ('big','semantic',3072,'x')")
    with pytest.raises(ValueError, match="cannot index 3072"):
        rag.create_vector_index(pcon, "big")


# --- Time window and the candidate scope for the full texts ---------------------------------------------------------------------------------

def test_the_operational_window_and_the_candidates_scope_choose_the_full_texts(world, con):
    assert fulltext.parse_years("2023-2025") == (2023, 2025) and fulltext.parse_years("2024") == (2024, 2024) and fulltext.parse_years(None) is None
    for bad in ("2025-2023", "abc", "1800-1900", "2020-2021-2022"):
        with pytest.raises(ValueError):
            fulltext.parse_years(bad)
    everyone = fulltext.select_papers(con, world.run_id, "eligible")
    window = fulltext.select_papers(con, world.run_id, "eligible", years=(2023, 2025))
    years = {cid: y for cid, y in con.execute("SELECT canonical_id,publication_year FROM canonical_paper WHERE run_id=%s", (world.run_id,))}
    assert window and len(window) < len(everyone) and all(2023 <= years[cid] <= 2025 for cid, _ in window)
    with pytest.raises(ValueError, match="No pre-assignment run"):
        fulltext.select_papers(con, world.run_id, "candidates")
    make_preassignment(con, world)
    picked = {cid for cid, _ in fulltext.select_papers(con, world.run_id, "candidates")}
    members = {p.canonical_id for p in world.eligible() if p.pdf is None and (p.truth["CTO"] or p.truth["TIM"] or p.abstract == "[No abstract available]")}
    assert picked and picked <= {p.canonical_id for p in world.eligible()}
    assert not {p.canonical_id for p in world.eligible() if p.pdf} & picked                           # papers that already have a file are not fetched again
    nobody = {p.canonical_id for p in world.eligible() if not (p.truth["CTO"] or p.truth["TIM"]) and p.abstract != "[No abstract available]"}
    assert not picked & nobody                                                                        # a paper the pre-assignment excluded is not a candidate


def test_the_review_set_holds_the_pairs_the_model_calls_a_conceptual_fallacy_and_is_never_prevalence(piped, pcon, tmp_path):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id, hints=proposals.TARGETED_HINTS)
    with pytest.raises(ValueError, match="No model proposal"):
        pairsample.create_review_set(pcon, run_id)
    manifest = proposals.prepare(pcon, run_id, tmp_path / "p", hints=proposals.TARGETED_HINTS)
    lines = (tmp_path / "p" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "p" / "batch_output.jsonl").write_text("\n".join(scripted_proposal_output(lines)) + "\n", encoding="utf-8")
    proposals.collect_file(pcon, tmp_path / "p")
    said = proposals.latest_classes(pcon, run_id)
    sample, design, existed = pairsample.create_review_set(pcon, run_id)
    items = {p: (k, s) for p, k, s in pcon.execute("SELECT pair_id,sample_kind,stratum FROM pair_sample_item WHERE pair_sample_id=%s", (sample,))}
    assert set(items) == {p for p, c in said.items() if c in pairs.POSITIVE} and items and not existed
    assert all(k == "challenge" and s == "llm_" + said[p] for p, (k, s) in items.items())
    assert pairsample.latest(pcon, run_id) == sample and pairsample.report(pcon, sample)["probabilistic"]["pairs"] == 0      # the queue serves it; never prevalence


def test_the_excel_export_holds_only_the_requested_model_classes_with_both_uses_side_by_side(piped, pcon, tmp_path):
    from openpyxl import load_workbook
    from research0 import proposal_export
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id, hints=proposals.TARGETED_HINTS)
    proposals.prepare(pcon, run_id, tmp_path / "p", hints=proposals.TARGETED_HINTS)
    lines = (tmp_path / "p" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "p" / "batch_output.jsonl").write_text("\n".join(scripted_proposal_output(lines)) + "\n", encoding="utf-8")
    proposals.collect_file(pcon, tmp_path / "p")
    said = proposals.latest_classes(pcon, run_id)
    out = proposal_export.export_xlsx(pcon, run_id, tmp_path / "x.xlsx")
    wb = load_workbook(tmp_path / "x.xlsx")
    sheet = wb["Pairs"]
    header = [c.value for c in sheet[1]]
    body = list(sheet.iter_rows(min_row=2, values_only=True))
    assert out["pairs"] == len(body) == sum(1 for c in said.values() if c in pairs.POSITIVE) > 0
    assert {r[0] for r in body} <= {pairs.CLASS_LABELS[c] for c in pairs.POSITIVE}                          # only jingle and jangle
    assert {r[header.index("Pair ID")] for r in body} == {str(p) for p, c in said.items() if c in pairs.POSITIVE}
    assert all(r[header.index("A: Label")] and r[header.index("B: Label")] and r[header.index("A: Journal")] for r in body)
    assert all(r[header.index("Reasons (model)")] for r in body) and "not decisions" in wb["Read me"]["A2"].value
    with pytest.raises(ValueError, match="Classes are"):
        proposal_export.export_xlsx(pcon, run_id, tmp_path / "y.xlsx", ("bogus",))


def test_a_corpus_tag_gives_a_rebuilt_signal_run_its_own_id_and_leaves_the_untagged_one_alone(piped, pcon):
    """Rows added to an extraction run after a build make the same inputs an 'already built' run; a tag in the params is the way to a new, separate run."""
    first, counts, existed = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    again, _, existed_again = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    tagged, tagged_counts, tagged_existed = signals.build(pcon, piped.extraction_run_id, HashEmbedder(), {"corpus_tag": "rows-added-later"})
    assert again == first and existed_again                                 # same inputs: the earlier run is returned untouched
    assert tagged != first and not tagged_existed and tagged_counts == counts
    assert pcon.execute("SELECT params->>'corpus_tag' FROM signal_run WHERE signal_run_id=%s", (tagged,)).fetchone()[0] == "rows-added-later"
    assert pcon.execute("SELECT params ? 'corpus_tag' FROM signal_run WHERE signal_run_id=%s", (first,)).fetchone()[0] is False
