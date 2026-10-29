import json
import random
import uuid

import pytest

from research0 import pairs, quality, release, signals
from research0.embeddings import HashEmbedder
from research0.mock_world import TRUTH_DECISIONS, TRUTH_PAIRS
from research0.mock_world import mock_audit


def test_every_gate_is_preregistered_with_the_documented_threshold():
    assert {g["id"]: g["threshold"] for g in quality.GATES} == {
        "seed_recovery": 0.95, "screening_recall": 0.95, "human_agreement": 0.80, "membership_macro_f1": 0.85, "multilabel_jaccard": 0.80, "multilabel_micro_f1": 0.85,
        "multilabel_macro_f1": 0.80, "extraction_f1": 0.90, "extraction_complete": 0.90, "extraction_definition_precision": 0.95, "jj_recall": 0.90, "jj_precision": 0.75,
        "jj_f1": 0.80, "rating_icc": 0.75, "rating_spearman": 0.70, "calibration_error": 0.10}
    assert quality.verdict("calibration_error", 0.10)["status"] == "pass" and quality.verdict("calibration_error", 0.11)["status"] == "fail"   # a lower-is-better gate
    assert quality.verdict("jj_f1", 0.80)["status"] == "pass" and quality.verdict("jj_f1", 0.7999)["status"] == "fail"
    assert quality.verdict("jj_f1", None)["status"] == "not_evaluable"


def test_gold_sets_need_a_provider_and_items_and_are_stored_once(world, con):
    with pytest.raises(ValueError):
        quality.store_gold(con, "pairs", "x", {"items": []}, "Ana")
    with pytest.raises(ValueError):
        quality.store_gold(con, "pairs", "x", {"items": [{"a": 1}]}, " ")
    one = quality.store_gold(con, "pairs", "x", {"items": [{"a": 1}]}, "Ana")
    assert quality.store_gold(con, "pairs", "x", {"items": [{"a": 1}]}, "Ana") == one
    assert con.execute("SELECT count(*) FROM gold_set").fetchone()[0] == 1


JINGLE = {"same_referent": "no", "compatible_properties": "no", "substitution_changes": "yes"}
JANGLE = {"same_referent": "yes", "compatible_properties": "yes", "substitution_changes": "no"}
UNCLEAR = {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}


def test_pair_decisions_follow_the_codebook_and_settle_only_by_agreement_or_an_adjudicator(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id, limit=60)
    rows = pcon.execute("SELECT d.pair_id,a.label_normalized=b.label_normalized FROM pair_dossier d JOIN candidate_pair p USING (pair_id) JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
                        "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b ORDER BY d.pair_id").fetchall()
    same = [pid for pid, eq in rows if eq]
    other = [pid for pid, eq in rows if not eq]
    a, c, d = same[:3]
    b = other[0]
    with pytest.raises(ValueError, match="dossier"):
        pairs.decide(pcon, uuid.uuid4(), "Ana", "distinct", "r", UNCLEAR)
    for bad in (("Ana", "twin", "r", UNCLEAR), ("Ana", "distinct", " ", UNCLEAR), (" ", "distinct", "r", UNCLEAR),
                ("Ana", "distinct", "r", {**UNCLEAR, "same_referent": "maybe"}),            # every comparison question needs yes, no or unclear
                ("Ana", "conceptual_jingle", "r", JINGLE),                                  # a conceptual fallacy needs its theoretical consequence
                ("Ana", "conceptual_jangle", "r", JANGLE, "a consequence"),                 # one label cannot be a jangle
                ("Ana", "conceptual_jingle", "r", {**JINGLE, "substitution_changes": "no"}, "a consequence"),   # nothing changes: measurement or scope, not a jingle
                ("Ana", "conceptual_jingle", "r", {**JINGLE, "compatible_properties": "yes"}, "a consequence")):
        with pytest.raises(ValueError):
            pairs.decide(pcon, a, *bad)
    with pytest.raises(ValueError, match="different labels"):
        pairs.decide(pcon, b, "Ana", "conceptual_jingle", "r", JINGLE, "c")
    with pytest.raises(ValueError, match="empirical redundancy"):
        pairs.decide(pcon, b, "Ana", "conceptual_jangle", "r", {**JANGLE, "same_referent": "unclear"}, "c")
    with pytest.raises(ValueError, match="conceptual jangle"):
        pairs.decide(pcon, b, "Ana", "same_meaning", "r", JANGLE)
    pairs.decide(pcon, a, "Ana", "conceptual_jingle", "different definitions", JINGLE, "Two outcomes are predicted, not one.")
    assert pairs.settled(pcon, run_id) == {}                                         # one opinion settles nothing
    pairs.decide(pcon, a, "Ben", "conceptual_jingle", "agree", JINGLE, "The mechanisms differ.", "A paper that measures both")
    pairs.decide(pcon, b, "Ana", "conceptual_jangle", "same def", JANGLE, "The antecedents transfer.")
    pairs.decide(pcon, b, "Ben", "empirical_redundancy", "they only correlate", UNCLEAR)
    pairs.decide(pcon, c, "Ana", "related_but_distinct", "x", UNCLEAR)
    pairs.decide(pcon, c, "Chi", "measurement_divergence", "adjudicated", UNCLEAR, is_final=True)
    pairs.decide(pcon, c, "Ben", "unresolved", "cannot tell", UNCLEAR)
    pairs.decide(pcon, d, "Ana", "distinct", "first", UNCLEAR)
    pairs.decide(pcon, d, "Ana", "unresolved", "changed my mind", UNCLEAR)          # a changed decision is a new row; the last counts
    settled = pairs.settled(pcon, run_id)
    assert settled == {a: "conceptual_jingle", c: "measurement_divergence"} and pairs.state_of(pairs.current(pcon, run_id)[b])[0] == "contested"
    pairs.decide(pcon, d, "Ben", "scope_variation", "x", UNCLEAR)
    assert pairs.state_of(pairs.current(pcon, run_id)[d])[0] == "contested"
    assert pcon.execute("SELECT count(*) FROM pair_decision WHERE pair_id=%s", (d,)).fetchone()[0] == 3
    tally = pairs.counts(pcon, run_id)
    assert tally["settled"] == 2 and tally["contested"] == 2 and tally["pending"] == 0 and tally["by_class"] == {"conceptual_jingle": 1, "measurement_divergence": 1}
    info = pairs.details(pcon, [a])[a]
    assert {x["validator"] for x in info} == {"Ana", "Ben"} and all(x["substitution_changes"] == "yes" and x["theoretical_consequence"] for x in info)
    queue = pairs.queue(pcon, run_id, validator="Ana")
    assert all(item["pair_id"] not in (str(a), str(b), str(c), str(d)) for item in queue)
    assert "signals" not in pairs.blind(queue[0]["dossier"]) if queue else True


def test_decisions_of_the_earlier_codebook_are_kept_as_history_and_never_counted(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    signals.build_dossiers(pcon, run_id, limit=3)
    first = pcon.execute("SELECT pair_id FROM pair_dossier ORDER BY pair_id").fetchone()[0]
    for who in ("Ana", "Ben"):            # as the dashboard recorded them before 13 Oct 2026
        pcon.execute("INSERT INTO pair_decision(pair_id,relation,flag,rationale,validator) VALUES (%s,'distinct','jingle','old scale',%s)", (first, who))
    assert pcon.execute("SELECT DISTINCT codebook_version FROM pair_decision").fetchall() == [(pairs.LEGACY_CODEBOOK,)]
    assert pairs.settled(pcon, run_id) == {} and pairs.current(pcon, run_id) == {}
    assert pairs.counts(pcon, run_id)["earlier_codebook"]["decisions"] == 2 and pairs.history(pcon, run_id) == (2, 1)
    assert str(first) in {q["pair_id"] for q in pairs.queue(pcon, run_id, validator="Ana")}              # the pair waits to be decided again
    with pytest.raises(Exception):                                                                      # the schema refuses a v2 row without its checklist
        with pcon.transaction():
            pcon.execute("INSERT INTO pair_decision(pair_id,codebook_version,decision_class,rationale,validator) VALUES (%s,'pair-codebook-v2','distinct','r','Cy')", (first,))


def test_blind_view_hides_signals_and_hint():
    content = {"pair_id": "x", "signals": {"s_sem_op": 0.1}, "triage_hint": "jingle_candidate", "retrieval_reasons": ["name_neighbor"], "a": {}, "b": {}}
    assert set(pairs.blind(content)) == {"pair_id", "a", "b"}


def gold_for_planted_pairs(con, run_id):
    """Gold built from the planted truth (the one thing the mock world knows for sure)."""
    items = []
    rows = con.execute("SELECT a.canonical_id,a.label_normalized,b.canonical_id,b.label_normalized FROM candidate_pair p "
                       "JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b "
                       "WHERE p.signal_run_id=%s", (run_id,)).fetchall()
    for ca, la, cb, lb in rows:
        for pair, relation in TRUTH_PAIRS.items():
            if {la, lb} == {key[0] for key in pair}:
                items.append({"canonical_id_a": str(ca), "label_a": la, "canonical_id_b": str(cb), "label_b": lb, "class": TRUTH_DECISIONS[relation][0]})
    return items


def test_pair_gates_measure_recall_and_precision_against_gold(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    items = gold_for_planted_pairs(pcon, run_id)
    assert items
    gold_id = quality.store_gold(pcon, "pairs", "planted", {"items": items}, "Ana")
    gates = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, signal_run_id=run_id, gold_pairs=gold_id)}
    assert gates["jj_recall"]["value"] is not None and gates["jj_precision"]["value"] is not None
    assert 0 <= gates["jj_f1"]["value"] <= 1
    per_class = gates["jj_recall"]["per_class"]
    assert set(per_class) == {"conceptual_jingle", "conceptual_jangle"} and gates["jj_recall"]["value"] == min(c["recall"] for c in per_class.values())   # each class must pass
    assert set(gates["jj_f1"]["near_negatives"]) == {"measurement_divergence", "scope_variation", "empirical_redundancy"}
    only_jingle = quality.store_gold(pcon, "pairs", "jingle only", {"items": [i for i in items if i["class"] == "conceptual_jingle"]}, "Ana")
    lonely = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, signal_run_id=run_id, gold_pairs=only_jingle)}
    assert lonely["jj_f1"]["status"] == "not_evaluable" and "both positive classes" in lonely["jj_f1"]["detail"]
    legacy = quality.store_gold(pcon, "pairs", "earlier codebook", {"items": [{**{k: v for k, v in i.items() if k != "class"}, "flag": {"conceptual_jingle": "jingle", "conceptual_jangle": "jangle"}.get(i["class"], "none")} for i in items]}, "Ana")
    old = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, signal_run_id=run_id, gold_pairs=legacy)}
    assert old["jj_f1"]["value"] == gates["jj_f1"]["value"] and any("earlier codebook" in w for w in old["jj_f1"]["warnings"])
    # a gold pair the pipeline never proposed counts as a miss
    ghost = {"canonical_id_a": str(uuid.uuid4()), "label_a": "ghost one", "canonical_id_b": str(uuid.uuid4()), "label_b": "ghost two", "class": "conceptual_jangle"}
    worse = quality.store_gold(pcon, "pairs", "planted+ghost", {"items": items + [ghost]}, "Ana")
    again = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, signal_run_id=run_id, gold_pairs=worse)}
    assert again["jj_recall"]["per_class"]["conceptual_jangle"]["recall"] < gates["jj_recall"]["per_class"]["conceptual_jangle"]["recall"]
    assert again["jj_recall"]["value"] <= gates["jj_recall"]["value"]


def test_rating_gates(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    rows = pcon.execute("SELECT a.canonical_id,a.label_normalized,b.canonical_id,b.label_normalized,p.s_def FROM candidate_pair p "
                        "JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b "
                        "WHERE p.signal_run_id=%s AND p.s_def IS NOT NULL ORDER BY p.pair_id LIMIT 30", (run_id,)).fetchall()
    rng = random.Random(2)
    items = [{"canonical_id_a": str(ca), "label_a": la, "canonical_id_b": str(cb), "label_b": lb,
              "ratings": {"Ana": min(1, max(0, s + rng.uniform(-.05, .05))), "Ben": min(1, max(0, s + rng.uniform(-.05, .05)))}} for ca, la, cb, lb, s in rows]
    gates = {g["id"]: g for g in quality.rating_gates(pcon, {"items": items}, run_id)}
    assert gates["rating_icc"]["status"] == "pass" and gates["rating_spearman"]["status"] == "pass"   # ratings built around the pipeline value agree with it
    sparse = {g["id"]: g for g in quality.rating_gates(pcon, {"items": items[:2]}, run_id)}
    assert sparse["rating_icc"]["status"] == "not_evaluable"


def test_release_is_blocked_without_gold_and_never_validated_by_signoff_alone(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    release_id, created, passed, gates, manifest = release.build(pcon, piped.world.run_id, "mock-1", ingestion_run_id=piped.ingestion_id,
                                                                  extraction_run_id=piped.extraction_run_id, signal_run_id=run_id)
    assert created and not passed and release.status(pcon, release_id) == "blocked"
    assert {g["status"] for g in gates} >= {"not_evaluable"} and manifest["signals"]["warning"].startswith("S_def, S_op and S_sem-op were computed with a lexical")
    assert set(manifest["signals"]["availability"]) == {"s_def", "s_op", "s_sem_op", "c_cit", "n_nom"} and manifest["signals"]["retrieval_strata"]
    with pytest.raises(ValueError, match="blocked"):
        release.sign_off(pcon, release_id, "Ana")
    assert release.latest_validated(pcon, piped.world.run_id) is None
    same, created_again, *_ = release.build(pcon, piped.world.run_id, "mock-1", ingestion_run_id=piped.ingestion_id, extraction_run_id=piped.extraction_run_id, signal_run_id=run_id)
    assert same == release_id and not created_again                                    # idempotent: same content, same release


def test_release_passes_and_is_validated_only_with_every_gate_and_a_signoff(piped, pcon, monkeypatch, tmp_path):
    """With every gate forced to a pass, a signed-off release becomes validated; the chain blocked -> candidate -> validated is the only path."""
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    monkeypatch.setattr(quality, "evaluate", lambda *a, **k: [quality.verdict(g["id"], quality.passing_value(g), "forced for the test") for g in quality.GATES])
    release_id, _, passed, _, _ = release.build(pcon, piped.world.run_id, "forced", signal_run_id=run_id, extraction_run_id=piped.extraction_run_id)
    assert passed and release.status(pcon, release_id) == "candidate" and release.latest_validated(pcon, piped.world.run_id) is None
    release.sign_off(pcon, release_id, "Ana", "checked")
    assert release.status(pcon, release_id) == "validated" and release.latest_validated(pcon, piped.world.run_id) == release_id
    out = release.write_report(pcon, release_id, tmp_path / "rel")
    text = (out / "release_report.md").read_text(encoding="utf-8")
    assert "validated" in text and "Ana" in text and json.loads((out / "manifest.json").read_text())["schema"] == release.SCHEMA


def test_release_report_for_a_blocked_release_says_what_to_do(piped, pcon, tmp_path):
    release_id, *_ = release.build(pcon, piped.world.run_id, "blocked-one")
    text = (release.write_report(pcon, release_id, tmp_path / "rel") / "release_report.md").read_text(encoding="utf-8")
    assert "suspended" in text and "not_evaluable" in text


def test_membership_gates_come_from_the_audit(world, con):
    pre, audit_id = mock_audit(con, world)
    gates = {g["id"]: g for g in quality.membership_gates(con, audit_id)}
    assert gates["human_agreement"]["value"] is not None and gates["membership_macro_f1"]["value"] is not None
    assert gates["human_agreement"]["status"] in ("pass", "fail")
    assert {g["id"]: g["status"] for g in quality.membership_gates(con, None)} == dict.fromkeys(quality.AUDIT_GATES, "not_evaluable")
    seed = quality.seed_gate(con, world.run_id)
    assert seed["status"] == "pass" and seed["value"] == 1.0


def test_extraction_gates_against_gold_built_from_the_planted_truth(piped, pcon):
    papers = [p for p in piped.world.papers if p.pdf][:10]
    items = [{"canonical_id": str(p.canonical_id), "constructs": [c["label"] for c in p.planted["constructs"]],
              "measures": [{"construct": m["construct"][0], "name": m["name"]} for m in p.planted["measures"]],
              "claims": [{"subject": c["subject"][0], "object": c["object"][0], "relation": c["relation"], "direction": c["direction"]} for c in p.planted["claims"]]} for p in papers]
    gold = quality.store_gold(pcon, "extraction", "planted", {"items": items}, "Ana")
    gates = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, extraction_run_id=piped.extraction_run_id, gold_extraction=gold)}
    assert gates["extraction_f1"]["value"] == pytest.approx(1.0) and gates["extraction_complete"]["value"] == pytest.approx(1.0)
    items[0]["claims"].append({"subject": "ghost", "object": "phantom", "relation": "predicts", "direction": "positive"})   # a gold item the pipeline missed
    worse = quality.store_gold(pcon, "extraction", "planted+1", {"items": items}, "Ana")
    again = {g["id"]: g for g in quality.evaluate(pcon, piped.world.run_id, extraction_run_id=piped.extraction_run_id, gold_extraction=worse)}
    assert again["extraction_f1"]["value"] < 1 and again["extraction_complete"]["value"] == pytest.approx(0.9)
