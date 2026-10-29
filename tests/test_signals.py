import uuid

import pytest

from research0 import signals
from research0.embeddings import HashEmbedder
from research0.mock_world import TRUTH_PAIRS

P = signals.DEFAULTS


def entry(sign="positive", status="supported"):
    return {"sign": sign, "status": status}


def test_jaccard_missing_data_is_none_not_zero():
    assert signals.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert signals.jaccard(set(), {"a"}) is None


def test_n_nom_compares_homologous_roles_only_and_skips_missing_ones():
    sim = signals.LabelSimilarity()
    a = {"outcomes": {"job performance": entry()}, "antecedents": {"algorithmic control": entry()}}
    b = {"outcomes": {"job performance": entry("negative")}, "moderators": {"tenure": entry()}}
    value, detail = signals.n_nom(a, b, sim, 0.8)
    assert value == pytest.approx(1.0) and detail["roles_compared"] == 1                  # only the outcomes exist on both sides
    assert detail["antecedents"]["compared"] is False and detail["moderators"]["compared"] is False
    assert detail["outcomes"]["sign_agreement"] == 0                                        # the sign is reported apart; it does not change the value
    assert "not approved identities" in detail["basis"]
    assert signals.n_nom({"antecedents": {"x": entry()}}, {"outcomes": {"x": entry()}}, sim, 0.8)[0] is None   # an antecedent is never compared with an outcome
    assert signals.n_nom({}, {"outcomes": {"x": entry()}}, sim, 0.8)[0] is None


def test_role_similarity_is_a_soft_weighted_jaccard_with_proposed_links():
    sim = signals.LabelSimilarity()
    value, detail = signals.role_similarity({"job performance": entry(), "trust": entry()}, {"job performance": entry(), "slack": entry()}, sim, 0.8)
    assert value == pytest.approx(1 / 3) and detail["matched"][0][:2] == ["job performance", "job performance"]
    near, _ = signals.role_similarity({"job performance": entry()}, {"job performances": entry()}, sim, 0.8)
    far, _ = signals.role_similarity({"job performance": entry()}, {"nursing staff": entry()}, sim, 0.8)
    assert near > 0.8 > far == 0


def test_hint_reads_s_sem_op_only_and_n_nom_is_not_part_of_it():
    h = signals.hint_for
    assert h(1.0, 0.1, P) == "jingle_candidate"               # same label, operationalisation diverges
    assert h(1.0, 0.9, P) == "none"                           # the same label used consistently is not a candidate
    assert h(1.0, 0.45, P) == "none"                          # between the thresholds: nothing to flag
    assert h(0.2, 0.8, P) == "jangle_candidate"               # different labels, operationalisation close
    assert h(0.2, 0.45, P) == "none" and h(0.2, 0.1, P) == "none"
    assert h(1.0, None, P) == "unresolved" and h(0.2, None, P) == "unresolved"     # a missing signal is not "different"
    assert "n_hi" not in P and "n_lo" not in P and signals.HINTS_V3 == ("jingle_candidate", "jangle_candidate", "related_but_distinct", "unresolved", "none")
    assert signals.citation_reading(0.2, P) == "same_conversation" and signals.citation_reading(0.0, P) == "separate_conversations" and signals.citation_reading(None, P) is None


def test_the_v4_hint_is_a_retrieval_stratum_read_from_the_meaning_first():
    h = lambda name, s_def, s_op=None, ca=frozenset(), cb=frozenset(): signals.hint_v4(name, s_def, s_op, set(ca), set(cb), P)
    assert signals.VERSION == "signals-v4" and signals.version_number(signals.VERSION) == 4 and signals.version_number("signals-v3") == 3 and signals.version_number(None) == 0
    assert h(1.0, 0.1) == "same_label_divergent" and h(0.2, 0.8) == "different_label_close"
    assert h(1.0, 0.9) == "hard_negative" and h(1.0, 0.9, None, {"hospital", "nurses"}, {"platform", "drivers"}) == "context_shifted"
    assert h(1.0, 0.9, None, {"hospital"}, set()) == "hard_negative"                        # an unknown context is not a different one
    assert h(0.2, 0.2, 0.9) == "close_measures_distinct_meaning"                           # close measures do not make one construct
    assert h(0.2, 0.45, 0.3) == "none" and h(1.0, 0.45) == "none"
    assert h(1.0, None, 0.9) == "unresolved" and h(0.2, None, 0.1) == "unresolved"          # without a meaning to compare, nothing is a candidate
    assert set(signals.HINTS) == {"same_label_divergent", "different_label_close", "close_measures_distinct_meaning", "context_shifted", "hard_negative", "unresolved", "none"}


def planted_kind(label_a, label_b):
    for pair, relation in TRUTH_PAIRS.items():
        if {label_a, label_b} == {key[0] for key in pair}:
            return relation
    return None


def test_candidate_pairs_contain_the_planted_jingle_jangle_and_related_pair(piped, pcon):
    run_id, counts, existed = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    assert not existed and counts["pairs"] > 0
    rows = pcon.execute(
        "SELECT a.label_normalized,b.label_normalized,p.name_sim,p.s_sem_op,p.c_cit,p.n_nom,p.hint,p.retrieval_reasons,a.canonical_id,b.canonical_id,p.c_cit_reading,p.n_nom_detail,p.s_def,p.s_op "
        "FROM candidate_pair p JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
        "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b WHERE p.signal_run_id=%s", (run_id,)).fetchall()
    assert all(r[8] != r[9] for r in rows)                                       # never two uses from the same paper
    found = {}
    for la, lb, name, sem, cit, nom, hint, reasons, *_ in rows:
        kind = planted_kind(la, lb)
        if kind:
            found.setdefault(kind, []).append((name, sem, nom, hint))
    assert set(found) == {"jingle", "jangle", "related_but_distinct"}              # retrieval reaches all three planted situations
    assert any(h == "same_label_divergent" for *_, h in found["jingle"])           # one label, diverging meanings
    assert any(h == "different_label_close" for *_, h in found["jangle"])          # different labels, close meanings
    jingle = [r for r in rows if r[0] == r[1] == "engagement"]
    assert jingle and all(r[2] == pytest.approx(1) for r in jingle) and min(r[12] for r in jingle if r[12] is not None) < 0.3
    assert any(r[12] is not None for r in rows) and any(r[13] is not None for r in rows) and any(r[3] is not None for r in rows)   # S_def, S_op and S_sem-op all exist
    reasons = {x for r in rows for x in r[7]}
    assert {"definition_neighbor", "same_label_divergent_definition", "measure_neighbor"} <= reasons
    assert any(r[4] is not None for r in rows) and any(r[5] is not None for r in rows)    # coupling and typed profiles exist for some pairs
    detailed = next(r for r in rows if r[5] is not None)
    assert detailed[11]["roles_compared"] >= 1 and detailed[11]["basis"]
    assert {r[10] for r in rows if r[4] is not None} <= {"same_conversation", "separate_conversations"}
    assert all(r[10] is None for r in rows if r[4] is None)
    again, _, existed = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    assert existed and again == run_id


def test_dossiers_hold_only_evidence_with_spans_and_the_candidate_relation_checklist(piped, pcon):
    run_id, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    created = signals.build_dossiers(pcon, run_id, hints=["same_label_divergent"], limit=3)
    assert 0 < created <= 3
    content, sha = pcon.execute("SELECT content,content_sha256 FROM pair_dossier LIMIT 1").fetchone()
    assert "not a verdict" in content["notice"] and content["triage_hint"] == "same_label_divergent" and content["version"] == "dossier-v3"
    for side in ("a", "b"):
        node = content[side]
        assert node["paper"]["title"] and node["mentions"] and all(m["quote"] and m["page"] >= 1 for m in node["mentions"])
        assert any(m["definition"] for m in node["mentions"])
        assert any(a["kind"] == "level" and a["basis"] == "extracted" and a["page"] >= 1 for a in node["attributes"])
        assert any(a["kind"] == "level" for a in node["meaning_profile"]) and "operationalization" in node        # meaning compared apart from the measures
    checklist = content["candidate_relation"]
    assert checklist["name_similarity"] >= P["name_hi"] and set(checklist) >= {"semantic_similarity_definition", "semantic_similarity_measures", "definition_and_measures_disagree",
                                                                 "same_scale", "nomological_network_similarity", "shared_citations", "contrary_evidence"}
    assert content["signals"]["c_cit_basis"].startswith("bibliographic coupling") and isinstance(content["signals"]["unavailable"], list) and "s_def" in content["signals"]
    assert len(sha) == 64 and any(q.startswith("Substitute:") for q in content["questions"]) and any("same meaning (no fallacy)" in q for q in content["questions"])


def test_different_parameters_make_a_new_run(piped, pcon):
    one, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    two, _, existed = signals.build(pcon, piped.extraction_run_id, HashEmbedder(), {"k_sem": 3})
    assert two != one and not existed


def test_uses_without_a_definition_are_left_out_before_pairing_and_counted(piped, pcon, monkeypatch):
    load = signals.load_nodes

    def some_undefined(con, run):              # every mock construct is defined: blank the meaning of one use in three
        nodes = load(con, run)
        for i, key in enumerate(sorted(nodes, key=str)):
            if i % 3 == 0:
                nodes[key]["def_text"] = None
        return nodes
    monkeypatch.setattr(signals, "load_nodes", some_undefined)
    kept, counts, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder(), {"k_sem": 7})          # params unlike the other tests: a fresh run
    everything, all_counts, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder(), {"k_sem": 7, "require": "none"})
    nodes = some_undefined(pcon, piped.extraction_run_id)
    undefined = sum(1 for n in nodes.values() if not n["def_text"])
    assert undefined > 0 and counts["excluded_nodes"] == undefined and all_counts["excluded_nodes"] == 0
    assert counts["nodes"] + undefined == all_counts["nodes"] == len(nodes)
    params = pcon.execute("SELECT params FROM signal_run WHERE signal_run_id=%s", (kept,)).fetchone()[0]
    assert params["require"] == "definition" and params["excluded_nodes"] == undefined
    hints = lambda run: {h for (h,) in pcon.execute("SELECT DISTINCT hint FROM candidate_pair WHERE signal_run_id=%s", (run,))}
    assert "unresolved" not in hints(kept) and "unresolved" in hints(everything)          # every kept pair has S_def on both sides
    assert pcon.execute("SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s AND s_def IS NULL", (kept,)).fetchone()[0] == 0
    with pytest.raises(ValueError, match="require"):
        signals.build(pcon, piped.extraction_run_id, HashEmbedder(), {"require": "bogus"})
