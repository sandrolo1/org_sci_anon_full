"""The statistics of the construct-pair tab and the blind validation sample, over the MOCK world only."""
import pytest

from research0 import pairs, pairsample, pairstats, signals
from research0.embeddings import HashEmbedder


def test_equal_allocation_takes_a_small_stratum_whole_and_shares_the_rest():
    out = pairsample.equal_allocation({"a": 100, "b": 100, "c": 3, "d": 100}, 30)
    assert out["c"] == 3 and sum(out.values()) == 30 and max(out.values()) - min(v for k, v in out.items() if k != "c") <= 1
    assert pairsample.equal_allocation({"a": 2, "b": 2}, 30) == {"a": 2, "b": 2}            # never more than the population
    assert sum(pairsample.equal_allocation({h: 1000 for h in "abcdef"}, 200).values()) == 200


@pytest.fixture
def sampled(piped, pcon):
    run, _, _ = signals.build(pcon, piped.extraction_run_id, HashEmbedder())
    sample, design, _ = pairsample.create(pcon, run, n_probabilistic=30, n_challenge=8, seed=11)
    ids = [r[0] for r in pcon.execute("SELECT pair_id FROM pair_sample_item WHERE pair_sample_id=%s", (sample,))]
    signals.build_dossiers(pcon, run, pair_ids=ids)
    return run, sample, set(ids)


def test_the_sample_holds_every_stratum_including_pairs_that_are_neither_jingle_nor_jangle(sampled, pcon):
    run, sample, _ = sampled
    design = pcon.execute("SELECT design FROM pair_sample WHERE pair_sample_id=%s", (sample,)).fetchone()[0]
    assert design["allocation"] == "equal"
    population = {f"{h}|{'same_label' if n >= 0.8 else 'different_label'}" for h, n in pcon.execute("SELECT hint,name_sim FROM candidate_pair WHERE signal_run_id=%s", (run,))}
    in_sample = {s for (s,) in pcon.execute("SELECT DISTINCT stratum FROM pair_sample_item WHERE pair_sample_id=%s AND sample_kind='probabilistic'", (sample,))}
    assert in_sample == population                                         # equal allocation reaches every stratum, the "none" ones included
    assert any(s.startswith("none|") for s in in_sample)
    rows = pcon.execute("SELECT DISTINCT stratum,population_size,sample_size FROM pair_sample_item WHERE pair_sample_id=%s AND sample_kind='probabilistic'", (sample,)).fetchall()
    partial = [size for _, population, size in rows if size < population]
    assert not partial or max(partial) - min(partial) <= 1                 # strata that are not taken whole get the same number of pairs, not a share of the population


def test_the_blind_queue_holds_only_sample_pairs_in_an_order_that_ignores_the_hint(sampled, pcon):
    run, sample, ids = sampled
    queue = pairs.queue(pcon, run, "Ana", 1000, sample)
    assert {q["pair_id"] for q in queue} == {str(i) for i in ids}
    assert [q["pair_id"] for q in queue] == [q["pair_id"] for q in pairs.queue(pcon, run, "Ben", 1000, sample)]     # the same order for everybody
    hints = [q["hint"] for q in queue]
    assert len(set(hints)) > 1 and hints != sorted(hints, key=pairs.PRIORITY.get)                                       # not "jingles first"
    unrestricted = pairs.queue(pcon, run, "Ana", 1000)
    assert [q["hint"] for q in unrestricted] == sorted((q["hint"] for q in unrestricted), key=pairs.PRIORITY.get)      # the curator's queue keeps the old order


def test_statistics_describe_the_run_and_hold_back_comparisons_until_two_people_finish(sampled, pcon):
    run, sample, ids = sampled
    out = pairstats.stats(pcon, run, "Ana")
    total = pcon.execute("SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s", (run,)).fetchone()[0]
    assert out["run"]["version"] == "signals-v4" and out["availability"]["pairs"] == total == sum(h["total"] for h in out["hints"]) and out["signal"] == "s_def"
    assert out["availability"]["n_nom_used_in_hint"] is False and out["availability"]["s_def"] == sum(b["same_label"] + b["different_label"] for b in out["s_sem_op"]["bins"])
    assert sum(b["pairs"] for b in out["c_cit"]["buckets"]) == out["availability"]["c_cit"] and out["thresholds"]["s_hi"] == signals.DEFAULTS["s_hi"]
    assert sum(r["pairs"] for r in out["retrieval_reasons"]) >= total
    review = out["review"]
    assert review["sample"]["pairs"] == len(ids) and review["states"]["undecided"] == len(ids) and review["revealed"] is False
    assert "triage_vs_human" not in review and "agreement" not in review and "estimates" not in review and review["held_back"]
    ordered = [q["pair_id"] for q in pairs.queue(pcon, run, None, 1000, sample)]
    unclear = {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}
    for pid in ordered:
        pairs.decide(pcon, pid, "Ana", "distinct", "two different constructs", unclear)
    assert pairstats.stats(pcon, run, "Ana")["review"]["revealed"] is False                    # one person is not enough
    assert pairstats.stats(pcon, run, "Ana")["review"]["my_decided"] == len(ids)
    for i, pid in enumerate(ordered):
        pairs.decide(pcon, pid, "Ben", "distinct" if i % 4 else "scope_variation", "x", unclear)
    review = pairstats.stats(pcon, run, "Ben")["review"]
    assert review["revealed"] is True and review["states"]["pending"] == 0
    assert review["agreement"][0]["annotators"] == ["Ana", "Ben"] and review["agreement"][0]["class"]["agreement"] == pytest.approx(1 - len(range(0, len(ids), 4)) / len(ids), abs=1e-4)
    assert review["agreement"][0]["fallacy"]["agreement"] == 1                          # nobody called a conceptual fallacy
    assert set(review["triage_vs_human"]) == {"probabilistic", "challenge"} and "warning" in review["estimates"] and review["codebook"] == "pair-codebook-v2"
    assert review["human"]["by_class"] == {"distinct": len(ids) - len(range(0, len(ids), 4))} and review["states"]["contested"] == len(range(0, len(ids), 4))     # only agreed pairs are settled
