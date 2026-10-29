import random
import uuid

import pytest

from research0 import audit
from research0.mock_world import make_preassignment


def test_allocation_respects_minimums_and_size():
    population = {"CTO=yes|TIM=no": 4000, "CTO=no|TIM=no": 20000, "CTO=uncertain|TIM=uncertain": 15, "CTO=yes|TIM=yes": 900}
    sizes = audit.allocate(population, 300, 20, 60)
    assert sum(sizes.values()) == 300 and sizes["CTO=uncertain|TIM=uncertain"] == 15     # a small stratum is taken whole
    assert sizes["CTO=no|TIM=no"] >= 60 and all(sizes[h] >= 20 for h in sizes if population[h] >= 20)
    assert all(sizes[h] <= population[h] for h in sizes)
    assert audit.allocate(population, 300, 20, 60) == sizes                                 # deterministic
    with pytest.raises(ValueError, match="minimums need"):
        audit.allocate(population, 50, 20, 60)
    assert audit.allocate({"a": 5, "CTO=no|TIM=no": 3}, 100, 20, 60) == {"a": 5, "CTO=no|TIM=no": 3}   # never more than exists


def test_estimates_recover_a_known_population():
    """Build a population with known truth, audit it by the design, and check the true value lies in the reported interval."""
    rng = random.Random(5)
    # strata: (model label, N, true yes rate). The 'no' stratum hides 6% false negatives.
    spec = [("yes", 3000, 0.80), ("no", 20000, 0.06), ("uncertain", 500, 0.30)]
    truth = {m: [rng.random() < p for _ in range(N)] for m, N, p in spec}
    tp = sum(truth["yes"]); fp = 3000 - tp; fn = sum(truth["no"]) + sum(truth["uncertain"]); tn = 20500 - fn
    true_macro = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp)) / 2
    rows = []
    for m, N, _ in spec:
        n = {"yes": 120, "no": 400, "uncertain": 60}[m]
        picked = rng.sample(truth[m], n)
        rows.append({"stratum": f"CTO={m}|TIM=no", "model": m, "N": N, "n": n, "yes": sum(picked), "uncertain": 0})
    result = audit.stratified_estimates(rows, seed=3)
    low, high = result["strict"]["macro_f1_ci95"]
    assert low <= true_macro <= high and high - low < 0.08
    fn_rate = result["false_negatives_among_no"]
    assert fn_rate["ci95"][0] <= 0.06 <= fn_rate["ci95"][1] + 0.01 and fn_rate["population"] == 20000
    assert result["design_warnings"] == []


def test_warnings_for_unaudited_and_tiny_strata():
    rows = [{"stratum": "a", "model": "yes", "N": 100, "n": 0, "yes": 0}, {"stratum": "b", "model": "no", "N": 100, "n": 4, "yes": 1}]
    result = audit.stratified_estimates(rows, draws=50)
    assert len(result["design_warnings"]) == 2 and "no audited paper" in result["design_warnings"][0]
    census = audit.stratified_estimates([{"stratum": "c", "model": "yes", "N": 10, "n": 10, "yes": 8}], draws=20)
    assert census["design_warnings"] == [] and census["strict"]["precision"] == pytest.approx(0.8)


def test_full_audit_workflow_on_mock_world(world, con, tmp_path):
    pre = make_preassignment(con, world)
    audit_id, design, existed = audit.create_sample(con, world.run_id, pre, n_total=120, min_per_stratum=8, min_excluded=15, seed=4)
    assert not existed and sum(design["sample"].values()) == 120
    assert sum(n for h, n in design["sample"].items() if h.startswith(audit.EXCLUDED)) >= 15
    assert {h.split("|")[2] for h in design["sample"]} >= {"conf=high", "conf=low"}          # strata by predicted membership and confidence band
    assert audit.create_sample(con, world.run_id, pre, n_total=120, min_per_stratum=8, min_excluded=15, seed=4)[2]
    papers = audit.blind_papers(con, audit_id, "Ana")
    assert len(papers) == 120 and not any(k in papers[0] for k in ("stratum", "label", "model"))
    assert [p["position"] for p in papers] == list(range(1, 121))
    with pytest.raises(ValueError):
        audit.agreement(con, audit_id)                         # nobody has rated yet

    truth = {p.canonical_id: p.truth for p in world.papers}
    flipped = 0
    for who, error_every in (("Ana", 0), ("Ben", 7)):
        for i, paper in enumerate(papers):
            cid = uuid.UUID(paper["id"])
            for community in ("CTO", "TIM"):
                label = "yes" if truth[cid][community] else "no"
                if error_every and (i + (community == "TIM")) % error_every == 0:
                    label = "no" if label == "yes" else "yes"; flipped += 1
                audit.rate(con, audit_id, cid, community, who, label)
    audit.rate(con, audit_id, uuid.UUID(papers[0]["id"]), "CTO", "Ana", "uncertain")      # a changed rating is a new row; the last one counts
    audit.rate(con, audit_id, uuid.UUID(papers[0]["id"]), "CTO", "Ana", "yes" if truth[uuid.UUID(papers[0]["id"])]["CTO"] else "no")
    assert audit.status(con, audit_id, "Ana")["my_completed"] == 120
    assert con.execute("SELECT count(*) FROM audit_rating WHERE audit_id=%s", (audit_id,)).fetchone()[0] == 240 * 2 + 2

    agreement = audit.agreement(con, audit_id)
    assert agreement["annotators"] == ["Ana", "Ben"] and agreement["disagreements"] > 0 and agreement["pending_adjudication"] == agreement["disagreements"]
    assert 0.3 < agreement["CTO"]["three_class"]["kappa"] < 1
    with pytest.raises(ValueError, match="waiting for an adjudicator"):
        audit.estimates(con, audit_id)

    for item in audit.adjudication_items(con, audit_id):
        assert "stratum" not in item and set(item["ratings"]) == {"Ana", "Ben"}
        audit.adjudicate(con, audit_id, uuid.UUID(item["id"]), item["community"], "Chi", "yes" if truth[uuid.UUID(item["id"])][item["community"]] else "no")
    with pytest.raises(ValueError, match="agree"):
        agreed = next(p for p in papers if (uuid.UUID(p["id"]), "TIM") not in set(audit.disagreements(con, audit_id)))
        audit.adjudicate(con, audit_id, uuid.UUID(agreed["id"]), "TIM", "Chi", "yes")
    final, pending = audit.final_labels(con, audit_id)
    assert not pending and all(final[(uuid.UUID(p["id"]), c)] == ("yes" if truth[uuid.UUID(p["id"])][c] else "no") for p in papers for c in ("CTO", "TIM"))
    estimate = audit.estimates(con, audit_id)
    screening = estimate["screening"]["strict"]
    assert 0 < screening["recall"] <= 1 and screening["recall_ci95"][0] <= screening["recall"] <= screening["recall_ci95"][1]
    multi = estimate["multilabel"]
    assert 0 < multi["mean_jaccard"] <= 1 and multi["included"] < multi["papers"] and multi["abstention_rate_weighted"] > 0   # abstentions are left out and reported
    assert multi["gold_classes_audited"] and multi["mean_jaccard_ci95"][0] <= multi["mean_jaccard"] <= multi["mean_jaccard_ci95"][1]
    ece = estimate["calibration"]
    assert ece["points"] > 0 and 0 <= ece["ece"] <= 1
    # the pre-assignment missed about 10% of true members, so recall must be below 1 and the missed members visible among the 'no' papers
    assert estimate["CTO"]["strict"]["recall"] < 1 and estimate["CTO"]["false_negatives_among_no"]["human_yes"] > 0
    assert 0 < estimate["uncertain_rate_of_preassignment"]["CTO"] < 0.2
    report = audit.write_report(estimate, agreement, tmp_path / "audit.md")
    text = report.read_text(encoding="utf-8")
    assert "Macro-F1" in text and "design-based" in text and "CTO=no|TIM=no" in text


def test_the_calibration_sample_never_enters_the_audit(world, con):
    pre = make_preassignment(con, world)
    calibration = [p.canonical_id for p in world.eligible()[:40]]
    con.execute("INSERT INTO blind_sample(run_id,threshold) VALUES (%s,0.8)", (world.run_id,))
    for i, cid in enumerate(calibration, 1):
        con.execute("INSERT INTO blind_sample_paper(run_id,canonical_id,position) VALUES (%s,%s,%s)", (world.run_id, cid, i))
    audit_id, design, _ = audit.create_sample(con, world.run_id, pre, n_total=60, min_per_stratum=4, min_excluded=8, seed=9)
    chosen = {cid for (cid,) in con.execute("SELECT canonical_id FROM audit_sample_paper WHERE audit_id=%s", (audit_id,))}
    assert not chosen & set(calibration) and design["excluded_calibration_papers"] == 40
    other, other_design, _ = audit.create_sample(con, world.run_id, pre, n_total=60, min_per_stratum=4, min_excluded=8, seed=9, exclude_calibration=False)
    assert other != audit_id and other_design["excluded_calibration_papers"] == 0


def test_metrics_for_sets_calibration_and_screening_on_hand_made_cases():
    rows = [{"stratum": "s", "weight": 2.0, "pred": {"CTO": "yes", "TIM": "no"}, "confidence": {"CTO": 0.9, "TIM": 0.9}, "gold": {"CTO": "yes", "TIM": "no"}},
            {"stratum": "s", "weight": 2.0, "pred": {"CTO": "yes", "TIM": "yes"}, "confidence": {"CTO": 0.9, "TIM": 0.9}, "gold": {"CTO": "yes", "TIM": "no"}},
            {"stratum": "s", "weight": 1.0, "pred": {"CTO": "no", "TIM": "no"}, "confidence": {"CTO": 0.5, "TIM": 0.5}, "gold": {"CTO": "no", "TIM": "no"}},
            {"stratum": "s", "weight": 1.0, "pred": {"CTO": "uncertain", "TIM": "no"}, "confidence": {"CTO": None, "TIM": None}, "gold": {"CTO": "yes", "TIM": "no"}}]
    m = audit.multilabel_metrics(rows, draws=20)
    # included: the first three; Jaccard = (1 and 1/2 and 1) weighted 2,2,1 over 5
    assert m["mean_jaccard"] == pytest.approx((2 * 1 + 2 * 0.5 + 1 * 1) / 5) and m["included"] == 3 and m["abstention_rate_weighted"] == pytest.approx(1 / 6)
    assert m["exact_set_agreement"] == pytest.approx(3 / 5)
    assert audit.calibration_error([{"confidence": {"CTO": 0.9, "TIM": 0.9}, "pred": {"CTO": "yes", "TIM": "yes"}, "gold": {"CTO": "yes", "TIM": "yes"}, "weight": 1}])["ece"] == pytest.approx(0.1)
    assert audit.calibration_error([{"confidence": {"CTO": None, "TIM": None}, "pred": {}, "gold": {}, "weight": 1}]) is None
