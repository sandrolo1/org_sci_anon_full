"""Phase 4: stratified human audit of the pre-assignment.

Design. Papers are stratified by the pair of pre-assignment labels (CTO yes/no/uncertain x TIM yes/no/uncertain) and, when the pre-assignment gave
confidences, by a confidence band (high from 0.8, else low), as the proposal asks (predicted membership and confidence). Every non-empty stratum is
sampled (a minimum per stratum), and the strata of papers the pre-assignment EXCLUDED from both communities get a larger mandatory minimum, because
that is where the false negatives hide. Papers of the calibration sample (the blind sample used to develop the codebook) never enter the audit. Inside a stratum the draw is simple random, so each paper's
inclusion probability is n_h / N_h and the estimates below are design-based.

Two people rate each sampled paper blind (they never see the model label, the stratum or the other person's rating), a third person
settles the disagreements, and the model is compared with the final human labels. Estimates carry intervals; with few papers per
stratum the intervals are wide and the report says so instead of hiding it.
"""
import json
import logging
import random
from collections import Counter

import numpy as np

from .core import stable_id
from .stats import cohen_kappa, krippendorff_alpha_nominal, wilson

LOG = logging.getLogger("research0")
LABELS = ("yes", "no", "uncertain")
COMMUNITIES = ("CTO", "TIM")
EXCLUDED = "CTO=no|TIM=no"      # prefix: with confidence bands the stratum is CTO=no|TIM=no|conf=high / conf=low
HIGH_CONFIDENCE = 0.8
BOOTSTRAP = 2000


def stratum_of(cto, tim, band=None):
    return f"CTO={cto}|TIM={tim}" + (f"|conf={band}" if band else "")


def labels_of(stratum):
    """(CTO label, TIM label) of a stratum name, whatever follows."""
    parts = dict(p.split("=") for p in stratum.split("|"))
    return parts["CTO"], parts["TIM"]


def strata(con, preassignment_run_id, by_confidence=True, exclude=()):
    rows = con.execute(
        "SELECT canonical_id,max(label) FILTER (WHERE community='CTO'),max(label) FILTER (WHERE community='TIM'),min(confidence) FROM community_preassignment "
        "WHERE preassignment_run_id=%s GROUP BY canonical_id HAVING count(*)=2 ORDER BY canonical_id", (preassignment_run_id,)).fetchall()
    skip = set(exclude)
    out = {}
    for cid, cto, tim, confidence in rows:
        if cid in skip:
            continue
        band = None if not by_confidence else ("na" if confidence is None else ("high" if confidence >= HIGH_CONFIDENCE else "low"))
        out.setdefault(stratum_of(cto, tim, band), []).append(cid)
    return out


def allocate(populations, n_total, min_per_stratum, min_excluded):
    """Sample size per stratum: the minimums first (the excluded stratum has its own), the rest in proportion to what is left to draw."""
    base = {h: min(N, min_excluded if h.startswith(EXCLUDED) else min_per_stratum) for h, N in populations.items()}
    need = sum(base.values())
    if need > n_total:
        raise ValueError(f"The minimums need {need} papers but the sample size is {n_total}; raise the size or lower the minimums")
    room = {h: populations[h] - base[h] for h in populations}
    spare = n_total - need
    total_room = sum(room.values())
    if spare and total_room:
        spare = min(spare, total_room)
        shares = {h: spare * room[h] / total_room for h in room}
        extra = {h: int(shares[h]) for h in shares}
        for h in sorted(shares, key=lambda h: (-(shares[h] - extra[h]), h))[:spare - sum(extra.values())]:
            extra[h] += 1
        base = {h: base[h] + min(extra[h], room[h]) for h in base}
    return base


def create_sample(con, run_id, preassignment_run_id, n_total=400, min_per_stratum=15, min_excluded=40, seed=20261010, by_confidence=True, exclude_calibration=True):
    calibration = [cid for (cid,) in con.execute("SELECT canonical_id FROM blind_sample_paper WHERE run_id=%s", (run_id,))] if exclude_calibration else []
    populations = {h: ids for h, ids in strata(con, preassignment_run_id, by_confidence, calibration).items()}
    if not populations:
        raise ValueError("The pre-assignment run has no complete papers (CTO and TIM) to sample from")
    sizes = allocate({h: len(ids) for h, ids in populations.items()}, n_total, min_per_stratum, min_excluded)
    design = {"version": "audit-design-v2", "n_total": n_total, "min_per_stratum": min_per_stratum, "min_excluded": min_excluded, "by_confidence": by_confidence,
              "excluded_calibration_papers": len(calibration),
              "population": {h: len(ids) for h, ids in sorted(populations.items())}, "sample": dict(sorted(sizes.items()))}
    audit_id = stable_id("audit", run_id, preassignment_run_id, seed, json.dumps(design, sort_keys=True))
    if con.execute("SELECT 1 FROM audit_sample WHERE audit_id=%s", (audit_id,)).fetchone():
        return audit_id, design, True
    rng = random.Random(seed)
    chosen = [(h, cid) for h in sorted(populations) for cid in rng.sample(sorted(populations[h]), sizes[h])]
    rng.shuffle(chosen)  # position carries no information about the stratum
    with con.transaction():
        con.execute("INSERT INTO audit_sample(audit_id,run_id,preassignment_run_id,seed,design) VALUES (%s,%s,%s,%s,%s::jsonb)",
                    (audit_id, run_id, preassignment_run_id, seed, json.dumps(design)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO audit_sample_paper(audit_id,canonical_id,stratum,population_size,sample_size,position) VALUES (%s,%s,%s,%s,%s,%s)",
                            [(audit_id, cid, h, len(populations[h]), sizes[h], i) for i, (h, cid) in enumerate(chosen, 1)])
    return audit_id, design, False


# --- People -------------------------------------------------------------------------------------------------------------

def blind_papers(con, audit_id, validator):
    """Text only: no model label, no stratum, no other rater."""
    rows = con.execute(
        "SELECT s.canonical_id,s.position,c.title,c.publication_year,r.raw_fields->>'Abstract',r.raw_fields->>'Author Keywords',r.raw_fields->>'Index Keywords' "
        "FROM audit_sample_paper s JOIN canonical_paper c USING (canonical_id) JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE s.audit_id=%s ORDER BY s.position", (audit_id,)).fetchall()
    mine = {}
    for cid, community, label, note in con.execute(
            "SELECT DISTINCT ON (canonical_id,community) canonical_id,community,label,note FROM audit_rating WHERE audit_id=%s AND validator=%s "
            "ORDER BY canonical_id,community,rated_at DESC,rating_id DESC", (audit_id, validator)):
        mine.setdefault(str(cid), {})[community] = {"label": label, "note": note}
    return [{"id": str(cid), "position": pos, "title": t, "year": y, "abstract": a, "author_keywords": ak, "index_keywords": ik, "ratings": mine.get(str(cid), {})}
            for cid, pos, t, y, a, ak, ik in rows]


def rate(con, audit_id, canonical_id, community, validator, label, note=""):
    validator = (validator or "").strip()
    if not validator or len(validator) > 80 or community not in COMMUNITIES or label not in LABELS or len(note or "") > 2000:
        raise ValueError("Validator name, community (CTO/TIM) and label (yes/no/uncertain) are required")
    if not con.execute("SELECT 1 FROM audit_sample_paper WHERE audit_id=%s AND canonical_id=%s", (audit_id, canonical_id)).fetchone():
        raise ValueError("Paper is not in this audit sample")
    con.execute("INSERT INTO audit_rating(audit_id,canonical_id,community,validator,label,note) VALUES (%s,%s,%s,%s,%s,%s)",
                (audit_id, canonical_id, community, validator, label, note or ""))


def current_ratings(con, audit_id):
    grouped = {}
    for validator, cid, community, label in con.execute(
            "SELECT DISTINCT ON (validator,canonical_id,community) validator,canonical_id,community,label FROM audit_rating WHERE audit_id=%s "
            "ORDER BY validator,canonical_id,community,rated_at DESC,rating_id DESC", (audit_id,)):
        grouped.setdefault(validator, {})[(cid, community)] = label
    return grouped


def status(con, audit_id, validator=None):
    row = con.execute("SELECT count(*) FROM audit_sample_paper WHERE audit_id=%s", (audit_id,)).fetchone()
    if not row[0]:
        return {"created": False}
    size = row[0]
    grouped = current_ratings(con, audit_id)
    complete = sorted(v for v, r in grouped.items() if len(r) == 2 * size)
    return {"created": True, "size": size, "reviewers": {v: len(r) // 2 for v, r in grouped.items()}, "completed_reviewers": complete,
            "my_completed": len(grouped.get(validator, {})) // 2 if validator else None}


def _two_complete(con, audit_id):
    info = status(con, audit_id)
    if not info["created"] or len(info["completed_reviewers"]) != 2:
        raise ValueError("Exactly two annotators must complete every sampled paper (both communities)")
    grouped = current_ratings(con, audit_id)
    a, b = info["completed_reviewers"]
    return a, b, grouped[a], grouped[b]


def disagreements(con, audit_id):
    a, b, ra, rb = _two_complete(con, audit_id)
    return [(cid, community) for (cid, community) in sorted(ra) if ra[(cid, community)] != rb[(cid, community)]]


def adjudication_items(con, audit_id):
    """Disagreements with the text and both ratings, never the model label or the stratum."""
    a, b, ra, rb = _two_complete(con, audit_id)
    latest = {(cid, community): label for cid, community, label in con.execute(
        "SELECT DISTINCT ON (canonical_id,community) canonical_id,community,label FROM audit_adjudication WHERE audit_id=%s "
        "ORDER BY canonical_id,community,decided_at DESC,adjudication_id DESC", (audit_id,))}
    text = {cid: (t, ab) for cid, t, ab in con.execute(
        "SELECT s.canonical_id,c.title,r.raw_fields->>'Abstract' FROM audit_sample_paper s JOIN canonical_paper c USING (canonical_id) "
        "JOIN export_record r ON r.record_id=c.representative_record_id WHERE s.audit_id=%s", (audit_id,))}
    return [{"id": str(cid), "community": community, "title": text[cid][0], "abstract": text[cid][1], "ratings": {a: ra[(cid, community)], b: rb[(cid, community)]},
             "final": latest.get((cid, community))} for cid, community in disagreements(con, audit_id)]


def adjudicate(con, audit_id, canonical_id, community, adjudicator, label, note=""):
    adjudicator = (adjudicator or "").strip()
    if not adjudicator or len(adjudicator) > 80 or label not in LABELS or community not in COMMUNITIES:
        raise ValueError("Adjudicator name, community and label are required")
    if (canonical_id, community) not in set(disagreements(con, audit_id)):
        raise ValueError("The two annotators agree on this paper and community")
    con.execute("INSERT INTO audit_adjudication(audit_id,canonical_id,community,adjudicator,label,note) VALUES (%s,%s,%s,%s,%s,%s)",
                (audit_id, canonical_id, community, adjudicator, label, note or ""))


def final_labels(con, audit_id):
    a, b, ra, rb = _two_complete(con, audit_id)
    settled = {(cid, community): label for cid, community, label in con.execute(
        "SELECT DISTINCT ON (canonical_id,community) canonical_id,community,label FROM audit_adjudication WHERE audit_id=%s "
        "ORDER BY canonical_id,community,decided_at DESC,adjudication_id DESC", (audit_id,))}
    out, pending = {}, []
    for key in ra:
        if ra[key] == rb[key]:
            out[key] = ra[key]
        elif key in settled:
            out[key] = settled[key]
        else:
            pending.append(key)
    return out, pending


def agreement(con, audit_id, expected=0.8):
    a, b, ra, rb = _two_complete(con, audit_id)
    result = {"annotators": [a, b], "threshold": expected}
    for community in COMMUNITIES:
        keys = [k for k in sorted(ra) if k[1] == community]
        three = cohen_kappa([(ra[k], rb[k]) for k in keys])
        binary = cohen_kappa([(ra[k] == "yes", rb[k] == "yes") for k in keys])
        alpha = krippendorff_alpha_nominal([[ra[k], rb[k]] for k in keys])
        result[community] = {"three_class": three, "yes_vs_rest": binary, "alpha_three_class": alpha["alpha"],
                             "meets_threshold": three["kappa"] is not None and three["kappa"] >= expected}
    result["disagreements"] = len(disagreements(con, audit_id))
    result["pending_adjudication"] = len(final_labels(con, audit_id)[1])
    return result


# --- Estimation ---------------------------------------------------------------------------------------------------------

def stratified_estimates(rows, seed=1, draws=BOOTSTRAP):
    """rows: one dict per stratum {stratum, model, N, n, yes, uncertain} for ONE community, where `model` is the pre-assignment label of
    every paper in the stratum, N its size, n the papers audited, yes/uncertain how many final human labels are yes / uncertain.

    Human 'uncertain' is reported twice: strict (counts as not-yes) and lenient (counts as yes). Pre-assignment 'uncertain' counts as
    not-yes in the comparison and is reported separately as a rate, because such papers go to people anyway."""
    def counts(p_of):
        tp = sum(r["N"] * p_of(r) for r in rows if r["model"] == "yes")
        fp = sum(r["N"] * (1 - p_of(r)) for r in rows if r["model"] == "yes")
        fn = sum(r["N"] * p_of(r) for r in rows if r["model"] != "yes")
        tn = sum(r["N"] * (1 - p_of(r)) for r in rows if r["model"] != "yes")
        return tp, fp, fn, tn

    def scores(tp, fp, fn, tn):
        f_yes = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        f_no = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else None
        valid = [f for f in (f_yes, f_no) if f is not None]
        return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None, "f1_yes": f_yes,
                "macro_f1": sum(valid) / len(valid) if valid else None, "estimated_members": tp + fn}

    out = {"strata": [], "design_warnings": []}
    for r in rows:
        r.setdefault("uncertain", 0)
        p, low, high = wilson(r["yes"], r["n"])
        out["strata"].append({**r, "human_yes_rate": p, "ci95": [low, high], "estimated_yes_in_stratum": None if p is None else r["N"] * p})
        if r["n"] == 0:
            out["design_warnings"].append(f"stratum {r['stratum']} ({r['N']} papers) has no audited paper: it is missing from the estimates")
        elif r["n"] < 10 and r["n"] < r["N"]:
            out["design_warnings"].append(f"stratum {r['stratum']} has only {r['n']} audited papers: its interval is very wide")
    audited = [r for r in rows if r["n"] > 0]
    strict = counts(lambda r: r["yes"] / r["n"] if r["n"] else 0.0)
    lenient = counts(lambda r: (r["yes"] + r["uncertain"]) / r["n"] if r["n"] else 0.0)
    out["strict"], out["lenient"] = scores(*strict), scores(*lenient)
    excluded = [r for r in audited if r["model"] == "no"]
    n_ex, k_ex = sum(r["n"] for r in excluded), sum(r["yes"] for r in excluded)
    p, low, high = wilson(k_ex, n_ex)
    population_ex = sum(r["N"] for r in excluded)
    out["false_negatives_among_no"] = {"audited": n_ex, "human_yes": k_ex, "rate": p, "ci95": [low, high], "population": population_ex,
                                       "estimated_missed": None if p is None else population_ex * p,
                                       "estimated_missed_ci95": None if p is None else [population_ex * low, population_ex * high]}
    rng = np.random.default_rng(seed)
    samples = {"macro_f1": [], "recall": [], "precision": []}
    for _ in range(draws):
        drawn = {}
        for r in audited:
            drawn[r["stratum"]] = r["yes"] if r["n"] >= r["N"] else rng.binomial(r["n"], r["yes"] / r["n"])
        drawn_scores = scores(*counts(lambda r: drawn[r["stratum"]] / r["n"] if r["n"] else 0.0))
        for key in samples:
            if drawn_scores[key] is not None:
                samples[key].append(drawn_scores[key])
    interval = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None
    out["strict"]["macro_f1_ci95"] = interval(samples["macro_f1"])
    out["strict"]["recall_ci95"] = interval(samples["recall"])
    out["strict"]["precision_ci95"] = interval(samples["precision"])
    return out


def _audit_rows(con, audit_id, final):
    """One dict per audited paper: stratum, weight, pre-assignment label and confidence per community, final human label per community."""
    pre_run = con.execute("SELECT preassignment_run_id FROM audit_sample WHERE audit_id=%s", (audit_id,)).fetchone()[0]
    pre = {(cid, community): (label, confidence) for cid, community, label, confidence in con.execute(
        "SELECT canonical_id,community,label,confidence FROM community_preassignment WHERE preassignment_run_id=%s", (pre_run,))}
    rows = []
    for cid, stratum, population, sample in con.execute("SELECT canonical_id,stratum,population_size,sample_size FROM audit_sample_paper WHERE audit_id=%s", (audit_id,)).fetchall():
        rows.append({"id": cid, "stratum": stratum, "weight": population / sample, "N": population,
                     "pred": {c: pre[(cid, c)][0] for c in COMMUNITIES}, "confidence": {c: pre[(cid, c)][1] for c in COMMUNITIES},
                     "gold": {c: final[(cid, c)] for c in COMMUNITIES}})
    return rows


def screening_estimates(rows, populations, seed=1):
    """Screening recall of the candidate selection: a paper is *selected* when the pre-assignment says yes or uncertain for either community
    (uncertain goes on to the full text, so it is not lost); a *member* is a paper the final human labels call yes for either community. Estimated by
    stratum like every other figure, so the excluded strata count."""
    per = {}
    for stratum, population in populations.items():
        labels = list(labels_of(stratum))
        per[stratum] = {"stratum": stratum, "model": "yes" if any(l in ("yes", "uncertain") for l in labels) else "no", "N": population, "n": 0, "yes": 0, "uncertain": 0}
    for r in rows:
        entry = per[r["stratum"]]
        entry["n"] += 1
        entry["yes"] += any(r["gold"][c] == "yes" for c in COMMUNITIES)
        entry["uncertain"] += (not any(r["gold"][c] == "yes" for c in COMMUNITIES)) and any(r["gold"][c] == "uncertain" for c in COMMUNITIES)
    return stratified_estimates([per[k] for k in sorted(per)], seed)


def _set_metrics(items):
    """items: [(weight, predicted set, gold set)] -> exact agreement, mean Jaccard, micro-F1, macro-F1 over CTO and TIM (weighted)."""
    total = sum(w for w, _, _ in items)
    if not total:
        return None
    jaccard = sum(w * (1.0 if not (p | g) else len(p & g) / len(p | g)) for w, p, g in items) / total
    exact = sum(w for w, p, g in items if p == g) / total
    tp = sum(w * len(p & g) for w, p, g in items)
    fp = sum(w * len(p - g) for w, p, g in items)
    fn = sum(w * len(g - p) for w, p, g in items)
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if 2 * a + b + c else None
    per = []
    for c in COMMUNITIES:
        v = f1(sum(w for w, p, g in items if c in p and c in g), sum(w for w, p, g in items if c in p and c not in g), sum(w for w, p, g in items if c not in p and c in g))
        if v is not None:
            per.append(v)
    return {"exact_set_agreement": exact, "mean_jaccard": jaccard, "micro_f1": f1(tp, fp, fn), "macro_f1": sum(per) / len(per) if per else None}


def multilabel_metrics(rows, seed=1, draws=500):
    """Set of memberships per paper (CTO, TIM, both, neither). Papers on which the pipeline abstained (uncertain for either community) or the final human
    label is uncertain are left out of the set metrics and counted as abstentions; sampling weights are N_h / n_h of the stratum."""
    classes = {}
    included, abstained = [], 0.0
    for r in rows:
        pred = {c for c in COMMUNITIES if r["pred"][c] == "yes"}
        gold = {c for c in COMMUNITIES if r["gold"][c] == "yes"}
        gold_unsure = any(r["gold"][c] == "uncertain" for c in COMMUNITIES)
        pred_unsure = any(r["pred"][c] == "uncertain" for c in COMMUNITIES)
        name = "uncertain" if gold_unsure else ("both" if len(gold) == 2 else ("neither" if not gold else next(iter(gold))))
        classes[name] = classes.get(name, 0) + 1
        if pred_unsure or gold_unsure:
            abstained += r["weight"]
        else:
            included.append((r["stratum"], r["weight"], pred, gold))
    total_weight = sum(r["weight"] for r in rows)
    point = _set_metrics([(w, p, g) for _, w, p, g in included])
    out = {"papers": len(rows), "included": len(included), "abstention_rate_weighted": abstained / total_weight if total_weight else None, "gold_classes_audited": classes, **(point or {})}
    if point:
        rng = np.random.default_rng(seed)
        by_stratum = {}
        for item in included:
            by_stratum.setdefault(item[0], []).append(item)
        draws_out = {k: [] for k in ("mean_jaccard", "micro_f1", "macro_f1")}
        for _ in range(draws):
            sample = []
            for items in by_stratum.values():
                picks = rng.integers(0, len(items), len(items))
                sample.extend((items[i][1], items[i][2], items[i][3]) for i in picks)
            m = _set_metrics(sample)
            for k in draws_out:
                if m and m[k] is not None:
                    draws_out[k].append(m[k])
        for k, v in draws_out.items():
            out[k + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    return out


def calibration_error(rows, bins=10):
    """Expected calibration error of the pipeline's confidences against the final human labels (weighted by sampling weights). None without confidences."""
    points = [(r["confidence"][c], r["pred"][c] == r["gold"][c], r["weight"]) for r in rows for c in COMMUNITIES if r["confidence"][c] is not None]
    if not points:
        return None
    total = sum(w for _, _, w in points)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        inside = [(c, ok, w) for c, ok, w in points if low <= c < high or (b == bins - 1 and c == 1.0)]
        weight = sum(w for _, _, w in inside)
        if weight:
            error += weight / total * abs(sum(w * ok for _, ok, w in inside) / weight - sum(w * c for c, _, w in inside) / weight)
    return {"ece": error, "points": len(points), "bins": bins}


def estimates(con, audit_id, seed=1):
    final, pending = final_labels(con, audit_id)
    if pending:
        raise ValueError(f"{len(pending)} disagreements are still waiting for an adjudicator")
    info = con.execute("SELECT design FROM audit_sample WHERE audit_id=%s", (audit_id,)).fetchone()[0]
    members = con.execute("SELECT canonical_id,stratum,population_size FROM audit_sample_paper WHERE audit_id=%s", (audit_id,)).fetchall()
    result = {"design": info, "uncertain_rate_of_preassignment": {}}
    rows = _audit_rows(con, audit_id, final)
    result["screening"] = screening_estimates(rows, info["population"], seed)
    result["multilabel"] = multilabel_metrics(rows, seed)
    result["calibration"] = calibration_error(rows)
    for community in COMMUNITIES:
        per = {}
        for cid, stratum, population in members:
            entry = per.setdefault(stratum, {"stratum": stratum, "model": labels_of(stratum)[0 if community == "CTO" else 1], "N": population, "n": 0, "yes": 0, "uncertain": 0})
            entry["n"] += 1
            entry["yes"] += final[(cid, community)] == "yes"
            entry["uncertain"] += final[(cid, community)] == "uncertain"
        for stratum, population in info["population"].items():
            per.setdefault(stratum, {"stratum": stratum, "model": labels_of(stratum)[0 if community == "CTO" else 1], "N": population, "n": 0, "yes": 0, "uncertain": 0})
        result[community] = stratified_estimates([per[k] for k in sorted(per)], seed)
        total = sum(info["population"].values())
        result["uncertain_rate_of_preassignment"][community] = sum(
            n for s, n in info["population"].items() if labels_of(s)[0 if community == "CTO" else 1] == "uncertain") / total
    return result


def write_report(estimate, agreement_result, path):
    lines = ["# Audit of the pre-assignment", "",
             "Estimates are design-based from a stratified random sample; read the intervals, not only the point values.", ""]
    if agreement_result:
        lines += ["## Agreement between the two annotators", "", "| Community | kappa (yes/no/uncertain) | kappa (yes vs rest) | alpha | observed agreement |", "|---|---:|---:|---:|---:|"]
        for c in COMMUNITIES:
            a = agreement_result[c]
            fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
            lines.append(f"| {c} | {fmt(a['three_class']['kappa'])} | {fmt(a['yes_vs_rest']['kappa'])} | {fmt(a['alpha_three_class'])} | {a['three_class']['agreement']:.3f} |")
        lines.append("")
    for c in COMMUNITIES:
        e = estimate[c]
        fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
        fn = e["false_negatives_among_no"]
        lines += [f"## {c}", "", f"- Macro-F1 (strict): {fmt(e['strict']['macro_f1'])}, 95% interval {e['strict']['macro_f1_ci95'] and [round(x, 3) for x in e['strict']['macro_f1_ci95']]}",
                  f"- Precision {fmt(e['strict']['precision'])}, recall {fmt(e['strict']['recall'])}; with human 'uncertain' counted as yes: recall {fmt(e['lenient']['recall'])}",
                  f"- Papers labelled 'no': {fn['audited']} audited, {fn['human_yes']} are human 'yes' (rate {fmt(fn['rate'])}, 95% interval {fn['ci95'] and [round(x, 3) for x in fn['ci95']]}); "
                  f"estimated missed members: {fn['estimated_missed'] and round(fn['estimated_missed'])}", ""]
        lines += ["| Stratum | Population | Audited | Human yes | Human uncertain | Yes rate | 95% interval |", "|---|---:|---:|---:|---:|---:|---|"]
        for s in e["strata"]:
            lines.append(f"| {s['stratum']} | {s['N']} | {s['n']} | {s['yes']} | {s['uncertain']} | {fmt(s['human_yes_rate'])} | {s['ci95'] if s['n'] else 'n/a'} |")
        lines += [""] + [f"- Warning: {w}" for w in e["design_warnings"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- Reporting aids required by the proposal ----------------------------------------------------------------------------------------------

def confusion_matrices(con, audit_id):
    """Pre-assignment label (rows) against the final human label (columns) for each community: the audited counts and the sampling-weighted
    estimates of the whole population. Numerators and denominators are published, not only percentages."""
    final, pending = final_labels(con, audit_id)
    if pending:
        raise ValueError(f"{len(pending)} disagreements are still waiting for an adjudicator")
    rows = _audit_rows(con, audit_id, final)
    out = {}
    for community in COMMUNITIES:
        counts, weighted = {}, {}
        for r in rows:
            key = (r["pred"][community], r["gold"][community])
            counts[key] = counts.get(key, 0) + 1
            weighted[key] = weighted.get(key, 0.0) + r["weight"]
        out[community] = {"audited": {f"{a}->{b}": n for (a, b), n in sorted(counts.items())},
                          "estimated_population": {f"{a}->{b}": round(w, 1) for (a, b), w in sorted(weighted.items())},
                          "audited_papers": len(rows)}
    return out


def breakdowns(con, audit_id):
    """Human 'yes' rate by publication period and by source among the audited papers (reported per strand, period and source, as the proposal asks).
    Descriptive: the strata of the design are not these groups, so no population estimate is attached."""
    final, pending = final_labels(con, audit_id)
    if pending:
        raise ValueError("Adjudication is not finished")
    info = {cid: (year, source) for cid, year, source in con.execute(
        "SELECT p.canonical_id,c.publication_year,c.source_title FROM audit_sample_paper p JOIN canonical_paper c USING (canonical_id) WHERE p.audit_id=%s", (audit_id,))}
    result = {}
    for community in COMMUNITIES:
        by_period, by_source = {}, {}
        for cid, (year, source) in info.items():
            hit = final[(cid, community)] == "yes"
            period = "unknown" if year is None else f"{year // 3 * 3}-{year // 3 * 3 + 2}"
            for table, key in ((by_period, period), (by_source, source or "unknown")):
                n, k = table.get(key, (0, 0))
                table[key] = (n + 1, k + hit)
        result[community] = {"by_period": {k: {"audited": n, "human_yes": y} for k, (n, y) in sorted(by_period.items())},
                             "by_source": {k: {"audited": n, "human_yes": y} for k, (n, y) in sorted(by_source.items())}}
    return result


def extraction_validation_sample(con, run_id, preassignment_run_id, seed=20261010, fraction=0.10, minimum=100):
    """Papers for the in-depth check of the extractions: 10% of each community's members (at least 100 where possible, otherwise all of them).
    A paper shared by both communities is one unit, drawn once. Members are the papers the pre-assignment calls yes for the community and that have a parsed full text."""
    rng = random.Random(seed)
    members = {c: sorted(cid for (cid,) in con.execute(
        "SELECT DISTINCT p.canonical_id FROM community_preassignment p JOIN paper_version v ON v.canonical_id=p.canonical_id AND v.status='parsed' "
        "WHERE p.preassignment_run_id=%s AND p.community=%s AND p.label='yes'", (preassignment_run_id, c))) for c in COMMUNITIES}
    chosen, design = set(), {}
    for c in COMMUNITIES:
        pool = members[c]
        take = len(pool) if len(pool) <= minimum else max(minimum, round(len(pool) * fraction))
        already = [x for x in pool if x in chosen]
        need = max(0, take - len(already))            # shared papers drawn for the other community count here too
        fresh = rng.sample([x for x in pool if x not in chosen], min(need, len([x for x in pool if x not in chosen])))
        chosen.update(fresh)
        design[c] = {"members": len(pool), "target": take, "drawn_for_this_community": len(fresh), "complete": take >= len(pool),
                     "note": "the whole community is validated; the lower statistical precision is declared" if take >= len(pool) else None}
    return sorted(chosen), design
