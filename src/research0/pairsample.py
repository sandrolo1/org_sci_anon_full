"""Pair sampling for the jingle/jangle gates (Appendix B, "Pair sampling").

Two samples are drawn from the candidate pairs of a signal run and are never mixed:
  * a probabilistic sample, stratified by the triage hint and by whether the two labels are the same; every pair has a known inclusion probability
    (sample_size / population_size of its stratum), so prevalence among the candidate pairs can be estimated;
  * an enriched challenge set to stress the system. For a signals-v4 run, the kinds of the proposal v4: same-label pairs whose meanings diverge most,
    different-label pairs whose meanings are closest, close measures under distinct definitions, context-shifted pairs, pairs near the S_def thresholds,
    hard negatives (one label, one meaning) and cross-community pairs. For an earlier run (no S_def): the closest operationalisations, pairs near the
    S_sem-op thresholds, homonyms and cross-community pairs. Percentages on it are NOT prevalence unless reweighted, and the report says so.
The proposal asks for at least 50 adjudicated cases per positive conceptual class (conceptual jingle, conceptual jangle); the report states how many there are.
Decisions are read with the current pair codebook (pairs.CODEBOOK); those of the earlier one are history.
"""
import json
import random
from collections import defaultdict

from . import pairs
from .signals import HINTS, version_number
from .core import stable_id
from .stats import wilson


def _stratum(hint, name_sim, name_hi):
    return f"{hint}|{'same_label' if name_sim >= name_hi else 'different_label'}"


def equal_allocation(populations, n):
    """Split n over the strata as evenly as the populations allow (a small stratum is taken whole and its share goes to the others).
    The sample is still probabilistic: every pair has a known inclusion probability, sample_size / population_size of its stratum."""
    left, out = n, dict.fromkeys(populations, 0)
    open_ = sorted(populations)
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for h in list(open_):
            take = min(share, populations[h] - out[h], left)
            out[h] += take
            left -= take
            if out[h] >= populations[h]:
                open_.remove(h)
            if left == 0:
                break
    return out


def latest(con, signal_run_id):
    row = con.execute("SELECT pair_sample_id FROM pair_sample WHERE signal_run_id=%s ORDER BY created_at DESC, pair_sample_id DESC LIMIT 1", (signal_run_id,)).fetchone()
    return row[0] if row else None


def create(con, signal_run_id, n_probabilistic=200, n_challenge=100, seed=20261010, min_per_stratum=5, allocation="equal"):
    """allocation "equal": the same number of pairs in each stratum (jingle/jangle candidates, "none" pairs with the same and different labels, unresolved ones),
    so that the blind validation holds both the positives and the pairs that are just two different constructs. "proportional": by population size,
    which on real data is mostly "none" and yields few positives."""
    if allocation not in ("equal", "proportional"):
        raise ValueError("allocation is 'equal' or 'proportional'")
    params, version = con.execute("SELECT params,version FROM signal_run WHERE signal_run_id=%s", (signal_run_id,)).fetchone()
    name_hi, s_hi, s_lo = params["name_hi"], params["s_hi"], params["s_lo"]
    rows = con.execute(
        "SELECT p.pair_id,p.hint,p.name_sim,p.s_sem_op,a.canonical_id,b.canonical_id,p.s_def,p.s_op FROM candidate_pair p JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
        "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b WHERE p.signal_run_id=%s ORDER BY p.pair_id", (signal_run_id,)).fetchall()
    v4 = version_number(version) >= 4
    if not rows:
        raise ValueError("The signal run has no candidate pairs")
    design = {"version": "pair-sample-v3" if v4 else "pair-sample-v2", "n_probabilistic": n_probabilistic, "n_challenge": n_challenge, "min_per_stratum": min_per_stratum, "allocation": allocation}
    sample_id = stable_id("pair-sample", signal_run_id, seed, json.dumps(design, sort_keys=True))
    if con.execute("SELECT 1 FROM pair_sample WHERE pair_sample_id=%s", (sample_id,)).fetchone():
        return sample_id, design, True
    rng = random.Random(seed)
    strata = defaultdict(list)
    for pair_id, hint, name_sim, *_ in rows:
        strata[_stratum(hint, name_sim, name_hi)].append(pair_id)
    if allocation == "equal":
        base = equal_allocation({h: len(ids) for h, ids in strata.items()}, n_probabilistic)
    else:
        base = {h: min(len(ids), min_per_stratum) for h, ids in strata.items()}
        if sum(base.values()) > n_probabilistic:
            raise ValueError(f"The minimums need {sum(base.values())} pairs but the probabilistic sample is {n_probabilistic}")
        spare, room = n_probabilistic - sum(base.values()), {h: len(ids) - base[h] for h, ids in strata.items()}
        total_room = sum(room.values())
        if spare and total_room:
            for h in sorted(room):
                base[h] += min(room[h], int(spare * room[h] / total_room))
    chosen = {}
    for h in sorted(strata):
        for pair_id in rng.sample(sorted(strata[h]), base[h]):
            chosen[pair_id] = ("probabilistic", h, len(strata[h]), base[h])
    # challenge set: four kinds in equal shares, never repeating a pair already in the probabilistic sample
    comm = defaultdict(set)
    for cid, community in con.execute("SELECT DISTINCT p.canonical_id,p.community FROM community_preassignment p WHERE p.label='yes' AND p.canonical_id IN "
                                      "(SELECT canonical_id FROM construct_node WHERE signal_run_id=%s)", (signal_run_id,)):
        comm[cid].add(community)
    free = [r for r in rows if r[0] not in chosen]
    cross = sorted([r for r in free if comm.get(r[4]) and comm.get(r[5]) and comm[r[4]] != comm[r[5]]], key=lambda r: str(r[0]))
    if v4:      # the challenge kinds of the proposal v4 (Appendix B, Pair sampling), on S_def and S_op
        defined = [r for r in free if r[6] is not None]
        same, other = [r for r in defined if r[2] >= name_hi], [r for r in defined if r[2] < name_hi]
        kinds = {
            "same_label_meaning_different": sorted(same, key=lambda r: (r[6], str(r[0]))),
            "different_label_meaning_similar": sorted(other, key=lambda r: (-r[6], str(r[0]))),
            "same_measure_different_definition": sorted([r for r in other if r[7] is not None and r[6] < s_hi], key=lambda r: (-(r[7] - r[6]), str(r[0]))),
            "context_shifted": sorted([r for r in free if r[1] == "context_shifted"], key=lambda r: str(r[0])),
            "near_threshold": sorted(defined, key=lambda r: (min(abs(r[6] - s_hi), abs(r[6] - s_lo)), str(r[0]))),
            "hard_negative": sorted([r for r in same if r[1] == "hard_negative"], key=lambda r: (-r[6], str(r[0]))),
            "cross_community": cross,
        }
    else:
        scored = [r for r in free if r[3] is not None]
        kinds = {
            "closest_operationalization": sorted(scored, key=lambda r: (-r[3], str(r[0]))),
            "near_threshold": sorted(scored, key=lambda r: (min(abs(r[3] - s_hi), abs(r[3] - s_lo)), str(r[0]))),
            "homonym": sorted([r for r in free if r[2] >= name_hi], key=lambda r: (r[3] if r[3] is not None else 2, str(r[0]))),
            "cross_community": cross,
        }
    share = max(1, n_challenge // len(kinds))
    for kind, ordered in kinds.items():
        taken = 0
        for r in ordered:
            if r[0] in chosen:
                continue
            chosen[r[0]] = ("challenge", kind, len(ordered), share)
            taken += 1
            if taken == share:
                break
    design["probabilistic_strata"] = {h: {"population": len(ids), "sample": base[h]} for h, ids in sorted(strata.items())}
    design["challenge_kinds"] = {k: share for k in kinds}
    with con.transaction():
        con.execute("INSERT INTO pair_sample(pair_sample_id,signal_run_id,seed,design) VALUES (%s,%s,%s,%s::jsonb)", (sample_id, signal_run_id, seed, json.dumps(design)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO pair_sample_item(pair_sample_id,pair_id,sample_kind,stratum,population_size,sample_size) VALUES (%s,%s,%s,%s,%s,%s)",
                            [(sample_id, pid, kind, stratum, population, size) for pid, (kind, stratum, population, size) in chosen.items()])
    return sample_id, design, False


CASCADE = "pair-sample-cascade-v1"
GROUPS = ("llm_promoted", "llm_missing", "llm_rejected", "not_targeted")
PASSED = ("llm_promoted", "llm_missing")       # what the cascade hands to people whole; the other two groups are its audit


def create_cascade(con, signal_run_id, targeted=None, n_audit=50, n_promoted=None, seed=20261014, max_missing_share=0.05):
    """The blind sample of the cascade (14 Oct 2026): signals target pairs, the model filters them, people decide. Every pair falls in one group:
      llm_promoted   targeted, the model proposes a class worth a person's time (proposals.PROMOTED_CLASSES): taken whole, or n_promoted at random;
      llm_missing    targeted, but the model's answer failed or was rejected: taken whole (no model excluded it);
      llm_rejected   targeted, the model says distinct or same meaning: n_audit at random;
      not_targeted   every other pair of the run (hard negatives, unresolved, none): n_audit at random.
    Strata are the model's class (promoted), the hint (missing, not targeted) or the label kind (rejected); each pair has a known inclusion probability,
    so report() estimates how many conceptual jingles and jangles lie in each group and the recall of the cascade. Without the audit groups the model
    would exclude pairs on its own (the internal project notes principle 3)."""
    from . import proposals
    targeted = tuple(targeted or proposals.TARGETED_HINTS)
    if n_audit < 1:
        raise ValueError("The audit of the rejected and untargeted pairs needs at least one pair per group (principle 3)")
    params, version = con.execute("SELECT params,version FROM signal_run WHERE signal_run_id=%s", (signal_run_id,)).fetchone()
    rows = con.execute("SELECT pair_id,hint,name_sim FROM candidate_pair WHERE signal_run_id=%s ORDER BY pair_id", (signal_run_id,)).fetchall()
    if not rows:
        raise ValueError("The signal run has no candidate pairs")
    if version_number(version) < 4:
        raise ValueError(f"The cascade works on signals-v4 runs only; this run is {version}")
    unknown = set(targeted) - set(HINTS)
    if unknown:
        raise ValueError("Unknown hints: " + ", ".join(sorted(unknown)))
    classes = proposals.latest_classes(con, signal_run_id)
    strata = defaultdict(list)
    for pair_id, hint, name_sim in rows:
        if hint not in targeted:
            strata[f"not_targeted|{hint}"].append(pair_id)
        elif pair_id not in classes:
            strata[f"llm_missing|{hint}"].append(pair_id)
        elif classes[pair_id] in proposals.PROMOTED_CLASSES:
            strata[f"llm_promoted|{classes[pair_id]}"].append(pair_id)
        else:
            strata[f"llm_rejected|{'same_label' if name_sim >= params['name_hi'] else 'different_label'}"].append(pair_id)
    group = lambda s: s.split("|", 1)[0]
    size = lambda g: sum(len(ids) for s, ids in strata.items() if group(s) == g)
    n_targeted = sum(size(g) for g in ("llm_promoted", "llm_missing", "llm_rejected"))
    if not n_targeted:
        raise ValueError("No pair of the run is in the targeted strata: " + ", ".join(targeted))
    if size("llm_missing") > max_missing_share * n_targeted:
        raise ValueError(f"{size('llm_missing')} of {n_targeted} targeted pairs have no model proposal (more than {max_missing_share:.0%}): run proposal-prepare and "
                         "proposal-collect on all of them first, or raise max_missing_share (those pairs go to people whole)")
    design = {"version": CASCADE, "targeted_hints": sorted(targeted), "prompt_version": proposals.PROMPT_VERSION, "promoted_classes": list(proposals.PROMOTED_CLASSES),
              "n_audit": n_audit, "n_promoted": n_promoted, "max_missing_share": max_missing_share}
    sample_id = stable_id("pair-sample", signal_run_id, seed, json.dumps(design, sort_keys=True), json.dumps(sorted((str(p), c) for p, c in classes.items())))
    if con.execute("SELECT 1 FROM pair_sample WHERE pair_sample_id=%s", (sample_id,)).fetchone():
        return sample_id, design, True
    rng = random.Random(seed)
    take = {}
    for g in GROUPS:
        populations = {s: len(ids) for s, ids in strata.items() if group(s) == g}
        if g == "llm_missing" or (g == "llm_promoted" and n_promoted is None):
            take.update(populations)
        else:
            take.update(equal_allocation(populations, n_promoted if g == "llm_promoted" else n_audit))
    chosen = {}
    for s in sorted(strata):
        for pair_id in rng.sample(sorted(strata[s]), take[s]):
            chosen[pair_id] = (s, len(strata[s]), take[s])
    design["strata"] = {s: {"population": len(ids), "sample": take[s]} for s, ids in sorted(strata.items())}
    design["groups"] = {g: {"population": size(g), "sample": sum(take[s] for s in strata if group(s) == g)} for g in GROUPS}
    with con.transaction():
        con.execute("INSERT INTO pair_sample(pair_sample_id,signal_run_id,seed,design) VALUES (%s,%s,%s,%s::jsonb)", (sample_id, signal_run_id, seed, json.dumps(design)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO pair_sample_item(pair_sample_id,pair_id,sample_kind,stratum,population_size,sample_size) VALUES (%s,%s,'probabilistic',%s,%s,%s)",
                            [(sample_id, pid, s, population, n) for pid, (s, population, n) in chosen.items()])
    return sample_id, design, False


REVIEW_SET = "pair-review-llm-classes-v1"


def create_review_set(con, signal_run_id, classes=pairs.POSITIVE):
    """Every pair of the run whose latest model proposal is one of `classes` (default: conceptual jingle and jangle), for people to look at before the
    validation design is chosen (user's request, 14 Oct 2026). Stored as an enriched set (sample_kind 'challenge', stratum 'llm_<class>'), so report()
    never reads it as prevalence; it is not the cascade's audit, and the people who review it are no longer blind to the model's class on these pairs."""
    from . import proposals
    if not set(classes) <= set(pairs.CLASSES):
        raise ValueError("Unknown class; classes are: " + ", ".join(pairs.CLASSES))
    chosen = {pid: cls for pid, cls in proposals.latest_classes(con, signal_run_id).items() if cls in classes}
    if not chosen:
        raise ValueError("No model proposal of these classes on this run: " + ", ".join(classes))
    population = {cls: sum(1 for c in chosen.values() if c == cls) for cls in classes}
    design = {"version": REVIEW_SET, "classes": list(classes), "prompt_version": proposals.PROMPT_VERSION, "pairs": population}
    sample_id = stable_id("pair-sample", signal_run_id, json.dumps(design, sort_keys=True), json.dumps(sorted(map(str, chosen))))
    if con.execute("SELECT 1 FROM pair_sample WHERE pair_sample_id=%s", (sample_id,)).fetchone():
        return sample_id, design, True
    with con.transaction():
        con.execute("INSERT INTO pair_sample(pair_sample_id,signal_run_id,seed,design) VALUES (%s,%s,0,%s::jsonb)", (sample_id, signal_run_id, json.dumps(design)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO pair_sample_item(pair_sample_id,pair_id,sample_kind,stratum,population_size,sample_size) VALUES (%s,%s,'challenge',%s,%s,%s)",
                            [(sample_id, pid, f"llm_{cls}", population[cls], population[cls]) for pid, cls in chosen.items()])
    return sample_id, design, False


def cascade_recall(strata):
    """Recall of the cascade for each positive class: the estimated positives in the groups handed to people whole over the estimated positives in all
    groups. The range combines the Wilson bounds of each stratum the conservative way (an unjudged stratum can hold every pair of it)."""
    out = {}
    for cls in pairs.POSITIVE:
        est = {"passed": [0.0, 0.0, 0.0], "audited": [0.0, 0.0, 0.0]}
        for stratum, entry in strata.items():
            side = est["passed" if stratum.split("|", 1)[0] in PASSED else "audited"]
            if entry["settled"]:
                p, lo, hi = wilson(entry["classes"].get(cls, 0), entry["settled"])
            else:
                p, lo, hi = 0.0, 0.0, 1.0
            for i, v in enumerate((p, lo, hi)):
                side[i] += entry["population"] * v
        (passed, passed_lo, passed_hi), (audited, audited_lo, audited_hi) = est["passed"], est["audited"]
        ratio = lambda a, b: round(a / (a + b), 3) if a + b else None
        out[cls] = {"recall": ratio(passed, audited), "range": [ratio(passed_lo, audited_hi), ratio(passed_hi, audited_lo)],
                    "estimated_in_passed": round(passed, 1), "estimated_in_audit_groups": round(audited, 1)}
    return out


def report(con, pair_sample_id):
    """Settled decisions on the sample. Prevalence among candidate pairs is estimated from the probabilistic sample only; the challenge set is counted apart."""
    signal_run, design = con.execute("SELECT signal_run_id,design FROM pair_sample WHERE pair_sample_id=%s", (pair_sample_id,)).fetchone()
    decided = {pid: pairs.state_of(per) for pid, per in pairs.current(con, signal_run).items()}
    items = con.execute("SELECT pair_id,sample_kind,stratum,population_size,sample_size FROM pair_sample_item WHERE pair_sample_id=%s", (pair_sample_id,)).fetchall()
    out = {"codebook": pairs.CODEBOOK, "probabilistic": {"pairs": 0, "settled": 0, "strata": {}}, "challenge": {"pairs": 0, "settled": 0, "by_class": {}}}
    for pid, kind, stratum, population, size in items:
        part = out[kind]
        part["pairs"] += 1
        state, decision = decided.get(pid, ("pending", None))
        part["settled"] += state == "settled"
        if kind == "challenge":
            if decision:
                part["by_class"][decision] = part["by_class"].get(decision, 0) + 1
        else:
            entry = part["strata"].setdefault(stratum, {"population": population, "sample": size, "settled": 0, "classes": {}})
            if decision:
                entry["settled"] += 1
                entry["classes"][decision] = entry["classes"].get(decision, 0) + 1
    estimate = {}
    for cls in pairs.POSITIVE:
        total, low, high = 0.0, 0.0, 0.0
        for entry in out["probabilistic"]["strata"].values():
            if entry["settled"]:
                p, lo, hi = wilson(entry["classes"].get(cls, 0), entry["settled"])
                total += entry["population"] * p
                low += entry["population"] * lo
                high += entry["population"] * hi
            else:
                high += entry["population"]       # unjudged strata can only widen the interval
        estimate[cls] = {"estimated_pairs": round(total, 1), "range": [round(low, 1), round(high, 1)]}
    out["estimated_candidate_pairs_by_class"] = estimate
    out["adjudicated_per_positive_class"] = {c: sum(1 for st, d in decided.values() if st == "settled" and d == c) for c in pairs.POSITIVE}
    out["warning"] = ("Percentages on the challenge set are not prevalence. " +
                      "; ".join(f"only {n} adjudicated {c.replace('_', ' ')} cases (the proposal asks for at least 50)" for c, n in out["adjudicated_per_positive_class"].items() if n < 50))
    if design.get("version") == CASCADE:
        out["cascade"] = {"recall_by_class": cascade_recall(out["probabilistic"]["strata"]), "groups": design["groups"],
                          "note": "Recall of signals + model together, from the audit samples of the pairs the model rejected and of the pairs never targeted. "
                                  "A conceptual jingle or jangle found there is a miss of the cascade; a few of them weigh as much as their stratum's population."}
    return out
