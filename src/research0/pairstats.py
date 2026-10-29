"""Statistics of a signal run for the "Construct pairs" tab: how many pairs, how available and how distributed the signals are, what the triage hint does,
and how the blind validation is going.

Everything here is descriptive. Two things are held back until enough people have finished, because they could anchor someone still deciding:
the comparison between the triage hint and the human decisions, and the agreement between annotators (`review.revealed`).
N_nom is reported but flagged as not used by the hint (signals-v3). For a signals-v4 run the histogram reads S_def (the meaning), the hint is a retrieval
stratum, and the human side counts the classes of the current pair codebook (decisions of the earlier codebook are reported as history only).
"""
from collections import Counter

from . import pairs, pairsample
from .signals import version_number
from .stats import cohen_kappa

BINS = 20
CIT_BUCKETS = [("0", None, 0), ("up to 0.01", 0, 0.01), ("0.01 to 0.02", 0.01, 0.02), ("0.02 to 0.05", 0.02, 0.05), ("0.05 to 0.10", 0.05, 0.10), ("above 0.10", 0.10, None)]
MIN_PEOPLE_TO_REVEAL = 2


def _quantiles(con, column, run, where="", points=(0.05, 0.25, 0.5, 0.75, 0.95)):
    row = con.execute(f"SELECT percentile_cont(%s::float8[]) WITHIN GROUP (ORDER BY {column}) FROM candidate_pair WHERE signal_run_id=%s AND {column} IS NOT NULL {where}",
                      (list(points), run)).fetchone()[0]
    return {f"p{int(p * 100):02d}": (None if v is None else round(v, 4)) for p, v in zip(points, row or [None] * len(points))}


def signal_stats(con, run):
    name_hi, cit_hi, s_hi, s_lo, version = con.execute("SELECT (params->>'name_hi')::float8,(params->>'cit_hi')::float8,(params->>'s_hi')::float8,(params->>'s_lo')::float8,version "
                                                       "FROM signal_run WHERE signal_run_id=%s", (run,)).fetchone()
    main = "s_def" if version_number(version) >= 4 else "s_sem_op"           # the signal the hint reads; a constant column name, never user input
    total, s_n, c_n, n_n, n_pos, same, d_n, o_n, disagree = con.execute(
        "SELECT count(*),count(s_sem_op),count(c_cit),count(n_nom),count(*) FILTER (WHERE n_nom>0),count(*) FILTER (WHERE name_sim>=%s),count(s_def),count(s_op),"
        "count(*) FILTER (WHERE abs(s_def-s_op)>=0.3) FROM candidate_pair WHERE signal_run_id=%s", (name_hi, run)).fetchone()
    bins = [{"from": round(i / BINS, 4), "to": round((i + 1) / BINS, 4), "same_label": 0, "different_label": 0} for i in range(BINS)]
    for b, is_same, n in con.execute(
            f"SELECT width_bucket(least(greatest({main},0),0.999999),0,1,%s),name_sim>=%s,count(*) FROM candidate_pair WHERE signal_run_id=%s AND {main} IS NOT NULL GROUP BY 1,2",
            (BINS, name_hi, run)):
        bins[b - 1]["same_label" if is_same else "different_label"] += n
    by_type = {}
    for is_same, n, median in con.execute(
            f"SELECT name_sim>=%s,count({main}),percentile_cont(0.5) WITHIN GROUP (ORDER BY {main}) FROM candidate_pair WHERE signal_run_id=%s GROUP BY 1", (name_hi, run)):
        by_type["same_label" if is_same else "different_label"] = {"with_signal": n, "median": None if median is None else round(median, 4)}
    cit = []
    for label, lo, hi in CIT_BUCKETS:
        cond = "c_cit=0" if label == "0" else (f"c_cit>{lo}" + (f" AND c_cit<={hi}" if hi is not None else ""))
        cit.append({"label": label, "pairs": con.execute(f"SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s AND c_cit IS NOT NULL AND {cond}", (run,)).fetchone()[0]})
    hints = {}
    for hint, is_same, n in con.execute("SELECT hint,name_sim>=%s,count(*) FROM candidate_pair WHERE signal_run_id=%s GROUP BY 1,2", (name_hi, run)):
        entry = hints.setdefault(hint, {"hint": hint, "same_label": 0, "different_label": 0, "median_s_sem_op": None})
        entry["same_label" if is_same else "different_label"] = n
    for hint, median in con.execute(f"SELECT hint,percentile_cont(0.5) WITHIN GROUP (ORDER BY {main}) FROM candidate_pair WHERE signal_run_id=%s GROUP BY 1", (run,)):
        hints[hint]["median_s_sem_op"] = hints[hint]["median_signal"] = None if median is None else round(median, 4)
    for entry in hints.values():
        entry["total"] = entry["same_label"] + entry["different_label"]
    order = {h: i for i, h in enumerate(pairs.PRIORITY)}
    return {
        "signal": main, "thresholds": {"name_hi": name_hi, "s_hi": s_hi, "s_lo": s_lo, "cit_hi": cit_hi},
        "availability": {"pairs": total, "same_label": same, "different_label": total - same, "s_sem_op": s_n, "s_def": d_n, "s_op": o_n, "def_and_op_disagree": disagree,
                         "c_cit": c_n, "n_nom": n_n, "n_nom_positive": n_pos, "n_nom_used_in_hint": False},
        "s_sem_op": {"bins": bins, "quantiles": _quantiles(con, main, run), "by_label_type": by_type, "signal": main},
        "c_cit": {"buckets": cit, "quantiles": _quantiles(con, "c_cit", run, points=(0.5, 0.75, 0.9, 0.95, 0.99)),
                  "at_or_above_threshold": con.execute("SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s AND c_cit>=%s", (run, cit_hi)).fetchone()[0]},
        "hints": sorted(hints.values(), key=lambda e: order.get(e["hint"], 99)),
        "retrieval_reasons": [{"reason": r, "pairs": n} for r, n in con.execute(
            "SELECT r,count(*) FROM candidate_pair,unnest(retrieval_reasons) r WHERE signal_run_id=%s GROUP BY 1 ORDER BY 2 DESC", (run,))],
    }


def review(con, run, sample_id, validator=None):
    """Progress of the blind validation of the sample. No pair-level information; the pipeline's guess is shown only once enough people have finished."""
    if not sample_id:
        return {"sample": None}
    items = con.execute("SELECT i.pair_id,i.sample_kind,p.hint FROM pair_sample_item i JOIN candidate_pair p USING (pair_id) WHERE i.pair_sample_id=%s", (sample_id,)).fetchall()
    ids = {pid for pid, *_ in items}
    decisions = {pid: per for pid, per in pairs.current(con, run).items() if pid in ids}
    people = Counter(name for per in decisions.values() for name in per)
    states = Counter(pairs.state_of(decisions[pid])[0] if pid in decisions else "undecided" for pid in ids)
    old_rows, old_pairs = pairs.history(con, run)
    out = {"codebook": pairs.CODEBOOK, "earlier_codebook": {"decisions": old_rows, "pairs": old_pairs},
           "sample": {"pair_sample_id": str(sample_id), "pairs": len(ids), "probabilistic": sum(1 for _, k, _ in items if k == "probabilistic"),
                      "challenge": sum(1 for _, k, _ in items if k == "challenge")},
           "people": [{"name": n, "decided": c} for n, c in sorted(people.items())],
           "my_decided": people.get(validator, 0) if validator else None,
           "states": {s: states.get(s, 0) for s in ("settled", "contested", "pending", "undecided")}}
    covered = sum(1 for pid in ids if len(decisions.get(pid, {})) >= MIN_PEOPLE_TO_REVEAL)
    out["revealed"] = covered == len(ids)
    if not out["revealed"]:
        out["held_back"] = (f"The comparison with the triage hint and the agreement between annotators appear when at least {MIN_PEOPLE_TO_REVEAL} people have decided every pair "
                            f"({covered} of {len(ids)} pairs have {MIN_PEOPLE_TO_REVEAL} decisions), so that nobody is anchored while deciding.")
        return out
    settled = {pid: pairs.state_of(per)[1] for pid, per in decisions.items() if pairs.state_of(per)[0] == "settled"}
    out["human"] = {"by_class": dict(Counter(settled.values()))}
    table = {"probabilistic": {}, "challenge": {}}
    for pid, kind, hint in items:
        if pid in settled:
            row = table[kind].setdefault(hint, Counter())
            row[settled[pid]] += 1
    out["triage_vs_human"] = {kind: {hint: dict(c) for hint, c in rows.items()} for kind, rows in table.items()}
    full = [n for n, c in people.items() if all(n in decisions.get(pid, {}) for pid in ids)]
    agreement = []
    for i, a in enumerate(sorted(full)):
        for b in sorted(full)[i + 1:]:
            both = [(decisions[pid][a][0], decisions[pid][b][0]) for pid in ids]
            entry = {"annotators": [a, b], "pairs": len(both)}
            fallacy = lambda d: d if d in pairs.POSITIVE else "other"           # the gate's view: conceptual jingle, conceptual jangle, or neither
            for field, pick in (("class", lambda d: d), ("fallacy", fallacy)):
                k = cohen_kappa([(pick(x), pick(y)) for x, y in both])
                entry[field] = {"agreement": round(k["agreement"], 4), "kappa": None if k["kappa"] is None else round(k["kappa"], 4)}
            agreement.append(entry)
    out["agreement"] = agreement
    report = pairsample.report(con, sample_id)
    out["estimates"] = {"candidate_pairs_by_class": report["estimated_candidate_pairs_by_class"], "adjudicated_per_positive_class": report["adjudicated_per_positive_class"], "warning": report["warning"]}
    return out


def stats(con, run, validator=None):
    row = con.execute("SELECT version,model_id,extraction_run_id,node_count,pair_count,created_at FROM signal_run WHERE signal_run_id=%s", (run,)).fetchone()
    if not row:
        raise ValueError("Unknown signal run")
    version, model_id, extraction_run_id, nodes, n_pairs, created = row
    return {"run": {"signal_run_id": str(run), "version": version, "model_id": model_id, "extraction_run_id": str(extraction_run_id), "nodes": nodes, "pairs": n_pairs,
                    "created_at": created.isoformat(timespec="seconds")},
            **signal_stats(con, run), "review": review(con, run, pairsample.latest(con, run), validator)}
