"""Phase 7 (second half): the preregistered quality gates (the internal project notes section 5) and the gold sets they are measured against.

A gate is `pass`, `fail` or `not_evaluable` (no gold data yet). `not_evaluable` is NOT a pass: a release with any gate that is not a
pass is blocked. Gold sets are human-made files (see `templates/`) stored with their checksum and the name of the person who supplied
them; nothing here invents a gold label.

Interpretations to confirm (the repository does not define them): "records complete" for extraction is the share of gold papers whose
gold items are ALL found by the extraction; the pair gate counts a candidate as a hit only when its retrieval stratum (or v3 triage hint) points to the
gold class. Proposal v4: the jingle/jangle gate holds SEPARATELY for each positive conceptual class, so its value is the lower of the two (and it cannot be
evaluated while either class has no gold case); the report adds how the near negatives (measurement divergence, scope variation, empirical redundancy)
were flagged. Gold pair items carry a `class` of the pair codebook v2; an item of the earlier codebook (`flag`) is read with LEGACY_FLAG_CLASS and said so.
"""
import json
import logging

from .core import digest, stable_id
from .stats import icc_agreement, prf, spearman

LOG = logging.getLogger("research0")
GATES = [  # Appendix C and the table of section "Human validation of the pipeline"; thresholds are ex ante targets, not results
    {"id": "seed_recovery", "name": "Query seed recovery", "threshold": 0.95},
    {"id": "screening_recall", "name": "Screening recall (false negatives among the excluded included)", "threshold": 0.95},
    {"id": "human_agreement", "name": "Human agreement on membership (kappa and alpha)", "threshold": 0.80},
    {"id": "membership_macro_f1", "name": "Pipeline membership macro-F1", "threshold": 0.85},
    {"id": "multilabel_jaccard", "name": "Multi-label set: mean Jaccard", "threshold": 0.80},
    {"id": "multilabel_micro_f1", "name": "Multi-label set: micro-F1", "threshold": 0.85},
    {"id": "multilabel_macro_f1", "name": "Multi-label set: macro-F1", "threshold": 0.80},
    {"id": "extraction_f1", "name": "Extraction F1 (strict)", "threshold": 0.90},
    {"id": "extraction_complete", "name": "Complete extraction records", "threshold": 0.90},
    {"id": "extraction_definition_precision", "name": "Definition and measure precision", "threshold": 0.95},
    {"id": "jj_recall", "name": "Conceptual jingle/jangle recall (each positive class)", "threshold": 0.90},
    {"id": "jj_precision", "name": "Conceptual jingle/jangle precision (each positive class)", "threshold": 0.75},
    {"id": "jj_f1", "name": "Conceptual jingle/jangle F1 (each positive class)", "threshold": 0.80},
    {"id": "rating_icc", "name": "Human ICC on continuous ratings", "threshold": 0.75},
    {"id": "rating_spearman", "name": "Pipeline-human Spearman on continuous ratings", "threshold": 0.70},
    {"id": "calibration_error", "name": "Calibration error of the pipeline's probabilities (lower is better)", "threshold": 0.10, "direction": "max"},
]
FLAG_OF_HINT = {"jingle_candidate": "conceptual_jingle", "jangle_candidate": "conceptual_jangle", "same_label_divergent": "conceptual_jingle",
                "different_label_close": "conceptual_jangle"}
LEGACY_FLAG_CLASS = {"jingle": "conceptual_jingle", "jangle": "conceptual_jangle", "none": "distinct", "unresolved": "unresolved"}
POSITIVE = ("conceptual_jingle", "conceptual_jangle")
NEAR_NEGATIVE = ("measurement_divergence", "scope_variation", "empirical_redundancy")
AGREEMENT_ACTIONS = ((0.80, "proceed"), (0.667, "revise the codebook and recode"), (float("-inf"), "block the variable"))


def passing_value(gate):
    """A value that passes the gate (for tests and the demo, which force every gate)."""
    return gate["threshold"] - 0.01 if gate.get("direction") == "max" else gate["threshold"] + 0.01


def verdict(gate_id, value, detail="", extra=None):
    gate = next(g for g in GATES if g["id"] == gate_id)
    if value is None:
        status = "not_evaluable"
    elif gate.get("direction") == "max":
        status = "pass" if value <= gate["threshold"] else "fail"
    else:
        status = "pass" if value >= gate["threshold"] else "fail"
    return {**gate, "value": value, "status": status, "detail": detail, **(extra or {})}


# --- Gold sets --------------------------------------------------------------------------------------------------------------

def store_gold(con, kind, label, content, provided_by):
    provided_by = (provided_by or "").strip()
    if kind not in ("extraction", "pairs", "ratings") or not provided_by or not isinstance(content, dict) or not content.get("items"):
        raise ValueError("A gold set needs a kind, the name of the person who provided it and a non-empty 'items' list")
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False)
    sha = digest(blob.encode())
    gold_id = stable_id("gold", kind, sha)
    con.execute("INSERT INTO gold_set(gold_id,kind,label,content,content_sha256,provided_by) VALUES (%s,%s,%s,%s::jsonb,%s,%s) ON CONFLICT DO NOTHING", (gold_id, kind, label, blob, sha, provided_by))
    return gold_id


def load_gold(con, gold_id):
    row = con.execute("SELECT kind,content FROM gold_set WHERE gold_id=%s", (gold_id,)).fetchone()
    if not row:
        raise ValueError("Unknown gold set")
    return row


# --- Gates computed from data already in the database -----------------------------------------------------------------------

def seed_gate(con, run_id):
    rows = con.execute("SELECT community,count(*) FILTER (WHERE outcome='found'),count(*) FROM catalog_seed_result WHERE run_id=%s AND seed_status='eligible' GROUP BY 1", (run_id,)).fetchall()
    if not rows:
        return verdict("seed_recovery", None, "No eligible seed results for this catalog run")
    rates = {c: found / total for c, found, total in rows if total}
    return verdict("seed_recovery", min(rates.values()), "; ".join(f"{c} {found}/{total}" for c, found, total in rows))


AUDIT_GATES = ("screening_recall", "human_agreement", "membership_macro_f1", "multilabel_jaccard", "multilabel_micro_f1", "multilabel_macro_f1", "calibration_error")


def membership_gates(con, audit_id):
    from . import audit
    if not audit_id:
        return [verdict(g, None, "No audit sample") for g in AUDIT_GATES]
    try:
        agreement = audit.agreement(con, audit_id)
        estimate = audit.estimates(con, audit_id)
    except ValueError as exc:
        return [verdict(g, None, str(exc)) for g in AUDIT_GATES]
    pieces = [(agreement[c]["three_class"]["kappa"], agreement[c]["alpha_three_class"]) for c in audit.COMMUNITIES]
    values = [v for pair in pieces for v in pair]
    agree = None if any(v is None for v in values) else min(values)
    action = next(text for floor, text in AGREEMENT_ACTIONS if agree is not None and agree >= floor) if agree is not None else None
    f1s = [estimate[c]["strict"]["macro_f1"] for c in audit.COMMUNITIES]
    macro = None if any(v is None for v in f1s) else min(f1s)
    cis = {c: estimate[c]["strict"]["macro_f1_ci95"] for c in audit.COMMUNITIES}
    screening = estimate["screening"]["strict"]
    multi = estimate["multilabel"] or {}
    calibration = estimate["calibration"]
    pick = lambda key: multi.get(key)
    return [
        verdict("screening_recall", screening["recall"], f"95% interval {screening.get('recall_ci95')}; precision {screening['precision']}; abstention (pipeline uncertain) rate by community {estimate['uncertain_rate_of_preassignment']}; "
                f"estimated missed members among the 'no' papers {[estimate[c]['false_negatives_among_no']['estimated_missed'] for c in audit.COMMUNITIES]}"),
        verdict("human_agreement", agree, f"lowest of kappa and alpha over CTO and TIM; if it fails: {action}", {"action": action}),
        verdict("membership_macro_f1", macro, "lowest community; 95% intervals " + json.dumps(cis), {"intervals": cis}),
        verdict("multilabel_jaccard", pick("mean_jaccard"), f"{multi.get('included')} of {multi.get('papers')} audited papers (rest abstained); 95% interval {multi.get('mean_jaccard_ci95')}"),
        verdict("multilabel_micro_f1", pick("micro_f1"), f"95% interval {multi.get('micro_f1_ci95')}"),
        verdict("multilabel_macro_f1", pick("macro_f1"), f"95% interval {multi.get('macro_f1_ci95')}"),
        verdict("calibration_error", None if calibration is None else calibration["ece"], "expected calibration error over 10 bins" if calibration else "the pre-assignment gave no probabilities"),
    ]


def definition_precision_gate(con, extraction_run_id):
    """Precision of the extracted definitions and measures, from the people's decisions: validated / (validated + rejected) among the records judged.
    A record counts as rejected if anybody's latest decision on it is a rejection."""
    if not extraction_run_id:
        return verdict("extraction_definition_precision", None, "No extraction run")
    judged = {}
    for subject_id, decision in con.execute("SELECT subject_id,decision FROM (SELECT DISTINCT ON (subject_id,validator) subject_id,decision FROM decision_event "
                                            "ORDER BY subject_id,validator,decided_at DESC,event_id DESC) t"):
        judged.setdefault(subject_id, set()).add(decision)
    mine = {r[0] for r in con.execute("SELECT mention_id FROM construct_mention WHERE extraction_run_id=%s AND definition_text IS NOT NULL", (extraction_run_id,))}
    mine |= {r[0] for r in con.execute("SELECT s.measure_id FROM construct_measure s JOIN construct_mention m USING (mention_id) WHERE m.extraction_run_id=%s", (extraction_run_id,))}
    decided = {k: v for k, v in judged.items() if k in mine and v & {"validated", "rejected"}}
    if not decided:
        return verdict("extraction_definition_precision", None, "No definition or measure has been judged by a person yet")
    good = sum(1 for v in decided.values() if "rejected" not in v)
    return verdict("extraction_definition_precision", good / len(decided), f"{good} of {len(decided)} judged definitions and measures were validated and none rejected")


def _extraction_sets(con, extraction_run_id, canonical_ids):
    rejected = {sid for sid, in con.execute(
        "SELECT DISTINCT subject_id FROM (SELECT DISTINCT ON (subject_id,validator) subject_id,decision FROM decision_event ORDER BY subject_id,validator,decided_at DESC,event_id DESC) t WHERE decision='rejected'")}
    from .extraction import normalize_label
    mentions = {(c, l) for m, c, l in con.execute("SELECT mention_id,canonical_id,label_normalized FROM construct_mention WHERE extraction_run_id=%s AND canonical_id = ANY(%s)",
                                                  (extraction_run_id, canonical_ids)) if m not in rejected}
    measures = {(c, l, normalize_label(n)) for m, c, l, n in con.execute(
        "SELECT s.measure_id,m.canonical_id,m.label_normalized,s.name FROM construct_measure s JOIN construct_mention m USING (mention_id) WHERE m.extraction_run_id=%s AND m.canonical_id = ANY(%s)",
        (extraction_run_id, canonical_ids)) if m not in rejected}
    claims = {(c, s, o, r, d) for k, c, s, o, r, d in con.execute(
        "SELECT n.claim_id,n.canonical_id,s.label_normalized,o.label_normalized,n.relation,n.direction FROM nomological_claim n JOIN construct_mention s ON s.mention_id=n.subject_mention_id "
        "JOIN construct_mention o ON o.mention_id=n.object_mention_id WHERE n.extraction_run_id=%s AND n.canonical_id = ANY(%s)", (extraction_run_id, canonical_ids)) if k not in rejected}
    return mentions, measures, claims


def _relaxed(predicted, gold):
    """F1 of construct mentions when labels match softly (character n-gram similarity >= 0.8) within the same paper."""
    from .signals import LabelSimilarity
    similarity = LabelSimilarity({l for _, l in predicted} | {l for _, l in gold})
    matched_gold = set()
    tp = 0
    for paper, label in predicted:
        options = [g for g in gold if g[0] == paper and g not in matched_gold and similarity.between(label, g[1]) >= 0.8]
        if options:
            matched_gold.add(options[0])
            tp += 1
    return prf(tp, len(predicted) - tp, len(gold) - tp)["f1"]


def extraction_gates(con, gold, extraction_run_id):
    import uuid
    from .extraction import normalize_label
    if not gold or not extraction_run_id:
        return [verdict("extraction_f1", None, "No gold extraction set"), verdict("extraction_complete", None, "No gold extraction set")]
    ids = [uuid.UUID(i["canonical_id"]) for i in gold["items"]]
    mentions, measures, claims = _extraction_sets(con, extraction_run_id, ids)
    g_mentions, g_measures, g_claims, covered = set(), set(), set(), 0
    for item in gold["items"]:
        c = uuid.UUID(item["canonical_id"])
        mine = ({(c, normalize_label(x)) for x in item.get("constructs", [])}, {(c, normalize_label(m["construct"]), normalize_label(m["name"])) for m in item.get("measures", [])},
                {(c, normalize_label(x["subject"]), normalize_label(x["object"]), x["relation"], x["direction"]) for x in item.get("claims", [])})
        g_mentions |= mine[0]; g_measures |= mine[1]; g_claims |= mine[2]
        covered += mine[0] <= mentions and mine[1] <= measures and mine[2] <= claims
    tp = len(mentions & g_mentions) + len(measures & g_measures) + len(claims & g_claims)
    fp = len(mentions - g_mentions) + len(measures - g_measures) + len(claims - g_claims)
    fn = len(g_mentions - mentions) + len(g_measures - measures) + len(g_claims - claims)
    score = prf(tp, fp, fn)
    relaxed = _relaxed(mentions, g_mentions)
    return [verdict("extraction_f1", score["f1"], f"strict micro match over constructs, measures and claims of {len(ids)} gold papers: tp={tp} fp={fp} fn={fn}; "
                                                  f"relaxed construct match (label similarity >= 0.8) F1 {relaxed}; every stored record carries a quote and a page by construction"),
            verdict("extraction_complete", covered / len(ids), f"{covered} of {len(ids)} gold papers fully covered")]


def _pair_key(item, signal_run_id=None):
    return tuple(sorted([(item["canonical_id_a"], item["label_a"]), (item["canonical_id_b"], item["label_b"])]))


def candidate_index(con, signal_run_id):
    from .extraction import normalize_label
    index = {}
    for ca, la, cb, lb, sem, hint, pair_id in con.execute(
            "SELECT a.canonical_id,a.label_normalized,b.canonical_id,b.label_normalized,coalesce(p.s_def,p.s_sem_op),p.hint,p.pair_id FROM candidate_pair p "
            "JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b "
            "WHERE p.signal_run_id=%s", (signal_run_id,)):
        index[tuple(sorted([(str(ca), la), (str(cb), lb)]))] = (sem, hint, pair_id)
    return index


def pipeline_flags(con, signal_run_id, index):
    """What the pipeline flags each candidate pair as. Once the model has filtered the targeted pairs (the cascade, 14 Oct 2026), its class is the pipeline's
    answer on those pairs and a pair it did not read is not flagged; before any model proposal exists on the run, the hint's class is the answer."""
    from .proposals import latest_classes
    classes = latest_classes(con, signal_run_id)
    if not classes:
        return {k: FLAG_OF_HINT.get(v[1]) for k, v in index.items()}, "hint"
    return {k: (classes.get(v[2]) if classes.get(v[2]) in POSITIVE else None) for k, v in index.items()}, "model"


def gold_class(item):
    """The class of a gold pair item: `class` (codebook v2), else the earlier `flag` read with LEGACY_FLAG_CLASS."""
    return item.get("class") or LEGACY_FLAG_CLASS.get(item.get("flag"))


def pair_gates(con, gold, signal_run_id):
    """Recall, precision and F1 for flagging conceptual jingle and conceptual jangle pairs, computed SEPARATELY for each positive class (proposal v4); the gate
    takes the lower of the two classes. A gold pair the pipeline never proposed is a miss; precision counts only pairs the gold set covers. The near negatives
    (measurement divergence, scope variation, empirical redundancy) are reported with what the pipeline flagged them as."""
    from .extraction import normalize_label
    if not gold or not signal_run_id:
        return [verdict(g, None, "No gold pair set") for g in ("jj_recall", "jj_precision", "jj_f1")]
    index = candidate_index(con, signal_run_id)
    norm = lambda it: {**it, "label_a": normalize_label(it["label_a"]), "label_b": normalize_label(it["label_b"])}
    gold_items = {_pair_key(norm(it)): gold_class(it) for it in gold["items"]}
    legacy = sum(1 for it in gold["items"] if not it.get("class"))
    flags, source = pipeline_flags(con, signal_run_id, index)
    flagged = flags.get
    per_class = {}
    for cls in POSITIVE:
        positives = {k for k, c in gold_items.items() if c == cls}
        hits = {k for k in positives if flagged(k) == cls}
        predicted = [k for k in gold_items if flagged(k) == cls]
        correct = [k for k in predicted if gold_items[k] == cls]
        tp, fp, fn = len(correct), len(predicted) - len(correct), len(positives) - len(hits)
        per_class[cls] = {**prf(tp, fp, fn), "gold_positives": len(positives),
                          "warning": None if len(positives) >= 50 else f"only {len(positives)} gold {cls.replace('_', ' ')} cases (the proposal asks for at least 50 adjudicated per positive class)"}
    near = {cls: {"gold": sum(1 for c in gold_items.values() if c == cls), "flagged_as": {p: sum(1 for k, c in gold_items.items() if c == cls and flagged(k) == p) for p in POSITIVE}}
            for cls in NEAR_NEGATIVE}
    lowest = lambda metric: None if any(per_class[c]["gold_positives"] == 0 for c in POSITIVE) else min(per_class[c][metric] for c in POSITIVE)
    note = "; ".join(f"{c.replace('_', ' ')}: recall {v['recall']}, precision {v['precision']}, F1 {v['f1']} ({v['gold_positives']} gold)" for c, v in per_class.items())
    note += "; near negatives flagged as a fallacy: " + ", ".join(f"{c.replace('_', ' ')} {sum(v['flagged_as'].values())}/{v['gold']}" for c, v in near.items())
    if any(per_class[c]["gold_positives"] == 0 for c in POSITIVE):
        note = "Not evaluable until both positive classes have gold cases. " + note
    warnings = [c["warning"] for c in per_class.values() if c["warning"]] + ([f"{legacy} gold item(s) use the earlier codebook (flag), read as {LEGACY_FLAG_CLASS}"] if legacy else [])
    extra = {"per_class": per_class, "near_negatives": near, "warnings": warnings, "flagged_by": source}
    return [verdict("jj_recall", lowest("recall"), note, extra), verdict("jj_precision", lowest("precision"), note, extra), verdict("jj_f1", lowest("f1"), note, extra)]


def rating_gates(con, gold, signal_run_id):
    from .extraction import normalize_label
    if not gold or not signal_run_id:
        return [verdict("rating_icc", None, "No gold ratings"), verdict("rating_spearman", None, "No gold ratings")]
    index = candidate_index(con, signal_run_id)
    raters = sorted({r for it in gold["items"] for r in it["ratings"]})
    complete = [it for it in gold["items"] if set(it["ratings"]) == set(raters)]
    if len(raters) < 2 or len(complete) < 3:
        return [verdict("rating_icc", None, "Needs at least 2 raters and 3 items rated by all of them"), verdict("rating_spearman", None, "Needs at least 3 rated items")]
    icc = icc_agreement([[it["ratings"][r] for r in raters] for it in complete])["icc"]
    pairs = []
    for it in complete:
        key = _pair_key({**it, "label_a": normalize_label(it["label_a"]), "label_b": normalize_label(it["label_b"])})
        if key in index and index[key][0] is not None:
            pairs.append((sum(it["ratings"].values()) / len(raters), index[key][0]))
    rho = spearman([h for h, _ in pairs], [p for _, p in pairs])["rho"] if len(pairs) >= 3 else None
    return [verdict("rating_icc", icc, f"{len(complete)} items, {len(raters)} raters"),
            verdict("rating_spearman", rho, f"{len(pairs)} of {len(complete)} rated pairs have a pipeline S_def (or, for an earlier run, S_sem-op)")]


def evaluate(con, run_id, audit_id=None, extraction_run_id=None, signal_run_id=None, gold_extraction=None, gold_pairs=None, gold_ratings=None):
    """All gates. gold_* are gold_set ids (or None)."""
    load = lambda g: load_gold(con, g)[1] if g else None
    gates = [seed_gate(con, run_id), *membership_gates(con, audit_id), *extraction_gates(con, load(gold_extraction), extraction_run_id), definition_precision_gate(con, extraction_run_id),
             *pair_gates(con, load(gold_pairs), signal_run_id), *rating_gates(con, load(gold_ratings), signal_run_id)]
    order = {g["id"]: i for i, g in enumerate(GATES)}
    return sorted(gates, key=lambda g: order[g["id"]])
