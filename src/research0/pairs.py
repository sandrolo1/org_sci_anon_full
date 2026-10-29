"""Phase 7 (first half): human adjudication of candidate pairs.

Codebook v2 (proposal v4, Appendix B, 13 Oct 2026). A person looks at a dossier and records ONE class of the codebook - conceptual jingle, conceptual jangle,
measurement divergence, scope variation, empirical redundancy, part-whole, related but distinct, distinct, same meaning (no fallacy) or unresolved - with the
answers of the pair comparison (same referent? compatible defining properties? does substituting the labels, or separating the meanings, change a central
proposition or prediction?), the theoretical consequence, a counterexample and a rationale. "Same meaning" is not in the proposal's list: it is the case of one
label used with one meaning, which the jingle test needs as its negative (an interpretation of 13 Oct 2026). The two conceptual classes need a meaning-level
argument and a theoretical consequence, so the checklist must be consistent with them:
  conceptual jingle  same label, properties not compatible, separating the meanings changes a proposition or prediction;
  conceptual jangle  different labels, same referent, compatible properties, substitution preserves the central propositions.
Different items, a different context or a high correlation alone are measurement divergence, scope variation or empirical redundancy, never a fallacy.

Decisions are append-only; the latest row per (pair, validator) of the CURRENT codebook counts. The decisions recorded with codebook v1 (relation + flag,
before 13 Oct 2026) stay in the table as history and are never counted by these readers: the user decided that the pairs are decided again.

A pair is *settled* when a person marked a decision final (an adjudicator), or when at least two different people's latest decisions agree (same class)
and nobody's latest decision differs. Otherwise it is *contested* or *pending*. No score settles anything, and a settled pair is a taxonomic decision about
two construct uses: it does not merge anything by itself.
"""
import csv
import hashlib
import json
from pathlib import Path

CODEBOOK = "pair-codebook-v2"
LEGACY_CODEBOOK = "pair-codebook-v1"
CLASSES = ("conceptual_jingle", "conceptual_jangle", "measurement_divergence", "scope_variation", "empirical_redundancy", "part_whole", "related_but_distinct",
           "distinct", "same_meaning", "unresolved")
POSITIVE = ("conceptual_jingle", "conceptual_jangle")
NEAR_NEGATIVE = ("measurement_divergence", "scope_variation", "empirical_redundancy")      # what a conceptual fallacy is most often mistaken for
ANSWERS = ("yes", "no", "unclear")
CHECKLIST = ("same_referent", "compatible_properties", "substitution_changes")
CLASS_LABELS = {"conceptual_jingle": "Conceptual jingle", "conceptual_jangle": "Conceptual jangle", "measurement_divergence": "Measurement divergence",
                "scope_variation": "Scope variation", "empirical_redundancy": "Empirical redundancy", "part_whole": "Part-whole", "related_but_distinct": "Related but distinct",
                "distinct": "Distinct", "same_meaning": "Same meaning (no fallacy)", "unresolved": "Unresolved"}
# Codebook v1, kept to read the history.
RELATIONS = ("equivalent", "partially_overlapping", "related_but_distinct", "distinct", "insufficient_evidence")
FLAGS = ("jingle", "jangle", "none", "unresolved")
PRIORITY = {"same_label_divergent": 0, "different_label_close": 1, "close_measures_distinct_meaning": 2, "context_shifted": 3, "jingle_candidate": 0, "jangle_candidate": 1,
            "related_but_distinct": 2, "unresolved": 4, "hard_negative": 5, "none": 6}


def blind(content):
    """The dossier without signals, hint, retrieval reasons or candidate-relation scores, so that a rater is not anchored on them (use for gold labelling)."""
    return {k: v for k, v in content.items() if k not in ("signals", "triage_hint", "retrieval_reasons", "candidate_relation")}


def check(decision_class, labels_equal, checklist, consequence):
    """The codebook's decision rule on one answer. Raises ValueError with the reason when the class and the checklist contradict each other."""
    if decision_class not in CLASSES:
        raise ValueError("The class must be one of: " + ", ".join(CLASSES))
    if any(checklist.get(k) not in ANSWERS for k in CHECKLIST):
        raise ValueError("Answer the three comparison questions (same referent, compatible properties, substitution changes a proposition): yes, no or unclear")
    if decision_class in POSITIVE and not (consequence or "").strip():
        raise ValueError("A conceptual jingle or jangle needs its theoretical consequence: which proposition, explanation or prediction changes")
    if decision_class == "conceptual_jingle":
        if not labels_equal:
            raise ValueError("A conceptual jingle is one label over different meanings: these two uses have different labels")
        if checklist["compatible_properties"] != "no" or checklist["substitution_changes"] != "yes":
            raise ValueError("A conceptual jingle needs defining properties that are not compatible and a separation that changes a proposition or prediction; "
                             "otherwise it is measurement divergence, scope variation or unresolved")
    if decision_class == "conceptual_jangle":
        if labels_equal:
            raise ValueError("A conceptual jangle is different labels for one meaning: these two uses have the same label")
        if checklist["same_referent"] != "yes" or checklist["compatible_properties"] != "yes" or checklist["substitution_changes"] != "no":
            raise ValueError("A conceptual jangle needs the same referent, compatible defining properties and a substitution that preserves the central propositions; "
                             "a high correlation or shared items alone are empirical redundancy")
    if decision_class == "same_meaning" and not labels_equal:
        raise ValueError("Same meaning is for one label used with one meaning; different labels with one meaning are a conceptual jangle")


def labels_equal(con, pair_id):
    row = con.execute("SELECT a.label_normalized=b.label_normalized FROM candidate_pair p JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
                      "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b WHERE p.pair_id=%s", (pair_id,)).fetchone()
    if not row:
        raise ValueError("Unknown pair")
    return row[0]


def decide(con, pair_id, validator, decision_class, rationale, checklist=None, consequence="", counterexample="", is_final=False):
    """Records one decision under the current codebook (append-only)."""
    validator, rationale = (validator or "").strip(), (rationale or "").strip()
    checklist = checklist or {}
    if not validator or len(validator) > 80 or not rationale or len(rationale) > 4000 or len(consequence or "") > 4000 or len(counterexample or "") > 4000:
        raise ValueError("Validator name, a class and a rationale are required")
    if not con.execute("SELECT 1 FROM pair_dossier WHERE pair_id=%s", (pair_id,)).fetchone():
        raise ValueError("The pair has no dossier yet: build it first, so that the decision is taken on the evidence")
    check(decision_class, labels_equal(con, pair_id), checklist, consequence)
    con.execute("INSERT INTO pair_decision(pair_id,codebook_version,decision_class,same_referent,compatible_properties,substitution_changes,theoretical_consequence,counterexample,"
                "rationale,validator,is_final) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (pair_id, CODEBOOK, decision_class, checklist["same_referent"], checklist["compatible_properties"], checklist["substitution_changes"],
                 (consequence or "").strip() or None, (counterexample or "").strip() or None, rationale, validator, bool(is_final)))


def current(con, signal_run_id):
    """{pair_id: {validator: (class, is_final, rationale)}}: the latest decision of each person under the current codebook."""
    rows = con.execute(
        "SELECT DISTINCT ON (d.pair_id,d.validator) d.pair_id,d.validator,d.decision_class,d.is_final,d.rationale FROM pair_decision d JOIN candidate_pair p USING (pair_id) "
        "WHERE p.signal_run_id=%s AND d.codebook_version=%s ORDER BY d.pair_id,d.validator,d.decided_at DESC,d.decision_id DESC", (signal_run_id, CODEBOOK)).fetchall()
    out = {}
    for pair_id, validator, decision_class, final, rationale in rows:
        out.setdefault(pair_id, {})[validator] = (decision_class, final, rationale)
    return out


def details(con, pair_ids):
    """{pair_id: [the latest current-codebook decision of each person, with its checklist]} for the release and the forum."""
    rows = con.execute(
        "SELECT DISTINCT ON (pair_id,validator) pair_id,validator,decision_class,same_referent,compatible_properties,substitution_changes,theoretical_consequence,counterexample,"
        "rationale,is_final FROM pair_decision WHERE codebook_version=%s AND pair_id = ANY(%s) ORDER BY pair_id,validator,decided_at DESC,decision_id DESC",
        (CODEBOOK, list(pair_ids))).fetchall()
    out = {}
    for pid, validator, cls, ref, props, subst, consequence, counter, rationale, final in rows:
        out.setdefault(pid, []).append({"validator": validator, "class": cls, "same_referent": ref, "compatible_properties": props, "substitution_changes": subst,
                                        "theoretical_consequence": consequence, "counterexample": counter, "rationale": rationale, "final": final})
    return out


def history(con, signal_run_id):
    """How many decisions were recorded with the earlier codebook (kept, not counted)."""
    return con.execute("SELECT count(*), count(DISTINCT d.pair_id) FROM pair_decision d JOIN candidate_pair p USING (pair_id) WHERE p.signal_run_id=%s AND d.codebook_version=%s",
                       (signal_run_id, LEGACY_CODEBOOK)).fetchone()


def state_of(per_validator):
    """('settled', class) | ('contested', None) | ('pending', None)"""
    finals = {decision for decision, final, _ in per_validator.values() if final}
    if finals:
        return ("settled", next(iter(finals))) if len(finals) == 1 else ("contested", None)
    decisions = [decision for decision, _, _ in per_validator.values()]
    if len(decisions) >= 2 and len(set(decisions)) == 1:
        return "settled", decisions[0]
    if len(set(decisions)) > 1:
        return "contested", None
    return "pending", None


def settled(con, signal_run_id):
    """{pair_id: class}"""
    return {pair_id: decision for pair_id, per in current(con, signal_run_id).items() for state, decision in [state_of(per)] if state == "settled"}


def counts(con, signal_run_id):
    decisions = current(con, signal_run_id)
    total = con.execute("SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s", (signal_run_id,)).fetchone()[0]
    old_rows, old_pairs = history(con, signal_run_id)
    tally = {"pairs": total, "with_dossier": con.execute("SELECT count(*) FROM pair_dossier d JOIN candidate_pair p USING (pair_id) WHERE p.signal_run_id=%s", (signal_run_id,)).fetchone()[0],
             "settled": 0, "contested": 0, "pending": 0, "by_class": {}, "codebook": CODEBOOK,
             "earlier_codebook": {"decisions": old_rows, "pairs": old_pairs, "note": "decisions with the earlier codebook (relation + flag) are kept as history and not counted"}}
    for per in decisions.values():
        state, decision = state_of(per)
        tally[state] += 1
        if decision:
            tally["by_class"][decision] = tally["by_class"].get(decision, 0) + 1
    tally["no_decision"] = total - len(decisions)
    return tally


def shuffled_key(pair_sample_id, pair_id):
    """A fixed pseudo-random position of a pair inside a sample, the same for everybody: the order carries no information about the pair."""
    return hashlib.sha256(f"{pair_sample_id}|{pair_id}".encode()).hexdigest()


def queue(con, signal_run_id, validator=None, limit=50, pair_sample_id=None):
    """Pairs with a dossier that still need this person's decision under the current codebook (or anyone's, if no name is given). A pair decided only
    with the earlier codebook is waiting again.
    Without a sample: most informative hint first (for a curator, not for a blind rater). With a sample (the blind validation): only the pairs of that
    sample, in a fixed shuffled order, so that neither the selection nor the order tells what the pipeline thought."""
    decisions = current(con, signal_run_id)
    if pair_sample_id:
        rows = con.execute("SELECT p.pair_id,p.hint,d.content FROM pair_sample_item i JOIN candidate_pair p USING (pair_id) JOIN pair_dossier d USING (pair_id) "
                           "WHERE i.pair_sample_id=%s AND p.signal_run_id=%s", (pair_sample_id, signal_run_id)).fetchall()
    else:
        rows = con.execute("SELECT p.pair_id,p.hint,d.content FROM candidate_pair p JOIN pair_dossier d USING (pair_id) WHERE p.signal_run_id=%s ORDER BY p.pair_id", (signal_run_id,)).fetchall()
    out = []
    for pair_id, hint, content in rows:
        per = decisions.get(pair_id, {})
        if state_of(per)[0] == "settled" or (validator and validator in per):
            continue
        out.append((shuffled_key(pair_sample_id, pair_id) if pair_sample_id else PRIORITY.get(hint, 9), str(pair_id), pair_id, hint, content))
    out.sort(key=lambda x: x[:2])
    return [{"pair_id": str(pid), "hint": hint, "dossier": content} for _, _, pid, hint, content in out[:limit]]


def export_blind(con, signal_run_id, path, hints=None, limit=None):
    """Blind dossiers as JSON lines for annotation outside the database (no signals, no hint)."""
    rows = con.execute("SELECT p.pair_id,p.hint,d.content FROM candidate_pair p JOIN pair_dossier d USING (pair_id) WHERE p.signal_run_id=%s ORDER BY p.pair_id", (signal_run_id,)).fetchall()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as stream:
        for pair_id, hint, content in rows:
            if hints and hint not in hints:
                continue
            stream.write(json.dumps(blind(content), ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def export_settled(con, signal_run_id, path):
    """The typed construct relations decided for the release (current codebook): settled pairs with the comparison checklist, the theoretical consequence,
    the counterexample and the reasons of the people who decided, and the signals that were available."""
    rows = con.execute(
        "SELECT p.pair_id,a.label_normalized,a.canonical_id,b.label_normalized,b.canonical_id,p.name_sim,p.s_def,p.s_op,p.s_sem_op,p.c_cit,p.n_nom,p.hint "
        "FROM candidate_pair p JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
        "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b WHERE p.signal_run_id=%s ORDER BY p.pair_id", (signal_run_id,)).fetchall()
    decided = current(con, signal_run_id)
    done = {pid for pid, per in decided.items() if state_of(per)[0] == "settled"}
    info = details(con, done)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pair_id", "label_a", "paper_a", "label_b", "paper_b", "class", "conceptual_fallacy", "same_referent", "compatible_properties", "substitution_changes",
                         "theoretical_consequence", "counterexample", "rationale", "decided_by", "codebook", "name_sim", "s_def", "s_op", "s_sem_op", "c_cit", "n_nom", "retrieval_stratum"])
        n = 0
        for pid, la, ca, lb, cb, name, s_def, s_op, sem, cit, nom, hint in rows:
            if pid not in done:
                continue
            cls = state_of(decided[pid])[1]
            agreeing = [d for d in info.get(pid, []) if d["class"] == cls]
            first = (sorted(agreeing, key=lambda d: not d["final"]) or [{}])[0]
            writer.writerow([pid, la, ca, lb, cb, cls, cls in POSITIVE, first.get("same_referent"), first.get("compatible_properties"), first.get("substitution_changes"),
                             first.get("theoretical_consequence"), first.get("counterexample"), " | ".join(d["rationale"] for d in agreeing), "; ".join(sorted(decided[pid])), CODEBOOK,
                             name, s_def, s_op, sem, cit, nom, hint])
            n += 1
    return n
