"""The construct database as the agents of conditions A, B and C read it: a map of the constructs in an agent's view and the network of one construct across papers.

Read-only. Aggregates go over the papers of the agent's view and never merge anything: a label is grouped by its normalized text only so that it can be browsed, and two
papers that use the same label are not shown to mean the same construct (principle 3). Counts are counts of what the papers claim, not a sample of the world.
The functions return plain data; the forum gives ids (E#) to the records it hands over and charges them to the agent's budget.
"""
from collections import Counter

from .extraction import normalize_label

SORTS = ("papers", "links", "thin")
NOTE = ("Aggregated over the papers of your view by identical label: the same label in two papers is not shown to be one construct, and the counts are counts of what "
        "the papers claim (extracted by a model, not validated), not a sample of the world.")


def _labels(con, run, papers):
    """{label: {"papers": set, "mentions": int}} over the view."""
    out = {}
    for cid, label in con.execute("SELECT canonical_id,label_normalized FROM construct_mention WHERE extraction_run_id=%s AND canonical_id = ANY(%s::uuid[])", (run, papers)):
        entry = out.setdefault(label, {"papers": set(), "mentions": 0})
        entry["papers"].add(cid)
        entry["mentions"] += 1
    return out


def _claims(con, run, papers, label=None):
    """Claims between two different labels in the view: (claim_id, canonical_id, subject, subject surface, object, object surface, relation, direction, evidence_status)."""
    extra, params = "", [run, papers]
    if label is not None:
        extra, params = " AND (s.label_normalized=%s OR o.label_normalized=%s)", params + [label, label]
    return con.execute(
        "SELECT k.claim_id,k.canonical_id,s.label_normalized,s.surface_form,o.label_normalized,o.surface_form,k.relation,k.direction,k.evidence_status FROM nomological_claim k "
        "JOIN construct_mention s ON s.mention_id=k.subject_mention_id JOIN construct_mention o ON o.mention_id=k.object_mention_id "
        "WHERE k.extraction_run_id=%s AND k.canonical_id = ANY(%s::uuid[]) AND s.label_normalized <> o.label_normalized" + extra + " ORDER BY k.canonical_id,k.claim_id", params).fetchall()


def overview(con, run, papers, contains=None, sort="papers", limit=15):
    """The map of the constructs in the view: for each label, papers, mentions, distinct antecedents and outcomes (relation 'predicts'), other links, and how often it
    qualifies a claim of others as moderator or mediator. `sort`: papers (most used), links (most connected), thin (used in at least two papers, fewest links)."""
    if sort not in SORTS:
        raise ValueError("sort must be one of " + ", ".join(SORTS))
    labels = _labels(con, run, papers)
    links = {}
    claims = _claims(con, run, papers)
    for _id, _cid, s, _ss, o, _os, relation, _d, _e in claims:
        if relation == "predicts":
            links.setdefault(o, {}).setdefault("antecedents", set()).add(s)
            links.setdefault(s, {}).setdefault("outcomes", set()).add(o)
        else:
            links.setdefault(s, {}).setdefault("other", set()).add(o)
            links.setdefault(o, {}).setdefault("other", set()).add(s)
    as_modifier = {}
    for label, kind, n in con.execute(
            "SELECT x.label_normalized,m.kind,count(*) FROM claim_modifier m JOIN construct_mention x ON x.mention_id=m.mention_id JOIN nomological_claim k ON k.claim_id=m.claim_id "
            "WHERE k.extraction_run_id=%s AND k.canonical_id = ANY(%s::uuid[]) GROUP BY 1,2", (run, papers)):
        as_modifier.setdefault(label, {})[kind] = n
    wanted = normalize_label(contains) if contains else ""
    rows = []
    for label, entry in labels.items():
        if wanted and wanted not in label:
            continue
        mine = links.get(label, {})
        row = {"construct": label, "papers": len(entry["papers"]), "mentions": entry["mentions"], "antecedents": len(mine.get("antecedents", ())),
               "outcomes": len(mine.get("outcomes", ())), "other_links": len(mine.get("other", ())), "as_modifier": as_modifier.get(label, {})}
        row["links"] = row["antecedents"] + row["outcomes"] + row["other_links"]
        rows.append(row)
    if sort == "papers":
        rows.sort(key=lambda r: (-r["papers"], -r["links"], r["construct"]))
    elif sort == "links":
        rows.sort(key=lambda r: (-r["links"], -r["papers"], r["construct"]))
    else:
        rows = [r for r in rows if r["papers"] >= 2]
        rows.sort(key=lambda r: (r["links"], -r["papers"], r["construct"]))
    return {"constructs_in_view": len(labels), "papers_in_view": len({p for e in labels.values() for p in e["papers"]}), "claims_in_view": len(claims),
            "matching": len(rows), "sorted_by": sort, "contains": wanted or None, "rows": rows[:limit], "note": NOTE}


def closest_labels(con, run, papers, label, limit=8):
    """Labels of the view that share a word with `label`, most shared words first, then most papers: what to try when a label is not in the view."""
    words = {w for w in label.split() if len(w) > 2}
    found = []
    for other, entry in _labels(con, run, papers).items():
        shared = len(words & set(other.split()))
        if shared:
            found.append((-shared, -len(entry["papers"]), other, len(entry["papers"])))
    return [{"construct": o, "papers": n} for *_, o, n in sorted(found)[:limit]]


def network(con, run, papers, label, limit=6):
    """The network of one label across the papers of the view. Records worth citing are returned as (kind, record_id, canonical_id) references, strongest first, so that the
    forum can give them ids within the agent's budget: definitions under "definitions", one example claim for each link under "example"."""
    label = normalize_label(label)
    mentions = con.execute("SELECT mention_id,canonical_id,surface_form,role,definition_text FROM construct_mention WHERE extraction_run_id=%s AND label_normalized=%s "
                           "AND canonical_id = ANY(%s::uuid[]) ORDER BY canonical_id,mention_id", (run, label, papers)).fetchall()
    if not mentions:
        return {"construct": label, "found": False, "closest": closest_labels(con, run, papers, label)}
    ids = [m[0] for m in mentions]
    groups = {"antecedents": {}, "outcomes": {}, "other_links": {}}
    claims = _claims(con, run, papers, label)
    for claim_id, cid, s, s_surface, o, o_surface, relation, direction, status in claims:
        if s == label:
            group, other, surface, this_is = ("outcomes" if relation == "predicts" else "other_links"), o, o_surface, "subject"
        else:
            group, other, surface, this_is = ("antecedents" if relation == "predicts" else "other_links"), s, s_surface, "object"
        key = (other, relation, this_is)
        e = groups[group].setdefault(key, {"construct": other, "names": Counter(), "relation": relation, "papers": set(), "claims": 0, "directions": Counter(), "evidence": Counter(),
                                           "example": ("claim", claim_id, cid)})
        if group == "other_links":
            e["this_is"] = this_is
        e["names"][surface] += 1
        e["papers"].add(cid)
        e["claims"] += 1
        e["directions"][direction] += 1
        e["evidence"][status] += 1
    edges = {}
    for group, found in groups.items():
        rows = sorted(found.values(), key=lambda e: (-len(e["papers"]), -e["claims"], e["construct"]))
        edges[group] = [{**{k: v for k, v in e.items() if k != "names"}, "name": e["names"].most_common(1)[0][0], "papers": len(e["papers"]),
                         "directions": dict(e["directions"]), "evidence": dict(e["evidence"])} for e in rows[:limit]]
        edges[group + "_not_shown"] = max(0, len(rows) - limit)
    claim_ids = [c[0] for c in claims]
    on = {c[0]: f"{c[2]} -> {c[4]}" for c in claims}
    qualifiers = Counter()
    for claim_id, kind, value, other in con.execute(
            "SELECT m.claim_id,m.kind,m.value,x.label_normalized FROM claim_modifier m LEFT JOIN construct_mention x ON x.mention_id=m.mention_id WHERE m.claim_id = ANY(%s)", (claim_ids,)):
        qualifiers[(kind, other or value, on[claim_id])] += 1
    as_modifier = con.execute(
        "SELECT m.kind,s.label_normalized,o.label_normalized,count(*),count(DISTINCT k.canonical_id),"
        "(array_agg(k.claim_id ORDER BY k.canonical_id,k.claim_id))[1],(array_agg(k.canonical_id ORDER BY k.canonical_id,k.claim_id))[1] FROM claim_modifier m "
        "JOIN construct_mention x ON x.mention_id=m.mention_id JOIN nomological_claim k ON k.claim_id=m.claim_id "
        "JOIN construct_mention s ON s.mention_id=k.subject_mention_id JOIN construct_mention o ON o.mention_id=k.object_mention_id "
        "WHERE x.label_normalized=%s AND k.extraction_run_id=%s AND k.canonical_id = ANY(%s::uuid[]) GROUP BY 1,2,3 ORDER BY 5 DESC,4 DESC,2,3 LIMIT %s", (label, run, papers, limit)).fetchall()
    defined = [m for m in mentions if m[4]]
    measures = con.execute("SELECT count(*) FROM construct_measure WHERE mention_id = ANY(%s)", (ids,)).fetchone()[0]
    return {"construct": label, "found": True, "name": Counter(m[2] for m in mentions).most_common(1)[0][0], "papers": len({m[1] for m in mentions}), "mentions": len(mentions),
            "roles": dict(Counter(m[3] for m in mentions)), "with_definition": len(defined), "measures": measures,
            "definitions": [("mention", m[0], m[1]) for m in defined[:2]], **edges,
            "qualifiers_of_its_claims": [{"kind": k, "construct_or_value": v, "on": o, "claims": n} for (k, v, o), n in qualifiers.most_common(limit)],
            "as_modifier": [{"kind": k, "of": f"{s} -> {o}", "claims": n, "papers": p, "example": ("claim", claim, cid)} for k, s, o, n, p, claim, cid in as_modifier],
            "note": NOTE}
