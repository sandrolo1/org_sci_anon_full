"""Read-only browser over the constructs extracted in Phase 5, for the dashboard's "Constructs" tab.

It answers four questions with the records already in the database and writes nothing:
  * what is there (`meta`): the extraction run, how much of the corpus it covers, where the CTO/TIM labels come from;
  * which constructs match a set of filters (`search`): community, role, year, journal, text, whether a definition / measure / claim exists;
  * what one label looks like across papers (`detail`): every use with its quote and page, measures, attributes, claims, and the
    antecedents / outcomes it is linked to;
  * what one paper contains (`paper`);
  * how the labels are linked by the papers' claims (`network`): the most connected (or most used) labels, or the neighbourhood of the labels a search names.

Two rules shape it. Constructs are grouped by the *identical normalised label* only to make them browsable: two papers using the same label
are not thereby using the same construct, and two labels are not thereby different (that is the human decision of Phase 6-7, never made
here). And CTO and TIM are two independent memberships of a paper (yes / no / uncertain / unlabelled), never one category.

Where the community labels come from, in this order: the latest pre-assignment run of the catalog run that is not a v1 suggestion; else the latest
v1 suggestion run in the database; else the collected Batch v1 file (`assignments.csv`), read in place. Whatever the source, the labels are
the pipeline's, not validated membership; the human ratings of the blind sample (where they exist) are shown next to them, never merged in.
"""
import threading
from collections import Counter
from typing import NamedTuple

from . import fulltext
from .community_dashboard import run_dir
from .community_recovery import assignment_rows
from .community_v2 import paper_fields
from .extraction import ATTRIBUTE_KINDS, RELATIONS, ROLES, normalize_label, validation_state

COMMUNITIES = ("CTO", "TIM")
STATUSES = ("yes", "no", "uncertain", "unlabelled")
SORTS = ("papers", "mentions", "label", "year", "claims", "antecedents", "outcomes")
MAX_PAGE = 100
USAGE_PAGE = 25
RELATED_LIMIT = 15
EDGE_LIMIT = 40
NETWORK_RANKS = ("links", "papers")
NETWORK_NODES = 200
NETWORK_MAX = 400
NOTES = (
    "Constructs are grouped by identical normalised label so that they can be browsed. The same label in two papers is not evidence of the same construct, "
    "and different labels are not evidence of different constructs: that is decided by people (Phases 6-7).",
    "Every record is a proposal extracted by a language model from the paper's full text, with a verbatim quote and page. None of it is validated unless a "
    "validation state says so.",
    "Only papers whose PDF was obtained and parsed appear here. A paper that is missing is unavailable, not irrelevant.",
)


class Mention(NamedTuple):
    mention_id: object
    canonical_id: object
    label: str          # label_normalized
    surface: str
    role: str
    definition: str     # '' when the paper gives none
    definition_norm: str
    measures: int
    claims: int


class Claim(NamedTuple):
    canonical_id: object
    subject: str        # label_normalized of the subject mention
    object: str
    relation: str
    direction: str
    evidence: str


class Index(NamedTuple):
    signature: tuple
    run: dict
    mentions: list
    by_label: dict       # label -> [Mention]
    papers: dict         # canonical_id -> {title, authors, year, source, doi}
    member: dict         # canonical_id -> {"CTO": label, "TIM": label}
    membership: dict     # where the labels come from
    human: dict          # canonical_id -> {"CTO": 0/1/None, "TIM": ..., "basis": str}
    coverage: dict
    claims: list         # [Claim] between two different labels, for the network


_CACHE = {}
_LOCK = threading.Lock()


# --- Index ---------------------------------------------------------------------------------------------------------------------

def latest_extraction_run(con, run_id):
    row = con.execute("SELECT e.extraction_run_id,e.model,e.prompt_version,e.schema_version,e.created_at,i.ingestion_run_id FROM extraction_run e "
                      "JOIN ingestion_run i USING (ingestion_run_id) WHERE i.run_id=%s AND NOT e.manifest ? 'delta_of' ORDER BY e.created_at DESC LIMIT 1", (run_id,)).fetchone()   # a delta run (extract-promote --delta) only adds rows to its parent
    if not row:
        raise ValueError("No extraction run yet (research0 extract-prepare, then extract-collect)")
    return {"extraction_run_id": row[0], "model": row[1], "prompt_version": row[2], "schema_version": row[3], "created_at": row[4].isoformat(timespec="seconds"),
            "ingestion_run_id": row[5]}


def _file_stamp(run_id):
    directory = run_dir(run_id)
    paths = [directory / "assignments.csv", *directory.glob("retry_[0-9][0-9][0-9]/assignments.csv")]
    return tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())


def _signature(con, run_id, extraction_run_id):
    mentions = con.execute("SELECT count(*),max(created_at) FROM construct_mention WHERE extraction_run_id=%s", (extraction_run_id,)).fetchone()
    claims = con.execute("SELECT count(*) FROM nomological_claim WHERE extraction_run_id=%s", (extraction_run_id,)).fetchone()
    pre = con.execute("SELECT count(*),max(created_at) FROM preassignment_run WHERE run_id=%s", (run_id,)).fetchone()
    human = con.execute("SELECT (SELECT count(*) FROM blind_rating WHERE run_id=%s),(SELECT count(*) FROM community_adjudication WHERE run_id=%s)", (run_id, run_id)).fetchone()
    return (str(extraction_run_id), *mentions, *claims, *pre, *human, _file_stamp(run_id))


def _membership(con, run_id, cids):
    """({canonical_id: {"CTO": label, "TIM": label}}, description of the source)."""
    for v1 in (False, True):
        row = con.execute(f"SELECT preassignment_run_id,source,model,prompt_version,scope_version,created_at FROM preassignment_run WHERE run_id=%s AND source {'=' if v1 else '<>'} 'v1_suggestion' "
                          "ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
        if not row:
            continue
        member = {}
        for cid, community, label in con.execute("SELECT canonical_id,community,label FROM community_preassignment WHERE preassignment_run_id=%s AND canonical_id = ANY(%s)", (row[0], list(cids))):
            member.setdefault(cid, {})[community] = label
        kind = "v1_suggestion" if v1 else row[1]
        text = ("Batch v1 suggestions (binary yes/no from title, abstract and keywords; not validated)" if v1 else
                f"Pre-assignment {row[3]} ({row[2]}, codebook {row[4]}): the pipeline's label for people to audit, not a membership decision")
        return member, {"kind": kind, "description": text, "validated": False, "preassignment_run_id": str(row[0]), "created_at": row[5].isoformat(timespec="seconds")}
    try:
        labels = assignment_rows(run_dir(run_id))
    except (OSError, ValueError, KeyError):
        labels = {}
    mine = {cid: labels[str(cid)] for cid in cids if str(cid) in labels}
    if not mine:
        return {}, {"kind": "none", "description": "No CTO/TIM label is available for these papers (collect the Batch or run the pre-assignment).", "validated": False}
    # same rule as community_v2.import_v1_suggestions: a paper without a usable abstract gets no trusted binary label
    insufficient = set()
    for cid, title, abstract, keywords, index_keywords in con.execute(
            "SELECT c.canonical_id,c.title,r.raw_fields->>'Abstract',r.raw_fields->>'Author Keywords',r.raw_fields->>'Index Keywords' FROM canonical_paper c "
            "JOIN export_record r ON r.record_id=c.representative_record_id WHERE c.canonical_id = ANY(%s)", (list(mine),)):
        if paper_fields(title, abstract, keywords, index_keywords)[1] != "abstract":
            insufficient.add(cid)
    member = {cid: {community: ("uncertain" if cid in insufficient else "yes" if row[community.lower()] == "1" else "no") for community in COMMUNITIES} for cid, row in mine.items()}
    return member, {"kind": "v1_suggestion_file", "validated": False,
                    "description": "Batch v1 suggestions read from the collected Batch file (binary yes/no from title, abstract and keywords; not in the database; not validated)"}


def _human(con, run_id, cids):
    """What people said about a paper's membership in the blind sample: {cid: {"CTO": 0/1/None, "TIM": ..., "basis": ...}}. Display only, never mixed into the labels."""
    cids = list(cids)
    ratings, final = {}, {}
    for cid, validator, cto, tim in con.execute("SELECT DISTINCT ON (validator,canonical_id) canonical_id,validator,cto,tim FROM blind_rating WHERE run_id=%s AND canonical_id = ANY(%s) "
                                                "ORDER BY validator,canonical_id,rated_at DESC,rating_id DESC", (run_id, cids)):
        ratings.setdefault(cid, []).append((cto, tim))
    for cid, cto, tim in con.execute("SELECT DISTINCT ON (canonical_id) canonical_id,cto,tim FROM community_adjudication WHERE run_id=%s AND canonical_id = ANY(%s) "
                                     "ORDER BY canonical_id,decided_at DESC,adjudication_id DESC", (run_id, cids)):
        final[cid] = (cto, tim)
    out = {}
    for cid in set(ratings) | set(final):
        given = ratings.get(cid, [])
        if cid in final:
            out[cid] = {"CTO": final[cid][0], "TIM": final[cid][1], "basis": "decided by an adjudicator"}
            continue
        values = [{c: v[i] for i, c in enumerate(COMMUNITIES)} for v in given]
        agreed = {c: (values[0][c] if len({v[c] for v in values}) == 1 else None) for c in COMMUNITIES}
        basis = "one annotator" if len(given) == 1 else "annotators agree" if None not in agreed.values() else "annotators disagree (not adjudicated)"
        out[cid] = {**agreed, "basis": basis}
    return out


def build_index(con, run_id):
    run = latest_extraction_run(con, run_id)
    extraction_run_id = run["extraction_run_id"]
    # Counts are grouped once, not computed per mention: there is no index on the claim's mention columns and a correlated count took ~14 s on 18k mentions.
    mention_rows = con.execute(
        "SELECT m.mention_id,m.canonical_id,m.label_normalized,m.surface_form,m.role,coalesce(m.definition_text,''),coalesce(ms.n,0),coalesce(cl.n,0) FROM construct_mention m "
        "LEFT JOIN (SELECT mention_id,count(*) n FROM construct_measure GROUP BY mention_id) ms ON ms.mention_id=m.mention_id "
        "LEFT JOIN (SELECT mid,count(*) n FROM (SELECT subject_mention_id mid FROM nomological_claim WHERE extraction_run_id=%(run)s "
        "UNION ALL SELECT object_mention_id FROM nomological_claim WHERE extraction_run_id=%(run)s) u GROUP BY mid) cl ON cl.mid=m.mention_id "
        "WHERE m.extraction_run_id=%(run)s ORDER BY m.canonical_id,m.char_start", {"run": extraction_run_id}).fetchall()
    mentions = [Mention(mid, cid, label, surface, role, definition, normalize_label(definition), measures, claims) for mid, cid, label, surface, role, definition, measures, claims in mention_rows]
    by_label = {}
    for m in mentions:
        by_label.setdefault(m.label, []).append(m)
    cids = sorted({m.canonical_id for m in mentions}, key=str)
    papers = {cid: {"title": title, "authors": authors, "year": year, "source": source, "doi": doi} for cid, title, authors, year, source, doi in con.execute(
        "SELECT canonical_id,title,authors,publication_year,source_title,coalesce(doi,all_dois[1]) FROM canonical_paper WHERE canonical_id = ANY(%s)", (cids,))}
    member, membership = _membership(con, run_id, cids)
    counts = {c: Counter(member.get(cid, {}).get(c, "unlabelled") for cid in cids) for c in COMMUNITIES}
    membership["counts"] = {c: {s: counts[c].get(s, 0) for s in STATUSES} for c in COMMUNITIES}
    eligible = fulltext.totals(con, run_id)["eligible"]
    parsed = con.execute("SELECT count(DISTINCT v.canonical_id) FROM paper_version v WHERE v.ingestion_run_id=%s AND v.status='parsed'", (run["ingestion_run_id"],)).fetchone()[0]
    coverage = {"eligible_papers": eligible, "parsed_papers": parsed, "papers_with_constructs": len(cids), "mentions": len(mentions), "labels": len(by_label)}
    claims = [Claim(*row) for row in con.execute(
        "SELECT k.canonical_id,s.label_normalized,o.label_normalized,k.relation,k.direction,k.evidence_status FROM nomological_claim k "
        "JOIN construct_mention s ON s.mention_id=k.subject_mention_id JOIN construct_mention o ON o.mention_id=k.object_mention_id "
        "WHERE k.extraction_run_id=%s AND s.label_normalized <> o.label_normalized ORDER BY k.claim_id", (extraction_run_id,))]
    coverage["labels_shown"] = len({label for c in claims if c.relation == "predicts" for label in (c.subject, c.object)})   # with an antecedent or an outcome
    return Index(_signature(con, run_id, extraction_run_id), run, mentions, by_label, papers, member, membership, _human(con, run_id, cids), coverage, claims)


def get_index(con, run_id):
    """The index is rebuilt only when the extraction run, the labels or the human ratings changed (all of it is append-only)."""
    with _LOCK:
        run = latest_extraction_run(con, run_id)
        signature = _signature(con, run_id, run["extraction_run_id"])
        cached = _CACHE.get(run_id)
        if cached and cached.signature == signature:
            return cached
        index = build_index(con, run_id)
        _CACHE[run_id] = index
        return index


# --- Parameters ----------------------------------------------------------------------------------------------------------------

def _one(query, key, default=""):
    return ((query.get(key) or [default])[0] or default).strip()


def _int(query, key, default=None, low=None, high=None):
    raw = _one(query, key)
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a whole number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{key} is out of range")
    return value


def _status(query, key):
    value = _one(query, key, "any").lower()
    if value not in ("any", *STATUSES):
        raise ValueError(f"{key} must be one of: any, " + ", ".join(STATUSES))
    return value


def parse_filters(query):
    roles = [r for r in _one(query, "roles").split(",") if r]
    if any(r not in ROLES for r in roles):
        raise ValueError("Unknown role; roles are: " + ", ".join(ROLES))
    sort = _one(query, "sort", "papers")
    if sort not in SORTS:
        raise ValueError("sort must be one of: " + ", ".join(SORTS))
    spans = _one(query, "spans", "any")
    if spans not in ("any", "both"):
        raise ValueError("spans must be any or both")
    return {"terms": normalize_label(_one(query, "q")).split(), "in_definitions": _one(query, "defs") == "1", "CTO": _status(query, "cto"), "TIM": _status(query, "tim"),
            "roles": set(roles), "year_from": _int(query, "year_from"), "year_to": _int(query, "year_to"), "source": _one(query, "source"),
            "has_definition": _one(query, "has_definition") == "1", "has_measure": _one(query, "has_measure") == "1", "has_claim": _one(query, "has_claim") == "1",
            "min_papers": _int(query, "min_papers", 1, low=1), "spans": spans, "sort": sort, "descending": _one(query, "dir", "desc") != "asc"}


def status_of(index, cid, community):
    return index.member.get(cid, {}).get(community) or "unlabelled"


def membership_group(index, cid):
    """CTO and TIM are independent; this only names the combinations for the summary bar. 'open' = at least one side is uncertain or unlabelled without a clear yes/no pair."""
    cto, tim = status_of(index, cid, "CTO"), status_of(index, cid, "TIM")
    if cto == "yes" and tim == "yes":
        return "both"
    if cto == "yes" and tim == "no":
        return "cto_only"
    if tim == "yes" and cto == "no":
        return "tim_only"
    if cto == "no" and tim == "no":
        return "neither"
    return "open"


# --- Search --------------------------------------------------------------------------------------------------------------------

def _paper_matches(index, cid, f):
    paper = index.papers.get(cid) or {}
    year = paper.get("year")
    if f["year_from"] is not None and (year is None or year < f["year_from"]):
        return False
    if f["year_to"] is not None and (year is None or year > f["year_to"]):
        return False
    if f["source"] and paper.get("source") != f["source"]:
        return False
    return all(f[c] == "any" or status_of(index, cid, c) == f[c] for c in COMMUNITIES)


def _matches(index, m, f):
    if not _paper_matches(index, m.canonical_id, f):
        return False
    if f["roles"] and m.role not in f["roles"]:
        return False
    if (f["has_definition"] and not m.definition) or (f["has_measure"] and not m.measures) or (f["has_claim"] and not m.claims):
        return False
    if f["terms"]:
        in_label = all(t in m.label for t in f["terms"])
        in_definition = f["in_definitions"] and all(t in m.definition_norm for t in f["terms"])
        if not (in_label or in_definition):
            return False
    return True


def _predicts(index, groups):
    """Distinct antecedent / outcome labels of each label in `groups` (relation 'predicts', as in forum_constructs.overview), counted only in the papers where
    the label matched the filters. A label with neither is left out of the list, its counts and the network (the user's request of 14 Oct 2026)."""
    linked = {label: {"antecedents": set(), "outcomes": set()} for label in groups}
    for c in index.claims:
        if c.relation != "predicts":
            continue
        if c.object in groups and c.canonical_id in groups[c.object]["papers"]:
            linked[c.object]["antecedents"].add(c.subject)
        if c.subject in groups and c.canonical_id in groups[c.subject]["papers"]:
            linked[c.subject]["outcomes"].add(c.object)
    return linked


def search(index, query):
    f = parse_filters(query)
    offset, limit = _int(query, "offset", 0, low=0), _int(query, "limit", 50, low=1, high=MAX_PAGE)
    groups = {}
    for m in index.mentions:
        if not _matches(index, m, f):
            continue
        g = groups.setdefault(m.label, {"papers": set(), "cto": set(), "tim": set(), "both": set(), "roles": Counter(), "surfaces": Counter(), "mentions": 0,
                                        "definitions": 0, "measures": 0, "claims": 0, "years": []})
        g["mentions"] += 1
        g["papers"].add(m.canonical_id)
        g["roles"][m.role] += 1
        g["surfaces"][m.surface] += 1
        g["definitions"] += bool(m.definition)
        g["measures"] += m.measures
        g["claims"] += m.claims
        year = (index.papers.get(m.canonical_id) or {}).get("year")
        if year:
            g["years"].append(year)
        cto, tim = status_of(index, m.canonical_id, "CTO") == "yes", status_of(index, m.canonical_id, "TIM") == "yes"
        if cto:
            g["cto"].add(m.canonical_id)
        if tim:
            g["tim"].add(m.canonical_id)
        if cto and tim:
            g["both"].add(m.canonical_id)
    linked = _predicts(index, groups)
    rows = []
    for label, g in groups.items():
        if len(g["papers"]) < f["min_papers"] or (f["spans"] == "both" and not (g["cto"] and g["tim"])):
            continue
        if not (linked[label]["antecedents"] or linked[label]["outcomes"]):
            continue
        rows.append({"label": label, "name": g["surfaces"].most_common(1)[0][0], "variants": len({s.casefold() for s in g["surfaces"]}), "papers": len(g["papers"]), "mentions": g["mentions"],
                     "cto_papers": len(g["cto"]), "tim_papers": len(g["tim"]), "both_papers": len(g["both"]), "spans_both": bool(g["cto"] and g["tim"]),
                     "roles": dict(g["roles"]), "with_definition": g["definitions"], "measures": g["measures"], "claims": g["claims"],
                     "antecedents": len(linked[label]["antecedents"]), "outcomes": len(linked[label]["outcomes"]),
                     "year_min": min(g["years"], default=None), "year_max": max(g["years"], default=None)})
    key = {"papers": lambda r: (r["papers"], r["mentions"]), "mentions": lambda r: (r["mentions"], r["papers"]), "label": lambda r: r["label"],
           "year": lambda r: (r["year_max"] or 0, r["papers"]), "claims": lambda r: (r["claims"], r["papers"]),
           "antecedents": lambda r: (r["antecedents"], r["papers"]), "outcomes": lambda r: (r["outcomes"], r["papers"])}[f["sort"]]
    rows.sort(key=lambda r: r["label"])      # a stable, predictable order inside ties
    rows.sort(key=key, reverse=f["descending"] and f["sort"] != "label")
    return {"filters": {k: (sorted(v) if isinstance(v, set) else v) for k, v in f.items()}, "total": len(rows), "offset": offset, "rows": rows[offset:offset + limit],
            "summary": _summary(index, rows, groups)}


def _summary(index, rows, groups):
    kept = {r["label"] for r in rows}
    papers, roles, mentions = set(), Counter(), 0
    for label in kept:
        g = groups[label]
        papers |= g["papers"]
        roles.update(g["roles"])
        mentions += g["mentions"]
    membership = Counter(membership_group(index, cid) for cid in papers)
    return {"labels": len(kept), "papers": len(papers), "mentions": mentions, "roles": {r: roles.get(r, 0) for r in ROLES},
            "membership": {k: membership.get(k, 0) for k in ("both", "cto_only", "tim_only", "neither", "open")}}


# --- Meta ----------------------------------------------------------------------------------------------------------------------

def meta(index):
    roles = Counter(m.role for m in index.mentions)
    years = Counter(p["year"] for p in index.papers.values() if p["year"])
    sources = Counter(p["source"] for p in index.papers.values() if p["source"])
    run = dict(index.run, extraction_run_id=str(index.run["extraction_run_id"]), ingestion_run_id=str(index.run["ingestion_run_id"]))
    groups = Counter(membership_group(index, cid) for cid in index.papers)
    membership = dict(index.membership, groups={k: groups.get(k, 0) for k in ("both", "cto_only", "tim_only", "neither", "open")})
    return {"run": run, "coverage": index.coverage, "membership": membership, "notes": list(NOTES),
            "roles": [{"role": r, "mentions": roles.get(r, 0)} for r in ROLES],
            "years": [{"year": y, "papers": n} for y, n in sorted(years.items())],
            "sources": [{"source": s, "papers": n} for s, n in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0]))],
            "human_rated_papers": len(index.human)}


# --- Detail --------------------------------------------------------------------------------------------------------------------

def _paper_card(index, cid):
    paper = index.papers.get(cid) or {}
    return {"id": str(cid), "title": paper.get("title"), "authors": paper.get("authors"), "year": paper.get("year"), "source": paper.get("source"), "doi": paper.get("doi"),
            "membership": {c: status_of(index, cid, c) for c in COMMUNITIES}, "human": index.human.get(cid)}


def _claims_for(con, mention_ids):
    return con.execute(
        "SELECT c.claim_id,c.canonical_id,c.relation,c.direction,c.evidence_status,c.confidence,c.quote,c.page,s.mention_id,s.label_normalized,s.surface_form,o.mention_id,o.label_normalized,o.surface_form "
        "FROM nomological_claim c JOIN construct_mention s ON s.mention_id=c.subject_mention_id JOIN construct_mention o ON o.mention_id=c.object_mention_id "
        "WHERE c.subject_mention_id = ANY(%(ids)s) OR c.object_mention_id = ANY(%(ids)s) ORDER BY c.canonical_id,c.page,c.claim_id", {"ids": mention_ids}).fetchall()


def _claim_dict(row, modifiers):
    claim_id, cid, relation, direction, status, confidence, quote, page, s_id, s_label, s_surface, o_id, o_label, o_surface = row
    return {"claim_id": str(claim_id), "paper_id": str(cid), "relation": relation, "direction": direction, "evidence_status": status, "confidence": confidence, "quote": quote, "page": page,
            "subject": {"label": s_label, "name": s_surface}, "object": {"label": o_label, "name": o_surface}, "modifiers": modifiers.get(claim_id, [])}


def _modifiers(con, claim_ids):
    out = {}
    for claim_id, kind, value, quote, label in con.execute(
            "SELECT m.claim_id,m.kind,m.value,m.quote,x.label_normalized FROM claim_modifier m LEFT JOIN construct_mention x ON x.mention_id=m.mention_id WHERE m.claim_id = ANY(%s) ORDER BY m.claim_id,m.kind,m.value",
            (claim_ids,)):
        out.setdefault(claim_id, []).append({"kind": kind, "value": value, "quote": quote, "label": label})
    return out


def _edges(label, claim_rows, index):
    """The neighbourhood of `label` over all its papers: who is linked to it, how, in how many papers and with which sign. Aggregated, never merged."""
    edges = {}
    for row in claim_rows:
        claim_id, cid, relation, direction, status, _confidence, _quote, _page, _s_id, s_label, s_surface, _o_id, o_label, o_surface = row
        if s_label == o_label:
            continue
        side, other, name = ("out", o_label, o_surface) if s_label == label else ("in", s_label, s_surface)
        e = edges.setdefault((side, other, relation), {"side": side, "label": other, "name": Counter(), "relation": relation, "papers": set(), "claims": 0, "directions": Counter(), "evidence": Counter()})
        e["name"][name] += 1
        e["papers"].add(cid)
        e["claims"] += 1
        e["directions"][direction] += 1
        e["evidence"][status] += 1
    out = [{"side": e["side"], "label": e["label"], "name": e["name"].most_common(1)[0][0], "relation": e["relation"], "papers": len(e["papers"]), "claims": e["claims"],
            "directions": dict(e["directions"]), "evidence": dict(e["evidence"]), "mentions_in_corpus": len(index.by_label.get(e["label"], []))} for e in edges.values()]
    out.sort(key=lambda e: (-e["papers"], -e["claims"], e["label"]))
    return out


def _related(index, label):
    padded = f" {label} "
    found = []
    for other, items in index.by_label.items():
        if other == label:
            continue
        if padded in f" {other} ":
            how = "contains this label"
        elif f" {other} " in padded:
            how = "is contained in this label"
        else:
            continue
        found.append({"label": other, "name": Counter(m.surface for m in items).most_common(1)[0][0], "papers": len({m.canonical_id for m in items}), "how": how})
    found.sort(key=lambda r: (-r["papers"], r["label"]))
    return found[:RELATED_LIMIT]


def detail(con, index, query):
    label = normalize_label(_one(query, "label"))
    mentions = index.by_label.get(label)
    if not mentions:
        raise ValueError("Unknown construct label")
    offset, limit = _int(query, "offset", 0, low=0), _int(query, "limit", USAGE_PAGE, low=1, high=MAX_PAGE)
    ids = [m.mention_id for m in mentions]
    claim_rows = _claims_for(con, ids)
    by_paper = {}
    for m in mentions:
        by_paper.setdefault(m.canonical_id, []).append(m)

    def order(cid):
        paper = index.papers.get(cid) or {}
        return (-(paper.get("year") or 0), (paper.get("title") or ""), str(cid))
    ordered = sorted(by_paper, key=order)
    page = ordered[offset:offset + limit]
    page_ids = [m.mention_id for cid in page for m in by_paper[cid]]
    full = {r[0]: r for r in con.execute("SELECT mention_id,surface_form,role,definition_text,quote,page FROM construct_mention WHERE mention_id = ANY(%s)", (page_ids,))}
    measures, attributes = {}, {}
    for mention_id, name, items, scale, quote, page_no in con.execute("SELECT mention_id,name,items_text,scale,quote,page FROM construct_measure WHERE mention_id = ANY(%s) ORDER BY name", (page_ids,)):
        measures.setdefault(mention_id, []).append({"name": name, "items": items, "scale": scale, "quote": quote, "page": page_no})
    for mention_id, kind, basis, value, quote, page_no in con.execute("SELECT mention_id,kind,basis,value,quote,page FROM extraction_attribute WHERE mention_id = ANY(%s) ORDER BY kind,basis", (page_ids,)):
        attributes.setdefault(mention_id, []).append({"kind": kind, "basis": basis, "value": value, "quote": quote, "page": page_no})
    page_claims = [r for r in claim_rows if r[1] in set(page)]
    modifiers = _modifiers(con, [r[0] for r in page_claims])
    claims_by_paper = {}
    for row in page_claims:
        claims_by_paper.setdefault(row[1], []).append(_claim_dict(row, modifiers))
    states = validation_state(con, "construct_mention", page_ids)
    usages = []
    for cid in page:
        uses = []
        for m in by_paper[cid]:
            _id, surface, role, definition, quote, page_no = full[m.mention_id]
            uses.append({"mention_id": str(m.mention_id), "surface": surface, "role": role, "definition": definition, "quote": quote, "page": page_no, "measures": measures.get(m.mention_id, []),
                         "attributes": attributes.get(m.mention_id, []), "validation": states.get(m.mention_id, {})})
        usages.append({"paper": _paper_card(index, cid), "mentions": uses, "claims": claims_by_paper.get(cid, [])})
    as_modifier = con.execute(
        "SELECT m.kind,count(*),count(DISTINCT c.canonical_id) FROM claim_modifier m JOIN nomological_claim c USING (claim_id) WHERE m.mention_id = ANY(%s) GROUP BY m.kind ORDER BY m.kind", (ids,)).fetchall()
    surfaces = Counter(m.surface for m in mentions)
    cto = {cid for cid in by_paper if status_of(index, cid, "CTO") == "yes"}
    tim = {cid for cid in by_paper if status_of(index, cid, "TIM") == "yes"}
    edges = _edges(label, claim_rows, index)
    return {"label": label, "name": surfaces.most_common(1)[0][0], "surface_forms": [{"form": f, "mentions": n} for f, n in surfaces.most_common(12)],
            "papers": len(by_paper), "mentions": len(mentions), "cto_papers": len(cto), "tim_papers": len(tim), "both_papers": len(cto & tim),
            "roles": dict(Counter(m.role for m in mentions)), "with_definition": sum(bool(m.definition) for m in mentions), "measures": sum(m.measures for m in mentions),
            "claims": len(claim_rows), "antecedents": sum(e["side"] == "in" and e["relation"] == "predicts" for e in edges),
            "outcomes": sum(e["side"] == "out" and e["relation"] == "predicts" for e in edges), "as_modifier": [{"kind": k, "claims": n, "papers": p} for k, n, p in as_modifier],
            "edges": edges[:EDGE_LIMIT], "related": _related(index, label),
            "usages": usages, "usage_offset": offset, "usage_total": len(ordered), "notes": [NOTES[0]]}


# --- One paper -----------------------------------------------------------------------------------------------------------------

def paper(con, index, query):
    raw = _one(query, "id")
    cid = next((c for c in index.papers if str(c) == raw), None)
    if cid is None:
        raise ValueError("This paper has no extracted constructs in the latest extraction run")
    rows = con.execute("SELECT mention_id,surface_form,label_normalized,role,definition_text,quote,page FROM construct_mention WHERE extraction_run_id=%s AND canonical_id=%s ORDER BY char_start,mention_id",
                       (index.run["extraction_run_id"], cid)).fetchall()
    ids = [r[0] for r in rows]
    measures, attributes = {}, {}
    for mention_id, name, items, scale, quote, page_no in con.execute("SELECT mention_id,name,items_text,scale,quote,page FROM construct_measure WHERE mention_id = ANY(%s) ORDER BY name", (ids,)):
        measures.setdefault(mention_id, []).append({"name": name, "items": items, "scale": scale, "quote": quote, "page": page_no})
    for mention_id, kind, basis, value, quote, page_no in con.execute("SELECT mention_id,kind,basis,value,quote,page FROM extraction_attribute WHERE mention_id = ANY(%s) AND basis <> 'missing' ORDER BY kind", (ids,)):
        attributes.setdefault(mention_id, []).append({"kind": kind, "basis": basis, "value": value, "quote": quote, "page": page_no})
    claim_rows = _claims_for(con, ids)
    modifiers = _modifiers(con, [r[0] for r in claim_rows])
    states = validation_state(con, "construct_mention", ids)
    in_corpus = {label: len({m.canonical_id for m in index.by_label[label]}) for label in {r[2] for r in rows}}
    return {"paper": _paper_card(index, cid),
            "mentions": [{"mention_id": str(mid), "surface": surface, "label": label, "role": role, "definition": definition, "quote": quote, "page": page_no,
                          "papers_with_label": in_corpus[label], "measures": measures.get(mid, []), "attributes": attributes.get(mid, []), "validation": states.get(mid, {})}
                         for mid, surface, label, role, definition, quote, page_no in rows],
            "claims": [_claim_dict(r, modifiers) for r in claim_rows], "attribute_kinds": list(ATTRIBUTE_KINDS)}


# --- Network -------------------------------------------------------------------------------------------------------------------

def _community(g):
    """Which community's papers mostly use the label (yes labels only): cto or tim when at least 3 in 4 of the CTO-or-TIM papers are of that side, mixed otherwise,
    open when no paper is labelled yes for either. For colouring only: one CTO paper among forty TIM ones would otherwise make a label look shared."""
    cto, tim = len(g["cto"]), len(g["tim"])
    if not cto and not tim:
        return "open"
    share = cto / (cto + tim)
    return "cto" if share >= 0.75 else "tim" if share <= 0.25 else "mixed"


def network(index, query):
    """Labels as nodes, the papers' claims between two labels as links, aggregated over the papers that pass the filters (never merged).

    The filters of the list apply here too: the paper filters (community, year, journal) to both the mentions and the claims, the mention filters (role, definition,
    measure, claim) and the label filters (minimum papers, both communities) to the nodes. The text search does not cut the network: it names the labels at the centre,
    shown with their neighbours up to `hops` links away. Without a search (or `focus`), the `max_nodes` most connected (`rank=links`) or most used (`rank=papers`)
    labels are shown with the links among them; labels without a link among them are left out unless `isolated=1`. As in the list, a label with no antecedent
    and no outcome (relation 'predicts') under the filters is never a node."""
    f = parse_filters(query)
    relations = [r for r in _one(query, "relations").split(",") if r] or list(RELATIONS)
    if any(r not in RELATIONS for r in relations):
        raise ValueError("Unknown relation; relations are: " + ", ".join(RELATIONS))
    rank = _one(query, "rank", "links")
    if rank not in NETWORK_RANKS:
        raise ValueError("rank must be one of: " + ", ".join(NETWORK_RANKS))
    hops = _int(query, "hops", 1, low=1, high=2)
    max_nodes = _int(query, "max_nodes", NETWORK_NODES, low=10, high=NETWORK_MAX)
    min_link = _int(query, "min_link_papers", 1, low=1)
    isolated = _one(query, "isolated") == "1"
    focus = normalize_label(_one(query, "focus"))
    terms, f["terms"] = f["terms"], []

    groups = {}
    for m in index.mentions:
        if not _matches(index, m, f):
            continue
        g = groups.setdefault(m.label, {"papers": set(), "cto": set(), "tim": set(), "surfaces": Counter(), "roles": Counter(), "mentions": 0, "seed": False})
        g["papers"].add(m.canonical_id)
        g["surfaces"][m.surface] += 1
        g["roles"][m.role] += 1
        g["mentions"] += 1
        if status_of(index, m.canonical_id, "CTO") == "yes":
            g["cto"].add(m.canonical_id)
        if status_of(index, m.canonical_id, "TIM") == "yes":
            g["tim"].add(m.canonical_id)
        if terms and (all(t in m.label for t in terms) or (f["in_definitions"] and all(t in m.definition_norm for t in terms))):
            g["seed"] = True
    linked = _predicts(index, groups)
    nodes = {label: g for label, g in groups.items() if len(g["papers"]) >= f["min_papers"] and not (f["spans"] == "both" and not (g["cto"] and g["tim"]))
             and (linked[label]["antecedents"] or linked[label]["outcomes"])}

    edges = {}
    for c in index.claims:
        if c.relation in relations and c.subject in nodes and c.object in nodes and _paper_matches(index, c.canonical_id, f):
            e = edges.setdefault((c.subject, c.object, c.relation), {"papers": set(), "claims": 0, "directions": Counter(), "evidence": Counter()})
            e["papers"].add(c.canonical_id)
            e["claims"] += 1
            e["directions"][c.direction] += 1
            e["evidence"][c.evidence] += 1
    edges = {k: e for k, e in edges.items() if len(e["papers"]) >= min_link}
    neighbours = {}
    for s, o, _relation in edges:
        neighbours.setdefault(s, set()).add(o)
        neighbours.setdefault(o, set()).add(s)

    def prominence(label):
        papers, links = len(nodes[label]["papers"]), len(neighbours.get(label, ()))
        return (-links, -papers, label) if rank == "links" else (-papers, -links, label)

    distance, not_shown = {}, 0
    if focus and focus not in nodes:
        raise ValueError("This construct is not in the network with these filters; loosen them or clear the centre")
    seeds = [focus] if focus else sorted((label for label, g in nodes.items() if g["seed"]), key=prominence) if terms else []
    if focus or terms:
        mode = "focus" if focus else "search"
        frontier = seeds[:max(1, max_nodes // 3)]           # a broad search leaves room for the neighbours
        not_shown += len(seeds) - len(frontier)
        distance = {label: 0 for label in frontier}
        for step in range(1, hops + 1):
            reached = sorted({n for label in frontier for n in neighbours.get(label, ()) if n not in distance}, key=prominence)
            frontier = reached[:max_nodes - len(distance)]
            not_shown += len(reached) - len(frontier)
            distance.update((label, step) for label in frontier)
        chosen = sorted(distance, key=lambda label: (distance[label], prominence(label)))
    else:
        mode = "overview"
        pool = sorted((label for label in nodes if isolated or label in neighbours), key=prominence)
        chosen, not_shown = pool[:max_nodes], max(0, len(pool) - max_nodes)
    kept = set(chosen)
    shown = [(k, e) for k, e in edges.items() if k[0] in kept and k[1] in kept]
    linked = {label for (s, o, _r), _e in shown for label in (s, o)}
    hidden_isolated = 0
    if mode == "overview" and not isolated:
        hidden_isolated = len(kept - linked)
        chosen = [label for label in chosen if label in linked]
    shown.sort(key=lambda ke: (-len(ke[1]["papers"]), -ke[1]["claims"], ke[0]))

    def node(label):
        g = nodes[label]
        return {"label": label, "name": g["surfaces"].most_common(1)[0][0], "papers": len(g["papers"]), "mentions": g["mentions"], "cto_papers": len(g["cto"]),
                "tim_papers": len(g["tim"]), "both_papers": len(g["cto"] & g["tim"]), "community": _community(g), "role": g["roles"].most_common(1)[0][0],
                "links": len(neighbours.get(label, ())), "distance": distance.get(label), "seed": label in distance and distance[label] == 0}

    def link(key, e):
        s, o, relation = key
        directions = dict(e["directions"])
        return {"source": s, "target": o, "relation": relation, "papers": len(e["papers"]), "claims": e["claims"],
                "direction": next(iter(directions)) if len(directions) == 1 else "mixed", "directions": directions, "evidence": dict(e["evidence"])}

    links = [link(k, e) for k, e in shown]
    return {"mode": mode, "focus": focus or None, "seeds": len(seeds), "rank": rank, "hops": hops, "max_nodes": max_nodes, "relations": relations, "min_link_papers": min_link,
            "nodes": [node(label) for label in chosen], "links": links,
            "summary": {"labels_matching_filters": len(nodes), "labels_with_links": len(neighbours), "links_matching_filters": len(edges),
                        "links_shown": len(links), "single_paper_links_shown": sum(1 for x in links if x["papers"] == 1),
                        "by_relation": {r: sum(1 for x in links if x["relation"] == r) for r in RELATIONS}, "not_shown": not_shown, "isolated_hidden": hidden_isolated},
            "notes": [NOTES[0], "A link is what at least one paper claims (extracted by a model, not validated). Its width counts the papers that make the claim, not its truth."]}


# --- Entry point ---------------------------------------------------------------------------------------------------------------

def get(path, query, con, run_id):
    """JSON-able dict for /api/constructs/{meta,list,detail,paper,network}; None if the path is not one of ours."""
    handlers = {"/api/constructs/meta": lambda i: meta(i), "/api/constructs/list": lambda i: search(i, query),
                "/api/constructs/detail": lambda i: detail(con, i, query), "/api/constructs/paper": lambda i: paper(con, i, query),
                "/api/constructs/network": lambda i: network(i, query)}
    handler = handlers.get(path)
    return None if handler is None else handler(get_index(con, run_id))
