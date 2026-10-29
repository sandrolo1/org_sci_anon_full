"""The "Methodology cards" and "Impact cards" pages: a read-only browser over `paper_card` (GET /api/cards/{meta,list,detail}?kind=method|impact).

A card says what a paper declares (its method, or the change it claims to make), field by field, with exact quotations that were checked against the text when the Batch was collected.
It is a description, not an evaluation of the paper. Nothing here writes to the database or calls a model. The newest valid card of a paper is the one shown; a paper whose only cards
were rejected is listed as rejected, with the reason, and every attempt stays visible in the detail.
"""
import threading

from . import paper_processing

KINDS = tuple(paper_processing.PROFILES)
SORTS = ("title", "year", "source", "fields")
FIELD_STATUSES = ("extracted", "inferred", "missing")
STATUS_FILTERS = ("valid", "rejected", "all")
MAX_PAGE = 100

# The page is in English, the cards are in Italian (that is what the model was asked for); the names of the fields follow the page (and the forum, which shows the cards of a proposal with the same fields and labels).
FIELD_LABELS = {
    "question": "Question", "objective": "Objective of the analysis", "units_levels": "Units and levels", "data_sample": "Data and sample", "design": "Design",
    "measures": "Measures", "procedure": "Analytical procedure", "assumptions_controls": "Assumptions and controls",
    "conversation": "Scientific conversation", "prior_explanation": "Prior explanation challenged", "claimed_change": "Claimed change",
    "consequences": "Stated consequences", "scope_conditions": "Scope conditions"}
TITLES = {"method": "Methodology cards", "impact": "Impact cards"}
STATUS_WORDS = {"extracted": "declared", "inferred": "inferred", "missing": "not stated"}
NOTES = {
    "common": (
        "A card describes what the paper says; it does not judge the paper. «Declared» means the text says it, «inferred» means the model derived it from passages it quotes, «not stated» means the text does not say it.",
        "Every quotation is an exact piece of the text the card was made from (letters and digits compared, so the PDF's formatting does not matter). A field whose quotation could not be found was turned into «not stated» when the Batch was collected, and the card says so.",
        "The texts of the cards are in Italian, as the model was asked; the quotations are in the paper's own language. A card is made from part of the paper (see «Made from» in the detail), not always the whole text.",
        "These cards are the baselines of the evaluator. They are not a release and are never shown to the discovery agents."),
    "method": ("The methodological card is made from the start of the paper and the sections on method, data, measures, analysis, robustness and appendices; the heading detection is a heuristic.",),
    "impact": ("The card of contributions is made from the start of the paper and the sections that state the contribution (discussion, implications, conclusions). It reports the change the authors claim, not the impact the paper had.",)}

COVERAGE_NOTES = {
    "method": "Cards never say a design is new or inadequate: that is for the evaluator, which compares a card with the cards of similar papers.",
    "impact": "Cards report what the authors claim to change, not the impact the paper had."}

_CACHE = {}
_LOCK = threading.Lock()


def _check_kind(query):
    kind = _one(query, "kind", "method")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of: {', '.join(KINDS)}")
    return kind


def _one(query, key, default=""):
    return ((query.get(key) or [default])[0] or default).strip()


def _int(query, key, default=None, low=None, high=None):
    raw = _one(query, key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a whole number")
    return max(low, value) if low is not None and value < low else min(high, value) if high is not None and value > high else value


# --- Index: the newest valid card of each paper, built once per change of the table --------------------------------------------------------------------

def _signature(con, kind, run_id):
    row = con.execute("SELECT count(*),max(c.created_at) FROM paper_card c JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) WHERE c.kind=%s AND i.run_id=%s",
                      (kind, run_id)).fetchone()
    return tuple(row)


def build_index(con, kind, run_id):
    profile = paper_processing.PROFILES[kind]
    rows = con.execute(
        "SELECT DISTINCT ON (c.canonical_id) c.card_id,c.canonical_id,c.status,c.card,c.rejection,c.profile_version,c.model,c.source_info,c.created_at "
        "FROM paper_card c JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) WHERE c.kind=%s AND i.run_id=%s "
        "ORDER BY c.canonical_id,(c.status='valid') DESC,c.created_at DESC,c.card_id", (kind, run_id)).fetchall()
    cids = [r[1] for r in rows]
    papers = {cid: (title, authors, year, source, doi) for cid, title, authors, year, source, doi in con.execute(
        "SELECT canonical_id,title,authors,publication_year,source_title,coalesce(doi,all_dois[1]) FROM canonical_paper WHERE canonical_id = ANY(%s)", (cids,))}
    entries = {}
    for card_id, cid, status, card, rejection, version, model, info, created in rows:
        title, authors, year, source, doi = papers.get(cid, (None,) * 5)
        fields = {f: (card[f]["status"] if card else None) for f in profile.fields}
        texts = " ".join(card[f]["text"] or "" for f in profile.fields) if card else ""
        entries[cid] = {"id": str(cid), "card_id": str(card_id), "title": title, "authors": authors, "year": year, "source": source, "doi": doi, "status": status, "rejection": rejection,
                        "fields": fields, "present": sum(1 for s in fields.values() if s in ("extracted", "inferred")), "gaps": len(card["info_gaps"]) if card else 0,
                        "demoted": len((info or {}).get("demoted_fields", [])), "profile_version": version, "model": model, "created_at": created.isoformat(timespec="seconds"),
                        "haystack": " ".join(str(x or "") for x in (title, authors, doi)).lower(), "texts": texts.lower()}
    return entries


def get_index(con, kind, run_id):
    """Cards are append-only, so the index is rebuilt only when a card was added."""
    with _LOCK:
        signature = _signature(con, kind, run_id)
        cached = _CACHE.get((kind, run_id))
        if cached and cached[0] == signature:
            return cached[1]
        index = build_index(con, kind, run_id)
        _CACHE[(kind, run_id)] = (signature, index)
        return index


# --- Meta ----------------------------------------------------------------------------------------------------------------------------------------------

def meta(con, kind, run_id):
    profile = paper_processing.PROFILES[kind]
    index = get_index(con, kind, run_id)
    valid = [e for e in index.values() if e["status"] == "valid"]
    stats = {f: {s: sum(1 for e in valid if e["fields"][f] == s) for s in FIELD_STATUSES} for f in profile.fields}
    attempts = con.execute("SELECT count(*) FILTER (WHERE c.status='rejected') FROM paper_card c JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) "
                           "WHERE c.kind=%s AND i.run_id=%s", (kind, run_id)).fetchone()[0]
    parsed = con.execute("SELECT count(DISTINCT v.canonical_id) FROM paper_version v JOIN ingestion_run i USING (ingestion_run_id) WHERE i.run_id=%s AND v.status='parsed'", (run_id,)).fetchone()[0]
    sent = con.execute("SELECT count(DISTINCT v.canonical_id) FROM processing_item p JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) "
                       "WHERE p.kind=%s AND i.run_id=%s", (kind, run_id)).fetchone()[0]
    years = sorted({e["year"] for e in index.values() if e["year"] is not None})
    sources = {}
    for e in index.values():
        if e["source"]:
            sources[e["source"]] = sources.get(e["source"], 0) + 1
    versions = sorted({e["profile_version"] for e in index.values()})
    models = sorted({e["model"] for e in index.values()})
    dates = sorted(e["created_at"] for e in index.values())
    return {"kind": kind, "title": TITLES[kind], "profile_version": profile.version, "versions_seen": versions, "models": models,
            "created_from": dates[0] if dates else None, "created_to": dates[-1] if dates else None,
            "fields": [{"key": f, "label": FIELD_LABELS.get(f, f)} for f in profile.fields],
            "coverage": {"parsed_papers": parsed, "sent_papers": sent, "papers_with_valid_card": len(valid), "papers_only_rejected": len(index) - len(valid),
                         "rejected_attempts": attempts, "waiting_for_batch": max(0, sent - len(index)),
                         "valid_with_demoted_fields": sum(1 for e in valid if e["demoted"])},
            "field_stats": stats, "years": years, "sources": [{"source": s, "papers": n} for s, n in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0]))],
            "status_words": STATUS_WORDS, "notes": list(NOTES["common"] + NOTES[kind]), "coverage_note": COVERAGE_NOTES[kind]}


# --- List ----------------------------------------------------------------------------------------------------------------------------------------------

def parse_filters(query, profile):
    f = {"q": _one(query, "q").lower(), "in_text": _one(query, "in_text") == "1", "year_from": _int(query, "year_from"), "year_to": _int(query, "year_to"), "source": _one(query, "source"),
         "status": _one(query, "status", "valid"), "field": _one(query, "field"), "field_status": _one(query, "field_status"),
         "min_fields": _int(query, "min_fields", 0, 0, len(profile.fields)), "sort": _one(query, "sort", "title"), "dir": _one(query, "dir", "asc")}
    if f["status"] not in STATUS_FILTERS:
        raise ValueError(f"status must be one of: {', '.join(STATUS_FILTERS)}")
    if f["field"] and f["field"] not in profile.fields:
        raise ValueError("unknown field")
    if f["field_status"] and f["field_status"] not in FIELD_STATUSES:
        raise ValueError(f"field_status must be one of: {', '.join(FIELD_STATUSES)}")
    if f["sort"] not in SORTS or f["dir"] not in ("asc", "desc"):
        raise ValueError("unknown sort")
    return f


def _matches(e, f):
    if f["status"] != "all" and e["status"] != f["status"]:
        return False
    if f["q"] and f["q"] not in e["haystack"] and not (f["in_text"] and f["q"] in e["texts"]):
        return False
    if f["year_from"] is not None and (e["year"] is None or e["year"] < f["year_from"]):
        return False
    if f["year_to"] is not None and (e["year"] is None or e["year"] > f["year_to"]):
        return False
    if f["source"] and e["source"] != f["source"]:
        return False
    if f["field"] and f["field_status"] and e["fields"][f["field"]] != f["field_status"]:
        return False
    return e["present"] >= f["min_fields"]


def _row(e):
    return {k: e[k] for k in ("id", "title", "authors", "year", "source", "doi", "status", "rejection", "fields", "present", "gaps", "demoted")}


def search(con, kind, run_id, query):
    profile = paper_processing.PROFILES[kind]
    f = parse_filters(query, profile)
    index = get_index(con, kind, run_id)
    rows = [e for e in index.values() if _matches(e, f)]
    key = {"title": lambda e: (e["title"] or "").lower(), "year": lambda e: (e["year"] is None, e["year"] or 0), "source": lambda e: ((e["source"] or "").lower(), (e["title"] or "").lower()),
           "fields": lambda e: (e["present"], (e["title"] or "").lower())}[f["sort"]]
    rows.sort(key=key, reverse=f["dir"] == "desc")
    offset, limit = _int(query, "offset", 0, 0), _int(query, "limit", 50, 1, MAX_PAGE)
    valid = [e for e in rows if e["status"] == "valid"]
    summary = {"papers": len(rows), "valid": len(valid), "rejected": len(rows) - len(valid), "fields": {
        fld: {s: sum(1 for e in valid if e["fields"][fld] == s) for s in FIELD_STATUSES} for fld in profile.fields}}
    return {"total": len(rows), "offset": offset, "limit": limit, "summary": summary, "rows": [_row(e) for e in rows[offset:offset + limit]]}


# --- One paper -----------------------------------------------------------------------------------------------------------------------------------------

def detail(con, kind, run_id, query):
    profile = paper_processing.PROFILES[kind]
    index = get_index(con, kind, run_id)
    raw = _one(query, "id")
    entry = next((e for e in index.values() if e["id"] == raw), None)
    if entry is None:
        raise ValueError(f"This paper has no {TITLES[kind].lower()[:-1]} in the latest run")
    attempts = [{"card_id": str(cid), "status": s, "rejection": r, "created_at": at.isoformat(timespec="seconds"), "model": m, "profile_version": v, "job": j}
                for cid, s, r, at, m, v, j in con.execute(
                    "SELECT c.card_id,c.status,c.rejection,c.created_at,c.model,c.profile_version,c.job FROM paper_card c JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) "
                    "WHERE c.kind=%s AND i.run_id=%s AND c.canonical_id=%s ORDER BY c.created_at DESC,c.card_id", (kind, run_id, entry["id"]))]
    card_id = entry["card_id"]
    shown = con.execute("SELECT card,source_info FROM paper_card WHERE card_id=%s", (card_id,)).fetchone()
    card, info = shown
    fields = []
    if card:
        for f in profile.fields:
            c = card[f]
            fields.append({"key": f, "label": FIELD_LABELS.get(f, f), "status": c["status"], "text": c["text"], "passages": c["passages"]})
    other = next(k for k in KINDS if k != kind)
    has_other = con.execute("SELECT EXISTS (SELECT 1 FROM paper_card c JOIN paper_version v USING (version_id) JOIN ingestion_run i USING (ingestion_run_id) "
                            "WHERE c.kind=%s AND i.run_id=%s AND c.canonical_id=%s AND c.status='valid')", (other, run_id, entry["id"])).fetchone()[0]
    return {"kind": kind, "paper": {k: entry[k] for k in ("id", "title", "authors", "year", "source", "doi")},
            "card": {"card_id": card_id, "status": entry["status"], "rejection": entry["rejection"], "profile_version": entry["profile_version"], "model": entry["model"],
                     "created_at": entry["created_at"], "fields": fields, "info_gaps": card["info_gaps"] if card else [], "source_info": info or {}},
            "attempts": attempts, "other": {"kind": other, "title": TITLES[other], "has_valid_card": bool(has_other)}}


def get(path, query, con, run_id):
    """JSON-able dict for /api/cards/{meta,list,detail}; None if the path is not one of ours."""
    handlers = {"/api/cards/meta": lambda k: meta(con, k, run_id), "/api/cards/list": lambda k: search(con, k, run_id, query), "/api/cards/detail": lambda k: detail(con, k, run_id, query)}
    handler = handlers.get(path)
    return handler(_check_kind(query)) if handler else None
