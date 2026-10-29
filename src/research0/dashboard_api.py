"""JSON endpoints of the live dashboard for the human work of Phases 4, 6 and 7: the stratified audit, the construct-pair review and the release status.

Annotators never receive the model label, the stratum, the triage hint or the signals: they see the text and give their own judgement.
Every write is an append to a decision table (audit_rating, audit_adjudication, pair_decision), as everywhere else.
"""
import uuid

from . import audit, card_browser, construct_browser, pairs, pairsample, pairstats, release

AUDIT_POSTS = ("/api/audit/rate", "/api/audit/adjudicate", "/api/audit/agreement")
PAIR_POSTS = ("/api/pairs/decide",)
POSTS = AUDIT_POSTS + PAIR_POSTS


def latest_audit(con, run_id):
    row = con.execute("SELECT audit_id FROM audit_sample WHERE run_id=%s ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
    return row[0] if row else None


def latest_signal_run(con, run_id):
    row = con.execute("SELECT s.signal_run_id FROM signal_run s JOIN extraction_run e USING (extraction_run_id) JOIN ingestion_run i USING (ingestion_run_id) "
                      "WHERE i.run_id=%s ORDER BY s.created_at DESC LIMIT 1", (run_id,)).fetchone()
    return row[0] if row else None


def _name(value):
    value = (value or "").strip()
    if not value or len(value) > 80:
        raise ValueError("Enter your name (up to 80 characters)")
    return value


def get(path, query, con, run_id):
    """Returns a JSON-able dict, or None if the path is not one of ours."""
    one = lambda key, default="": (query.get(key) or [default])[0]
    if path.startswith("/api/constructs/"):
        return construct_browser.get(path, query, con, run_id)
    if path.startswith("/api/cards/"):
        return card_browser.get(path, query, con, run_id)
    if path.startswith("/api/audit/"):
        audit_id = latest_audit(con, run_id)
        if path == "/api/audit/status":
            return {"audit_id": None, "created": False} if not audit_id else {"audit_id": str(audit_id), **audit.status(con, audit_id, one("validator").strip() or None)}
        if not audit_id:
            raise ValueError("No audit sample has been drawn yet (research0 audit-sample)")
        if path == "/api/audit/papers":
            return {"audit_id": str(audit_id), "papers": audit.blind_papers(con, audit_id, _name(one("validator")))}
        if path == "/api/audit/adjudication":
            return {"audit_id": str(audit_id), "items": audit.adjudication_items(con, audit_id)}
    if path == "/api/pairs/status":
        signal_run = latest_signal_run(con, run_id)
        return {"signal_run_id": None} if not signal_run else {"signal_run_id": str(signal_run), **pairs.counts(con, signal_run)}
    if path == "/api/pairs/stats":
        signal_run = latest_signal_run(con, run_id)
        return {"signal_run_id": None} if not signal_run else {"signal_run_id": str(signal_run), **pairstats.stats(con, signal_run, one("validator").strip() or None)}
    if path == "/api/pairs/queue":
        signal_run = latest_signal_run(con, run_id)
        if not signal_run:
            raise ValueError("No signal run yet (research0 signals-build and dossiers-build)")
        sample = pairsample.latest(con, signal_run)
        if not sample:
            raise ValueError("No blind sample has been drawn yet (research0 pair-sample, then research0 dossiers-build --pair-sample-id)")
        limit = max(1, min(int(one("limit", "20")), 100))
        total = con.execute("SELECT count(*) FROM pair_sample_item WHERE pair_sample_id=%s", (sample,)).fetchone()[0]
        waiting = con.execute("SELECT count(*) FROM pair_sample_item i WHERE i.pair_sample_id=%s AND NOT EXISTS (SELECT 1 FROM pair_dossier d WHERE d.pair_id=i.pair_id)", (sample,)).fetchone()[0]
        if waiting:
            raise ValueError(f"{waiting} pairs of the sample have no dossier yet (research0 dossiers-build --pair-sample-id)")
        validator = one("validator").strip() or None
        waiting_for_me = pairs.queue(con, signal_run, validator, total, sample)
        return {"signal_run_id": str(signal_run), "sample_size": total, "remaining": len(waiting_for_me),
                "pairs": [{"pair_id": q["pair_id"], "dossier": pairs.blind(q["dossier"])} for q in waiting_for_me[:limit]]}
    if path == "/api/release/status":
        row = con.execute("SELECT release_id,label,gates_passed,gates,created_at FROM release WHERE run_id=%s ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
        if not row:
            return {"release": None}
        return {"release": {"release_id": str(row[0]), "label": row[1], "status": release.status(con, row[0]), "gates": row[3], "created_at": row[4].isoformat(timespec="seconds")}}
    return None


def post(path, body, con, run_id):
    if path.startswith("/api/audit/"):
        audit_id = latest_audit(con, run_id)
        if not audit_id:
            raise ValueError("No audit sample has been drawn yet (research0 audit-sample)")
        if path == "/api/audit/rate":
            audit.rate(con, audit_id, uuid.UUID(body["paper_id"]), body.get("community"), _name(body.get("validator")), body.get("label"), str(body.get("note") or ""))
            return audit.status(con, audit_id, body["validator"].strip())
        if path == "/api/audit/adjudicate":
            audit.adjudicate(con, audit_id, uuid.UUID(body["paper_id"]), body.get("community"), _name(body.get("adjudicator")), body.get("label"), str(body.get("note") or ""))
            return {"saved": True}
        result = {"agreement": audit.agreement(con, audit_id)}
        if result["agreement"]["pending_adjudication"] == 0:
            result["estimates"] = audit.estimates(con, audit_id)
        return result
    if path == "/api/pairs/decide":
        if body.get("relation") or body.get("flag"):
            raise ValueError("The relation + flag scale was replaced by the codebook of the proposal v4: send a class and the three comparison answers (reload the page)")
        pairs.decide(con, uuid.UUID(body["pair_id"]), _name(body.get("validator")), body.get("decision_class"), body.get("rationale"),
                     {k: body.get(k) for k in pairs.CHECKLIST}, body.get("consequence") or "", body.get("counterexample") or "", bool(body.get("final")))
        return {"saved": True}
    return None
