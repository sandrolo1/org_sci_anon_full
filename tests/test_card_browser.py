"""The "Methodology cards" and "Impact cards" pages: a read-only browser over `paper_card`. Checked against the MOCK world (a throw-away schema) by comparing what the API says
with an independent SQL count of the same rows. The cards are written by the test itself, committed, and deleted at the end; nothing reaches a model."""
import json
import threading
import urllib.error
import urllib.request
import uuid
from types import SimpleNamespace

import pytest

from research0 import card_browser as cbr
from research0 import paper_processing as pp
from research0.dashboard_server import serve


def args(**kw):
    return {k: [str(v)] for k, v in kw.items()}


def make(kind, statuses=None, gaps=()):
    """A card in the shape the Batch collector stores: every field {text, status, passages}, plus info_gaps."""
    statuses, card = statuses or {}, {}
    for f in pp.PROFILES[kind].fields:
        s = statuses.get(f, "extracted")
        card[f] = {"text": None, "status": "missing", "passages": []} if s == "missing" else {"text": f"testo di {f}", "status": s, "passages": [f"a passage for the field {f} of the paper"]}
    card["info_gaps"] = list(gaps)
    return card


def insert(con, kind, version, canonical, status, card=None, rejection=None, created="2026-10-18T10:00:00+00", demoted=()):
    con.execute("INSERT INTO paper_card(card_id,kind,version_id,canonical_id,profile_version,model,source_sha256,source_info,status,card,rejection,job,created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s)",
                (uuid.uuid4(), kind, version, canonical, pp.PROFILES[kind].version, "gpt-test", "0" * 64,
                 json.dumps({"kind": "sections", "total_chars": 5000, "used_chars": 3000, "truncated": False, "demoted_fields": list(demoted)}), status,
                 json.dumps(card) if card else None, rejection, "job-1", created))


@pytest.fixture(autouse=True)
def fresh_cache():
    cbr._CACHE.clear()
    yield
    cbr._CACHE.clear()


@pytest.fixture
def cards(piped):
    """Four papers: A has a methodology and an impact card; B was rejected once and then got a valid methodology card (design not stated, measures inferred);
    C has only a rejected methodology card; D has only an impact card. Committed in the piped world and deleted afterwards."""
    world = piped.world
    with world.connect() as c:
        papers = c.execute("SELECT DISTINCT ON (canonical_id) version_id,canonical_id FROM paper_version WHERE ingestion_run_id=%s AND status='parsed' ORDER BY canonical_id,version_id",
                           (piped.ingestion_id,)).fetchall()[:4]
        (va, ca), (vb, cb_), (vc, cc), (vd, cd) = papers
        insert(c, "method", va, ca, "valid", make("method"))
        insert(c, "impact", va, ca, "valid", make("impact"))
        insert(c, "method", vb, cb_, "rejected", rejection="only 3 of 8 fields have an exact quotation (at least 4 are needed)", created="2026-10-17T10:00:00+00")
        insert(c, "method", vb, cb_, "valid", make("method", {"design": "missing", "measures": "inferred"}, gaps=["Disegno: la citazione proposta non e un passaggio esatto del testo"]), demoted=["design"])
        insert(c, "method", vc, cc, "rejected", rejection="the card is not a JSON object")
        insert(c, "impact", vd, cd, "valid", make("impact", {"claimed_change": "inferred"}))
        for v in (va, vb, vc, vd):
            c.execute("INSERT INTO processing_item(kind,version_id,prompt_version,job) VALUES ('method',%s,%s,'job-1')", (v, pp.PROFILES["method"].version))
        c.commit()
    try:
        yield SimpleNamespace(world=world, run=world.run_id, a=str(ca), b=str(cb_), c=str(cc), d=str(cd))
    finally:
        with world.connect() as c:
            c.execute("DELETE FROM paper_card")
            c.execute("DELETE FROM processing_item")
            c.commit()


@pytest.fixture
def con(cards):
    connection = cards.world.connect()
    yield connection
    connection.rollback()
    connection.close()


def call(con, cards, endpoint, **kw):
    return cbr.get(f"/api/cards/{endpoint}", args(**kw), con, cards.run)


def titles(con, ids):
    return {str(cid): t for cid, t in con.execute("SELECT canonical_id,title FROM canonical_paper WHERE canonical_id = ANY(%s)", (list(ids),))}


# --- meta ---------------------------------------------------------------------------------------------------------------------------------------------

def test_meta_counts_match_the_table_and_name_the_fields_of_the_profile(con, cards):
    m = call(con, cards, "meta", kind="method")
    profile = pp.PROFILES["method"]
    assert [f["key"] for f in m["fields"]] == list(profile.fields) and all(f["label"] for f in m["fields"])
    assert "axes" not in m                                                                                       # a flat list of fields, as the forum shows the card of a proposal
    c = m["coverage"]
    assert (c["papers_with_valid_card"], c["papers_only_rejected"], c["rejected_attempts"]) == (2, 1, 2)
    assert (c["sent_papers"], c["waiting_for_batch"], c["valid_with_demoted_fields"]) == (4, 1, 1)            # D was sent but has no methodology card yet
    sql = con.execute("SELECT count(*) FILTER (WHERE card->'design'->>'status'='missing'), count(*) FILTER (WHERE card->'measures'->>'status'='inferred') "
                      "FROM paper_card WHERE kind='method' AND status='valid'").fetchone()
    assert (m["field_stats"]["design"]["missing"], m["field_stats"]["measures"]["inferred"]) == tuple(sql)
    assert all(sum(s.values()) == 2 for s in m["field_stats"].values())                                          # the rejected card is in no statistic
    assert m["profile_version"] == profile.version and m["models"] == ["gpt-test"] and any("does not judge" in n or "does not judge the paper" in n for n in m["notes"])


def test_the_impact_page_has_its_own_fields(con, cards):
    m = call(con, cards, "meta", kind="impact")
    assert [f["key"] for f in m["fields"]] == list(pp.PROFILES["impact"].fields)
    assert (m["coverage"]["papers_with_valid_card"], m["coverage"]["papers_only_rejected"]) == (2, 0)


# --- list ---------------------------------------------------------------------------------------------------------------------------------------------

def test_the_list_shows_valid_cards_by_default_and_the_newest_valid_card_of_a_paper(con, cards):
    rows = call(con, cards, "list", kind="method")["rows"]
    assert {r["id"] for r in rows} == {cards.a, cards.b}                         # C has only a rejected card, D has no methodology card
    b = next(r for r in rows if r["id"] == cards.b)
    assert b["status"] == "valid" and b["present"] == 7 and b["fields"]["design"] == "missing" and b["fields"]["measures"] == "inferred" and (b["gaps"], b["demoted"]) == (1, 1)
    names = [r["title"].lower() for r in rows]
    assert names == sorted(names)                                                  # title A to Z by default


def test_rejected_papers_are_listed_on_request_with_the_reason_and_no_fields(con, cards):
    out = call(con, cards, "list", kind="method", status="rejected")
    assert [r["id"] for r in out["rows"]] == [cards.c] and out["rows"][0]["rejection"] == "the card is not a JSON object"
    assert set(out["rows"][0]["fields"].values()) == {None} and out["rows"][0]["present"] == 0
    assert out["summary"] == {**out["summary"], "papers": 1, "valid": 0, "rejected": 1}
    assert call(con, cards, "list", kind="method", status="all")["total"] == 3


def test_summary_counts_the_valid_cards_of_the_view_field_by_field(con, cards):
    s = call(con, cards, "list", kind="method")["summary"]
    assert (s["papers"], s["valid"], s["rejected"]) == (2, 2, 0)
    assert s["fields"]["design"] == {"extracted": 1, "inferred": 0, "missing": 1} and s["fields"]["question"]["extracted"] == 2
    only_b = call(con, cards, "list", kind="method", field="design", field_status="missing")
    assert [r["id"] for r in only_b["rows"]] == [cards.b] and only_b["summary"]["fields"]["design"]["missing"] == 1


def test_filters_on_fields_search_text_year_and_journal(con, cards):
    assert call(con, cards, "list", kind="method", min_fields=8)["total"] == 1                                    # only A has every field
    assert call(con, cards, "list", kind="method", field="measures", field_status="inferred")["rows"][0]["id"] == cards.b
    title_b = titles(con, [cards.b])[cards.b]
    word = max(title_b.split(), key=len).lower()
    assert cards.b in {r["id"] for r in call(con, cards, "list", kind="method", q=word)["rows"]}
    assert call(con, cards, "list", kind="method", q="testo di question")["total"] == 0                            # the card texts are searched only when asked
    assert call(con, cards, "list", kind="method", q="testo di question", in_text=1)["total"] == 2
    assert [r["id"] for r in call(con, cards, "list", kind="method", q="testo di design", in_text=1)["rows"]] == [cards.a]   # B does not state its design: no text to find
    year, source = con.execute("SELECT publication_year,source_title FROM canonical_paper WHERE canonical_id=%s", (cards.a,)).fetchone()
    expected = con.execute("SELECT count(*) FROM canonical_paper WHERE canonical_id = ANY(%s) AND publication_year=%s", ([cards.a, cards.b], year)).fetchone()[0]
    assert call(con, cards, "list", kind="method", year_from=year, year_to=year)["total"] == expected
    assert cards.a in {r["id"] for r in call(con, cards, "list", kind="method", source=source)["rows"]}
    assert call(con, cards, "list", kind="method", year_from=2999)["total"] == 0


def test_sorting_and_paging(con, cards):
    by_fields = call(con, cards, "list", kind="method", sort="fields", dir="desc")["rows"]
    assert [r["id"] for r in by_fields] == [cards.a, cards.b]
    assert [r["id"] for r in call(con, cards, "list", kind="method", sort="fields", dir="asc")["rows"]] == [cards.b, cards.a]
    page = call(con, cards, "list", kind="method", limit=1, offset=1, sort="fields", dir="desc")
    assert page["total"] == 2 and [r["id"] for r in page["rows"]] == [cards.b]


def test_bad_input_is_refused_with_a_message(con, cards):
    for endpoint, bad in (("list", {"kind": "other"}), ("list", {"status": "maybe"}), ("list", {"sort": "random"}), ("list", {"field": "nothing"}), ("list", {"field_status": "x"}),
                          ("list", {"year_from": "abc"}), ("detail", {"id": "zzz"}), ("detail", {"id": str(uuid.uuid4())})):
        with pytest.raises(ValueError):
            call(con, cards, endpoint, **{"kind": "method", **bad})
    assert cbr.get("/api/cards/unknown", {}, con, cards.run) is None


# --- detail -------------------------------------------------------------------------------------------------------------------------------------------

def test_detail_gives_every_field_with_its_status_text_and_exact_passages(con, cards):
    d = call(con, cards, "detail", kind="method", id=cards.b)
    assert d["paper"]["id"] == cards.b and d["card"]["status"] == "valid" and d["card"]["info_gaps"]
    by_key = {f["key"]: f for f in d["card"]["fields"]}
    assert list(by_key) == list(pp.PROFILES["method"].fields)
    assert by_key["design"]["status"] == "missing" and by_key["design"]["text"] is None and by_key["design"]["passages"] == []
    assert by_key["measures"]["status"] == "inferred" and by_key["measures"]["passages"]
    assert all("axis" not in f for f in d["card"]["fields"])
    assert d["card"]["source_info"]["demoted_fields"] == ["design"] and d["card"]["source_info"]["used_chars"] == 3000


def test_detail_keeps_every_attempt_and_points_to_the_other_card(con, cards):
    d = call(con, cards, "detail", kind="method", id=cards.b)
    assert [(a["status"], bool(a["rejection"])) for a in d["attempts"]] == [("valid", False), ("rejected", True)]      # newest first
    assert d["other"] == {"kind": "impact", "title": "Impact cards", "has_valid_card": False}
    a = call(con, cards, "detail", kind="method", id=cards.a)
    assert a["other"]["has_valid_card"] is True
    assert call(con, cards, "detail", kind="impact", id=cards.a)["other"] == {"kind": "method", "title": "Methodology cards", "has_valid_card": True}
    assert call(con, cards, "detail", kind="impact", id=cards.d)["card"]["fields"][2]["status"] == "inferred"          # claimed_change


def test_a_paper_with_only_rejected_cards_shows_the_reason_and_no_fields(con, cards):
    d = call(con, cards, "detail", kind="method", id=cards.c)
    assert d["card"]["status"] == "rejected" and d["card"]["rejection"] == "the card is not a JSON object" and d["card"]["fields"] == [] and len(d["attempts"]) == 1


def test_a_new_card_shows_up_without_restarting(con, cards):
    assert call(con, cards, "list", kind="impact")["total"] == 2
    with cards.world.connect() as c:
        vid = c.execute("SELECT version_id FROM paper_version WHERE canonical_id=%s AND ingestion_run_id IN (SELECT ingestion_run_id FROM ingestion_run WHERE run_id=%s) LIMIT 1", (cards.b, cards.run)).fetchone()[0]
        insert(c, "impact", vid, cards.b, "valid", make("impact"), created="2026-10-18T11:00:00+00")
        c.commit()
    assert call(con, cards, "list", kind="impact")["total"] == 3


# --- over HTTP, as the dashboard serves it -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def server(cards, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", cards.world.url)
    srv, run_id = serve(str(cards.run), 0, read_only=True)                  # read-only: the pages must work for reviewers
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], str(run_id)
    srv.shutdown()
    srv.server_close()


def http(port, method, path):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=b"{}" if method == "POST" else None, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_the_endpoints_work_in_a_read_only_deployment_and_refuse_bad_input(server, cards):
    port, _ = server
    for kind in ("method", "impact"):
        status, body = http(port, "GET", f"/api/cards/meta?kind={kind}")
        assert status == 200 and json.loads(body)["coverage"]["papers_with_valid_card"] == 2
        status, body = http(port, "GET", f"/api/cards/list?kind={kind}&limit=5")
        assert status == 200 and json.loads(body)["total"] == 2
    assert http(port, "GET", f"/api/cards/detail?kind=method&id={cards.a}")[0] == 200
    for bad in ("/api/cards/list?kind=other", "/api/cards/detail?kind=method&id=nothing", "/api/cards/list?kind=method&status=maybe"):
        status, body = http(port, "GET", bad)
        assert status == 400 and "error" in json.loads(body)
    assert http(port, "GET", "/api/cards/unknown")[0] == 404
    assert http(port, "POST", "/api/cards/list")[0] == 404                      # reading only: there is nothing to write


def test_the_menu_carries_the_two_pages_under_the_step_and_none_of_the_card_text_is_inserted_as_html(server):
    port, _ = server
    status, body = http(port, "GET", "/")
    html = body.decode("utf-8")
    assert status == 200 and "Data Extraction" in html
    for tab in ("method", "impact"):
        assert f'id="t-{tab}"' in html and f'id="p-{tab}"' in html and f'id="cd-{tab}-root"' in html
    assert "api/cards/" in html
    script = html[html.index("// Paper cards:"):]
    script = script[:script.index("</script>")]
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script and "document.write" not in script        # the cards' words are data, set with textContent only
