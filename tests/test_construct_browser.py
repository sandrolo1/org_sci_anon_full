"""The "Constructs" tab: a read-only browser over the extracted constructs. Everything is checked against the MOCK world (a throw-away schema),
by comparing what the API says with an independent SQL count of the same records."""
import json
import threading
import urllib.error
import urllib.request
import uuid

import pytest

from research0 import construct_browser as cb
from research0.community_v2 import paper_fields
from research0.dashboard_server import serve
from research0.extraction import ROLES, normalize_label
from research0.mock_world import make_preassignment


@pytest.fixture(autouse=True)
def fresh_cache():
    cb._CACHE.clear()
    yield
    cb._CACHE.clear()


@pytest.fixture
def ctx(piped, pcon):
    """The piped mock world with a pre-assignment (batch_v2) on top; rolled back at the end of the test."""
    make_preassignment(pcon, piped.world)
    return pcon, piped.world.run_id


def args(**kw):
    return {k: [str(v)] for k, v in kw.items()}


def listing(con, run, **kw):
    return cb.get("/api/constructs/list", args(**kw), con, run)


def test_meta_matches_the_database_and_says_the_labels_are_not_validated(ctx):
    con, run = ctx
    meta = cb.get("/api/constructs/meta", {}, con, run)
    mentions, papers, labels = con.execute("SELECT count(*),count(DISTINCT canonical_id),count(DISTINCT label_normalized) FROM construct_mention").fetchone()
    assert (meta["coverage"]["mentions"], meta["coverage"]["papers_with_constructs"], meta["coverage"]["labels"]) == (mentions, papers, labels) and mentions > 0
    assert meta["membership"]["kind"] == "batch_v2" and meta["membership"]["validated"] is False
    assert sum(r["mentions"] for r in meta["roles"]) == mentions and [r["role"] for r in meta["roles"]] == list(ROLES)
    assert sum(meta["membership"]["groups"].values()) == papers
    for community in ("CTO", "TIM"):
        assert sum(meta["membership"]["counts"][community].values()) == papers
    assert any("not evidence of the same construct" in n for n in meta["notes"]) and meta["run"]["prompt_version"]


def test_cto_and_tim_filters_are_independent_and_partition_the_papers(ctx):
    con, run = ctx
    total = listing(con, run, limit=1)["summary"]["papers"]
    cells = {(c, t): listing(con, run, cto=c, tim=t, limit=1)["summary"]["papers"] for c in cb.STATUSES for t in cb.STATUSES}
    assert sum(cells.values()) == total and total > 0
    assert len({v for v in cells.values() if v}) > 1                      # the mock world has papers in several cells, so the filters do something
    only_cto = con.execute("SELECT count(DISTINCT m.canonical_id) FROM construct_mention m "
                           "JOIN community_preassignment a ON a.canonical_id=m.canonical_id AND a.community='CTO' AND a.label='yes' "
                           "JOIN community_preassignment b ON b.canonical_id=m.canonical_id AND b.community='TIM' AND b.label='no'").fetchone()[0]
    assert cells[("yes", "no")] == only_cto
    assert listing(con, run, cto="yes", limit=1)["summary"]["papers"] == sum(cells[("yes", t)] for t in cb.STATUSES)   # TIM left open = every TIM status together


def test_summary_groups_name_the_combinations_without_merging_them(ctx):
    con, run = ctx
    summary = listing(con, run, limit=1)["summary"]
    assert sum(summary["membership"].values()) == summary["papers"]
    both = listing(con, run, cto="yes", tim="yes", limit=1)["summary"]
    assert both["membership"]["both"] == both["papers"]
    assert listing(con, run, cto="yes", tim="no", limit=1)["summary"]["membership"]["cto_only"] == listing(con, run, cto="yes", tim="no", limit=1)["summary"]["papers"]


def test_role_filter_counts_only_the_matching_mentions(ctx):
    con, run = ctx
    out = listing(con, run, roles="antecedent", limit=100)
    mentions, labels = con.execute("SELECT count(*),count(DISTINCT label_normalized) FROM construct_mention WHERE role='antecedent'").fetchone()
    assert out["summary"]["mentions"] == mentions and out["total"] == labels
    assert all(set(r["roles"]) == {"antecedent"} for r in out["rows"])
    two = listing(con, run, roles="antecedent,outcome", limit=1)["summary"]["mentions"]
    assert two == con.execute("SELECT count(*) FROM construct_mention WHERE role IN ('antecedent','outcome')").fetchone()[0]


def test_text_search_uses_the_label_and_only_optionally_the_definition(ctx):
    con, run = ctx
    label, definition = con.execute("SELECT label_normalized,definition_text FROM construct_mention WHERE coalesce(definition_text,'')<>'' ORDER BY label_normalized LIMIT 1").fetchone()
    word = label.split()[0]
    by_label = listing(con, run, q=word, limit=100)
    assert by_label["total"] > 0 and all(word in r["label"] for r in by_label["rows"])
    whole_definition = normalize_label(definition)
    assert whole_definition not in label
    assert label not in {r["label"] for r in listing(con, run, q=whole_definition, limit=100)["rows"]}
    assert label in {r["label"] for r in listing(con, run, q=whole_definition, defs=1, limit=100)["rows"]}


def test_other_filters_agree_with_the_database(ctx):
    con, run = ctx
    both = listing(con, run, spans="both", limit=100)
    assert all(r["cto_papers"] > 0 and r["tim_papers"] > 0 and r["spans_both"] for r in both["rows"])
    repeated = listing(con, run, min_papers=2, limit=100)
    assert all(r["papers"] >= 2 for r in repeated["rows"])
    assert repeated["total"] == con.execute("SELECT count(*) FROM (SELECT 1 FROM construct_mention GROUP BY label_normalized HAVING count(DISTINCT canonical_id) >= 2) x").fetchone()[0]
    defined = listing(con, run, has_definition=1, limit=1)
    assert defined["summary"]["mentions"] == con.execute("SELECT count(*) FROM construct_mention WHERE coalesce(definition_text,'')<>''").fetchone()[0]
    claimed = listing(con, run, has_claim=1, limit=1)["summary"]["mentions"]
    assert claimed == con.execute("SELECT count(*) FROM construct_mention m WHERE EXISTS (SELECT 1 FROM nomological_claim c WHERE c.subject_mention_id=m.mention_id OR c.object_mention_id=m.mention_id)").fetchone()[0]
    first, last = con.execute("SELECT min(publication_year),max(publication_year) FROM canonical_paper WHERE canonical_id IN (SELECT canonical_id FROM construct_mention)").fetchone()
    assert listing(con, run, year_from=last + 1, limit=1)["total"] == 0
    assert listing(con, run, year_from=first, year_to=last, limit=1)["summary"]["mentions"] == listing(con, run, limit=1)["summary"]["mentions"]
    source = con.execute("SELECT source_title FROM canonical_paper WHERE canonical_id IN (SELECT canonical_id FROM construct_mention) LIMIT 1").fetchone()[0]
    assert listing(con, run, source=source, limit=1)["summary"]["mentions"] == con.execute(
        "SELECT count(*) FROM construct_mention m JOIN canonical_paper c USING (canonical_id) WHERE c.source_title=%s", (source,)).fetchone()[0]


def test_sorting_and_paging_are_stable(ctx):
    con, run = ctx
    first = listing(con, run, sort="papers", limit=5)
    assert [r["papers"] for r in first["rows"]] == sorted((r["papers"] for r in first["rows"]), reverse=True)
    pages = [listing(con, run, sort="label", dir="asc", offset=o, limit=7)["rows"] for o in (0, 7, 14)]
    names = [r["label"] for page in pages for r in page]
    assert names == sorted(names) and len(set(names)) == len(names)


def test_antecedents_and_outcomes_are_the_distinct_labels_linked_by_predicts(ctx):
    con, run = ctx
    sql = ("SELECT count(DISTINCT {other}.label_normalized) FROM nomological_claim k JOIN construct_mention s ON s.mention_id=k.subject_mention_id "
           "JOIN construct_mention o ON o.mention_id=k.object_mention_id WHERE k.relation='predicts' AND {me}.label_normalized=%s AND s.label_normalized <> o.label_normalized")
    rows = listing(con, run, sort="antecedents", limit=100)["rows"]
    assert [r["antecedents"] for r in rows] == sorted((r["antecedents"] for r in rows), reverse=True) and rows[0]["antecedents"] > 0
    assert any(r["outcomes"] for r in rows)
    for r in rows:
        assert r["antecedents"] == con.execute(sql.format(other="s", me="o"), (r["label"],)).fetchone()[0]
        assert r["outcomes"] == con.execute(sql.format(other="o", me="s"), (r["label"],)).fetchone()[0]
    detail = cb.get("/api/constructs/detail", args(label=rows[0]["label"]), con, run)
    assert (detail["antecedents"], detail["outcomes"]) == (rows[0]["antecedents"], rows[0]["outcomes"])


def test_labels_without_antecedents_and_outcomes_are_left_out_of_list_counts_and_network(ctx):
    con, run = ctx
    index = cb.get_index(con, run)
    m = index.mentions[0]
    lonely = m._replace(mention_id=uuid.uuid4(), label="a label nobody links", surface="A label nobody links", claims=0)
    correlated = m._replace(mention_id=uuid.uuid4(), label="a label only correlating", surface="A label only correlating", claims=1)
    other = index.claims[0]
    fake = index._replace(mentions=[*index.mentions, lonely, correlated],
                          claims=[*index.claims, cb.Claim(m.canonical_id, correlated.label, other.object, "correlates", "positive", "hypothesized")])
    before, after = cb.search(index, args(limit=100)), cb.search(fake, args(limit=100))
    assert all(r["antecedents"] or r["outcomes"] for r in after["rows"])
    assert after["total"] == before["total"] and after["summary"] == before["summary"]
    assert {lonely.label, correlated.label}.isdisjoint(r["label"] for r in after["rows"])
    net = cb.network(fake, args(isolated=1, max_nodes=400, relations="correlates"))
    assert {lonely.label, correlated.label}.isdisjoint(n["label"] for n in net["nodes"])
    with pytest.raises(ValueError, match="not in the network"):
        cb.network(fake, args(focus=correlated.label))
    assert cb.get("/api/constructs/meta", {}, con, run)["coverage"]["labels_shown"] == len({x for c in index.claims if c.relation == "predicts" for x in (c.subject, c.object)})


@pytest.mark.parametrize("bad", [{"cto": "maybe"}, {"tim": "x"}, {"roles": "bogus"}, {"sort": "zzz"}, {"limit": 0}, {"limit": 1000}, {"offset": -1}, {"year_from": "abc"}, {"spans": "x"}, {"min_papers": 0}])
def test_bad_parameters_are_rejected_not_ignored(ctx, bad):
    con, run = ctx
    with pytest.raises(ValueError):
        listing(con, run, **bad)


def test_detail_lists_every_use_with_quote_page_and_the_claims(ctx):
    con, run = ctx
    label = con.execute("SELECT m.label_normalized FROM construct_mention m JOIN nomological_claim c ON c.subject_mention_id=m.mention_id GROUP BY 1 ORDER BY count(*) DESC,1 LIMIT 1").fetchone()[0]
    ids = [r[0] for r in con.execute("SELECT mention_id FROM construct_mention WHERE label_normalized=%s", (label,))]
    detail = cb.get("/api/constructs/detail", args(label=label.upper()), con, run)          # the label is normalised: case and spacing do not matter
    assert detail["label"] == label and detail["mentions"] == len(ids)
    assert detail["papers"] == detail["usage_total"] == con.execute("SELECT count(DISTINCT canonical_id) FROM construct_mention WHERE label_normalized=%s", (label,)).fetchone()[0]
    uses = [u for usage in detail["usages"] for u in usage["mentions"]]
    assert len(uses) == len(ids) and all(u["quote"] and u["page"] and u["role"] in ROLES for u in uses)
    assert detail["claims"] == con.execute("SELECT count(*) FROM nomological_claim WHERE subject_mention_id = ANY(%s) OR object_mention_id = ANY(%s)", (ids, ids)).fetchone()[0] > 0
    assert sum(e["claims"] for e in detail["edges"]) == detail["claims"]                      # the mock label has few neighbours, so nothing is cut off
    assert all(e["label"] != label for e in detail["edges"]) and {e["side"] for e in detail["edges"]} <= {"in", "out"}
    assert any(usage["claims"] for usage in detail["usages"]) and all(u["validation"] == {} for u in uses)   # nothing has been validated: the page must not imply it


def test_detail_pages_do_not_repeat_papers(ctx):
    con, run = ctx
    label = con.execute("SELECT label_normalized FROM construct_mention GROUP BY 1 HAVING count(DISTINCT canonical_id) >= 3 ORDER BY count(*) DESC,1 LIMIT 1").fetchone()[0]
    seen = []
    for offset in range(3):
        page = cb.get("/api/constructs/detail", args(label=label, offset=offset, limit=1), con, run)
        assert len(page["usages"]) == 1 and page["usage_offset"] == offset
        seen.append(page["usages"][0]["paper"]["id"])
    assert len(set(seen)) == 3
    assert cb.get("/api/constructs/detail", args(label=label, offset=10000), con, run)["usages"] == []


def test_detail_edges_name_related_labels_without_merging_them(ctx):
    con, run = ctx
    out = cb.get("/api/constructs/list", args(sort="label", dir="asc", limit=100), con, run)
    index = cb.get_index(con, run)
    for row in out["rows"]:
        related = cb._related(index, row["label"])
        assert all(r["label"] != row["label"] and r["how"] in ("contains this label", "is contained in this label") for r in related)


def test_unknown_things_are_refused(ctx):
    con, run = ctx
    with pytest.raises(ValueError):
        cb.get("/api/constructs/detail", args(label="no such construct"), con, run)
    with pytest.raises(ValueError):
        cb.get("/api/constructs/paper", args(id=str(uuid.uuid4())), con, run)
    with pytest.raises(ValueError):
        cb.get("/api/constructs/paper", args(id="not-a-uuid"), con, run)
    assert cb.get("/api/constructs/other", {}, con, run) is None


def test_paper_view_matches_the_database(ctx):
    con, run = ctx
    cid = con.execute("SELECT canonical_id FROM construct_mention GROUP BY 1 ORDER BY count(*) DESC, 1::text LIMIT 1").fetchone()[0]
    out = cb.get("/api/constructs/paper", args(id=cid), con, run)
    assert len(out["mentions"]) == con.execute("SELECT count(*) FROM construct_mention WHERE canonical_id=%s", (cid,)).fetchone()[0]
    assert len(out["claims"]) == con.execute("SELECT count(*) FROM nomological_claim WHERE canonical_id=%s", (cid,)).fetchone()[0]
    assert out["paper"]["id"] == str(cid) and set(out["paper"]["membership"]) == {"CTO", "TIM"} and set(out["paper"]["membership"].values()) <= set(cb.STATUSES)
    assert all(m["papers_with_label"] >= 1 and m["quote"] for m in out["mentions"])
    starts = [r[0] for r in con.execute("SELECT char_start FROM construct_mention WHERE canonical_id=%s ORDER BY char_start,mention_id", (cid,))]
    assert starts == sorted(starts)                                                           # reading order


def test_human_ratings_are_shown_beside_the_labels_never_merged_into_them(ctx):
    con, run = ctx
    a, b, c = [r[0] for r in con.execute("SELECT DISTINCT canonical_id FROM construct_mention ORDER BY 1 LIMIT 3")]
    con.execute("INSERT INTO blind_sample(run_id,threshold) VALUES (%s,0.8) ON CONFLICT DO NOTHING", (run,))
    for position, cid in enumerate((a, b, c), start=9001):
        con.execute("INSERT INTO blind_sample_paper(run_id,canonical_id,position) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", (run, cid, position))
    for name in ("Ana", "Ben"):
        con.execute("INSERT INTO blind_rating(run_id,canonical_id,validator,cto,tim) VALUES (%s,%s,%s,1,0)", (run, a, name))
    con.execute("INSERT INTO blind_rating(run_id,canonical_id,validator,cto,tim) VALUES (%s,%s,'Ana',1,1),(%s,%s,'Ben',0,1)", (run, b, run, b))
    con.execute("INSERT INTO blind_rating(run_id,canonical_id,validator,cto,tim) VALUES (%s,%s,'Ana',1,1),(%s,%s,'Ben',0,1)", (run, c, run, c))
    con.execute("INSERT INTO community_adjudication(run_id,canonical_id,adjudicator,cto,tim) VALUES (%s,%s,'Chi',0,1)", (run, c))
    human = {cid: cb.get("/api/constructs/paper", args(id=cid), con, run)["paper"] for cid in (a, b, c)}
    assert human[a]["human"] == {"CTO": 1, "TIM": 0, "basis": "annotators agree"}
    assert human[b]["human"] == {"CTO": None, "TIM": 1, "basis": "annotators disagree (not adjudicated)"}
    assert human[c]["human"] == {"CTO": 0, "TIM": 1, "basis": "decided by an adjudicator"}
    for cid in (a, b, c):                                                                       # the pipeline label is still the pipeline's
        expected = dict(con.execute("SELECT community,label FROM community_preassignment WHERE canonical_id=%s", (cid,)).fetchall())
        assert human[cid]["membership"] == expected
    assert cb.get("/api/constructs/meta", {}, con, run)["human_rated_papers"] == 3


def test_labels_come_from_the_best_source_available(ctx, tmp_path, monkeypatch):
    con, run = ctx
    assert cb.get("/api/constructs/meta", {}, con, run)["membership"]["kind"] == "batch_v2"
    # a v1 suggestion run, even a newer one, never outranks the pre-assignment
    v1 = uuid.uuid4()
    con.execute("INSERT INTO preassignment_run(preassignment_run_id,run_id,source,model,prompt_version,scope_version,codebook_sha256) VALUES (%s,%s,'v1_suggestion','m','p','s','0')", (v1, run))
    assert cb.get("/api/constructs/meta", {}, con, run)["membership"]["kind"] == "batch_v2"
    con.execute("DELETE FROM community_preassignment")
    con.execute("DELETE FROM preassignment_run WHERE source <> 'v1_suggestion'")
    assert cb.get("/api/constructs/meta", {}, con, run)["membership"]["kind"] == "v1_suggestion"
    con.execute("DELETE FROM preassignment_run")
    monkeypatch.setattr(cb, "run_dir", lambda _run: tmp_path)
    meta = cb.get("/api/constructs/meta", {}, con, run)
    assert meta["membership"]["kind"] == "none" and meta["membership"]["counts"]["CTO"]["unlabelled"] == meta["coverage"]["papers_with_constructs"]
    assert listing(con, run, cto="yes", limit=1)["total"] == 0


def test_the_collected_batch_file_is_read_in_place_and_weak_abstracts_stay_uncertain(ctx, tmp_path, monkeypatch):
    con, run = ctx
    con.execute("DELETE FROM community_preassignment")
    con.execute("DELETE FROM preassignment_run")
    ids = [r[0] for r in con.execute("SELECT DISTINCT canonical_id FROM construct_mention")]
    (tmp_path / "assignments.csv").write_text("record_id,cto,tim\n" + "".join(f"{cid},1,0\n" for cid in ids), encoding="utf-8")
    monkeypatch.setattr(cb, "run_dir", lambda _run: tmp_path)
    meta = cb.get("/api/constructs/meta", {}, con, run)
    assert meta["membership"]["kind"] == "v1_suggestion_file" and meta["membership"]["validated"] is False
    weak = {cid for cid, title, abstract, kw, ikw in con.execute(
        "SELECT c.canonical_id,c.title,r.raw_fields->>'Abstract',r.raw_fields->>'Author Keywords',r.raw_fields->>'Index Keywords' FROM canonical_paper c "
        "JOIN export_record r ON r.record_id=c.representative_record_id WHERE c.canonical_id = ANY(%s)", (ids,)) if paper_fields(title, abstract, kw, ikw)[1] != "abstract"}
    assert weak, "the mock world is expected to contain papers without a usable abstract"
    counts = meta["membership"]["counts"]
    assert counts["CTO"]["uncertain"] == len(weak) == counts["TIM"]["uncertain"]
    assert counts["CTO"]["yes"] == len(ids) - len(weak) and counts["TIM"]["no"] == len(ids) - len(weak)


def test_the_index_is_cached_until_something_it_depends_on_changes(ctx):
    con, run = ctx
    first = cb.get_index(con, run)
    assert cb.get_index(con, run) is first
    con.execute("INSERT INTO preassignment_run(preassignment_run_id,run_id,source,model,prompt_version,scope_version,codebook_sha256) VALUES (%s,%s,'v1_suggestion','m','p','s','0')", (uuid.uuid4(), run))
    assert cb.get_index(con, run) is not first


def network(con, run, **kw):
    return cb.get("/api/constructs/network", args(**kw), con, run)


CLAIM_LINKS = ("SELECT s.label_normalized,o.label_normalized,k.relation,count(DISTINCT k.canonical_id),count(*) FROM nomological_claim k "
               "JOIN construct_mention s ON s.mention_id=k.subject_mention_id JOIN construct_mention o ON o.mention_id=k.object_mention_id "
               "WHERE s.label_normalized <> o.label_normalized {where} GROUP BY 1,2,3")


def test_network_links_are_the_papers_claims_aggregated_never_merged(ctx):
    con, run = ctx
    out = network(con, run, isolated=1, max_nodes=400)
    expected = {(s, o, r): (p, n) for s, o, r, p, n in con.execute(CLAIM_LINKS.format(where=""))}
    assert expected and out["mode"] == "overview" and out["summary"]["links_matching_filters"] == len(expected)
    labels = {n["label"] for n in out["nodes"]}
    assert len(labels) == len(out["nodes"]) <= 400
    for link in out["links"]:
        assert link["source"] in labels and link["target"] in labels and link["source"] != link["target"]
        assert (link["papers"], link["claims"]) == expected[(link["source"], link["target"], link["relation"])]
    linked = {x for link in out["links"] for x in (link["source"], link["target"])}
    plain = network(con, run, max_nodes=400)                                  # by default a label without a link among those shown is left out
    assert {n["label"] for n in plain["nodes"]} == linked and plain["summary"]["isolated_hidden"] == len(labels - linked)
    papers = dict(con.execute("SELECT label_normalized,count(DISTINCT canonical_id) FROM construct_mention GROUP BY 1").fetchall())
    assert all(n["papers"] == papers[n["label"]] and n["community"] in ("cto", "tim", "mixed", "open") for n in out["nodes"])


def test_network_respects_the_filters_of_the_list(ctx):
    con, run = ctx
    only = network(con, run, relations="predicts", max_nodes=400)
    assert only["links"] and {link["relation"] for link in only["links"]} == {"predicts"}
    cto = network(con, run, cto="yes", tim="no", isolated=1, max_nodes=400)
    expected = {(s, o, r): p for s, o, r, p, _n in con.execute(CLAIM_LINKS.format(where=
        "AND EXISTS (SELECT 1 FROM community_preassignment a WHERE a.canonical_id=k.canonical_id AND a.community='CTO' AND a.label='yes') "
        "AND EXISTS (SELECT 1 FROM community_preassignment b WHERE b.canonical_id=k.canonical_id AND b.community='TIM' AND b.label='no')"))}
    assert cto["summary"]["links_matching_filters"] == len(expected)
    assert all(expected[(link["source"], link["target"], link["relation"])] == link["papers"] for link in cto["links"])
    two = network(con, run, min_papers=2, isolated=1, max_nodes=400)
    assert all(n["papers"] >= 2 for n in two["nodes"])
    assert all(link["papers"] >= 2 for link in network(con, run, min_link_papers=2, max_nodes=400)["links"])
    total = con.execute("SELECT count(DISTINCT label_normalized) FROM construct_mention").fetchone()[0]
    small = network(con, run, max_nodes=10, isolated=1)
    assert len(small["nodes"]) == min(10, total) and small["summary"]["not_shown"] == max(0, total - 10)


def test_network_search_and_focus_show_the_neighbourhood(ctx):
    con, run = ctx
    label = con.execute("SELECT s.label_normalized FROM nomological_claim k JOIN construct_mention s ON s.mention_id=k.subject_mention_id "
                        "JOIN construct_mention o ON o.mention_id=k.object_mention_id WHERE s.label_normalized <> o.label_normalized GROUP BY 1 ORDER BY count(*) DESC,1 LIMIT 1").fetchone()[0]
    neighbours = {r[0] for r in con.execute(
        "SELECT CASE WHEN s.label_normalized=%(l)s THEN o.label_normalized ELSE s.label_normalized END FROM nomological_claim k JOIN construct_mention s ON s.mention_id=k.subject_mention_id "
        "JOIN construct_mention o ON o.mention_id=k.object_mention_id WHERE s.label_normalized <> o.label_normalized AND %(l)s IN (s.label_normalized,o.label_normalized)", {"l": label})}
    out = network(con, run, focus=label.upper(), max_nodes=400)
    assert out["mode"] == "focus" and out["focus"] == label
    assert {n["label"]: n["distance"] for n in out["nodes"]} == {label: 0, **{n: 1 for n in neighbours}}
    assert [n["label"] for n in out["nodes"] if n["seed"]] == [label]
    wider = network(con, run, focus=label, hops=2, max_nodes=400)
    assert {n["label"] for n in out["nodes"]} <= {n["label"] for n in wider["nodes"]} and {n["distance"] for n in wider["nodes"]} <= {0, 1, 2}
    word = label.split()[0]
    found = network(con, run, q=word, max_nodes=400)
    assert found["mode"] == "search" and found["seeds"] >= 1
    assert all(word in n["label"] for n in found["nodes"] if n["seed"])
    with pytest.raises(ValueError, match="not in the network"):
        network(con, run, focus="no such construct")


@pytest.mark.parametrize("bad", [{"relations": "causes"}, {"rank": "zzz"}, {"hops": 3}, {"max_nodes": 5}, {"max_nodes": 5000}, {"min_link_papers": 0}, {"cto": "maybe"}])
def test_network_refuses_bad_parameters(ctx, bad):
    con, run = ctx
    with pytest.raises(ValueError):
        network(con, run, **bad)


def test_without_an_extraction_run_the_tab_says_what_to_do(pcon):
    pcon.execute("DELETE FROM extraction_run WHERE false")                        # nothing deleted; just opens the transaction
    with pytest.raises(ValueError, match="extract-prepare"):
        cb.get("/api/constructs/meta", {}, pcon, uuid.uuid4())


# --- Over HTTP, as the dashboard serves it ---------------------------------------------------------------------------------------

@pytest.fixture
def server(piped, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", piped.world.url)
    srv, run_id = serve(str(piped.world.run_id), 0, read_only=True)           # read-only: the tab must work for reviewers
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


def test_the_endpoints_work_in_a_read_only_deployment_and_refuse_bad_input(server):
    port, _ = server
    status, body = http(port, "GET", "/api/constructs/meta")
    assert status == 200 and json.loads(body)["coverage"]["mentions"] > 0
    status, body = http(port, "GET", "/api/constructs/list?sort=label&dir=asc&limit=3")
    assert status == 200 and len(json.loads(body)["rows"]) == 3
    label = json.loads(body)["rows"][0]["label"]
    assert http(port, "GET", "/api/constructs/detail?label=" + urllib.request.quote(label))[0] == 200
    status, body = http(port, "GET", "/api/constructs/network?rank=papers&max_nodes=50")
    assert status == 200 and json.loads(body)["links"]
    for bad in ("/api/constructs/list?cto=maybe", "/api/constructs/detail?label=nothing+like+this", "/api/constructs/paper?id=zzz", "/api/constructs/network?hops=9"):
        status, body = http(port, "GET", bad)
        assert status == 400 and "error" in json.loads(body)
    assert http(port, "GET", "/api/constructs/unknown")[0] == 404
    assert http(port, "POST", "/api/constructs/list")[0] == 404                # reading only: there is nothing to write


def test_the_page_carries_the_tab_and_none_of_its_text_is_inserted_as_html(server):
    port, _ = server
    status, body = http(port, "GET", "/")
    html = body.decode("utf-8")
    assert status == 200 and 'id="t-constructs"' in html and 'id="p-constructs"' in html and "api/constructs/" in html
    script = html[html.index("// Constructs tab:"):]
    script = script[:script.index("</script>")]
    assert "innerHTML" not in script and "insertAdjacentHTML" not in script and "document.write" not in script   # papers' words are data, set with textContent only


def test_the_staging_view_shows_the_staged_run_where_the_browser_expects_the_live_one_and_nobody_else_sees_it(piped, pcon, tmp_path):
    """dashboard-serve --staged (the user's request of 14 Oct 2026): views with the live names over the staged twins, read through the search_path."""
    import shutil
    from research0 import extraction, staging_view
    directory = tmp_path / "staged"
    shutil.copytree(piped.directory, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["prompt_version"] = "construct-extraction-staging-view-test"
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    staged = uuid.UUID(extraction.collect_file(pcon, directory, staged=True)["extraction_run_id"])
    base = pcon.execute("SELECT current_schema()").fetchone()[0]
    view = staging_view.create(pcon)
    assert view == f"staging_view_{base}" and staging_view.schema_name("public") == "staging_view"
    assert pcon.execute("SELECT has_schema_privilege('research_agent', %s, 'USAGE')", (view,)).fetchone()[0] is False
    live = cb.latest_extraction_run(pcon, piped.world.run_id)["extraction_run_id"]
    pcon.execute(f'SET LOCAL search_path = "{view}", "{base}"')
    seen = cb.latest_extraction_run(pcon, piped.world.run_id)
    assert seen["extraction_run_id"] == staged != live and seen["prompt_version"] == "construct-extraction-staging-view-test"
    index = cb.build_index(pcon, piped.world.run_id)
    assert index.mentions and pcon.execute("SELECT count(*) FROM construct_mention").fetchone()[0] == \
        pcon.execute(f'SELECT count(*) FROM "{base}".staged_construct_mention').fetchone()[0]          # the live name reads the staged rows
    assert pcon.execute("SELECT count(*) FROM canonical_paper").fetchone()[0] > 0                         # everything else is the live schema
    assert staging_view.describe(pcon)[-1]["extraction_run_id"] == str(staged)
    url = staging_view.connection_url("postgresql://u:p@h:1/db?sslmode=disable&options=-csearch_path%3Dmock_x,public", "staging_view", "public")
    assert "sslmode=disable" in url and url.count("search_path") == 1 and "staging_view%2Cpublic" in url
