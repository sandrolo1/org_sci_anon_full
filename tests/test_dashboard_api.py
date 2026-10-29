"""The live dashboard's endpoints for the audit, the construct pairs and the release, over the MOCK world only."""
import json
import threading
import urllib.error
import urllib.request
import uuid

import pytest

from research0 import audit, pairsample, signals
from research0.dashboard_server import serve
from research0.embeddings import HashEmbedder
from research0.mock_world import make_preassignment


def call(port, method, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def live(piped, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", piped.world.url)
    with piped.world.connect() as con:
        pre = make_preassignment(con, piped.world)
        audit_id, _, _ = audit.create_sample(con, piped.world.run_id, pre, n_total=80, min_per_stratum=4, min_excluded=6, seed=2)
        signal_run, _, _ = signals.build(con, piped.extraction_run_id, HashEmbedder())
        sample_id, _, _ = pairsample.create(con, signal_run, n_probabilistic=12, n_challenge=4, seed=3)
        sample_ids = [r[0] for r in con.execute("SELECT pair_id FROM pair_sample_item WHERE pair_sample_id=%s", (sample_id,))]
        signals.build_dossiers(con, signal_run, pair_ids=sample_ids)
        con.commit()
    server, run_id = serve(str(piped.world.run_id), 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1], str(run_id), audit_id, signal_run
    server.shutdown()
    server.server_close()
    with piped.world.connect() as con:                      # leave the shared mock world as it was
        for table in ("pair_decision", "pair_dossier", "pair_sample_item", "pair_sample", "candidate_pair", "construct_node", "signal_run", "audit_adjudication", "audit_rating", "audit_sample_paper", "audit_sample",
                      "community_preassignment", "preassignment_run"):
            con.execute(f"DELETE FROM {table}")
        con.commit()


def test_audit_endpoints_hide_everything_that_could_anchor_the_rater(live):
    port, run_id, audit_id, _ = live
    status, out = call(port, "GET", "/api/audit/status?validator=Ana")
    assert status == 200 and out["size"] == 80 and out["my_completed"] == 0 and "design" not in out and "stratum" not in json.dumps(out)
    status, out = call(port, "GET", "/api/audit/papers?validator=Ana")
    assert status == 200 and len(out["papers"]) == 80 and "stratum" not in json.dumps(out) and "label" not in out["papers"][0]
    assert call(port, "GET", "/api/audit/papers")[0] == 400                       # a name is required
    paper = out["papers"][0]["id"]
    ok = call(port, "POST", "/api/audit/rate", {"run_id": run_id, "paper_id": paper, "community": "CTO", "validator": "Ana", "label": "yes"})
    assert ok[0] == 200
    for bad in ({"community": "XYZ", "label": "yes"}, {"community": "CTO", "label": "maybe"}, {"community": "CTO", "label": "yes", "validator": " "},
                {"community": "CTO", "label": "yes", "paper_id": str(uuid.uuid4())}):
        body = {"run_id": run_id, "paper_id": paper, "validator": "Ana", **bad}
        assert call(port, "POST", "/api/audit/rate", body)[0] == 400
    assert call(port, "POST", "/api/audit/rate", {"run_id": str(uuid.uuid4()), "paper_id": paper, "community": "CTO", "validator": "Ana", "label": "yes"})[0] == 409
    assert call(port, "POST", "/api/audit/agreement", {"run_id": run_id})[0] == 400  # nobody has finished


def test_pair_endpoints_are_blind_and_append_decisions(live, piped):
    port, run_id, _, signal_run = live
    status, out = call(port, "GET", "/api/pairs/queue?validator=Ana&limit=10")
    assert status == 200 and out["pairs"]
    text = json.dumps(out)
    assert "s_sem_op" not in text and "triage_hint" not in text and "retrieval_reasons" not in text and "name_sim" not in text
    pair = out["pairs"][0]["pair_id"]
    checks = {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}
    assert call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": pair, "validator": "Ana", "decision_class": "distinct", **checks, "rationale": "different definitions, p. 1"})[0] == 200
    assert call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": pair, "validator": "Ana", "decision_class": "same", **checks, "rationale": "x"})[0] == 400
    assert call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": pair, "validator": "Ana", "decision_class": "distinct", **checks, "rationale": " "})[0] == 400
    assert call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": pair, "validator": "Ana", "decision_class": "distinct", "rationale": "no checklist"})[0] == 400
    old = call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": pair, "validator": "Ana", "relation": "distinct", "flag": "jingle", "rationale": "the earlier scale"})
    assert old[0] == 400 and "codebook" in old[1]["error"]                             # a page left open before the change says to reload
    again = call(port, "GET", "/api/pairs/queue?validator=Ana&limit=10")[1]
    assert pair not in [p["pair_id"] for p in again["pairs"]]                         # already answered by Ana
    other = call(port, "GET", "/api/pairs/queue?validator=Ben&limit=10")[1]
    assert pair in [p["pair_id"] for p in other["pairs"]]                             # still waiting for a second opinion
    assert call(port, "GET", "/api/pairs/status")[1]["pending"] == 1
    assert out["remaining"] == out["sample_size"] and len(out["pairs"]) == min(10, out["sample_size"])        # the queue is the blind sample, 10 at a time
    assert again["remaining"] == out["remaining"] - 1 and other["remaining"] == out["remaining"]
    assert call(port, "GET", "/api/release/status")[1] == {"release": None}


def test_stats_endpoint_shows_the_signals_and_holds_back_the_comparison_with_the_pipeline(live):
    port, run_id, _, signal_run = live
    status, out = call(port, "GET", "/api/pairs/stats?validator=Ana")
    assert status == 200 and out["signal_run_id"] == out["run"]["signal_run_id"] == str(signal_run)      # the page reads the top-level id and out["availability"]["n_nom_used_in_hint"] is False
    assert out["review"]["revealed"] is False and out["review"]["sample"]["pairs"] > 0 and out["review"]["my_decided"] == 0
    assert "triage_vs_human" not in out["review"] and "agreement" not in out["review"]
    queue = call(port, "GET", "/api/pairs/queue?validator=Ana&limit=3")[1]
    assert call(port, "POST", "/api/pairs/decide", {"run_id": run_id, "pair_id": queue["pairs"][0]["pair_id"], "validator": "Ana", "decision_class": "distinct", **{"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}, "rationale": "two different constructs, p. 1"})[0] == 200
    assert call(port, "GET", "/api/pairs/stats?validator=Ana")[1]["review"]["my_decided"] == 1


def test_the_queue_refuses_to_serve_pairs_without_a_blind_sample(live, piped):
    port, _, _, _ = live
    with piped.world.connect() as con:
        con.execute("DELETE FROM pair_sample_item")
        con.execute("DELETE FROM pair_sample")
        con.commit()
    status, out = call(port, "GET", "/api/pairs/queue?validator=Ana&limit=3")
    assert status == 400 and "No blind sample" in out["error"]
