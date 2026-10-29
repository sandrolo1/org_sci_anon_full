import http.client
import json
import threading

import pytest

from research0.dashboard_server import hostname_of, serve


def test_hostname_of():
    assert hostname_of("Review.Example.org:8765") == "review.example.org" and hostname_of("review.example.org") == "review.example.org"
    assert hostname_of("[::1]") == "[::1]" and hostname_of(None) == ""


@pytest.fixture
def server(world, monkeypatch):
    """The dashboard served over the MOCK catalog, never the real one."""
    monkeypatch.setenv("DATABASE_URL", world.url)
    started = []

    def start(**kwargs):
        srv, run_id = serve(str(world.run_id), 0, **kwargs)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv.server_address[1], run_id
    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def request(port, method, path, host, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host)
    payload = json.dumps(body).encode() if body is not None else None
    for k, v in {"Content-Type": "application/json", **(headers or {})}.items():
        conn.putheader(k, v)
    if payload is not None:
        conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders(payload)
    response = conn.getresponse()
    data = response.read()
    return response.status, (json.loads(data) if data[:1] in (b"{", b"[") else data)


def test_default_is_loopback_only(server):
    port, _ = server()
    assert request(port, "GET", "/api/snapshot", f"127.0.0.1:{port}")[0] == 200
    assert request(port, "GET", "/api/snapshot", "reviewers.example.org")[0] == 403


def test_public_host_is_accepted_only_when_configured_and_read_only_blocks_every_write(server, world):
    port, run_id = server(allowed_hosts=["reviewers.example.org"], read_only=True)
    assert request(port, "GET", "/api/snapshot", "reviewers.example.org")[0] == 200
    assert request(port, "GET", "/api/snapshot", "other.example.org")[0] == 403
    body = {"run_id": str(run_id), "item_id": "x", "decision": "x", "validator": "Ana"}
    status, out = request(port, "POST", "/api/decisions", "reviewers.example.org", body, {"Origin": "https://reviewers.example.org"})
    assert status == 403 and "read-only" in out["error"]
    for path in ("/api/community/start", "/api/community/sample", "/api/fulltext/start", "/api/community/rate"):
        assert request(port, "POST", path, "reviewers.example.org", body, {"Origin": "https://reviewers.example.org"})[0] == 403
    status, _ = request(port, "POST", "/api/fulltext/upload?run_id=x", "reviewers.example.org", None, {"Content-Type": "application/pdf", "Origin": "https://reviewers.example.org", "Content-Length": "0"})
    assert status == 403
    with world.connect() as con:
        assert con.execute("SELECT count(*) FROM validation_decision").fetchone()[0] == 0


def test_full_text_download_is_locked_until_an_admin_mode_exists(server):
    port, run_id = server(read_only=False)                                                                 # writable: only the lock stops it
    local = f"127.0.0.1:{port}"
    status, out = request(port, "POST", "/api/fulltext/start", local, {"run_id": str(run_id), "scope": "sample"}, {"Origin": f"http://{local}"})
    assert status == 403 and "admin mode" in out["error"]
    assert request(port, "GET", "/api/fulltext/status", local)[0] == 200                                   # the figures the Overview shows stay readable


def test_cto_tim_labelling_batch_is_locked_until_an_admin_mode_exists(server):
    port, run_id = server(read_only=False)
    local = f"127.0.0.1:{port}"
    status, out = request(port, "POST", "/api/community/start", local, {"run_id": str(run_id)}, {"Origin": f"http://{local}"})
    assert status == 403 and "admin mode" in out["error"]


def test_evaluation_admin_page_is_locked_until_an_admin_mode_exists_but_the_results_stay_readable(server, monkeypatch):
    port, run_id = server(read_only=False)                                                                 # writable: only the lock stops it
    local = f"127.0.0.1:{port}"
    for path in ("start", "stop", "fulltext", "download", "cards", "cards-batch", "cards-refresh"):
        status, out = request(port, "POST", "/api/evaluation/" + path, local, {"run_id": str(run_id)}, {"Origin": f"http://{local}"})
        assert status == 403 and "admin mode" in out["error"], path
    for path in ("meta", "status", "pipeline", "reconcile", "to-fetch"):                                   # what only the admin page reads
        status, out = request(port, "GET", "/api/evaluation/" + path, local)
        assert status == 403 and "admin mode" in out["error"], path
    status, out = request(port, "GET", "/api/evaluation/runs", local)                                       # the results of a run (page "Pilot results")
    assert status == 200 and "runs" in out
    monkeypatch.setattr("research0.dashboard_server.EVALUATION_ADMIN_MODE", True)
    assert request(port, "GET", "/api/evaluation/status", local)[0] == 200                                 # the flag is the only thing that keeps the page closed


def test_community_statistics_are_readable_even_on_a_read_only_deployment(server, world, monkeypatch, tmp_path):
    from research0 import community_dashboard
    monkeypatch.setattr(community_dashboard, "RUN_ROOT", tmp_path)
    port, run_id = server(allowed_hosts=["reviewers.example.org"], read_only=True)
    host = "reviewers.example.org"
    status, out = request(port, "GET", "/api/community/stats", host)
    assert status == 200 and out["available"] is False                                                     # no Batch collected yet
    with world.connect() as con:
        ids = [row[0] for row in con.execute("SELECT canonical_id FROM canonical_paper WHERE run_id=%s ORDER BY canonical_id LIMIT 4", (run_id,))]
    directory = tmp_path / str(run_id)
    directory.mkdir()
    (directory / "assignments.csv").write_text("record_id,cto,tim\n" + "".join(f"{cid},{c},{t}\n" for cid, (c, t) in zip(ids, [(1, 0), (0, 1), (1, 1), (0, 0)])), encoding="utf-8")
    status, out = request(port, "GET", "/api/community/stats?seed=2", host)
    assert status == 200 and out["labelled"] == len(ids) and {g["papers"] for g in out["groups"]} == {1}
    assert request(port, "GET", "/api/community/stats?seed=x", host)[0] == 400                             # a bad seed is a client error, not a crash
    assert request(port, "POST", "/api/community/start", host, {"run_id": str(run_id)}, {"Origin": f"https://{host}"})[0] == 403   # the Batch stays locked


def test_origins_are_checked_against_the_host(server):
    port, run_id = server(allowed_hosts=["reviewers.example.org"], read_only=False)
    body = {"run_id": str(run_id), "item_id": "nope", "decision": "x", "validator": "Ana"}
    ok = request(port, "POST", "/api/decisions", "reviewers.example.org", body, {"Origin": "https://reviewers.example.org"})
    assert ok[0] == 404                                                                                    # passed the checks, unknown item
    assert request(port, "POST", "/api/decisions", "reviewers.example.org", body, {"Origin": "https://evil.example"})[0] == 403
    assert request(port, "POST", "/api/decisions", "reviewers.example.org", body, {"Origin": "http://evil.example"})[0] == 403
    local = f"127.0.0.1:{port}"
    assert request(port, "POST", "/api/decisions", local, body, {"Origin": f"https://{local}"})[0] == 403   # https is only for configured hosts
    assert request(port, "POST", "/api/decisions", local, body, {"Origin": f"http://{local}"})[0] == 404
