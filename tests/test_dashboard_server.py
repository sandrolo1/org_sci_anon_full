import json
import os
import threading
import urllib.error
import urllib.request
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="Set local PostgreSQL environment to run integration test")


def call(port, method, path, body=None, headers=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_decisions_are_appended_validated_and_origin_checked():
    from research0.cli import database
    from research0.dashboard_server import serve

    with database() as con:
        for name in ("003_phase1_catalog", "004_validation_decisions"):
            con.execute(open(f"migrations/{name}.sql", encoding="utf-8").read())
        row = con.execute("SELECT run_id FROM catalog_run ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        pytest.skip("no catalog run in the database")
    server, run_id = serve(str(row[0]), 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        status, snapshot = call(port, "GET", "/api/snapshot")
        assert status == 200 and snapshot["live"] and snapshot["items"]
        item = snapshot["items"][0]
        name = "pytest-" + uuid.uuid4().hex[:8]
        post = lambda **kw: call(port, "POST", "/api/decisions", {"run_id": str(run_id), "item_id": item["id"], "decision": item["decisions"][0], "note": "n", "validator": name, **kw})
        assert post()[0] == 200
        status, out = post(decision="withdrawn")
        assert status == 200 and out["decisions"][-1]["decision"] == "withdrawn" or any(d["validator"] == name and d["decision"] == "withdrawn" for d in out["decisions"])
        assert post(decision="not_a_decision")[0] == 400
        assert post(validator=" ")[0] == 400
        assert post(item_id="nope")[0] == 404
        assert post(run_id=str(uuid.uuid4()))[0] == 409
        assert call(port, "POST", "/api/decisions", {}, {"Origin": "http://evil.example"})[0] == 403
        assert call(port, "POST", "/api/decisions", {}, {"Host": "evil.example"})[0] == 403
        with database() as con:
            assert con.execute("SELECT count(*) FROM validation_decision WHERE validator=%s", (name,)).fetchone()[0] == 2  # append-only
            con.execute("DELETE FROM validation_decision WHERE validator=%s", (name,))  # remove test rows only
    finally:
        server.shutdown()
