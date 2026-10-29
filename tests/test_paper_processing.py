"""One click, three Batches: what each kind sends, the ceiling, the record that stops a paper from being sent twice, the resumption after a failed submission, and the validation of the
cards that come back. The submission itself is replaced by a stand-in: nothing here reaches OpenAI. Needs the local PostgreSQL (a throw-away mock schema, dropped at the end)."""
import json
import os
from types import SimpleNamespace

import pytest

from research0 import documents, extraction, paper_cards, paper_processing as pp


@pytest.fixture(scope="module")
def pworld():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set local PostgreSQL environment to run integration tests")
    from research0 import mock_world
    w = mock_world.create_world(n=60, seed=5)
    try:
        with w.connect() as c:
            ingestion_id, _ = documents.ingest(c, w.run_id, w.store)
            c.commit()
        yield SimpleNamespace(world=w, ingestion_id=ingestion_id)
    finally:
        w.drop()


@pytest.fixture
def fresh(pworld):
    """A connection for tests that commit: the tables the click writes are emptied before and after, so each test starts from a world where nothing was sent."""
    def wipe():
        with pworld.world.connect() as c:
            c.execute("DELETE FROM paper_card")
            c.execute("DELETE FROM processing_item")
            c.commit()
    wipe()
    connection = pworld.world.connect()
    yield connection
    connection.rollback()
    connection.close()
    wipe()


@pytest.fixture
def fake_submit(monkeypatch):
    """Replaces the Batch submission: writes a batch.json like the real one and records the calls."""
    calls = []

    def submit(directory):
        calls.append(directory.name)
        state = {"input_file_id": "file_x", "batch_id": f"batch_{directory.name}", "status": "validating", "request_counts": None}
        (directory / "batch.json").write_text(json.dumps(state), encoding="utf-8")
        return state
    monkeypatch.setattr(pp, "submit", submit)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    return calls


def prepared(pworld, fresh, tmp_path, limit=None):
    return pp.prepare_job(fresh, pworld.world.run_id, None, limit, base=tmp_path)


# --- what each kind would send --------------------------------------------------------------------------------------------------------------------

def test_the_estimate_counts_the_same_papers_for_the_three_kinds_and_writes_nothing(pworld, fresh, tmp_path):
    out = pp.estimate(fresh, pworld.world.run_id)
    parsed = len(pp.parsed_versions(fresh, pworld.ingestion_id))
    assert parsed > 20 and out["pdfs_to_parse"] == 0 and out["ingestion_run_id"] == str(pworld.ingestion_id)
    assert {k: v["papers"] for k, v in out["kinds"].items()} == {"constructs": parsed, "method": parsed, "impact": parsed} and out["estimated_input_tokens"] > 0
    assert not list(tmp_path.iterdir())
    assert pp.estimate(fresh, pworld.world.run_id, limit=5)["kinds"]["method"]["papers"] == 5


def test_the_job_prepares_three_directories_in_the_layout_the_existing_commands_read(pworld, fresh, tmp_path):
    plan = prepared(pworld, fresh, tmp_path)
    root = tmp_path / plan["job"]
    for kind in pp.KINDS:
        manifest = json.loads((root / kind / "manifest.json").read_text(encoding="utf-8"))
        lines = (root / kind / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        assert manifest["request_count"] == len(lines) == plan["kinds"][kind]["requests"] > 0 and set(manifest["requests"]) == {json.loads(l)["custom_id"] for l in lines}
        assert not (root / kind / "batch.json").exists()
    assert plan["estimated_input_tokens"] == sum(k["estimated_input_tokens"] for k in plan["kinds"].values()) > 0 and (root / "job.json").exists()


def test_a_card_request_carries_the_card_instructions_the_strict_schema_and_the_text_it_will_be_checked_against(pworld, fresh, tmp_path):
    plan = prepared(pworld, fresh, tmp_path)
    for kind, profile in (("method", paper_cards.METHOD), ("impact", pp.PROFILES["impact"])):
        request = json.loads((tmp_path / plan["job"] / kind / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
        body = request["body"]
        assert request["url"] == "/v1/responses" and body["instructions"] == profile.card_system and body["text"]["format"]["strict"] is True and body["text"]["format"]["name"] == profile.card_task
        assert set(body["text"]["format"]["schema"]["properties"]) == set(profile.fields) | {"info_gaps"} and "minimum" not in json.dumps(body["text"]["format"]["schema"])
        assert f"INPUT_ID: {request['custom_id']}" in body["input"] and "<<<TESTO" in body["input"]
    constructs = json.loads((tmp_path / plan["job"] / "constructs" / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert constructs["body"]["text"]["format"]["name"] == "construct_extraction"


def test_the_limit_caps_the_papers_and_all_three_kinds_get_the_same_ones(pworld, fresh, tmp_path):
    plan = prepared(pworld, fresh, tmp_path, limit=4)
    ids = {kind: set(json.loads((tmp_path / plan["job"] / kind / "manifest.json").read_text(encoding="utf-8"))["requests"]) for kind in pp.KINDS}
    assert all(plan["kinds"][k]["requests"] == 4 for k in pp.KINDS) and ids["method"] == ids["impact"] == ids["constructs"]


# --- the ceiling and the record of what was sent ----------------------------------------------------------------------------------------------------

def test_nothing_is_sent_above_the_ceiling(pworld, fresh, tmp_path, fake_submit):
    plan = prepared(pworld, fresh, tmp_path)
    with pytest.raises(ValueError, match="above the ceiling"):
        pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"] - 1)
    with pytest.raises(ValueError, match="whole number"):
        pp.submit_job(fresh, plan["directory"], 0)
    assert fake_submit == [] and fresh.execute("SELECT count(*) FROM processing_item").fetchone()[0] == 0


def test_no_key_no_submission(pworld, fresh, tmp_path, fake_submit, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    plan = prepared(pworld, fresh, tmp_path)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"])
    assert fake_submit == []


def test_a_click_submits_three_batches_and_records_every_paper_so_a_second_click_has_nothing_to_send(pworld, fresh, tmp_path, fake_submit):
    plan = prepared(pworld, fresh, tmp_path)
    submitted = pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"])
    assert fake_submit == ["constructs", "method", "impact"] and set(submitted) == set(pp.KINDS)
    recorded = dict(fresh.execute("SELECT kind,count(*) FROM processing_item GROUP BY kind").fetchall())
    assert recorded == {k: plan["kinds"][k]["requests"] for k in pp.KINDS}
    assert all(not v for v in pp.pending(fresh, pworld.ingestion_id).values())
    with pytest.raises(ValueError, match="already been sent"):
        prepared(pworld, fresh, tmp_path)
    assert pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"]) == submitted and fake_submit == ["constructs", "method", "impact"]      # a repeat sends nothing


def test_new_papers_are_the_only_ones_a_later_click_sends(pworld, fresh, tmp_path, fake_submit):
    first = prepared(pworld, fresh, tmp_path, limit=10)
    pp.submit_job(fresh, first["directory"], first["estimated_input_tokens"])
    second = prepared(pworld, fresh, tmp_path / "later", limit=10)
    sent = {kind: set(json.loads((tmp_path / first["job"] / kind / "manifest.json").read_text(encoding="utf-8"))["requests"]) for kind in pp.KINDS}
    again = {kind: set(json.loads((tmp_path / "later" / second["job"] / kind / "manifest.json").read_text(encoding="utf-8"))["requests"]) for kind in pp.KINDS}
    assert all(sent[k] and again[k] and not sent[k] & again[k] for k in pp.KINDS)


def test_a_failed_submission_keeps_what_was_sent_and_a_retry_sends_only_the_rest(pworld, fresh, tmp_path, fake_submit, monkeypatch):
    plan = prepared(pworld, fresh, tmp_path)
    good = pp.submit

    def flaky(directory):
        if directory.name == "method":
            raise RuntimeError("the provider refused")
        return good(directory)
    monkeypatch.setattr(pp, "submit", flaky)
    with pytest.raises(RuntimeError):
        pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"])
    assert dict(fresh.execute("SELECT kind,count(*) FROM processing_item GROUP BY kind").fetchall()) == {"constructs": plan["kinds"]["constructs"]["requests"]}
    monkeypatch.setattr(pp, "submit", good)
    pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"])
    assert fake_submit == ["constructs", "method", "impact"]                       # constructs once (before the failure), then only the two that were left
    assert set(dict(fresh.execute("SELECT kind,count(*) FROM processing_item GROUP BY kind").fetchall())) == set(pp.KINDS)


# --- the cards that come back -----------------------------------------------------------------------------------------------------------------------

def batch_line(custom_id, card):
    body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(card)}]}]}
    return json.dumps({"custom_id": custom_id, "response": {"status_code": 200, "body": body}})


def card_from(source, missing=()):
    sentences = [s.strip() + "." for s in source.replace("\n", " ").split(".") if len(s.strip()) > 40]
    fields = pp.PROFILES["method"].fields
    return {**{f: ({"text": None, "status": "missing", "passages": []} if f in missing else {"text": f"descrizione di {f}", "status": "extracted", "passages": [sentences[i % len(sentences)]]})
               for i, f in enumerate(fields)}, "info_gaps": []}


def collect_setup(pworld, fresh, tmp_path, fake_submit, answers):
    plan = prepared(pworld, fresh, tmp_path, limit=3)
    directory = tmp_path / plan["job"] / "method"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted(manifest["requests"])
    sources = {str(v): paper_cards.method_source(t or "")[0] for v, t in fresh.execute("SELECT version_id,full_text FROM paper_version WHERE version_id = ANY(%s)", (ids,))}
    (directory / "batch_output.jsonl").write_text("\n".join(answers(ids, sources)) + "\n", encoding="utf-8")
    return directory, ids, sources


def test_an_inexact_quotation_demotes_its_field_and_a_card_that_loses_most_of_its_fields_is_rejected(pworld, fresh, tmp_path, fake_submit):
    def answers(ids, sources):
        one_bad = card_from(sources[ids[1]])
        one_bad["design"]["passages"] = ["a sentence that the paper never contained, invented by the model"]
        mostly_bad = card_from(sources[ids[2]])
        for name in list(mostly_bad)[:5]:
            mostly_bad[name]["passages"] = ["an invented sentence number %d that is not in the paper" % len(name)]
        return [batch_line(ids[0], card_from(sources[ids[0]])), batch_line(ids[1], one_bad), batch_line(ids[2], mostly_bad)]
    directory, ids, _ = collect_setup(pworld, fresh, tmp_path, fake_submit, answers)
    out = pp.collect_cards(fresh, directory)
    assert out == {"kind": "method", "valid": 2, "rejected": 1, "failed_requests": 0, "fields_demoted": 1, "missing_requests": 0}
    rows = {str(v): (s, r, c, i) for v, s, r, c, i in fresh.execute("SELECT version_id,status,rejection,card,source_info FROM paper_card WHERE kind='method'")}
    assert rows[ids[0]][0] == "valid" and rows[ids[0]][3]["demoted_fields"] == []
    status, _, card, info = rows[ids[1]]
    assert status == "valid" and info["demoted_fields"] == ["design"] and card["design"] == {"text": None, "status": "missing", "passages": []}
    assert any(g.startswith("design:") and "not asserted" in g for g in card["info_gaps"]) and card["measures"]["status"] == "extracted"       # the other fields are untouched
    assert rows[ids[2]][0] == "rejected" and "fields have an exact quotation" in rows[ids[2]][1]


def test_collecting_twice_stores_nothing_new(pworld, fresh, tmp_path, fake_submit):
    directory, ids, _ = collect_setup(pworld, fresh, tmp_path, fake_submit, lambda ids, sources: [batch_line(i, card_from(sources[i])) for i in ids])
    pp.collect_cards(fresh, directory)
    pp.collect_cards(fresh, directory)
    assert fresh.execute("SELECT count(*) FROM paper_card").fetchone()[0] == 3


def test_a_changed_text_or_an_unreadable_or_failed_answer_never_becomes_a_valid_card(pworld, fresh, tmp_path, fake_submit):
    def answers(ids, sources):
        garbage = json.dumps({"custom_id": ids[1], "response": {"status_code": 200, "body": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "not json"}]}]}}})
        failed = json.dumps({"custom_id": ids[2], "error": {"message": "boom"}, "response": None})
        return [batch_line(ids[0], card_from(sources[ids[0]])), garbage, failed]
    directory, ids, _ = collect_setup(pworld, fresh, tmp_path, fake_submit, answers)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requests"][ids[0]]["source_sha256"] = "0" * 64                  # the text the request was made from is no longer the text in the database
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = pp.collect_cards(fresh, directory)
    assert out["valid"] == 0 and out["rejected"] == 2 and out["failed_requests"] == 1
    reasons = {str(v): r for v, r in fresh.execute("SELECT version_id,rejection FROM paper_card")}
    assert "changed" in reasons[ids[0]] and "could not be read" in reasons[ids[1]]


def test_a_card_for_a_foreign_custom_id_stops_the_collection(pworld, fresh, tmp_path, fake_submit):
    directory, ids, sources = collect_setup(pworld, fresh, tmp_path, fake_submit, lambda ids, sources: [batch_line(ids[0], card_from(sources[ids[0]]))])
    with (directory / "batch_output.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(batch_line("00000000-0000-0000-0000-000000000000", card_from(sources[ids[0]])) + "\n")
    with pytest.raises(ValueError, match="Unexpected or repeated"):
        pp.collect_cards(fresh, directory)


def test_a_paper_with_a_valid_card_is_not_pending_for_that_kind_any_more(pworld, fresh, tmp_path, fake_submit):
    directory, ids, _ = collect_setup(pworld, fresh, tmp_path, fake_submit, lambda ids, sources: [batch_line(i, card_from(sources[i])) for i in ids])
    before = {str(v) for v, _ in pp.pending(fresh, pworld.ingestion_id)["method"]}
    pp.collect_cards(fresh, directory)
    after = {str(v) for v, _ in pp.pending(fresh, pworld.ingestion_id)["method"]}
    assert set(ids) <= before and not set(ids) & after and {str(v) for v, _ in pp.pending(fresh, pworld.ingestion_id)["impact"]} >= set(ids)


# --- state and refresh --------------------------------------------------------------------------------------------------------------------------------

def test_the_status_reads_the_files_and_refresh_collects_only_the_batches_that_ended(pworld, fresh, tmp_path, fake_submit, monkeypatch):
    plan = prepared(pworld, fresh, tmp_path, limit=3)
    pp.submit_job(fresh, plan["directory"], plan["estimated_input_tokens"])
    status = pp.job_status(plan["directory"])
    assert status["kinds"]["method"]["batch_id"] == "batch_method" and "collected" not in status["kinds"]["method"] and status["kinds"]["constructs"]["requests"] == 3
    calls = []
    monkeypatch.setattr(extraction, "collect_remote", lambda con, d, staged=False: calls.append(("constructs", staged)) or {"status": "in_progress", "papers_answered": 0})
    monkeypatch.setattr(pp, "collect_cards_remote", lambda con, d: calls.append((d.name, None)) or {"status": "completed", "valid": 3, "rejected": 0})
    after = pp.refresh_job(fresh, plan["directory"])
    assert ("constructs", True) in calls                                         # constructs are always collected into staging
    assert "collected" not in after["kinds"]["constructs"] and after["kinds"]["method"]["collected"]["valid"] == 3 and after["kinds"]["impact"]["collected"]["status"] == "completed"
    calls.clear()
    pp.refresh_job(fresh, plan["directory"])
    assert calls == [("constructs", True)]                                       # the collected ones are not asked again


def test_the_latest_job_is_the_newest_directory(tmp_path):
    assert pp.latest_job(tmp_path) is None
    for name in ("20261018T100000000000Z", "20261018T120000000000Z"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "job.json").write_text("{}", encoding="utf-8")
    assert pp.latest_job(tmp_path).name == "20261018T120000000000Z"


# --- the background job -------------------------------------------------------------------------------------------------------------------------------

def test_the_background_job_refuses_a_bad_request_before_starting(monkeypatch):
    job = pp.ProcessingJob()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        job.start("run", None, 1000, "postgresql://x", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    with pytest.raises(ValueError, match="ceiling"):
        job.start("run", None, None, "postgresql://x", None)
    with pytest.raises(ValueError, match="Limit"):
        job.start("run", 0, 1000, "postgresql://x", None)
    assert job.snapshot() == {"state": "idle"}


def test_the_background_job_reports_a_failure_to_the_page(monkeypatch):
    job = pp.ProcessingJob()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    job._run("run", None, 1000, "postgresql://nobody@127.0.0.1:1/none", None)
    state = job.snapshot()
    assert state["state"] == "failed" and state["error"] and state["finished_at"]


# --- through the dashboard server ----------------------------------------------------------------------------------------------------------------------

import threading
import time
import urllib.error
import urllib.request


def http(port, method, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def server(pworld, fresh, tmp_path, monkeypatch):
    """The dashboard server on the mock schema; PROCESSING_DIR points into tmp_path and the S3 store is the mock world's memory store."""
    from research0 import dashboard_server, fulltext
    monkeypatch.setenv("DATABASE_URL", pworld.world.url)
    monkeypatch.setenv("PROCESSING_DIR", str(tmp_path / "processing"))
    monkeypatch.setattr(fulltext, "S3Store", lambda: pworld.world.store)
    started = []

    def start(read_only=False):
        srv, run_id = dashboard_server.serve(str(pworld.world.run_id), 0, read_only=read_only)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv.server_address[1], str(run_id)
    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def test_the_server_reports_the_state_and_the_estimate_without_sending_anything(server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    port, run_id = server()
    status, out = http(port, "GET", "/api/papers/status")
    assert status == 200 and out["job"] == {"state": "idle"} and out["last"] is None and out["openai_configured"] is False and out["read_only"] is False
    status, est = http(port, "GET", "/api/papers/estimate")
    assert status == 200 and set(est["kinds"]) == set(pp.KINDS) and est["kinds"]["method"]["papers"] > 0 and est["estimated_input_tokens"] > 0
    assert http(port, "GET", "/api/papers/estimate?limit=2")[1]["kinds"]["impact"]["papers"] == 2


def test_the_click_is_refused_without_a_key_with_a_bad_ceiling_and_on_a_read_only_deployment(server, monkeypatch, fake_submit):
    monkeypatch.delenv("OPENAI_API_KEY")
    port, run_id = server()
    status, out = http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 2, "max_input_tokens": 10**9})
    assert status == 400 and "OPENAI_API_KEY" in out["error"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    assert http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 2, "max_input_tokens": 0})[0] == 400
    assert http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 2})[0] == 400
    assert http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 0, "max_input_tokens": 10**9})[0] == 400
    assert http(port, "POST", "/api/papers/process", {"run_id": "00000000-0000-0000-0000-000000000000", "limit": 2, "max_input_tokens": 10**9})[0] == 409
    ro_port, ro_run = server(read_only=True)
    assert http(ro_port, "GET", "/api/papers/status")[1]["read_only"] is True
    assert http(ro_port, "POST", "/api/papers/process", {"run_id": ro_run, "limit": 2, "max_input_tokens": 10**9})[0] == 403
    assert http(ro_port, "POST", "/api/papers/refresh", {"run_id": ro_run})[0] == 403
    assert fake_submit == []


def wait_finished(port):
    for _ in range(300):
        state = http(port, "GET", "/api/papers/status")[1]["job"]
        if state["state"] != "running":
            return state
        time.sleep(0.1)
    raise AssertionError("the job did not finish")


def test_one_click_through_the_server_submits_the_three_batches_and_a_second_click_has_nothing_to_send(server, pworld, fresh, fake_submit):
    port, run_id = server()
    status, out = http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 3, "max_input_tokens": 10**9})
    assert status == 200 and out["state"] == "running"
    state = wait_finished(port)
    assert state["state"] == "finished" and state["plan"]["kinds"]["method"]["requests"] == 3 and sorted(fake_submit) == sorted(pp.KINDS)
    last = http(port, "GET", "/api/papers/status")[1]["last"]
    assert {k: v["batch_id"] for k, v in last["kinds"].items()} == {k: f"batch_{k}" for k in pp.KINDS}
    assert dict(fresh.execute("SELECT kind,count(*) FROM processing_item GROUP BY kind").fetchall()) == {k: 3 for k in pp.KINDS}
    http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 3, "max_input_tokens": 10**9})
    again = wait_finished(port)
    assert again["state"] == "finished" and fake_submit.count("method") == 2                       # the next 3 papers, not the same 3
    assert dict(fresh.execute("SELECT kind,count(*) FROM processing_item GROUP BY kind").fetchall()) == {k: 6 for k in pp.KINDS}


def test_a_click_above_the_ceiling_prepares_but_sends_nothing_and_says_so(server, fresh, fake_submit):
    port, run_id = server()
    http(port, "POST", "/api/papers/process", {"run_id": run_id, "limit": 3, "max_input_tokens": 1000})
    state = wait_finished(port)
    assert state["state"] == "failed" and "above the ceiling" in state["error"] and fake_submit == []
    assert fresh.execute("SELECT count(*) FROM processing_item").fetchone()[0] == 0


def test_refresh_without_a_job_is_an_error_the_page_can_show(server, fake_submit):
    port, run_id = server()
    status, out = http(port, "POST", "/api/papers/refresh", {"run_id": run_id})
    assert status == 400 and "No job" in out["error"]


# --- a Batch submitted before this module existed ------------------------------------------------------------------------------------------------------

def test_a_batch_sent_before_the_record_existed_can_be_registered_so_its_papers_are_not_sent_again(pworld, fresh, tmp_path, fake_submit):
    plan = prepared(pworld, fresh, tmp_path)
    directory = tmp_path / plan["job"] / "constructs"
    with pytest.raises(ValueError, match="nothing was submitted"):
        pp.register_sent(fresh, directory, "constructs")
    pp.submit(directory)                                                                   # the stand-in writes a batch.json, as the real submission did in October
    assert fresh.execute("SELECT count(*) FROM processing_item").fetchone()[0] == 0
    n = plan["kinds"]["constructs"]["requests"]
    assert pp.register_sent(fresh, directory, "constructs") == n and pp.register_sent(fresh, directory, "constructs") == 0
    todo = pp.pending(fresh, pworld.ingestion_id)
    assert todo["constructs"] == [] and len(todo["method"]) == len(todo["impact"]) == n
    with pytest.raises(ValueError, match="does not cover this prompt version"):
        pp.register_sent(fresh, directory, "method")
