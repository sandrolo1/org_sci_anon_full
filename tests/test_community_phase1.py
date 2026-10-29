import csv
import json

from research0.community_phase1 import collect, prepare


def write_export(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["EID", "DOI", "Title", "Year", "Abstract", "Author Keywords", "Index Keywords"])
        writer.writeheader()
        writer.writerows(rows)


def write_scope(path):
    path.write_text(json.dumps({"version": "test", "language": "en", "decision_rule": "Assess each independently.",
        "communities": {
            "CTO": {"name": "CTO", "source_url": "https://example.org/cto", "scope": "Technology and organizing"},
            "TIM": {"name": "TIM", "source_url": "https://example.org/tim", "scope": "Technology and innovation"},
        }}), encoding="utf-8")


def test_prepare_deduplicates_eid_and_quarantines_conflicting_doi(tmp_path):
    scope = tmp_path / "scope.json"
    write_scope(scope)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_export(first, [
        {"EID": "e1", "DOI": "10.1/a", "Title": "One", "Year": "2020", "Abstract": "Short"},
        {"EID": "e2", "DOI": "10.1/b", "Title": "Two", "Year": "2021", "Abstract": "B"},
    ])
    write_export(second, [
        {"EID": "e1", "DOI": "10.1/a", "Title": "One", "Year": "2020", "Abstract": "Longer abstract"},
        {"EID": "e3", "DOI": "10.1/b", "Title": "Three", "Year": "2022", "Abstract": "C"},
    ])
    run = tmp_path / "run"
    manifest = prepare([first, second], scope, run, batch_size=10)
    assert (manifest["all_unique_records"], manifest["ready_records"], manifest["selected_records"]) == (3, 1, 1)
    assert manifest["exports"][0]["rows"] == manifest["exports"][1]["rows"] == 2
    assert len(manifest["identity_conflicts"]) == 1
    request = json.loads((run / "requests.jsonl").read_text(encoding="utf-8"))
    paper = json.loads(request["body"]["input"])[0]
    assert paper["abstract"] == "Longer abstract"
    assert request["body"]["model"] == "gpt-6-luna"
    assert {"cto", "tim"}.issubset(request["body"]["text"]["format"]["schema"]["properties"]["assignments"]["items"]["properties"])


def test_collect_accepts_overlapping_membership_and_rejects_missing_assignment(tmp_path):
    scope = tmp_path / "scope.json"
    write_scope(scope)
    export = tmp_path / "export.csv"
    write_export(export, [{"EID": f"e{i}", "DOI": f"10.1/{i}", "Title": f"Paper {i}", "Year": "2020", "Abstract": "Abstract"} for i in range(2)])
    run = tmp_path / "run"
    manifest = prepare([export], scope, run)
    custom_id = next(iter(manifest["requests"]))
    ids = manifest["requests"][custom_id]
    assignments = [{"record_id": record_id, "cto": 1, "tim": 1, "cto_rationale": "Organization", "tim_rationale": "Innovation"} for record_id in ids]
    output = {"custom_id": custom_id, "response": {"status_code": 200, "body": {"output": [
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps({"assignments": assignments})}]}]}}}
    output_path = run / "output.jsonl"
    output_path.write_text(json.dumps(output) + "\n", encoding="utf-8")
    result = collect(run / "manifest.json", scope, output_path, run / "assignments.csv")
    assert result["assignments"] == 2
    with (run / "assignments.csv").open(encoding="utf-8", newline="") as stream:
        assert all(row["cto"] == row["tim"] == "1" for row in csv.DictReader(stream))
    output["response"]["body"]["output"][0]["content"][0]["text"] = json.dumps({"assignments": assignments[:1]})
    output_path.write_text(json.dumps(output) + "\n", encoding="utf-8")
    result = collect(run / "manifest.json", scope, output_path, run / "assignments.csv")
    assert result["assignments"] == 0
    assert result["failed_requests"]
