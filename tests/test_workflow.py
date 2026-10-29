import json

from research0.workflow import build_demo, demo_snapshot, item_status, pilot_preview, review_metrics


def test_review_metrics_distinguish_resolution_discussion_and_conflict():
    items = [{"id": str(i), "type": "seed", "severity": "high"} for i in range(4)]
    decisions = {
        "1": [{"validator": "A", "decision": "found", "at": "1"}],
        "2": [{"validator": "A", "decision": "needs_discussion", "at": "1"}],
        "3": [{"validator": "A", "decision": "found", "at": "1"},
              {"validator": "B", "decision": "missing", "at": "2"}],
    }
    metrics = review_metrics({"items": items, "decisions": decisions})
    assert [metrics[k] for k in ("open", "decided", "discussion", "conflict", "open_high")] == [1, 1, 1, 1, 1]
    assert metrics["completion"] == .25
    assert item_status([{"validator": "A", "decision": "found", "at": "1"},
                        {"validator": "A", "decision": "withdrawn", "at": "2"}]) == "open"


def test_demo_is_explicit_and_does_not_change_source(tmp_path):
    source = {"run": {"run_id": "sample"}, "items": [
        {"id": str(i), "decisions": ["yes", "needs_discussion"]} for i in range(8)],
        "decisions": {}, "limits": []}
    original = json.dumps(source)
    preview = demo_snapshot(source)
    assert json.dumps(source) == original
    assert preview["demo"] is True and preview["live"] is False
    assert len(preview["decisions"]) > 0
    assert all("Simulated" in entry[0]["note"] for entry in preview["decisions"].values())
    path = tmp_path / "source.json"
    path.write_text(original, encoding="utf-8")
    result = build_demo(path, tmp_path / "site")
    assert result["demo"] is True
    assert (tmp_path / "site" / "index.html").exists()


def test_pilot_preview_separates_simulated_from_collected_assignments(tmp_path):
    run = tmp_path / "pilot"
    run.mkdir()
    request = {"body": {"input": json.dumps([{"record_id": "paper-1", "title": "Example", "abstract": "Text"}])}}
    (run / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")
    simulated = pilot_preview(run)
    assert simulated["simulated"] is True
    assert "Simulated" in simulated["papers"][0]["cto_rationale"]
    assignments = run / "assignments.csv"
    assignments.write_text("record_id,cto,tim,cto_rationale,tim_rationale\npaper-1,1,0,Reason CTO,Reason TIM\n", encoding="utf-8")
    collected = pilot_preview(run, assignments)
    assert collected["simulated"] is False
    assert collected["papers"][0]["cto"] == 1
    assert collected["papers"][0]["tim_rationale"] == "Reason TIM"
