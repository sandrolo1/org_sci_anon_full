import json

import pytest

from research0.community_recovery import assignment_rows, prepare_retry


def fixture_run(tmp_path, status="expired"):
    root = tmp_path / "run"
    root.mkdir()
    manifest = {"model": "test", "prompt_version": "v1", "scope_version": "v1", "scope_sha256": "sha",
                "requests": {"group-a": ["a", "b"], "group-b": ["c", "d"]}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    requests = [{"custom_id": key, "body": {"input": ids}} for key, ids in manifest["requests"].items()]
    (root / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
    (root / "batch.json").write_text(json.dumps({"status": status, "output_file_id": "output"}), encoding="utf-8")
    (root / "assignments.csv").write_text("record_id,cto,tim\na,1,0\nb,0,1\n", encoding="utf-8")
    return root


def test_retry_contains_only_missing_request_and_keeps_completed(tmp_path):
    root = fixture_run(tmp_path)
    result = prepare_retry(root)
    assert (result["missing_requests"], result["missing_papers"], result["assigned"]) == (1, 2, 2)
    retry = root / "retry_001"
    assert [json.loads(line)["custom_id"] for line in (retry / "requests.jsonl").read_text().splitlines()] == ["group-b"]
    assert json.loads((retry / "manifest.json").read_text())["requests"] == {"group-b": ["c", "d"]}
    assert not (retry / "batch.json").exists()  # preparation does not submit
    (retry / "batch.json").write_text(json.dumps({"status": "completed", "output_file_id": "output-2"}), encoding="utf-8")
    (retry / "assignments.csv").write_text("record_id,cto,tim\nc,1,1\nd,0,0\n", encoding="utf-8")
    assert len(assignment_rows(root)) == 4
    assert prepare_retry(root)["ready"] is False


def test_retry_refuses_active_batch_and_missing_collected_output(tmp_path):
    root = fixture_run(tmp_path, "in_progress")
    with pytest.raises(ValueError, match="still active"):
        prepare_retry(root)
    (root / "batch.json").write_text(json.dumps({"status": "expired", "output_file_id": "output"}), encoding="utf-8")
    (root / "assignments.csv").unlink()
    with pytest.raises(ValueError, match="Collect the available output"):
        prepare_retry(root)
