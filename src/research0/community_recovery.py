"""Prepare a safe, explicit Batch retry using only requests without valid assignments."""
import csv
import json
from pathlib import Path


def assignment_rows(run_dir):
    """Merge validated outputs across attempts, refusing contradictory duplicates."""
    rows = {}
    attempts = [run_dir, *sorted(run_dir.glob("retry_[0-9][0-9][0-9]"))]
    for attempt in attempts:
        path = attempt / "assignments.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                paper_id = row["record_id"]
                if paper_id in rows and any(rows[paper_id][field] != row[field] for field in ("cto", "tim")):
                    raise ValueError(f"Conflicting assignments for paper {paper_id}")
                rows[paper_id] = row
    return rows


def prepare_retry(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("requests"):
        raise ValueError("Original request manifest is missing")
    attempts = [run_dir, *sorted(run_dir.glob("retry_[0-9][0-9][0-9]"))]
    for attempt in attempts:
        state_path = attempt / "batch.json"
        if not state_path.exists():
            raise ValueError(f"Attempt not submitted or not collected: {attempt}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in ("completed", "expired", "cancelled"):
            raise ValueError(f"Attempt is still active or failed validation ({state.get('status')}): {attempt}")
        if state.get("output_file_id") and not (attempt / "assignments.csv").exists():
            raise ValueError(f"Collect the available output before preparing a retry: {attempt}")
    completed = set(assignment_rows(run_dir))
    expected = {paper_id for ids in manifest["requests"].values() for paper_id in ids}
    if not completed <= expected:
        raise ValueError("Collected assignments contain papers outside the original manifest")
    missing = {}
    for custom_id, ids in manifest["requests"].items():
        found = sum(paper_id in completed for paper_id in ids)
        if 0 < found < len(ids):
            raise ValueError(f"Partial assignments within request {custom_id}; review manually")
        if found == 0:
            missing[custom_id] = ids
    if not missing:
        return {"ready": False, "reason": "All eligible papers have assignments", "assigned": len(completed)}
    attempt_number = len(attempts)
    output = run_dir / f"retry_{attempt_number:03d}"
    if output.exists():
        raise ValueError(f"Retry directory already exists: {output}")
    selected = set()
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item["custom_id"] in missing:
                if item["custom_id"] in selected:
                    raise ValueError(f"Repeated request ID: {item['custom_id']}")
                selected.add(item["custom_id"])
    if selected != set(missing):
        raise ValueError("Original request file does not match manifest")
    output.mkdir()
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as source, (output / "requests.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            if json.loads(line)["custom_id"] in missing:
                target.write(line)
    retry_manifest = {key: manifest[key] for key in ("model", "prompt_version", "scope_version", "scope_sha256")}
    retry_manifest.update({"parent_run": str(run_dir), "all_unique_records": len(expected),
                           "selected_records": sum(map(len, missing.values())),
                           "request_count": len(missing), "requests": missing})
    (output / "manifest.json").write_text(json.dumps(retry_manifest, indent=2) + "\n", encoding="utf-8")
    return {"ready": True, "directory": str(output), "assigned": len(completed),
            "missing_papers": retry_manifest["selected_records"], "missing_requests": len(missing)}
