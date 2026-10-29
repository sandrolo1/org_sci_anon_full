"""Human review progress and an explicitly simulated dashboard preview."""

import copy
import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from .dashboard import render


def latest_decisions(entries):
    latest = {}
    for entry in sorted(entries, key=lambda value: value.get("at", "")):
        latest[entry["validator"]] = entry
    return [entry for entry in latest.values() if entry["decision"] != "withdrawn"]


def item_status(entries):
    active = latest_decisions(entries)
    substantive = {entry["decision"] for entry in active if entry["decision"] != "needs_discussion"}
    if len(substantive) > 1:
        return "conflict"
    if any(entry["decision"] == "needs_discussion" for entry in active):
        return "discussion"
    return "decided" if substantive else "open"


def review_metrics(snapshot):
    decisions = snapshot.get("decisions", {})
    statuses = {item["id"]: item_status(decisions.get(item["id"], [])) for item in snapshot["items"]}
    counts = Counter(statuses.values())
    by_type = Counter(item["type"] for item in snapshot["items"])
    by_severity = Counter(item["severity"] for item in snapshot["items"] if statuses[item["id"]] == "open")
    validators = {entry["validator"] for item in snapshot["items"] for entry in latest_decisions(decisions.get(item["id"], []))}
    total = len(snapshot["items"])
    return {"total": total, "open": counts["open"], "decided": counts["decided"],
            "discussion": counts["discussion"], "conflict": counts["conflict"],
            "completion": counts["decided"] / total if total else 1,
            "open_high": by_severity["high"], "validators": len(validators),
            "by_type": dict(by_type)}


def pilot_preview(run_dir: Path, assignments_path: Path | None = None):
    """Read prepared pilot papers; simulate labels only when no collected assignments exist."""
    assignments = {}
    if assignments_path:
        with assignments_path.open(encoding="utf-8-sig", newline="") as stream:
            assignments = {row["record_id"]: row for row in csv.DictReader(stream)}
    papers = []
    with (run_dir / "requests.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            request = json.loads(line)
            for paper in json.loads(request["body"]["input"]):
                record_id = paper["record_id"]
                row = assignments.get(record_id)
                if row is None and assignments_path:
                    continue
                if row is None:
                    digest = hashlib.sha256(record_id.encode()).digest()
                    row = {"cto": str(digest[0] % 2), "tim": str(digest[1] % 2),
                           "cto_rationale": "Simulated placeholder; assess the paper against the CTO scope.",
                           "tim_rationale": "Simulated placeholder; assess the paper against the TIM scope."}
                papers.append({"record_id": record_id, "title": paper.get("title", ""),
                               "abstract": paper.get("abstract", ""), "cto": int(row["cto"]), "tim": int(row["tim"]),
                               "cto_rationale": row["cto_rationale"], "tim_rationale": row["tim_rationale"]})
    return {"papers": papers, "simulated": assignments_path is None, "source": str(run_dir)}


def demo_snapshot(snapshot, pilot=None):
    """Return a preview without altering or purporting to resolve actual review work."""
    preview = copy.deepcopy(snapshot)
    preview["demo"] = True
    preview["live"] = False
    preview["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    preview["decisions"] = {}
    if pilot is not None:
        preview["pilot"] = pilot
    for index, item in enumerate(preview["items"]):
        if index % 5 == 0:
            continue
        decision = "needs_discussion" if index % 7 == 0 else next(
            (value for value in item["decisions"] if value != "needs_discussion"), item["decisions"][0])
        preview["decisions"][item["id"]] = [{"validator": "Demo reviewer", "decision": decision,
                                                "note": "Simulated example; verify the evidence before a real decision.",
                                                "at": preview["generated_at"]}]
    preview["limits"] = ["DEMO: review decisions are simulated and are not scientific or human validation."] + preview.get("limits", [])
    return preview


def build_demo(snapshot_path: Path, output: Path, pilot_run: Path | None = None, assignments: Path | None = None):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    preview = demo_snapshot(snapshot, pilot_preview(pilot_run, assignments) if pilot_run else None)
    page = render(preview, output)
    return {"page": str(page), "items": len(preview["items"]), "demo": True}
