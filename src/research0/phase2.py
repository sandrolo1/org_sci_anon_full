"""Phase 2: reproducible human review packet from a catalog dashboard snapshot."""

import csv
import json
import re
from collections import Counter
from pathlib import Path


def _csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_review_packet(snapshot_path: Path, output: Path, decisions_path: Path | None = None):
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    run = snapshot["run"]["run_id"]
    items = snapshot["items"]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("Snapshot contains duplicate item IDs")
    decisions = {item["id"]: [] for item in items}
    for item_id, entries in snapshot.get("decisions", {}).items():
        if item_id in decisions:
            decisions[item_id].extend(entries)
    if decisions_path:
        with decisions_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["run_id"] != run:
                    raise ValueError("Decision CSV belongs to a different catalog run")
                if row["item_id"] not in decisions:
                    raise ValueError(f"Unknown decision item: {row['item_id']}")
                item = next(i for i in items if i["id"] == row["item_id"])
                if row["decision"] not in item["decisions"] + ["withdrawn"]:
                    raise ValueError(f"Invalid decision for {row['item_id']}")
                if not row["validator"].strip():
                    raise ValueError("Every decision needs a validator")
                decisions[row["item_id"]].append({"validator": row["validator"].strip(), "decision": row["decision"],
                                                  "note": row.get("note", ""), "at": row.get("decided_at", "")})
    rows = []
    for item in items:
        latest = {}
        for entry in sorted(decisions[item["id"]], key=lambda e: e.get("at", "")):
            latest[entry["validator"]] = entry
        active = [entry for entry in latest.values() if entry["decision"] != "withdrawn"]
        substantive = {entry["decision"] for entry in active if entry["decision"] != "needs_discussion"}
        status = "conflicting_decisions" if len(substantive) > 1 else "discussion" if any(
            entry["decision"] == "needs_discussion" for entry in active) else "recorded" if substantive else "open"
        rows.append({"run_id": run, "item_id": item["id"], "type": item["type"], "severity": item["severity"],
                     "title": item["title"], "status": status, "decision": next(iter(substantive)) if status == "recorded" else "",
                     "validators": "; ".join(sorted(entry["validator"] for entry in active)),
                     "evidence": json.dumps(item.get("evidence", []), ensure_ascii=False),
                     "allowed_decisions": "; ".join(item["decisions"])})
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "review_queue.csv", list(rows[0]) if rows else ["run_id", "item_id"], rows)
    srcids = re.findall(r"SRCID\((\d+)\)", snapshot["run"].get("query_text", ""))
    _csv(output / "source_validation_worksheet.csv",
         ["source_id", "journal_name", "inclusion_rationale", "active", "version", "validated_on", "validated_by"],
         [{"source_id": sid, "journal_name": "", "inclusion_rationale": "", "active": "", "version": "",
           "validated_on": "", "validated_by": ""} for sid in dict.fromkeys(srcids)])
    counts = Counter(row["status"] for row in rows)
    by_type = Counter(row["type"] for row in rows)
    gates = snapshot.get("gates", [])
    lines = ["# Phase 2 — human review packet", "", f"Catalog run: `{run}`", "",
             f"Snapshot: `{snapshot_path.name}`", "", "## Retrieval benchmark", "",
             "| Community | Recovered | Coverage |", "|---|---:|---:|"]
    lines += [f"| {g['community']} | {g['found']}/{g['eligible']} | {g['coverage']:.1%} |" for g in gates]
    lines += ["", "## Review queue", "", f"{len(rows)} items: {counts['open']} open, {counts['recorded']} with a recorded decision, "
              f"{counts['discussion']} for discussion, {counts['conflicting_decisions']} with conflicting decisions.", ""]
    lines += [f"- {kind}: {count}" for kind, count in sorted(by_type.items())]
    lines += ["", "## Actions", "", "1. Verify the three unrecovered benchmark seeds in Scopus and document the cause.",
              "2. Resolve each deduplication conflict from the paired records; keep the original exports intact.",
              f"3. Validate all {len(dict.fromkeys(srcids))} query SRCIDs in `source_validation_worksheet.csv`: journal, rationale, version, date and validator.",
              "4. Review the journal-name variant and records without DOI.",
              "5. Record decisions in the live dashboard, or supply its downloaded CSV to this command. A recorded decision documents review; it does not alter the catalog or certify the source list.",
              "", "Phase 3 remains pending human sign-off on the review and validated source list.", ""]
    (output / "phase2_review.md").write_text("\n".join(lines), encoding="utf-8")
    return {"run_id": run, "items": len(rows), "statuses": dict(counts), "source_ids": len(dict.fromkeys(srcids))}
