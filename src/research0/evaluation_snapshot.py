"""Build a private, read-only HTML view of a dated forum evaluation export.

The generated page embeds unpublished proposals. Write it under ignored
``reports/`` and do not publish it or commit it to Git.
"""

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


TEMPLATE = Path(__file__).with_name("evaluation_snapshot_template.html")
MARKER = "/*__DATA__*/null"


def followup_for(flags, novelty):
    """Turn mechanical audit findings into explicit next steps, not quality labels."""
    flags = set(flags)
    hybrid = (novelty or {}).get("hybrid") or {}
    delta = None
    if isinstance((novelty or {}).get("vote"), (int, float)) and isinstance(hybrid.get("vote"), (int, float)):
        delta = round(hybrid["vote"] - novelty["vote"], 2)
    if "hybrid:failed" in flags:
        return {"kind": "rerun", "label": "Hybrid check failed",
                "action": "Rerun the hybrid judge with valid cited passages; retain the failed attempt in the record.",
                "detail": hybrid.get("reason") or "No valid hybrid judgment was retained.", "delta": delta}
    if "hybrid:shift_at_least_two" in flags:
        return {"kind": "inspect", "label": "Novelty score shifted",
                "action": "Compare both rationales and cited papers; repeat with a frozen rubric before interpreting the shift.",
                "detail": f"The hybrid vote differs from the primary novelty vote by {delta:+g} points." if delta is not None else "Large vote difference.",
                "delta": delta}
    if flags:
        return {"kind": "inspect", "label": "Integrity check flagged",
                "action": "Inspect the listed source or arithmetic mismatch before using this evaluation.",
                "detail": "; ".join(sorted(flags)), "delta": delta}
    return {"kind": "none", "label": "No mechanical flag", "action": "", "detail": "", "delta": delta}


def retrieval_diagnostics(novelty):
    """Count retained papers and newly added papers used in hybrid comparisons."""
    novelty = novelty or {}
    hybrid = novelty.get("hybrid") or {}
    primary_ids = {paper.get("id") for paper in novelty.get("neighbours") or []}
    candidates = hybrid.get("candidates") or []
    added = {paper.get("label"): paper for paper in candidates if paper.get("id") not in primary_ids}
    compared = {comparison.get("neighbour_id") for judge in (hybrid.get("judges") or {}).values()
                if isinstance(judge, dict) for comparison in judge.get("comparisons") or []}
    added_compared = [paper for label, paper in added.items() if label in compared]
    return {"primary_retained": sum(paper.get("id") in primary_ids for paper in candidates),
            "added_candidates": len(added), "added_compared": len(added_compared),
            "added_compared_titles": [paper.get("title") or "Untitled" for paper in added_compared]}


def _json_member(archive, suffix):
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected exactly one {suffix} in export")
    return json.loads(archive.read(names[0]).decode("utf-8-sig"))


def load_payload(zip_path, *, integrity_aggregate=None, integrity_cases=None):
    source = Path(zip_path)
    with zipfile.ZipFile(source) as archive:
        proposals = _json_member(archive, "proposals.json")
        evaluations = _json_member(archive, "evaluations.json")
    if not isinstance(proposals, list) or not isinstance(evaluations, list):
        raise ValueError("proposals and evaluations must be lists")
    by_id = {row.get("proposal_id"): row for row in evaluations}
    if len(by_id) != len(evaluations) or None in by_id:
        raise ValueError("evaluation IDs must be unique and present")
    ids = [row.get("proposal_id") for row in proposals]
    if not ids or None in ids or len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise ValueError("proposal and evaluation IDs must match one-to-one")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    integrity = None
    flags_by_id = {}
    if bool(integrity_aggregate) != bool(integrity_cases):
        raise ValueError("provide both integrity files or neither")
    if integrity_aggregate:
        integrity = json.loads(Path(integrity_aggregate).read_text(encoding="utf-8"))
        if integrity.get("source_sha256") != source_hash:
            raise ValueError("integrity report was produced from another export")
        cases = json.loads(Path(integrity_cases).read_text(encoding="utf-8"))
        flags_by_id = {row["proposal_id"]: row["flags"] for row in cases}
        if len(flags_by_id) != len(cases) or not set(flags_by_id).issubset(set(ids)):
            raise ValueError("integrity case IDs are duplicated or missing from export")
    items = []
    for row in proposals:
        detail = by_id[row["proposal_id"]]["evaluation"]
        if detail.get("proposal_id") != row["proposal_id"]:
            raise ValueError("evaluation document belongs to another proposal")
        score = row["evaluation"]
        flags = flags_by_id.get(row["proposal_id"], [])
        novelty = (detail.get("criteria") or {}).get("novelty") or {}
        items.append({"id": row["proposal_id"], "title": row.get("title"), "abstract": row.get("abstract"),
                      "condition": row.get("condition"), "trigger": row.get("trigger"),
                      "episode": row.get("episode_title"), "warnings": row.get("warnings") or [],
                      "mean": score.get("mean_of_three"), "votes": score.get("votes") or {},
                      "scored_on": score.get("scored_on"), "hybrid_vote": score.get("novelty_hybrid_vote"),
                      "criteria": detail.get("criteria") or {}, "cards": row.get("cards") or {},
                      "integrity_flags": flags, "followup": followup_for(flags, novelty),
                      "retrieval_diagnostics": retrieval_diagnostics(novelty),
                      "schema": detail.get("schema")})
    payload = {"source": source.name, "source_sha256": source_hash,
               "items": items, "integrity": integrity}
    return payload


def build(zip_path, output, *, integrity_aggregate=None, integrity_cases=None):
    output = Path(output)
    payload = load_payload(zip_path, integrity_aggregate=integrity_aggregate,
                           integrity_cases=integrity_cases)
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise ValueError("snapshot template marker is missing or duplicated")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(MARKER, encoded), encoding="utf-8")
    return {"proposals": len(payload["items"]), "hybrid_scored": sum(row["hybrid_vote"] is not None for row in payload["items"]),
            "output": str(output)}


def build_json(zip_path, output, *, integrity_aggregate=None, integrity_cases=None):
    """Write the same private payload for the local Vite app."""
    output = Path(output)
    payload = load_payload(zip_path, integrity_aggregate=integrity_aggregate,
                           integrity_cases=integrity_cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"proposals": len(payload["items"]), "output": str(output)}


def build_followup_csv(payload, output):
    """Export the exception queue for repeatable follow-up without publishing proposal text."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source_sha256", "proposal_id", "title", "condition", "issue", "primary_novelty",
              "hybrid_novelty", "delta", "primary_papers_retained", "added_papers_compared",
              "added_paper_titles_compared", "failure_or_shift_detail", "next_action")
    cases = [item for item in payload["items"] if item["followup"]["kind"] != "none"]
    cases.sort(key=lambda item: (0 if item["followup"]["kind"] == "rerun" else 1, item["title"] or ""))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in cases:
            followup = item["followup"]
            writer.writerow({"source_sha256": payload["source_sha256"], "proposal_id": item["id"],
                             "title": item["title"], "condition": item["condition"],
                             "issue": followup["label"], "primary_novelty": item["votes"].get("novelty"),
                             "hybrid_novelty": item["hybrid_vote"], "delta": followup["delta"],
                             "primary_papers_retained": item["retrieval_diagnostics"]["primary_retained"],
                             "added_papers_compared": item["retrieval_diagnostics"]["added_compared"],
                             "added_paper_titles_compared": "; ".join(item["retrieval_diagnostics"]["added_compared_titles"]),
                             "failure_or_shift_detail": followup["detail"], "next_action": followup["action"]})
    return {"cases": len(cases), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--integrity-aggregate", type=Path)
    parser.add_argument("--integrity-cases", type=Path)
    args = parser.parse_args(argv)
    if bool(args.output) == bool(args.output_json):
        parser.error("choose exactly one of --output and --output-json")
    command = build if args.output else build_json
    print(json.dumps(command(args.export, args.output or args.output_json,
                             integrity_aggregate=args.integrity_aggregate,
                             integrity_cases=args.integrity_cases)))


if __name__ == "__main__":
    main()
