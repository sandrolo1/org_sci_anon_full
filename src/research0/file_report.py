"""Phase-0 coverage across export files, without building a canonical catalog."""
import csv
import json
from pathlib import Path

from .core import coverage, file_digest, mapped, match, norm_doi, norm_eid, read_scopus, stable_id


def load_seeds(path, community):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"seed_id", "community", "version", "doi", "eid", "title", "publication_year", "status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing benchmark columns in {path}")
        rows = list(reader)
    for n, row in enumerate(rows, 2):
        if row["community"] != community:
            raise ValueError(f"{path}:{n}: expected community {community}")
        if row["status"] not in ("eligible", "out_of_scope", "duplicate", "to_verify"):
            raise ValueError(f"{path}:{n}: unvalidated status {row['status']}")
        row["doi"] = norm_doi(row["doi"])
        row["eid"] = norm_eid(row["eid"])
        row["publication_year"] = int(row["publication_year"])
    if not rows:
        raise ValueError(f"No benchmark rows in {path}")
    return rows


def build_report(cto_path, tim_path, export_paths, output):
    seeds = load_seeds(cto_path, "CTO") + load_seeds(tim_path, "TIM")
    versions = {row["version"] for row in seeds}
    if len(versions) != 1:
        raise ValueError("CTO and TIM benchmarks must use one version")
    records = []
    exports = []
    for path in export_paths:
        sha = file_digest(path)
        count = 0
        for row_number, raw in read_scopus(path):
            rec = mapped(raw)
            rec["record_id"] = str(stable_id("file-record", sha, row_number))
            rec["export_file"] = path.name
            records.append(rec)
            count += 1
        exports.append({"file": str(path), "sha256": sha, "records": count})
    results = []
    by_id = {r["record_id"]: r for r in records}
    for community in ("CTO", "TIM"):
        for seed in (s for s in seeds if s["community"] == community):
            outcome, method, ids, cause = match(seed, records)
            results.append({"community": community, "seed_id": seed["seed_id"], "title": seed["title"],
                            "seed_status": seed["status"], "outcome": outcome, "match_method": method or "",
                            "candidate_record_ids": ";".join(ids),
                            "candidate_export_files": ";".join(sorted({by_id[x]["export_file"] for x in ids})),
                            "possible_cause": cause or ""})
    summary = {"benchmark_version": versions.pop(), "exports": exports, "combined_export_rows": len(records),
               "CTO": coverage([r for r in results if r["community"] == "CTO"]),
               "TIM": coverage([r for r in results if r["community"] == "TIM"]),
               "query_source_list_validated": False}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# CTO and TIM benchmark coverage", "", f"Benchmark version: {summary['benchmark_version']}",
             f"Rows across both exports: {len(records)}", "", "| Community | Found / eligible | Coverage | 95% target | Ambiguous |",
             "|---|---:|---:|---|---:|"]
    for community in ("CTO", "TIM"):
        stats = summary[community]
        pct = "N/A" if stats["coverage"] is None else f"{stats['coverage']:.1%}"
        target = "N/A" if stats["target_met"] is None else ("met" if stats["target_met"] else "not met")
        lines.append(f"| {community} | {stats['found']} / {stats['eligible']} | {pct} | {target} | {stats['ambiguous']} |")
    lines.extend(["", "## Missing or ambiguous seeds", ""])
    for r in results:
        if r["seed_status"] == "eligible" and r["outcome"] != "found":
            lines.append(f"- {r['community']} {r['seed_id']}: {r['title']} — {r['outcome']}. {r['possible_cause']}")
    if all(r["outcome"] == "found" for r in results if r["seed_status"] == "eligible"):
        lines.append("None.")
    lines.extend(["", "This comparison uses rows from both exports without creating a canonical catalog or modifying the original files.",
                  "The versioned source ID list has not been validated: coverage measures retrieval in the supplied files, not completeness of the Scopus query."])
    (output / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, results
