"""Normalize the evaluator's dated CSV exports for an offline, read-only audit.

The output stays local under reports/: it contains licensed abstracts and
unpublished forum questions. This is an export adapter, not a live database sync.
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(stream, required):
    reader = csv.DictReader(stream)
    if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
        raise ValueError("CSV lacks required columns: " + ", ".join(sorted(set(required) - set(reader.fieldnames or []))))
    return list(reader)


def _unique(rows, key, name):
    ids = [row[key] for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} has missing or duplicate {key}")
    return {row[key]: row for row in rows}


def convert(papers_path, forum_zip_path, *, literature_cutoff_year=2025):
    """Return (catalog, questions, summary) without mutating either source."""
    if isinstance(literature_cutoff_year, bool) or not isinstance(literature_cutoff_year, int):
        raise ValueError("literature cutoff year must be an integer")
    with Path(papers_path).open(encoding="utf-8-sig", newline="") as stream:
        paper_rows = _rows(stream, {"catalog_run_id", "extraction_run_id", "paper_id", "title", "doi",
                                    "year", "abstract", "eligible", "n_constructs_v4"})
    paper_by_id = _unique(paper_rows, "paper_id", "paper export")
    catalog_runs = {row["catalog_run_id"] for row in paper_rows}
    extraction_runs = {row["extraction_run_id"] for row in paper_rows}
    if len(catalog_runs) != 1 or len(extraction_runs) != 1:
        raise ValueError("paper export must have exactly one catalog and extraction run")
    papers = []
    for row in paper_rows:
        if row["eligible"] not in {"True", "False"}:
            raise ValueError(f"invalid eligibility for paper {row['paper_id']}")
        papers.append({"id": row["paper_id"], "title": row["title"], "doi": row["doi"] or None,
                       "year": int(row["year"]) if row["year"] else None,
                       "abstract": row["abstract"] or None, "eligible": row["eligible"] == "True",
                       "n_constructs_v4": int(row["n_constructs_v4"] or 0)})
    with zipfile.ZipFile(forum_zip_path) as archive:
        def read_table(name, required):
            matches = [entry for entry in archive.namelist() if entry.endswith("/" + name + ".csv")]
            if len(matches) != 1:
                raise ValueError(f"archive needs one {name}.csv")
            with archive.open(matches[0]) as raw:
                return _rows(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), required)
        episodes = read_table("forum_episode", {"episode_id", "release_id", "condition", "config", "status"})
        question_rows = read_table("question", {"question_id", "episode_id", "content"})
        proposals = read_table("forum_proposal", {"proposal_id", "question_id", "episode_id", "status"})
        evaluations = read_table("forum_proposal_evaluation", {"evaluation_id", "proposal_id", "status",
                                                                 "rubric_version", "mean_score"})
    episode_by_id = _unique(episodes, "episode_id", "episodes")
    question_by_id = _unique(question_rows, "question_id", "questions")
    proposal_by_id = _unique(proposals, "proposal_id", "proposals")
    _unique(evaluations, "evaluation_id", "evaluations")
    questions = []
    for row in question_rows:
        if row["episode_id"] not in episode_by_id:
            raise ValueError(f"question {row['question_id']} has unknown episode")
        questions.append({"id": row["question_id"], "content": json.loads(row["content"]),
                          "episode_id": row["episode_id"]})
    for row in proposals:
        if row["question_id"] not in question_by_id or row["episode_id"] not in episode_by_id:
            raise ValueError(f"proposal {row['proposal_id']} has unknown question or episode")
    for row in evaluations:
        if row["proposal_id"] not in proposal_by_id:
            raise ValueError(f"evaluation {row['evaluation_id']} has unknown proposal")
    releases = {row["release_id"] for row in episodes}
    batches = defaultdict(lambda: {"episodes": 0, "arms": Counter(), "scores": []})
    for row in episodes:
        config = json.loads(row["config"])
        batch = config.get("batch") or {}
        batch_id = batch.get("id") or "unbatched"
        batches[batch_id]["episodes"] += 1
        batches[batch_id]["arms"][batch.get("arm") or row["condition"]] += 1
    for row in evaluations:
        proposal = proposal_by_id[row["proposal_id"]]
        episode = episode_by_id[proposal["episode_id"]]
        batch = (json.loads(episode["config"]).get("batch") or {}).get("id") or "unbatched"
        if row["status"] == "ok" and row["mean_score"]:
            batches[batch]["scores"].append(float(row["mean_score"]))
    source_date = re.search(r"(20\d{2}-\d{2}-\d{2})", Path(papers_path).name)
    catalog = {"catalog_run_id": next(iter(catalog_runs)), "extraction_run_id": next(iter(extraction_runs)),
               "exported_at": "unknown", "source_filename_date": source_date.group(1) if source_date else None,
               "literature_cutoff_year": literature_cutoff_year, "papers": papers}
    summary = {"source_sha256": {"papers_csv": _sha256(papers_path), "forum_zip": _sha256(forum_zip_path)},
               "converted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "catalog_run_id": catalog["catalog_run_id"], "extraction_run_id": catalog["extraction_run_id"],
               "release_ids": sorted(releases), "papers": len(papers),
               "papers_with_abstract": sum(bool(p["abstract"]) and p["abstract"].strip().casefold() != "[no abstract available]" for p in papers),
               "papers_with_v4_constructs": sum(p["n_constructs_v4"] > 0 for p in papers),
               "episodes": len(episodes), "questions": len(questions), "proposals": len(proposals),
               "evaluations": len(evaluations),
               "evaluation_rubrics": dict(Counter(r["rubric_version"] for r in evaluations)),
               "batches": {key: {"episodes": value["episodes"], "arms": dict(value["arms"]),
                                 "blind_mean_score": round(statistics.mean(value["scores"]), 3) if value["scores"] else None}
                           for key, value in sorted(batches.items())},
               "limitations": ["Paper CSV is a 1,466-paper extraction subset, not the full canonical catalog.",
                               "Paper export time is unknown; the date in its filename is not verified as the export time.",
                               "The forum ZIP contains proposal-blind-v2 ratings, not the evaluator's new three-metric evaluation."]}
    return catalog, questions, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--forum-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--literature-cutoff-year", type=int, default=2025)
    args = parser.parse_args(argv)
    catalog, questions, summary = convert(args.papers, args.forum_zip,
                                          literature_cutoff_year=args.literature_cutoff_year)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("catalog.json", catalog), ("questions.json", questions), ("summary.json", summary)):
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"papers": len(catalog["papers"]), "questions": len(questions),
                      "batches": len(summary["batches"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
