"""Check the private 2026 benchmark archive without a database or model calls.

The report contains aggregate counts and paper IDs for failures, never publisher text.
Keep the source ZIP outside Git. This checks source integrity and literal quotations;
it cannot establish whether a quoted passage supports an evaluator's inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile


LABELS = {
    "question": "Question", "objective": "Objective of the analysis", "units_levels": "Units and levels",
    "data_sample": "Data and sample", "design": "Design", "measures": "Measures",
    "procedure": "Analytical procedure", "assumptions_controls": "Assumptions and controls",
    "conversation": "Scientific conversation", "prior_explanation": "Prior explanation challenged",
    "claimed_change": "Claimed change", "consequences": "Stated consequences", "scope_conditions": "Scope conditions",
}
FIELDS = {
    "method": ("question", "objective", "units_levels", "data_sample", "design", "measures", "procedure", "assumptions_controls"),
    "impact": ("conversation", "prior_explanation", "claimed_change", "consequences", "scope_conditions"),
}
STATUS = {"extracted": "declared", "inferred": "inferred", "missing": "not stated"}
MIN_PASSAGE = 20


def _squash(value):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def _card_text(card, kind, *, gaps=False):
    lines = []
    for field in FIELDS[kind]:
        value = card[field]
        if value["status"] == "missing":
            continue
        quotes = " ".join(f"«{passage.strip()}»" for passage in value["passages"])
        lines.append(f"{LABELS[field]} ({STATUS[value['status']]}): {value['text']} {quotes}".rstrip())
    missing = [LABELS[field] for field in FIELDS[kind] if card[field]["status"] == "missing"]
    if missing:
        lines.append("Not stated: " + "; ".join(missing))
    if gaps and card.get("info_gaps"):
        lines.append("Left open by the author: " + "; ".join(card["info_gaps"][:4]))
    return "\n".join(lines)


def _records(archive, name):
    return [json.loads(line) for line in archive.read(name).splitlines() if line]


def _member(archive, prefix, suffix):
    matches = [name for name in archive.namelist() if name.startswith(prefix) and name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {prefix}*{suffix}; found {len(matches)}")
    return matches[0]


def _source_text(abstract, card, criterion):
    if criterion == "novelty":
        return f"TITLE: {abstract['title']}\n\nABSTRACT:\n{abstract['abstract']}"
    kind = "method" if criterion == "methodology" else "impact"
    return f"ABSTRACT OF THE PROPOSAL:\n{abstract['abstract']}\n\nTHE PROPOSAL'S {kind.upper()} CARD:\n{_card_text(card, kind, gaps=True)}"


def audit_archive(path):
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("inputs/evaluation_2026/manifest.json"))
        hash_failures = [name for name, expected in manifest["sha256"].items()
                         if name not in archive.namelist() or hashlib.sha256(archive.read(name)).hexdigest() != expected]
        abstracts = {r["paper_id"]: r for r in _records(archive, "inputs/evaluation_2026/evaluator_input/abstracts.jsonl")}
        cards = {(r["paper_id"], r["kind"]): r for r in _records(archive, _member(archive, "evaluation_cards_run_", ".jsonl"))}
        parsed = {r["paper_id"]: r for name in archive.namelist() if name.startswith("inputs/evaluation_2026/parsed/") and name.endswith(".json")
                  for r in [json.loads(archive.read(name))]}
        corpora = {
            "novelty": {r["id"]: r["text"] for r in _records(archive, _member(archive, "reports/anvur_live/corpora/cor_", "/texts.jsonl"))},
            "methodology": {r["id"]: r["card"] for r in _records(archive, _member(archive, "reports/anvur_live/corpora/fsm_", "/cards.jsonl"))},
            "impact": {r["id"]: r["card"] for r in _records(archive, _member(archive, "reports/anvur_live/corpora/fsi_", "/cards.jsonl"))},
        }
        score_rows = _records(archive, _member(archive, "results/evaluation_scores_run_", ".jsonl"))
        index = list(csv.DictReader(StringIO(archive.read("inputs/evaluation_2026/fulltext/fulltext_index.csv").decode("utf-8-sig"))))
        counts, failures = Counter(), set()

        for (paper_id, kind), record in cards.items():
            source = parsed.get(paper_id)
            if source is None:
                failures.add(paper_id)
                continue
            if hashlib.sha256(source["full_text"].encode("utf-8")).hexdigest() != source["text_sha256"]:
                failures.add(paper_id)
            text = _squash(source["full_text"])
            for field in FIELDS[kind]:
                for passage in record["card"][field]["passages"]:
                    counts["card_passages"] += 1
                    if _squash(passage) in text:
                        counts["card_passages_matched"] += 1
                    else:
                        failures.add(paper_id)

        def check_quote(quote, hay, paper_id, *, optional=False):
            counts["judge_quote_fields"] += 1
            if not quote and optional:
                counts["permitted_blank_quotes"] += 1
            elif len(_squash(quote)) >= MIN_PASSAGE and _squash(quote) in _squash(hay):
                counts["judge_quotes_matched"] += 1
            else:
                counts["judge_quotes_unmatched"] += 1
                failures.add(paper_id)

        for row in score_rows:
            paper_id = row["paper_id"]
            if paper_id not in abstracts:
                failures.add(paper_id)
                continue
            document = row["document"]["criteria"]
            for criterion in ("novelty", "methodology", "impact", "hybrid"):
                parent = document["novelty"] if criterion == "hybrid" else document[criterion]
                result = parent["hybrid"] if criterion == "hybrid" else parent
                if not result.get("judges"):
                    continue
                source_kind = "novelty" if criterion == "hybrid" else criterion
                card_kind = "method" if criterion == "methodology" else "impact"
                own_card = cards.get((paper_id, card_kind)) if criterion not in ("novelty", "hybrid") else None
                if criterion not in ("novelty", "hybrid") and own_card is None:
                    failures.add(paper_id)
                    continue
                own = _source_text(abstracts[paper_id], own_card["card"] if own_card else None, source_kind)
                neighbours = {}
                for item in result.get("candidates" if criterion == "hybrid" else "neighbours", []):
                    candidate = corpora[source_kind].get(item["id"])
                    if candidate is None:
                        failures.add(paper_id)
                    else:
                        neighbours[item["label"]] = candidate if source_kind == "novelty" else _card_text(candidate, card_kind)
                for answer in result["judges"].values():
                    if not answer:
                        continue
                    counts["judge_answers"] += 1
                    for comparison in answer["comparisons"]:
                        other = neighbours.get(comparison["neighbour_id"], "")
                        dimensions = [comparison[dimension] for dimension in ("question", "mechanism", "predictions")] if criterion == "hybrid" else [comparison]
                        for dimension in dimensions:
                            optional = dimension.get("assessment") == "not_stated"
                            check_quote(dimension["proposal_passage"], own, paper_id, optional=optional)
                            check_quote(dimension["neighbour_passage"], other, paper_id, optional=optional)
                    if criterion == "methodology":
                        test = answer["mechanism_test"]
                        for field in ("mechanism_passage", "design_passage"):
                            check_quote(test[field], own, paper_id, optional=test["verdict"] == "mechanism_not_stated")

        return {
            "run_id": manifest["run_id"],
            "papers": len(score_rows),
            "manifest_files_checked": len(manifest["sha256"]),
            "manifest_hash_failures": hash_failures,
            "fulltext_origins": dict(Counter(r["origin"] for r in index)),
            "fulltext_versions": dict(Counter(r["version"] for r in index)),
            "cards": len(cards),
            **dict(counts),
            "paper_ids_with_source_or_quote_failures": sorted(failures),
            "scope": "Literal source and quote verification only; semantic correctness and independent replication remain untested.",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_archive(args.archive)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if result["manifest_hash_failures"] or result["paper_ids_with_source_or_quote_failures"]:
        raise SystemExit("Archive source or quote audit found failures")


if __name__ == "__main__":
    main()
