"""Audit a private forum evaluation export without calling a model or database.

The detailed output contains unpublished proposal IDs and stays under ignored
``reports/``. The aggregate output contains counts only and may be reviewed in
Git. This audits stored evidence and score arithmetic, not scientific quality.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CRITERIA = ("novelty", "methodology", "impact")


def _member(archive, suffix):
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected exactly one {suffix} member")
    return archive.read(names[0])


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            raise ValueError(f"{label} has a missing or duplicate {key}")
        result[value] = row
    return result


def _squash(value):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def _quote_in(quote, source):
    needle = _squash(quote)
    return len(needle) >= 20 and needle in _squash(source)


def _card_text(card):
    parts = []
    for value in card.values():
        if not isinstance(value, dict):
            continue
        parts.append(str(value.get("text") or ""))
        parts.extend(str(p) for p in value.get("passages") or [])
    return parts


def _cards(path):
    rows = (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return _unique(rows, "canonical_id", "paper cards")


def _paper(catalog, neighbor):
    for key in (neighbor.get("id"), neighbor.get("eid"), neighbor.get("canonical_id")):
        if key and key in catalog:
            return catalog[key]
    return None


def _reference_dois(references):
    result = set()
    for reference in references or []:
        for doi in re.findall(r"\bdoi:\s*(10\.\S+)", str(reference), flags=re.IGNORECASE):
            result.add(doi.rstrip(".,;)]}").casefold())
    return result


def _comparisons(judges):
    for judge in (judges or {}).values():
        if isinstance(judge, dict):
            yield from judge.get("comparisons") or []


def _check_judgement(*, criterion, document, proposal_text, sources, counters, flags, hybrid=False):
    label = "hybrid" if hybrid else criterion
    if document.get("status") != "scored":
        return
    comparisons = list(_comparisons(document.get("judges")))
    counters[f"{label}_comparison_count"] += len(comparisons)
    for comp in comparisons:
        ref = comp.get("neighbour_id")
        if ref not in sources:
            flags.append(f"{label}:unknown_comparison_paper")
            counters[f"{label}_unknown_comparison_paper"] += 1
            continue
        fields = [comp] if not hybrid else [comp.get(k) or {} for k in ("question", "mechanism", "predictions")]
        for field in fields:
            if not field.get("proposal_passage") and not field.get("neighbour_passage"):
                continue
            counters[f"{label}_quote_pairs"] += 1
            if not _quote_in(field.get("proposal_passage"), proposal_text):
                flags.append(f"{label}:proposal_quote_mismatch")
                counters[f"{label}_proposal_quote_mismatch"] += 1
            source = sources[ref]
            if source is None:
                flags.append(f"{label}:missing_paper_source")
                counters[f"{label}_missing_paper_source"] += 1
            elif not any(_quote_in(field.get("neighbour_passage"), part) for part in source):
                flags.append(f"{label}:paper_quote_mismatch")
                counters[f"{label}_paper_quote_mismatch"] += 1


def audit_export(export, method_cards, impact_cards):
    """Return aggregate counts and private case flags for one frozen export."""
    export = Path(export)
    method = _cards(method_cards)
    impact = _cards(impact_cards)
    with zipfile.ZipFile(export) as archive:
        proposals = json.loads(_member(archive, "proposals.json"))
        evaluations = json.loads(_member(archive, "evaluations.json"))
        catalog_rows = list(csv.DictReader(io.StringIO(_member(archive, "catalog_novelty_corpus.csv").decode("utf-8-sig"))))
    by_proposal = _unique(proposals, "proposal_id", "proposals")
    by_evaluation = _unique(evaluations, "proposal_id", "evaluations")
    if set(by_proposal) != set(by_evaluation):
        raise ValueError("proposal and evaluation ID sets differ")
    catalog = {}
    by_doi = defaultdict(set)
    for row in catalog_rows:
        if row.get("doi"):
            by_doi[row["doi"].strip().casefold()].add(row.get("eid"))
        for key in (row.get("eid"), row.get("canonical_id")):
            if key:
                if key in catalog and catalog[key] is not row:
                    raise ValueError(f"duplicate catalog identifier: {key}")
                catalog[key] = row

    counters = Counter()
    score_deltas = []
    case_flags = []
    for pid, proposal in by_proposal.items():
        document = by_evaluation[pid].get("evaluation") or {}
        flags = []
        counters["proposals"] += 1
        if document.get("proposal_id") != pid:
            flags.append("evaluation_id_mismatch")
        criteria = document.get("criteria") or {}
        summary = proposal.get("evaluation") or {}
        votes = summary.get("votes") or {}
        if document.get("status") != "ok":
            flags.append("primary_evaluation_not_ok")
        if all(isinstance(votes.get(k), (int, float)) for k in CRITERIA):
            expected = round(sum(votes[k] for k in CRITERIA) / 3, 2)
            if abs(float(summary.get("mean_of_three") or 0) - expected) > 0.011:
                flags.append("mean_mismatch")
            for kind in CRITERIA:
                if abs(float(criteria.get(kind, {}).get("vote") or 0) - votes[kind]) > 0.011:
                    flags.append(f"{kind}:vote_mismatch")
        else:
            flags.append("missing_primary_vote")

        for kind in CRITERIA:
            criterion = criteria.get(kind) or {}
            if criterion.get("status") != "scored":
                flags.append(f"{kind}:not_scored")
            neighbors = criterion.get("neighbours") or []
            counters[f"{kind}_neighbors"] += len(neighbors)
            sources = {}
            for neighbor in neighbors:
                label = neighbor.get("label")
                if not label or label in sources:
                    flags.append(f"{kind}:duplicate_neighbor_label")
                    continue
                if kind == "novelty":
                    paper = _paper(catalog, neighbor)
                    source = [paper.get("abstract") or ""] if paper else None
                else:
                    cards = method if kind == "methodology" else impact
                    found = cards.get(neighbor.get("id"))
                    source = _card_text(found["card"]) if found else None
                sources[label] = source
                if source is None:
                    flags.append(f"{kind}:neighbor_not_mapped")
                    counters[f"{kind}_neighbor_not_mapped"] += 1
            _check_judgement(criterion=kind, document=criterion,
                             proposal_text=criterion.get("proposal_text") or "", sources=sources,
                             counters=counters, flags=flags)
            judge_votes = [j.get("vote") for j in (criterion.get("judges") or {}).values() if isinstance(j, dict)]
            if len(judge_votes) == 2 and all(isinstance(v, (int, float)) for v in judge_votes):
                counters[f"{kind}_double_judged"] += 1
                counters[f"{kind}_double_abs_gap_sum"] += abs(judge_votes[0] - judge_votes[1])

        novelty = criteria.get("novelty") or {}
        hybrid = novelty.get("hybrid") or {}
        referenced = _reference_dois(proposal.get("references"))
        reference_ids = {next(iter(by_doi[doi])) for doi in referenced if len(by_doi.get(doi, ())) == 1}
        semantic_ids = {n.get("id") for n in novelty.get("neighbours") or []}
        counters["distinct_reference_dois"] += len(referenced)
        counters["reference_dois_in_corpus"] += len(reference_ids)
        counters["reference_dois_in_semantic_five"] += len(reference_ids & semantic_ids)
        counters["proposals_with_corpus_references"] += bool(reference_ids)
        if hybrid.get("status") == "scored":
            counters["hybrid_scored"] += 1
            if not isinstance(hybrid.get("vote"), (int, float)) or not isinstance(novelty.get("vote"), (int, float)):
                flags.append("hybrid:missing_vote")
            else:
                delta = hybrid["vote"] - novelty["vote"]
                score_deltas.append(delta)
                if delta > 0:
                    counters["hybrid_vote_up"] += 1
                elif delta < 0:
                    counters["hybrid_vote_down"] += 1
                else:
                    counters["hybrid_vote_same"] += 1
                if abs(delta) >= 2:
                    flags.append("hybrid:shift_at_least_two")
            candidates = hybrid.get("candidates") or []
            hybrid_ids = {n.get("id") for n in candidates}
            counters["paired_reference_dois_in_corpus"] += len(reference_ids)
            counters["paired_reference_dois_in_semantic_five"] += len(reference_ids & semantic_ids)
            counters["paired_reference_dois_in_hybrid_ten"] += len(reference_ids & hybrid_ids)
            if len(candidates) != len(hybrid_ids):
                flags.append("hybrid:duplicate_candidate")
            if not semantic_ids.issubset(hybrid_ids):
                flags.append("hybrid:semantic_neighbor_missing")
            counters["hybrid_lexical_only_candidates"] += sum(
                "lexical" in (c.get("found_by") or []) and c.get("id") not in semantic_ids for c in candidates)
            sources = {}
            for candidate in candidates:
                label = candidate.get("label")
                paper = _paper(catalog, candidate)
                if label in sources:
                    flags.append("hybrid:duplicate_candidate_label")
                sources[label] = [paper.get("abstract") or ""] if paper else None
                if paper is None:
                    flags.append("hybrid:candidate_not_mapped")
                    counters["hybrid_candidate_not_mapped"] += 1
            _check_judgement(criterion="novelty", document=hybrid,
                             proposal_text=novelty.get("proposal_text") or "", sources=sources,
                             counters=counters, flags=flags, hybrid=True)
            lexical_labels = {c.get("label") for c in candidates if c.get("id") not in semantic_ids}
            cited = {c.get("neighbour_id") for c in _comparisons(hybrid.get("judges"))}
            counters["hybrid_lexical_comparisons"] += sum(c.get("neighbour_id") in lexical_labels for c in _comparisons(hybrid.get("judges")))
            if lexical_labels & cited:
                counters["hybrid_cases_with_added_paper_compared"] += 1
                if isinstance(hybrid.get("vote"), (int, float)) and isinstance(novelty.get("vote"), (int, float)):
                    counters["hybrid_changed_with_added_paper_compared" if hybrid["vote"] != novelty["vote"] else
                             "hybrid_unchanged_with_added_paper_compared"] += 1
            else:
                flags.append("hybrid:no_lexical_candidate_cited")
        elif hybrid.get("status") == "failed":
            counters["hybrid_failed"] += 1
            flags.append("hybrid:failed")
        else:
            counters["hybrid_absent_or_other"] += 1

        method_card = ((proposal.get("cards") or {}).get("method") or {}).get("card") or {}
        for field in ("design", "data_sample", "procedure", "measures"):
            if (method_card.get(field) or {}).get("status") == "inferred":
                counters[f"method_plan_{field}_inferred"] += 1
        for flag in set(flags):
            counters[f"flag_{flag}"] += 1
        if flags:
            case_flags.append({"proposal_id": pid, "flags": sorted(set(flags))})

    aggregate = {
        "source_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
        "corpus_papers": len(catalog_rows),
        "counts": dict(sorted(counters.items())),
        "hybrid_mean_delta": round(sum(score_deltas) / len(score_deltas), 3) if score_deltas else None,
        "hybrid_max_abs_delta": max((abs(x) for x in score_deltas), default=None),
        "interpretation": "Integrity and sensitivity audit only; passage matches, score agreement and journal tiers are not scientific-validity labels.",
    }
    return {"aggregate": aggregate, "case_flags": case_flags}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--method-cards", type=Path, required=True)
    parser.add_argument("--impact-cards", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_export(args.export, args.method_cards, args.impact_cards)
    for output, value in ((args.aggregate, result["aggregate"]), (args.private_detail, result["case_flags"])):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"proposals": result["aggregate"]["counts"]["proposals"],
                      "case_flags": len(result["case_flags"])}))


if __name__ == "__main__":
    main()
