"""Offline, read-only novelty retrieval from a frozen catalog export.

This prepares evidence dossiers; it does not decide whether an idea is novel.
Run with ``python -m research0.novelty_offline --help``.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .novelty_baseline import lexical_candidates, question_card


CLAIM_LABELS = {"apparent_prior_answer", "partial_overlap", "related_distinct"}


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _papers(catalog):
    if not isinstance(catalog, dict) or not catalog.get("catalog_run_id") or not catalog.get("exported_at"):
        raise ValueError("catalog_run_id and exported_at are required")
    year = catalog.get("literature_cutoff_year")
    if isinstance(year, bool) or not isinstance(year, int) or year < 1900:
        raise ValueError("literature_cutoff_year must be an integer")
    rows = catalog.get("papers")
    if not isinstance(rows, list):
        raise ValueError("papers must be a list")
    seen = set()
    eligible = []
    counts = {"exported": len(rows), "eligible": 0, "abstract_available": 0,
              "title_only": 0, "excluded": 0, "dated_after_cutoff": 0}
    for paper in rows:
        if not isinstance(paper, dict) or not paper.get("id") or not paper.get("title"):
            raise ValueError("each paper needs id and title")
        paper_id = str(paper["id"])
        if paper_id in seen:
            raise ValueError(f"duplicate canonical paper id: {paper_id}")
        seen.add(paper_id)
        publication_year = paper.get("year")
        if publication_year is not None and (isinstance(publication_year, bool) or
                                             not isinstance(publication_year, int)):
            raise ValueError(f"paper {paper_id} year must be an integer or null")
        if paper.get("eligible") is not True:
            counts["excluded"] += 1
            continue
        if publication_year is not None and publication_year > year:
            counts["dated_after_cutoff"] += 1
            continue
        counts["eligible"] += 1
        abstract = str(paper.get("abstract") or "").strip()
        if abstract and abstract.casefold() != "[no abstract available]":
            counts["abstract_available"] += 1
        else:
            counts["title_only"] += 1
        eligible.append(paper)
    return eligible, counts


def validate_claims(claims, papers, question_ids):
    """Check exact source pointers; quote presence is not semantic correctness."""
    paper_by_id = {str(paper["id"]): paper for paper in papers}
    results = []
    for claim in claims:
        if not isinstance(claim, dict):
            results.append({"valid_pointer": False, "reason": "claim_not_object"})
            continue
        paper_id = str(claim.get("paper_id", ""))
        question_id = str(claim.get("question_id", ""))
        quote = claim.get("quote")
        reason = None
        if question_id not in question_ids:
            reason = "unknown_question"
        elif claim.get("label") not in CLAIM_LABELS:
            reason = "unknown_label"
        elif paper_id not in paper_by_id:
            reason = "unknown_or_ineligible_paper"
        elif not isinstance(quote, str) or not quote.strip():
            reason = "missing_quote"
        elif quote not in str(paper_by_id[paper_id].get("abstract") or ""):
            reason = "quote_not_in_abstract"
        results.append({"question_id": question_id, "paper_id": paper_id,
                        "label": claim.get("label"), "valid_pointer": reason is None,
                        "reason": reason})
    return results


def build_dossier(catalog, questions, *, top_k=20, claims=None):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    papers, coverage = _papers(catalog)
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")
    paper_by_id = {str(paper["id"]): paper for paper in papers}
    seen = set()
    dossiers = []
    for row in questions:
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("each question needs an id")
        question_id = str(row["id"])
        if question_id in seen:
            raise ValueError(f"duplicate question id: {question_id}")
        seen.add(question_id)
        try:
            card = question_card(row.get("content"))
        except ValueError as exc:
            dossiers.append({"question_id": question_id, "status": "abstain",
                             "reason": str(exc), "candidates": []})
            continue
        content = row["content"]
        queries = [content["rq"], content["puzzle"], content["mechanism"]]
        queries.extend(item["statement"] for item in content["predictions"])
        aliases = row.get("author_stated_aliases", [])
        if not isinstance(aliases, list):
            raise ValueError(f"question {question_id} aliases must be a list")
        quote_backed_aliases = []
        for alias in aliases:
            if not isinstance(alias, dict):
                raise ValueError(f"question {question_id} alias needs text, paper_id, and quote")
            phrase, paper_id, quote = alias.get("text"), str(alias.get("paper_id", "")), alias.get("quote")
            abstract = str(paper_by_id.get(paper_id, {}).get("abstract") or "")
            if (not isinstance(phrase, str) or not phrase.strip() or
                    not isinstance(quote, str) or not quote.strip() or
                    quote not in abstract or phrase.casefold() not in quote.casefold()):
                raise ValueError(f"question {question_id} alias lacks an exact abstract quote")
            queries.append(phrase)
            quote_backed_aliases.append(alias)
        hits = lexical_candidates(queries, papers, top_k=top_k) if papers else []
        dossiers.append({"question_id": question_id, "status": "retrieval_only",
                         "question_card": card, "question_card_sha256": hashlib.sha256(card.encode()).hexdigest(),
                         "quote_backed_aliases": quote_backed_aliases,
                         "candidates": hits, "uncertainty": "No prior-answer or novelty verdict has been validated."})
    if claims is not None and not isinstance(claims, list):
        raise ValueError("claims must be a list")
    return {"method": "offline-lexical-dossier-v1", "catalog_run_id": catalog["catalog_run_id"],
            "exported_at": catalog["exported_at"],
            "literature_cutoff_year": catalog["literature_cutoff_year"],
            "coverage": coverage, "questions": dossiers,
            "claim_pointer_checks": validate_claims(claims or [], papers, seen),
            "interpretation": "Candidate retrieval only; no verified novelty score."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True, help="Frozen catalog JSON export")
    parser.add_argument("--questions", type=Path, required=True, help="Structured question JSON list")
    parser.add_argument("--claims", type=Path, help="Optional model claims JSON list for pointer checks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    catalog, catalog_hash = _read(args.catalog)
    questions, questions_hash = _read(args.questions)
    claims, claims_hash = _read(args.claims) if args.claims else (None, None)
    report = build_dossier(catalog, questions, top_k=args.top_k, claims=claims)
    report["input_sha256"] = {"catalog": catalog_hash, "questions": questions_hash,
                              "claims": claims_hash}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['questions'])} dossiers to {args.output}")


if __name__ == "__main__":
    main()
