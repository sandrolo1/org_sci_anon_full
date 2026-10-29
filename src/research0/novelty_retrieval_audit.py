"""Read-only comparison of lexical whole-catalog retrieval with forum neighbors.

The forum's semantic neighbors and this audit may cover different paper sets.
An overlap is a retrieval diagnostic, never a novelty score or a validity test.
No model, database, or network call occurs here.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .novelty_baseline import lexical_candidates
from .scoring_io import clean_proposal_text


def doi_key(value):
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or None


def audit(catalog, proposals, *, top_k=20):
    """Compare candidates for proposals with abstracts and stored new scores."""
    if not isinstance(catalog, dict) or not catalog.get("catalog_run_id"):
        raise ValueError("catalog_run_id is required")
    cutoff = catalog.get("literature_cutoff_year")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        raise ValueError("literature_cutoff_year must be an integer")
    if not isinstance(proposals, list) or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 5:
        raise ValueError("proposals must be a list and top_k at least five")
    if not isinstance(catalog.get("papers"), list):
        raise ValueError("papers must be a list")
    seen, papers, coverage = set(), [], Counter()
    doi_ids, eid_ids = defaultdict(set), defaultdict(set)
    for paper in catalog["papers"]:
        pid = str(paper.get("id") or "")
        if not pid or pid in seen:
            raise ValueError("paper IDs must be unique and nonempty")
        seen.add(pid)
        coverage["exported"] += 1
        if paper.get("eligible") is not True:
            coverage["excluded"] += 1
            continue
        year = paper.get("year")
        if not isinstance(year, int) or isinstance(year, bool):
            coverage["unknown_year"] += 1
            continue
        if year > cutoff:
            coverage["after_cutoff"] += 1
            continue
        papers.append(paper)
        coverage["eligible_within_cutoff"] += 1
        if not paper.get("abstract") or str(paper["abstract"]).strip().casefold() == "[no abstract available]":
            coverage["title_only"] += 1
        key = doi_key(paper.get("doi"))
        if key:
            doi_ids[key].add(pid)
        eid = str(paper.get("eid") or "").strip()
        if eid:
            eid_ids[eid].add(pid)
    eligible_ids = {paper["id"] for paper in papers}
    results = []
    seen_proposals = set()
    for row in proposals:
        pid = str(row.get("id") or "")
        if not pid or pid in seen_proposals:
            raise ValueError("proposal IDs must be unique and nonempty")
        seen_proposals.add(pid)
        title = clean_proposal_text(row.get("title"))
        abstract = clean_proposal_text(row.get("abstract"))
        item = {"proposal_id": pid, "status": None, "lexical_candidates": [],
                "semantic_mapped": [], "semantic_unmapped": [], "overlap_at_5": None,
                "semantic_corpus_id": None, "score_schema": None}
        if not abstract:
            item["status"] = "missing_proposal_abstract"
            results.append(item)
            continue
        hits = lexical_candidates([title + "\n" + abstract], papers, top_k=top_k) if papers else []
        item["lexical_candidates"] = [{key: hit.get(key) for key in ("id", "title", "doi", "year", "bm25_score", "evidence_available")}
                                      for hit in hits]
        score = row.get("score")
        if isinstance(score, dict) and score.get("synthetic") is True:
            item["status"] = "synthetic_score_excluded"
            results.append(item)
            continue
        item["score_schema"] = score.get("schema") if isinstance(score, dict) else None
        novelty = (score.get("criteria") or {}).get("novelty") if isinstance(score, dict) else None
        if not isinstance(novelty, dict) or not novelty.get("neighbours"):
            item["status"] = "lexical_only_no_new_score"
            results.append(item)
            continue
        item["semantic_corpus_id"] = ((score.get("corpora") or {}).get("novelty") or {}).get("corpus_id")
        for rank, neighbor in enumerate(novelty["neighbours"], 1):
            canonical = str(neighbor.get("canonical_id") or "")
            mapped_by = "canonical_id" if canonical in eligible_ids else None
            if canonical not in eligible_ids:
                eid = str(neighbor.get("eid") or neighbor.get("id") or "").strip()
                matches = eid_ids.get(eid, set()) if eid else set()
                canonical = next(iter(matches)) if len(matches) == 1 else ""
                mapped_by = "eid" if canonical else None
            key = doi_key(neighbor.get("doi"))
            if canonical not in eligible_ids:
                matches = doi_ids.get(key, set()) if key else set()
                canonical = next(iter(matches)) if len(matches) == 1 else ""
                mapped_by = "doi" if canonical else None
            if canonical in eligible_ids:
                item["semantic_mapped"].append({"rank": rank, "id": canonical, "neighbor_id": neighbor.get("id"), "mapped_by": mapped_by})
            else:
                item["semantic_unmapped"].append({"rank": rank, "neighbor_id": neighbor.get("id"), "doi": key})
        semantic_top = {entry["id"] for entry in item["semantic_mapped"] if entry["rank"] <= 5}
        lexical_top = {hit["id"] for hit in hits[:5]}
        item["overlap_at_5"] = len(semantic_top & lexical_top) if semantic_top else None
        item["status"] = "comparison_ready" if semantic_top else "semantic_ids_unmapped"
        results.append(item)
    statuses = dict(Counter(item["status"] for item in results))
    return {"method": "whole-catalog-lexical-vs-forum-semantic-v1", "catalog_run_id": catalog["catalog_run_id"],
            "literature_cutoff_year": cutoff, "coverage": dict(coverage), "proposals": results,
            "summary": {"proposals": len(results), "statuses": statuses},
            "interpretation": "Retrieval comparison only. A low overlap does not establish which method found better prior art; DOI gaps and corpus scope affect it."}


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--scored-proposals", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    catalog, catalog_hash = _read(args.catalog)
    proposals, hashes = [], {}
    for path in args.scored_proposals:
        part, hashes[str(path)] = _read(path)
        if not isinstance(part, list):
            parser.error(f"{path} must contain a JSON list")
        proposals.extend(part)
    result = audit(catalog, proposals, top_k=args.top_k)
    result["input_sha256"] = {"catalog": catalog_hash, "scored_proposals": hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
