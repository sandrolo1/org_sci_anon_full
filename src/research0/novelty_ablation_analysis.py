"""Validate and summarize blinded novelty-ablation results after a judge runs.

No model calls. Refuses incomplete or mixed-configuration experiments. All
row-level output is private because it maps unpublished proposals to arms.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .evaluation_integrity import _quote_in, _unique


ARMS = ("semantic_five", "semantic_plus_lexical", "semantic_plus_placebo")
CONTRASTS = (("lexical_vs_placebo", ARMS[1], ARMS[2]),
             ("lexical_vs_semantic", ARMS[1], ARMS[0]),
             ("placebo_vs_semantic", ARMS[2], ARMS[0]))
DIMENSIONS = ("question", "mechanism", "predictions")
ASSESSMENTS = {"overlap", "difference", "unclear", "not_addressed"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield value


def _bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("packet_layout_version") != "fixed-semantic-slots-v2":
        raise ValueError("unknown or obsolete packet layout version")
    packet_path, key_path = directory / "packets.jsonl", directory / "private_key.json"
    if manifest.get("packets_sha256") != _sha(packet_path) or manifest.get("private_key_sha256") != _sha(key_path):
        raise ValueError("packet bundle hash mismatch; regenerate the frozen bundle")
    packets = _unique(list(_jsonl(packet_path)), "packet_id", "packets")
    key_rows = json.loads(key_path.read_text(encoding="utf-8"))
    if not isinstance(key_rows, list):
        raise ValueError("private key must be a list")
    keys = _unique(key_rows, "packet_id", "private key")
    if set(packets) != set(keys) or len(packets) != manifest.get("packets"):
        raise ValueError("packet IDs and private key differ")
    by_proposal = defaultdict(dict)
    for packet_id, key in keys.items():
        if key.get("source_sha256") != manifest.get("source_sha256") or key.get("arm") not in ARMS:
            raise ValueError("private key has a mismatched source or unknown arm")
        packet = packets[packet_id]
        if packet.get("rubric_version") != "novelty-three-dimensions-ablation-v1":
            raise ValueError("unknown packet rubric version")
        if not packet.get("rubric") or not packet.get("proposal_text"):
            raise ValueError("packet lacks rubric or proposal text")
        refs = {paper.get("ref") for paper in packet.get("papers") or []}
        if len(refs) != len(packet.get("papers") or []) or refs != set(key.get("paper_ids") or {}):
            raise ValueError("packet paper references do not match private key")
        if any(set(paper) != {"ref", "title", "year", "abstract"} for paper in packet["papers"]):
            raise ValueError("a blinded packet contains unexpected paper fields")
        if key["arm"] in by_proposal[key["proposal_id"]]:
            raise ValueError("duplicate proposal arm in private key")
        by_proposal[key["proposal_id"]][key["arm"]] = (key, packet)
    if len(by_proposal) != manifest.get("proposals") or any(set(arms) != set(ARMS) for arms in by_proposal.values()):
        raise ValueError("each proposal must have exactly three arms")
    for arms in by_proposal.values():
        identities = {arm: set(arms[arm][0]["paper_ids"].values()) for arm in ARMS}
        if [len(identities[arm]) for arm in ARMS] != [5, 10, 10]:
            raise ValueError("packet arms must contain five, ten and ten distinct papers")
        if (identities[ARMS[0]] != identities[ARMS[1]] & identities[ARMS[2]] or
                not identities[ARMS[0]].issubset(identities[ARMS[1]]) or
                not identities[ARMS[0]].issubset(identities[ARMS[2]])):
            raise ValueError("two ten-paper arms must share only the semantic five")
        ten_refs = [{ref: paper_id for ref, paper_id in arms[arm][0]["paper_ids"].items()
                     if paper_id in identities[ARMS[0]]} for arm in ARMS[1:]]
        if ten_refs[0] != ten_refs[1]:
            raise ValueError("semantic papers moved between ten-paper arms")
        if len({arms[arm][1].get("proposal_text") for arm in ARMS}) != 1 or len({arms[arm][1].get("rubric") for arm in ARMS}) != 1:
            raise ValueError("proposal text and rubric differ across arms")
    return manifest, packets, keys


def check_bundle(directory):
    manifest, _, _ = _bundle(directory)
    return {"source_sha256": manifest["source_sha256"], "proposals": manifest["proposals"],
            "packets": manifest["packets"], "status": "verified_prepared_only_no_model_calls"}


def _validate(row, packet):
    errors = []
    status = row.get("status")
    vote = row.get("vote")
    pairs, compared = 0, set()
    if status == "scored":
        if isinstance(vote, bool) or not isinstance(vote, (int, float)) or not 1 <= vote <= 10:
            errors.append("vote must be numeric from 1 to 10")
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            errors.append("scored answer needs structured comparisons")
            comparisons = []
        papers = {paper["ref"]: paper for paper in packet["papers"]}
        for index, comp in enumerate(comparisons):
            if not isinstance(comp, dict):
                errors.append(f"comparison {index} is not an object")
                continue
            ref = comp.get("neighbour_id")
            if ref not in papers:
                errors.append(f"comparison {index} names an unknown paper")
                continue
            for dimension in DIMENSIONS:
                field = comp.get(dimension)
                if not isinstance(field, dict) or field.get("assessment") not in ASSESSMENTS or not field.get("explanation"):
                    errors.append(f"comparison {index}.{dimension} lacks assessment or explanation")
                    continue
                proposal_quote = field.get("proposal_passage")
                paper_quote = field.get("neighbour_passage")
                if field["assessment"] in ("overlap", "difference") and (not proposal_quote or not paper_quote):
                    errors.append(f"comparison {index}.{dimension} needs both passages")
                if proposal_quote and not _quote_in(proposal_quote, packet["proposal_text"]):
                    errors.append(f"comparison {index}.{dimension} proposal passage mismatch")
                if paper_quote and not _quote_in(paper_quote, papers[ref]["abstract"]):
                    errors.append(f"comparison {index}.{dimension} paper passage mismatch")
                if proposal_quote and paper_quote:
                    pairs += 1
                    compared.add(ref)
        if pairs == 0:
            errors.append("scored answer has no paired supporting passages")
    elif status in ("insufficient_evidence", "failed"):
        if vote is not None or not row.get("reason"):
            errors.append("no-verdict answer needs a reason and null vote")
    else:
        errors.append("unknown status")
    return {"status": "invalid" if errors else status, "vote": vote if status == "scored" and not errors else None,
            "issues": errors, "quote_pairs": pairs if not errors else 0,
            "compared_refs": sorted(compared) if not errors else []}


def _contrast(rows, positive, negative):
    paired = defaultdict(list)
    for (proposal_id, repeat), arms in rows.items():
        left, right = arms.get(positive), arms.get(negative)
        if left and right and left["status"] == right["status"] == "scored":
            paired[proposal_id].append(left["vote"] - right["vote"])
    values = [statistics.mean(deltas) for deltas in paired.values()]
    return {"proposals_paired": len(values), "paired_repeats": sum(map(len, paired.values())),
            "mean_difference": round(statistics.mean(values), 3) if values else None,
            "median_difference": round(statistics.median(values), 3) if values else None,
            "positive": sum(value > 0 for value in values), "negative": sum(value < 0 for value in values),
            "unchanged": sum(value == 0 for value in values),
            "interpretation": "Descriptive paired contrast, one mean per proposal; invalid and no-verdict results are excluded and counted separately."}


def analyze(bundle_dir, results_path, output_dir):
    manifest, packets, keys = _bundle(bundle_dir)
    raw = list(_jsonl(results_path))
    if not raw:
        raise ValueError("results file is empty; no experiment has run")
    model_ids = {row.get("model_id") for row in raw}
    prompt_hashes = {row.get("prompt_sha256") for row in raw}
    config_hashes = {row.get("config_sha256") for row in raw}
    if (len(model_ids) != 1 or not next(iter(model_ids)) or
            len(prompt_hashes) != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(next(iter(prompt_hashes)))) or
            len(config_hashes) != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(next(iter(config_hashes))))):
        raise ValueError("results must use one model ID, prompt hash, and configuration hash")
    if any(row.get("packets_sha256") != manifest["packets_sha256"] for row in raw):
        raise ValueError("results reference a different packet bundle")
    expected_rubric = next(iter(packets.values()))["rubric_version"]
    if any(row.get("rubric_version") != expected_rubric for row in raw):
        raise ValueError("results mixed rubric versions")
    indexed = {}
    for row in raw:
        packet_id, repeat = row.get("packet_id"), row.get("repeat")
        if packet_id not in packets or isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
            raise ValueError("result has an unknown packet or invalid repeat number")
        identity = (packet_id, repeat)
        if identity in indexed:
            raise ValueError("duplicate packet/repeat result")
        indexed[identity] = row
    repeats = {repeat for _, repeat in indexed}
    if repeats != set(range(1, max(repeats) + 1)) or set(indexed) != {(packet_id, repeat) for packet_id in packets for repeat in repeats}:
        raise ValueError("incomplete or unbalanced packet/repeat coverage")

    semantic = {key["proposal_id"]: set(key["paper_ids"].values())
                for key in keys.values() if key["arm"] == "semantic_five"}
    statuses = {arm: Counter() for arm in ARMS}
    pair_rows = defaultdict(dict)
    private_rows = []
    quote_pairs = 0
    added_use = Counter()
    for (packet_id, repeat), row in indexed.items():
        packet, key = packets[packet_id], keys[packet_id]
        result = _validate(row, packet)
        arm, proposal_id = key["arm"], key["proposal_id"]
        statuses[arm][result["status"]] += 1
        quote_pairs += result["quote_pairs"]
        refs = key["paper_ids"]
        added_compared = sum(refs[ref] not in semantic[proposal_id] for ref in result["compared_refs"])
        if arm != "semantic_five" and result["status"] == "scored":
            added_use[arm] += bool(added_compared)
        pair_rows[(proposal_id, repeat)][arm] = result
        private_rows.append({"proposal_id": proposal_id, "packet_id": packet_id, "arm": arm,
                             "repeat": repeat, "status": result["status"], "vote": result["vote"],
                             "quote_pairs": result["quote_pairs"], "added_papers_compared": added_compared,
                             "validation_issues": "; ".join(result["issues"])})
    contrasts = {name: _contrast(pair_rows, positive, negative) for name, positive, negative in CONTRASTS}
    report = {"source_sha256": manifest["source_sha256"], "packets_sha256": manifest["packets_sha256"],
              "results_sha256": _sha(results_path), "model_id": next(iter(model_ids)),
              "prompt_sha256": next(iter(prompt_hashes)), "config_sha256": next(iter(config_hashes)),
              "rubric_version": expected_rubric,
              "proposals": manifest["proposals"], "repeats": len(repeats),
              "arm_status": {arm: dict(statuses[arm]) for arm in ARMS},
              "valid_quote_pairs": quote_pairs,
              "scored_ten_paper_runs_quoting_added_papers": dict(added_use),
              "contrasts": contrasts,
              "interpretation": "Exploratory sensitivity analysis. No contrast establishes novelty accuracy or generalizes beyond these proposals."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "private_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=private_rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(private_rows, key=lambda row: (row["proposal_id"], row["repeat"], row["arm"])))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Directory made by novelty_ablation.prepare")
    parser.add_argument("--check-bundle", action="store_true", help="Verify packets and hashes without judgments")
    parser.add_argument("--results", type=Path, help="Private JSONL model outputs")
    parser.add_argument("--output", type=Path, help="Private report directory")
    args = parser.parse_args(argv)
    if args.check_bundle:
        if args.results or args.output:
            parser.error("--check-bundle takes no results or output")
        print(json.dumps(check_bundle(args.bundle)))
        return
    if not args.results or not args.output:
        parser.error("analysis requires --results and --output")
    report = analyze(args.bundle, args.results, args.output)
    print(json.dumps({"proposals": report["proposals"], "repeats": report["repeats"],
                      "arm_status": report["arm_status"]}))


if __name__ == "__main__":
    main()
