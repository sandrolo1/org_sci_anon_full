"""Make a frozen, balanced execution schedule for matched 0/A episodes.

This schedules runs only. It neither executes models nor verifies source equivalence.
The JSON input is a list of opaque records with pair_id, problem_id, stratum,
packet_sha256, trigger, and seed. Do not put source text in this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


REQUIRED = ("pair_id", "problem_id", "stratum", "packet_sha256", "trigger", "seed")
OPAQUE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
TRIGGER_KEY = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
STRATA = frozenset(("cto", "tim", "cross"))


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def plan(blocks: list[dict], assignment_seed: str) -> dict:
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("blocks must be a nonempty JSON list")
    if (not isinstance(assignment_seed, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", assignment_seed)):
        raise ValueError("assignment seed must be a short recorded token")
    seen: set[str] = set()
    strata: dict[str, list[dict]] = defaultdict(list)
    for block in blocks:
        if not isinstance(block, dict) or set(block) != set(REQUIRED):
            raise ValueError(f"each block must have exactly these keys: {', '.join(REQUIRED)}")
        if any(not isinstance(block[k], str) or not OPAQUE_ID.fullmatch(block[k])
               for k in ("pair_id", "problem_id")):
            raise ValueError("pair_id and problem_id must be opaque short identifiers")
        if not isinstance(block["stratum"], str) or block["stratum"] not in STRATA:
            raise ValueError("stratum must be cto, tim, or cross")
        if not isinstance(block["trigger"], str) or not TRIGGER_KEY.fullmatch(block["trigger"]):
            raise ValueError("trigger must be a short machine-readable key")
        if (not isinstance(block["seed"], int) or isinstance(block["seed"], bool)
                or not 0 <= block["seed"] <= 2_147_483_647):
            raise ValueError("block seed must fit the forum's seed range")
        if (not isinstance(block["packet_sha256"], str) or len(block["packet_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in block["packet_sha256"])):
            raise ValueError("packet_sha256 must be a lowercase SHA-256 hex digest")
        if block["pair_id"] in seen:
            raise ValueError("pair_id must be unique")
        seen.add(block["pair_id"])
        strata[block["stratum"]].append(block)

    orientation: dict[str, tuple[str, str]] = {}
    for stratum, group in strata.items():
        ranked = sorted(group, key=lambda b: _key(assignment_seed, "orientation", stratum, b["pair_id"]))
        first = "A" if int(_key(assignment_seed, "start", stratum), 16) % 2 else "0"
        second = "0" if first == "A" else "A"
        for index, block in enumerate(ranked):
            orientation[block["pair_id"]] = (first, second) if index % 2 == 0 else (second, first)

    execution = sorted(blocks, key=lambda b: _key(assignment_seed, "execution", b["pair_id"]))
    episodes = []
    for block in execution:
        for order, condition in enumerate(orientation[block["pair_id"]], 1):
            episodes.append({**block, "condition": condition, "within_pair_order": order,
                             "execution_slot": len(episodes) + 1,
                             "run_id": f"{block['pair_id']}:{condition}"})
    return {"schema_version": 1, "assignment_seed": assignment_seed,
            "blocks_sha256": digest(blocks), "episodes_sha256": digest(episodes),
            "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True, help="frozen JSON block registry")
    parser.add_argument("--assignment-seed", required=True, help="outcome-independent recorded random seed")
    parser.add_argument("--out", type=Path, required=True, help="new schedule JSON; existing file is refused")
    args = parser.parse_args()
    result = plan(json.loads(args.blocks.read_text(encoding="utf-8")), args.assignment_seed)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"{len(result['episodes']) // 2} matched pairs; schedule SHA-256 {result['episodes_sha256']}")


if __name__ == "__main__":
    main()
