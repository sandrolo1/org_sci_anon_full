"""Catalog-bound Batch assignment and blinded inter-rater validation."""
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path

from .community_phase1 import MODEL, PROMPT_VERSION, collect_remote, load_scope, make_request, submit
from .community_recovery import assignment_rows
from .core import file_digest

SCOPE = Path(__file__).resolve().parents[2] / "config" / "community_scope_v1.json"
RUN_ROOT = Path("reports/community_dashboard")
SAMPLE_SIZE = 100
EXPECTED_KAPPA = 0.8
LOG = logging.getLogger("research0")


def run_dir(run_id):
    return RUN_ROOT / str(run_id)


def current_attempt(directory):
    retries = sorted(directory.glob("retry_[0-9][0-9][0-9]")) if directory.exists() else []
    return (retries[-1], len(retries)) if retries else (directory, 0)


def prepare_catalog(con, run_id, directory, scope_path=SCOPE):
    """Use the actual canonical IDs, excluding unresolved identity and perimeter flags."""
    if (directory / "manifest.json").exists():
        raise ValueError("Assignment run already prepared")
    scope = load_scope(scope_path)
    records = []
    for cid, title, year, raw in con.execute(
        "SELECT c.canonical_id,c.title,c.publication_year,r.raw_fields FROM canonical_paper c "
        "JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE c.run_id=%s AND c.status='retained' AND cardinality(c.perimeter_flags)=0 ORDER BY c.canonical_id", (run_id,)
    ):
        records.append({"record_id": str(cid), "title": title or "", "year": str(year or ""),
                        "abstract": (raw.get("Abstract") or "").strip(),
                        "author_keywords": (raw.get("Author Keywords") or "").strip(),
                        "index_keywords": (raw.get("Index Keywords") or "").strip()})
    if not records:
        raise ValueError("No eligible canonical papers")
    directory.mkdir(parents=True, exist_ok=True)
    requests = {}
    with (directory / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for offset in range(0, len(records), 10):
            group = records[offset:offset + 10]
            key = f"community-{offset // 10:06d}"
            stream.write(json.dumps(make_request(group, scope, key), ensure_ascii=False) + "\n")
            requests[key] = [r["record_id"] for r in group]
    manifest = {"catalog_run_id": str(run_id), "model": MODEL, "prompt_version": PROMPT_VERSION,
                "scope_version": scope["version"], "scope_sha256": file_digest(scope_path),
                "all_unique_records": con.execute("SELECT count(*) FROM canonical_paper WHERE run_id=%s", (run_id,)).fetchone()[0],
                "selected_records": len(records), "request_count": len(requests), "requests": requests}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def assignment_status(run_id):
    directory = run_dir(run_id)
    if not (directory / "manifest.json").exists():
        return {"status": "not_started", "assigned": 0}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    attempt, attempt_number = current_attempt(directory)
    active_manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((attempt / "batch.json").read_text(encoding="utf-8")) if (attempt / "batch.json").exists() else {}
    counts = Counter((row["cto"], row["tim"]) for row in assignment_rows(directory).values())
    assigned = sum(counts.values())
    return {"status": state.get("status", "prepared"), "attempt": attempt_number,
            "total": manifest["all_unique_records"],
            "eligible": manifest["selected_records"], "excluded": manifest["all_unique_records"] - manifest["selected_records"],
            "request_total": active_manifest["request_count"], "request_counts": state.get("request_counts"),
            "checked_at": state.get("checked_at"),
            "assigned": assigned, "unassigned": manifest["selected_records"] - assigned,
            "cto_only": counts[("1", "0")], "tim_only": counts[("0", "1")],
            "both": counts[("1", "1")], "neither": counts[("0", "0")]}


GROUPS = (("both", "CTO and TIM", ("1", "1")), ("cto_only", "CTO only", ("1", "0")),
          ("tim_only", "TIM only", ("0", "1")), ("neither", "Neither", ("0", "0")))
EXAMPLES_PER_GROUP = 5


def _share(count, total):
    return round(count / total, 4) if total else None


def stats(con, run_id, seed=0, per_group=EXAMPLES_PER_GROUP):
    """Read-only figures and examples of the Batch labels (suggestions, not validated membership): groups, per-community totals, per year, per journal.

    Examples are a fixed pseudo-random pick for a given seed (the same seed always shows the same papers); nothing is written.
    The human ratings of the blind sample are not read here: they stay hidden until the annotators have finished.
    """
    labels = assignment_rows(run_dir(run_id))
    status = assignment_status(run_id)
    base = {"status": status["status"], "model": MODEL, "prompt_version": PROMPT_VERSION, "validated": False, "seed": seed,
            "eligible": status.get("eligible"), "excluded": status.get("excluded"), "unassigned": status.get("unassigned")}
    if not labels:
        return {**base, "available": False, "labelled": 0}
    papers = {str(cid): (title, authors, year, source) for cid, title, authors, year, source in con.execute(
        "SELECT canonical_id,title,authors,publication_year,source_title FROM canonical_paper WHERE run_id=%s", (run_id,))}
    group_of = {pair: key for key, _, pair in GROUPS}
    rows = [(cid, group_of[(row["cto"], row["tim"])]) + papers[cid][2:] for cid, row in labels.items() if cid in papers and (row["cto"], row["tim"]) in group_of]
    total = len(rows)
    counts = Counter(group for _, group, *_ in rows)
    by_year, by_source = {}, {}
    for _, group, year, source in rows:
        by_year.setdefault(year, Counter())[group] += 1
        by_source.setdefault(source or "(no journal)", Counter())[group] += 1
    shape = lambda c: {"total": sum(c.values()), **{key: c.get(key, 0) for key, _, _ in GROUPS}}
    examples = {}
    for key, _, _ in GROUPS:
        ranked = sorted((cid for cid, group, *_ in rows if group == key), key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode()).hexdigest())
        examples[key] = [{"canonical_id": cid, "title": papers[cid][0], "authors": papers[cid][1], "year": papers[cid][2], "source": papers[cid][3]}
                         for cid in ranked[:per_group]]
    return {**base, "available": True, "labelled": total,
            "groups": [{"key": key, "label": label, "papers": counts.get(key, 0), "share": _share(counts.get(key, 0), total)} for key, label, _ in GROUPS],
            "communities": {name: {"papers": counts["both"] + counts[only], "share": _share(counts["both"] + counts[only], total)}
                            for name, only in (("CTO", "cto_only"), ("TIM", "tim_only"))},
            "by_year": [{"year": year, **shape(c)} for year, c in sorted(by_year.items(), key=lambda kv: (kv[0] is None, kv[0]))],
            "by_source": sorted(({"source": source, **shape(c)} for source, c in by_source.items()), key=lambda r: (-r["total"], r["source"])),
            "examples": examples}


def start(con, run_id):
    directory = run_dir(run_id)
    if (directory / "batch.json").exists():
        raise ValueError("Assignment already started; inspect status before retrying")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required to start assignment")
    if not (directory / "manifest.json").exists():
        LOG.info("Community assignment: preparing canonical papers for run %s", run_id)
        prepare_catalog(con, run_id, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    LOG.info("Community assignment: submitting %s requests for %s papers", manifest["request_count"], manifest["selected_records"])
    result = submit(directory)
    LOG.info("Community assignment: batch submitted, status=%s", result["status"])
    return result


def refresh(run_id):
    directory = run_dir(run_id)
    attempt, _ = current_attempt(directory)
    if not (attempt / "batch.json").exists():
        raise ValueError("No submitted batch")
    result = collect_remote(attempt, SCOPE)
    after = assignment_status(run_id)
    counts = after["request_counts"] or {}
    LOG.info("Community assignment: status=%s requests=%s/%s completed, %s failed; papers=%s/%s assigned",
             after["status"], counts.get("completed", 0), counts.get("total") or after["request_total"],
             counts.get("failed", 0), after["assigned"], after["eligible"])
    return result


def create_sample(con, run_id, size=SAMPLE_SIZE, threshold=EXPECTED_KAPPA):
    if not 0 <= threshold <= 1 or size < 2:
        raise ValueError("Invalid sample size or threshold")
    if con.execute("SELECT 1 FROM blind_sample WHERE run_id=%s", (run_id,)).fetchone():
        return sample_status(con, run_id)
    # The sample is drawn from every eligible paper, not only those the Batch has already labelled:
    # Batch results arrive in request order, so sampling from them would not be random.
    # Model comparisons are computed later, once the sampled papers have Batch assignments.
    candidates = [cid for (cid,) in con.execute(
        "SELECT canonical_id FROM canonical_paper WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0", (run_id,)
    )]
    manifest_path = run_dir(run_id) / "manifest.json"
    if manifest_path.exists():
        submitted = {rid for ids in json.loads(manifest_path.read_text(encoding="utf-8")).get("requests", {}).values() for rid in ids}
        if submitted:
            candidates = [cid for cid in candidates if str(cid) in submitted]
    if len(candidates) < size:
        raise ValueError(f"Only {len(candidates)} eligible papers; need {size} for the blind sample")
    # PostgreSQL generates a fresh random order; the selected order is stored permanently.
    chosen = [cid for (cid,) in con.execute(
        "SELECT canonical_id FROM canonical_paper WHERE run_id=%s AND canonical_id=ANY(%s) ORDER BY random() LIMIT %s",
        (run_id, candidates, size))]
    with con.transaction():
        con.execute("INSERT INTO blind_sample(run_id,threshold) VALUES (%s,%s)", (run_id, threshold))
        for position, cid in enumerate(chosen, 1):
            con.execute("INSERT INTO blind_sample_paper(run_id,canonical_id,position) VALUES (%s,%s,%s)", (run_id, cid, position))
    return sample_status(con, run_id)


def sample_status(con, run_id, validator=None):
    row = con.execute("SELECT threshold FROM blind_sample WHERE run_id=%s", (run_id,)).fetchone()
    if not row:
        return {"created": False}
    count = con.execute("SELECT count(*) FROM blind_sample_paper WHERE run_id=%s", (run_id,)).fetchone()[0]
    ratings = list(con.execute(
        "SELECT DISTINCT ON (validator,canonical_id) validator,canonical_id,cto,tim "
        "FROM blind_rating WHERE run_id=%s ORDER BY validator,canonical_id,rated_at DESC,rating_id DESC", (run_id,)))
    progress = Counter(r[0] for r in ratings)
    result = {"created": True, "size": count, "threshold": row[0], "reviewers": len(progress),
              "completed_reviewers": sum(n == count for n in progress.values()),
              "my_completed": progress.get(validator, 0)}
    return result


def blind_papers(con, run_id, validator):
    rows = con.execute(
        "SELECT s.canonical_id,s.position,c.title,c.publication_year,r.raw_fields->>'Abstract',"
        "r.raw_fields->>'Author Keywords',r.raw_fields->>'Index Keywords' "
        "FROM blind_sample_paper s JOIN canonical_paper c USING (canonical_id) "
        "JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE s.run_id=%s ORDER BY s.position", (run_id,))
    ratings = {str(cid): {"cto": cto, "tim": tim} for cid, cto, tim in con.execute(
        "SELECT DISTINCT ON (canonical_id) canonical_id,cto,tim FROM blind_rating "
        "WHERE run_id=%s AND validator=%s ORDER BY canonical_id,rated_at DESC,rating_id DESC", (run_id, validator))}
    return [{"id": str(cid), "position": pos, "title": title, "year": year, "abstract": abstract,
             "author_keywords": ak, "index_keywords": ik, "rating": ratings.get(str(cid))}
            for cid, pos, title, year, abstract, ak, ik in rows]


def rate(con, run_id, canonical_id, validator, cto, tim):
    if not validator or len(validator) > 80 or type(cto) is not int or type(tim) is not int or cto not in (0, 1) or tim not in (0, 1):
        raise ValueError("Validator and binary CTO/TIM ratings are required")
    if not con.execute("SELECT 1 FROM blind_sample_paper WHERE run_id=%s AND canonical_id=%s", (run_id, canonical_id)).fetchone():
        raise ValueError("Paper is not in this blind sample")
    con.execute("INSERT INTO blind_rating(run_id,canonical_id,validator,cto,tim) VALUES (%s,%s,%s,%s,%s)",
                (run_id, canonical_id, validator, cto, tim))


def kappa(pairs):
    n = len(pairs)
    if not n:
        raise ValueError("At least one paired rating is required")
    observed = sum(a == b for a, b in pairs) / n
    a1 = sum(a for a, _ in pairs) / n
    b1 = sum(b for _, b in pairs) / n
    expected = a1 * b1 + (1 - a1) * (1 - b1)
    return {"n": n, "agreement": observed, "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def completed_ratings(con, run_id, minimum=2):
    """minimum=1 lets the agreement report run with a single annotator (annotator vs Batch only)."""
    sample = sample_status(con, run_id)
    if not sample["created"] or sample["completed_reviewers"] < minimum:
        raise ValueError("Two annotators must complete every blind paper" if minimum == 2
                         else "At least one annotator must complete every blind paper")
    rows = list(con.execute(
        "SELECT DISTINCT ON (validator,canonical_id) validator,canonical_id,cto,tim FROM blind_rating "
        "WHERE run_id=%s ORDER BY validator,canonical_id,rated_at DESC,rating_id DESC", (run_id,)))
    grouped = {}
    for name, cid, cto, tim in rows:
        grouped.setdefault(name, {})[cid] = (cto, tim)
    complete = sorted(name for name, values in grouped.items() if len(values) == sample["size"])
    if len(complete) not in range(minimum, 3):
        raise ValueError("Exactly two complete annotators are required for pairwise Cohen kappa" if minimum == 2
                         else "One or two complete annotators are required (more than two is not supported)")
    return sample, complete, grouped


def adjudication_items(con, run_id):
    _, names, grouped = completed_ratings(con, run_id)
    a, b = (grouped[name] for name in names)
    latest = {cid: (cto, tim) for cid, cto, tim in con.execute(
        "SELECT DISTINCT ON (canonical_id) canonical_id,cto,tim FROM community_adjudication "
        "WHERE run_id=%s ORDER BY canonical_id,decided_at DESC,adjudication_id DESC", (run_id,))}
    papers = {cid: (title, abstract) for cid, title, abstract in con.execute(
        "SELECT s.canonical_id,c.title,r.raw_fields->>'Abstract' "
        "FROM blind_sample_paper s JOIN canonical_paper c USING (canonical_id) "
        "JOIN export_record r ON r.record_id=c.representative_record_id WHERE s.run_id=%s", (run_id,))}
    return [{"id": str(cid), "title": papers[cid][0], "abstract": papers[cid][1],
             "ratings": {names[0]: a[cid], names[1]: b[cid]},
             "final": latest.get(cid)}
            for cid in sorted(a) if a[cid] != b[cid]]


def adjudicate(con, run_id, canonical_id, adjudicator, cto, tim):
    if not adjudicator or len(adjudicator) > 80 or type(cto) is not int or type(tim) is not int or cto not in (0, 1) or tim not in (0, 1):
        raise ValueError("Adjudicator and binary CTO/TIM ratings are required")
    _, names, grouped = completed_ratings(con, run_id)
    if canonical_id not in grouped[names[0]] or grouped[names[0]][canonical_id] == grouped[names[1]][canonical_id]:
        raise ValueError("Paper has no annotator disagreement")
    con.execute("INSERT INTO community_adjudication(run_id,canonical_id,adjudicator,cto,tim) VALUES (%s,%s,%s,%s,%s)",
                (run_id, canonical_id, adjudicator, cto, tim))


def agreement(con, run_id, allow_partial=False):
    """allow_partial=True compares with the Batch on the papers that already have a collected label (the result says how many)."""
    sample, complete, grouped = completed_ratings(con, run_id, minimum=1)
    a = grouped[complete[0]]
    ids = sorted(a)
    result = {"annotators": complete, "threshold": sample["threshold"]}
    model = assignment_rows(run_dir(run_id))
    missing = [str(cid) for cid in ids if str(cid) not in model]
    result["batch_missing"] = len(missing)
    # Papers used for the Batch comparison: all of them, or (explicit opt-in) only those with a collected Batch label.
    compared = [cid for cid in ids if str(cid) in model] if allow_partial else ([] if missing else ids)
    result["batch_compared"] = len(compared)
    result["batch_partial"] = bool(allow_partial and missing)
    if result["batch_partial"]:
        result["batch_missing_ids"] = missing
    batch = lambda cid, label: int(model[str(cid)][label.lower()])
    if len(complete) == 1:
        # One annotator: no human-human kappa and no disagreements to settle; only annotator vs Batch.
        result["mode"] = "single_annotator"
        for label, index in (("CTO", 0), ("TIM", 1)):
            result[label] = {}
            if compared:
                result[label]["annotator_a_batch"] = kappa([(a[cid][index], batch(cid, label)) for cid in compared])
        return result
    result["mode"] = "two_annotators"
    b = grouped[complete[1]]
    resolved = {cid: (cto, tim) for cid, cto, tim in con.execute(
        "SELECT DISTINCT ON (canonical_id) canonical_id,cto,tim FROM community_adjudication "
        "WHERE run_id=%s ORDER BY canonical_id,decided_at DESC,adjudication_id DESC", (run_id,))}
    disagreements = [cid for cid in ids if a[cid] != b[cid]]
    pending = [cid for cid in disagreements if cid not in resolved]
    result["disagreements"] = len(disagreements)
    result["pending_adjudication"] = len(pending)
    for label, index in (("CTO", 0), ("TIM", 1)):
        metric = kappa([(a[cid][index], b[cid][index]) for cid in ids])
        metric["meets_threshold"] = metric["kappa"] is not None and metric["kappa"] >= sample["threshold"]
        result[label] = {"annotators": metric}
        if not compared:
            continue  # Batch comparison needs a collected Batch label (every sampled paper unless allow_partial)
        result[label]["annotator_a_batch"] = kappa([(a[cid][index], batch(cid, label)) for cid in compared])
        result[label]["annotator_b_batch"] = kappa([(b[cid][index], batch(cid, label)) for cid in compared])
        if not any(cid in pending for cid in compared):
            final = {cid: a[cid] if a[cid] == b[cid] else resolved[cid] for cid in compared}
            result[label]["final_batch"] = kappa([(final[cid][index], batch(cid, label)) for cid in compared])
    return result
