"""One click, three Batches over the same parsed papers: construct extraction, the methodological card and the card of contributions (18 Oct 2026, the user's request).

What the click does, in order: it parses the PDFs that were never parsed (free, local), looks for the papers that each of the three kinds has not seen yet, writes the three request
files, checks the estimate against the ceiling the person typed, and only then submits the three Batches. Nothing is sent before the ceiling check, and nothing is sent twice: a paper
that is in a submitted Batch is recorded in `processing_item`, so a second click, or a click after new papers were imported, sends only what is new.

The three Batches share one directory per click (`<PROCESSING_DIR>/<job>/constructs|method|impact`), each laid out like the extraction runs already are (requests.jsonl, manifest.json,
batch.json, batch_output.jsonl), so `community-submit` and `extract-collect` keep working on the constructs one. Collecting a constructs Batch goes to staging, as always, and waits for a
person to promote it. Collecting a card Batch validates every card against the text it was made from (an exact quotation or nothing) and stores it in `paper_card`, valid or rejected
with the reason.

Not done here, and said where it matters: a rejected card is not sent again by a later click (the paper stays recorded as submitted; retrying it is a deliberate step), and the
constructs of papers added later are collected into the same staged run as before, which `extract-promote` refuses as "already in the live tables" (an open decision, the internal project notes §10).
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import documents, extraction, paper_cards, scoring_io
from .community_phase1 import MODEL, extract_output_text, submit
from .core import digest, stable_id

LOG = logging.getLogger("research0")
KINDS = ("constructs", "method", "impact")
CARD_KINDS = ("method", "impact")
PROFILES = paper_cards.PROFILES
SOURCES = {"method": paper_cards.method_source, "impact": paper_cards.impact_source}
CARD_MAX_OUTPUT_TOKENS = 8000
MIN_SOURCE_CHARS = 1500          # a text shorter than this has no section worth a card
CHARS_PER_TOKEN = 3              # deliberately high, as in scoring_judges.estimate_tokens: the estimate must not undershoot
END_STATES = ("completed", "failed", "expired", "cancelled")


def prompt_version(kind):
    return extraction.PROMPT_VERSION if kind == "constructs" else PROFILES[kind].version


def base_directory():
    return Path(os.environ.get("PROCESSING_DIR") or "reports/processing")


# --- which papers each kind has not seen -------------------------------------------------------------------------------------------------------

def latest_ingestion(con, run_id):
    row = con.execute("SELECT ingestion_run_id FROM ingestion_run WHERE run_id=%s ORDER BY created_at DESC LIMIT 1", (run_id,)).fetchone()
    return row[0] if row else None


def parsed_versions(con, ingestion_run_id):
    """[(version_id, canonical_id)] of the parsed papers of an ingestion run, those excluded from the full-text work left out (the latest decision wins), as in `extraction.prepare`."""
    return con.execute(
        "SELECT v.version_id,v.canonical_id FROM paper_version v JOIN ingestion_run i USING (ingestion_run_id) WHERE v.ingestion_run_id=%s AND v.status='parsed' "
        "AND NOT coalesce((SELECT x.excluded FROM fulltext_exclusion x WHERE x.run_id=i.run_id AND x.canonical_id=v.canonical_id ORDER BY x.decided_at DESC, x.exclusion_id DESC LIMIT 1), false) "
        "ORDER BY v.canonical_id,v.version_id", (ingestion_run_id,)).fetchall()


def done_versions(con, kind):
    """The versions this kind must not send again: already in a submitted Batch, already extracted by this prompt version, or already holding a valid card."""
    version = prompt_version(kind)
    done = {str(r[0]) for r in con.execute("SELECT version_id FROM processing_item WHERE kind=%s AND prompt_version=%s", (kind, version))}
    if kind == "constructs":
        for mention, run in (("construct_mention", "extraction_run"), ("staged_construct_mention", "staged_extraction_run")):
            done |= {str(r[0]) for r in con.execute(f"SELECT DISTINCT m.version_id FROM {mention} m JOIN {run} r USING (extraction_run_id) WHERE r.prompt_version=%s", (version,))}
    else:
        done |= {str(r[0]) for r in con.execute("SELECT version_id FROM paper_card WHERE kind=%s AND profile_version=%s AND status='valid'", (kind, version))}
    return done


def register_sent(con, directory, kind, job=None):
    """Record as already sent the papers of a Batch that was submitted before `processing_item` existed (the construct extraction of 13 Oct 2026, for instance). Without this a paper
    that came back with no construct at all looks like one never sent and the next click would send it again. The directory's own manifest says what was sent and with which prompt
    version, and that version must be the one `kind` uses now. -> how many papers were recorded (those already recorded count as zero)."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_version"] != prompt_version(kind):
        raise ValueError(f"That Batch used {manifest['prompt_version']}, and {kind} is now {prompt_version(kind)}: it does not cover this prompt version")
    if not (directory / "batch.json").exists():
        raise ValueError("That directory has no batch.json: nothing was submitted from it")
    job = job or f"registered:{directory.name}"
    ids = [v for (v,) in con.execute("SELECT version_id::text FROM paper_version WHERE version_id = ANY(%s)", (list(manifest["requests"]),))]
    recorded = 0
    for version_id in ids:
        recorded += con.execute("INSERT INTO processing_item(kind,version_id,prompt_version,job) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                                (kind, version_id, manifest["prompt_version"], job)).rowcount
    return recorded


def pending(con, ingestion_run_id, limit=None):
    """{kind: [(version_id, canonical_id)]}. `limit` caps the PAPERS, so the three kinds work on the same ones: the first `limit` papers that at least one kind has not seen."""
    versions = parsed_versions(con, ingestion_run_id)
    done = {k: done_versions(con, k) for k in KINDS}
    todo = [(v, c) for v, c in versions if any(str(v) not in done[k] for k in KINDS)]
    if limit:
        todo = todo[:limit]
    return {k: [(v, c) for v, c in todo if str(v) not in done[k]] for k in KINDS}


def unparsed_pdfs(con, run_id, ingestion_run_id):
    """How many stored PDFs the click would parse first (none when there is no ingestion run yet: all of them)."""
    return con.execute(
        "SELECT count(*) FROM fulltext_file f WHERE f.run_id=%(run)s "
        "AND NOT coalesce((SELECT x.excluded FROM fulltext_exclusion x WHERE x.run_id=f.run_id AND x.canonical_id=f.canonical_id ORDER BY x.decided_at DESC, x.exclusion_id DESC LIMIT 1), false) "
        "AND NOT EXISTS (SELECT 1 FROM paper_version v WHERE v.file_id=f.file_id AND v.ingestion_run_id=%(ing)s)", {"run": run_id, "ing": ingestion_run_id}).fetchone()[0]


# --- the card requests ------------------------------------------------------------------------------------------------------------------------

def card_request(kind, version_id, source, model=MODEL):
    profile = PROFILES[kind]
    return {"custom_id": str(version_id), "method": "POST", "url": "/v1/responses", "body": {
        "model": model, "instructions": profile.card_system, "input": f"INPUT_ID: {version_id}\n\n<<<TESTO\n{source}\n>>>FINE_TESTO\n",
        "text": {"format": {"type": "json_schema", "name": profile.card_task, "strict": True, "schema": scoring_io.provider_schema(paper_cards.card_schema(profile.fields))}},
        "max_output_tokens": CARD_MAX_OUTPUT_TOKENS}}


def request_tokens(request):
    return (len(request["body"]["instructions"]) + len(request["body"]["input"])) // CHARS_PER_TOKEN + 1


def prepare_cards(con, ingestion_run_id, directory, kind, version_ids, job, model=MODEL):
    """Write the requests of one kind of card (one per paper) and a manifest that remembers, for each, the hash of the exact text it was made from. -> manifest."""
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("Requests already prepared in this directory")
    profile, requests, entries, skipped, tokens = PROFILES[kind], [], {}, {}, 0
    ids = sorted(str(v) for v in version_ids)
    for start in range(0, len(ids), 50):                 # in groups, so the full texts of the whole run are never held at once
        for vid, text in con.execute("SELECT version_id,full_text FROM paper_version WHERE version_id = ANY(%s) ORDER BY version_id", (ids[start:start + 50],)):
            source, info = SOURCES[kind](text or "")
            if len(source) < MIN_SOURCE_CHARS:
                skipped[str(vid)] = f"the text has only {len(source)} usable characters"
                continue
            request = card_request(kind, vid, source, model)
            requests.append(json.dumps(request, ensure_ascii=False))
            tokens += request_tokens(request)
            entries[str(vid)] = {"source_sha256": digest(source.encode("utf-8")), "source": {k: info[k] for k in ("kind", "total_chars", "used_chars", "truncated")}}
    if not requests:
        raise ValueError(f"No paper has a usable text for the {kind} card")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "requests.jsonl").write_text("\n".join(requests) + "\n", encoding="utf-8")
    manifest = {"kind": kind, "job": job, "ingestion_run_id": str(ingestion_run_id), "model": model, "prompt_version": profile.version, "unit": "paper",
                "instructions_sha256": digest(profile.card_system.encode("utf-8")), "request_count": len(requests), "estimated_input_tokens": tokens, "requests": entries, "skipped": skipped}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def collect_cards(con, directory, output_jsonl=None):
    """Validate the answers of a card Batch against the text each card was made from, and store them. A card is valid only if the text, rebuilt now from the database, still has the
    hash it had when the request was written, and every quotation is an exact piece of it. -> counts."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    kind, profile = manifest["kind"], PROFILES[manifest["kind"]]
    path = Path(output_jsonl) if output_jsonl else directory / "batch_output.jsonl"
    pending_ids, counts = set(manifest["requests"]), {"valid": 0, "rejected": 0, "failed_requests": 0, "fields_demoted": 0}
    for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else []):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item["custom_id"]
        if custom_id not in pending_ids:
            raise ValueError(f"Unexpected or repeated custom_id: {custom_id}")
        pending_ids.discard(custom_id)
        if item.get("error") or not item.get("response") or item["response"].get("status_code") != 200:
            counts["failed_requests"] += 1
            continue
        row = con.execute("SELECT canonical_id,full_text FROM paper_version WHERE version_id=%s", (custom_id,)).fetchone()
        source, info = SOURCES[kind](row[1] or "") if row else ("", {})
        card, demoted, problems = None, [], []
        try:
            answer = json.loads(extract_output_text(item["response"]["body"]))
            if digest(source.encode("utf-8")) != manifest["requests"][custom_id]["source_sha256"]:
                problems = ["the paper's text changed after the request was written"]
            else:
                card, demoted, problems = paper_cards.repair_card(answer, source, profile)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            problems = [f"the answer could not be read: {str(exc)[:120]}"]
        valid = not problems
        info = {**manifest["requests"][custom_id]["source"], "demoted_fields": demoted}
        con.execute("INSERT INTO paper_card(card_id,kind,version_id,canonical_id,profile_version,model,source_sha256,source_info,status,card,rejection,job) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s) ON CONFLICT DO NOTHING",
                    (stable_id("paper-card", manifest["job"], kind, custom_id), kind, custom_id, row[0], profile.version, manifest["model"], manifest["requests"][custom_id]["source_sha256"],
                     json.dumps(info), "valid" if valid else "rejected", json.dumps(card, ensure_ascii=False) if valid else None,
                     None if valid else "; ".join(problems)[:1000], manifest["job"]))
        counts["valid" if valid else "rejected"] += 1
        counts["fields_demoted"] += len(demoted) if valid else 0
    return {"kind": kind, **counts, "missing_requests": len(pending_ids)}


def collect_cards_remote(con, directory):
    """Fetch a card Batch (explicitly requested), save its output next to the requests and validate it."""
    from openai import OpenAI
    directory = Path(directory)
    state = json.loads((directory / "batch.json").read_text(encoding="utf-8"))
    if not state.get("batch_id"):
        raise ValueError("No batch_id saved; inspect the uploaded file before resubmitting")
    client = OpenAI()
    batch = client.batches.retrieve(state["batch_id"])
    state.update(status=batch.status, output_file_id=batch.output_file_id, error_file_id=batch.error_file_id)
    (directory / "batch.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if not batch.output_file_id:
        return {"status": batch.status, "valid": 0, "rejected": 0}
    (directory / "batch_output.jsonl").write_bytes(client.files.content(batch.output_file_id).read())
    return {"status": batch.status, **collect_cards(con, directory)}


# --- the click ------------------------------------------------------------------------------------------------------------------------------------

def new_job_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")           # microseconds: two clicks in the same second must not share a directory


def estimate(con, run_id, limit=None):
    """What a click would do now, without writing or sending anything: PDFs to parse, papers and approximate input tokens per kind. The tokens of papers that are not parsed yet cannot
    be counted and are not in the figures."""
    ingestion = latest_ingestion(con, run_id)
    to_parse = unparsed_pdfs(con, run_id, ingestion) if ingestion else con.execute("SELECT count(*) FROM fulltext_file WHERE run_id=%s", (run_id,)).fetchone()[0]
    out = {"ingestion_run_id": str(ingestion) if ingestion else None, "pdfs_to_parse": to_parse, "kinds": {}}
    todo = pending(con, ingestion, limit) if ingestion else {k: [] for k in KINDS}
    for kind in KINDS:
        ids = [str(v) for v, _ in todo[kind]]
        tokens = 0
        for start in range(0, len(ids), 50):
            for (text,) in con.execute("SELECT full_text FROM paper_version WHERE version_id = ANY(%s)", (ids[start:start + 50],)):
                if kind == "constructs":
                    tokens += len(text or "") // CHARS_PER_TOKEN + 1                 # the whole text: an upper bound, the request leaves out references and appendices
                else:
                    source, _ = SOURCES[kind](text or "")
                    tokens += (len(source) + len(PROFILES[kind].card_system)) // CHARS_PER_TOKEN + 1 if len(source) >= MIN_SOURCE_CHARS else 0
        out["kinds"][kind] = {"papers": len(ids), "estimated_input_tokens": tokens}
    out["estimated_input_tokens"] = sum(k["estimated_input_tokens"] for k in out["kinds"].values())
    return out


def prepare_job(con, run_id, store=None, limit=None, base=None):
    """Parse the new PDFs (when a store is given), then write the requests of the three kinds into a new job directory. Sends nothing. -> plan."""
    base = Path(base) if base else base_directory()
    ingestion = documents.ingest(con, run_id, store)[0] if store is not None else latest_ingestion(con, run_id)
    if ingestion is None:
        raise ValueError("No PDF has been parsed yet and no store was given to parse them")
    con.commit()
    todo = pending(con, ingestion, limit)
    if not any(todo.values()):
        raise ValueError("Every parsed paper has already been sent for all three kinds")
    job = new_job_name()
    root = base / job
    root.mkdir(parents=True, exist_ok=False)
    plan = {"job": job, "directory": str(root), "run_id": str(run_id), "ingestion_run_id": str(ingestion), "limit": limit, "kinds": {}}
    for kind in KINDS:
        if not todo[kind]:
            plan["kinds"][kind] = {"requests": 0, "estimated_input_tokens": 0, "note": "nothing new"}
            continue
        directory = root / kind
        try:
            if kind == "constructs":
                manifest = extraction.prepare(con, ingestion, directory, canonical_ids=[c for _, c in todo[kind]])
                tokens = sum(request_tokens(json.loads(line)) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
                manifest["estimated_input_tokens"] = tokens
                manifest["job"] = job
                (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            else:
                manifest = prepare_cards(con, ingestion, directory, kind, [v for v, _ in todo[kind]], job)
        except ValueError as exc:
            plan["kinds"][kind] = {"requests": 0, "estimated_input_tokens": 0, "note": str(exc)[:200]}
            continue
        plan["kinds"][kind] = {"requests": manifest["request_count"], "estimated_input_tokens": manifest["estimated_input_tokens"], "skipped": len(manifest.get("skipped", {}))}
    plan["estimated_input_tokens"] = sum(k["estimated_input_tokens"] for k in plan["kinds"].values())
    root.mkdir(parents=True, exist_ok=True)
    (root / "job.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def submit_job(con, root, max_input_tokens):
    """Submit the prepared Batches of a job, but only if the estimate is within the ceiling the person gave. Resumable: a kind that already has a batch.json is not sent again. The
    papers of each Batch are recorded in `processing_item` right after it was accepted. -> {kind: batch_id}."""
    root = Path(root)
    plan = json.loads((root / "job.json").read_text(encoding="utf-8"))
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("Give the ceiling of input tokens as a whole number")
    if plan["estimated_input_tokens"] > max_input_tokens:
        raise ValueError(f"The three Batches are estimated at about {plan['estimated_input_tokens']:,} input tokens, above the ceiling of {max_input_tokens:,}. Nothing was sent: "
                         "raise the ceiling or lower the number of papers")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required to submit the Batches")
    submitted = {}
    for kind in KINDS:
        directory = root / kind
        if not (directory / "manifest.json").exists():
            continue
        if (directory / "batch.json").exists():
            submitted[kind] = json.loads((directory / "batch.json").read_text(encoding="utf-8")).get("batch_id")
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        state = submit(directory)
        for version_id in manifest["requests"]:
            con.execute("INSERT INTO processing_item(kind,version_id,prompt_version,job) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", (kind, version_id, manifest["prompt_version"], plan["job"]))
        con.commit()          # right away: the Batch exists, so a later failure must not undo the record that stops these papers from being sent again
        submitted[kind] = state["batch_id"]
        LOG.info(json.dumps({"event": "processing_batch_submitted", "job": plan["job"], "kind": kind, "batch_id": state["batch_id"], "requests": manifest["request_count"]}))
    plan["submitted"] = submitted
    (root / "job.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return submitted


def job_status(root):
    """Read-only state of a job, from its files."""
    root = Path(root)
    plan = json.loads((root / "job.json").read_text(encoding="utf-8"))
    kinds = {}
    for kind in KINDS:
        directory = root / kind
        entry = {"requests": plan["kinds"].get(kind, {}).get("requests", 0), "estimated_input_tokens": plan["kinds"].get(kind, {}).get("estimated_input_tokens", 0)}
        if (directory / "batch.json").exists():
            state = json.loads((directory / "batch.json").read_text(encoding="utf-8"))
            entry.update(batch_id=state.get("batch_id"), status=state.get("status"), request_counts=state.get("request_counts"))
        if (directory / "collected.json").exists():
            entry["collected"] = json.loads((directory / "collected.json").read_text(encoding="utf-8"))
        kinds[kind] = entry
    return {"job": plan["job"], "directory": str(root), "limit": plan.get("limit"), "estimated_input_tokens": plan["estimated_input_tokens"], "kinds": kinds}


def refresh_job(con, root):
    """Ask OpenAI for the state of each submitted Batch and collect those that ended: constructs into staging, cards into `paper_card`. A kind already collected is left alone."""
    root = Path(root)
    for kind in KINDS:
        directory = root / kind
        if not (directory / "batch.json").exists() or (directory / "collected.json").exists():
            continue
        result = extraction.collect_remote(con, directory, staged=True) if kind == "constructs" else collect_cards_remote(con, directory)
        con.commit()
        if result.get("status") in END_STATES:
            (directory / "collected.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return job_status(root)


def latest_job(base=None):
    base = Path(base) if base else base_directory()
    jobs = sorted(p for p in base.glob("*/job.json")) if base.is_dir() else []
    return jobs[-1].parent if jobs else None


class ProcessingJob:
    """The click, in the background: parse, prepare, check the ceiling, submit. One at a time per server. Progress is in memory; the files and `processing_item` are the record."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"state": "idle"}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def _set(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, run_id, limit, max_input_tokens, database_url, store):
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 100000):
            raise ValueError("Limit must be a whole number between 1 and 100000")
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError("Give the ceiling of input tokens as a whole number")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required to submit the Batches")
        with self.lock:
            if self.state["state"] == "running":
                raise ValueError("The papers are already being processed")
            self.state = {"state": "running", "phase": "parsing", "limit": limit, "max_input_tokens": max_input_tokens, "plan": None, "error": None,
                          "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        threading.Thread(target=self._run, args=(run_id, limit, max_input_tokens, database_url, store), daemon=True).start()
        return self.snapshot()

    def _run(self, run_id, limit, max_input_tokens, database_url, store):
        import psycopg
        try:
            with psycopg.connect(database_url) as con:
                self._set(phase="preparing")
                plan = prepare_job(con, run_id, store, limit)
                self._set(plan=plan, phase="submitting")
                submit_job(con, plan["directory"], max_input_tokens)
            self._set(state="finished", phase="submitted")
        except Exception as exc:        # reported to the page; what was prepared or submitted stays on disk and in the database
            LOG.error(json.dumps({"event": "processing_job_failed", "run_id": str(run_id), "error": str(exc)[:300]}))
            self._set(state="failed", error=str(exc)[:300])
        finally:
            self._set(finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
