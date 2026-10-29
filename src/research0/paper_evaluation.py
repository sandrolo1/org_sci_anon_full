"""Evaluation of published papers (20 Oct 2026, the user's requests): the "Evaluation" page of the dashboard, the steps that prepare its papers, and the runs.

What it does: it takes the 2026 papers of `inputs/evaluation_2026/` (outside the Scopus perimeter of the corpus, which ends in 2025) and gives each the SAME evaluation a forum proposal gets,
through the same code (`forum_scoring.score_proposal`, prompts, dossiers and checks untouched):
  * NOVELTY from the title and abstract: the closest abstracts of the CTO/TIM corpus (by embedding, and by BM25 for the hybrid judge), two votes - first judge and hybrid judge;
  * METHODOLOGY and IMPACT from the paper's two cards, against the cards of the corpus papers. A proposal's cards are written by its author; a paper's cards are made exactly as the
    corpus cards were (18 Oct 2026, job 20261018T134621203476Z): from the full text, the same section selection (`paper_cards.method_source` / `impact_source`), the same instructions and
    profile versions, the same input format, model and output allowance (`config/evaluation_cards.json`), the same check of the quotations (`paper_cards.repair_card`).

The steps of a sample, in order (each a button on the page; nothing runs by itself):
  1. the abstracts are already in the folder;
  2. FULL TEXTS: only those the project already has (the user's decision, 20 Oct 2026: nothing is downloaded). The PDFs of the 2026 papers of the ANVUR package of 17 Oct 2026
     (open-access copies collected then, kept in git at commit 87c4aaf), copied with their index to `fulltext_archive/`; and any PDF / .docx a person puts in `fulltext_manual/`
     named after the paper id or the DOI (principle 8 (b)), which wins over the archive. By default the sample is drawn among the papers that have one;
     2b. DOWNLOAD (21 Oct 2026, the user's request, replacing "nothing is downloaded"): a button tries the papers that have no full text with the resolver of the corpus
     (`fulltext.acquire`: only what Unpaywall, OpenAlex or Semantic Scholar declare open access, from the DOI; principle 8 (a)), in the order of the sample's draw, and imports and
     parses each PDF found (`fulltext_download/`); every attempt is a row, the failed ones too. The papers still without one are exported as an Excel list to fetch by hand;
     the RECONCILIATION (`reconcile`) says, stage by stage, how many papers reach the benchmark and why each other one does not;
  3. PARSING, local and free, when the full texts are imported (`documents.parse_document`, the parser of the corpus); the PDF and the parsed text stay in the folder, outside git;
  4. CARDS: one methodology and one impact card per parsed paper (by default one OpenAI Batch, collected by a later click; or synchronous calls; a token ceiling and a confirmation);
  5. the RUN: novelty alone, or the three criteria of a proposal.

Rules of the user, kept here and tested:
  * The judge sees ONLY what a proposal's judge sees: title and abstract, and for methodology and impact the card. Never the journal, its ABS grade, the DOI (its prefix names the publisher)
    or the year: the file of the abstracts has no such field (a field outside {paper_id, title, abstract, abstract_sha256} stops everything), and a card field whose text or quotation
    names the paper's journal is turned into "not stated" (`guard_card`). The journal and the grade are copied to `evaluation_paper` and used only to SHOW the results.
  * No internet for the judges: no tool and no web search (`scoring_judges.Judges.ask`); the neighbours come from the local corpora.
  * Judges whose knowledge ends before the papers: each judge of `config/evaluation_pre2026.json` declares its `knowledge_cutoff`, and a run is refused unless it precedes the earliest
    online date of the sample. A precondition, not a proof (preprints; a cutoff is the provider's statement). The card model is the corpus's own (it describes, with exact quotations).
  * A paper already in the corpus (same DOI or title) is left out of the sample.

Runs, scores, full-text attempts and cards are rows (migrations 031 and 032), append-only; the newest of each paper counts. Every step that spends or downloads is refused on a read-only
deployment (the server answers 403 before it gets here).
"""
import csv
import datetime as dt
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import statistics
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import forum_scoring as fscore
from . import paper_cards, scoring_corpora, scoring_io, scoring_judges
from .forum_providers import BudgetExceeded, Cancelled

LOG = logging.getLogger("research0")
VERSION = "paper-evaluation-v1"
ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_FIELDS = frozenset({"paper_id", "title", "abstract", "abstract_sha256"})
GRADES = ("1", "2", "3", "4", "4*")                  # ABS / AJG 2024, lowest to highest
CRITERIA = fscore.CRITERIA                            # novelty, methodology, impact
KINDS = ("novelty", "full")                           # what a run judges: novelty alone, or the three criteria of a proposal
CARD_KINDS = ("method", "impact")
DEFAULT_N = 80
DEFAULT_SEED = 20261020
DEFAULT_MIN_WORDS = 40                                # below this the text is a column's lead-in, not an abstract
DEFAULT_MAX_TOKENS = 3_000_000
MIN_SOURCE_CHARS = 1500                               # as paper_processing: a text shorter than this has no section worth a card
SALT = "evaluation-2026"
DOCUMENT_SUFFIXES = (".pdf", ".docx")


def settings():
    """Where the papers, the configurations, the register of the calls and the three corpora are: the environment or the project's own copies."""
    def pick(env, default):
        value = os.environ.get(env)
        return Path(value) if value else default
    directory = pick("EVALUATION_DIR", ROOT / "inputs" / "evaluation_2026")
    return {"directory": directory, "config": pick("EVALUATION_CONFIG", ROOT / "config" / "evaluation_pre2026.json"),
            "cards_config": pick("EVALUATION_CARDS_CONFIG", ROOT / "config" / "evaluation_cards.json"),
            "workspace": pick("EVALUATION_WORKSPACE", ROOT / "reports" / "evaluation_2026" / "workspace"),
            "novelty": pick("FORUM_SCORING_NOVELTY_CORPUS", fscore.DEFAULTS["novelty"][1]),
            "methodology": pick("FORUM_SCORING_METHOD_CORPUS", scoring_corpora.newest_card_corpus(fscore.CORPORA, "method")),
            "impact": pick("FORUM_SCORING_IMPACT_CORPUS", scoring_corpora.newest_card_corpus(fscore.CORPORA, "impact"))}


def _dir(where, name):
    return Path(where["directory"]) / name             # fulltext/, parsed/, fulltext_manual/ live next to the abstracts


# --- what the judge reads, and what only the page shows ------------------------------------------------------------------------------------------

def read_inputs(directory):
    """{paper_id: {"title", "abstract", "abstract_sha256"}} from `evaluator_input/abstracts.jsonl`. A field outside the four allowed stops everything: the judge must not be able to see more."""
    rows, out = scoring_io.read_jsonl(Path(directory) / "evaluator_input" / "abstracts.jsonl"), {}
    for r in rows:
        extra = set(r) - EVALUATOR_FIELDS
        if extra:
            raise ValueError(f"evaluator_input/abstracts.jsonl has fields the judge must not see: {sorted(extra)}")
        out[r["paper_id"]] = {"title": r["title"], "abstract": r["abstract"], "abstract_sha256": r.get("abstract_sha256", "")}
    return out


def read_display(directory):
    """{paper_id: row} from `display_metadata/papers.csv`: journal, ABS grade, DOI, date, source sets. For the results page and the steps (the DOI finds the PDF), never for a judge."""
    path = Path(directory) / "display_metadata" / "papers.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8", newline="") as stream:
        return {r["paper_id"]: r for r in csv.DictReader(stream)}


def words(text):
    return len(str(text or "").split())


def eligible(inputs, display, *, min_words=DEFAULT_MIN_WORDS, require_grade=True, exclude=(), only=None):
    """The ids a sample may be drawn from: an abstract of at least `min_words` words, an ABS grade when `require_grade`, not in `exclude`, and in `only` when given (the papers whose
    full text the project already has)."""
    out = []
    for pid, r in inputs.items():
        if pid in exclude or words(r["abstract"]) < min_words or (only is not None and pid not in only):
            continue
        if require_grade and not (display.get(pid) or {}).get("ajg2024"):
            continue
        out.append(pid)
    return sorted(out)


def draw(ids, n, seed):
    """n ids, chosen by a hash of (seed, id): the same seed and the same candidates give the same sample, and nothing depends on the order of the file."""
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())[:n]


def already_in_corpus(inputs, display, corpus_entries):
    """Ids of the papers whose DOI or title is an entry of the abstract corpus: they would be their own closest neighbour."""
    dois = {str(e.get("doi") or "").lower() for e in corpus_entries} - {""}
    titles = {scoring_io.title_key(e.get("title")) for e in corpus_entries} - {""}
    return {pid for pid, r in inputs.items() if ((display.get(pid) or {}).get("doi") or "").lower() in dois or scoring_io.title_key(r["title"]) in titles}


def check_cutoffs(cfg, earliest_online):
    """-> (the latest knowledge cutoff of the judges, as a date). Each judge must DECLARE one and it must precede `earliest_online` (a date), or ValueError."""
    cutoffs = []
    for j in cfg["judges"]:
        declared = (j["model"] or {}).get("knowledge_cutoff")
        if not declared:
            raise ValueError(f"{j['role']}: the model's knowledge_cutoff is not declared in the configuration; a run is refused without it")
        cutoffs.append(dt.date.fromisoformat(declared))
    latest = max(cutoffs)
    if earliest_online is not None and latest >= earliest_online:
        raise ValueError(f"the model's knowledge cutoff ({latest}) does not precede the earliest paper of the sample ({earliest_online}): it could have seen them")
    return latest


def kept_ids(con, run_id):
    """The papers of an earlier run, in the order of that run: what `sample(keep=...)` keeps when a run is extended."""
    try:
        rid = uuid.UUID(str(run_id))
    except ValueError:
        raise ValueError("the run to extend is not a valid run id")
    row = con.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (rid,)).fetchone()
    if row is None:
        raise ValueError("the run to extend is unknown")
    return list(row[0]["paper_ids"])


def with_keep(con, options, where=None):
    """`options` of a sample with `extend_run` (a run id) replaced by `keep` (the ids of that run); the rest unchanged."""
    options = dict(options)
    run_id = options.pop("extend_run", None)
    if run_id == "judged":                                      # TEMPORARY: remove with evaluation_reuse.py
        from . import evaluation_reuse
        options["keep"] = evaluation_reuse.keep_judged(con, where or settings(), **options)
    elif run_id:
        options["keep"] = kept_ids(con, run_id)
    return options


UNGRADED = "not graded"


def stratum(display, pid):
    """The ABS 2024 grade of a paper, the stratum of a stratified sample; `UNGRADED` without one."""
    return (display.get(pid) or {}).get("ajg2024") or UNGRADED


def strata_order(names):
    """The strata from the highest grade down, an unknown grade and then `UNGRADED` last: the order that breaks the ties of `allocate`."""
    return sorted(names, key=lambda g: (g == UNGRADED, -GRADES.index(g) if g in GRADES else 1, g))


def allocate(n, capacity, floor=None):
    """{stratum: papers} adding up to n (or to the whole capacity when smaller): as equal as the strata allow. A stratum never gets more than its `capacity` (the papers it has) nor
    less than its `floor` (the papers it already keeps); what a small stratum cannot take goes to the others. Ties go to the higher grade."""
    floor = {g: min(int((floor or {}).get(g, 0)), capacity[g]) for g in capacity}
    out, left = dict(floor), int(n) - sum(floor.values())
    order = strata_order(capacity)
    while left > 0:
        open_ = [g for g in order if out[g] < capacity[g]]
        if not open_:
            break
        g = min(open_, key=lambda x: out[x])                    # the smallest stratum first; min keeps the first of the order on a tie
        out[g] += 1
        left -= 1
    return out


def pool_of(where, *, min_words=DEFAULT_MIN_WORDS, require_grade=True, require_fulltext=True):
    """The papers a sample may be drawn from (before `keep`): -> (pool, inputs, display, corpus, papers already in the corpus)."""
    inputs, display = read_inputs(where["directory"]), read_display(where["directory"])
    if not inputs:
        raise ValueError(f"no abstracts in {where['directory']}/evaluator_input (the folder is local, outside git)")
    if not (Path(where["novelty"]) / "corpus.json").is_file():
        raise ValueError(f"the abstract corpus is not at {where['novelty']}")
    corpus = scoring_corpora.AbstractCorpus(where["novelty"])
    own = already_in_corpus(inputs, display, corpus.entries)
    with_text = set(available_files(where, {p: (display.get(p) or {}).get("doi", "") for p in inputs})) if require_fulltext else None
    return eligible(inputs, display, min_words=int(min_words), require_grade=bool(require_grade), exclude=own, only=with_text), inputs, display, corpus, own


def sample(where, *, n=DEFAULT_N, seed=DEFAULT_SEED, min_words=DEFAULT_MIN_WORDS, require_grade=True, require_fulltext=True, keep=None, stratify=False):
    """The sample every step and the run share: the same filters and seed give the same papers. `require_fulltext`: only papers whose full text the project already has (needed for the
    methodology and impact cards). `keep`: ids of an earlier run to extend (21 Oct 2026): they stay in whatever the filters, the pool or the seed say, and the other n - len(keep) papers
    are drawn by the usual hash among the rest of the pool, so a paper already judged never leaves the sample. `stratify` (21 Oct 2026): the same number of papers for each ABS grade
    (`allocate`), the kept papers counting in their grade, the usual hash within each grade.
    -> {"ids", "pool", "left_out_in_corpus", "strata", "inputs", "display", "corpus"}; `strata` = [{"grade", "pool", "kept", "papers"}], highest grade first."""
    pool, inputs, display, corpus, own = pool_of(where, min_words=min_words, require_grade=require_grade, require_fulltext=require_fulltext)
    keep = list(dict.fromkeys(keep or ()))
    gone = [p for p in keep if p not in inputs]
    if gone:
        raise ValueError(f"{len(gone)} papers of the run to extend are no longer among the abstracts ({gone[0]}...)")
    if not max(1, len(keep)) <= int(n) <= len(inputs):
        raise ValueError(f"n must be between {max(1, len(keep))}" + (" (the papers of the run to extend)" if keep else "") + f" and {len(inputs)}")
    pool = [p for p in pool if p not in set(keep)]
    if len(pool) + len(keep) < int(n):
        raise ValueError(f"only {len(pool) + len(keep)} papers pass the filters ({len(own)} are already in the corpus): lower n or relax the filters")
    by = lambda ids: {g: [p for p in ids if stratum(display, p) == g] for g in {stratum(display, p) for p in pool + keep}}
    pools, kept = by(pool), by(keep)
    if stratify:
        target = allocate(int(n), {g: len(pools[g]) + len(kept[g]) for g in pools}, {g: len(kept[g]) for g in kept})
        ids = keep + draw([p for g in pools for p in draw(pools[g], target[g] - len(kept[g]), int(seed))], int(n) - len(keep), int(seed))
    else:
        ids = keep + draw(pool, int(n) - len(keep), int(seed))
    taken = by(ids)
    strata = [{"grade": g, "pool": len(pools[g]) + len(kept[g]), "kept": len(kept[g]), "papers": len(taken[g])} for g in strata_order(pools)]
    return {"ids": ids, "pool": len(pool) + len(keep), "left_out_in_corpus": len(own), "strata": strata, "inputs": inputs, "display": display, "corpus": corpus}


def _sample_for(database_url, where, options):
    """`sample` for a job: an `extend_run` among the options is resolved against the database first; an empty one (the page's "no, draw a new sample") is dropped."""
    if options.get("extend_run"):
        import psycopg
        with psycopg.connect(database_url) as con:
            options = with_keep(con, options, where)
    options = {k: v for k, v in options.items() if k != "extend_run"}
    return sample(where, **options)


def sync_papers(con, ids, inputs, display):
    """Copy what the page shows of the papers (title, journal, grade...) to `evaluation_paper`. Not the abstract. A grade filled in later replaces an empty one."""
    for pid in ids:
        d, r = display.get(pid) or {}, inputs[pid]
        online = d.get("online_date") or None
        con.execute(
            "INSERT INTO evaluation_paper(paper_id,title,journal,ajg2024,doi,online_date,source_sets,abstract_words,abstract_sha256) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (paper_id) DO UPDATE SET journal=EXCLUDED.journal, ajg2024=EXCLUDED.ajg2024, doi=EXCLUDED.doi, online_date=EXCLUDED.online_date, source_sets=EXCLUDED.source_sets, updated_at=clock_timestamp()",
            (pid, r["title"], d.get("journal", ""), d.get("ajg2024", ""), d.get("doi", ""), online, [s for s in (d.get("source_sets") or "").split(";") if s], words(r["abstract"]), r["abstract_sha256"]))


def _jsonb(value):
    from psycopg.types.json import Jsonb
    return Jsonb(value)


# --- step 2 and 3: the full texts, and their parsing -------------------------------------------------------------------------------------------------

def latest_fulltext(con, ids):
    """{paper_id: the newest full-text attempt, as a dict}."""
    cols = ("origin", "outcome", "detail", "found_by", "version_label", "pdf_url", "landing_url", "sha256", "path", "parse_status", "parse_detail", "text_sha256", "page_count", "created_at")
    rows = con.execute(f"SELECT DISTINCT ON (paper_id) paper_id,{','.join(cols)} FROM evaluation_fulltext WHERE paper_id = ANY(%s) ORDER BY paper_id, created_at DESC", (list(ids),)).fetchall()
    return {r[0]: dict(zip(cols, r[1:])) for r in rows}


def parsed_path(where, pid):
    return _dir(where, "parsed") / f"{pid}.json"


def parsed_text(where, pid):
    """The full text of a parsed paper, or None. It lives in the folder (publisher text, outside git and outside the database)."""
    path = parsed_path(where, pid)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("full_text")


def store_document(where, pid, data):
    """Save a file of a paper in the folder and parse it there (local, free). -> what the attempt row records about the file and its parsing."""
    from . import documents, fulltext
    kind = fulltext.check_document(data)                 # recognised from the content, never from the name
    if kind == "doc":
        raise ValueError("a legacy .doc is not read: save it as .docx or PDF")
    sha = hashlib.sha256(data).hexdigest()
    folder = _dir(where, "fulltext") / pid
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha}.{kind}"
    path.write_bytes(data)
    parsed = documents.parse_document(data, uri=path.name)
    out = {"sha256": sha, "size_bytes": len(data), "path": str(path.relative_to(Path(where["directory"]))), "parse_status": parsed["status"], "parse_detail": parsed.get("detail"),
           "page_count": parsed.get("page_count"), "text_sha256": None}
    if parsed["status"] == "parsed":
        text = parsed["full_text"]
        out["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parsed_path(where, pid).parent.mkdir(parents=True, exist_ok=True)
        parsed_path(where, pid).write_text(json.dumps({"paper_id": pid, "file_sha256": sha, "text_sha256": out["text_sha256"], "page_count": parsed.get("page_count"),
                                                       "language": parsed.get("language"), "full_text": text}, ensure_ascii=False), encoding="utf-8")
    return out


def record_fulltext(con, pid, origin, r):
    con.execute(
        "INSERT INTO evaluation_fulltext(attempt_id,paper_id,origin,outcome,detail,found_by,oa_status,license,version_label,pdf_url,landing_url,sha256,size_bytes,path,parse_status,parse_detail,"
        "text_sha256,page_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (uuid.uuid4(), pid, origin, r["outcome"], (r.get("detail") or None) and str(r["detail"])[:500], r.get("found_by"), r.get("oa_status"), r.get("license"), r.get("version"),
         r.get("pdf_url"), r.get("landing_url"), r.get("sha256"), r.get("size_bytes"), r.get("path"), r.get("parse_status"), (r.get("parse_detail") or None) and str(r["parse_detail"])[:500],
         r.get("text_sha256"), r.get("page_count")))


VERSIONS = {"publishedVersion": "published", "acceptedVersion": "accepted", "submittedVersion": "submitted"}
ARCHIVE_SOURCE = "research_materials/anvur_2026-10-17 (git 87c4aaf)"


def doi_slug(doi):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in (doi or "").lower())


def archive_index(where):
    """{doi: row} of the PDFs the project already had (`fulltext_archive/archive_index.csv`): the 2026 papers of the ANVUR package of 17 Oct 2026, open-access copies collected then and
    kept in git (commit 87c4aaf), copied here with their version, licence and source URL. Nothing is downloaded by this module."""
    path = _dir(where, "fulltext_archive") / "archive_index.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8", newline="") as stream:
        return {r["doi"].lower(): r for r in csv.DictReader(stream)}


def manual_files(where):
    """{name without the suffix, lower case: path} of the files a person put in `fulltext_manual/`."""
    folder = _dir(where, "fulltext_manual")
    return {p.stem.lower(): p for p in sorted(folder.glob("*")) if p.suffix.lower() in DOCUMENT_SUFFIXES} if folder.is_dir() else {}


DOWNLOAD_COLUMNS = ("doi", "paper_id", "file", "found_by", "version", "oa_status", "license", "pdf_url", "landing_url", "sha256", "downloaded_at")


def download_index(where):
    """{doi: row} of the PDFs the download step found (`fulltext_download/download_index.csv`, one row per file, the newest of a DOI counts)."""
    path = _dir(where, "fulltext_download") / "download_index.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8", newline="") as stream:
        return {r["doi"].lower(): r for r in csv.DictReader(stream)}


def available_files(where, dois):
    """{paper_id: (path, origin, index row or None)} of the papers whose full text is already in the project. `dois` = {paper_id: doi}. A file put by hand (named after the paper id or
    the DOI, every character that is not a letter, a digit, '-', '.' or '_' as '_') wins over the archive: a person chose it, possibly a later version. The archive wins over a
    download (a download is only tried for a paper with neither)."""
    manual, index, downloads, out = manual_files(where), archive_index(where), download_index(where), {}
    folder, fetched = _dir(where, "fulltext_archive"), _dir(where, "fulltext_download")
    for pid, doi in dois.items():
        path = manual.get(pid.lower()) or manual.get(doi_slug(doi))
        if path is not None:
            out[pid] = (path, "manual_upload", None)
            continue
        row = index.get((doi or "").lower())
        if row and (folder / row["file"]).is_file():
            out[pid] = (folder / row["file"], "project_archive", row)
            continue
        row = downloads.get((doi or "").lower())
        if row and (fetched / row["file"]).is_file():
            out[pid] = (fetched / row["file"], "open_access_download", row)
    return out


def version_of(row):
    """The version of an index row as stored: the archive keeps the DRIVER words (submittedVersion), a download the short ones (submitted)."""
    value = (row or {}).get("version") or ""
    return VERSIONS.get(value) or (value if value in VERSIONS.values() else None)


def import_fulltexts(con, ids, where, *, gate=None, progress=None):
    """Step 2: copy into the evaluation the full texts the project already has for `ids` (the archive, and the files put by hand), and parse each. A file whose content was already
    imported for that paper is not imported again. -> counts. Nothing is downloaded."""
    dois = dict(con.execute("SELECT paper_id, doi FROM evaluation_paper WHERE paper_id = ANY(%s)", (list(ids),)).fetchall())
    files, latest = available_files(where, dois), latest_fulltext(con, ids)
    counts, problems = {"project_archive": 0, "manual_upload": 0, "open_access_download": 0, "unchanged": 0, "not_in_the_project": len(ids) - len(files)}, []
    for done, (pid, (path, origin, row)) in enumerate(sorted(files.items()), start=1):
        if gate:
            gate()
        data = path.read_bytes()
        if (latest.get(pid) or {}).get("sha256") == hashlib.sha256(data).hexdigest():
            counts["unchanged"] += 1
            continue
        try:
            stored = store_document(where, pid, data)
        except ValueError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        if origin == "project_archive":
            facts = {"found_by": f"{ARCHIVE_SOURCE}: {row['source'].split(':', 1)[-1]}", "detail": f"open-access copy collected on 17 Oct 2026 ({row['oa_status']})",
                     "oa_status": row["oa_status"] or None, "license": row["license"] or None, "version": VERSIONS.get(row["version"]), "pdf_url": row["download_url"] or None}
        elif origin == "open_access_download":
            facts = {"found_by": row["found_by"] or None, "detail": f"open-access copy downloaded on {row['downloaded_at'][:10]}", "oa_status": row["oa_status"] or None,
                     "license": row["license"] or None, "version": version_of(row), "pdf_url": row["pdf_url"] or None, "landing_url": row["landing_url"] or None}
        else:
            facts = {"found_by": f"fulltext_manual/{path.name}", "detail": "put in the folder by a person"}
        record_fulltext(con, pid, origin, {"outcome": "imported", **facts, **stored})
        con.commit()
        counts[origin] += 1
        if progress:
            progress(done, len(files), {})
    return {**counts, "problems": problems[:20]}


# --- step 2b: downloading the open-access PDFs the project does not have, and the papers left to fetch by hand -----------------------------------------

def latest_downloads(con, ids):
    """{paper_id: the newest download attempt}, whatever its outcome."""
    rows = con.execute("SELECT DISTINCT ON (paper_id) paper_id,outcome,detail,landing_url,created_at FROM evaluation_fulltext WHERE paper_id = ANY(%s) AND origin='open_access_download' "
                       "ORDER BY paper_id, created_at DESC", (list(ids),)).fetchall()
    return {r[0]: {"outcome": r[1], "detail": r[2], "landing_url": r[3], "created_at": r[4]} for r in rows}


def to_fetch(con, where, *, min_words=DEFAULT_MIN_WORDS, require_grade=True, seed=DEFAULT_SEED):
    """The papers that would be usable but have no full text, in the order of the sample's draw (the first ones are those a larger sample takes next; the order does not favour any
    journal or grade). -> (ids, inputs, display, {paper_id: newest download attempt})."""
    inputs, display = read_inputs(where["directory"]), read_display(where["directory"])
    own = already_in_corpus(inputs, display, scoring_corpora.AbstractCorpus(where["novelty"]).entries) if (Path(where["novelty"]) / "corpus.json").is_file() else set()
    pool = eligible(inputs, display, min_words=int(min_words), require_grade=bool(require_grade), exclude=own)
    have = available_files(where, {p: (display.get(p) or {}).get("doi", "") for p in pool})
    ids = [p for p in draw(pool, len(pool), int(seed)) if p not in have]
    return ids, inputs, display, latest_downloads(con, ids) if cards_tables_exist(con) else {}


def download_todo(con, where, *, limit=None, **options):
    """The papers the download step tries: no full text, a DOI, and no final answer yet (never tried, or a failure that is worth trying again, as `fulltext.RETRY`)."""
    from . import fulltext
    ids, inputs, display, tried = to_fetch(con, where, **options)
    todo = [p for p in ids if (display.get(p) or {}).get("doi") and (p not in tried or tried[p]["outcome"] in fulltext.RETRY)]
    return (todo[:int(limit)] if limit else todo), inputs, display


def _append_download(where, row):
    path = _dir(where, "fulltext_download") / "download_index.csv"
    new = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DOWNLOAD_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) or "" for k in DOWNLOAD_COLUMNS})


def download_fulltexts(con, ids, where, email, *, get=None, workers=None, gate=None, progress=None):
    """Step 2b: try `ids` with the resolver of the corpus (`fulltext.acquire`) and, for each PDF found, keep it in `fulltext_download/`, import it and parse it, as step 2 does with the
    archive. Every attempt is a row of `evaluation_fulltext` (origin open_access_download), the failed ones with the resolver's reason. Stop ends it after the
    papers in progress; twenty lookups failed in a row end it too (the network or the services, not the papers). -> {"outcomes": {outcome: papers}, "stopped"}."""
    from collections import Counter
    from . import fulltext
    if not email or "@" not in email:
        raise ValueError("Set UNPAYWALL_EMAIL in .env (a real address; Unpaywall asks for it) and restart the server")
    get = get or fulltext.http_get
    dois = dict(con.execute("SELECT paper_id, doi FROM evaluation_paper WHERE paper_id = ANY(%s)", (list(ids),)).fetchall())
    keys = (os.environ.get("OPENALEX_API_KEY", "").strip() or None, os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None)
    stop, counts, streak = threading.Event(), Counter(), 0

    def work(pid):
        return None if stop.is_set() else (pid, fulltext.acquire(dois.get(pid), email, get, *keys))
    folder = _dir(where, "fulltext_download")
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers or fulltext.WORKERS) as pool:
        try:
            for item in pool.map(work, ids):
                if item is None:
                    continue
                pid, r = item
                data, outcome = r.pop("data", None), r["outcome"]
                facts = {"found_by": r.get("found_by"), "oa_status": r.get("oa_status"), "license": r.get("license"), "version": r.get("version"), "pdf_url": r.get("pdf_url"),
                         "landing_url": r.get("landing_url")}
                if data:
                    try:
                        stored = store_document(where, pid, data)
                    except ValueError as exc:
                        record_fulltext(con, pid, "open_access_download", {"outcome": "not_a_pdf", "detail": str(exc), **facts})
                        outcome = "not_a_pdf"
                    else:
                        name = f"{doi_slug(dois[pid])}.pdf"
                        (folder / name).write_bytes(data)
                        _append_download(where, {"doi": dois[pid], "paper_id": pid, "file": name, "sha256": stored["sha256"], "downloaded_at": _now(), **facts})
                        record_fulltext(con, pid, "open_access_download", {"outcome": "imported", "detail": f"open-access copy downloaded by {fulltext.RESOLVER_VERSION}", **facts, **stored})
                        outcome = "imported"
                else:
                    record_fulltext(con, pid, "open_access_download", {"outcome": outcome, "detail": f"{fulltext.REASONS.get(outcome, outcome)}" + (f": {r['detail']}" if r.get("detail") else ""), **facts})
                con.commit()
                counts[outcome] += 1
                streak = streak + 1 if outcome == "lookup_failed" else 0
                if progress:
                    progress(sum(counts.values()), len(ids), dict(counts))
                if streak >= fulltext.ABORT_AFTER_LOOKUP_FAILURES:
                    raise RuntimeError(f"{streak} lookups failed in a row ({r.get('detail')}); check UNPAYWALL_EMAIL, SEMANTIC_SCHOLAR_API_KEY and the network")
                if gate:
                    gate()
        except Cancelled:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            return {"outcomes": dict(counts), "stopped": True}
        except BaseException:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return {"outcomes": dict(counts), "stopped": False}


OUTCOME_WORDS = {"imported": "downloaded", "lookup_failed": "the services could not be asked (tried again next time)", "download_failed": "the download failed (tried again next time)"}


def _attempt_words(attempt):
    from . import fulltext
    if not attempt:
        return "not tried yet"
    return OUTCOME_WORDS.get(attempt["outcome"]) or fulltext.REASONS.get(attempt["outcome"], attempt["outcome"])


def read_titles(directory):
    """{paper_id: title} of every paper of the set, from `raw/` (the papers without an abstract too). For the page and the files only, never for a judge."""
    path = Path(directory) / "raw" / "abstracts_2026_full.jsonl"
    return {r["paper_id"]: r.get("title") or "" for r in scoring_io.read_jsonl(path)} if path.is_file() else {}


# --- the reconciliation: from the set of 2026 papers to the papers judged, stage by stage -------------------------------------------------------------

STAGES = (("abstract", "With an abstract", "no abstract in Crossref or OpenAlex: the novelty judge has nothing to read"),
          ("words", "Abstract long enough", "abstract under the minimum words (a column's lead-in, not an abstract)"),
          ("not_in_corpus", "Not already in the corpus", "already in the CTO/TIM corpus: it would be its own closest neighbour"),
          ("grade", "Journal with an ABS 2024 grade", "journal without an ABS 2024 grade (only when the sample asks for one)"),
          ("fulltext", "Full text in the project", "no full text: neither in the archive, nor downloaded, nor put by hand"),
          ("parsed", "Full text imported and parsed", "full text in the project but not imported yet, or not readable"),
          ("cards", "Both cards valid", "methodology or impact card not made, or refused"),
          ("judged", "Judged on the three criteria", "never judged by a run on three criteria (test runs excluded)"))


def _judged(con):
    """{paper_id: {"full": best status over the runs on three criteria, "novelty": the same over the runs on novelty only, "runs": [names]}}, test runs excluded. The status of a paper in
    a run is its newest score there; across runs, ok beats partial beats failed."""
    out, best = {}, {"ok": 3, "partial": 2, "failed": 1}
    if not tables_exist(con):
        return out
    for pid, kind, status, run_id, label in con.execute(
            "SELECT DISTINCT ON (s.paper_id, s.run_id) s.paper_id, r.kind, s.status, r.run_id, r.label FROM evaluation_score s JOIN evaluation_run r USING (run_id) "
            "WHERE NOT r.synthetic ORDER BY s.paper_id, s.run_id, s.created_at DESC").fetchall():
        p = out.setdefault(pid, {"full": None, "novelty": None, "runs": []})
        if best.get(status, 0) > best.get(p[kind], 0):
            p[kind] = status
        p["runs"].append(label or f"Run {str(run_id)[:8]}")
    return out


def reconcile(con, where=None, *, min_words=DEFAULT_MIN_WORDS, require_grade=True, seed=DEFAULT_SEED, active=None):
    """Every paper of the 2026 set, the first stage it does not pass and why, and the counts of the funnel; the files of the folders that match no paper; the runs and what they judged.
    It reads, it changes nothing. The stages follow the order of the pipeline: a paper counts at a stage only if it passed every stage before."""
    where = where or settings()
    inputs, display, titles = read_inputs(where["directory"]), read_display(where["directory"]), read_titles(where["directory"])
    corpus_present = (Path(where["novelty"]) / "corpus.json").is_file()
    own = already_in_corpus(inputs, display, scoring_corpora.AbstractCorpus(where["novelty"]).entries) if corpus_present else set()
    dois = {p: (d.get("doi") or "") for p, d in display.items()}
    have = available_files(where, dois)
    db = cards_tables_exist(con)
    files = latest_fulltext(con, list(display)) if db else {}
    tried = latest_downloads(con, list(display)) if db else {}
    cards = latest_cards(con, list(display)) if db else {}
    judged = _judged(con)
    rank = {p: i + 1 for i, p in enumerate(draw(list(display), len(display), int(seed)))}
    rows = []
    for pid, d in display.items():
        f, j = files.get(pid), judged.get(pid) or {}
        parsed = parsed_path(where, pid).is_file() and bool(f and f["outcome"] == "imported")
        both = all((cards.get((pid, k)) or {}).get("status") == "valid" for k in CARD_KINDS)
        passes = {"abstract": pid in inputs, "words": pid in inputs and words(inputs[pid]["abstract"]) >= int(min_words), "not_in_corpus": pid not in own,
                  "grade": bool(d.get("ajg2024")) or not require_grade, "fulltext": pid in have, "parsed": parsed, "cards": both, "judged": j.get("full") in ("ok", "partial")}
        stop = next((key for key, _, _ in STAGES if not passes[key]), None)
        source = have.get(pid)
        rows.append({"paper_id": pid, "rank": rank[pid], "title": (inputs.get(pid) or {}).get("title") or titles.get(pid, ""), "doi": d.get("doi", ""), "journal": d.get("journal", ""),
                     "ajg2024": d.get("ajg2024", ""), "online_date": d.get("online_date", ""), "source_sets": d.get("source_sets", ""),
                     "words": words(inputs[pid]["abstract"]) if pid in inputs else 0, "stops_at": stop,
                     "fulltext": source[1] if source else None, "version": version_of(source[2]) if source and source[2] else None,
                     "imported": bool(f and f["outcome"] == "imported"), "parse_status": f["parse_status"] if f else None,
                     "cards": {k: (cards.get((pid, k)) or {}).get("status") for k in CARD_KINDS}, "judged_full": j.get("full"), "judged_novelty": j.get("novelty"), "runs": j.get("runs", []),
                     "download": None if pid not in tried else {"outcome": tried[pid]["outcome"], "words": _attempt_words(tried[pid]), "landing_url": tried[pid]["landing_url"],
                                                               "at": tried[pid]["created_at"].isoformat(timespec="seconds")}})
    rows.sort(key=lambda r: r["rank"])
    order = [key for key, _, _ in STAGES]
    funnel, reached = [], len(rows)
    for key, label, why in STAGES:
        out = sum(1 for r in rows if r["stops_at"] == key)
        reached -= out
        funnel.append({"stage": key, "label": label, "papers": reached, "left_out": out, "why": why, "skipped": key == "grade" and not require_grade})
    # the same funnel by ABS grade: how many papers of each grade have a full text, and how far they go
    past = lambda r, key: r["stops_at"] is None or order.index(r["stops_at"]) > order.index(key)
    grades = sorted({r["ajg2024"] for r in rows if r["ajg2024"]}, key=lambda g: (-GRADES.index(g) if g in GRADES else 1, g))    # highest first, an unknown grade last
    by_grade = []
    for grade in grades + [""]:
        of = [r for r in rows if r["ajg2024"] == grade]
        if of:
            by_grade.append({"grade": grade or "not graded", "papers": len(of), "with_fulltext": sum(1 for r in of if r["fulltext"]),
                             **{name: sum(1 for r in of if past(r, key)) for name, key in (("usable", "grade"), ("usable_with_fulltext", "fulltext"), ("parsed", "parsed"), ("cards", "cards"))},
                             "judged": sum(1 for r in of if r["stops_at"] is None)})
    # the files of the folders, against the papers of the set
    known = {doi_slug(x) for x in dois.values() if x} | {p.lower() for p in display}
    index = archive_index(where)
    archive_files = sorted(p.name for p in _dir(where, "fulltext_archive").glob("*.pdf")) if _dir(where, "fulltext_archive").is_dir() else []
    set_dois = {x.lower() for x in dois.values() if x}
    by_doi = {x.lower(): p for p, x in dois.items() if x}
    usable_ids = {r["paper_id"] for r in rows if r["stops_at"] not in order[:order.index("fulltext") + 1]}
    archive = {"files": len(archive_files), "indexed": len(index), "not_indexed": sorted(set(archive_files) - {r["file"] for r in index.values()}),
               "no_paper": sorted(r["file"] for doi, r in index.items() if doi not in set_dois),
               "by_stage": {}}
    stops = {r["paper_id"]: r["stops_at"] for r in rows}
    for doi in index:
        if doi in by_doi:
            stage = stops[by_doi[doi]]
            key = stage if stage in order[:order.index("fulltext")] else "usable"
            archive["by_stage"][key] = archive["by_stage"].get(key, 0) + 1
    manual = manual_files(where)
    folders = {"archive": archive, "manual": {"files": len(manual), "not_recognised": sorted(p.name for stem, p in manual.items() if stem not in known), "folder": str(_dir(where, "fulltext_manual"))},
               "download": {"files": len(download_index(where))}}
    runs = []
    for r in (list_runs(con, active)["runs"] if tables_exist(con) else []):
        if r["synthetic"]:
            continue
        ids = con.execute("SELECT parameters->'paper_ids' FROM evaluation_run WHERE run_id=%s", (uuid.UUID(r["run_id"]),)).fetchone()[0]
        runs.append({**{k: r[k] for k in ("run_id", "label", "kind", "status", "papers", "scored", "complete", "partial", "failed", "created_at")},
                     "not_usable_now": sum(1 for p in ids if p not in usable_ids)})
    count = lambda pred: sum(1 for r in rows if pred(r))
    todo = {"to_fetch": count(lambda r: r["stops_at"] == "fulltext"),
            "to_fetch_never_tried": count(lambda r: r["stops_at"] == "fulltext" and not r["download"]),
            "to_fetch_by_hand": count(lambda r: r["stops_at"] == "fulltext" and r["download"] and r["download"]["outcome"] not in ("lookup_failed", "download_failed")),
            "to_import": count(lambda r: r["stops_at"] == "parsed" and not r["imported"]), "unreadable": count(lambda r: r["stops_at"] == "parsed" and r["imported"]),
            "to_card": count(lambda r: r["stops_at"] == "cards"), "to_judge": count(lambda r: r["stops_at"] == "judged"),
            "judged_novelty_only": count(lambda r: r["judged_novelty"] in ("ok", "partial") and r["judged_full"] not in ("ok", "partial"))}
    return {"filters": {"min_words": int(min_words), "require_grade": bool(require_grade), "seed": int(seed)}, "total": len(rows), "funnel": funnel, "by_grade": by_grade, "folders": folders, "runs": runs,
            "todo": todo, "rows": rows, "corpus_present": corpus_present, "migration_applied": db, "download_migration_applied": db and download_allowed(con),
            "email_configured": "@" in os.environ.get("UNPAYWALL_EMAIL", ""), "semantic_scholar": bool(os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip())}


GRADE_COLUMNS = (("grade", "ABS 2024 grade"), ("papers", "Papers of the set"), ("with_fulltext", "With a full text in the project"), ("usable", "Usable"),
                 ("usable_with_fulltext", "Usable with a full text"), ("parsed", "Parsed"), ("cards", "Both cards valid"), ("judged", "Judged on three criteria"))


def _xlsx_cell(sheet, row, col, value, font=None, link=None):
    """One cell. A text is always stored as text, so a title that starts with '=' cannot become a formula."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)[:32000]
    cell = sheet.cell(row=row, column=col, value=value)
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"
    if font is not None:
        cell.font = font
    if link:
        cell.hyperlink, cell.style = link, "Hyperlink"
    return cell


def export_to_fetch(con, where=None, *, min_words=DEFAULT_MIN_WORDS, require_grade=True, seed=DEFAULT_SEED, active=None):
    """The Excel file of the papers to fetch by hand: (file name, content type, bytes). Sheet 1 says what to do; sheet 2 lists the usable papers without a full text in the order of
    the sample's draw, with the outcome of the automatic attempt; sheet 3 is the reconciliation of every paper of the set. It reads `reconcile`, so the file and the page agree."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    where = where or settings()
    rec = reconcile(con, where, min_words=min_words, require_grade=require_grade, seed=seed, active=active)
    head, bold, plain = Font(bold=True, color="FFFFFF"), Font(bold=True), Font()
    fill = PatternFill("solid", fgColor="44546A")
    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    missing = [r for r in rec["rows"] if r["stops_at"] == "fulltext"]
    by_hand = [r for r in missing if r["download"] and r["download"]["outcome"] not in ("lookup_failed", "download_failed")]
    lines = [("Evaluation benchmark (step 6): papers whose full text must be fetched by hand", bold),
             (f"Exported {_now()} · filters: abstract of at least {rec['filters']['min_words']} words" + (", journal with an ABS 2024 grade" if rec["filters"]["require_grade"] else "") +
              f", seed {rec['filters']['seed']}.", plain),
             (f"{len(missing)} usable papers have no full text: {len(by_hand)} were tried automatically (Unpaywall, OpenAlex, Semantic Scholar) without success, "
              f"{rec['todo']['to_fetch_never_tried']} were never tried (press the download button on the page first, it may find them).", plain),
             ("How to add one: download the PDF (or .docx) from the DOI link with your own access, name it as in the column 'File name to use' and put it in:", plain),
             (rec["folders"]["manual"]["folder"], bold),
             ("then press 'Import and parse the full texts' on the page. A file put by hand wins over every other copy.", plain),
             ("Rank = the order of the sample's draw: the first rows are the papers a larger sample takes next. It does not follow the journal or the grade.", plain),
             ("", plain), ("Reconciliation, stage by stage", bold)]
    for i, (text, font) in enumerate(lines, 1):
        _xlsx_cell(readme, i, 1, text, font)
    top = len(lines) + 1
    for j, h in enumerate(("Stage", "Papers that reach it", "Left out here", "Why they are left out"), 1):
        _xlsx_cell(readme, top, j, h, bold)
    _xlsx_cell(readme, top + 1, 1, "Papers of the 2026 set", plain)
    _xlsx_cell(readme, top + 1, 2, rec["total"], plain)
    for i, f in enumerate(rec["funnel"], top + 2):
        _xlsx_cell(readme, i, 1, f["label"] + (" (not asked)" if f["skipped"] else ""), plain)
        _xlsx_cell(readme, i, 2, f["papers"], plain)
        _xlsx_cell(readme, i, 3, f["left_out"], plain)
        _xlsx_cell(readme, i, 4, f["why"], plain)
    top += len(rec["funnel"]) + 3
    _xlsx_cell(readme, top, 1, "By ABS 2024 grade", bold)
    for j, h in enumerate(GRADE_COLUMNS, 1):
        _xlsx_cell(readme, top + 1, j, h[1], bold)
    for i, g in enumerate(rec["by_grade"], top + 2):
        for j, (key, _) in enumerate(GRADE_COLUMNS, 1):
            _xlsx_cell(readme, i, j, g[key], plain)
    readme.column_dimensions["A"].width, readme.column_dimensions["B"].width, readme.column_dimensions["C"].width, readme.column_dimensions["D"].width = 40, 20, 14, 80

    def table(title, columns, data):
        sheet = wb.create_sheet(title)
        for col, (name, width, _) in enumerate(columns, 1):
            c = _xlsx_cell(sheet, 1, col, name, head)
            c.fill, c.alignment = fill, Alignment(wrap_text=True, vertical="top")
            sheet.column_dimensions[get_column_letter(col)].width = width
        for r, item in enumerate(data, 2):
            for col, (_, _, get) in enumerate(columns, 1):
                value, link = get(item)
                _xlsx_cell(sheet, r, col, value, None, link)
        sheet.freeze_panes = "C2"
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(data) + 1}"
        return sheet
    doi_url = lambda r: "https://doi.org/" + r["doi"] if r["doi"] else None
    plain_value = lambda key: (lambda r: (r[key] or None, None))
    page = lambda r: ((r["download"] or {}).get("landing_url") or None, (r["download"] or {}).get("landing_url") or None)
    fetch_columns = [("Rank", 7, plain_value("rank")), ("Paper id", 18, plain_value("paper_id")), ("File name to use", 22, lambda r: (f"{r['paper_id']}.pdf", None)),
                     ("DOI", 26, lambda r: (r["doi"] or None, doi_url(r))), ("Title", 60, plain_value("title")), ("Journal", 34, plain_value("journal")), ("ABS 2024", 9, plain_value("ajg2024")),
                     ("Online", 11, plain_value("online_date")), ("Automatic download", 46, lambda r: (_attempt_words(r["download"]), None)),
                     ("Tried on", 11, lambda r: ((r["download"] or {}).get("at", "")[:10] or None, None)), ("Open-access page found", 40, page)]
    table("To fetch by hand", fetch_columns, by_hand + [r for r in missing if r not in by_hand])
    stage_label = {key: label for key, label, _ in STAGES}
    all_columns = [("Rank", 7, plain_value("rank")), ("Paper id", 18, plain_value("paper_id")), ("DOI", 26, lambda r: (r["doi"] or None, doi_url(r))), ("Title", 60, plain_value("title")),
                   ("Journal", 34, plain_value("journal")), ("ABS 2024", 9, plain_value("ajg2024")), ("Abstract words", 9, plain_value("words")),
                   ("Stops at", 30, lambda r: ("in the benchmark, judged" if r["stops_at"] is None else "not passed: " + stage_label[r["stops_at"]], None)),
                   ("Full text from", 18, lambda r: ({"project_archive": "archive", "manual_upload": "put by hand", "open_access_download": "downloaded"}.get(r["fulltext"]), None)),
                   ("Version", 11, plain_value("version")), ("Parsed", 9, lambda r: (r["parse_status"], None)),
                   ("Methodology card", 12, lambda r: (r["cards"]["method"], None)), ("Impact card", 12, lambda r: (r["cards"]["impact"], None)),
                   ("Judged (three criteria)", 12, plain_value("judged_full")), ("Judged (novelty only)", 12, plain_value("judged_novelty")),
                   ("Runs", 30, lambda r: ("; ".join(r["runs"]) or None, None)), ("Automatic download", 40, lambda r: (_attempt_words(r["download"]) if r["download"] or r["stops_at"] == "fulltext" else None, None))]
    table("All papers", all_columns, rec["rows"])
    out = io.BytesIO()
    wb.save(out)
    name = f"evaluation_pdfs_to_fetch_{dt.date.today().isoformat()}.xlsx"
    return name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", out.getvalue()


# --- step 4: the cards ----------------------------------------------------------------------------------------------------------------------------

def card_source(text, kind):
    from .paper_processing import SOURCES                # the same section selection as the cards of the corpus papers
    return SOURCES[kind](text or "")


def card_prompt(pid, source):
    """The message of a card request, in the format of the Batch that made the corpus cards; the id is opaque."""
    return f"INPUT_ID: {scoring_io.opaque_id(SALT + '-card', pid)}\n\n<<<TESTO\n{source}\n>>>FINE_TESTO\n"


VENUE_WORDS = r"(?:journal|published in|publish in|appeared in|appears in|appear in|forthcoming in|accepted in|accepted at|accepted for publication in|submitted to|special issue of|issue of)"


def journal_matcher(journal):
    """-> a function telling whether a text names the journal, or None when the name cannot be looked for. A name of several words is looked for as a whole; a name of ONE word
    (Innovation, Organization, Technovation) only where it is used as a venue ("published in Organization", "the journal Innovation", "Innovation journal"): as a plain word it is the
    paper's own vocabulary, and looking for it everywhere emptied every card of a paper about innovations (20 Oct 2026)."""
    # Display metadata can contain HTML entities ("&amp;") while a card quotes
    # the decoded journal title ("&"). Compare the same decoded form on both
    # sides, and treat "&" and "and" as equivalent in venue names.
    key = lambda value: scoring_io.title_key(html.unescape(value or "").replace("&", " and "))
    name = key(journal)
    if len(name) < 8:
        return None
    word = re.escape(name)
    pattern = re.compile(rf"\b{word}\b" if " " in name else rf"\b{VENUE_WORDS}(?: the)?(?: journal)? {word}\b|\b{word} journal\b")
    return lambda text: bool(pattern.search(key(text)))


def guard_card(card, journal, profile):
    """A field whose text or quotation names the paper's own journal becomes "not stated": the judge must not learn where the paper was published. -> (card, fields turned off)."""
    names = journal_matcher(journal)
    if names is None or card is None:
        return card, []
    off = []
    for f in profile.fields:
        c = card[f]
        if c["status"] != "missing" and any(names(t) for t in [c.get("text") or "", *c["passages"]]):
            card[f] = {"text": None, "status": "missing", "passages": []}
            off.append(f)
    if off:
        card["info_gaps"] = [g for g in card.get("info_gaps", []) if not names(g)] + [f"{f}: removed because it named the publication venue" for f in off]
    return card, off


def check_card(answer, source, journal, profile):
    """The check of the corpus cards (`paper_cards.repair_card`), then the journal guard; a card left with less than half of its fields is refused, as there. -> (card, off, problems)."""
    card, demoted, problems = paper_cards.repair_card(answer, source, profile)
    if problems:
        return None, demoted, problems
    card, off = guard_card(card, journal, profile)
    kept = sum(1 for f in profile.fields if card[f]["status"] != "missing")
    needed = math.ceil(len(profile.fields) * paper_cards.MIN_VERIFIED_SHARE)
    if kept < needed:
        return None, demoted + off, [f"only {kept} of {len(profile.fields)} fields are left once the fields naming the journal are removed"]
    return card, demoted + off, []


def latest_cards(con, ids):
    """{(paper_id, kind): newest card row of the current profile version, as a dict}."""
    out = {}
    for kind in CARD_KINDS:
        for pid, status, card, model, source_sha, info, rejection, at in con.execute(
                "SELECT DISTINCT ON (paper_id) paper_id,status,card,model,source_sha256,source_info,rejection,created_at FROM evaluation_card "
                "WHERE paper_id = ANY(%s) AND kind=%s AND profile_version=%s ORDER BY paper_id, created_at DESC", (list(ids), kind, paper_cards.PROFILES[kind].version)).fetchall():
            out[(pid, kind)] = {"status": status, "card": card, "model": model, "source_sha256": source_sha, "source_info": info, "rejection": rejection, "created_at": at}
    return out


def cards_todo(con, ids, where):
    """[(paper_id, kind, source, info)] of the cards still to make: a parsed paper with no valid card of the current profile for the same text. A refused card is asked again by a later
    click (the register keeps every attempt and stops a paper after three failures)."""
    have, todo, short = latest_cards(con, ids), [], []
    for pid in ids:
        text = parsed_text(where, pid)
        if not text:
            continue
        for kind in CARD_KINDS:
            source, info = card_source(text, kind)
            if len(source) < MIN_SOURCE_CHARS:
                short.append((pid, kind))
                continue
            row = have.get((pid, kind))
            if row and row["status"] == "valid" and row["source_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest():
                continue
            todo.append((pid, kind, source, info))
    return todo, short


def estimate_cards(todo, cfg):
    """Input tokens (characters / 3, deliberately high, as the judges' limits) and the output allowance of the cards still to make."""
    tokens = sum((len(paper_cards.PROFILES[k].card_system) + len(card_prompt(p, s))) // 3 + 1 for p, k, s, _ in todo)
    return {"cards": len(todo), "input_tokens": tokens, "max_output_tokens": len(todo) * cfg["judges"][0]["params"]["max_output_tokens"]}


def make_cards(con, ids, where, *, provider=None, synthetic_provider=None, max_tokens=None, gate=None, progress=None):
    """Step 4: make the missing methodology and impact cards of `ids`, as the corpus cards were made, and store each (valid, or refused with the reason). -> counts. A limit, a refusal of
    the provider or Stop ends the step; what is stored stays."""
    cfg = scoring_judges.load_config(where["cards_config"])
    if max_tokens:
        cfg = {**cfg, "budget": {**cfg["budget"], "max_tokens": int(max_tokens)}}
    if synthetic_provider is not None:
        cfg = {**cfg, "run_live": False, "budget": {k: 0 for k in cfg["budget"]}, "judges": [{**j, "provider": "synthetic"} for j in cfg["judges"]]}
        providers = {r: synthetic_provider for r in scoring_judges.JUDGE_ROLES}
    else:
        if getattr(provider, "name", None) != "openai":
            raise ValueError("the cards are made on the OpenAI provider")
        providers = {r: provider for r in scoring_judges.JUDGE_ROLES}
    todo, short = cards_todo(con, ids, where)
    held = cards_in_batch(where)
    todo = [t for t in todo if (t[0], t[1]) not in held]                    # already in a Batch: it will be paid for there
    journals = dict(con.execute("SELECT paper_id, journal FROM evaluation_paper WHERE paper_id = ANY(%s)", (list(ids),)).fetchall())
    session ="evc_" + scoring_io.sha256_obj([VERSION, sorted(ids), dt.datetime.now(dt.timezone.utc).isoformat()])[:12]
    judges = scoring_judges.Judges(where["workspace"], cfg, providers, session_id=session, gate=gate)
    model = cfg["judges"][0]["model"]["id"]
    counts, lock, done = {"valid": 0, "rejected": 0, "too_short": len(short)}, threading.Lock(), [0]

    def one(item):
        pid, kind, source, info = item
        profile = paper_cards.PROFILES[kind]
        journal = journals.get(pid, "")
        try:
            obj, _ = judges.ask("judge_1", task=f"evaluation_{kind}_card", system=profile.card_system, prompt=card_prompt(pid, source), schema_name=profile.card_task,
                                schema=paper_cards.card_schema(profile.fields), validate=lambda o: check_card(o, source, journal, profile)[2],
                                key_parts=[VERSION, "card", kind, profile.version])
            card, changed, problems = check_card(obj, source, journal, profile)
        except scoring_judges.JudgeError as exc:
            card, changed, problems = None, [], [str(exc)]
        return pid, kind, source, info, card, changed, problems

    def store(result):
        pid, kind, source, info, card, changed, problems = result
        with lock:
            con.execute("INSERT INTO evaluation_card(card_id,paper_id,kind,profile_version,model,source_sha256,source_info,status,card,rejection) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (uuid.uuid4(), pid, kind, paper_cards.PROFILES[kind].version, model, hashlib.sha256(source.encode("utf-8")).hexdigest(),
                         _jsonb({**{k: info[k] for k in ("kind", "total_chars", "used_chars", "truncated") if k in info}, "demoted_fields": changed}),
                         "valid" if card else "rejected", _jsonb(card) if card else None, None if card else "; ".join(problems)[:1000]))
            con.commit()
            counts["valid" if card else "rejected"] += 1
            done[0] += 1
            if progress:
                progress(done[0], len(todo), judges.budget.used())
    state, note = "completed", ""
    try:
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            futures = [pool.submit(one, item) for item in todo]
            first_error = None
            for f in futures:
                try:
                    store(f.result())
                except (BudgetExceeded, Cancelled) as exc:
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error
    except (BudgetExceeded, Cancelled) as exc:
        state, note = "stopped", str(exc)[:300]
    return {"state": state, "note": note, "asked": len(todo), **counts, "usage": judges.budget.used(), "model": model}


# --- step 4 by Batch (20 Oct 2026, the user's request): the same requests, half the price, the answers within 24 hours -----------------------------------
# A click writes the requests of the cards still to make into `<workspace>/card_batches/<job>/` (requests.jsonl, manifest.json), checks the ceiling and submits them (batch.json); a
# later click asks OpenAI for the state and, once the Batch has ended, checks every answer as the synchronous step does (`check_card`) and stores it (collected.json). A card that is in
# a Batch not yet collected is sent by neither path. A request the Batch could not answer (an API error, an expired Batch) leaves no row: the next click sends it again.

BATCH_END_STATES = ("completed", "failed", "expired", "cancelled")
MAX_CARD_REFUSALS = 3                                 # as the register of the synchronous calls stops a card after three failures: a card refused three times is not sent again


def card_batch_root(where):
    return Path(where["workspace"]) / "card_batches"


def card_custom_id(pid, kind):
    return f"{kind}-{scoring_io.opaque_id(SALT + '-card', pid)}"                     # the id of the prompt: opaque, the paper id stays here


def card_batch_request(pid, kind, source, cfg):
    """One line of the Batch: the body of the synchronous call (`forum_providers` OpenAI provider for `make_cards`), field for field."""
    profile, judge = paper_cards.PROFILES[kind], cfg["judges"][0]
    params = judge["params"]
    body = {"model": judge["model"]["id"], "instructions": profile.card_system, "input": [{"role": "user", "content": card_prompt(pid, source)}],
            "max_output_tokens": params["max_output_tokens"], "store": False,
            "text": {"format": {"type": "json_schema", "name": profile.card_task, "strict": True, "schema": scoring_io.provider_schema(paper_cards.card_schema(profile.fields))}}}
    if params.get("reasoning_effort"):
        body["reasoning"] = {"effort": params["reasoning_effort"]}
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    return {"custom_id": card_custom_id(pid, kind), "method": "POST", "url": "/v1/responses", "body": body}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def card_batch_dirs(where):
    root = card_batch_root(where)
    return sorted(p.parent for p in root.glob("*/manifest.json")) if root.is_dir() else []


def cards_in_batch(where):
    """{(paper_id, kind)} of the cards in a submitted Batch not yet collected: neither path sends them again. A directory whose submission failed before upload has no batch.json and
    holds nothing back."""
    out = set()
    for d in card_batch_dirs(where):
        if (d / "batch.json").exists() and not (d / "collected.json").exists():
            out |= {(e["paper_id"], e["kind"]) for e in _read(d / "manifest.json")["requests"].values()}
    return out


def card_refusals(con, ids):
    """{(paper_id, kind, source_sha256): refused cards of the current profile versions}."""
    out = {}
    for kind in CARD_KINDS:
        for pid, sha, n in con.execute("SELECT paper_id,source_sha256,count(*) FROM evaluation_card WHERE paper_id = ANY(%s) AND kind=%s AND profile_version=%s AND status='rejected' "
                                       "GROUP BY 1,2", (list(ids), kind, paper_cards.PROFILES[kind].version)).fetchall():
            out[(pid, kind, sha)] = n
    return out


def cards_for_batch(con, ids, where):
    """-> (todo, held, stopped): the cards a Batch would send, those already in a Batch, and those refused MAX_CARD_REFUSALS times for the same text."""
    todo, _ = cards_todo(con, ids, where)
    busy, refused = cards_in_batch(where), card_refusals(con, ids)
    send, held, stopped = [], [], []
    for item in todo:
        pid, kind, source, _ = item
        if (pid, kind) in busy:
            held.append((pid, kind))
        elif refused.get((pid, kind, hashlib.sha256(source.encode("utf-8")).hexdigest()), 0) >= MAX_CARD_REFUSALS:
            stopped.append((pid, kind))
        else:
            send.append(item)
    return send, held, stopped


def submit_card_batch(con, ids, where, *, max_tokens, submit=None):
    """Write the requests of the cards still to make and submit them as one Batch, only if the estimate (input tokens plus the whole output allowance) is within `max_tokens`. Nothing
    is sent before that check. -> {"job", "batch_id", "cards", ...}. ValueError says why nothing was sent."""
    cfg = scoring_judges.load_config(where["cards_config"])
    send, held, stopped = cards_for_batch(con, ids, where)
    if not send:
        raise ValueError("no card to send: every parsed paper of the sample has its two cards" + (f", {len(held)} are in a Batch not yet collected" if held else "") +
                         (f", {len(stopped)} were refused {MAX_CARD_REFUSALS} times" if stopped else ""))
    est = estimate_cards(send, cfg)
    total = est["input_tokens"] + est["max_output_tokens"]
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("give the token ceiling as a whole number")
    if total > max_tokens:
        raise ValueError(f"the Batch of {len(send)} cards is estimated at up to {total:,} tokens ({est['input_tokens']:,} in, at most {est['max_output_tokens']:,} out), above the ceiling of "
                         f"{max_tokens:,}. Nothing was sent: raise the ceiling or lower the number of papers")
    job = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")              # microseconds: two clicks in the same second must not share a directory
    directory = card_batch_root(where) / job
    directory.mkdir(parents=True, exist_ok=False)
    lines, entries = [], {}
    for pid, kind, source, info in send:
        request = card_batch_request(pid, kind, source, cfg)
        lines.append(json.dumps(request, ensure_ascii=False))
        entries[request["custom_id"]] = {"paper_id": pid, "kind": kind, "profile_version": paper_cards.PROFILES[kind].version,
                                         "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                                         "source": {k: info[k] for k in ("kind", "total_chars", "used_chars", "truncated") if k in info}}
    (directory / "requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write(directory / "manifest.json", {"version": VERSION, "job": job, "model": cfg["judges"][0]["model"]["id"], "created_at": _now(), "request_count": len(lines),
                                         "estimate": est, "max_tokens": max_tokens, "requests": entries})
    if submit is None:
        from .community_phase1 import submit
    state = submit(directory)
    LOG.info(json.dumps({"event": "evaluation_card_batch_submitted", "job": job, "batch_id": state.get("batch_id"), "requests": len(lines)}))
    return {"job": job, "batch_id": state.get("batch_id"), "status": state.get("status"), "cards": len(lines), "held": len(held), "stopped": len(stopped), **est}


def store_card_answers(con, directory, where, output_text):
    """Check every answer of a Batch as the synchronous step checks it (the text it was made from must still have the same hash; `check_card`) and store it, valid or refused with the
    reason. Idempotent: a card row's id is fixed by the Batch and the request. -> counts."""
    from .community_phase1 import extract_output_text
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    batch_id = _read(directory / "batch.json").get("batch_id") if (directory / "batch.json").exists() else None
    pids = sorted({e["paper_id"] for e in manifest["requests"].values()})
    journals = dict(con.execute("SELECT paper_id, journal FROM evaluation_paper WHERE paper_id = ANY(%s)", (pids,)).fetchall())
    seen, counts = set(), {"valid": 0, "rejected": 0, "failed_requests": 0}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cid = item.get("custom_id")
        entry = manifest["requests"].get(cid)
        if entry is None or cid in seen:
            raise ValueError(f"unexpected or repeated custom_id in the Batch output: {cid}")
        seen.add(cid)
        response = item.get("response") or {}
        if item.get("error") or response.get("status_code") != 200:
            counts["failed_requests"] += 1
            continue
        pid, kind, body = entry["paper_id"], entry["kind"], response.get("body") or {}
        profile = paper_cards.PROFILES[kind]
        source, info = card_source(parsed_text(where, pid), kind)
        card, changed, problems = None, [], []
        if profile.version != entry["profile_version"]:
            problems = [f"the card profile changed after the request was written ({entry['profile_version']} -> {profile.version})"]
        elif hashlib.sha256(source.encode("utf-8")).hexdigest() != entry["source_sha256"]:
            problems = ["the paper's text changed after the request was written"]
        elif body.get("status") == "incomplete":
            problems = ["the answer was cut off at max_output_tokens"]
        else:
            try:
                obj, why = scoring_io.parse_json(extract_output_text(body))
                card, changed, problems = (None, [], [why]) if why else check_card(obj, source, journals.get(pid, ""), profile)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                card, changed, problems = None, [], [f"the answer could not be read: {str(exc)[:120]}"]
        con.execute("INSERT INTO evaluation_card(card_id,paper_id,kind,profile_version,model,source_sha256,source_info,status,card,rejection) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (card_id) DO NOTHING",
                    (uuid.uuid5(uuid.NAMESPACE_URL, f"{VERSION}/card-batch/{manifest['job']}/{cid}"), pid, kind, entry["profile_version"], manifest["model"],
                     entry["source_sha256"], _jsonb({**entry["source"], "demoted_fields": changed, "batch": {"job": manifest["job"], "batch_id": batch_id}}),
                     "valid" if card else "rejected", _jsonb(card) if card else None, None if card else "; ".join(problems)[:1000]))
        counts["valid" if card else "rejected"] += 1
    con.commit()
    return {**counts, "missing_requests": len(set(manifest["requests"]) - seen)}


def refresh_card_batches(con, where, client=None):
    """Ask OpenAI for the state of each Batch not yet collected; store the answers of those that ended and mark them collected (their unanswered cards can then be sent again). -> the
    list of `card_batches`."""
    for d in card_batch_dirs(where):
        if not (d / "batch.json").exists() or (d / "collected.json").exists():
            continue
        state = _read(d / "batch.json")
        if not state.get("batch_id"):
            continue                                   # uploaded but not created: a person must look before anything is sent again
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        batch = client.batches.retrieve(state["batch_id"])
        counts = batch.request_counts
        state.update(status=batch.status, output_file_id=batch.output_file_id, error_file_id=batch.error_file_id, checked_at=_now(),
                     request_counts={"total": counts.total, "completed": counts.completed, "failed": counts.failed} if counts else None)
        _write(d / "batch.json", state)
        if batch.status not in BATCH_END_STATES:
            continue
        result = {"status": batch.status, "valid": 0, "rejected": 0, "failed_requests": 0, "missing_requests": len(_read(d / "manifest.json")["requests"])}
        if batch.output_file_id:
            output = client.files.content(batch.output_file_id).read().decode("utf-8")
            (d / "batch_output.jsonl").write_text(output, encoding="utf-8")
            result.update(store_card_answers(con, d, where, output))
        if batch.error_file_id:
            (d / "batch_errors.jsonl").write_bytes(client.files.content(batch.error_file_id).read())
        _write(d / "collected.json", {**result, "collected_at": _now()})
        LOG.info(json.dumps({"event": "evaluation_card_batch_collected", "job": d.name, **result}))
    return card_batches(where)


def card_batches(where):
    """The Batches of cards, newest first, from their files (read-only)."""
    out = []
    for d in reversed(card_batch_dirs(where)):
        m = _read(d / "manifest.json")
        state = _read(d / "batch.json") if (d / "batch.json").exists() else {}
        out.append({"job": m["job"], "created_at": m.get("created_at"), "cards": m["request_count"], "estimate": m.get("estimate"), "batch_id": state.get("batch_id"),
                    "status": state.get("status") or "not submitted", "request_counts": state.get("request_counts"), "checked_at": state.get("checked_at"),
                    "collected": _read(d / "collected.json") if (d / "collected.json").exists() else None})
    return out


# --- step 5: the run -----------------------------------------------------------------------------------------------------------------------------------

def create_run(con, *, n=DEFAULT_N, seed=DEFAULT_SEED, min_words=DEFAULT_MIN_WORDS, require_grade=True, require_fulltext=True, max_tokens=DEFAULT_MAX_TOKENS, label="", criteria="novelty", where=None, synthetic=False, extend_run=None, stratify=False):
    """Check everything that can be checked without a model, draw the sample and record the run as `running`. `criteria`: "novelty" (two novelty votes) or "full" (novelty, methodology
    and impact, as for a proposal; a paper without a valid card gets no vote on that criterion). `extend_run`: the id of an earlier run whose papers all stay in the sample (a new run, the
    earlier one is not touched; what the judges already answered comes from the register, not from a new call). -> {"run_id", "papers", ...}. ValueError says what is wrong."""
    where = where or settings()
    if criteria not in KINDS:
        raise ValueError(f"criteria must be one of {KINDS}")
    if not Path(where["config"]).is_file():
        raise ValueError(f"configuration not found ({where['config']})")
    cfg = scoring_judges.load_config(where["config"])
    if not 100_000 <= int(max_tokens) <= 20_000_000:
        raise ValueError("the token ceiling must be between 100,000 and 20,000,000")
    if criteria == "full":
        for c, kind in fscore.CARD_OF.items():
            if not scoring_corpora.is_card_corpus(Path(where[c]), kind):
                raise ValueError(f"the {kind} card corpus is not at {where[c]} (research0 forum-score-corpus --kind {kind})")
    options = with_keep(con, {"n": n, "seed": seed, "min_words": min_words, "require_grade": require_grade, "require_fulltext": require_fulltext, "stratify": bool(stratify), "extend_run": extend_run}, where)
    keep = options.get("keep")
    s = sample(where, **options)
    ids, display = s["ids"], s["display"]
    dates = [dt.date.fromisoformat(display[i]["online_date"]) for i in ids if (display.get(i) or {}).get("online_date")]
    cutoff = check_cutoffs(cfg, min(dates) if dates else None)
    sync_papers(con, ids, s["inputs"], display)
    run_id = uuid.uuid4()
    cards = latest_cards(con, ids) if criteria == "full" else {}
    params = {"version": VERSION, "criteria": criteria, "n": int(n), "seed": int(seed), "min_words": int(min_words), "require_grade": bool(require_grade), "require_fulltext": bool(require_fulltext),
              "max_tokens": int(max_tokens), "kept": len(keep or ()),
              "extends_run": (extend_run if extend_run == "judged" else str(uuid.UUID(str(extend_run)))) if extend_run else None,     # TEMPORARY "judged": remove with evaluation_reuse.py
              "stratify": bool(stratify), "strata": s["strata"], "paper_ids": ids, "pool": s["pool"], "left_out_in_corpus": s["left_out_in_corpus"], "k": cfg["k"], "lexical_k": cfg.get("lexical_k", 0), "double_share": fscore.double_share(),
              "embedding": cfg["embedding"], "corpus": s["corpus"].manifest["corpus_id"], "web_search": False, "tools": [], "earliest_online": min(dates).isoformat() if dates else None,
              "with_cards_at_start": {k: sum(1 for p in ids if (cards.get((p, k)) or {}).get("status") == "valid") for k in CARD_KINDS} if criteria == "full" else None}
    model = "+".join(sorted({j["model"]["id"] for j in cfg["judges"]}))
    con.execute("INSERT INTO evaluation_run(run_id,kind,label,status,synthetic,config_name,config_sha256,model,knowledge_cutoff,parameters) VALUES (%s,%s,%s,'running',%s,%s,%s,%s,%s,%s)",
                (run_id, criteria, (label or "").strip()[:120], synthetic, cfg["name"], scoring_judges.fingerprint(cfg), model, cutoff, _jsonb(params)))
    con.commit()
    return {"run_id": str(run_id), "papers": len(ids), "model": model, "knowledge_cutoff": cutoff.isoformat(), "pool": s["pool"], "criteria": criteria}


def corpora_for(where, criteria):
    """The corpora of a run: the abstract corpus, and for a full run the two card corpora (the same the proposals are compared with)."""
    na = "not judged in this run (novelty only)"
    if criteria == "full":
        return fscore.load_corpora({c: where[c] for c in CRITERIA}, where["workspace"])
    novelty = scoring_corpora.AbstractCorpus(where["novelty"])
    distribution = fscore.nearest_distribution(novelty.vectors, Path(where["workspace"]) / "scoring" / f"nearest_{novelty.manifest['corpus_id']}.npy")
    return fscore.Corpora(novelty=novelty, distributions={"novelty": distribution}, missing={"methodology": na, "impact": na})


def _status(criteria, kind):
    asked = CRITERIA if kind == "full" else ("novelty",)
    given = [c for c in asked if criteria[c]["vote"] is not None]
    return "ok" if len(given) == len(asked) else "partial" if given else "failed"


def execute(con, run_id, *, where=None, provider=None, synthetic_provider=None, embedder=None, gate=None, progress=None):
    """Score the papers of a run that have no usable score yet (a resumed run does only the rest). Commits after each paper. -> {"scored", "failed", "state", ...}; `state` is completed,
    stopped (Stop or a limit) or failed. The run row gets its final status. `synthetic_provider` and `embedder` stand in for the real ones in the tests."""
    where = where or settings()
    run = con.execute("SELECT parameters, status, synthetic, kind FROM evaluation_run WHERE run_id=%s", (run_id,)).fetchone()
    if run is None:
        raise ValueError("Unknown evaluation run")
    params, synthetic, kind = run[0], run[2], run[3]
    cfg = scoring_judges.load_config(where["config"])
    cfg = {**cfg, "budget": {**cfg["budget"], "max_tokens": params["max_tokens"]}}
    if synthetic_provider is not None:
        providers = {r: synthetic_provider for r in scoring_judges.JUDGE_ROLES}
    else:
        if not cfg["run_live"]:
            raise ValueError("the configuration has run_live false")
        if getattr(provider, "name", None) != "openai":
            raise ValueError("a live evaluation runs on the OpenAI provider")
        from . import embeddings
        providers, embedder = {r: provider for r in scoring_judges.JUDGE_ROLES}, embeddings.get_embedder(cfg["embedding"])
    corpora = corpora_for(where, kind)
    for c in CRITERIA:
        corpus = corpora.get(c)
        if corpus is not None and corpus.manifest["embedder"] != embedder.model_id:
            raise ValueError(f"the {c} corpus was embedded with {corpus.manifest['embedder']}, the embedder is {embedder.model_id}")
    inputs = read_inputs(where["directory"])
    done = {r[0] for r in con.execute("SELECT paper_id FROM evaluation_score WHERE run_id=%s AND status='ok'", (run_id,)).fetchall()}
    todo = [p for p in params["paper_ids"] if p not in done]
    cards = latest_cards(con, todo) if kind == "full" else {}
    session = "evr_" + run_id.hex[:12]
    judges = scoring_judges.Judges(where["workspace"], cfg, providers, session_id=session, gate=gate)
    models = {j["role"]: {"model": j["model"].get("id"), "knowledge_cutoff": j["model"].get("knowledge_cutoff"), "reasoning_effort": j["params"].get("reasoning_effort")} for j in cfg["judges"]}
    summaries = {c: fscore.corpus_summary(corpora, c) for c in CRITERIA}
    state, note, scored, failed = "completed", "", 0, 0
    con.execute("UPDATE evaluation_run SET status='running', finished_at=NULL WHERE run_id=%s", (run_id,))
    con.commit()
    try:
        for pid in todo:
            roles = fscore.panel(pid, params["double_share"])
            content = {"title": inputs[pid]["title"], "abstract": inputs[pid]["abstract"]}
            used_cards = {}
            if kind == "full":
                content["cards"] = {}
                for k in CARD_KINDS:
                    row = cards.get((pid, k))
                    if row and row["status"] == "valid":
                        content["cards"][k] = {"status": "valid", "card": row["card"]}
                        used_cards[k] = {"model": row["model"], "source_sha256": row["source_sha256"], "created_at": row["created_at"].isoformat(timespec="seconds")}
            criteria = fscore.score_proposal(judges, corpora, embedder, params["k"], scoring_io.opaque_id(SALT, pid), content, workers=cfg["concurrency"], roles=roles, lexical_k=params["lexical_k"])
            for c in CRITERIA:
                criteria[c].pop("proposal_text", None)              # the abstract and the card are publisher text and derived text: the card is in evaluation_card, the abstract nowhere
            hybrid = criteria["novelty"].get("hybrid") or {}
            status = _status(criteria, kind)
            votes = {c: criteria[c]["vote"] for c in CRITERIA}
            doc = {"schema": VERSION, "paper_id": pid, "scored_on": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "synthetic": bool(synthetic), "kind": kind,
                   "config": cfg["name"], "config_sha256": scoring_judges.fingerprint(cfg), "models": {r: models[r] for r in roles}, "session_id": session, "k": params["k"],
                   "lexical_k": params["lexical_k"], "panel": {"roles": list(roles), "double": len(roles) > 1}, "corpora": summaries, "cards": used_cards,
                   "criteria": criteria, "status": status}
            failed_row = status == "failed"
            con.execute("INSERT INTO evaluation_score(score_id,run_id,paper_id,status,novelty,novelty_hybrid,methodology,impact,document) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (uuid.uuid4(), run_id, pid, status, votes["novelty"], None if failed_row else hybrid.get("vote"), votes["methodology"] if kind == "full" else None,
                         votes["impact"] if kind == "full" else None, _jsonb(doc)))
            con.commit()
            scored += not failed_row
            failed += failed_row
            if progress:
                progress(len(done) + scored + failed, len(params["paper_ids"]), judges.budget.used())
    except (BudgetExceeded, Cancelled) as exc:                 # a limit, a refusal of the provider or Stop: what is stored stays, the run can be resumed
        state, note = "stopped", str(exc)[:300]
    except Exception as exc:
        state, note = "failed", f"{type(exc).__name__}: {str(exc)[:280]}"
        LOG.error("evaluation run %s failed: %s", run_id, note)
    con.rollback()
    usage = judges.budget.used()
    con.execute("UPDATE evaluation_run SET status=%s, note=%s, finished_at=clock_timestamp(), stats=%s WHERE run_id=%s", (state, note, _jsonb({"usage": usage, "scored_now": scored, "failed_now": failed}), run_id))
    con.commit()
    return {"run_id": str(run_id), "state": state, "note": note, "scored": scored, "failed": failed, "usage": usage}


# --- the background job of the dashboard ----------------------------------------------------------------------------------------------------------

TASKS = ("run", "fulltext", "cards")


class EvaluationJob:
    """One task at a time - a run, the full-text downloads or the cards - in a thread of the dashboard server. The state is in memory: a restart of the server ends the task (a run row then
    reads `running` with no job behind it, and the page shows it as interrupted and offers to resume; downloads and cards are simply asked again by the next click)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = {"state": "idle"}
        self._thread = None
        self._refresh_lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def running_id(self):
        s = self.snapshot()
        return s.get("run_id") if s.get("state") in ("starting", "running") and s.get("task", "run") == "run" else None

    def busy(self):
        return self.snapshot().get("state") in ("starting", "running")

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def _claim(self, task):
        with self._lock:
            if self._state.get("state") in ("starting", "running"):
                raise ValueError(f"a {self._state.get('task', 'run')} task is already going; stop it or wait")
            self._state = {"state": "starting", "task": task}

    def _launch(self, target, *args):
        self._stop.clear()
        self._thread = threading.Thread(target=target, args=args, daemon=True, name="evaluation-task")
        self._thread.start()

    def start(self, database_url, *, resume=None, provider_factory=None, **options):
        """Create the run (or take `resume`, an earlier run id) and start it. The checks run here, in the caller's thread, so a refusal comes back as an error. -> snapshot."""
        import psycopg
        self._claim("run")
        try:
            if not os.environ.get("OPENAI_API_KEY") and provider_factory is None:
                raise ValueError("OPENAI_API_KEY is not set on this server")
            with psycopg.connect(database_url) as con:
                if resume:
                    if con.execute("SELECT status FROM evaluation_run WHERE run_id=%s", (uuid.UUID(str(resume)),)).fetchone() is None:
                        raise ValueError("Unknown evaluation run")
                    run_id = uuid.UUID(str(resume))
                else:
                    run_id = uuid.UUID(create_run(con, **options)["run_id"])
            provider_factory = provider_factory or _openai
            self._set(run_id=str(run_id), state="running", done=0, total=0, message="", usage={}, started_at=_now())
            self._launch(self._work_run, database_url, run_id, provider_factory)
        except Exception:
            self._set(state="idle")
            raise
        return self.snapshot()

    def start_fulltext(self, database_url, **sample_options):
        """Steps 2 and 3 for the sample: import and parse the full texts the project already has (the archive, and the files put by hand). Nothing is downloaded."""
        import psycopg
        self._claim("fulltext")
        try:
            where = settings()
            s = _sample_for(database_url, where, sample_options)
            with psycopg.connect(database_url) as con:
                sync_papers(con, s["ids"], s["inputs"], s["display"])
                con.commit()
            self._set(state="running", done=0, total=len(s["ids"]), message="", started_at=_now())
            self._launch(self._work_fulltext, database_url, s["ids"], where)
        except Exception:
            self._set(state="idle")
            raise
        return self.snapshot()

    def start_download(self, database_url, *, limit=None, email=None, get=None, min_words=DEFAULT_MIN_WORDS, require_grade=True, seed=DEFAULT_SEED):
        """Step 2b: try to download the open-access PDFs of the usable papers that have none, at most `limit`, in the order of the draw. Free (no model), but it goes on the internet."""
        import psycopg
        email = (email if email is not None else os.environ.get("UNPAYWALL_EMAIL", "")).strip()
        if not email or "@" not in email:
            raise ValueError("Set UNPAYWALL_EMAIL in .env (a real address; Unpaywall asks for it) and restart the server")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 100000):
            raise ValueError("At most must be a whole number between 1 and 100000")
        self._claim("download")
        try:
            where = settings()
            with psycopg.connect(database_url) as con:
                if not cards_tables_exist(con) or not download_allowed(con):
                    raise ValueError("Migration 033 is not applied (research0 init)")
                ids, inputs, display = download_todo(con, where, limit=limit, min_words=min_words, require_grade=require_grade, seed=seed)
                if not ids:
                    raise ValueError("no paper to try: every usable paper has a full text or was already tried (the others are in the Excel list to fetch by hand)")
                sync_papers(con, ids, inputs, display)
                con.commit()
            self._set(state="running", done=0, total=len(ids), message="", started_at=_now())
            self._launch(self._work_download, database_url, ids, where, email, get)
        except Exception:
            self._set(state="idle")
            raise
        return self.snapshot()

    def start_cards(self, database_url, *, max_tokens=None, provider_factory=None, **sample_options):
        """Step 4 for the sample: the cards still to make, within `max_tokens`."""
        import psycopg
        self._claim("cards")
        try:
            if not os.environ.get("OPENAI_API_KEY") and provider_factory is None:
                raise ValueError("OPENAI_API_KEY is not set on this server")
            where = settings()
            s = _sample_for(database_url, where, sample_options)
            with psycopg.connect(database_url) as con:
                sync_papers(con, s["ids"], s["inputs"], s["display"])
                con.commit()
                todo, _ = cards_todo(con, s["ids"], where)
            if not todo:
                raise ValueError("no card to make: every parsed paper of the sample has its two cards (or no paper is parsed yet)")
            self._set(state="running", done=0, total=len(todo), message="", usage={}, started_at=_now())
            self._launch(self._work_cards, database_url, s["ids"], where, max_tokens, provider_factory or _openai)
        except Exception:
            self._set(state="idle")
            raise
        return self.snapshot()

    def submit_cards_batch(self, database_url, *, max_tokens, submit=None, **sample_options):
        """Step 4 by Batch: write and submit the requests of the cards still to make. Quick (an upload), so it runs in the caller's thread, but it holds the task slot meanwhile so a
        synchronous card step cannot pick the same cards. -> snapshot."""
        import psycopg
        self._claim("cards_batch")
        try:
            if not os.environ.get("OPENAI_API_KEY") and submit is None:
                raise ValueError("OPENAI_API_KEY is not set on this server")
            where = settings()
            s = _sample_for(database_url, where, sample_options)
            with psycopg.connect(database_url) as con:
                sync_papers(con, s["ids"], s["inputs"], s["display"])
                con.commit()
                out = submit_card_batch(con, s["ids"], where, max_tokens=max_tokens, submit=submit)
        except Exception:
            self._set(state="idle")
            raise
        self._finish(out, f"Batch {out['batch_id'] or '?'} submitted: {out['cards']} cards; the answers come within 24 hours - press \"Check the Batches\" to collect them")
        return self.snapshot()

    def refresh_cards_batches(self, database_url, client=None):
        """Check the Batches of cards and collect those that ended. Not a task: it only stores answers, so it may run beside one; one check at a time."""
        import psycopg
        with self._refresh_lock, psycopg.connect(database_url) as con:
            return refresh_card_batches(con, settings(), client)

    def stop(self):
        if self.busy():
            self._stop.set()
            self._set(message="stopping after the step in progress")
        return self.snapshot()

    def _gate(self):
        if self._stop.is_set():
            raise Cancelled("stopped by a person")

    def _progress(self, done, total, usage):
        self._set(done=done, total=total, usage=usage)

    def _finish(self, out, message):
        self._set(state="idle", last=out, message=message)

    def _fail(self, exc):
        LOG.error("evaluation job: %s", exc)
        self._set(state="idle", message=f"{type(exc).__name__}: {str(exc)[:200]}")

    def _work_run(self, database_url, run_id, provider_factory):
        import psycopg
        try:
            with psycopg.connect(database_url) as con:
                self._set(total=len(con.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (run_id,)).fetchone()[0]["paper_ids"]))
                out = execute(con, run_id, provider=provider_factory(), gate=self._gate, progress=self._progress)
            self._finish(out, out["note"] or f"run {out['state']}: {out['scored']} scored, {out['failed']} failed")
        except Exception as exc:
            self._fail(exc)

    def _work_fulltext(self, database_url, ids, where):
        import psycopg
        try:
            with psycopg.connect(database_url) as con:
                out = import_fulltexts(con, ids, where, gate=self._gate, progress=self._progress)
            self._finish(out, f"full texts imported: {out['project_archive']} from the project's archive, {out['manual_upload']} put by hand, {out['open_access_download']} downloaded, {out['unchanged']} already there; "
                              f"{out['not_in_the_project']} papers have none" + (f"; {len(out['problems'])} files could not be read" if out["problems"] else ""))
        except Exception as exc:
            self._fail(exc)

    def _work_download(self, database_url, ids, where, email, get):
        import psycopg

        def progress(done, total, counts):
            self._set(done=done, total=total, message=", ".join(f"{v} {OUTCOME_WORDS.get(k, k).split(' (')[0].replace('_', ' ')}" for k, v in sorted(counts.items())))
        try:
            with psycopg.connect(database_url) as con:
                out = download_fulltexts(con, ids, where, email, get=get, gate=self._gate, progress=progress)
            got, tried = out["outcomes"].get("imported", 0), sum(out["outcomes"].values())
            self._finish(out, ("stopped: " if out["stopped"] else "") + f"{got} PDFs found and parsed out of {tried} papers tried; the others are in the Excel list to fetch by hand")
        except Exception as exc:
            self._fail(exc)

    def _work_cards(self, database_url, ids, where, max_tokens, provider_factory):
        import psycopg
        try:
            with psycopg.connect(database_url) as con:
                out = make_cards(con, ids, where, provider=provider_factory(), max_tokens=max_tokens, gate=self._gate, progress=self._progress)
            self._finish(out, out["note"] or f"cards: {out['valid']} valid, {out['rejected']} refused")
        except Exception as exc:
            self._fail(exc)


def _openai():
    from .forum_providers import get_provider
    return get_provider("openai")


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --- reading ---------------------------------------------------------------------------------------------------------------------------------------

def tables_exist(con):
    return bool(con.execute("SELECT to_regclass('evaluation_run') IS NOT NULL AND to_regclass('evaluation_score') IS NOT NULL").fetchone()[0])


def cards_tables_exist(con):
    return bool(con.execute("SELECT to_regclass('evaluation_card') IS NOT NULL AND to_regclass('evaluation_fulltext') IS NOT NULL").fetchone()[0])


def download_allowed(con):
    """Whether migration 033 is applied: the origin of a downloaded full text is accepted by the table."""
    row = con.execute("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname='evaluation_fulltext_origin_check' AND conrelid=to_regclass('evaluation_fulltext')").fetchone()
    return bool(row and "open_access_download" in row[0])


def meta(con):
    """What the page needs before any step: the folder, the models and their cutoff, the defaults, and what is missing."""
    where = settings()
    out = {"version": VERSION, "migration_applied": tables_exist(con), "cards_migration_applied": cards_tables_exist(con),
           "defaults": {"n": DEFAULT_N, "seed": DEFAULT_SEED, "min_words": DEFAULT_MIN_WORDS, "require_grade": True, "require_fulltext": True, "stratify": True, "max_tokens": DEFAULT_MAX_TOKENS},
           "openai_configured": bool(os.environ.get("OPENAI_API_KEY")), "problems": [], "manual_folder": str(_dir(where, "fulltext_manual"))}
    try:
        cfg = scoring_judges.load_config(where["config"])
        out["models"] = [{"role": j["role"], "model": j["model"]["id"], "knowledge_cutoff": j["model"].get("knowledge_cutoff"), "reasoning_effort": j["params"].get("reasoning_effort")} for j in cfg["judges"]]
        out["k"], out["lexical_k"], out["double_share"] = cfg["k"], cfg.get("lexical_k", 0), fscore.double_share()
    except (OSError, ValueError, scoring_judges.ConfigError) as exc:
        out["problems"].append(f"configuration: {exc}")
    try:
        cards_cfg = scoring_judges.load_config(where["cards_config"])
        out["card_model"] = {"model": cards_cfg["judges"][0]["model"]["id"], "reasoning_effort": cards_cfg["judges"][0]["params"].get("reasoning_effort"),
                             "max_output_tokens": cards_cfg["judges"][0]["params"]["max_output_tokens"], "profiles": {k: paper_cards.PROFILES[k].version for k in CARD_KINDS}}
    except (OSError, ValueError, scoring_judges.ConfigError) as exc:
        out["problems"].append(f"card configuration: {exc}")
    try:
        inputs, display = read_inputs(where["directory"]), read_display(where["directory"])
    except (OSError, ValueError) as exc:
        inputs, display = {}, {}
        out["problems"].append(str(exc))
    if not inputs:
        out["problems"].append("The abstracts are not on this server (inputs/evaluation_2026 is local and outside git): runs can be read here but not started.")
    out["papers"] = {"abstracts": len(inputs), "with_grade": sum(1 for p in inputs if (display.get(p) or {}).get("ajg2024")),
                     "usable": len(eligible(inputs, display, min_words=DEFAULT_MIN_WORDS, require_grade=False)),
                     "usable_with_grade": len(eligible(inputs, display, min_words=DEFAULT_MIN_WORDS, require_grade=True))}
    with_text = set(available_files(where, {p: (display.get(p) or {}).get("doi", "") for p in inputs}))
    out["papers"].update(with_fulltext=len(with_text), usable_with_fulltext=len(eligible(inputs, display, min_words=DEFAULT_MIN_WORDS, require_grade=False, only=with_text)),
                         usable_with_fulltext_and_grade=len(eligible(inputs, display, min_words=DEFAULT_MIN_WORDS, require_grade=True, only=with_text)))
    out["corpus_present"] = (Path(where["novelty"]) / "corpus.json").is_file()
    out["card_corpora_present"] = all(scoring_corpora.is_card_corpus(Path(where[c]), k) for c, k in fscore.CARD_OF.items())
    if not out["corpus_present"]:
        out["problems"].append("The abstract corpus is not on this server: runs cannot be started.")
    if not out["cards_migration_applied"]:
        out["problems"].append("Migration 032 has not been applied: the full texts, the cards and the runs on three criteria are not available.")
    out["grades"] = list(GRADES)
    return out


def pipeline(con, *, where=None, **sample_options):
    """The steps of a sample, paper by paper: abstract, full text (and how it was found or why not), parsing, the two cards; with the counts and the estimate of the cards still to make."""
    where = where or settings()
    options = with_keep(con, sample_options, where)
    s = sample(where, **options)
    ids, display = s["ids"], s["display"]
    files = latest_fulltext(con, ids) if cards_tables_exist(con) else {}
    cards = latest_cards(con, ids) if cards_tables_exist(con) else {}
    todo, short = cards_todo(con, ids, where) if cards_tables_exist(con) else ([], [])
    send, held, stopped = cards_for_batch(con, ids, where) if cards_tables_exist(con) else ([], [], [])
    short, held, stopped = set(short), set(held), set(stopped)
    available = available_files(where, {p: (display.get(p) or {}).get("doi", "") for p in ids})
    rows = []
    for pid in ids:
        f, d = files.get(pid), display.get(pid) or {}
        parsed = parsed_path(where, pid).is_file()
        rows.append({"paper_id": pid, "title": s["inputs"][pid]["title"], "doi": d.get("doi", ""), "journal": d.get("journal", ""), "ajg2024": d.get("ajg2024", ""),
                     "fulltext": None if f is None else {"origin": f["origin"], "outcome": f["outcome"], "detail": f["detail"], "found_by": f["found_by"], "version": f["version_label"],
                                                         "parse_status": f["parse_status"], "at": f["created_at"].isoformat(timespec="seconds")},
                     "parsed": parsed, "available": available[pid][1] if pid in available else None,
                     "version": version_of(available[pid][2]) if pid in available else None,
                     "cards": {k: ("too_short" if (pid, k) in short else "in_batch" if (pid, k) in held else "stopped" if (pid, k) in stopped else
                                   (cards.get((pid, k)) or {}).get("status") or ("to_make" if parsed else None)) for k in CARD_KINDS},
                     "rejections": {k: cards[(pid, k)]["rejection"] for k in CARD_KINDS if (cards.get((pid, k)) or {}).get("status") == "rejected"}})
    counts = {"papers": len(ids), "with_fulltext": sum(1 for r in rows if r["fulltext"] and r["fulltext"]["outcome"] == "imported"),
              "parsed": sum(r["parsed"] for r in rows), "not_imported": sum(1 for r in rows if r["available"] and r["fulltext"] is None),
              "available": {o: sum(1 for r in rows if r["available"] == o) for o in ("project_archive", "manual_upload", "open_access_download")}, "not_in_the_project": sum(r["available"] is None for r in rows),
              "unavailable": {}, "cards": {k: {st: sum(1 for r in rows if r["cards"][k] == st) for st in ("valid", "rejected", "to_make", "too_short", "in_batch", "stopped")} for k in CARD_KINDS},
              "both_cards": sum(1 for r in rows if all(r["cards"][k] == "valid" for k in CARD_KINDS))}
    for r in rows:
        if r["fulltext"] and r["fulltext"]["outcome"] != "imported":
            counts["unavailable"][r["fulltext"]["outcome"]] = counts["unavailable"].get(r["fulltext"]["outcome"], 0) + 1
    try:
        estimate = estimate_cards(send, scoring_judges.load_config(where["cards_config"]))      # what the next Batch would send
    except (OSError, ValueError, scoring_judges.ConfigError):
        estimate = None
    return {"sample": {"n": len(ids), "pool": s["pool"], "left_out_in_corpus": s["left_out_in_corpus"], "kept": len(options.get("keep") or ()), "strata": s["strata"],
                       "stratify": bool(options.get("stratify"))}, "counts": counts, "estimate": estimate, "rows": rows,
            "manual_folder": str(_dir(where, "fulltext_manual")), "migration_applied": cards_tables_exist(con), "batches": card_batches(where)}


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out, i = [0.0] * len(values), 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx, sy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sx == 0 or sy == 0 else round(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy) ** 0.5, 3)


def _num(v):
    return None if v is None else float(v)


def _desc(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": round(statistics.fmean(values), 2) if values else None, "sd": round(statistics.stdev(values), 2) if len(values) > 1 else None,
            "median": round(statistics.median(values), 2) if values else None}


def _mean3(r):
    values = [r["novelty"], r["methodology"], r["impact"]]
    return round(sum(values) / 3, 2) if all(v is not None for v in values) else None


def list_runs(con, active=None):
    if not tables_exist(con):
        return {"migration_applied": False, "runs": []}
    full = cards_tables_exist(con)                     # before migration 032 the score rows have no methodology / impact columns
    extra = "avg(s.methodology),avg(s.impact)" if full else "NULL::numeric,NULL::numeric"
    out = []
    for (rid, kind, label, status, synthetic, model, cutoff, params, note, created, finished, ok, partial, bad, mean_f, mean_h, mean_m, mean_i) in con.execute(
            "SELECT r.run_id,r.kind,r.label,r.status,r.synthetic,r.model,r.knowledge_cutoff,r.parameters,r.note,r.created_at,r.finished_at,"
            "count(s.*) FILTER (WHERE s.status='ok'),count(s.*) FILTER (WHERE s.status='partial'),count(s.*) FILTER (WHERE s.status='failed'),avg(s.novelty),avg(s.novelty_hybrid)," + extra + " "
            "FROM evaluation_run r LEFT JOIN LATERAL (SELECT DISTINCT ON (paper_id) * FROM evaluation_score WHERE run_id=r.run_id ORDER BY paper_id, created_at DESC) s ON true "
            "GROUP BY r.run_id ORDER BY r.created_at DESC").fetchall():
        shown = "interrupted" if status == "running" and str(rid) != active else status
        r2 = lambda v: round(float(v), 2) if v is not None else None
        out.append({"run_id": str(rid), "kind": kind, "label": label, "status": shown, "synthetic": synthetic, "model": model, "knowledge_cutoff": cutoff.isoformat(), "papers": len(params["paper_ids"]),
                    "scored": ok + partial, "complete": ok, "partial": partial, "failed": bad, "mean_first": r2(mean_f), "mean_hybrid": r2(mean_h), "mean_methodology": r2(mean_m), "mean_impact": r2(mean_i),
                    "created_at": created.isoformat(timespec="seconds"), "finished_at": finished.isoformat(timespec="seconds") if finished else None, "note": note,
                    "n": params["n"], "seed": params["seed"], "max_tokens": params["max_tokens"]})
    return {"migration_applied": True, "runs": out}


def run_detail(con, run_id, active=None):
    """One run: its settings, a row per paper (votes, indices, journal and ABS grade), and the descriptive statistics: each criterion overall and by ABS grade, the rank correlation of each
    vote with the ABS grade, the two novelty judges, the double sample."""
    run = con.execute("SELECT run_id,label,status,synthetic,config_name,model,knowledge_cutoff,parameters,stats,note,created_at,finished_at,kind FROM evaluation_run WHERE run_id=%s", (run_id,)).fetchone()
    if run is None:
        raise ValueError("Unknown evaluation run")
    params, kind = run[7], run[12]
    extra = ",methodology,impact" if cards_tables_exist(con) else ",NULL::numeric,NULL::numeric"
    latest = {r[0]: r for r in con.execute(
        f"SELECT DISTINCT ON (paper_id) paper_id,status,novelty,novelty_hybrid,document,created_at{extra} FROM evaluation_score WHERE run_id=%s ORDER BY paper_id, created_at DESC", (run_id,)).fetchall()}
    papers = {r[0]: r for r in con.execute("SELECT paper_id,title,journal,ajg2024,doi,online_date,source_sets,abstract_words FROM evaluation_paper WHERE paper_id = ANY(%s)", (params["paper_ids"],)).fetchall()}
    rows = []
    for pid in params["paper_ids"]:
        p, s = papers.get(pid), latest.get(pid)
        doc = s[4] if s else None
        crit = (doc or {}).get("criteria") or {"novelty": (doc or {}).get("novelty") or {}}
        nov = crit.get("novelty") or {}
        sim = nov.get("similarity") or {}
        hyb = nov.get("hybrid") or {}
        given = [v for v in (nov.get("votes") or {}).values() if v is not None]
        row = {"paper_id": pid, "title": p[1] if p else pid, "journal": p[2] if p else "", "ajg2024": p[3] if p else "", "doi": p[4] if p else "", "online_date": p[5].isoformat() if p and p[5] else None,
               "words": p[7] if p else None, "status": s[1] if s else "waiting", "novelty": _num(s[2]) if s else None, "hybrid": _num(s[3]) if s else None,
               "methodology": _num(s[6]) if s else None, "impact": _num(s[7]) if s else None,
               "indicative": nov.get("indicative_vote"), "top1": sim.get("top1"), "percentile": sim.get("percentile"), "mean_topk": sim.get("mean_topk"),
               "double": bool(len(given) == 2), "gap": abs(given[0] - given[1]) if len(given) == 2 else None, "lexical_only": len(hyb.get("lexical_only") or []) if hyb else None,
               "reasons": {c: (crit.get(c) or {}).get("reason") for c in CRITERIA if (crit.get(c) or {}).get("vote") is None and (crit.get(c) or {}).get("reason")} if s else {}}
        row["mean"] = _mean3(row)
        rows.append(row)
    done = [r for r in rows if r["status"] in ("ok", "partial")]
    order = {g: i + 1 for i, g in enumerate(GRADES)}
    grades = [_num(order.get(r["ajg2024"])) for r in done]
    keys = {"first": "novelty", "hybrid": "hybrid", "methodology": "methodology", "impact": "impact", "mean": "mean"}
    by_grade = [{"grade": g, **{k: _desc(r[f] for r in done if r["ajg2024"] == g) for k, f in keys.items()}} for g in GRADES]
    ungraded = [r for r in done if r["ajg2024"] not in order]
    if ungraded:
        by_grade.append({"grade": "not graded", **{k: _desc(r[f] for r in ungraded) for k, f in keys.items()}})
    pairs = [(r["novelty"], r["hybrid"]) for r in done if r["novelty"] is not None and r["hybrid"] is not None]
    diffs = [h - f for f, h in pairs]
    stats = {**{k: _desc(r[f] for r in done) for k, f in keys.items()}, "by_grade": by_grade,
             "spearman_with_grade": {**{k: _spearman(grades, [r[f] for r in done]) for k, f in keys.items()}, "n": sum(g is not None for g in grades)},
             "judges": {"n": len(pairs), "mean_difference": round(statistics.fmean(diffs), 2) if diffs else None, "mean_abs_difference": round(statistics.fmean(abs(d) for d in diffs), 2) if diffs else None,
                        "exact_share": round(sum(d == 0 for d in diffs) / len(diffs), 3) if diffs else None, "within_one_share": round(sum(abs(d) <= 1 for d in diffs) / len(diffs), 3) if diffs else None},
             "double_sample": {"n": sum(r["double"] for r in done), "mean_gap": round(statistics.fmean(r["gap"] for r in done if r["gap"] is not None), 2) if any(r["gap"] is not None for r in done) else None}}
    shown = "interrupted" if run[2] == "running" and str(run[0]) != active else run[2]
    return {"run": {"run_id": str(run[0]), "kind": kind, "label": run[1], "status": shown, "synthetic": run[3], "config": run[4], "model": run[5], "knowledge_cutoff": run[6].isoformat(),
                    "parameters": {k: v for k, v in params.items() if k != "paper_ids"}, "usage": (run[8] or {}).get("usage"), "note": run[9], "created_at": run[10].isoformat(timespec="seconds"),
                    "finished_at": run[11].isoformat(timespec="seconds") if run[11] else None},
            "rows": rows, "stats": stats, "waiting": sum(r["status"] == "waiting" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
            "partial": sum(r["status"] == "partial" for r in rows)}


EXPORT_FORMATS = ("csv", "json")
EXPORT_COLUMNS = (("run_id", "run_id"), ("run_label", "run_label"), ("test_data", "test_data"), ("model", "model"), ("knowledge_cutoff", "knowledge_cutoff"),
                  ("paper_id", "paper_id"), ("title", "title"), ("journal", "journal"), ("abs_grade_2024", "ajg2024"), ("doi", "doi"), ("online_date", "online_date"),
                  ("status", "status"), ("novelty_first", "novelty"), ("novelty_hybrid", "hybrid"), ("methodology", "methodology"), ("impact", "impact"), ("mean_of_three", "mean"),
                  ("closest_abstract_similarity", "top1"), ("closest_abstract_percentile", "percentile"), ("mean_similarity_topk", "mean_topk"), ("judged_twice", "double"),
                  ("gap_between_copies", "gap"), ("lexical_only_candidates", "lexical_only"), ("not_scored_reasons", "reasons"))


def _cell(value):
    """One CSV cell. Text that starts like a spreadsheet formula gets a leading apostrophe, so a paper's title cannot run in Excel."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        value = "; ".join(f"{k}: {v}" for k, v in value.items())
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def export_run(con, run_id, fmt="csv", active=None):
    """The results of one run as a file: (file name, content type, bytes). CSV = a row per paper of the sample (those not judged yet included, with their status); JSON = the same rows plus
    the run's settings and the descriptive statistics the page shows. It reads `run_detail`, so the file and the page cannot disagree, and it sends nothing anywhere."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError("Unknown export format (csv or json)")
    detail = run_detail(con, run_id, active)
    run, rows = detail["run"], detail["rows"]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", run["label"] or "").strip("-")[:40]
    name = f"evaluation_{slug + '_' if slug else ''}{run['run_id'][:8]}.{fmt}"
    if fmt == "json":
        body = {"exported_at": _now(), "version": VERSION, "run": run, "stats": detail["stats"], "counts": {"papers": len(rows), "waiting": detail["waiting"], "failed": detail["failed"], "partial": detail["partial"]},
                "rows": rows,
                "note": "Votes are 0-10. The journal and the ABS grade were not shown to the judges. A relation between grade and vote is a description of this sample, not a validation."}
        return name, "application/json; charset=utf-8", json.dumps(body, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([header for header, _ in EXPORT_COLUMNS])
    context = {"run_id": run["run_id"], "run_label": run["label"], "test_data": run["synthetic"], "model": run["model"], "knowledge_cutoff": run["knowledge_cutoff"]}
    for r in rows:
        merged = {**context, **r}
        writer.writerow([_cell(merged.get(key)) for _, key in EXPORT_COLUMNS])
    return name, "text/csv; charset=utf-8", b"\xef\xbb\xbf" + out.getvalue().encode("utf-8")      # the BOM lets Excel read the accents


def paper_detail(con, run_id, paper_id, where=None):
    """One paper of a run: what was shown to the judges (the abstract is read from the local folder when it is there), the cards, every judge's answer, the neighbours they read."""
    row = con.execute("SELECT document,status,created_at FROM evaluation_score WHERE run_id=%s AND paper_id=%s ORDER BY created_at DESC LIMIT 1", (run_id, paper_id)).fetchone()
    meta_row = con.execute("SELECT title,journal,ajg2024,doi,online_date FROM evaluation_paper WHERE paper_id=%s", (paper_id,)).fetchone()
    if meta_row is None:
        raise ValueError("Unknown paper")
    attempts = con.execute("SELECT status,created_at FROM evaluation_score WHERE run_id=%s AND paper_id=%s ORDER BY created_at", (run_id, paper_id)).fetchall()
    try:
        shown = read_inputs((where or settings())["directory"]).get(paper_id)
    except (OSError, ValueError):
        shown = None
    cards = {}
    if cards_tables_exist(con):
        for (pid, kind), c in latest_cards(con, [paper_id]).items():
            cards[kind] = {"status": c["status"], "card": c["card"], "model": c["model"], "rejection": c["rejection"], "created_at": c["created_at"].isoformat(timespec="seconds")}
    doc = row[0] if row else None
    if doc and "criteria" not in doc:                     # a novelty-only score made before migration 032
        doc = {**doc, "criteria": {"novelty": doc.get("novelty")}}
    return {"paper": {"paper_id": paper_id, "title": meta_row[0], "journal": meta_row[1], "ajg2024": meta_row[2], "doi": meta_row[3], "online_date": meta_row[4].isoformat() if meta_row[4] else None},
            "shown_to_the_judges": {"title": shown["title"], "abstract": shown["abstract"]} if shown else None, "cards": cards,
            "card_fields": {k: [{"key": f, "label": fscore.FIELD_LABELS.get(f, f)} for f in paper_cards.PROFILES[k].fields] for k in CARD_KINDS},
            "score": doc, "status": row[1] if row else "waiting", "scored_at": row[2].isoformat(timespec="seconds") if row else None,
            "attempts": [{"status": a, "at": t.isoformat(timespec="seconds")} for a, t in attempts]}


def describe():
    """For the page: how to read the evaluation."""
    return {"notes": [
        "Each paper is evaluated exactly as a forum proposal is, by the same code and prompts. NOVELTY: the judge reads its title and abstract and the abstracts of the closest published papers "
        "of the CTO/TIM corpus (2015-2025); distance from what was published is good, if it is a matter of substance. Two votes: the first judge, and the hybrid judge, which also reads papers "
        "that share the paper's own terms (BM25) and compares question, mechanism and predictions apart.",
        "METHODOLOGY and IMPACT (runs on three criteria): the judge reads the abstract and the paper's methodology or impact card against the cards of the closest corpus papers. The paper's "
        "cards are made as the corpus cards were: from the full text, the same sections, instructions, model and check of the quotations. The prompts still speak of a 'proposal' (they are "
        "the proposals' prompts, unchanged): a paper's methodology card describes a design that was carried out, and its impact card the change the paper claims having obtained results.",
        "A fifth of the papers is also judged by a second copy of each judge, to measure how far a judge agrees with itself.",
        "The judges never see the journal, its ABS grade, the DOI or the date (a card field that names the journal is removed). They are shown here, after the vote, to read the results. "
        "A relation between grade and vote is a description, not a validation.",
        "The judges' model has a declared knowledge cutoff before the papers and no web search or tool. A cutoff does not exclude that a paper circulated earlier as a preprint. "
        "The full texts are those the project already had (the open-access copies of the ANVUR package, 17 Oct 2026, most of them the authors' submitted manuscripts rather than "
        "the published version), those the download button finds (only links Unpaywall, OpenAlex or Semantic Scholar declare open access, from the DOI; never a paywall), and any file "
        "put in the folder by a person. A paper without one gets no methodology or impact vote, so by default the sample is drawn among the papers that have one; those are not a "
        "random part of the 2026 papers (open access is not random)."]}
