"""The discovery forum: agents work on one VALIDATED release, in read-only mode, and everything they say and do is stored.

Division of labour. The agents read through `rag.reader_connection` (the `research_agent` role when AGENT_DATABASE_URL is set: SELECT everywhere and INSERT on
its own retrieval log only). The orchestrator writes the discovery workspace (forum_episode, forum_agent, forum_message, question, agent_profile) with the
application's connection. Nothing here touches the release, the catalog, the extractions or the decisions.

A turn has two phases. In the private reflection the agent may use the tools (search the corpus; in A, B and C explore the construct database, and with the
construct-problem trigger list and read the dossiers of the construct relationships people decided); in the public contribution it writes to the forum with no tools,
and may cite only evidence ids that exist (E for a passage or a construct record, P for a dossier). Every episode follows one ideation trigger (proposal v4): it says how
to look for a puzzle, never what to study. After the rounds the question is written in a structured form and checked: a source nobody retrieved is flagged, not silently
kept. Budgets are enforced between model calls; a share of them is held back so the final question can always be written.
"""
import datetime
import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from . import embedding_jobs
from . import forum_cards as fcards
from . import forum_constructs as fcon
from . import forum_proposal as fprop
from . import forum_protocol as proto
from . import forum_sandbox as sandbox
from . import forum_scoring as fscore
from . import rag
from .embeddings import get_embedder
from .forum_providers import BudgetExceeded, Cancelled, ToolSpec, TurnRequest, get_provider

LOG = logging.getLogger("research0")
MODELS_PATH = Path(__file__).resolve().parents[2] / "config" / "forum_models.json"
SYNTHESIS_GRACE_SECONDS = 300
FINAL_TASKS = ("synthesize", "proposal_draft", "proposal_critique", "proposal_revise", "proposal_shorten",
               *fcards.TASKS.values())      # run after the rounds, paid from the part of the budget held back


def load_models(path=MODELS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- Creating an episode -----------------------------------------------------------------------------------------------------------------

def profile_id(episode_id, agent_key):
    return f"forum:{episode_id}:{agent_key}"


def release_choices(con, run_id):
    """Releases of the catalog run with their status. Any release can host an episode; one that is not validated is tagged as such everywhere."""
    from . import release as release_module
    out = []
    for release_id, label, created in con.execute("SELECT release_id,label,created_at FROM release WHERE run_id=%s ORDER BY created_at DESC", (run_id,)).fetchall():
        manifest = con.execute("SELECT manifest FROM release WHERE release_id=%s", (release_id,)).fetchone()[0]
        out.append({"release_id": str(release_id), "label": label, "status": release_module.status(con, release_id), "created_at": created.isoformat(timespec="seconds"),
                    "has_ingestion": bool(manifest["runs"].get("ingestion_run_id")), "has_signals": bool(manifest["runs"].get("signal_run_id"))})
    return out


def embedding_choices(con, release_id):
    """Embedding models that have vectors for the release's chunks (a model without them would silently leave the vector side out of the search)."""
    ingestion_id, _ = rag.release_scope(con, release_id, allow_unvalidated=True)
    return [{"model_id": m, "dimensions": d, "chunks": n} for m, d, n in con.execute(
        "SELECT e.model_id,m.dimensions,count(*) FROM chunk_embedding e JOIN embedding_model m USING (model_id) JOIN document_chunk c USING (chunk_id) JOIN paper_version v USING (version_id) "
        "WHERE v.ingestion_run_id=%s GROUP BY 1,2 ORDER BY 3 DESC", (ingestion_id,)).fetchall()]


# The retrieval strata (signals-v4) and triage hints (signals-v3) that can stand in for a decision in the signals-only option -> the class an agent is told
# the pair is a candidate for. Only these three say something a construct problem can start from.
CATEGORY_HINTS = {"conceptual_jingle": ("same_label_divergent", "jingle_candidate"), "conceptual_jangle": ("different_label_close", "jangle_candidate"),
                  "unresolved": ("unresolved",)}
# Which pairs of a stratum come first: the clearest by the signal it reads (S_def in v4, S_sem-op in v3). A jingle candidate is the same label with the most distant
# meanings, a jangle candidate different labels with the closest ones; unresolved has no meaning to compare, so the pairs whose papers cite most alike.
HINT_ORDER = {"conceptual_jingle": "coalesce(p.s_def, p.s_sem_op) ASC", "conceptual_jangle": "coalesce(p.s_def, p.s_sem_op) DESC", "unresolved": "p.c_cit DESC NULLS LAST"}
HINT_PAIRS_PER_CATEGORY = 20


def hint_pair_ids(con, release_id):
    """{category: [pair_id]}: for each hint, the clearest HINT_PAIRS_PER_CATEGORY pairs of the release's signal run that NO person has decided on (any decision,
    even a contested or single one, wins over a hint). Deterministic: the signal run is immutable and ties break on pair_id."""
    signal_run_id = rag.release_scope(con, release_id, allow_unvalidated=True)[1]["runs"].get("signal_run_id")
    out = {}
    if signal_run_id:
        for category, hints in CATEGORY_HINTS.items():
            out[category] = [r[0] for r in con.execute(
                f"SELECT p.pair_id FROM candidate_pair p WHERE p.signal_run_id=%s AND p.hint = ANY(%s) AND NOT EXISTS (SELECT 1 FROM pair_decision d WHERE d.pair_id=p.pair_id) "
                f"ORDER BY {HINT_ORDER[category]}, p.pair_id LIMIT %s", (signal_run_id, list(hints), HINT_PAIRS_PER_CATEGORY)).fetchall()]
    con.commit()
    return out


def prepare_hint_dossiers(con, release_id):
    """Builds the dossiers of the hint pairs that lack one (the signal run only has them for the pairs somebody sampled). `con` must be the application's
    connection: the agents read through a read-only one. A dossier is derived evidence and nothing else is written; no decision is touched."""
    from . import signals
    run = rag.release_scope(con, release_id, allow_unvalidated=True)[1]["runs"].get("signal_run_id")
    ids = [p for found in hint_pair_ids(con, release_id).values() for p in found]
    created = signals.build_dossiers(con, uuid.UUID(run), pair_ids=ids) if run and ids else 0
    con.commit()
    return created


def diagnosis_pool(con, release_id, provisional=False, pipeline_hints=False):
    """Settled pairs of the release as the agents see them, numbered P1.. in a fixed order. Reads through `con` (it commits the read transaction).
    `provisional` also takes the pairs one reviewer decided and nobody confirmed (derogation chosen per episode); those carry `provisional: True`.
    `pipeline_hints` adds pairs that only the automatic signals labelled and no person decided (derogation chosen per episode, `prepare_hint_dossiers` must have run);
    those carry `pipeline: True`, relation `not_reviewed` and the category of their hint."""
    pool = []
    for item in rag.settled_pairs(con, release_id, allow_unvalidated=True, provisional=provisional):
        d = item["dossier"]
        pool.append({"pair_id": item["pair_id"], "relation": item["class"], "flag": None, "category": item["class"], "provisional": item["provisional"],
                     "pipeline": False, "label_a": d["a"]["label"], "label_b": d["b"]["label"], "paper_a": d["a"]["paper"], "paper_b": d["b"]["paper"], "dossier": d,
                     "rationale": item.get("rationale") or [], "checklist": item.get("checklist") or {}})
    if pipeline_hints:
        by_pair = {str(pid): category for category, found in hint_pair_ids(con, release_id).items() for pid in found}
        for pair_id, content in con.execute("SELECT pair_id,content FROM pair_dossier WHERE pair_id = ANY(%s)", (list(by_pair),)).fetchall():
            pool.append({"pair_id": str(pair_id), "relation": "not_reviewed", "flag": by_pair[str(pair_id)], "category": by_pair[str(pair_id)], "provisional": False, "pipeline": True,
                         "label_a": content["a"]["label"], "label_b": content["b"]["label"], "paper_a": content["a"]["paper"], "paper_b": content["b"]["paper"], "dossier": content,
                         "rationale": []})
        con.commit()
    pool.sort(key=lambda x: x["pair_id"])
    for number, pair in enumerate(pool, 1):
        pair["id"] = f"P{number}"
    return pool


def _check_name(created_by):
    created_by = (created_by or "").strip()
    if not created_by or len(created_by) > 80:
        raise ValueError("Enter your name (up to 80 characters)")
    return created_by


def _release_facts(con, release_id):
    """What every episode on the release is checked against and frozen with, read once (a batch prepares many episodes on the same release)."""
    from . import release as release_module
    ingestion_id, manifest = rag.release_scope(con, release_id, allow_unvalidated=True)   # any release: the episode is tagged with its status
    status = release_module.status(con, release_id)
    basis = None
    if manifest["runs"].get("preassignment_run_id"):
        basis = con.execute("SELECT source FROM preassignment_run WHERE preassignment_run_id=%s", (manifest["runs"]["preassignment_run_id"],)).fetchone()
        basis = basis[0] if basis else None
    label = con.execute("SELECT label FROM release WHERE release_id=%s", (release_id,)).fetchone()[0]
    return {"manifest": manifest, "embedding_models": {c["model_id"] for c in embedding_choices(con, release_id)}, "pools": {},
            "release": {"label": label, "status": status, "validated": status == "validated", "community_basis": basis}}


def _prepare_episode(con, release_id, raw_config, models, facts):
    """Validate one configuration against the release and return it as it will be frozen. Writes no episode (the hint dossiers it may build are derived data)."""
    raw_config = raw_config or {}
    config = proto.normalize_config(raw_config, models)
    if config["analysis_tools"] != "none" and not sandbox.enabled():
        raise ValueError(f"Code execution is off on this server: agents run Python as this server's own user, which is only acceptable on a machine you trust. Set {sandbox.ENV_FLAG}=1 to allow it")
    spec, manifest = proto.CONDITIONS[config["condition"]], facts["manifest"]
    if config["retrieval"]["embedding_model"] not in facts["embedding_models"]:
        raise ValueError("The embedding model has no vectors for this release's chunks: choose one that does (embed-chunks)")
    if spec["start"] == "constructs" and not manifest["runs"].get("extraction_run_id"):
        raise ValueError("This release has no extraction run, so there is no construct database for the conditions that start from it (A, B, C)")
    construct_problem = proto.trigger_of(config) == "construct_problem"
    picked = " (the seed picked it: choose another trigger or seed)" if config["trigger"]["choice"] == "random" else ""
    if construct_problem and not manifest["runs"].get("signal_run_id"):
        raise ValueError("This release has no signal run, so there are no construct relationships for the construct-problem trigger" + picked)
    provisional, hints = config["diagnoses"]["provisional"], config["diagnoses"]["pipeline_hints"]
    if construct_problem and (provisional, hints) not in facts["pools"]:
        if hints:
            prepare_hint_dossiers(con, release_id)       # before the pool is read: the agents' connection is read-only and could not build them
        facts["pools"][(provisional, hints)] = bool(diagnosis_pool(con, release_id, provisional, hints))
    if construct_problem and not facts["pools"][(provisional, hints)]:
        raise ValueError("The release has no construct relationship decided by people for the construct-problem trigger" + picked
                         + ("" if provisional and hints else
                            " (in the trigger settings you can allow relationships decided by a single reviewer, or pairs labelled only by the automatic signals)"))
    if raw_config.get("agents", {}) and set(raw_config["agents"]) - set(config["agents"]):
        raise ValueError("Unknown agent: " + ", ".join(sorted(set(raw_config["agents"]) - set(config["agents"]))))
    alone = (" (" + config["protocol"]["reasoners"][0].upper() + ")") if spec["interaction"] == "isolated" else ""       # B: which specialist reasoned
    config["title"] = config["title"] or f"Condition {config['condition']} - {spec['name']}{alone}"
    config["release"] = dict(facts["release"])      # frozen: what the release was when the episode was made
    return config


def create_episode(con, release_id, raw_config, created_by, models=None):
    """Validate the configuration against the release, freeze it and register the agents. Calls no model. Returns the episode id."""
    created_by = _check_name(created_by)
    config = _prepare_episode(con, release_id, raw_config, models or load_models(), _release_facts(con, release_id))
    episode_id = _insert_episode(con, release_id, config, created_by)
    con.commit()
    return episode_id


def _insert_episode(con, release_id, config, created_by):
    """Register a prepared episode and its agents, without committing."""
    episode_id = uuid.uuid4()
    con.execute("INSERT INTO forum_episode(episode_id,release_id,condition,title,provider,seed,config,config_sha256,budgets_sha256,prompt_version,created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (episode_id, release_id, config["condition"], config["title"], config["provider"], config["seed"], Jsonb(config), proto.config_checksum(config), proto.budgets_checksum(config),
                 config["prompt_version"], created_by))
    for agent in proto.active_agents(config["condition"], config["protocol"]):
        pid = profile_id(episode_id, agent["key"])
        con.execute("INSERT INTO forum_agent(episode_id,agent_key,label,community,role,profile_id,settings) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (episode_id, agent["key"], agent["label"], agent["community"], agent["role"], pid, Jsonb(config["agents"][agent["key"]])))
        con.execute("INSERT INTO agent_profile(agent_id,community,retrieval_policy,system_prompt_version) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (pid, agent["community"], Jsonb({"max_per_paper": config["retrieval"]["max_per_paper"], "episode_id": str(episode_id), "condition": config["condition"],
                                                     "trigger": config["trigger"]["key"]}), config["prompt_version"]))
    return episode_id


def create_batch(con, release_id, raw_config, created_by, repetitions, seed_mode="vary", models=None):
    """The arms of the design (0, A, B with the CTO specialist, B with the TIM specialist, C) under the same settings, `repetitions` times (proto.batch_configs). Every configuration is validated before any episode is
    written, and all of them are written in one transaction: a batch exists whole or not at all. Each episode freezes `config.batch` (id, repetition, position) - the
    grouping lives in the frozen configuration, so it needs no table of its own. Calls no model. Returns (batch id, episode ids in the order they run)."""
    created_by = _check_name(created_by)
    raw_config = raw_config or {}
    models, facts = models or load_models(), _release_facts(con, release_id)
    plan = proto.batch_configs(raw_config, repetitions, seed_mode)
    batch_id, total = uuid.uuid4(), len(plan)
    base = proto._text(raw_config.get("title"), "title", 80) or f"Batch {str(batch_id)[:8]}"
    reps = plan[-1][0]
    prepared = []
    for position, (repetition, condition, raw) in enumerate(plan, 1):
        try:
            config = _prepare_episode(con, release_id, raw, models, facts)
        except ValueError as exc:
            raise ValueError(f"Repetition {repetition}, condition {condition} (seed {raw['seed']}): {exc}")
        arm = proto.arm_of(condition, config["protocol"])
        alone = " " + config["protocol"]["reasoners"][0].upper() if arm != condition else ""
        config["title"] = f"{base} - rep {repetition}/{reps} - {condition} {proto.CONDITIONS[condition]['short']}{alone}"
        config["batch"] = {"id": str(batch_id), "repetition": repetition, "repetitions": reps, "position": position, "total": total, "seed_mode": seed_mode, "title": base,
                           "arm": arm}
        prepared.append(config)
    con.commit()            # the reads above (the diagnosis pools commit on their own); nothing of the batch is written yet
    try:
        ids = [_insert_episode(con, release_id, config, created_by) for config in prepared]
        con.commit()
    except Exception:
        con.rollback()
        raise
    return batch_id, ids


def batch_episodes(con, batch_id):
    """[(episode id, status, condition, provider)] of a batch in the order it runs."""
    return con.execute("SELECT episode_id,status,condition,provider FROM forum_episode WHERE config->'batch'->>'id'=%s ORDER BY (config->'batch'->>'position')::int",
                       (str(batch_id),)).fetchall()


def batch_detail(con, batch_id):
    """The batch as a grid: one row per repetition, one cell per arm (B once for each specialist), with what each episode produced and the mean of the three votes of its proposals
    (forum_scoring.py), and per arm the totals. Proposals of the earlier format are not counted (they are not shown)."""
    rows = con.execute(f"SELECT {EPISODE_COLUMNS} FROM forum_episode WHERE config->'batch'->>'id'=%s ORDER BY (config->'batch'->>'position')::int", (str(batch_id),)).fetchall()
    if not rows:
        raise ValueError("Unknown batch")
    episodes = []
    for row in rows:
        e = _episode_dict(row)
        e["batch"], e["budgets"], e["usage"], e["cost_usd"] = e["config"]["batch"], e["config"]["budgets"], usage(con, row[0]), cost(con, row[0])
        proposals = [str(p) for (p,) in con.execute(f"SELECT p.proposal_id FROM forum_proposal p WHERE p.episode_id=%s AND {fscore.NEW_FORMAT_SQL}", (row[0],)).fetchall()]
        means = [r["mean"] for r in fscore.latest_for(con, proposals).values() if r["mean"] is not None]
        e["questions"] = con.execute("SELECT count(*) FROM question WHERE episode_id=%s", (row[0],)).fetchone()[0]
        e["proposals"], e["evaluated"] = len(proposals), len(means)
        e["mean"] = round(sum(means) / len(means), 2) if means else None
        e["seed"] = e["config"]["seed"]
        e["lead"] = e["config"]["protocol"].get("lead") if proto.CONDITIONS[e["condition"]].get("encounter") else None
        e["arm"] = proto.arm_of(e["condition"], e["config"]["protocol"])
        del e["config"]
        episodes.append(e)
    first = _episode_dict(rows[0])["config"]
    by_arm = {}
    for e in episodes:
        c = by_arm.setdefault(e["arm"], {"episodes": 0, "completed": 0, "questions": 0, "proposals": 0, "means": [], "tokens": 0})
        c["episodes"] += 1
        c["completed"] += e["status"] == "completed"
        c["questions"] += e["questions"]
        c["proposals"] += e["proposals"]
        c["tokens"] += e["usage"]["tokens"]
        if e["mean"] is not None:
            c["means"].append(e["mean"])
    for c in by_arm.values():
        means = c.pop("means")
        c["rated_episodes"], c["mean"] = len(means), round(sum(means) / len(means), 2) if means else None
    return {"batch_id": str(batch_id), "title": first["batch"]["title"], "repetitions": first["batch"]["repetitions"], "seed_mode": first["batch"]["seed_mode"],
            "budgets": first["budgets"], "trigger_choice": first["trigger"]["choice"], "created_by": episodes[0]["created_by"],
            "created_at": episodes[0]["created_at"], "episodes": episodes, "arms": list(dict.fromkeys(e["arm"] for e in episodes)), "by_arm": by_arm,
            "statuses": {s: sum(e["status"] == s for e in episodes) for s in {e["status"] for e in episodes}}}


def mark_orphans(con):
    """An episode left 'running' by a server that stopped can never finish: say so instead of showing a live badge forever."""
    n = con.execute("UPDATE forum_episode SET status='failed', status_detail='The server stopped while the episode was running', finished_at=now() WHERE status IN ('running','stopping')").rowcount
    con.commit()
    return n


# --- Running an episode ------------------------------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, episode_id, provider=None, app_url=None, agent_url=None, stop=None, on_activity=None, models=None):
        self.episode_id = episode_id
        self.stop = stop or threading.Event()
        self.on_activity = on_activity or (lambda text: None)
        self.models = models or load_models()
        self.app = psycopg.connect(app_url or os.environ["DATABASE_URL"], autocommit=True)
        self.agent_con = rag.reader_connection(agent_url or os.environ.get("AGENT_DATABASE_URL") or app_url or os.environ["DATABASE_URL"])
        row = self.app.execute("SELECT release_id,condition,provider,seed,config FROM forum_episode WHERE episode_id=%s", (episode_id,)).fetchone()
        if not row:
            raise ValueError("Unknown episode")
        self.release_id, self.condition, provider_name, self.seed, self.config = row
        self.provider = provider or get_provider(provider_name)
        self.spec = proto.CONDITIONS[self.condition]
        if self.spec.get("legacy"):
            raise ValueError(proto.LEGACY_NOTICE + " Episodes made with them can be read, not run.")
        self.trigger = proto.trigger_of(self.config)            # None for an episode created before the triggers (13 Oct 2026)
        # The construct-problem trigger hands the adjudicated construct relationships (dossiers) to the organized conditions; condition 0 gets the raw passages of the
        # same papers instead (matched opportunities, no map). With any other trigger nobody gets them.
        self.diagnoses = self.trigger == "construct_problem" and self.spec["organized"]
        self.matched_raw = self.trigger == "construct_problem" and not self.spec["organized"]
        self.analysis = self.config.get("analysis_tools", "none")
        if self.analysis != "none" and not sandbox.enabled():
            raise ValueError(f"Code execution is off on this server ({sandbox.ENV_FLAG} is not set)")
        self.code = self.config.get("code") or {}
        self.proposal = self.config.get("proposal") or {"enabled": False}          # an episode made before proposals existed has none
        self.agents = [dict(a, profile_id=profile_id(episode_id, a["key"])) for a in proto.active_agents(self.condition, self.config["protocol"])]
        if self.spec["interaction"] == "isolated" and len(self.agents) > 1:
            raise ValueError("This B episode was made with both specialists, each with half of the budget, which was removed on 14 Oct 2026: it can be read, not run. "
                             "Copy its settings into a new episode with one specialist, or run a batch, which runs B once for each")
        self.by_key = {a["key"]: a for a in self.agents}
        self.budgets = self.config["budgets"]
        self.rng = random.Random(self.seed)
        self.embedder = get_embedder(self.config["retrieval"]["embedding_model"])
        self.seq = (self.app.execute("SELECT coalesce(max(seq)+1,0) FROM forum_message WHERE episode_id=%s", (episode_id,)).fetchone()[0])
        self.t0 = time.monotonic()
        self.tokens = self.calls = self.retrievals = self.passages = 0
        self.evidence, self.chunk_ids = {}, {}          # E-id -> hit, chunk_id -> E-id
        self.pool, self.visible, self.read_pairs = {}, {}, set()
        self.premises = {}                  # agent key -> the P-ids handed over to it (the same set for everybody in a discussion)
        self.opening = None                 # what the encounter opened on, for the question archive: {"agent", "text", "citations"} or the drawn theme
        self.has_records = False
        self.extraction_run = None          # the construct database of the release, for the tools of A, B and C
        self.views = {}                     # agent key -> the papers of its view (str canonical ids)
        self.code_runs = 0                  # numbers the runs A1, A2... for the whole episode, whoever ran them
        self.theme_mode = (self.config.get("encounter") or {}).get("theme_mode", "random_passages")      # an episode made before the choice ran on random passages
        # Budgets are spent from one ledger for the whole episode, whoever spends them (until 14 Oct 2026 the two specialists of B had half each, in ledgers of their own).
        b = self.budgets
        self.caps = {"tokens": b["max_total_tokens"], "calls": b["max_llm_calls"], "retrievals": b["max_retrieval_calls"], "passages": b["max_passages"],
                     "pairs": b["max_pairs"], "code_runs": (self.config.get("code") or {}).get("max_runs", 0)}
        self.used = {"tokens": 0, "calls": 0, "retrievals": 0, "passages": 0, "code_runs": 0}
        self.runs = {}                      # A-id -> {"ok", "purpose", "code_sha256", "agent"}
        self.web, self.web_urls = {}, {}    # W-id -> {"url", "title", "agent", "round"} that a web search returned, url -> W-id (only with external_access open)
        self.web_on = self.config.get("external_access") == "open"
        self.today = datetime.date.today().isoformat()      # the agents are told the date, to reason about how recent a study is
        self.seen = {}                      # agent key -> E-ids that agent retrieved
        self.data_files = {}                # agent key -> {name: (bytes, description)}
        self.questions = []                 # (question_id, agent key, content) in the order they were written
        self.questions_by_author = {}
        self.proposal_ids = []              # the proposals written in this run, for the blind evaluation that follows
        self.auto_evaluate = os.environ.get("FORUM_AUTO_EVALUATE", "1").strip().lower() not in ("0", "false", "no")
        self.notes = {a["key"]: "" for a in self.agents}
        self.reserve_tokens = proto.synthesis_reserve(self.config["agents"], self.config["protocol"], self.spec, self.agents, self.proposal)
        self.reserve_calls = 1 if self.spec["interaction"] == "discussion" else len(self.agents)
        if self.proposal["enabled"]:
            self.reserve_calls += self.reserve_calls * (proto.proposal_calls(self.spec) + proto.CARD_CALLS)

    # -- storage --
    def emit(self, kind, agent_key=None, content="", data=None, audience="all", round_number=0, model=None, result=None, counts=None):
        data = dict(data or {})
        if counts:
            data["counts"] = counts
        seq, self.seq = self.seq, self.seq + 1
        self.app.execute(
            "INSERT INTO forum_message(episode_id,seq,agent_key,kind,audience,round,content,data,model,input_tokens,output_tokens,reasoning_tokens,llm_calls,latency_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (self.episode_id, seq, agent_key, kind, audience, round_number, content, Jsonb(data), model,
             getattr(result, "input_tokens", 0), getattr(result, "output_tokens", 0), getattr(result, "reasoning_tokens", 0), getattr(result, "llm_calls", 0), getattr(result, "latency_ms", None)))
        return seq

    def set_status(self, status, detail=None):
        self.app.execute("UPDATE forum_episode SET status=%s, status_detail=%s, started_at=coalesce(started_at, CASE WHEN %s='running' THEN now() END), "
                         "finished_at=CASE WHEN %s IN ('completed','stopped','failed','budget_exhausted') THEN now() END WHERE episode_id=%s",
                         (status, detail, status, status, self.episode_id))

    # -- budgets --
    def ledger(self, agent=None):
        """(spent, limits, reserve tokens, reserve calls) of the episode's budget, the same for every agent."""
        return self.used, self.caps, self.reserve_tokens, self.reserve_calls

    def gate(self, reserve=True, agent=None):
        if self.stop.is_set():
            raise Cancelled()
        b = self.budgets
        elapsed = time.monotonic() - self.t0
        if elapsed > b["max_wall_seconds"] + (0 if reserve else SYNTHESIS_GRACE_SECONDS):
            raise BudgetExceeded("the time limit was reached")
        used, cap, reserve_tokens, reserve_calls = self.ledger(agent)
        if used["calls"] + (reserve_calls if reserve else 0) >= cap["calls"]:
            raise BudgetExceeded("the limit on model calls was reached")
        if used["tokens"] + (reserve_tokens if reserve else 0) >= cap["tokens"]:
            raise BudgetExceeded("the token limit was reached")

    def on_usage(self, usage, agent=None):
        spent = usage["input_tokens"] + usage["output_tokens"]
        used = self.ledger(agent)[0]
        used["tokens"] += spent
        used["calls"] += 1
        self.tokens += spent
        self.calls += 1

    def clamp(self, wanted, reserve=True, agent=None):
        used, cap, reserve_tokens, _ = self.ledger(agent)
        room = cap["tokens"] - used["tokens"] - (reserve_tokens if reserve else 0)
        if room < 300:
            raise BudgetExceeded("the token limit was reached")
        return min(wanted, room)

    def left(self, agent):
        used, cap, reserve_tokens, *_ = self.ledger(agent)
        out = {"tokens": max(0, cap["tokens"] - used["tokens"] - reserve_tokens), "searches": max(0, cap["retrievals"] - used["retrievals"]),
               "passages": max(0, cap["passages"] - used["passages"])}
        if self.diagnoses:
            out["dossiers"] = max(0, cap["pairs"] - len(self.read_pairs))
        if self.analysis != "none":
            out["code_runs"] = max(0, cap["code_runs"] - used["code_runs"])
        return out

    def spend(self, agent, retrievals=0, passages=0):
        """Charge searches and new passages to the agent's ledger (and to the episode's totals)."""
        used = self.ledger(agent)[0]
        used["retrievals"] += retrievals
        used["passages"] += passages
        self.retrievals += retrievals
        self.passages += passages

    def passages_left(self, agent):
        used, cap = self.ledger(agent)[:2]
        return cap["passages"] - used["passages"]

    def searches_spent(self, agent):
        used, cap = self.ledger(agent)[:2]
        return used["retrievals"] >= cap["retrievals"] or used["passages"] >= cap["passages"]

    # -- tools --
    def tools_for(self, agent):
        tools = []
        if self.budgets["max_retrieval_calls"] > 0 and self.budgets["max_passages"] > 0:
            tools.append(ToolSpec("search_corpus", "Search the full-text passages of your corpus (hybrid keyword and semantic search). Returns passages with ids E1, E2, ... that you can cite.",
                                  {"type": "object", "properties": {"query": {"type": "string", "description": "What to look for, in plain words (up to 500 characters)"},
                                                                    "k": {"type": "integer", "description": "How many passages (default and maximum come from the episode settings)"},
                                                                    "section": {"type": "string", "description": "Optional: only this section, for example method, results or discussion"}},
                                   "required": ["query"]}))
        if self.spec["organized"] and self.has_records:
            tools.append(ToolSpec("search_constructs", "Search the construct database: the constructs, measures, nomological claims and attributes extracted from the papers, each with its evidence quote. "
                                  "Returns records with ids E#, citable like passages. Use it to see how a construct is defined, measured and related to others across papers.",
                                  {"type": "object", "properties": {"query": {"type": "string", "description": "A construct, a relation or a measure, in plain words"},
                                                                    "kind": {"type": "string", "enum": ["mention", "measure", "claim", "attribute"], "description": "Optional: only this kind of record"},
                                                                    "k": {"type": "integer"}}, "required": ["query"]}))
        if self.extraction_run and self.budgets["max_retrieval_calls"] > 0:
            tools.append(ToolSpec("construct_overview", "The map of the construct database in your view: for each construct (by label) the papers that use it, its distinct antecedents and "
                                  "outcomes, its other links, and how often it moderates or mediates a claim of others. Sort by papers (most used), links (most connected) or thin (used "
                                  "in at least two papers, fewest links). Counts are what the papers claim, not a sample of the world. Counts as one search.",
                                  {"type": "object", "properties": {"contains": {"type": "string", "description": "Optional: only labels containing these words"},
                                                                    "sort": {"type": "string", "enum": list(fcon.SORTS)},
                                                                    "limit": {"type": "integer", "description": "Rows, default 15, at most 30"}}, "required": []}))
            tools.append(ToolSpec("construct_network", "The network of one construct across the papers of your view: its antecedents, outcomes and other links (papers, claims, signs, "
                                  "evidence status), the moderators and mediators of its claims, the claims it moderates or mediates, and its definitions. The strongest links come "
                                  "with an example record E# you can cite, as many as one search returns. Counts as one search.",
                                  {"type": "object", "properties": {"construct": {"type": "string", "description": "The construct label, as construct_overview or search_constructs show it"},
                                                                    "limit": {"type": "integer", "description": "Links per group, default 6, at most 12"}}, "required": ["construct"]}))
        if self.diagnoses and self.visible.get(agent["key"]):
            categories = sorted({self.pool[p]["category"] for p in self.visible[agent["key"]]})
            tools.append(ToolSpec("list_diagnoses", "List the construct-problem dossiers: relationships between two construct uses that human reviewers decided in the release "
                                  "(a conceptual jingle or jangle, or another relationship such as related but distinct or unresolved).",
                                  {"type": "object", "properties": {"category": {"type": "string", "enum": categories}, "limit": {"type": "integer"}}, "required": []}))
            tools.append(ToolSpec("read_dossier", "Read one construct-problem dossier: the reviewers' rationale, the contrary evidence, and both uses of the constructs with definitions, "
                                  "measures, claims and pages. Counts against your dossier budget.",
                                  {"type": "object", "properties": {"id": {"type": "string", "description": "A dossier id such as P3"}}, "required": ["id"]}))
        if self.analysis != "none" and self.ledger(agent)[0]["code_runs"] < self.ledger(agent)[1]["code_runs"]:
            tools.append(ToolSpec("run_python", self.python_description(agent),
                                  {"type": "object", "properties": {"purpose": {"type": "string", "description": "One sentence: what this run is meant to show"},
                                                                    "code": {"type": "string", "description": "A complete Python 3 program. Read from data/, print what matters, save files to out/"}},
                                   "required": ["purpose", "code"]}))
        return tools

    def python_description(self, agent):
        files = "; ".join(f"data/{name}: {d['rows']} rows ({', '.join(d['columns'])})" if d.get("columns") else f"data/{name}" for name, (_, d) in self.data_for(agent).items())
        return (f"Run a Python 3 program and read what it prints. Libraries: {', '.join(sandbox.available_libraries()) or 'the standard library only'}. No network, no other files; "
                f"each run is a fresh process (nothing is remembered between runs) with a limit of {self.code['timeout_seconds']} s and {self.code['memory_mb']} MB. Save what you want kept "
                f"(png, csv, txt, json) in out/. Each run gets an id A1, A2, ... that you can cite. The tables hold what the papers say (extracted by a model, not validated): they are not "
                f"a sample of the world. Files: {files}.")

    def view(self, agent):
        """The papers of the agent's view (its community, or both for an agent without one), as text ids, read once per episode."""
        if agent["key"] not in self.views:
            manifest = rag.release_scope(self.agent_con, self.release_id, True)[1]
            communities = [agent["community"]] if agent["community"] else ["CTO", "TIM"]
            self.views[agent["key"]] = sorted({str(p) for c in communities for p in rag.community_papers(self.agent_con, manifest, c, True)})
            self.agent_con.commit()
        return self.views[agent["key"]]

    @staticmethod
    def record_hit(f, score=0.0):
        """A record of the construct database shaped like a passage, so that it gets an E-id and is cited like one."""
        return {"chunk_id": f["record_id"], "canonical_id": f["canonical_id"], "title": f["title"], "year": f["year"], "doi": f["doi"], "authors": f.get("authors"),
                "journal": f.get("journal"), "section": "construct " + f["record_kind"], "pages": [f["page"] or 0, f["page"] or 0], "text": f["text"], "score": score, "char_span": None, "record_kind": f["record_kind"]}

    def data_for(self, agent):
        """{name: (bytes, description)}: the tables built for this agent at the start, and the passages it has retrieved so far."""
        files = dict(self.data_files.get(agent["key"], {}))
        rows = [[i, h["title"], h["year"], h["section"], h["pages"][0], h["pages"][1], h["text"]] for i in sorted(self.seen.get(agent["key"], ()), key=lambda i: int(i[1:])) for h in [self.evidence[i]]]
        files["passages.csv"] = sandbox.csv_file(["id", "title", "year", "section", "page_from", "page_to", "text"], rows)
        return files

    def build_data(self, agent):
        """The tables of the agent's view of the release, as CSV: papers, extracted constructs, measures, claims, and the claims' mediators and moderators (A, B, C) or the
        diagnoses it can see (D, E, F). Condition 0 gets none of them (it has no database by design) and only the passages it retrieves."""
        files = {}
        if not self.spec["organized"]:
            return files
        _, manifest = rag.release_scope(self.agent_con, self.release_id, True)
        communities = [agent["community"]] if agent["community"] else ["CTO", "TIM"]
        papers = sorted({str(p) for c in communities for p in rag.community_papers(self.agent_con, manifest, c, True)})
        run = manifest["runs"].get("extraction_run_id")
        q = lambda sql, *params: self.agent_con.execute(sql, params).fetchall()
        files["papers.csv"] = sandbox.csv_file(["canonical_id", "title", "year", "source", "doi"], q(
            "SELECT canonical_id,title,publication_year,source_title,doi FROM canonical_paper WHERE canonical_id = ANY(%s::uuid[]) ORDER BY canonical_id", papers))
        if run:
            files["constructs.csv"] = sandbox.csv_file(["mention_id", "canonical_id", "label", "surface_form", "role", "definition", "page"], q(
                "SELECT mention_id,canonical_id,label_normalized,surface_form,role,definition_text,page FROM construct_mention WHERE extraction_run_id=%s AND canonical_id = ANY(%s::uuid[]) "
                "ORDER BY mention_id", run, papers))
            files["measures.csv"] = sandbox.csv_file(["measure_id", "mention_id", "canonical_id", "name", "items", "scale", "page"], q(
                "SELECT m.measure_id,m.mention_id,c.canonical_id,m.name,m.items_text,m.scale,m.page FROM construct_measure m JOIN construct_mention c USING (mention_id) "
                "WHERE c.extraction_run_id=%s AND c.canonical_id = ANY(%s::uuid[]) ORDER BY m.measure_id", run, papers))
            files["claims.csv"] = sandbox.csv_file(["claim_id", "canonical_id", "subject", "object", "relation", "direction", "evidence_status", "confidence", "page"], q(
                "SELECT k.claim_id,k.canonical_id,s.label_normalized,o.label_normalized,k.relation,k.direction,k.evidence_status,k.confidence,k.page FROM nomological_claim k "
                "JOIN construct_mention s ON s.mention_id=k.subject_mention_id JOIN construct_mention o ON o.mention_id=k.object_mention_id "
                "WHERE k.extraction_run_id=%s AND k.canonical_id = ANY(%s::uuid[]) ORDER BY k.claim_id", run, papers))
            if self.spec["start"] == "constructs":      # the mediators and moderators of the claims: the links A, B and C start from
                files["modifiers.csv"] = sandbox.csv_file(["claim_id", "canonical_id", "kind", "construct", "value"], q(
                    "SELECT m.claim_id,k.canonical_id,m.kind,x.label_normalized,m.value FROM claim_modifier m JOIN nomological_claim k USING (claim_id) "
                    "LEFT JOIN construct_mention x ON x.mention_id=m.mention_id WHERE k.extraction_run_id=%s AND k.canonical_id = ANY(%s::uuid[]) "
                    "ORDER BY m.claim_id,m.kind,x.label_normalized,m.value", run, papers))
        self.agent_con.commit()
        if not self.diagnoses:
            return files
        rows = []
        for pid in self.visible.get(agent["key"], []):
            d = self.pool[pid]
            sig = (d["dossier"].get("signals") or {})
            rows.append([pid, d["category"], d["relation"], d["flag"], "no" if d["pipeline"] else ("one reviewer" if d["provisional"] else "yes"), d["label_a"], d["label_b"],
                         d["paper_a"]["title"], d["paper_a"]["year"], d["paper_b"]["title"], d["paper_b"]["year"], sig.get("s_sem_op"), sig.get("c_cit"), sig.get("n_nom")])
        files["diagnoses.csv"] = sandbox.csv_file(["id", "category", "relation", "flag", "reviewed_by_people", "label_a", "label_b", "paper_a", "year_a", "paper_b", "year_b",
                                                   "s_sem_op", "c_cit", "n_nom"], rows)
        return files

    def register_hits(self, hits, allowance, agent=None):
        """Give every hit its E-id. `new` counts what is new to the budget: a passage nobody had seen."""
        out, new = [], 0
        for hit in hits:
            known = self.chunk_ids.get(hit["chunk_id"])
            costs = known is None
            if costs and new >= allowance:
                continue
            if known is None:
                known = f"E{len(self.evidence) + 1}"
                self.chunk_ids[hit["chunk_id"]] = known
                self.evidence[known] = hit
            new += costs
            out.append((known, hit))
        return out, new

    def passage_view(self, evidence_id, hit, chars):
        return {"id": evidence_id, "title": hit["title"], "year": hit["year"], "section": hit["section"], "pages": hit["pages"], "text": hit["text"][:chars]}

    def dispatch(self, agent, name, args):
        """(text for the model, data to store). Errors that the model can act on are returned as text, never raised."""
        cfg = self.config["retrieval"]
        if name == "search_corpus":
            query = str(args.get("query", "")).strip()[:500]
            if not query:
                return json.dumps({"error": "give a query"}), {"error": "empty query"}
            if self.searches_spent(agent):
                return json.dumps({"error": "your search or passage budget is spent: work with the evidence you already have"}), {"error": "search budget spent"}
            try:
                k = max(1, min(int(args.get("k") or cfg["k"]), cfg["k"]))
            except (TypeError, ValueError):
                k = cfg["k"]
            section = str(args["section"]).strip() if args.get("section") else None
            community = agent["community"] or ["CTO", "TIM"]
            try:
                hits = rag.search(self.agent_con, self.release_id, query, self.embedder, k=k, sections=[section] if section else None, agent=agent["profile_id"],
                                  community=community, max_per_paper=cfg["max_per_paper"], allow_unvalidated=True, include_v1=True)
            except ValueError as exc:
                return json.dumps({"error": str(exc)}), {"error": str(exc)}
            seen, new = self.register_hits(hits, self.passages_left(agent), agent)
            self.spend(agent, retrievals=1, passages=new)
            self.seen.setdefault(agent["key"], set()).update(i for i, _ in seen)
            shown = [self.passage_view(i, h, cfg["passage_chars"]) for i, h in seen]
            data = {"tool": "search_corpus", "query": query, "hits": [{**s, "chunk_id": h["chunk_id"], "canonical_id": h["canonical_id"], "doi": h["doi"], "char_span": h["char_span"],
                                                                      "score": h["score"]} for s, (_, h) in zip(shown, seen)], "new_passages": new}
            return json.dumps({"hits": shown, "left": self.left(agent)}, ensure_ascii=False), {**data, "counts": {"passages": new, "retrievals": 1}}
        if name == "search_constructs":
            query = str(args.get("query", "")).strip()[:500]
            if not query:
                return json.dumps({"error": "give a query"}), {"error": "empty query"}
            if self.searches_spent(agent):
                return json.dumps({"error": "your search or passage budget is spent: work with the evidence you already have"}), {"error": "search budget spent"}
            try:
                k = max(1, min(int(args.get("k") or cfg["k"]), cfg["k"]))
            except (TypeError, ValueError):
                k = cfg["k"]
            kind = args.get("kind") if args.get("kind") in embedding_jobs.KINDS else None
            ingestion_id, manifest = rag.release_scope(self.agent_con, self.release_id, True)
            communities = [agent["community"]] if agent["community"] else ["CTO", "TIM"]
            papers = sorted({str(p) for c in communities for p in rag.community_papers(self.agent_con, manifest, c, True)})
            found = embedding_jobs.search_records(self.agent_con, manifest["runs"]["extraction_run_id"], self.embedder.model_id, self.embedder.embed([query])[0], k, [kind] if kind else None, papers)
            self.agent_con.commit()
            hits = [self.record_hit(f, 1 - f["distance"]) for f in found]
            seen, new = self.register_hits(hits, self.passages_left(agent), agent)
            self.spend(agent, retrievals=1, passages=new)
            self.seen.setdefault(agent["key"], set()).update(i for i, _ in seen)
            shown = [self.passage_view(i, h, cfg["passage_chars"]) for i, h in seen]
            data = {"tool": "search_constructs", "query": query, "new_passages": new,
                    "hits": [{**s_, "chunk_id": h["chunk_id"], "canonical_id": h["canonical_id"], "doi": h["doi"], "char_span": None, "score": h["score"], "record_kind": h["record_kind"]} for s_, (_, h) in zip(shown, seen)]}
            return json.dumps({"records": shown, "left": self.left(agent)}, ensure_ascii=False), {**data, "counts": {"passages": new, "retrievals": 1}}
        if name in ("construct_overview", "construct_network") and self.extraction_run:
            return self.construct_tool(agent, name, args)
        if name == "list_diagnoses":
            wanted = args.get("category")
            try:
                limit = max(1, min(int(args.get("limit") or 10), 20))
            except (TypeError, ValueError):
                limit = 10
            rows = [self.pool[p] for p in self.visible[agent["key"]] if not wanted or self.pool[p]["category"] == wanted][:limit]
            listed = [{"id": p["id"], "category": p["category"], "relation": p["relation"], "flag": p["flag"], "label_a": p["label_a"], "label_b": p["label_b"],
                       "paper_a": f"{p['paper_a']['title']} ({p['paper_a']['year']})", "paper_b": f"{p['paper_b']['title']} ({p['paper_b']['year']})",
                       **({"reviewed": False, "note": "labelled by the automatic signals only; no person has reviewed this pair"} if p["pipeline"] else {})} for p in rows]
            return json.dumps({"diagnoses": listed}, ensure_ascii=False), {"tool": "list_diagnoses", "diagnoses": listed}
        if name == "read_dossier":
            pair = self.pool.get(str(args.get("id", "")))
            if not pair or pair["id"] not in self.visible[agent["key"]]:
                return json.dumps({"error": "unknown diagnosis id"}), {"error": "unknown diagnosis id"}
            new = pair["id"] not in self.read_pairs
            if new and len(self.read_pairs) >= self.ledger(agent)[1]["pairs"]:
                return json.dumps({"error": "your dossier budget is spent"}), {"error": "dossier budget spent"}
            self.read_pairs.add(pair["id"])
            text, cut = self.dossier_text(pair)
            return text, {"tool": "read_dossier", "id": pair["id"], "category": pair["category"], "labels": [pair["label_a"], pair["label_b"]], "truncated": cut,
                          "new_dossier": new, "counts": {"dossiers": 1 if new else 0}}
        if name == "run_python" and self.analysis != "none":
            return self.run_python(agent, args)
        return json.dumps({"error": f"unknown tool {name}"}), {"error": "unknown tool"}

    def dossier_text(self, pair):
        """A construct-problem dossier begins with the reviewers' rationale and the contrary evidence, without the similarity scores; a pair only the signals labelled keeps
        its scores and says that nobody reviewed it."""
        rationale = pair.get("rationale") or None
        checklist = {k: v for k, v in (pair.get("checklist") or {}).items() if v}
        if rationale and checklist:
            rationale = rationale + ["Comparison by the reviewers: " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in checklist.items())]
        return proto.compact_dossier(pair["dossier"], self.config["diagnoses"]["dossier_chars"], reviewed=not pair["pipeline"], rationale=rationale, signals=pair["pipeline"])

    def construct_tool(self, agent, name, args):
        """construct_overview and construct_network (A, B, C). Each call counts as one search; the records a network hands over get E-ids and count as passages, at most as
        many as one search returns (the strongest links first), so that exploring the database costs what searching it would."""
        if self.searches_spent(agent):
            return json.dumps({"error": "your search or passage budget is spent: work with the evidence you already have"}), {"error": "search budget spent"}

        def number(key, default, top):
            try:
                return max(1, min(int(args.get(key) or default), top))
            except (TypeError, ValueError):
                return default
        papers = self.view(agent)
        if name == "construct_overview":
            sort = args.get("sort") if args.get("sort") in fcon.SORTS else "papers"
            found = fcon.overview(self.agent_con, self.extraction_run, papers, str(args.get("contains") or "").strip()[:120] or None, sort, number("limit", 15, 30))
            self.agent_con.commit()
            self.spend(agent, retrievals=1)
            data = {"tool": name, **{k: found[k] for k in ("constructs_in_view", "matching", "sorted_by", "contains", "rows")}, "counts": {"retrievals": 1}}
            return json.dumps({**found, "left": self.left(agent)}, ensure_ascii=False), data
        construct = str(args.get("construct") or args.get("label") or "").strip()[:200]
        if not construct:
            return json.dumps({"error": "give a construct label"}), {"error": "empty construct"}
        net = fcon.network(self.agent_con, self.extraction_run, papers, construct, number("limit", 6, 12))
        if not net["found"]:
            self.agent_con.commit()
            self.spend(agent, retrievals=1)
            reply = {"construct": net["construct"], "found": False, "note": "No construct with this label in your view. Try one of the closest labels, or search_constructs.",
                     "closest": net["closest"], "left": self.left(agent)}
            return json.dumps(reply, ensure_ascii=False), {"tool": name, "construct": net["construct"], "found": False, "closest": net["closest"], "counts": {"retrievals": 1}}
        groups = ("antecedents", "outcomes", "other_links", "as_modifier")
        refs = list(net["definitions"][:1])
        for depth in range(max(len(net[g]) for g in groups)):            # the strongest link of each group first, then the second, ...
            refs += [net[g][depth]["example"] for g in groups if depth < len(net[g])]
        refs = list(dict.fromkeys(refs))[:self.config["retrieval"]["k"]]
        hits = [self.record_hit(f) for f in embedding_jobs.records(self.agent_con, self.extraction_run, refs)]
        self.agent_con.commit()
        seen, new = self.register_hits(hits, self.passages_left(agent), agent)
        self.spend(agent, retrievals=1, passages=new)
        self.seen.setdefault(agent["key"], set()).update(i for i, _ in seen)
        ids = {h["chunk_id"]: i for i, h in seen}
        cite = lambda ref: ids.get(str(ref[1]))
        for g in groups:
            for e in net[g]:
                e["example"] = cite(e["example"])
        net["definitions"] = [i for i in map(cite, net["definitions"]) if i]
        shown = [self.passage_view(i, h, self.config["retrieval"]["passage_chars"]) for i, h in seen]
        reply = {**net, "records": shown, "left": self.left(agent)}
        if len(refs) > len(seen):
            reply["records_note"] = "Some links have no record id: your passage budget did not allow more. Their counts stand; search_constructs can find their claims."
        data = {"tool": name, "construct": net["construct"], "found": True, "new_passages": new, "summary": {g: len(net[g]) for g in groups},
                "hits": [{**s_, "chunk_id": h["chunk_id"], "canonical_id": h["canonical_id"], "doi": h["doi"], "char_span": None, "score": 0.0, "record_kind": h["record_kind"]}
                         for s_, (_, h) in zip(shown, seen)], "counts": {"passages": new, "retrievals": 1}}
        return json.dumps(reply, ensure_ascii=False, default=str), data

    def run_python(self, agent, args):
        code, purpose = str(args.get("code") or ""), " ".join(str(args.get("purpose") or "").split())[:200]
        if not code.strip():
            return json.dumps({"error": "give the code to run"}), {"error": "empty code"}
        used, cap = self.ledger(agent)[:2]
        if used["code_runs"] >= cap["code_runs"]:
            return json.dumps({"error": "your code-run budget is spent: work with the results you already have"}), {"error": "code-run budget spent"}
        used["code_runs"] += 1
        self.code_runs += 1
        run_id = f"A{self.code_runs}"
        files = self.data_for(agent)
        result = sandbox.run_python(code, {n: b for n, (b, _) in files.items()}, timeout=self.code["timeout_seconds"], memory_mb=self.code["memory_mb"], cancel=self.stop)
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.runs[run_id] = {"ok": result.ok, "purpose": purpose, "code_sha256": digest, "agent": agent["key"], "stdout": result.stdout, "stderr": result.stderr}
        self.app.execute("INSERT INTO forum_analysis(episode_id,run_id,agent_key,round,purpose,code,code_sha256,inputs,environment,exit_code,timed_out,killed_for,stdout,stderr,duration_ms) "
                         "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (self.episode_id, run_id, agent["key"], getattr(self, "round_now", 0), purpose, code, digest,
                          Jsonb({n: {"rows": d["rows"], "sha256": d["sha256"]} for n, (_, d) in files.items()}), Jsonb(sandbox.environment()), result.exit_code, result.timed_out,
                          result.killed_for, result.stdout, result.stderr, result.duration_ms))
        for f in result.files:
            self.app.execute("INSERT INTO forum_analysis_file(episode_id,run_id,name,media_type,size,sha256,content) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                             (self.episode_id, run_id, f["name"], f["media_type"], f["size"], f["sha256"], f["content"]))
        kept = [{"name": f["name"], "size": f["size"]} for f in result.files]
        reply = {"run_id": run_id, "exit_code": result.exit_code, "stdout": result.stdout[:6000], "stderr": result.stderr[-2500:], "files": kept, "left": self.left(agent)}
        if result.killed_for:
            reply["stopped"] = "the run was stopped for " + result.killed_for
        if result.dropped:
            reply["not_kept"] = result.dropped
        data = {"tool": "run_python", "run_id": run_id, "purpose": purpose, "ok": result.ok, "exit_code": result.exit_code, "timed_out": result.timed_out, "killed_for": result.killed_for,
                "stdout": result.stdout[:2000], "stderr": result.stderr[-1500:], "files": kept, "duration_ms": result.duration_ms, "counts": {"code_runs": 1}}
        return json.dumps(reply, ensure_ascii=False), data

    def tool_runner(self, agent, round_number):
        def run(name, args):
            self.round_now = round_number
            self.emit("tool_call", agent["key"], f"{name}({json.dumps(args, ensure_ascii=False)[:300]})", {"tool": name, "arguments": args}, audience=agent["key"], round_number=round_number)
            text, data = self.dispatch(agent, name, args)
            counts = data.pop("counts", None)
            self.emit("tool_result", agent["key"], self.result_summary(name, data), {"tool": name, **data}, audience=agent["key"], round_number=round_number, counts=counts)
            return text
        return run

    @staticmethod
    def result_summary(name, data):
        if data.get("error"):
            return f"{name}: {data['error']}"
        if name in ("search_corpus", "search_constructs"):
            return f"{len(data['hits'])} {'records' if name == 'search_constructs' else 'passages'} ({data['new_passages']} new) for: {data['query']}"
        if name == "construct_overview":
            return f"{len(data['rows'])} of {data['matching']} constructs by {data['sorted_by']}" + (f" containing '{data['contains']}'" if data.get("contains") else "")
        if name == "construct_network":
            if not data["found"]:
                return f"no construct '{data['construct']}' in the view; closest: " + (", ".join(c["construct"] for c in data["closest"][:4]) or "none")
            n = data["summary"]
            return (f"network of {data['construct']}: {n['antecedents']} antecedents, {n['outcomes']} outcomes, {n['other_links']} other links; "
                    f"{len(data['hits'])} records ({data['new_passages']} new)")
        if name == "list_diagnoses":
            return f"{len(data['diagnoses'])} diagnoses listed"
        if name == "run_python":
            return f"{data['run_id']} {'finished' if data['ok'] else 'FAILED'}" + (f" ({data['purpose'][:90]})" if data.get("purpose") else "") + f"; {len(data['files'])} file(s) kept"
        return f"dossier {data.get('id')} ({data.get('category')}) read" + (" - truncated" if data.get("truncated") else "")

    # -- prompts --
    def visible_messages(self, agent):
        rows = self.app.execute("SELECT seq,agent_key,kind,round,content FROM forum_message WHERE episode_id=%s AND kind IN ('system','encounter','message') AND audience IN ('all',%s) ORDER BY seq",
                                (self.episode_id, agent["key"])).fetchall()
        lines = []
        for seq, who, kind, round_number, content in rows:
            if kind == "message":
                name = "You" if who == agent["key"] else self.by_key[who]["label"]
                lines.append(f"[round {round_number}] {name}: {content}")
            else:
                lines.append(f"[{'handed over' if kind == 'encounter' else 'moderator'}] {content}")
        text = "\n\n".join(lines)
        limit = self.config["protocol"]["context_chars"]
        if len(text) > limit:
            text = "[earlier messages omitted]\n\n" + text[-limit:]
        return text

    def evidence_block(self, ids, chars):
        blocks = []
        for i in ids:
            if i in self.evidence:
                h = self.evidence[i]
                blocks.append(f"[{i}] {h['title']} ({h['year']}), {h['section']}, pp. {h['pages'][0]}-{h['pages'][1]}\n{h['text'][:chars]}")
            elif i in self.pool:
                p = self.pool[i]
                blocks.append(f"[{i}] {proto.CATEGORY_LABELS.get(p['category'], p['category'])}: {p['label_a']} vs {p['label_b']}"
                              + (" (signals only, not reviewed)" if p["pipeline"] else ""))
            elif i in self.runs:
                r = self.runs[i]
                blocks.append(f"[{i}] Python run by {self.by_key[r['agent']]['label']} ({'finished' if r['ok'] else 'FAILED'}): {r['purpose']}\nIt printed:\n{r['stdout'][:1200] or '(nothing)'}"
                              + ("" if r["ok"] else f"\nError:\n{r['stderr'][-400:]}"))
            elif i in self.web:
                w = self.web[i]
                blocks.append(f"[{i}] Web source found by {self.by_key[w['agent']]['label']}: {w['title'] or '(no title)'} - {w['url']}")
        return "\n\n".join(blocks)

    def header(self, agent, round_number, step_title, step_text):
        left = self.left(agent)
        rounds = self.config["protocol"]["rounds"]
        focus = self.config.get("focus")
        return (f"ROUND {round_number} of {rounds} - {step_title}\n{step_text}\n"
                + (f"Focus of this episode: {focus}\n" if focus else "")
                + f"Budget left: about {left['tokens']} tokens, {left['searches']} searches, {left['passages']} new passages"
                + (f", {left['dossiers']} dossiers" if "dossiers" in left else "")
                + (f", {left['code_runs']} code runs." if "code_runs" in left else "."))

    def step(self, round_number):
        return proto.chain_step(self.condition, self.theme_mode, round_number, self.config["protocol"]["rounds"])

    # -- turns --
    def request(self, agent, task, model, prompt, round_number, **overrides):
        settings = self.config["agents"][agent["key"]]
        tools = overrides.pop("tools", [])
        reserve = task not in FINAL_TASKS
        self.gate(reserve, agent)
        maximum = overrides.pop("max_output_tokens")
        encounter = self.config.get("encounter") if self.condition == "C" else None
        system = overrides.pop("system", None) or proto.framing(self.condition, agent, self.analysis, encounter, self.web_on, self.today, trigger=self.trigger, dossiers=self.diagnoses,
                                                                lead=self.config["protocol"].get("lead"))
        effort = overrides.pop("reasoning_effort", settings["reasoning_effort"])
        return TurnRequest(
            model=model, system=system, prompt=prompt, task=task, tools=tools, run_tool=self.tool_runner(agent, round_number),
            gate=lambda: self.gate(reserve, agent), on_usage=lambda usage: self.on_usage(usage, agent), max_output_tokens=self.clamp(maximum, reserve, agent), reasoning_effort=effort,
            temperature=settings["temperature"], max_tool_rounds=self.budgets["max_tool_rounds"] if tools else 0,
            web_search=(self.config["external_access"] == "open" and task == "reflect"), **overrides)

    def turn(self, agent, round_number):
        key, settings = agent["key"], self.config["agents"][agent["key"]]
        step_title, step_text = self.step(round_number)
        shared = self.spec["interaction"] == "discussion"
        visible = self.visible_messages(agent)
        # The transcript comes first and the header (round, step, budget left) after it, next to the task (18 Oct 2026, the user's decision): the header changes at every call, and at the
        # head of the prompt it made every prompt differ from the first character, so the provider's prompt cache could never reuse the transcript.
        base = f"THE FORUM SO FAR:\n{visible or '(nothing yet)'}\n"
        before = set(self.evidence)
        if self.config["protocol"]["reflection"]:
            self.on_activity(f"{agent['label']} is reflecting ({settings['reflection_model']})")
            notes = f"\nYOUR PRIVATE NOTES SO FAR:\n{self.notes[key] or '(none)'}\n"
            hints = {"agent_label": agent["label"], "round": round_number, "topic": self.config.get("focus") or "digital technology organizing", "second_search": round_number > 1,
                     "wants_diagnoses": self.diagnoses, "read_dossier": round_number == 1, "search_terms": self.search_terms(agent)}
            head = self.header(agent, round_number, step_title, step_text)
            req = self.request(agent, "reflect", settings["reflection_model"], f"{base}{notes}\n{head}\n\n{proto.REFLECT_TASK}", round_number, tools=self.tools_for(agent),
                               max_output_tokens=settings["max_reflection_tokens"], hints=hints)
            result = self.provider.run(req)
            web_ids = self.register_web(result.sources, agent, round_number)
            self.notes[key] = (self.notes[key] + "\n\n" if self.notes[key] else "") + f"(round {round_number}) " + result.text
            if web_ids:
                self.notes[key] += "\nWeb sources of this search: " + ", ".join(f"[{i}]" for i in web_ids)
            self.emit("reflection", key, result.text, {"reasoning_summary": result.reasoning_summary, "sources": result.sources, "web": [{"id": i, "url": self.web[i]["url"], "title": self.web[i]["title"]} for i in web_ids], "finish": result.finish,
                                                       "citations": self.citations(result.text)},
                      audience=key, round_number=round_number, model=settings["reflection_model"], result=result)
        else:
            self.notes[key] = ""
        new_ids = [i for i in self.evidence if i not in before]
        mine = [p for p in sorted(self.read_pairs) if p in self.visible.get(key, [])]
        cited = list(dict.fromkeys(new_ids + proto.cited_ids(self.notes[key]) + proto.cited_ids(visible) + mine))
        evidence_text = self.evidence_block(cited, self.config["retrieval"]["passage_chars"])
        self.on_activity(f"{agent['label']} is writing ({settings['discussion_model']})")
        words = max(60, settings["max_message_tokens"] // 2)
        task = (proto.SPEAK_TASK if shared else proto.SOLO_SPEAK_TASK).format(words=words)
        notes_block = f"\nYOUR PRIVATE NOTES SO FAR:\n{self.notes[key] or '(none)'}\n" if self.notes[key] else ""     # the reflection's label: the same prefix
        head = self.header(agent, round_number, step_title, step_text)          # computed again: the reflection has spent part of the budget
        prompt = f"{base}{notes_block}\nEVIDENCE YOU MAY CITE:\n{evidence_text or '(none yet)'}\n\n{head}\n\n{task}"
        hints = {"agent_label": agent["label"], "round": round_number, "step": step_title, "evidence_text": evidence_text}
        no_reflection_tools = [] if self.config["protocol"]["reflection"] else self.tools_for(agent)
        req = self.request(agent, "speak", settings["discussion_model"], prompt, round_number, tools=no_reflection_tools, max_output_tokens=settings["max_message_tokens"], hints=hints)
        result = self.provider.run(req)
        self.emit("message", key, result.text, {"citations": self.citations(result.text), "finish": result.finish, "step": step_title, "sources": result.sources},
                  audience="all" if shared else key, round_number=round_number, model=settings["discussion_model"], result=result)
        if self.theme_mode == "opener_theme" and self.spec.get("encounter") and round_number == 1 and key == self.config["protocol"]["lead"] and not self.opening:
            self.opening = {"mode": "opener_theme", "agent": key, "text": result.text, "citations": proto.cited_ids(result.text)}     # the theme the opener proposed

    @property
    def premise_ids(self):
        return set().union(*self.premises.values()) if self.premises else set()

    def search_terms(self, agent):
        mine = [self.pool[p] for p in self.visible.get(agent["key"], []) if p in self.premises.get(agent["key"], ())]
        if mine:
            return list(dict.fromkeys([mine[0]["label_a"], mine[0]["label_b"]]))
        return []

    def citations(self, text):
        return [{"id": i, "valid": self.known(i)} for i in proto.cited_ids(text)]

    def known(self, source_id):
        return source_id in self.evidence or source_id in self.pool or source_id in self.runs or source_id in self.web

    def known_ids(self):
        return set(self.evidence) | set(self.pool) | set(self.runs) | set(self.web)

    def register_web(self, sources, agent, round_number):
        """Give every page a web search returned its W-id, so that it can be cited like a passage. The same URL keeps its id. Returns the ids of these sources."""
        ids = []
        for source in sources or []:
            url = str(source.get("url") or "").strip()
            if not url:
                continue
            if url not in self.web_urls:
                self.web_urls[url] = f"W{len(self.web) + 1}"
                self.web[self.web_urls[url]] = {"url": url, "title": str(source.get("title") or "").strip(), "agent": agent["key"], "round": round_number}
            ids.append(self.web_urls[url])
        return list(dict.fromkeys(ids))

    # -- the protocol of each interaction --
    def prepare(self):
        runs = rag.release_scope(self.agent_con, self.release_id, True)[1]["runs"]
        if self.spec["organized"] and runs.get("extraction_run_id"):
            self.has_records = bool(self.agent_con.execute("SELECT 1 FROM record_embedding WHERE model_id=%s AND extraction_run_id=%s LIMIT 1",
                                                           (self.embedder.model_id, runs["extraction_run_id"])).fetchone())
        if self.spec["start"] == "constructs":
            self.extraction_run = runs.get("extraction_run_id")
        self.agent_con.commit()
        if self.diagnoses or self.matched_raw:          # only the construct-problem trigger: no other episode sees a construct relationship
            self.pool = {p["id"]: p for p in diagnosis_pool(self.agent_con, self.release_id, self.config["diagnoses"].get("provisional", False),
                                                            self.config["diagnoses"].get("pipeline_hints", False))}
            for agent in self.agents:
                if agent["community"]:
                    members = set(rag.community_papers(self.agent_con, rag.release_scope(self.agent_con, self.release_id, True)[1], agent["community"], True))
                    self.agent_con.commit()
                    members = {str(m) for m in members}
                    self.visible[agent["key"]] = [p["id"] for p in self.pool.values() if {p["dossier"]["a"]["paper"]["canonical_id"], p["dossier"]["b"]["paper"]["canonical_id"]} & members]
                else:
                    self.visible[agent["key"]] = list(self.pool)
        if self.analysis != "none":
            for agent in self.agents:
                self.data_files[agent["key"]] = self.build_data(agent)
        providers = {a["key"]: self.config["agents"][a["key"]] for a in self.agents}
        self.emit("note", None, f"Condition {self.condition} - {self.spec['name']}. {self.spec['summary']}", {
            "tests": self.spec["tests"], "provider": self.config["provider"], "budgets": self.budgets, "agents": providers, "seed": self.seed, "focus": self.config.get("focus"),
            "trigger": self.config.get("trigger"),
            "retrieval": self.config["retrieval"], "external_access": self.config["external_access"], "reserve_tokens": self.reserve_tokens,
            "analysis_tools": self.analysis, "code": self.code or None, "proposal": self.proposal,
            "budget_split": "shared", "lead": self.config["protocol"].get("lead"),
            "theme_mode": self.theme_mode if self.condition == "C" else None, "today": self.today,
            "data_files": {a["key"]: {n: {k: v for k, v in d.items() if k != "bytes"} for n, (_, d) in self.data_files.get(a["key"], {}).items()} for a in self.agents}}, audience="none")
        if self.trigger:
            name, instruction, move = proto.TRIGGERS[self.trigger]
            self.emit("system", None, f"Trigger of this episode: {name} ({move}). {instruction}", {"role": "trigger", "trigger": self.trigger}, audience="all", round_number=1)
        if self.diagnoses:
            self.hand_over_construct_problems()
        elif self.matched_raw:
            self.hand_over_matched_passages()
        if self.spec.get("encounter"):
            self.open_random_theme()

    def premises_of(self, key):
        """The dossiers handed over that this agent's question may rest on: its own in B, everybody's otherwise."""
        return self.premises.get(key, set()) if self.spec["interaction"] == "isolated" else self.premise_ids

    def choose_pairs(self, items, limit):
        """`limit` pairs of `items`, the categories in turn and, within each, people's decisions before pairs only the signals labelled. The draw depends on the seed
        alone (its own generator), so the same seed and pool give the same pairs in every condition: the matched opportunities of the proposal."""
        rng = random.Random(f"dossiers:{self.seed}")
        buckets = {}
        for pair in sorted(items, key=lambda p: p["pair_id"]):
            buckets.setdefault(pair["category"], []).append(pair)
        for bucket in buckets.values():
            rng.shuffle(bucket)
            bucket.sort(key=lambda p: not p["pipeline"])      # stable: pop() takes from the end, so a pair people decided comes first
        chosen = []
        while len(chosen) < limit and any(buckets.values()):
            for category in sorted(buckets):
                if buckets[category] and len(chosen) < limit:
                    chosen.append(buckets[category].pop())
        return chosen

    def hand_over_construct_problems(self):
        """The construct-problem trigger in A, B and C: dossiers of the construct relationships people decided, of every kind - nothing says to separate or to integrate.
        One hand-over for everybody in A and C; in B each specialist gets its own, from the pairs that touch its community, in its own feed and from its own budget."""
        limit, isolated = self.config["diagnoses"]["limit"], self.spec["interaction"] == "isolated"
        groups = ([(a["key"], [a["key"]], self.visible.get(a["key"], [])) for a in self.agents] if isolated else
                  [("all", [a["key"] for a in self.agents], list(dict.fromkeys(p for a in self.agents for p in self.visible.get(a["key"], []))))])
        for audience, keys, ids in groups:
            chosen = self.choose_pairs([self.pool[p] for p in ids], limit)
            blocks, cards = [], []
            for pair in chosen:
                self.read_pairs.add(pair["id"])
                for key in keys:
                    self.premises.setdefault(key, set()).add(pair["id"])
                text, cut = self.dossier_text(pair)
                who = ("proposed by the automatic signals only: NO PERSON HAS REVIEWED IT" if pair["pipeline"] else
                       "decided by one human reviewer, not yet confirmed by a second" if pair["provisional"] else "decided by human reviewers")
                decided = f"decided: {proto.CATEGORY_LABELS[pair['category']]}" if not pair["pipeline"] else f"signal hint: {pair['category']}"
                blocks.append(f"[{pair['id']}] {pair['label_a']} vs {pair['label_b']} ({decided}; {who})\n{text}")
                cards.append({"id": pair["id"], "category": pair["category"], "relation": pair["relation"], "flag": pair["flag"], "label_a": pair["label_a"], "label_b": pair["label_b"],
                              "paper_a": pair["paper_a"], "paper_b": pair["paper_b"], "truncated": cut, "provisional": pair["provisional"], "pipeline": pair["pipeline"]})
            if not chosen:
                self.emit("note", None, "No construct relationship touches this agent's community: it works without a dossier.", {"agent": audience}, audience="none")
                continue
            intro = ("Construct-problem dossiers handed over for your trigger. Each records a relationship between two construct uses that human reviewers decided, with their "
                     "rationale and the contrary evidence. Examine what theoretically changes if the adjudication is right; accept, challenge or reason conditionally from it. "
                     "Nothing tells you to separate or to integrate.")
            if any(p["pipeline"] for p in chosen):
                intro += (" EXCEPTION to rule 3: the dossiers marked 'NO PERSON HAS REVIEWED IT' were not decided by anybody; they are proposals from automatic signals with uncalibrated "
                          "thresholds (the semantic closeness of definitions and measures). Treat them as hypotheses that may well be wrong: first check in the dossier whether the two "
                          "uses really look like one label for different constructs (jingle) or different labels for one construct (jangle), say so when they do not, and never present "
                          "them as established.")
            self.emit("system", None, intro + "\n\n" + "\n\n".join(blocks), {"role": "premises", "pairs": cards}, audience=audience, counts={"dossiers": len(cards)})

    def hand_over_matched_passages(self):
        """The construct-problem trigger in condition 0: the pairs the organized conditions get with this seed, but only as raw passages of their papers - no label, no
        relationship, no rationale (the proposal's matched opportunities: 0 sees the raw evidence, the organized conditions the structured records and dossiers). For each
        paper, the first passage that names the construct, else its first passage. They count as passages."""
        agent = self.agents[0]
        chosen = self.choose_pairs(list(self.pool.values()), self.config["diagnoses"]["limit"])
        ingestion_id, _ = rag.release_scope(self.agent_con, self.release_id, True)
        found = []
        for pair in chosen:
            for side in ("a", "b"):
                row = self.agent_con.execute(
                    "SELECT c.chunk_id FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s AND v.canonical_id=%s::uuid "
                    "ORDER BY (position(lower(%s) in lower(c.text)) > 0) DESC, c.chunk_id LIMIT 1",
                    (ingestion_id, pair["dossier"][side]["paper"]["canonical_id"], pair["dossier"][side]["label"])).fetchone()
                if row and row[0] not in found:
                    found.append(row[0])
        hits = rag.hydrate(self.agent_con, [(cid, 0.0, {"matched": True}) for cid in found]) if found else []
        self.agent_con.commit()
        allowance = min(len(hits), self.passages_left(agent))
        if not hits or allowance <= 0:
            self.emit("note", None, "No passage could be handed over for the construct-problem trigger: the agent works from its own searches.", {}, audience="none")
            return
        shown, new, text = self.present(hits, allowance, agent)
        self.emit("system", None, "Passages handed over for your trigger. They come from papers whose constructs may raise a construct problem; nobody tells you which construct "
                                  "or relationship is at stake, or what anybody decided.\n\n" + text,
                  {"role": "premises", "matched": True, "hits": shown, "pair_ids": [p["pair_id"] for p in chosen]}, audience="all", counts={"passages": new})

    def draw_random(self, agent, n):
        """n items drawn at random, with the episode's seed, from the agent's own community: passages of the PDFs and (if the release has them) records of the construct
        database. Items already seen in this episode keep their id. Returns hits shaped like the search tools'."""
        ingestion_id, manifest = rag.release_scope(self.agent_con, self.release_id, True)
        papers = [str(p) for p in rag.community_papers(self.agent_con, manifest, agent["community"], True)]
        n_records = n // 2 if self.has_records else 0
        hits = []
        rows = self.agent_con.execute("SELECT c.chunk_id FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s AND v.canonical_id::text = ANY(%s) AND length(c.text) > 400 ORDER BY c.chunk_id",
                                      (ingestion_id, papers)).fetchall()
        chosen = self.rng.sample([r[0] for r in rows], min(n - n_records, len(rows)))
        if chosen:
            hits += rag.hydrate(self.agent_con, [(cid, 0.0, {"random_draw": True}) for cid in chosen])
        if n_records:
            for f in embedding_jobs.random_records(self.agent_con, manifest["runs"]["extraction_run_id"], self.embedder.model_id, papers, n_records, self.rng):
                hits.append(self.record_hit(f))
        self.agent_con.commit()
        return hits

    def present(self, hits, allowance, agent):
        seen, new = self.register_hits(hits, allowance, agent)
        self.spend(agent, passages=new)
        shown = [self.passage_view(i, h, self.config["retrieval"]["passage_chars"]) for i, h in seen]
        for s_, (_, h) in zip(shown, seen):
            if h.get("record_kind"):
                s_["record_kind"] = h["record_kind"]
        text = "\n\n".join(f"[{s['id']}] {s['title']} ({s['year']}), {s['section']}, pp. {s['pages'][0]}-{s['pages'][1]}\n{s['text']}" for s in shown)
        return shown, new, text

    def open_random_theme(self):
        """Condition C. The opener is the lead agent: the one chosen in the form, or the one the seed picked. By default (opener_theme, proposal v4) nothing is drawn: the
        opener proposes a theme from evidence, following the trigger, in its first message. The auxiliary controls open on drawn material (random passages, or a random
        starting point with the studies closest to it) or on nothing (each starts from its own construct database)."""
        opener = self.by_key[self.config["protocol"]["lead"]]
        if self.theme_mode == "opener_theme":
            other = self.other_than(opener)
            self.emit("system", None, f"Opening (seed {self.seed}). {opener['label']} opens: following the trigger, propose one theme supported by evidence from your corpus and "
                                      f"say why the other community may matter to it. Then {other['label']} tests or reframes it from their own corpus.",
                      {"role": "opening", "mode": "opener_theme", "opener": opener["key"], "trigger": self.trigger}, audience="all", round_number=1)
            return
        if self.theme_mode == "constructs":          # no theme: each specialist starts from the construct database of its own community (the steps say so)
            return
        if self.theme_mode == "rag_cluster":
            return self.open_retrieved_theme(opener)
        n = min(2, self.passages_left(opener))
        if n <= 0:
            return
        shown, new, text = self.present(self.draw_random(opener, n), n, opener)
        self.opening = {"mode": "random_passages", "agent": opener["key"], "items": [s["id"] for s in shown]}
        self.emit("system", None, f"Opening theme, drawn at random (seed {self.seed}) from the {opener['community']} corpus. {opener['label']} opens the conversation from it, "
                                  f"and then {self.other_than(opener)['label']} answers.\n\n{text}", {"role": "theme", "mode": "random_passages", "hits": shown, "opener": opener["key"],
                                                                                                 "from_community": opener["community"]},
                  audience="all", round_number=1, counts={"passages": new})

    def open_retrieved_theme(self, opener):
        """A theme built by retrieval: one item drawn at random (seeded) from the opener's corpus - a construct record if the release has them, else a passage - and, as the studies
        of the theme, the passages of the same corpus closest to it, at most one per paper. The search is logged under the opener's profile like any other."""
        size = min(self.config["encounter"]["theme_size"], self.passages_left(opener))
        if size <= 0:
            return
        drawn = self.draw_random(opener, 2 if self.has_records else 1)
        if not drawn:
            self.emit("note", None, f"The {opener['community']} corpus has nothing to draw a theme from: the conversation opens without one.", {}, audience="none")
            return
        seed_hit = next((h for h in drawn if h.get("record_kind")), drawn[0])
        found, searched = [], False
        used, cap = self.ledger(opener)[:2]
        if used["retrievals"] < cap["retrievals"]:
            try:
                found = rag.search(self.agent_con, self.release_id, " ".join(seed_hit["text"].split())[:600], self.embedder, k=min(size, 20), agent=opener["profile_id"],
                                   community=opener["community"], max_per_paper=1, allow_unvalidated=True, include_v1=True)
                searched = True
            except ValueError as exc:
                self.emit("note", None, f"The studies closest to the starting point could not be searched ({exc}): the theme is the starting point alone.", {}, audience="none")
        hits = [seed_hit] + [h for h in found if h["chunk_id"] != seed_hit["chunk_id"]]
        shown, new, text = self.present(hits[:size], size, opener)
        if searched:
            self.spend(opener, retrievals=1)
        what = "a construct record" if seed_hit.get("record_kind") else "a passage"
        self.opening = {"mode": "rag_cluster", "agent": opener["key"], "items": [s["id"] for s in shown]}
        self.emit("system", None, f"Opening theme (seed {self.seed}). The starting point is {what} drawn at random from the {opener['community']} corpus, [{shown[0]['id']}]; the other items are the studies of the "
                                  f"same corpus closest to it. {opener['label']} opens: what do these studies find, what conclusion do they lead to, and what problem is there in that conclusion? "
                                  f"Then {self.other_than(opener)['label']} tests it against their own community's literature.\n\n{text}",
                  {"role": "theme", "mode": "rag_cluster", "hits": shown, "opener": opener["key"], "from_community": opener["community"], "starting_point": shown[0]["id"]},
                  audience="all", round_number=1, counts={"passages": new, "retrievals": 1 if searched else 0})

    def other_than(self, agent):
        return next(a for a in self.agents if a["key"] != agent["key"])

    def encounter(self, agent, round_number):
        """Condition C: random papers and constructs of the agent's own community, drawn with the episode's seed and brought into the conversation, where the other agent sees them."""
        n = min(self.config["encounter"]["items_per_round"], self.passages_left(agent))
        if n <= 0:
            return
        shown, new, text = self.present(self.draw_random(agent, n), n, agent)
        self.emit("encounter", agent["key"], f"{agent['label']} brings these items, drawn at random from the {agent['community']} corpus (not chosen for relevance), into the conversation:\n\n{text}",
                  {"hits": shown, "from_community": agent["community"]}, audience="all", round_number=round_number, counts={"passages": new})

    def synthesize(self, agent, round_number):
        key, settings = agent["key"], self.config["agents"][agent["key"]]
        shared = self.spec["interaction"] == "discussion"
        transcript = self.visible_messages(agent)
        premises = self.premises_of(key)
        ids = list(dict.fromkeys(proto.cited_ids(transcript) + proto.cited_ids(self.notes[key]) + sorted(premises)
                                 + [w for w, v in self.web.items() if v["agent"] == key]))
        if not shared:
            ids = list(dict.fromkeys(ids + list(self.evidence)[:12] + [r for r, v in self.runs.items() if v["agent"] == key]))
        evidence_text = self.evidence_block(ids, self.config["retrieval"]["passage_chars"])
        listing = ", ".join(sorted(self.known_ids()))
        prompt = (f"THE WORK SO FAR:\n{transcript or '(your notebook is empty)'}\n\nYOUR PRIVATE NOTES:\n{self.notes[key] or '(none)'}\n\nEVIDENCE:\n{evidence_text or '(none)'}\n\n"
                  f"Valid source ids: {listing or '(none)'}\n\n{proto.synth_task(self.condition, self.theme_mode, self.trigger, self.diagnoses)}")
        self.on_activity(f"{agent['label']} is formulating the question ({settings['synthesis_model']})")
        premise = "; ".join(f"{self.pool[p]['label_a']} vs {self.pool[p]['label_b']} ({self.pool[p]['category']})" for p in sorted(premises, key=lambda i: int(i[1:]))[:2]) or ("the question starts from the construct network, with no dossier handed over" if self.spec["start"] == "constructs" else "the question comes from the corpus, with no dossier handed over")
        req = self.request(agent, "synthesize", settings["synthesis_model"], prompt, round_number, max_output_tokens=settings["max_synthesis_tokens"],
                           json_schema={"name": "research_question", "schema": proto.QUESTION_SCHEMA}, hints={"agent_label": agent["label"], "premise": premise})
        result = self.provider.run(req)
        if result.finish == "incomplete":
            self.emit("note", key, "The final question was cut off by its output limit, so it is not valid JSON and was not stored.", {"finish": "incomplete"}, audience="none", result=result,
                      round_number=round_number, model=settings["synthesis_model"])
            return None
        try:
            content, warnings = proto.validate_question(json.loads(result.text), self.known_ids(), {r: v["ok"] for r, v in self.runs.items()} if self.analysis != "none" else None,
                                                        years={i: h.get("year") for i, h in self.evidence.items()}, web=self.web_on, trigger=self.trigger)

            def first(note):
                warnings.insert(0, note)
                content["_validation"]["warnings"].insert(0, note)
            trigger = self.config.get("trigger") or {}
            content["_validation"]["trigger"] = trigger.get("key")
            if trigger.get("cross_community"):          # the proposal: cases that need both communities are analysed apart, so that B is not penalised for an impossible task
                content["_validation"]["cross_community"] = True
                first("The trigger needs both communities and this specialist saw only one: analyse this question apart from the within-community ones.")
            release = self.config.get("release") or {}
            content["_validation"]["release"] = release
            if release and not release.get("validated"):
                first(f"The release this question was made on is not validated (status: {release.get('status')}): treat the question as exploratory.")
            unreviewed = sorted((p for p in premises if self.pool[p]["pipeline"]), key=lambda i: int(i[1:]))
            if unreviewed:
                first(f"Construct-problem dossier(s) {', '.join(unreviewed)} were NOT decided by any person: they are labels from the automatic signals (semantic closeness of definitions "
                      "and measures, thresholds not calibrated). The question rests on unreviewed hypotheses: treat it as exploratory, a trial of the forum and not a finding.")
            single = sorted((p for p in premises if self.pool[p]["provisional"]), key=lambda i: int(i[1:]))
            if single:
                first(f"Construct-problem dossier(s) {', '.join(single)} rest on the decision of a single reviewer, with no second reviewer or adjudication "
                      "(provisional relationships were allowed for this episode): treat the question as exploratory.")
        except (ValueError, TypeError) as exc:
            self.emit("note", key, f"The final question could not be read ({exc}); nothing was stored.", {"raw": result.text[:2000]}, audience="none", result=result,
                      round_number=round_number, model=settings["synthesis_model"])
            return None
        sources = [self.source_record(i) for i in content["sources"]]
        question_id = uuid.uuid4()
        agents = [a["key"] for a in self.agents] if shared else [key]
        opener = self.config["protocol"].get("lead") if self.spec.get("encounter") else None
        self.app.execute("INSERT INTO question(question_id,episode_id,release_id,condition,agents,rq,content,evidence,ideation_trigger,numeric_seed,opener,opening_theme,"
                         "alternative_puzzles,conceptual_signature) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (question_id, self.episode_id, self.release_id, self.condition, agents, content["rq"].strip(), Jsonb(content), Jsonb(sources), self.trigger, self.seed,
                          opener, Jsonb(self.opening) if self.opening else None, Jsonb(content.get("alternative_puzzles")) if "alternative_puzzles" in content else None,
                          Jsonb(content.get("conceptual_signature")) if "conceptual_signature" in content else None))
        self.emit("question", key, content["rq"].strip(), {"question_id": str(question_id), "warnings": warnings}, audience="all", round_number=round_number,
                  model=settings["synthesis_model"], result=result)
        self.questions.append((question_id, key, content))
        return question_id

    # -- the proposal --
    def exploratory_notice(self, question):
        release = (question.get("_validation") or {}).get("release") or {}
        parts = []
        if release and not release.get("validated"):
            parts.append(f"the release behind it is not validated ({release.get('status')})")
        if any(self.pool[p]["pipeline"] for p in self.premise_ids if p in self.pool):
            parts.append("its construct-problem dossiers were labelled by automatic signals and no person reviewed them")
        if any(self.pool[p]["provisional"] for p in self.premise_ids if p in self.pool):
            parts.append("its premises rest on a single reviewer")
        return ("EXPLORATORY, NOT A FINDING: " + "; ".join(parts) + ".") if parts else None

    def reference_record(self, source_id):
        if source_id in self.evidence:
            h = self.evidence[source_id]
            return {"kind": "construct" if h.get("record_kind") else "passage", "title": h["title"], "year": h["year"], "pages": h["pages"], "doi": h.get("doi"),
                    "authors": h.get("authors"), "journal": h.get("journal")}
        if source_id in self.pool:
            p = self.pool[source_id]
            basis = "labelled by the automatic signals only" if p["pipeline"] else "decided by one reviewer" if p["provisional"] else "decided by human reviewers"
            return {"kind": "diagnosis", "category": proto.CATEGORY_LABELS[p["category"]], "basis": basis, "labels": [p["label_a"], p["label_b"]]}
        if source_id in self.runs:
            r = self.runs[source_id]
            return {"kind": "analysis", "purpose": r["purpose"], "code_sha256": r["code_sha256"], "ok": r["ok"]}
        if source_id in self.web:
            w = self.web[source_id]
            return {"kind": "web", "title": w["title"], "url": w["url"]}
        return None

    def proposal_context(self, agent, question, extra=()):
        key = agent["key"]
        body = json.dumps({k: v for k, v in question.items() if k != "_validation"}, ensure_ascii=False)
        transcript = self.visible_messages(agent)
        asked = list(question.get("sources") or []) + [x for step in question.get("argument_steps") or [] for x in step.get("sources") or []]
        ids = list(dict.fromkeys(asked + proto.cited_ids(json.dumps(question.get("timeliness") or {})) + proto.cited_ids(json.dumps(question.get("trigger_evidence") or ""))
                                 + list(extra) + proto.cited_ids(transcript) + proto.cited_ids(self.notes[key])
                                 + sorted(self.premises_of(key)) + [r for r, v in self.runs.items() if v["agent"] == key]))
        ids = [i for i in ids if self.known(i)][:40]
        evidence_text = self.evidence_block(ids, self.config["retrieval"]["passage_chars"])
        return (f"THE QUESTION THE WORK SETTLED ON:\n{body}\n\nTHE WORK SO FAR:\n{transcript or '(your notebook is empty)'}\n\nEVIDENCE YOU MAY CITE:\n{evidence_text or '(none)'}\n\n"
                f"Valid source ids: {', '.join(ids) or '(none)'}"), ids

    @staticmethod
    def proposal_json(content):
        return json.dumps({k: content[k] for k in ("title", "abstract", *[x[0] for x in fprop.SECTIONS]) if k in content}, ensure_ascii=False)

    def proposal_answer(self, agent, task, prompt, round_number, label):
        """One structured call of the proposal. (content, warnings), or None if the answer was cut off or unreadable (a note says why)."""
        key, settings = agent["key"], self.config["agents"][agent["key"]]
        known = self.known_ids()
        self.on_activity(f"{agent['label']} is writing the {label} ({settings['synthesis_model']})")
        req = self.request(agent, task, settings["synthesis_model"], prompt, round_number, max_output_tokens=self.proposal["max_tokens"],
                           json_schema={"name": "research_proposal", "schema": fprop.PROPOSAL_SCHEMA}, hints={"agent_label": agent["label"], "pages": self.proposal["page_limit"]})
        result = self.provider.run(req)
        if result.finish == "incomplete":
            self.emit("note", key, f"The {label} was cut off by its output limit, so it is not valid JSON and was not used.", {"finish": "incomplete", "task": task}, audience="none",
                      result=result, round_number=round_number, model=settings["synthesis_model"])
            return None
        try:
            content, warnings = fprop.validate_proposal(json.loads(result.text), known, pages=self.proposal["page_limit"])
        except (ValueError, TypeError) as exc:
            self.emit("note", key, f"The {label} could not be used ({exc}).", {"raw": result.text[:2000], "task": task}, audience="none", result=result, round_number=round_number,
                      model=settings["synthesis_model"])
            return None
        self.emit("note", key, f"The {label}: {sum(content['_validation']['words'].values())} words.", {"role": "proposal_stage", "task": task, "proposal": self.proposal_json(content),
                                                                                                         "warnings": warnings}, audience="none", result=result,
                  round_number=round_number, model=settings["synthesis_model"])
        return content, warnings

    def critique_context(self, author, question, content, ids):
        """What the critic reads besides the draft (18 Oct 2026, the user's decision): the question, the evidence the draft cites and the dossiers it rests on, and the ids the author
        could cite - not the transcript of the discussion, which the critic took part in, nor the critic's own notes. An id the draft cites that is not in the list is one it invented."""
        body = json.dumps({k: v for k, v in question.items() if k != "_validation"}, ensure_ascii=False)
        cited = [i for i in dict.fromkeys(content["_validation"]["cited"] + sorted(self.premises_of(author["key"]))) if self.known(i)]
        evidence_text = self.evidence_block(cited, self.config["retrieval"]["passage_chars"])
        return (f"THE QUESTION THE WORK SETTLED ON:\n{body}\n\nTHE EVIDENCE THE DRAFT CITES (and the dossiers it rests on):\n{evidence_text or '(none)'}\n\n"
                f"Valid source ids: {', '.join(ids) or '(none)'}")

    def proposal_critique(self, critic, author, content, ids):
        settings = self.config["agents"][critic["key"]]
        rounds = self.config["protocol"]["rounds"]
        context = self.critique_context(author, self.questions_by_author[author["key"]], content, ids)
        self.on_activity(f"{critic['label']} is criticising the proposal ({settings['discussion_model']})")
        prompt = f"{context}\n\nTHE DRAFT PROPOSAL OF {author['label']}:\n{self.proposal_json(content)}\n\n{fprop.critique_task(self.diagnoses)}"
        result = self.provider.run(self.request(critic, "proposal_critique", settings["discussion_model"], prompt, rounds, max_output_tokens=settings["max_message_tokens"],
                                                hints={"agent_label": critic["label"]}))
        self.emit("message", critic["key"], result.text, {"citations": self.citations(result.text), "finish": result.finish, "step": "Proposal critique"}, audience="all",
                  round_number=rounds, model=settings["discussion_model"], result=result)
        return result.text if result.finish != "incomplete" and result.text.strip() else None

    def proposal_cards(self, agent, content, proposal_ref, round_number):
        """The methodology card and the card of contributions of the final proposal, written by its author (forum_cards.py). -> ({kind: stored record}, [warnings]). Never raises for a card
        that could not be made: the proposal is complete without them, so a stop, an exhausted budget, an unreadable answer or a refusal by the check is stored as such, with the reason."""
        key, settings = agent["key"], self.config["agents"][agent["key"]]
        source, model = fcards.source_text(content), settings["synthesis_model"]
        stored, warnings = {}, []
        for kind in fcards.KINDS:
            task, problems, outcome = fcards.TASKS[kind], [], None
            effort = fcards.effort(kind, settings["reasoning_effort"])
            for attempt in range(fcards.ATTEMPTS):
                self.on_activity(f"{agent['label']} is writing the {kind} card ({model})")
                try:
                    req = self.request(agent, task, model, fcards.prompt(kind, proposal_ref, source, problems), round_number, max_output_tokens=fcards.MAX_OUTPUT_TOKENS,
                                       json_schema={"name": task, "schema": fcards.provider_schema(kind)}, hints={"agent_label": agent["label"], "kind": kind},
                                       system=fcards.system(kind), reasoning_effort=effort)
                    result = self.provider.run(req)
                except (BudgetExceeded, Cancelled) as exc:
                    outcome = fcards.record(kind, "not_made", source=source, model=model, reasoning_effort=effort, reason=f"stopped before the card could be written: {exc}")
                    break
                except Exception as exc:        # the proposal is complete and not stored yet: an error of the provider in a card must not lose it
                    LOG.exception("card call failed")
                    outcome = fcards.record(kind, "not_made", source=source, model=model, reasoning_effort=effort, reason=f"the call failed: {type(exc).__name__}: {str(exc)[:200]}")
                    break
                if result.finish == "incomplete":
                    problems = ["the answer was cut off by its output limit, so it is not valid JSON"]
                    self.emit("note", key, f"The {kind} card was cut off by its output limit and was not used.", {"finish": "incomplete", "task": task}, audience="none", result=result,
                              round_number=round_number, model=model)
                    continue
                try:
                    card, demoted, problems = fcards.check(kind, json.loads(result.text), source)
                except (ValueError, TypeError) as exc:
                    card, demoted, problems = None, [], [f"the answer could not be read: {str(exc)[:120]}"]
                self.emit("note", key, f"The {kind} card: " + ("accepted" if card else "refused (" + "; ".join(problems)[:200] + ")") + ".",
                          {"role": "card_stage", "task": task, "kind": kind, "attempt": attempt + 1, "demoted_fields": demoted, "problems": problems}, audience="none", result=result,
                          round_number=round_number, model=model)
                if card:
                    outcome = fcards.record(kind, "valid", source=source, model=model, reasoning_effort=effort, card=card, demoted=demoted)
                    break
            stored[kind] = outcome or fcards.record(kind, "rejected", source=source, model=model, reasoning_effort=effort, problems=problems)
            if stored[kind]["status"] != "valid":
                warnings.append(f"The {kind} card could not be made ({stored[kind]['reason'] or '; '.join(stored[kind]['problems'])[:200]}).")
            elif stored[kind]["demoted_fields"]:
                warnings.append(f"In the {kind} card, {len(stored[kind]['demoted_fields'])} field(s) were set to not stated because their quotation was not an exact piece of the proposal.")
        return stored, warnings

    def write_proposal(self, question_id, key, question):
        """Draft, (in a discussion) critique by the other specialist and revision, then render and count the pages; shorten if over the limit. Returns the proposal id or None."""
        author = self.by_key[key]
        limit, rounds = self.proposal["page_limit"], self.config["protocol"]["rounds"]
        self.questions_by_author[key] = question
        context, ids = self.proposal_context(author, question)
        made = self.proposal_answer(author, "proposal_draft", f"{context}\n\n{fprop.draft_task(limit, self.diagnoses)}", rounds, "draft proposal")
        if not made:
            return None
        content, warnings = made
        critic = self.other_than(author) if self.spec["interaction"] == "discussion" else None
        if critic:
            try:
                critique = self.proposal_critique(critic, author, content, ids)
                revised = self.proposal_answer(author, "proposal_revise", f"{context}\n\nYOUR DRAFT:\n{self.proposal_json(content)}\n\nCRITIQUE BY {critic['label']}:\n{critique}\n\n"
                                                                           f"{fprop.revise_task(limit)}", rounds, "revised proposal") if critique else None
            except BudgetExceeded as exc:
                self.emit("note", key, f"The critique and revision of the proposal were skipped: {exc}", {"reason": str(exc)}, audience="none")
                revised = None
            if revised:
                content, warnings = revised
            else:
                warnings = warnings + ["The proposal was not revised after the critique (see the log)."]
        meta = {"condition": f"{self.condition} ({self.spec['name']})", "authors": [author["label"] + (" with " + critic["label"] if critic else "")],
                "episode": f"episode {str(self.episode_id)[:8]}", "date": datetime.date.today().isoformat(), "notice": self.exploratory_notice(question)}

        def render(c):
            lines = fprop.reference_lines(c, {i: r for i in c["_validation"]["cited"] for r in [self.reference_record(i)] if r})
            pdf, pages = fprop.render_pdf(c, meta, lines)
            return lines, pdf, pages
        refs, pdf, pages = render(content)
        for _ in range(2):
            if pages <= limit:
                break
            try:
                shorter = self.proposal_answer(author, "proposal_shorten", f"{context}\n\nYOUR PROPOSAL:\n{self.proposal_json(content)}\n\n{fprop.shorten_task(pages, limit, self.diagnoses)}", rounds, "shortened proposal")
            except BudgetExceeded as exc:
                self.emit("note", key, f"The proposal could not be shortened: {exc}", {"reason": str(exc)}, audience="none")
                break
            if not shorter:
                break
            new_refs, new_pdf, new_pages = render(shorter[0])
            if new_pages < pages:
                content, warnings, refs, pdf, pages = shorter[0], shorter[1], new_refs, new_pdf, new_pages
        content["cards"], card_warnings = self.proposal_cards(author, content, f"proposal-{str(question_id)[:8]}", rounds)      # the design and the expected change, as cards, from the final text
        warnings = warnings + card_warnings
        status = "ok" if pages <= limit else "over_limit"
        if status == "over_limit":
            warnings = [f"The proposal runs {pages} pages and the limit is {limit}: it was not cut by hand, this is what the agents wrote."] + warnings
        if meta["notice"]:
            warnings = [meta["notice"]] + warnings
        proposal_id = uuid.uuid4()
        self.app.execute("INSERT INTO forum_proposal(proposal_id,question_id,episode_id,author,critic,status,pages,page_limit,content,refs,meta,warnings) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (proposal_id, question_id, self.episode_id, key, critic["key"] if critic else None, status, pages, limit, Jsonb(content), Jsonb(refs), Jsonb(meta), Jsonb(warnings)))
        self.emit("note", key, f"Proposal written: {pages} page(s), limit {limit}" + ("" if status == "ok" else " - OVER THE LIMIT"), {"role": "proposal", "proposal_id": str(proposal_id),
                  "pages": pages, "status": status, "question_id": str(question_id), "pdf_bytes": len(pdf)}, audience="none", round_number=rounds)
        self.proposal_ids.append(proposal_id)
        return proposal_id

    def score_proposals(self):
        """The evaluation of the proposals just written: novelty, methodology and impact against their nearest published papers (forum_scoring.py). Outside the episode's budgets and
        usage (it has its own cap, in its configuration); it can never fail the episode: a criterion that cannot be judged is stored as such, an error is a note, and a Stop leaves the
        rest to the Evaluate button of the Results page."""
        def gate():
            if self.stop.is_set():
                raise Cancelled("stopped")
        self.on_activity(f"Evaluation of {len(self.proposal_ids)} proposal(s): novelty, methodology, impact")
        try:
            result = fscore.score_proposals(self.app, self.proposal_ids, self.provider, gate=gate,
                                            on_wait=lambda: self.on_activity("Evaluation of the proposals waits for the one of another episode"))
        except Exception as exc:    # the proposals are already stored: whatever goes wrong here must not turn the episode into a failed one
            LOG.exception("proposal evaluation failed")
            self.emit("note", None, f"The evaluation of the proposals could not run: {type(exc).__name__}: {str(exc)[:200]}", {"role": "evaluation"}, audience="none")
            return
        if "skipped" in result:
            self.emit("note", None, f"The proposals were not evaluated: {result['skipped']}", {"role": "evaluation", **result}, audience="none")
        elif "stopped_by" in result:
            self.emit("note", None, f"The evaluation stopped after {result['stored']} proposal(s): {result['stopped_by']}. The rest can be evaluated from the Results page.",
                      {"role": "evaluation", **result}, audience="none")
        else:
            self.emit("note", None, f"Evaluation stored for {result['stored']} proposal(s).", {"role": "evaluation", **{k: v for k, v in result.items() if k != "usage"}}, audience="none")

    def source_record(self, source_id):
        if source_id in self.evidence:
            h = self.evidence[source_id]
            return {"id": source_id, "kind": "passage", "verified": True, "chunk_id": h["chunk_id"], "canonical_id": h["canonical_id"], "title": h["title"], "year": h["year"], "pages": h["pages"]}
        if source_id in self.pool:
            p = self.pool[source_id]
            return {"id": source_id, "kind": "diagnosis", "verified": True, "pair_id": p["pair_id"], "labels": [p["label_a"], p["label_b"]], "category": p["category"]}
        if source_id in self.runs:
            r = self.runs[source_id]
            return {"id": source_id, "kind": "analysis", "verified": True, "agent": r["agent"], "purpose": r["purpose"], "ok": r["ok"], "code_sha256": r["code_sha256"]}
        if source_id in self.web:      # verified = the web search of this episode returned it; what the page says was not checked
            w = self.web[source_id]
            return {"id": source_id, "kind": "web", "verified": True, "agent": w["agent"], "title": w["title"], "url": w["url"], "round": w["round"]}
        return {"id": source_id, "kind": "unknown", "verified": False}

    def run(self):
        status, detail = "completed", None
        try:
            self.set_status("running")
            self.prepare()
            rounds = self.config["protocol"]["rounds"]
            produced = 0
            try:
                for round_number in range(1, rounds + 1):
                    if self.spec["interaction"] == "discussion":
                        title, text = self.step(round_number)
                        self.emit("system", None, f"Round {round_number} of {rounds} - {title}. {text}", {"role": "moderator", "step": title}, audience="all", round_number=round_number)
                    order = self.agents
                    if self.spec.get("encounter"):          # the lead opens each round
                        order = sorted(self.agents, key=lambda a: a["key"] != self.config["protocol"]["lead"])
                    for agent in order:
                        if self.spec.get("encounter") and not (round_number == 1 and agent["key"] == self.config["protocol"]["lead"]):
                            self.encounter(agent, round_number)       # the opener's first items are the opening theme
                        self.turn(agent, round_number)
            except BudgetExceeded as exc:
                status, detail = "budget_exhausted", f"Stopped before the last round: {exc}"
                self.emit("note", None, detail + " The question is still written from the work so far.", {"reason": str(exc)}, audience="none")
            writers = [self.by_key[self.config["protocol"]["lead"]]] if self.spec["interaction"] == "discussion" else self.agents
            for agent in writers:
                if self.synthesize(agent, rounds):
                    produced += 1
            if self.proposal["enabled"]:
                missing = 0
                for question_id, key, question in list(self.questions):
                    try:
                        if not self.write_proposal(question_id, key, question):
                            missing += 1
                    except BudgetExceeded as exc:
                        missing += 1
                        self.emit("note", key, f"The proposal was not written: {exc}", {"reason": str(exc)}, audience="none")
                if missing:
                    detail = (detail + "; " if detail else "") + f"{missing} proposal(s) could not be written (see the log)"
                if self.auto_evaluate and self.proposal_ids:
                    self.score_proposals()
            if not produced:
                status, detail = "failed", (detail + "; " if detail else "") + "No valid research question was produced"
        except Cancelled:
            status, detail = "stopped", "Stopped by a person"
            self.emit("note", None, detail, {}, audience="none")
        except BudgetExceeded as exc:
            status, detail = "budget_exhausted", f"No question could be written: {exc}"
            self.emit("note", None, detail, {}, audience="none")
        except Exception as exc:    # the episode must end in a visible state whatever went wrong
            LOG.exception("forum episode failed")
            status, detail = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            try:
                self.emit("note", None, "The episode failed: " + detail, {}, audience="none")
            except psycopg.Error:
                pass
        finally:
            self.set_status(status, detail)
            self.on_activity("")
            self.agent_con.close()
            self.app.close()
        return status


# --- Reading (for the interface and the reports) ----------------------------------------------------------------------------------------

COUNTS = ("coalesce(sum((data->'counts'->>'passages')::int),0), coalesce(sum((data->'counts'->>'retrievals')::int),0), coalesce(sum((data->'counts'->>'dossiers')::int),0)")


def usage(con, episode_id):
    row = con.execute(f"SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0), coalesce(sum(reasoning_tokens),0), coalesce(sum(llm_calls),0), {COUNTS}, count(*), max(seq) "
                      "FROM forum_message WHERE episode_id=%s", (episode_id,)).fetchone()
    names = ("input_tokens", "output_tokens", "reasoning_tokens", "llm_calls", "passages", "retrievals", "dossiers", "messages", "last_seq")
    out = dict(zip(names, (int(v) if v is not None else -1 for v in row)))
    out["tokens"] = out["input_tokens"] + out["output_tokens"]
    out["by_agent"] = {k: {"tokens": int(t), "calls": int(c)} for k, t, c in con.execute(
        "SELECT agent_key,sum(input_tokens+output_tokens),sum(llm_calls) FROM forum_message WHERE episode_id=%s AND agent_key IS NOT NULL GROUP BY agent_key", (episode_id,)).fetchall()}
    return out


def cost(con, episode_id, models=None):
    """Dollar cost from the prices in config/forum_models.json; None as soon as any model used has no price (nothing is guessed)."""
    models = models or load_models()
    total = 0.0
    for model, i, o in con.execute("SELECT model,sum(input_tokens),sum(output_tokens) FROM forum_message WHERE episode_id=%s AND model IS NOT NULL GROUP BY model", (episode_id,)).fetchall():
        entry = proto.model_entry(models, model)
        price = (entry or {}).get("price_per_mtok") or {}
        if price.get("input") is None or price.get("output") is None:
            return None
        total += i / 1e6 * price["input"] + o / 1e6 * price["output"]
    return round(total, 4)


EPISODE_COLUMNS = "episode_id,release_id,condition,title,provider,seed,config,config_sha256,budgets_sha256,prompt_version,status,status_detail,created_by,created_at,started_at,finished_at"


def _episode_dict(row):
    keys = EPISODE_COLUMNS.split(",")
    out = dict(zip(keys, row))
    for k in ("episode_id", "release_id"):
        out[k] = str(out[k])
    for k in ("created_at", "started_at", "finished_at"):
        out[k] = out[k].isoformat(timespec="seconds") if out[k] else None
    spec = proto.CONDITIONS[out["condition"]]
    out["condition_name"], out["interaction"], out["tests"] = spec["name"], spec["interaction"], spec["tests"]
    out["legacy"] = bool(spec.get("legacy"))                     # D, E, F: removed from the design, readable only
    out["trigger"] = out["config"].get("trigger")                 # None for an episode made before the triggers
    return out


def list_episodes(con, release_ids):
    rows = con.execute(f"SELECT {EPISODE_COLUMNS} FROM forum_episode WHERE release_id = ANY(%s) ORDER BY created_at DESC", (release_ids,)).fetchall()
    out = []
    for row in rows:
        episode = _episode_dict(row)
        episode["usage"] = usage(con, row[0])
        episode["questions"] = con.execute("SELECT count(*) FROM question WHERE episode_id=%s", (row[0],)).fetchone()[0]
        episode["proposals"] = con.execute(f"SELECT count(*) FROM forum_proposal p WHERE p.episode_id=%s AND {fscore.NEW_FORMAT_SQL}", (row[0],)).fetchone()[0]      # the older format is not shown
        episode["budgets"] = episode["config"]["budgets"]
        episode["provisional"] = bool((episode["config"].get("diagnoses") or {}).get("provisional"))
        episode["pipeline_hints"] = bool((episode["config"].get("diagnoses") or {}).get("pipeline_hints"))
        episode["release"] = episode["config"].get("release") or {"label": "", "status": "validated", "validated": True}
        episode["models"] = sorted({m for a in episode["config"]["agents"].values() for m in (a["discussion_model"], a["reflection_model"], a["synthesis_model"])})
        episode["batch"] = episode["config"].get("batch")             # None for an episode created on its own
        episode["arm"] = proto.arm_of(episode["condition"], episode["config"].get("protocol"))      # "B-cto" / "B-tim" for one specialist: the page groups by it
        del episode["config"]
        out.append(episode)
    return out


def episode_settings(con, episode_id):
    """The episode with its frozen configuration and its agents' settings, without the transcript."""
    row = con.execute(f"SELECT {EPISODE_COLUMNS} FROM forum_episode WHERE episode_id=%s", (episode_id,)).fetchone()
    if not row:
        raise ValueError("Unknown episode")
    episode = _episode_dict(row)
    episode["agents"] = [{"key": k, "label": label, "community": community, "role": role, "settings": settings} for k, label, community, role, settings in con.execute(
        "SELECT agent_key,label,community,role,settings FROM forum_agent WHERE episode_id=%s", (episode_id,)).fetchall()]
    order = [a["key"] for a in proto.layout(episode["condition"])]
    episode["agents"].sort(key=lambda a: order.index(a["key"]))
    return episode


def episode_detail(con, episode_id, after=-1):
    episode = episode_settings(con, episode_id)
    episode["messages"] = [{"seq": seq, "agent": agent, "kind": kind, "audience": audience, "round": round_number, "content": content, "data": data, "model": model,
                            "input_tokens": i, "output_tokens": o, "reasoning_tokens": r, "latency_ms": latency, "at": at.isoformat(timespec="milliseconds")}
                           for seq, agent, kind, audience, round_number, content, data, model, i, o, r, latency, at in con.execute(
        "SELECT seq,agent_key,kind,audience,round,content,data,model,input_tokens,output_tokens,reasoning_tokens,latency_ms,created_at FROM forum_message WHERE episode_id=%s AND seq>%s ORDER BY seq LIMIT 500",
        (episode_id, after)).fetchall()]
    episode["usage"] = usage(con, episode_id)
    episode["cost_usd"] = cost(con, episode_id)
    episode["proposals"] = [{"proposal_id": str(pid), "question_id": str(qid), "author": author, "critic": critic, "status": status, "pages": pages, "page_limit": limit, "content": content,
                             "refs": refs, "meta": meta, "warnings": warnings, "created_at": at.isoformat(timespec="seconds")}
                            for pid, qid, author, critic, status, pages, limit, content, refs, meta, warnings, at in con.execute(
        f"SELECT proposal_id,question_id,author,critic,status,pages,page_limit,content,refs,meta,warnings,created_at FROM forum_proposal p WHERE episode_id=%s AND {fscore.NEW_FORMAT_SQL} "
        "ORDER BY created_at", (episode_id,)).fetchall()]
    # proposals of the earlier format (four sections, no cards) are not shown (the user's decision, 18 Oct 2026); they stay in the database
    episode["hidden_proposals"] = con.execute(f"SELECT count(*) FROM forum_proposal p WHERE episode_id=%s AND NOT ({fscore.NEW_FORMAT_SQL})", (episode_id,)).fetchone()[0]
    scores = fscore.latest_for(con, [p["proposal_id"] for p in episode["proposals"]])
    for p in episode["proposals"]:
        p["score"] = scores.get(p["proposal_id"])
    files = {}
    for run_id, name, media_type, size, sha in con.execute("SELECT run_id,name,media_type,size,sha256 FROM forum_analysis_file WHERE episode_id=%s ORDER BY run_id,name", (episode_id,)).fetchall():
        files.setdefault(run_id, []).append({"name": name, "media_type": media_type, "size": size, "sha256": sha})
    episode["analyses"] = [{"run_id": run_id, "agent": agent, "round": round_number, "purpose": purpose, "code": code, "code_sha256": digest, "inputs": inputs, "environment": environment,
                            "exit_code": exit_code, "timed_out": timed_out, "killed_for": killed_for, "stdout": stdout, "stderr": stderr, "duration_ms": duration,
                            "files": files.get(run_id, []), "created_at": at.isoformat(timespec="seconds")}
                           for run_id, agent, round_number, purpose, code, digest, inputs, environment, exit_code, timed_out, killed_for, stdout, stderr, duration, at in con.execute(
        "SELECT run_id,agent_key,round,purpose,code,code_sha256,inputs,environment,exit_code,timed_out,killed_for,stdout,stderr,duration_ms,created_at FROM forum_analysis "
        "WHERE episode_id=%s ORDER BY created_at,run_id", (episode_id,)).fetchall()]
    episode["questions"] = [{"question_id": str(qid), "agents": agents, "rq": rq, "content": content, "evidence": evidence, "evaluation": evaluation, "created_at": at.isoformat(timespec="seconds"),
                             "trigger": trigger, "seed": seed, "opener": opener, "opening_theme": opening}
                            for qid, agents, rq, content, evidence, evaluation, at, trigger, seed, opener, opening in con.execute(
        "SELECT question_id,agents,rq,content,evidence,evaluation,created_at,ideation_trigger,numeric_seed,opener,opening_theme FROM question WHERE episode_id=%s ORDER BY created_at",
        (episode_id,)).fetchall()]
    return episode


def comparable(con, episode_id):
    """Other episodes of the same release whose budgets and tool rules are identical (the proposal's condition of comparability)."""
    return [{"episode_id": str(e), "condition": c, "title": t, "status": s} for e, c, t, s in con.execute(
        "SELECT o.episode_id,o.condition,o.title,o.status FROM forum_episode o JOIN forum_episode e ON e.release_id=o.release_id AND e.budgets_sha256=o.budgets_sha256 "
        "WHERE e.episode_id=%s AND o.episode_id<>e.episode_id ORDER BY o.created_at", (episode_id,)).fetchall()]
