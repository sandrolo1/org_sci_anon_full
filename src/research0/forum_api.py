"""JSON endpoints of the discovery forum, served by the dashboard server (see dashboard_server.py).

Reading is open to anyone who can open the dashboard (including the read-only deployment for reviewers). Everything that starts a run is a POST, which the
server already refuses in read-only mode. An episode runs in a background thread of the server and can be stopped (one job at a time: an episode, or a batch whose episodes
run FORUM_BATCH_WORKERS at a time, one after the other by default); its progress is read by
polling `/api/forum/episode?after=<last seq>`. A run that uses OpenAI sends prompts to a third party and costs money, so it starts only with an explicit
acknowledgement from the page.
"""
import datetime as dt
import json
import os
import re
import threading
import uuid
from pathlib import Path

import psycopg

from . import forum, forum_proposal, forum_protocol as proto, forum_sandbox, forum_scoring
from .forum_providers import get_provider

ACKNOWLEDGE = "send-text-to-openai"
POSTS = ("/api/forum/episode/create", "/api/forum/episode/start", "/api/forum/episode/stop", "/api/forum/evaluation/run",
         "/api/forum/batch/create", "/api/forum/batch/start", "/api/forum/batch/stop")
BATCH_FAILURES_IN_A_ROW = 3     # a batch halts after this many failed episodes in a row: the cause is then likely common (a key, a quota, a model id), not the episode
MAX_BATCH_WORKERS = 4           # episodes of one batch that may run at the same time, whatever the setting says


def batch_workers():
    """How many episodes of a batch run at the same time: FORUM_BATCH_WORKERS, default 1 (one after the other, as before). An unreadable value is 1; more than MAX_BATCH_WORKERS is cut."""
    try:
        return max(1, min(MAX_BATCH_WORKERS, int(os.environ.get("FORUM_BATCH_WORKERS", "1"))))
    except ValueError:
        return 1


class ForumJobs:
    """The job being run by this server, if any: one episode, or one batch whose episodes run `workers` at a time."""

    def __init__(self, workers=None):
        self.lock = threading.Lock()
        self.workers = max(1, min(MAX_BATCH_WORKERS, workers)) if workers else batch_workers()
        self.current = None      # {"episode_id" (the first one), "thread", "stop", "active": {episode_id: activity}, "batch": None or {"id", "done", "total", "workers", "halted"}}
        self.rating = None       # the evaluation (novelty, methodology, impact) being run from the Results page: {"thread", "done", "total", "failed"}

    def alive(self):
        with self.lock:
            return bool(self.current and self.current["thread"].is_alive())

    def snapshot(self, episode_id=None):
        """{"running": an episode that runs (the asked one if it does), "running_ids": all of them, "activity": that episode's, "batch": ...}."""
        with self.lock:
            cur = self.current
            batch = dict(cur["batch"]) if cur and cur.get("batch") else None
            alive = bool(cur and cur["thread"].is_alive())
            if batch:
                batch["running"] = alive
            if not alive:
                return {"running": None, "running_ids": [], "activity": "", "batch": batch}
            active = dict(cur["active"])
            ids = list(active) or [cur["episode_id"]]          # between two episodes of a batch nothing is active for a moment
            asked = str(episode_id) if episode_id else None
            running = asked if asked in active else ids[0]
            out = {"running": running, "running_ids": ids, "activity": active.get(running, "") if not asked or asked == running else "", "batch": batch}
            if not asked or asked == running:
                out["stopping"] = cur["stop"].is_set()
            return out

    def start_batch(self, batch_id, episode_ids, provider_name):
        """Run the given episodes of a batch, `self.workers` at a time, each worker taking the next one in order. One stop event serves them all: stopping the episode that runs stops
        the batch, and the episodes not reached stay 'created' (the batch can be resumed). An episode that is no longer 'created' when its turn comes is skipped. The batch halts, with
        no new episode started (those running finish), after BATCH_FAILURES_IN_A_ROW failures in a row, in the order the episodes end."""
        with self.lock:
            if self.current and self.current["thread"].is_alive():
                raise ValueError("Another episode is running: wait for it or stop it first")
            stop, halt, book = threading.Event(), threading.Event(), threading.Lock()
            workers, upcoming, failures = max(1, min(self.workers, len(episode_ids))), iter(episode_ids), [0]
            state = {"episode_id": str(episode_ids[0]), "stop": stop, "active": {},
                     "batch": {"id": str(batch_id), "done": 0, "total": len(episode_ids), "workers": workers, "halted": None}}

            def next_episode():
                with book:
                    return None if stop.is_set() or halt.is_set() else next(upcoming, None)

            def run_episode(episode_id):
                key = str(episode_id)
                state["active"][key] = "starting"
                try:
                    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
                        row = con.execute("SELECT status FROM forum_episode WHERE episode_id=%s", (episode_id,)).fetchone()
                    if not row or row[0] != "created":
                        return "skipped"
                    runner = forum.Runner(episode_id, get_provider(provider_name), stop=stop, on_activity=lambda text: state["active"].__setitem__(key, text))
                    return runner.run()
                except Exception as exc:          # one episode must not end its worker silently: count it as a failure and go on
                    forum.LOG.exception("batch episode could not start")
                    state["batch"]["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    return "failed"
                finally:
                    state["active"].pop(key, None)

            def worker():
                while (episode_id := next_episode()) is not None:
                    status = run_episode(episode_id)
                    with book:
                        state["batch"]["done"] += 1
                        failures[0] = failures[0] + 1 if status == "failed" else 0
                        if failures[0] >= BATCH_FAILURES_IN_A_ROW and not halt.is_set():
                            state["batch"]["halted"] = f"{failures[0]} episodes failed in a row: the batch stopped so as not to spend on a common cause. Read their notes, then run the rest"
                            halt.set()

            def work():
                pool = [threading.Thread(target=worker, name=f"forum-batch-{str(batch_id)[:8]}-{i + 1}", daemon=True) for i in range(workers)]
                for thread in pool:
                    thread.start()
                for thread in pool:
                    thread.join()
            state["thread"] = threading.Thread(target=work, name=f"forum-batch-{str(batch_id)[:8]}", daemon=True)
            self.current = state
            state["thread"].start()

    def stop_batch(self, batch_id):
        with self.lock:
            cur = self.current
            if not cur or not cur["thread"].is_alive() or not cur.get("batch") or cur["batch"]["id"] != str(batch_id):
                raise ValueError("That batch is not running")
            cur["stop"].set()

    def start(self, episode_id, provider_name):
        with self.lock:
            if self.current and self.current["thread"].is_alive():
                raise ValueError("Another episode is running: wait for it or stop it first")
            stop = threading.Event()
            state = {"episode_id": str(episode_id), "stop": stop, "active": {str(episode_id): "starting"}, "batch": None}
            runner = forum.Runner(episode_id, get_provider(provider_name), stop=stop, on_activity=lambda text: state["active"].__setitem__(str(episode_id), text))
            state["thread"] = threading.Thread(target=runner.run, name=f"forum-{str(episode_id)[:8]}", daemon=True)
            self.current = state
            state["thread"].start()

    def rating_snapshot(self):
        with self.lock:
            cur = self.rating
            if not cur:
                return {"running": False}
            return {"running": cur["thread"].is_alive(), "done": cur["done"], "total": cur["total"], "failed": cur["failed"], "error": cur.get("error")}

    def start_rating(self, proposal_ids, provider_name, judge=None):
        """Evaluate the given proposals (novelty, methodology, impact) in a thread of this server, one at a time, so the page can show the progress. Its own connection; the proposals'
        episodes are not touched."""
        with self.lock:
            if self.rating and self.rating["thread"].is_alive():
                raise ValueError("An evaluation is already running: wait for it to finish")
            state = {"done": 0, "total": len(proposal_ids), "failed": 0}
            where = forum_scoring.settings_for(judge)

            def work():
                try:
                    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
                        for proposal_id in proposal_ids:
                            try:
                                result = forum_scoring.score_proposals(con, [proposal_id], get_provider(provider_name), settings_=where)
                                if "stored" not in result or result.get("stopped_by"):
                                    raise ValueError(result.get("skipped") or result.get("stopped_by") or "not evaluated")
                                state["failed"] += result["statuses"]["failed"]
                            except Exception as exc:        # one proposal must not stop the others
                                state["failed"] += 1
                                state["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                            state["done"] += 1
                except Exception as exc:
                    state["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            state["thread"] = threading.Thread(target=work, name="forum-rating", daemon=True)
            self.rating = state
            state["thread"].start()

    def stop(self, episode_id):
        """Stops the episode that runs; in a batch, the rest of the batch too (one stop event)."""
        with self.lock:
            cur = self.current
            if not cur or not cur["thread"].is_alive() or str(episode_id) not in (*cur["active"], cur["episode_id"]):
                raise ValueError("That episode is not running")
            cur["stop"].set()


def condition_cards():
    """Every condition, so that old episodes can be read; `legacy` ones (D, E, F) cannot be created and the form hides them."""
    return [{"id": cid, "name": c["name"], "short": c["short"], "summary": c["summary"], "tests": c["tests"], "interaction": c["interaction"], "organized": c["organized"], "start": c["start"],
             "categories": list(c["categories"]) if c["categories"] else None, "agents": proto.layout(cid), "legacy": bool(c.get("legacy"))} for cid, c in proto.CONDITIONS.items()]


THEME_MODE_LABELS = {"opener_theme": "Theme proposed by the opener from evidence (the design)", "constructs": "Auxiliary: each starts from its own construct database",
                     "rag_cluster": "Auxiliary: a theme built by retrieval", "random_passages": "Auxiliary: random passages"}


def settings_payload(models):
    defaults = proto.normalize_config({"condition": "C", "provider": "openai"}, models)
    return {"conditions": condition_cards(), "design": list(proto.DESIGN), "models": models["models"], "limits": {k: list(v) for k, v in proto.LIMITS.items()},
            "triggers": [{"key": k, "name": name, "instruction": instruction, "move": move, "note": proto.TRIGGER_NOTES.get(k, "")} for k, (name, instruction, move) in proto.TRIGGERS.items()],
            "theme_modes": [{"key": m, "label": THEME_MODE_LABELS[m], "auxiliary": m in proto.AUXILIARY_THEME_MODES} for m in proto.THEME_MODES],
            "code": {"available": forum_sandbox.enabled(), "flag": forum_sandbox.ENV_FLAG, "libraries": forum_sandbox.available_libraries(), "defaults": proto.DEFAULT_CODE},
            "proposal": {"defaults": proto.DEFAULT_PROPOSAL, "sections": [label for _, label, _ in forum_proposal.SECTIONS]},
            "defaults": {"budgets": defaults["budgets"], "protocol": defaults["protocol"], "retrieval": defaults["retrieval"], "diagnoses": defaults["diagnoses"],
                         "encounter": defaults["encounter"], "agent": {k: v for k, v in defaults["agents"]["cto"].items() if not k.endswith("_model")}, "seed": defaults["seed"]},
            "steps": [{"title": t, "text": x} for t, x in proto.STEPS_V4_SOLO], "efforts": ["none", "low", "medium", "high"],
            "providers": {"openai": {"available": bool(os.environ.get("OPENAI_API_KEY")), "label": "OpenAI (sends text to OpenAI, costs money)", "acknowledge": ACKNOWLEDGE}},
            "scoring": forum_scoring.describe(),
            "category_labels": proto.CATEGORY_LABELS, "prompt_version": proto.PROMPT_VERSION_V4}


TEMPLATE = Path(__file__).with_name("forum_template.html")


def render_page(run_id, read_only):
    boot = json.dumps({"run_id": str(run_id), "read_only": bool(read_only)}).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__BOOT__*/null", boot)


def _uuid(value, what="id"):
    try:
        return uuid.UUID(str(value))
    except ValueError:
        raise ValueError(f"Invalid {what}")


def get(path, query, con, ctx):
    """A JSON-able dict, or None if the path is not one of ours."""
    one = lambda key, default="": (query.get(key) or [default])[0]
    if path == "/api/forum/config":
        return {**settings_payload(forum.load_models()), "read_only": ctx.read_only, "run_id": str(ctx.run_id)}
    if path == "/api/forum/releases":
        return {"releases": forum.release_choices(con, ctx.run_id)}
    if path == "/api/forum/release":
        release_id = _uuid(one("id"), "release id")
        row = con.execute("SELECT run_id FROM release WHERE release_id=%s", (release_id,)).fetchone()
        if not row or row[0] != ctx.run_id:
            raise ValueError("Unknown release for this catalog run")
        out = {"release_id": str(release_id), "embedding_models": [], "diagnoses": {}, "diagnoses_with_single_reviewer": {}, "diagnoses_from_signals": {}, "error": None}
        try:
            out["embedding_models"] = forum.embedding_choices(con, release_id)
            counts, with_single = {}, {}
            for item in forum.diagnosis_pool(con, release_id, provisional=True):
                with_single[item["category"]] = with_single.get(item["category"], 0) + 1
                if not item["provisional"]:
                    counts[item["category"]] = counts.get(item["category"], 0) + 1
            out["diagnoses"] = counts
            out["diagnoses_with_single_reviewer"] = with_single
            out["diagnoses_from_signals"] = {c: len(ids) for c, ids in forum.hint_pair_ids(con, release_id).items() if ids}
        except ValueError as exc:
            out["error"] = str(exc)
        return out
    if path == "/api/forum/episodes":
        releases = [uuid.UUID(r["release_id"]) for r in forum.release_choices(con, ctx.run_id)]
        return {"episodes": forum.list_episodes(con, releases), "job": ctx.jobs.snapshot()}
    if path == "/api/forum/episode":
        episode_id = _uuid(one("id"), "episode id")
        try:
            after = int(one("after", "-1"))
        except ValueError:
            raise ValueError("Invalid cursor")
        owner = con.execute("SELECT r.run_id FROM forum_episode e JOIN release r USING (release_id) WHERE e.episode_id=%s", (episode_id,)).fetchone()
        if not owner or owner[0] != ctx.run_id:
            raise ValueError("Unknown episode for this catalog run")
        detail = forum.episode_detail(con, episode_id, after)
        detail["job"] = ctx.jobs.snapshot(episode_id)
        if after < 0:           # the first, full read also carries the episodes that are comparable with this one
            detail["comparable"] = forum.comparable(con, episode_id)
        return detail
    if path == "/api/forum/batch":
        batch_id = _uuid(one("id"), "batch id")
        _batch_owned(con, batch_id, ctx)
        detail = forum.batch_detail(con, batch_id)
        job = ctx.jobs.snapshot()
        detail["job"] = job if job.get("batch") and job["batch"]["id"] == str(batch_id) else {"running": job.get("running"), "batch": None}
        return detail
    if path == "/api/forum/evaluations":          # the Results page: proposals of the current format from real episodes, their three votes and similarity indices
        out = forum_scoring.leaderboard(con, ctx.run_id, budgets=one("budgets") or None)
        out["job"] = ctx.jobs.rating_snapshot()
        out["openai_available"] = bool(os.environ.get("OPENAI_API_KEY"))
        return out
    if path == "/api/forum/evaluation":
        proposal_id = _uuid(one("proposal"), "proposal id")
        _proposal_of_run(con, proposal_id, ctx)
        out = forum_scoring.detail(con, proposal_id)
        if "episode_id" in out:
            out["episode"] = forum.episode_settings(con, out["episode_id"])        # the settings the proposal was generated under, for the reader (never sent to the judges)
        return out
    return None


BINARIES = ("/api/forum/proposal.pdf", "/api/forum/analysis/file", "/api/forum/evaluations.csv")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def binary(path, query, con, ctx):
    """(bytes, content type, file name, inline) for the downloads, or None if the path is not one of ours. The PDF is rendered from the stored text, not stored itself."""
    one = lambda key: (query.get(key) or [""])[0]
    if path == "/api/forum/evaluations.csv":          # the Results table for regressions and other analyses; same rows as /api/forum/evaluations, same budgets filter, nothing sent anywhere
        board = forum_scoring.leaderboard(con, ctx.run_id, budgets=one("budgets") or None)
        return forum_scoring.export_csv(board["rows"]), "text/csv; charset=utf-8", f"forum-results-{dt.date.today().isoformat()}.csv", False
    if path == "/api/forum/proposal.pdf":
        proposal_id = _uuid(one("id"), "proposal id")
        row = con.execute("SELECT p.content,p.refs,p.meta,r.run_id,p.episode_id FROM forum_proposal p JOIN forum_episode e USING (episode_id) JOIN release r ON r.release_id=e.release_id "
                          "WHERE p.proposal_id=%s", (proposal_id,)).fetchone()
        if not row or row[3] != ctx.run_id:
            raise ValueError("Unknown proposal for this catalog run")
        pdf, _ = forum_proposal.render_pdf(row[0], row[2], row[1])
        return pdf, "application/pdf", f"proposal-{str(row[4])[:8]}-{str(proposal_id)[:8]}.pdf", True
    if path == "/api/forum/analysis/file":
        episode_id, run_id, name = _uuid(one("episode"), "episode id"), one("run"), one("name")
        row = con.execute("SELECT f.content,f.media_type,r.run_id FROM forum_analysis_file f JOIN forum_episode e USING (episode_id) JOIN release r ON r.release_id=e.release_id "
                          "WHERE f.episode_id=%s AND f.run_id=%s AND f.name=%s", (episode_id, run_id, name)).fetchone()
        if not row or row[2] != ctx.run_id:
            raise ValueError("Unknown file for this catalog run")
        return bytes(row[0]), row[1], f"{run_id}-{SAFE_NAME.sub('_', name)}", row[1] in ("image/png", "text/plain")
    return None


def post(path, body, con, ctx):
    if path == "/api/forum/episode/create":
        release_id = _uuid(body.get("release_id"), "release id")
        row = con.execute("SELECT run_id FROM release WHERE release_id=%s", (release_id,)).fetchone()
        if not row or row[0] != ctx.run_id:
            raise ValueError("Unknown release for this catalog run")
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config is required")
        if body.get("start") is True:          # creating sends nothing anywhere; the confirmation is asked when the episode starts (here or with /start)
            _check_acknowledged(config.get("provider", "openai"), body)
        episode_id = forum.create_episode(con, release_id, config, str(body.get("created_by", "")))
        out = {"episode_id": str(episode_id), "started": False}
        if body.get("start") is True:
            ctx.jobs.start(episode_id, config.get("provider", "openai"))
            out["started"] = True
        return out
    if path == "/api/forum/episode/start":
        episode_id = _uuid(body.get("episode_id"), "episode id")
        row = con.execute("SELECT e.status,e.provider,r.run_id FROM forum_episode e JOIN release r USING (release_id) WHERE e.episode_id=%s", (episode_id,)).fetchone()
        if not row or row[2] != ctx.run_id:
            raise ValueError("Unknown episode for this catalog run")
        if row[0] != "created":
            raise ValueError(f"An episode that is {row[0]} cannot be started again: create a new one (copy its settings)")
        if row[1] not in proto.PROVIDERS:
            raise ValueError("This is a simulated episode: the simulation was removed, so it can be read but not run. Copy its settings into a new episode with a real model")
        _check_acknowledged(row[1], body)
        ctx.jobs.start(episode_id, row[1])
        return {"episode_id": str(episode_id), "started": True}
    if path == "/api/forum/episode/stop":
        episode_id = _uuid(body.get("episode_id"), "episode id")
        ctx.jobs.stop(episode_id)
        return {"episode_id": str(episode_id), "stopping": True}
    if path == "/api/forum/batch/create":
        release_id = _uuid(body.get("release_id"), "release id")
        row = con.execute("SELECT run_id FROM release WHERE release_id=%s", (release_id,)).fetchone()
        if not row or row[0] != ctx.run_id:
            raise ValueError("Unknown release for this catalog run")
        config = body.get("config")
        if not isinstance(config, dict):
            raise ValueError("config is required")
        if body.get("start") is True:
            _check_acknowledged(config.get("provider", "openai"), body)
        batch_id, ids = forum.create_batch(con, release_id, config, str(body.get("created_by", "")), body.get("repetitions"), body.get("seed_mode") or "vary")
        out = {"batch_id": str(batch_id), "episode_ids": [str(i) for i in ids], "started": False}
        if body.get("start") is True:
            ctx.jobs.start_batch(batch_id, ids, config.get("provider", "openai"))
            out["started"] = True
        return out
    if path == "/api/forum/batch/start":          # also resumes a batch that was stopped, halted or cut by a server restart: only its 'created' episodes run
        batch_id = _uuid(body.get("batch_id"), "batch id")
        _batch_owned(con, batch_id, ctx)
        rows = forum.batch_episodes(con, batch_id)
        todo = [(e, provider) for e, status, _, provider in rows if status == "created"]
        if not todo:
            raise ValueError("Every episode of this batch has already run: create a new batch to run it again")
        if any(provider not in proto.PROVIDERS for _, provider in todo):
            raise ValueError("This batch has simulated episodes: the simulation was removed, so they can be read but not run")
        _check_acknowledged(todo[0][1], body)
        ctx.jobs.start_batch(batch_id, [e for e, _ in todo], todo[0][1])
        return {"batch_id": str(batch_id), "started": len(todo)}
    if path == "/api/forum/batch/stop":
        batch_id = _uuid(body.get("batch_id"), "batch id")
        ctx.jobs.stop_batch(batch_id)
        return {"batch_id": str(batch_id), "stopping": True}
    if path == "/api/forum/evaluation/run":          # novelty, methodology, impact; the texts go to OpenAI (embeddings and the judges) and cost money
        if body.get("proposal_id"):
            wanted = _uuid(body["proposal_id"], "proposal id")
            row = con.execute(f"SELECT e.provider, {forum_scoring.NEW_FORMAT_SQL} FROM forum_proposal p JOIN forum_episode e USING (episode_id) JOIN release r ON r.release_id=e.release_id "
                              "WHERE p.proposal_id=%s AND r.run_id=%s", (wanted, ctx.run_id)).fetchone()
            if not row:
                raise ValueError("Unknown proposal for this catalog run")
            if row[0] != "openai":
                raise ValueError("Only proposals of OpenAI episodes are evaluated")
            if not row[1]:
                raise ValueError("This proposal is of the earlier format (no cards): it is not evaluated")
            jobs = [wanted]                          # asked for by name: evaluated again even if it already has a score (a new row, the old one stays)
        elif body.get("pending") is True:
            jobs = forum_scoring.pending(con, ctx.run_id)
        else:
            raise ValueError("Say which proposal (proposal_id) or ask for the pending ones (pending: true)")
        if not jobs:
            return {"started": 0}
        forum_scoring.settings_for(body.get("judge"))        # an unknown judge is refused before anything is sent
        _check_acknowledged("openai", body)
        ctx.jobs.start_rating([str(p) for p in jobs], "openai", body.get("judge"))
        return {"started": len(jobs)}
    return None


def _proposal_of_run(con, proposal_id, ctx):
    owner = con.execute("SELECT r.run_id FROM forum_proposal p JOIN forum_episode e USING (episode_id) JOIN release r ON r.release_id=e.release_id WHERE p.proposal_id=%s", (proposal_id,)).fetchone()
    if not owner or owner[0] != ctx.run_id:
        raise ValueError("Unknown proposal for this catalog run")


def _batch_owned(con, batch_id, ctx):
    rows = con.execute("SELECT DISTINCT r.run_id FROM forum_episode e JOIN release r USING (release_id) WHERE e.config->'batch'->>'id'=%s", (str(batch_id),)).fetchall()
    if not rows or any(r[0] != ctx.run_id for r in rows):
        raise ValueError("Unknown batch for this catalog run")


def _check_acknowledged(provider, body):
    if provider == "openai":
        if body.get("acknowledge") != ACKNOWLEDGE:
            raise ValueError("This run sends the prompts, with passages from the corpus, to OpenAI and costs money: confirm it in the page")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set on the server")
