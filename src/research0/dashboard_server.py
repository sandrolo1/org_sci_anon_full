"""Local live dashboard: reads the catalog from the database in DATABASE_URL and stores validator decisions there.

Bound to the loopback interface unless DASHBOARD_BIND says otherwise (a container behind a reverse proxy). Decisions are appended to
validation_decision, never updated. For a deployment other people read, set DASHBOARD_ALLOWED_HOSTS to the public host name(s) (the Host and
Origin headers are checked against them) and DASHBOARD_READ_ONLY=1 so that nobody can write, spend money (Batch, Unpaywall) or upload through it.
"""
import json
import logging
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import psycopg
from botocore.exceptions import BotoCoreError, ClientError

from .dashboard import build_snapshot, current_decisions, render_html
from . import community_dashboard as community
from . import dashboard_api
from . import fulltext
from . import paper_evaluation
from . import paper_processing

LOG = logging.getLogger("research0")
MAX_BODY = 16 * 1024
# Full-text download mode (Unpaywall, OpenAlex and Semantic Scholar by DOI, started from the dashboard) can only be activated in an admin mode, which does not exist yet:
# until it does, nobody can start it from here. Manual uploads and the command line (research0 fulltext-fetch) are not affected.
FULLTEXT_DOWNLOAD_ADMIN_MODE = False
# Same for the CTO/TIM labelling Batch (POST /api/community/start sends the catalog's titles and abstracts to OpenAI and costs money).
COMMUNITY_BATCH_ADMIN_MODE = False
# Same for the admin page of the evaluation of published papers (sample, full texts, cards, runs: it downloads, spends and writes). Every write under /api/evaluation/ and the
# reads that only that page makes are refused; the four reads of the results of a run (page "Pilot results") stay open.
EVALUATION_ADMIN_MODE = False
EVALUATION_RESULT_READS = ("/api/evaluation/runs", "/api/evaluation/run", "/api/evaluation/paper", "/api/evaluation/export")
EVALUATION_LOCKED = "the evaluation admin page is locked: it can only be activated in admin mode"


def hostname_of(host_header):
    host = (host_header or "").strip().lower()
    return host.rsplit(":", 1)[0] if not host.endswith("]") and ":" in host else host


def make_handler(run_id, allowed_hosts=(), read_only=False):
    allowed_hosts = {h.strip().lower() for h in allowed_hosts if h.strip()}
    # A catalog run is immutable, so statistics and items are computed once; only decisions are re-read.
    cache = {}
    lock = threading.Lock()
    job = fulltext.FulltextJob()
    papers_job = paper_processing.ProcessingJob()       # the one click that sends constructs, methodology and impact to three Batches
    evaluation_job = paper_evaluation.EvaluationJob()   # one run of the novelty judges over the 2026 papers, in a thread of this server
    stores = {}

    def get_store():
        with lock:
            if "s3" not in stores:
                stores["s3"] = fulltext.S3Store()
            return stores["s3"]

    def base_snapshot(con):
        with lock:
            if "snapshot" not in cache:
                cache["snapshot"] = build_snapshot(con, run_id)
            return cache["snapshot"]

    class Handler(BaseHTTPRequestHandler):
        server_version = "research0-dashboard"

        def log_message(self, fmt, *args):
            LOG.info(json.dumps({"event": "http", "run_id": str(run_id), "request": fmt % args}))

        def _send(self, status, body, content_type, headers=None):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, ensure_ascii=False, default=str), "application/json; charset=utf-8")

        def _host_ok(self):
            # Refuse requests whose Host is not this loopback server (DNS rebinding).
            port = self.server.server_address[1]
            host = self.headers.get("Host")
            return host in (f"127.0.0.1:{port}", f"localhost:{port}") or hostname_of(host) in allowed_hosts

        def _origin_ok(self):
            # A page served from this server sends Origin = scheme://Host. https is accepted only for a host we were told to serve.
            origin, host = self.headers.get("Origin"), self.headers.get("Host")
            return origin is None or origin == f"http://{host}" or (origin == f"https://{host}" and hostname_of(host) in allowed_hosts)

        def _live_snapshot(self):
            with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                snapshot = dict(base_snapshot(con), live=True, decisions=current_decisions(con, run_id))
            snapshot["limits"] = ["Decisions are stored as append-only records in PostgreSQL; a human decision does not change the catalog or validate a Scopus source automatically."] + snapshot["limits"][1:]
            snapshot["assignment"] = community.assignment_status(run_id)
            if os.environ.get("DASHBOARD_STAGING") == "1":       # the parallel dashboard of a staged extraction run (cli dashboard-serve --staged)
                from .staging_view import describe
                with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                    snapshot["staging"] = describe(con)
            return snapshot

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "unexpected Host header"})
            try:
                if self.path in ("/", "/index.html"):
                    return self._send(200, render_html(self._live_snapshot()), "text/html; charset=utf-8")
                if self.path == "/api/snapshot":
                    return self._json(200, self._live_snapshot())
                if self.path == "/api/community/status":
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, {"assignment": community.assignment_status(run_id),
                                                "sample": community.sample_status(con, run_id)})
                if self.path.startswith("/api/community/stats"):    # read-only figures and examples of the Batch labels; the labelling Batch itself stays locked
                    seed = int(parse_qs(urlsplit(self.path).query).get("seed", ["0"])[0] or 0)
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, community.stats(con, run_id, seed))
                if self.path.startswith("/api/community/blind?validator="):
                    name = parse_qs(urlsplit(self.path).query).get("validator", [""])[0].strip()
                    if not name or len(name) > 80:
                        return self._json(400, {"error": "validator name required"})
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, {"papers": community.blind_papers(con, run_id, name),
                                                "sample": community.sample_status(con, run_id, name)})
                if self.path == "/api/community/adjudication":
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, {"papers": community.adjudication_items(con, run_id)})
                if self.path.startswith(("/api/audit/", "/api/pairs/", "/api/release/", "/api/constructs/", "/api/cards/")):
                    url = urlsplit(self.path)
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        result = dashboard_api.get(url.path, parse_qs(url.query), con, run_id)
                    if result is not None:
                        return self._json(200, result)
                if self.path.startswith("/api/evaluation/"):
                    url = urlsplit(self.path)
                    query = parse_qs(url.query)
                    one = lambda key: query.get(key, [""])[0]
                    if not EVALUATION_ADMIN_MODE and url.path not in EVALUATION_RESULT_READS:
                        return self._json(403, {"error": EVALUATION_LOCKED})
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        active = evaluation_job.running_id()
                        if url.path == "/api/evaluation/meta":
                            return self._json(200, {**paper_evaluation.meta(con), **paper_evaluation.describe(), "job": evaluation_job.snapshot(), "read_only": read_only})
                        if url.path == "/api/evaluation/status":
                            return self._json(200, {"job": evaluation_job.snapshot()})
                        if url.path == "/api/evaluation/runs":
                            return self._json(200, paper_evaluation.list_runs(con, active))
                        if url.path == "/api/evaluation/run":
                            return self._json(200, paper_evaluation.run_detail(con, uuid.UUID(one("id")), active))
                        if url.path == "/api/evaluation/export":      # a file of the results of one run; a read like the others, so it also works on a read-only deployment
                            name, content_type, data = paper_evaluation.export_run(con, uuid.UUID(one("id")), one("format") or "csv", active)
                            return self._send(200, data, content_type, {"Content-Disposition": f'attachment; filename="{name}"'})
                        filters = lambda: {"min_words": int(one("min_words") or paper_evaluation.DEFAULT_MIN_WORDS), "require_grade": one("require_grade") != "0",
                                           "seed": int(one("seed") or paper_evaluation.DEFAULT_SEED)}
                        if url.path == "/api/evaluation/reconcile":    # every paper of the set, the stage it stops at and why; reads only
                            return self._json(200, paper_evaluation.reconcile(con, active=active, **filters()))
                        if url.path == "/api/evaluation/to-fetch":     # the Excel list of the papers to fetch by hand; reads only
                            name, content_type, data = paper_evaluation.export_to_fetch(con, active=active, **filters())
                            return self._send(200, data, content_type, {"Content-Disposition": f'attachment; filename="{name}"'})
                        if url.path == "/api/evaluation/paper":
                            return self._json(200, paper_evaluation.paper_detail(con, uuid.UUID(one("run")), one("id")))
                        if url.path == "/api/evaluation/pipeline":
                            return self._json(200, paper_evaluation.pipeline(con, n=int(one("n") or paper_evaluation.DEFAULT_N), seed=int(one("seed") or paper_evaluation.DEFAULT_SEED),
                                                                             min_words=int(one("min_words") or paper_evaluation.DEFAULT_MIN_WORDS), require_grade=one("require_grade") != "0",
                                                                             require_fulltext=one("require_fulltext") != "0", extend_run=one("extend_run") or None, stratify=one("stratify") == "1"))
                if self.path.startswith("/api/papers/"):
                    url = urlsplit(self.path)
                    if url.path == "/api/papers/status":          # light: the job in memory and the files of the latest job, no database
                        last = paper_processing.latest_job()
                        return self._json(200, {"job": papers_job.snapshot(), "last": paper_processing.job_status(last) if last else None,
                                                "openai_configured": bool(os.environ.get("OPENAI_API_KEY")), "read_only": read_only})
                    if url.path == "/api/papers/estimate":        # heavier: reads the texts of the papers still to send, so the page asks for it only when it needs it
                        limit = parse_qs(url.query).get("limit", [""])[0]
                        with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                            return self._json(200, paper_processing.estimate(con, run_id, int(limit) if limit else None))
                if self.path.startswith("/api/fulltext/"):
                    url = urlsplit(self.path)
                    query = parse_qs(url.query)
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        if url.path == "/api/fulltext/status":
                            return self._json(200, {"job": job.snapshot(), "totals": fulltext.totals(con, run_id), "scopes": list(fulltext.SCOPES),
                                                    "storage": f"s3://{os.environ.get('S3_BUCKET', '')}/{fulltext.S3Store.prefix}",
                                                    "unpaywall_configured": bool(os.environ.get("UNPAYWALL_EMAIL", "").strip())})
                        if url.path == "/api/fulltext/missing":
                            offset, limit = int(query.get("offset", ["0"])[0]), min(int(query.get("limit", ["50"])[0]), 200)
                            if offset < 0 or limit < 1:
                                raise ValueError("invalid paging")
                            return self._json(200, {"papers": fulltext.missing(con, run_id, offset, limit)})
                        if url.path == "/api/fulltext/missing.csv":
                            name = f"fulltext_to_get_by_hand_{str(run_id)[:8]}.csv"
                            return self._send(200, fulltext.missing_csv(con, run_id), "text/csv; charset=utf-8",
                                              {"Content-Disposition": f'attachment; filename="{name}"'})
            except psycopg.Error as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                return self._json(503, {"error": "database unavailable"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)[:300]})
            self._json(404, {"error": "not found"})

        def _upload(self):
            # A PDF chosen by a person for one paper; the paper, run and uploader travel in the query string.
            query = parse_qs(urlsplit(self.path).query)
            one = lambda key: query.get(key, [""])[0]
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/pdf":
                return self._json(415, {"error": "PDF body required"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = fulltext.MAX_PDF_BYTES + 1
            if not 0 < length <= fulltext.MAX_PDF_BYTES:
                return self._json(413, {"error": "the file is empty or larger than 60 MB"})
            if one("run_id") != str(run_id):
                return self._json(409, {"error": "this page belongs to a different catalog run; reload it"})
            try:
                data = self.rfile.read(length)
                with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                    return self._json(200, fulltext.save_upload(con, get_store(), run_id, uuid.UUID(one("paper_id")), data, one("filename"), one("uploader")))
            except ValueError as exc:
                return self._json(400, {"error": str(exc)[:300]})
            except psycopg.Error as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                return self._json(503, {"error": "database unavailable"})
            except (BotoCoreError, ClientError, KeyError) as exc:
                LOG.error(json.dumps({"event": "storage_error", "error": str(exc)[:200]}))
                return self._json(503, {"error": "file storage unavailable (check S3_BUCKET and the S3 settings)"})

        def do_POST(self):
            is_upload = urlsplit(self.path).path == "/api/fulltext/upload"
            if not is_upload and self.path not in ("/api/decisions", "/api/community/start", "/api/community/refresh",
                                                   "/api/community/sample", "/api/community/rate", "/api/community/agreement",
                                                   "/api/community/adjudicate", "/api/fulltext/start", "/api/papers/process", "/api/papers/refresh", "/api/evaluation/start", "/api/evaluation/stop", "/api/evaluation/fulltext", "/api/evaluation/download", "/api/evaluation/cards", "/api/evaluation/cards-batch", "/api/evaluation/cards-refresh", *dashboard_api.POSTS):
                return self._json(404, {"error": "not found"})
            if not self._host_ok() or not self._origin_ok():
                return self._json(403, {"error": "cross-origin request refused"})
            if read_only:
                return self._json(403, {"error": "this deployment is read-only"})
            if self.path == "/api/fulltext/start" and not FULLTEXT_DOWNLOAD_ADMIN_MODE:
                return self._json(403, {"error": "full-text download mode is locked: it can only be activated in admin mode"})
            if self.path == "/api/community/start" and not COMMUNITY_BATCH_ADMIN_MODE:
                return self._json(403, {"error": "the CTO/TIM labelling Batch is locked: it can only be activated in admin mode"})
            if self.path.startswith("/api/evaluation/") and not EVALUATION_ADMIN_MODE:
                return self._json(403, {"error": EVALUATION_LOCKED})
            if is_upload:
                return self._upload()
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "JSON body required"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = MAX_BODY + 1
            if not 0 < length <= MAX_BODY:
                return self._json(413, {"error": "body too large or empty"})
            try:
                body = json.loads(self.rfile.read(length))
                if body.get("run_id") != str(run_id):
                    return self._json(409, {"error": "this page belongs to a different catalog run; reload it"})
                if self.path == "/api/fulltext/start":
                    return self._json(200, job.start(run_id, str(body.get("scope", "")), body.get("limit"),
                                                     os.environ.get("UNPAYWALL_EMAIL", "").strip(), os.environ["DATABASE_URL"]))
                if self.path.startswith("/api/evaluation/") and self.path != "/api/evaluation/stop":
                    sample_options = {k: body[k] for k in ("n", "seed", "min_words", "require_grade", "require_fulltext", "extend_run", "stratify") if k in body}
                    if self.path == "/api/evaluation/start":
                        if body.get("acknowledge_send_to_openai") is not True:
                            raise ValueError("the titles, abstracts and cards are sent to OpenAI and this costs money: confirm it on the page")
                        options = {**sample_options, **{k: body[k] for k in ("max_tokens", "label", "criteria") if k in body}}
                        return self._json(200, evaluation_job.start(os.environ["DATABASE_URL"], resume=body.get("resume"), **options))
                    if self.path == "/api/evaluation/download":       # tries the open-access PDFs of the usable papers that have none (Unpaywall, OpenAlex, Semantic Scholar), imports and parses them
                        return self._json(200, evaluation_job.start_download(os.environ["DATABASE_URL"], limit=body.get("limit"),
                                                                             **{k: body[k] for k in ("min_words", "require_grade", "seed") if k in body}))
                    if self.path == "/api/evaluation/fulltext":       # imports the full texts the project already has (archive, downloads, files put by hand) and parses them
                        return self._json(200, evaluation_job.start_fulltext(os.environ["DATABASE_URL"], **sample_options))
                    if self.path == "/api/evaluation/cards":          # costs money: the page sends it only after the box is ticked
                        if body.get("acknowledge_send_to_openai") is not True:
                            raise ValueError("the full texts are sent to OpenAI to make the cards and this costs money: confirm it on the page")
                        return self._json(200, evaluation_job.start_cards(os.environ["DATABASE_URL"], max_tokens=body.get("max_tokens"), **sample_options))
                    if self.path == "/api/evaluation/cards-batch":    # the same cards by Batch: half the price, answers within 24 hours; costs money too
                        if body.get("acknowledge_send_to_openai") is not True:
                            raise ValueError("the full texts are sent to OpenAI to make the cards and this costs money: confirm it on the page")
                        return self._json(200, evaluation_job.submit_cards_batch(os.environ["DATABASE_URL"], max_tokens=body.get("max_tokens"), **sample_options))
                    if self.path == "/api/evaluation/cards-refresh":  # asks OpenAI for the state of the Batches and stores the answers of those that ended; sends nothing
                        return self._json(200, {"batches": evaluation_job.refresh_cards_batches(os.environ["DATABASE_URL"])})
                if self.path == "/api/evaluation/stop":
                    return self._json(200, evaluation_job.stop())
                if self.path == "/api/papers/process":
                    return self._json(200, papers_job.start(run_id, body.get("limit"), body.get("max_input_tokens"), os.environ["DATABASE_URL"], get_store()))
                if self.path == "/api/papers/refresh":
                    last = paper_processing.latest_job()
                    if last is None:
                        raise ValueError("No job has been started yet")
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, {"last": paper_processing.refresh_job(con, last)})
                if self.path in dashboard_api.POSTS:
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        return self._json(200, dashboard_api.post(self.path, body, con, run_id))
                if self.path.startswith("/api/community/"):
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        if self.path == "/api/community/start":
                            result = community.start(con, run_id)
                        elif self.path == "/api/community/refresh":
                            result = community.refresh(run_id)
                        elif self.path == "/api/community/sample":
                            result = community.create_sample(con, run_id, threshold=float(body.get("threshold", community.EXPECTED_KAPPA)))
                        elif self.path == "/api/community/rate":
                            community.rate(con, run_id, uuid.UUID(body["paper_id"]), str(body.get("validator", "")).strip(), body.get("cto"), body.get("tim"))
                            result = community.sample_status(con, run_id, str(body.get("validator", "")).strip())
                        elif self.path == "/api/community/adjudicate":
                            community.adjudicate(con, run_id, uuid.UUID(body["paper_id"]), str(body.get("adjudicator", "")).strip(), body.get("cto"), body.get("tim"))
                            result = {"saved": True}
                        else:
                            result = community.agreement(con, run_id, allow_partial=body.get("allow_partial") is True)
                    return self._json(200, result)
                validator, note, item_id, decision = (str(body.get(k, "")).strip() for k in ("validator", "note", "item_id", "decision"))
                if not validator or len(validator) > 80 or len(note) > 2000:
                    return self._json(400, {"error": "validator name (max 80) is required; note max 2000 characters"})
                with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                    item = next((i for i in base_snapshot(con)["items"] if i["id"] == item_id), None)
                    if item is None:
                        return self._json(404, {"error": "unknown item"})
                    if decision not in item["decisions"] + ["withdrawn"]:
                        return self._json(400, {"error": "decision not allowed for this item"})
                    con.execute("INSERT INTO validation_decision(run_id,item_id,item_type,decision,note,validator) VALUES (%s,%s,%s,%s,%s,%s)",
                                (run_id, item_id, item["type"], decision, note, validator))
                    decisions = current_decisions(con, run_id).get(item_id, [])
                LOG.info(json.dumps({"event": "decision_recorded", "run_id": str(run_id), "item_id": item_id, "decision": decision, "validator": validator}))
                self._json(200, {"decisions": decisions})
            except (ValueError, AttributeError, KeyError) as exc:
                self._json(400, {"error": str(exc)[:300] or "invalid JSON"})
            except psycopg.Error as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                self._json(503, {"error": "database unavailable"})

    return Handler


def serve(run_id, port=8765, host=None, allowed_hosts=None, read_only=None):
    run_id = uuid.UUID(run_id)
    host = host or os.environ.get("DASHBOARD_BIND", "127.0.0.1")
    if allowed_hosts is None:
        allowed_hosts = os.environ.get("DASHBOARD_ALLOWED_HOSTS", "").split(",")
    if read_only is None:
        read_only = os.environ.get("DASHBOARD_READ_ONLY", "").strip().lower() in ("1", "true", "yes")
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        if not con.execute("SELECT 1 FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone():
            raise ValueError("Unknown catalog run")
    server = ThreadingHTTPServer((host, port), make_handler(run_id, allowed_hosts, read_only))
    return server, run_id
