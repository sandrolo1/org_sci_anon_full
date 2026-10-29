"""The discovery forum as a service of its own: `research0 forum-serve`.

It shares nothing with the validation dashboard except the database it reads (a release, normally a validated one) and the code of the forum. It has its own
process, port, host name and settings, so it can be published on a different domain and behind different credentials, and the dashboard stays
unaware of it. Settings (all optional, safe defaults for a laptop):

  FORUM_BIND            interface to bind (default 127.0.0.1; a container behind a reverse proxy sets 0.0.0.0)
  FORUM_ALLOWED_HOSTS   comma-separated public host names; the Host and Origin headers are checked against them (DNS rebinding, cross-site writes)
  FORUM_READ_ONLY       1 = people can follow every episode but cannot create, start or stop one (and nothing can spend money through the page)
  FORUM_BATCH_WORKERS   episodes of one batch that run at the same time (default 1 = one after the other; at most 4). The cost is the same, the wait shorter; each one needs memory

An episode that was running when the server stopped is marked failed at start-up.
"""
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import uuid

import psycopg

from . import forum, forum_api

LOG = logging.getLogger("research0")
MAX_BODY = 32 * 1024


def hostname_of(host_header):
    host = (host_header or "").strip().lower()
    return host.rsplit(":", 1)[0] if not host.endswith("]") and ":" in host else host


def make_handler(run_id, allowed_hosts=(), read_only=False):
    allowed_hosts = {h.strip().lower() for h in allowed_hosts if h.strip()}
    ctx = SimpleNamespace(run_id=run_id, jobs=forum_api.ForumJobs(), read_only=read_only)

    class Handler(BaseHTTPRequestHandler):
        server_version = "research0-forum"

        def log_message(self, fmt, *args):
            LOG.info(json.dumps({"event": "http", "service": "forum", "request": fmt % args}))

        def _send(self, status, body, content_type, disposition=None):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if disposition:
                self.send_header("Content-Disposition", disposition)
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
            port = self.server.server_address[1]
            host = self.headers.get("Host")
            return host in (f"127.0.0.1:{port}", f"localhost:{port}") or hostname_of(host) in allowed_hosts

        def _origin_ok(self):
            origin, host = self.headers.get("Origin"), self.headers.get("Host")
            return origin is None or origin == f"http://{host}" or (origin == f"https://{host}" and hostname_of(host) in allowed_hosts)

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "unexpected Host header"})
            url = urlsplit(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, forum_api.render_page(run_id, read_only), "text/html; charset=utf-8")
                if url.path == "/healthz":
                    return self._json(200, {"ok": True})
                if url.path in forum_api.BINARIES:
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        found = forum_api.binary(url.path, parse_qs(url.query), con, ctx)
                    data, content_type, name, inline = found
                    return self._send(200, data, content_type, f'{"inline" if inline else "attachment"}; filename="{name}"')
                if url.path.startswith("/api/forum/"):
                    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                        result = forum_api.get(url.path, parse_qs(url.query), con, ctx)
                    if result is not None:
                        return self._json(200, result)
            except psycopg.errors.UndefinedTable as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                return self._json(503, {"error": "the database lacks a table the forum needs: run research0 init to apply the migrations"})
            except psycopg.Error as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                return self._json(503, {"error": "database unavailable"})
            except ValueError as exc:
                return self._json(400, {"error": str(exc)[:300]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in forum_api.POSTS:
                return self._json(404, {"error": "not found"})
            if not self._host_ok() or not self._origin_ok():
                return self._json(403, {"error": "cross-origin request refused"})
            if read_only:
                return self._json(403, {"error": "this deployment is read-only"})
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
                with psycopg.connect(os.environ["DATABASE_URL"]) as con:
                    return self._json(200, forum_api.post(self.path, body, con, ctx))
            except (ValueError, AttributeError, KeyError) as exc:
                self._json(400, {"error": str(exc)[:300] or "invalid JSON"})
            except psycopg.Error as exc:
                LOG.error(json.dumps({"event": "database_error", "error": str(exc)[:200]}))
                self._json(503, {"error": "database unavailable"})

    return Handler


def serve(run_id, port=8770, host=None, allowed_hosts=None, read_only=None):
    run_id = uuid.UUID(run_id)
    host = host or os.environ.get("FORUM_BIND", "127.0.0.1")
    if allowed_hosts is None:
        allowed_hosts = os.environ.get("FORUM_ALLOWED_HOSTS", "").split(",")
    if read_only is None:
        read_only = os.environ.get("FORUM_READ_ONLY", "").strip().lower() in ("1", "true", "yes")
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        if not con.execute("SELECT 1 FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone():
            raise ValueError("Unknown catalog run")
        try:
            if forum.mark_orphans(con):
                LOG.warning(json.dumps({"event": "forum_episodes_marked_failed", "reason": "server restarted while they were running"}))
        except psycopg.errors.UndefinedTable:
            con.rollback()      # the database predates migration 020; the page says so
    server = ThreadingHTTPServer((host, port), make_handler(run_id, allowed_hosts, read_only))
    return server, run_id
