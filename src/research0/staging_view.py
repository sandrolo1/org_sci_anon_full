"""A parallel, read-only view of an extraction run waiting in staging (migration 026), for the dashboard (13 Oct 2026, the user's request).

The staged rows live in the `staged_*` twin tables. This module creates a schema of views that carry the LIVE names (construct_mention, extraction_run, ...)
and read the staged twins; a connection whose search_path starts with that schema sees the staged constructs where the dashboard expects the live ones, and
everything else (catalog, memberships, decisions, signals) from the live schema. Nothing is copied and nothing is promoted: promotion stays
`research0 extract-promote`, a person's decision. The dashboard started on it is always read-only and says, on every page, that it shows staged data.
"""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .extraction import EXTRACTION_TABLES


def schema_name(base):
    """The schema of the views for the live schema `base` (one per schema, so that the tests' throw-away schemas do not share it)."""
    return "staging_view" if base == "public" else f"staging_view_{base}"


def create(con):
    """Creates (or refreshes) the views; returns the schema name. The agents' role gets no access to it."""
    base = con.execute("SELECT current_schema()").fetchone()[0]
    view = schema_name(base)
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{view}"')
    for table in EXTRACTION_TABLES:
        con.execute(f'CREATE OR REPLACE VIEW "{view}"."{table}" AS SELECT * FROM "{base}"."staged_{table}"')
    if con.execute("SELECT 1 FROM pg_roles WHERE rolname='research_agent'").fetchone():
        con.execute(f'REVOKE ALL ON SCHEMA "{view}" FROM research_agent')
        con.execute(f'REVOKE ALL ON ALL TABLES IN SCHEMA "{view}" FROM research_agent')
    return view


def connection_url(url, view, base):
    """`url` with a search_path that puts the views first and then the live schema (any search_path already in the URL is replaced)."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not (k == "options" and "search_path" in v)]
    query.append(("options", f"-csearch_path={view},{base}"))
    return urlunsplit(parts._replace(query=urlencode(query)))


def describe(con):
    """What the parallel view shows: the staged runs with their row counts and whether they were promoted."""
    from .extraction import staged_runs
    return staged_runs(con)
