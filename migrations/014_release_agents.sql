-- Phase 7 and 8. Human adjudication of candidate pairs, gold sets for the quality gates, the validated release, the
-- retrieval log, and the read-only agent role. pair_decision and release_signoff are append-only; the latest decision
-- per (pair, validator) is current.
CREATE TABLE IF NOT EXISTS pair_decision (
  decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  pair_id uuid NOT NULL REFERENCES candidate_pair(pair_id),
  relation text NOT NULL CHECK (relation IN ('jingle','jangle','partial_overlap','distinct','unresolved')),
  rationale text NOT NULL CHECK (length(btrim(rationale)) > 0),
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  is_final boolean NOT NULL DEFAULT false,
  decided_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS pair_decision_latest_idx ON pair_decision(pair_id, validator, decided_at DESC);
CREATE TABLE IF NOT EXISTS gold_set (
  gold_id uuid PRIMARY KEY,
  kind text NOT NULL CHECK (kind IN ('extraction','pairs','ratings')),
  label text NOT NULL,
  content jsonb NOT NULL,
  content_sha256 text NOT NULL,
  provided_by text NOT NULL CHECK (length(btrim(provided_by)) > 0),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (kind, content_sha256)
);
CREATE TABLE IF NOT EXISTS release (
  release_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  label text NOT NULL,
  gates_passed boolean NOT NULL,
  gates jsonb NOT NULL,
  manifest jsonb NOT NULL,
  manifest_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS release_signoff (
  signoff_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  release_id uuid NOT NULL REFERENCES release(release_id),
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  note text NOT NULL DEFAULT '',
  signed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS retrieval_run (
  retrieval_run_id uuid PRIMARY KEY,
  release_id uuid REFERENCES release(release_id),
  query_text text NOT NULL,
  method text NOT NULL,
  params jsonb NOT NULL,
  agent text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS retrieval_hit (
  retrieval_run_id uuid NOT NULL REFERENCES retrieval_run(retrieval_run_id),
  rank integer NOT NULL,
  chunk_id uuid NOT NULL REFERENCES document_chunk(chunk_id),
  score double precision NOT NULL,
  components jsonb NOT NULL,
  PRIMARY KEY (retrieval_run_id, rank)
);
-- Discovery agents connect as research_agent: they read everything and may only append their own retrieval log.
-- The role has no password and cannot log in until an administrator sets one (ALTER ROLE research_agent LOGIN PASSWORD ...).
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    CREATE ROLE research_agent NOLOGIN;
  END IF;
  EXECUTE format('GRANT USAGE ON SCHEMA %I TO research_agent', current_schema());
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
  EXECUTE format('GRANT INSERT ON %I.retrieval_run, %I.retrieval_hit TO research_agent', current_schema(), current_schema());
END
$$;
