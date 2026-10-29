-- Phase 1: canonical catalog derived from export records. Export records are never
-- altered or deleted; duplicates are linked to a canonical paper and every link is logged.
CREATE TABLE IF NOT EXISTS catalog_run (
  run_id uuid PRIMARY KEY,
  query_id uuid NOT NULL REFERENCES query_version(query_id),
  dedup_version text NOT NULL,
  batch_ids uuid[] NOT NULL,
  input_record_count integer NOT NULL,
  canonical_count integer NOT NULL,
  summary jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS canonical_paper (
  canonical_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  representative_record_id uuid NOT NULL REFERENCES export_record(record_id),
  doi text,
  eid text,
  all_dois text[] NOT NULL DEFAULT '{}',
  all_eids text[] NOT NULL DEFAULT '{}',
  title text,
  authors text,
  publication_year integer,
  source_title text,
  document_type text,
  member_count integer NOT NULL CHECK (member_count >= 1),
  batch_count integer NOT NULL CHECK (batch_count >= 1),
  status text NOT NULL CHECK (status IN ('retained','review_needed')),
  perimeter_flags text[] NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS canonical_paper_run_idx ON canonical_paper(run_id);
CREATE TABLE IF NOT EXISTS catalog_member (
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  record_id uuid NOT NULL REFERENCES export_record(record_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  link_method text NOT NULL CHECK (link_method IN ('representative','eid','doi','bibliographic_fallback')),
  PRIMARY KEY (run_id, record_id)
);
CREATE INDEX IF NOT EXISTS catalog_member_canonical_idx ON catalog_member(canonical_id);
CREATE TABLE IF NOT EXISTS dedup_log (
  log_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  decision text NOT NULL CHECK (decision IN ('merged','flagged_for_review')),
  method text NOT NULL,
  record_id uuid NOT NULL REFERENCES export_record(record_id),
  other_record_id uuid NOT NULL REFERENCES export_record(record_id),
  canonical_id uuid REFERENCES canonical_paper(canonical_id),
  reason text NOT NULL,
  details jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS dedup_log_run_idx ON dedup_log(run_id);
CREATE TABLE IF NOT EXISTS catalog_seed_result (
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  seed_key text NOT NULL,
  community text NOT NULL CHECK (community IN ('CTO','TIM')),
  benchmark_version text NOT NULL,
  seed_status text NOT NULL,
  outcome text NOT NULL CHECK (outcome IN ('found','missing','ambiguous','out_of_scope')),
  match_method text,
  canonical_ids uuid[] NOT NULL DEFAULT '{}',
  possible_cause text,
  PRIMARY KEY (run_id, community, seed_key)
);
