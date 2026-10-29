CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS source_list (
  source_id text NOT NULL,
  journal_name text NOT NULL,
  inclusion_rationale text NOT NULL,
  active boolean NOT NULL,
  version text NOT NULL,
  validated_on date NOT NULL,
  validated_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (version, source_id)
);
CREATE TABLE IF NOT EXISTS query_version (
  query_id uuid PRIMARY KEY,
  source_version text NOT NULL,
  query_text text NOT NULL,
  query_sha256 text NOT NULL,
  UNIQUE (source_version, query_sha256),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS export_batch (
  batch_id uuid PRIMARY KEY,
  query_id uuid NOT NULL REFERENCES query_version(query_id),
  original_uri text NOT NULL,
  original_sha256 text NOT NULL,
  original_filename text NOT NULL,
  exported_on date NOT NULL,
  provenance text NOT NULL,
  record_count integer NOT NULL CHECK (record_count >= 0),
  imported_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (query_id, original_sha256)
);
CREATE TABLE IF NOT EXISTS export_record (
  record_id uuid PRIMARY KEY,
  batch_id uuid NOT NULL REFERENCES export_batch(batch_id),
  row_number integer NOT NULL,
  doi text,
  eid text,
  title text,
  title_normalized text,
  authors text,
  publication_year integer,
  source_title text,
  source_id text,
  document_type text,
  source_type text,
  raw_fields jsonb NOT NULL,
  UNIQUE (batch_id, row_number)
);
CREATE INDEX IF NOT EXISTS export_record_doi_idx ON export_record(doi);
CREATE INDEX IF NOT EXISTS export_record_eid_idx ON export_record(eid);
CREATE INDEX IF NOT EXISTS export_record_title_year_idx ON export_record(title_normalized, publication_year);
CREATE TABLE IF NOT EXISTS benchmark_seed (
  seed_id uuid PRIMARY KEY,
  community text NOT NULL CHECK (community IN ('CTO','TIM')),
  version text NOT NULL,
  doi text,
  eid text,
  title text NOT NULL,
  authors text,
  publication_year integer NOT NULL,
  journal text NOT NULL,
  top_rank text NOT NULL,
  expert_rationale text NOT NULL,
  status text NOT NULL CHECK (status IN ('eligible','out_of_scope','duplicate','to_verify')),
  retrieval_note text NOT NULL,
  validated_by text NOT NULL,
  validated_on date NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (community, version, seed_id)
);
CREATE TABLE IF NOT EXISTS matching_run (
  run_id uuid PRIMARY KEY,
  batch_id uuid NOT NULL REFERENCES export_batch(batch_id),
  benchmark_version text NOT NULL,
  matcher_version text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (batch_id, benchmark_version, matcher_version)
);
CREATE TABLE IF NOT EXISTS matching_result (
  run_id uuid NOT NULL REFERENCES matching_run(run_id),
  seed_id uuid NOT NULL REFERENCES benchmark_seed(seed_id),
  outcome text NOT NULL CHECK (outcome IN ('found','missing','ambiguous','out_of_scope')),
  match_method text,
  candidate_record_ids uuid[] NOT NULL DEFAULT '{}',
  possible_cause text,
  PRIMARY KEY (run_id, seed_id)
);
