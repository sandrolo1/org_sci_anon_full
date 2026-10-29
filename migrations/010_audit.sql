-- Phase 4: stratified human audit of the pre-assignment, with a mandatory sample of the excluded papers.
-- Ratings and adjudications are append-only; the latest row per (audit, paper, community, validator) is current.
CREATE TABLE IF NOT EXISTS audit_sample (
  audit_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  preassignment_run_id uuid NOT NULL REFERENCES preassignment_run(preassignment_run_id),
  seed integer NOT NULL,
  design jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS audit_sample_paper (
  audit_id uuid NOT NULL REFERENCES audit_sample(audit_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  stratum text NOT NULL,
  population_size integer NOT NULL CHECK (population_size >= 1),
  sample_size integer NOT NULL CHECK (sample_size >= 1),
  position integer NOT NULL,
  PRIMARY KEY (audit_id, canonical_id),
  UNIQUE (audit_id, position)
);
CREATE TABLE IF NOT EXISTS audit_rating (
  rating_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  audit_id uuid NOT NULL,
  canonical_id uuid NOT NULL,
  community text NOT NULL CHECK (community IN ('CTO','TIM')),
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  label text NOT NULL CHECK (label IN ('yes','no','uncertain')),
  note text NOT NULL DEFAULT '',
  rated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  FOREIGN KEY (audit_id, canonical_id) REFERENCES audit_sample_paper(audit_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS audit_rating_latest_idx ON audit_rating(audit_id, canonical_id, community, validator, rated_at DESC);
CREATE TABLE IF NOT EXISTS audit_adjudication (
  adjudication_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  audit_id uuid NOT NULL,
  canonical_id uuid NOT NULL,
  community text NOT NULL CHECK (community IN ('CTO','TIM')),
  adjudicator text NOT NULL CHECK (length(btrim(adjudicator)) > 0),
  label text NOT NULL CHECK (label IN ('yes','no','uncertain')),
  note text NOT NULL DEFAULT '',
  decided_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  FOREIGN KEY (audit_id, canonical_id) REFERENCES audit_sample_paper(audit_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS audit_adjudication_latest_idx ON audit_adjudication(audit_id, canonical_id, community, decided_at DESC);
