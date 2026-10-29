-- Phase 5 (extraction). Constructs, measures and nomological claims extracted from chunks. Each record keeps the verbatim
-- quote, its offsets in paper_version.full_text and the page. Nothing here is a merge: construct identity is decided by people.
-- decision_event is append-only: validating or rejecting a record is a new event, never an update of the record.
CREATE TABLE IF NOT EXISTS extraction_run (
  extraction_run_id uuid PRIMARY KEY,
  ingestion_run_id uuid NOT NULL REFERENCES ingestion_run(ingestion_run_id),
  model text NOT NULL,
  prompt_version text NOT NULL,
  schema_version text NOT NULL,
  manifest jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_mention (
  mention_id uuid PRIMARY KEY,
  extraction_run_id uuid NOT NULL REFERENCES extraction_run(extraction_run_id),
  chunk_id uuid NOT NULL REFERENCES document_chunk(chunk_id),
  version_id uuid NOT NULL REFERENCES paper_version(version_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  surface_form text NOT NULL,
  label_normalized text NOT NULL,
  role text NOT NULL CHECK (role IN ('focal','measured','antecedent','outcome','moderator','mediator','control','mentioned')),
  definition_text text,
  quote text NOT NULL,
  char_start integer NOT NULL,
  char_end integer NOT NULL,
  page integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS construct_mention_label_idx ON construct_mention(extraction_run_id, label_normalized);
CREATE TABLE IF NOT EXISTS construct_measure (
  measure_id uuid PRIMARY KEY,
  mention_id uuid NOT NULL REFERENCES construct_mention(mention_id),
  name text NOT NULL,
  items_text text,
  scale text,
  quote text NOT NULL,
  char_start integer NOT NULL,
  char_end integer NOT NULL,
  page integer NOT NULL
);
CREATE TABLE IF NOT EXISTS nomological_claim (
  claim_id uuid PRIMARY KEY,
  extraction_run_id uuid NOT NULL REFERENCES extraction_run(extraction_run_id),
  chunk_id uuid NOT NULL REFERENCES document_chunk(chunk_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  subject_mention_id uuid NOT NULL REFERENCES construct_mention(mention_id),
  object_mention_id uuid NOT NULL REFERENCES construct_mention(mention_id),
  relation text NOT NULL CHECK (relation IN ('predicts','correlates','mediates','moderates','part_of','same_as_claimed','distinct_from_claimed')),
  direction text NOT NULL CHECK (direction IN ('positive','negative','nonlinear','none','unspecified')),
  quote text NOT NULL,
  char_start integer NOT NULL,
  char_end integer NOT NULL,
  page integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS claim_modifier (
  modifier_id uuid PRIMARY KEY,
  claim_id uuid NOT NULL REFERENCES nomological_claim(claim_id),
  kind text NOT NULL CHECK (kind IN ('moderator','boundary_condition','sample','method')),
  value text NOT NULL,
  quote text NOT NULL
);
CREATE TABLE IF NOT EXISTS decision_event (
  event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subject_type text NOT NULL CHECK (subject_type IN ('construct_mention','nomological_claim','construct_measure','paper_version')),
  subject_id uuid NOT NULL,
  decision text NOT NULL CHECK (decision IN ('validated','rejected','needs_review')),
  note text NOT NULL DEFAULT '',
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  decided_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS decision_event_subject_idx ON decision_event(subject_type, subject_id, decided_at DESC);
