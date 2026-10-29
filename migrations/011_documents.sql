-- Phase 5 (parsing and chunking). A paper_version is one parse of one stored PDF; text, offsets and hashes make every later
-- span checkable. A PDF without a text layer is 'no_text_layer' (the full text stays unavailable), never "irrelevant".
-- Everything here is derived from fulltext_file and can be rebuilt by a new ingestion_run.
CREATE TABLE IF NOT EXISTS ingestion_run (
  ingestion_run_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  parser text NOT NULL,
  parser_version text NOT NULL,
  chunker_version text NOT NULL,
  params jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_version (
  version_id uuid PRIMARY KEY,
  ingestion_run_id uuid NOT NULL REFERENCES ingestion_run(ingestion_run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  file_id uuid NOT NULL REFERENCES fulltext_file(file_id),
  status text NOT NULL CHECK (status IN ('parsed','no_text_layer','failed')),
  detail text,
  page_count integer,
  language text,
  full_text text,
  page_offsets integer[],
  text_sha256 text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (ingestion_run_id, file_id)
);
CREATE INDEX IF NOT EXISTS paper_version_paper_idx ON paper_version(canonical_id);
CREATE TABLE IF NOT EXISTS document_chunk (
  chunk_id uuid PRIMARY KEY,
  version_id uuid NOT NULL REFERENCES paper_version(version_id),
  ordinal integer NOT NULL,
  section text NOT NULL,
  page_start integer NOT NULL,
  page_end integer NOT NULL,
  char_start integer NOT NULL,
  char_end integer NOT NULL,
  text text NOT NULL,
  text_sha256 text NOT NULL,
  token_count integer NOT NULL,
  language text,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  UNIQUE (version_id, ordinal)
);
CREATE INDEX IF NOT EXISTS document_chunk_tsv_idx ON document_chunk USING gin (tsv);
CREATE TABLE IF NOT EXISTS chunk_embedding (
  chunk_id uuid NOT NULL REFERENCES document_chunk(chunk_id),
  model_id text NOT NULL REFERENCES embedding_model(model_id),
  embedding vector NOT NULL,
  PRIMARY KEY (chunk_id, model_id)
);
