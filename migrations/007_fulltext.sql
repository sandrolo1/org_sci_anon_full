-- Full-text acquisition. Both tables are append-only: a file or an attempt is never updated or deleted.
-- A missing PDF means "unavailable", never "irrelevant"; the latest attempt per paper explains why.
CREATE TABLE IF NOT EXISTS fulltext_file (
  file_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  origin text NOT NULL CHECK (origin IN ('open_access_download','manual_upload')),
  s3_uri text NOT NULL,
  sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
  size_bytes bigint NOT NULL CHECK (size_bytes > 0),
  source_url text,
  original_filename text,
  acquired_by text NOT NULL CHECK (length(btrim(acquired_by)) > 0),
  validation_status text NOT NULL DEFAULT 'unvalidated' CHECK (validation_status IN ('unvalidated','validated','rejected')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (canonical_id, sha256)
);
CREATE INDEX IF NOT EXISTS fulltext_file_paper_idx ON fulltext_file(run_id, canonical_id);
CREATE TABLE IF NOT EXISTS fulltext_attempt (
  attempt_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  doi text,
  source text NOT NULL,
  outcome text NOT NULL CHECK (outcome IN ('downloaded','no_doi','not_open_access','no_pdf_link','download_failed','not_a_pdf','too_large','lookup_failed')),
  oa_status text,
  license text,
  pdf_url text,
  landing_url text,
  http_status integer,
  detail text,
  file_id uuid REFERENCES fulltext_file(file_id),
  attempted_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fulltext_attempt_latest_idx ON fulltext_attempt(run_id, canonical_id, attempted_at DESC);
