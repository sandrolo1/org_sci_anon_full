-- Embeddings of the construct database's records, so that agents can search them (the chunks of the PDFs have chunk_embedding).
-- One row per (record, model). The text that was embedded is not stored: it is rebuilt from the record by research0.embedding_jobs and its
-- SHA-256 is kept, so a changed record or a changed recipe is detectable. Derived and rebuildable; never a source of record.
-- record_kind: mention | measure | claim | attribute. record_id is that table's key. canonical_id is denormalised for community filters.
CREATE TABLE IF NOT EXISTS record_embedding (
  record_kind text NOT NULL CHECK (record_kind IN ('mention','measure','claim','attribute')),
  record_id uuid NOT NULL,
  model_id text NOT NULL REFERENCES embedding_model(model_id),
  extraction_run_id uuid NOT NULL REFERENCES extraction_run(extraction_run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  text_sha256 text NOT NULL,
  recipe text NOT NULL,
  embedding vector NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (record_kind, record_id, model_id)
);
CREATE INDEX IF NOT EXISTS record_embedding_run_idx ON record_embedding(extraction_run_id, model_id);
CREATE INDEX IF NOT EXISTS record_embedding_paper_idx ON record_embedding(canonical_id);
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;
