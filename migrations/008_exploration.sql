-- Phase 2: exploratory clustering. Embeddings and clusters are DERIVED and rebuildable; they never decide membership
-- and never decide that two constructs are the same. Rows are inserted, not updated.
CREATE TABLE IF NOT EXISTS embedding_model (
  model_id text PRIMARY KEY,
  kind text NOT NULL CHECK (kind IN ('lexical','semantic')),
  dimensions integer NOT NULL CHECK (dimensions > 0),
  description text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS paper_embedding (
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  model_id text NOT NULL REFERENCES embedding_model(model_id),
  text_sha256 text NOT NULL,
  embedding vector NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (canonical_id, model_id, text_sha256)
);
CREATE TABLE IF NOT EXISTS cluster_run (
  cluster_run_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  model_id text NOT NULL REFERENCES embedding_model(model_id),
  algorithm text NOT NULL,
  params jsonb NOT NULL,
  seed integer NOT NULL,
  paper_count integer NOT NULL,
  cluster_count integer NOT NULL,
  summary jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS cluster_assignment (
  cluster_run_id uuid NOT NULL REFERENCES cluster_run(cluster_run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  cluster_label integer NOT NULL,
  distance double precision NOT NULL,
  PRIMARY KEY (cluster_run_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS cluster_assignment_label_idx ON cluster_assignment(cluster_run_id, cluster_label);
