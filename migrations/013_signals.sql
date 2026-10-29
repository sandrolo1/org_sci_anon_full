-- Phase 6: signals and candidate pairs. A node is one paper's use of one construct label, so the same label in two papers is two
-- nodes (that is what makes a jingle observable). A candidate pair is a request for human attention. The three signals and the name
-- similarity are descriptive numbers (NULL when the input data is missing, never 0); the hint is a triage label, not a verdict.
CREATE TABLE IF NOT EXISTS signal_run (
  signal_run_id uuid PRIMARY KEY,
  extraction_run_id uuid NOT NULL REFERENCES extraction_run(extraction_run_id),
  model_id text NOT NULL REFERENCES embedding_model(model_id),
  version text NOT NULL,
  params jsonb NOT NULL,
  node_count integer NOT NULL,
  pair_count integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS construct_node (
  signal_run_id uuid NOT NULL REFERENCES signal_run(signal_run_id),
  node_id uuid NOT NULL,
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  label_normalized text NOT NULL,
  mention_count integer NOT NULL,
  mention_ids uuid[] NOT NULL,
  PRIMARY KEY (signal_run_id, node_id)
);
CREATE TABLE IF NOT EXISTS candidate_pair (
  pair_id uuid PRIMARY KEY,
  signal_run_id uuid NOT NULL REFERENCES signal_run(signal_run_id),
  node_a uuid NOT NULL,
  node_b uuid NOT NULL,
  name_sim double precision,
  s_sem_op double precision,
  c_cit double precision,
  n_nom double precision,
  retrieval_reasons text[] NOT NULL,
  hint text NOT NULL CHECK (hint IN ('jingle_candidate','jangle_candidate','partial_candidate','none')),
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK (node_a < node_b),
  UNIQUE (signal_run_id, node_a, node_b),
  FOREIGN KEY (signal_run_id, node_a) REFERENCES construct_node(signal_run_id, node_id),
  FOREIGN KEY (signal_run_id, node_b) REFERENCES construct_node(signal_run_id, node_id)
);
CREATE TABLE IF NOT EXISTS pair_dossier (
  pair_id uuid PRIMARY KEY REFERENCES candidate_pair(pair_id),
  dossier_version text NOT NULL,
  content jsonb NOT NULL,
  content_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
