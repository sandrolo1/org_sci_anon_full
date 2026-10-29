-- Evaluation of published papers by the novelty judges (paper_evaluation.py, 20 Oct 2026, the user's request).
-- The papers are the 2026 papers of inputs/evaluation_2026 (outside the corpus perimeter). A judge sees only their title and abstract, the same input the forum proposals have; the journal and its
-- ABS grade are kept apart (evaluation_paper) and used only to SHOW the results. Each run has its model, the model's knowledge cutoff and the sample it drew, so runs can be compared.
-- evaluation_score is append-only: one row per paper and attempt (a failed attempt stays; the page shows the newest). evaluation_run is updated only in status / finished_at / stats / note.
-- The forum agents must not read any of it (REVOKE at the end): it is the yardstick their proposals are compared with.
CREATE TABLE IF NOT EXISTS evaluation_paper (
  paper_id text PRIMARY KEY,
  title text NOT NULL,
  journal text NOT NULL DEFAULT '',
  ajg2024 text NOT NULL DEFAULT '',
  doi text NOT NULL DEFAULT '',
  online_date date,
  source_sets text[] NOT NULL DEFAULT '{}',
  abstract_words integer,
  abstract_sha256 text NOT NULL DEFAULT '',
  updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS evaluation_run (
  run_id uuid PRIMARY KEY,
  kind text NOT NULL DEFAULT 'novelty' CHECK (kind IN ('novelty')),
  label text NOT NULL DEFAULT '',
  status text NOT NULL CHECK (status IN ('running', 'completed', 'stopped', 'failed')),
  synthetic boolean NOT NULL DEFAULT false,
  config_name text NOT NULL,
  config_sha256 text NOT NULL,
  model text NOT NULL,
  knowledge_cutoff date NOT NULL,
  parameters jsonb NOT NULL,
  stats jsonb,
  note text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  finished_at timestamptz
);
CREATE TABLE IF NOT EXISTS evaluation_score (
  score_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES evaluation_run(run_id),
  paper_id text NOT NULL REFERENCES evaluation_paper(paper_id),
  status text NOT NULL CHECK (status IN ('ok', 'failed')),
  novelty numeric(4,2) CHECK (novelty IS NULL OR (novelty >= 1 AND novelty <= 10)),
  novelty_hybrid numeric(4,2) CHECK (novelty_hybrid IS NULL OR (novelty_hybrid >= 1 AND novelty_hybrid <= 10)),
  document jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK ((status = 'ok' AND novelty IS NOT NULL) OR (status = 'failed' AND novelty IS NULL AND novelty_hybrid IS NULL))
);
CREATE INDEX IF NOT EXISTS evaluation_score_idx ON evaluation_score(run_id, paper_id, created_at DESC);
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    EXECUTE 'REVOKE ALL ON evaluation_paper, evaluation_run, evaluation_score FROM research_agent';
  END IF;
END
$$;
