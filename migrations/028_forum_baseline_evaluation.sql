-- Forum, fourth step: the "Baseline Agent Evaluation" of a research proposal (forum_baseline.py).
-- The abstract of the proposal is compared with the closest papers of the CTO/TIM corpus and judged, by two independent calls of one model, on the ANVUR VQR GEV13b criteria; ORIGINALITY is
-- what this evaluator is built for, method and impact are stored as provisional and no total is given. It runs from the command line (`research0 anvur-rag-run`), `anvur-rag-export`
-- writes it here, and the forum page only reads it: the page never starts a paid run.
-- Append-only like the rest of the forum workspace: a new run for the same proposal is a new row (one per proposal and ANVUR run), never an update; the page shows the latest. The whole
-- evaluation (abstract used and whether it was written by the forum agent or derived, the neighbours with their similarity and journal grade, both judges' votes, reasons and
-- comparisons, the models and the corpus) is in `document`, so a row can be read without the workspace files of the run.
-- The agents of the forum must not read it: like the other evaluations of their own proposals, it stays out of their reach (REVOKE at the end of this file).
CREATE TABLE IF NOT EXISTS forum_baseline_evaluation (
  evaluation_id uuid PRIMARY KEY,
  proposal_id uuid NOT NULL REFERENCES forum_proposal(proposal_id),
  status text NOT NULL CHECK (status IN ('simulated_not_official','no_valid_evaluation')),
  run_id text NOT NULL,
  config_sha256 text NOT NULL,
  corpus_id text NOT NULL,
  coverage text NOT NULL CHECK (coverage IN ('adeguata','limitata')),
  synthetic boolean NOT NULL DEFAULT false,
  document jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  UNIQUE (proposal_id, run_id)
);
CREATE INDEX IF NOT EXISTS forum_baseline_evaluation_idx ON forum_baseline_evaluation(proposal_id, created_at DESC);
DO $$
BEGIN
  EXECUTE format('REVOKE ALL ON forum_baseline_evaluation FROM research_agent');
END
$$;
