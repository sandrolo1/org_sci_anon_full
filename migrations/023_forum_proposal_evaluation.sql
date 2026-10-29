-- Forum, third step: a blind evaluator agent rates every research proposal on six indices (1-10).
-- Append-only like the rest of the forum workspace: a new evaluation of the same proposal is a new row, never an update; the interface shows the latest
-- successful one. A failed attempt is kept (status 'failed', with the error) so a retry leaves a trace. The evaluator is a separate call that sees only the
-- title and the four sections of the proposal (see forum_evaluation.py): its tokens are stored here and are NOT part of the episode's budgets or usage.
CREATE TABLE IF NOT EXISTS forum_proposal_evaluation (
  evaluation_id uuid PRIMARY KEY,
  proposal_id uuid NOT NULL REFERENCES forum_proposal(proposal_id),
  status text NOT NULL CHECK (status IN ('ok','failed')),
  provider text NOT NULL,
  model text NOT NULL,
  reasoning_effort text,
  rubric_version text NOT NULL,
  prompt_sha256 text NOT NULL,
  text_sha256 text NOT NULL,
  scores jsonb NOT NULL DEFAULT '{}',
  mean_score numeric(4,2) CHECK (mean_score IS NULL OR (mean_score >= 1 AND mean_score <= 10)),
  overall_comment text NOT NULL DEFAULT '',
  input_tokens integer NOT NULL DEFAULT 0,
  output_tokens integer NOT NULL DEFAULT 0,
  reasoning_tokens integer NOT NULL DEFAULT 0,
  latency_ms integer,
  error text,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK ((status = 'ok' AND mean_score IS NOT NULL) OR status = 'failed')
);
CREATE INDEX IF NOT EXISTS forum_proposal_evaluation_idx ON forum_proposal_evaluation(proposal_id, created_at DESC);
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;
