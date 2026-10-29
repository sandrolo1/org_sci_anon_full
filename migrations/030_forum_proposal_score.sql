-- Forum, fifth step: the evaluation of a research proposal (forum_scoring.py, 18 Oct 2026). It replaces the blind evaluator (023) and the Baseline Agent Evaluation (028) on the forum page;
-- their rows stay as history.
-- Three votes from 1 to 10, each the mean of two judges and each judged against its own nearest published papers: NOVELTY (the abstract against the abstracts of the CTO/TIM corpus;
-- distance is good), METHODOLOGY (the proposal's methodology card against the methodology cards of the papers; similarity is good) and IMPACT (the impact card against the impact
-- cards, with the citations those papers received as their realised impact). The similarity indices, the indicative votes, the neighbours and both judges' answers are in `document`,
-- so a row can be read without the workspace files. A criterion that could not be judged has a NULL vote; status says whether all three (ok), some (partial) or none (failed) exist.
-- Append-only like the rest of the forum workspace: a new evaluation of the same proposal is a new row; the page shows the newest one that is not failed.
-- The agents of the forum must not read it (REVOKE at the end): it is the yardstick their proposals are graded with.
CREATE TABLE IF NOT EXISTS forum_proposal_score (
  score_id uuid PRIMARY KEY,
  proposal_id uuid NOT NULL REFERENCES forum_proposal(proposal_id),
  version text NOT NULL,
  status text NOT NULL CHECK (status IN ('ok', 'partial', 'failed')),
  novelty numeric(4,2) CHECK (novelty IS NULL OR (novelty >= 1 AND novelty <= 10)),
  methodology numeric(4,2) CHECK (methodology IS NULL OR (methodology >= 1 AND methodology <= 10)),
  impact numeric(4,2) CHECK (impact IS NULL OR (impact >= 1 AND impact <= 10)),
  synthetic boolean NOT NULL DEFAULT false,
  config_sha256 text NOT NULL,
  document jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK ((status = 'ok' AND novelty IS NOT NULL AND methodology IS NOT NULL AND impact IS NOT NULL)
         OR (status = 'partial' AND num_nonnulls(novelty, methodology, impact) BETWEEN 1 AND 2)
         OR (status = 'failed' AND novelty IS NULL AND methodology IS NULL AND impact IS NULL))
);
CREATE INDEX IF NOT EXISTS forum_proposal_score_idx ON forum_proposal_score(proposal_id, created_at DESC);
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    EXECUTE 'REVOKE ALL ON forum_proposal_score FROM research_agent';
  END IF;
END
$$;
