-- One click on the dashboard, three Batches over the same parsed papers (18 Oct 2026): construct extraction, the methodological card, the card of contributions.
-- `processing_item` is what stops a second click from sending a paper again: one row per (kind, paper version, prompt version) that was put in a submitted Batch.
-- `paper_card` is where the two kinds of card land when their Batch is collected. Append-only: a card that failed its checks is kept, with the reason, next to the valid ones.
CREATE TABLE IF NOT EXISTS processing_item (
  kind text NOT NULL CHECK (kind IN ('constructs', 'method', 'impact')),
  version_id uuid NOT NULL REFERENCES paper_version(version_id),
  prompt_version text NOT NULL,
  job text NOT NULL,
  submitted_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, version_id, prompt_version)
);
CREATE TABLE IF NOT EXISTS paper_card (
  card_id uuid PRIMARY KEY,
  kind text NOT NULL CHECK (kind IN ('method', 'impact')),
  version_id uuid NOT NULL REFERENCES paper_version(version_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  profile_version text NOT NULL,
  model text NOT NULL,
  source_sha256 text NOT NULL,
  source_info jsonb NOT NULL DEFAULT '{}',
  status text NOT NULL CHECK (status IN ('valid', 'rejected')),
  card jsonb,
  rejection text,
  job text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CHECK ((status = 'valid' AND card IS NOT NULL AND rejection IS NULL) OR (status = 'rejected' AND rejection IS NOT NULL))
);
-- A paper has at most one valid card per kind, profile version and source text; rejected attempts may repeat.
CREATE UNIQUE INDEX IF NOT EXISTS paper_card_valid_idx ON paper_card(kind, version_id, profile_version, source_sha256) WHERE status = 'valid';
CREATE INDEX IF NOT EXISTS paper_card_canonical_idx ON paper_card(canonical_id, kind);
-- The discovery agents must not read the evaluator's baselines: the cards say what published papers claim to change and which methods they use, and an agent that saw them
-- would be writing proposals against the yardstick that grades them. The earlier migrations grant SELECT on every table each time they are re-run, so this revoke comes after them.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    EXECUTE 'REVOKE ALL ON paper_card, processing_item FROM research_agent';
  END IF;
END
$$;
