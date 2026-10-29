-- Papers taken out of the full-text work (download, missing-PDF list, coverage). Append-only: the latest row per paper decides,
-- so an exclusion is lifted by appending a row with excluded = false. The reason says who decided what and why.
-- An exclusion is not a statement about the paper's relevance (principle 5): it only says "do not spend effort on its PDF".
CREATE TABLE IF NOT EXISTS fulltext_exclusion (
  exclusion_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  excluded boolean NOT NULL DEFAULT true,
  reason text NOT NULL CHECK (length(btrim(reason)) > 0),
  decided_by text NOT NULL CHECK (length(btrim(decided_by)) > 0),
  decided_at timestamptz NOT NULL DEFAULT clock_timestamp()  -- not now(): rows written in one transaction must still be ordered
);
CREATE INDEX IF NOT EXISTS fulltext_exclusion_latest_idx ON fulltext_exclusion(run_id, canonical_id, decided_at DESC);
ALTER TABLE fulltext_exclusion ALTER COLUMN decided_at SET DEFAULT clock_timestamp();
