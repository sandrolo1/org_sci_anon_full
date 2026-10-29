CREATE TABLE IF NOT EXISTS community_adjudication (
  adjudication_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL,
  canonical_id uuid NOT NULL,
  adjudicator text NOT NULL CHECK (length(btrim(adjudicator)) > 0),
  cto integer NOT NULL CHECK (cto IN (0, 1)),
  tim integer NOT NULL CHECK (tim IN (0, 1)),
  decided_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (run_id, canonical_id) REFERENCES blind_sample_paper(run_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS community_adjudication_latest_idx
  ON community_adjudication(run_id, canonical_id, decided_at DESC);
