CREATE TABLE IF NOT EXISTS blind_sample (
  run_id uuid PRIMARY KEY REFERENCES catalog_run(run_id),
  threshold double precision NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS blind_sample_paper (
  run_id uuid NOT NULL REFERENCES blind_sample(run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  position integer NOT NULL,
  PRIMARY KEY (run_id, canonical_id),
  UNIQUE (run_id, position)
);
CREATE TABLE IF NOT EXISTS blind_rating (
  rating_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL,
  canonical_id uuid NOT NULL,
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  cto integer NOT NULL CHECK (cto IN (0, 1)),
  tim integer NOT NULL CHECK (tim IN (0, 1)),
  rated_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (run_id, canonical_id) REFERENCES blind_sample_paper(run_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS blind_rating_latest_idx ON blind_rating(run_id, validator, canonical_id, rated_at DESC);
