-- Human validation decisions from the dashboard. Append-only: a change or withdrawal is a new row;
-- the latest row per (run, item, validator) is the current decision.
CREATE TABLE IF NOT EXISTS validation_decision (
  decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  item_id text NOT NULL,
  item_type text NOT NULL,
  decision text NOT NULL,
  note text NOT NULL DEFAULT '',
  validator text NOT NULL CHECK (length(btrim(validator)) > 0),
  decided_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS validation_decision_run_idx ON validation_decision(run_id, item_id, validator, decided_at DESC);
