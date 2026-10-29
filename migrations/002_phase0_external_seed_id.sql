ALTER TABLE benchmark_seed ADD COLUMN IF NOT EXISTS external_seed_id text;
CREATE UNIQUE INDEX IF NOT EXISTS benchmark_external_seed_idx
  ON benchmark_seed(community, version, external_seed_id)
  WHERE external_seed_id IS NOT NULL;
