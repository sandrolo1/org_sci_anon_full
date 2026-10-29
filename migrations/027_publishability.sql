-- Publishability evaluator: a benchmark of CTO/TIM introductions labelled by venue tier, persistent splits, an autonomous calibration pipeline and the forum integration.
-- Everything here is append-only (the usual rule): a new version is a new row. The only "state" is derived from the event tables (pub_experiment_event, pub_activation).
-- The benchmark label is a proxy: 1 = published in a journal of the configured top tier, 0 = published in a journal of the configured lower tier. It is NOT a documented rejection;
-- documented rejections live apart, in pub_rejection, with their provenance.

-- The tiered journal list (configured by a person; the top tier can be taken from source_register, the lower tier is never invented).
CREATE TABLE IF NOT EXISTS pub_journal_list (
  list_version text NOT NULL,
  journal_key text NOT NULL,                    -- normalised exact title
  exact_title text NOT NULL,
  issn text,
  eissn text,
  scopus_source_id text,
  tier text NOT NULL CHECK (tier IN ('top','lower')),
  community text CHECK (community IS NULL OR community IN ('CTO','TIM','both')),
  title_variants text[] NOT NULL DEFAULT '{}',
  origin text NOT NULL,                         -- source_register:<version> | csv:<file name>
  configured_by text NOT NULL,
  configured_on date NOT NULL,
  notes text NOT NULL DEFAULT '',
  PRIMARY KEY (list_version, journal_key)
);

-- A dataset is one immutable import (content hash), with the explicit year convention it was built under.
CREATE TABLE IF NOT EXISTS pub_dataset (
  dataset_id uuid PRIMARY KEY,
  label text NOT NULL,
  content_sha256 text NOT NULL UNIQUE,
  journal_list_version text NOT NULL,
  benchmark_year integer NOT NULL,
  year_convention text NOT NULL,
  source_file text,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  stats jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pub_paper (
  dataset_id uuid NOT NULL REFERENCES pub_dataset(dataset_id),
  paper_id text NOT NULL,
  group_id text NOT NULL,                       -- papers that are versions or duplicates of each other share it and are never split apart
  scope text NOT NULL CHECK (scope IN ('CTO','TIM')),
  journal_key text NOT NULL,
  journal_title text NOT NULL,
  tier text NOT NULL CHECK (tier IN ('top','lower')),
  doi text,
  title text,
  authors text,
  online_date date,
  online_year integer,
  issue_year integer,
  issue_label text,
  label integer NOT NULL CHECK (label IN (0,1)),
  label_provenance text NOT NULL,
  intro_original text,
  intro_prepared text,
  intro_status text NOT NULL CHECK (intro_status IN ('ok','missing','incomplete','flagged_residue')),
  intro_flags jsonb NOT NULL DEFAULT '[]',
  intro_sha256 text,
  intro_source text,
  bibliographic_source text NOT NULL,
  verification_status text NOT NULL CHECK (verification_status IN ('verified','unverified')),
  verified_by text,
  verified_on date,
  PRIMARY KEY (dataset_id, paper_id)
);
CREATE INDEX IF NOT EXISTS pub_paper_group_idx ON pub_paper(dataset_id, group_id);

CREATE TABLE IF NOT EXISTS pub_rejection (
  dataset_id uuid NOT NULL,
  paper_id text NOT NULL,
  rejecting_journal text NOT NULL,
  rejection_date date,
  provenance text NOT NULL,                     -- who says so and where (author statement, editor letter, ...)
  note text NOT NULL DEFAULT '',
  recorded_by text NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (dataset_id, paper_id, rejecting_journal),
  FOREIGN KEY (dataset_id, paper_id) REFERENCES pub_paper(dataset_id, paper_id)
);

-- A split is persistent and immutable. Groups, not papers, are allocated.
CREATE TABLE IF NOT EXISTS pub_split (
  split_id uuid PRIMARY KEY,
  dataset_id uuid NOT NULL REFERENCES pub_dataset(dataset_id),
  seed integer NOT NULL,
  ratios jsonb NOT NULL,
  policy jsonb NOT NULL,
  content_sha256 text NOT NULL UNIQUE,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  report jsonb NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS pub_split_item (
  split_id uuid NOT NULL REFERENCES pub_split(split_id),
  paper_id text NOT NULL,
  group_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('dev','val','test')),
  stratum text NOT NULL,
  unseen_journal boolean NOT NULL DEFAULT false,   -- test paper from a journal with no paper in dev or validation
  PRIMARY KEY (split_id, paper_id)
);

-- Evaluator configurations (immutable). eval_sha256 leaves out the classification threshold: changing only the threshold re-uses every cached model call.
CREATE TABLE IF NOT EXISTS pub_config (
  config_id uuid PRIMARY KEY,
  config_sha256 text NOT NULL UNIQUE,
  eval_sha256 text NOT NULL,
  content jsonb NOT NULL,
  parent_config_id uuid,
  origin text NOT NULL,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS pub_experiment (
  experiment_id uuid PRIMARY KEY,
  label text NOT NULL,
  split_id uuid NOT NULL REFERENCES pub_split(split_id),
  spec jsonb NOT NULL,
  spec_sha256 text NOT NULL,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS pub_experiment_event (
  experiment_id uuid NOT NULL REFERENCES pub_experiment(experiment_id),
  seq integer NOT NULL,
  event text NOT NULL,
  data jsonb NOT NULL DEFAULT '{}',
  at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (experiment_id, seq)
);

CREATE TABLE IF NOT EXISTS pub_candidate (
  experiment_id uuid NOT NULL REFERENCES pub_experiment(experiment_id),
  iteration integer NOT NULL,
  idx integer NOT NULL,
  config_id uuid NOT NULL REFERENCES pub_config(config_id),
  origin text NOT NULL,                         -- initial | proposer | threshold
  summary text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (experiment_id, iteration, idx)
);
CREATE TABLE IF NOT EXISTS pub_result (
  experiment_id uuid NOT NULL REFERENCES pub_experiment(experiment_id),
  config_id uuid NOT NULL REFERENCES pub_config(config_id),
  role text NOT NULL CHECK (role IN ('dev','val','test')),
  metrics jsonb NOT NULL,
  n_scored integer NOT NULL,
  n_failed integer NOT NULL,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (experiment_id, config_id, role)
);

-- Every call to the model (and every failed attempt). The cache is "the latest ok row with this cache_key".
CREATE TABLE IF NOT EXISTS pub_call (
  call_id uuid PRIMARY KEY,
  cache_key text NOT NULL,
  purpose text NOT NULL,                        -- experiment | proposer | final_test | forum
  experiment_id uuid,
  config_id uuid,
  eval_sha256 text,
  paper_ref text,
  intro_sha256 text,
  status text NOT NULL CHECK (status IN ('ok','failed')),
  provider text NOT NULL,
  model text NOT NULL,
  model_resolved text,
  result jsonb,
  input_tokens integer NOT NULL DEFAULT 0,
  output_tokens integer NOT NULL DEFAULT 0,
  reasoning_tokens integer NOT NULL DEFAULT 0,
  cost_usd numeric(12,6),
  latency_ms integer,
  error text,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS pub_call_key_idx ON pub_call(cache_key, status);
CREATE INDEX IF NOT EXISTS pub_call_experiment_idx ON pub_call(experiment_id);

-- A configuration is frozen when it meets the validation criteria; the final test is used once per split, ever.
CREATE TABLE IF NOT EXISTS pub_freeze (
  config_id uuid NOT NULL REFERENCES pub_config(config_id),
  experiment_id uuid NOT NULL REFERENCES pub_experiment(experiment_id),
  config_sha256 text NOT NULL,
  validation jsonb NOT NULL,
  frozen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY (config_id, experiment_id)
);
CREATE TABLE IF NOT EXISTS pub_holdout_access (
  access_id uuid PRIMARY KEY,
  split_id uuid NOT NULL REFERENCES pub_split(split_id),
  experiment_id uuid NOT NULL,
  config_id uuid NOT NULL,
  purpose text NOT NULL,
  n_items integer NOT NULL,
  accessed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS pub_test_use (
  split_id uuid PRIMARY KEY REFERENCES pub_split(split_id),
  experiment_id uuid NOT NULL,
  config_id uuid NOT NULL,
  passed boolean NOT NULL,
  metrics jsonb NOT NULL,
  criteria jsonb NOT NULL,
  used_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

-- What the forum uses: promote a validated configuration, roll back to the previous one. The active one is the latest row.
CREATE TABLE IF NOT EXISTS pub_activation (
  activation_id uuid PRIMARY KEY,
  action text NOT NULL CHECK (action IN ('promote','rollback')),
  config_id uuid NOT NULL REFERENCES pub_config(config_id),
  previous_activation_id uuid,                  -- the activation that was current when this one was made
  restores_activation_id uuid,                  -- rollback only: the activation whose configuration is back in use
  done_by text NOT NULL,
  note text NOT NULL DEFAULT '',
  done_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE IF NOT EXISTS pub_forum_prediction (
  prediction_id uuid PRIMARY KEY,
  proposal_id uuid NOT NULL REFERENCES forum_proposal(proposal_id),
  config_id uuid NOT NULL REFERENCES pub_config(config_id),
  config_sha256 text NOT NULL,
  activation_id uuid NOT NULL,
  provider text NOT NULL,
  model text NOT NULL,
  model_resolved text,
  prediction integer NOT NULL CHECK (prediction IN (0,1)),
  model_prediction integer,
  probability numeric(6,5) NOT NULL,
  threshold numeric(6,5) NOT NULL,
  rationale text NOT NULL,
  limitations jsonb NOT NULL DEFAULT '[]',
  quotes_verified boolean,
  intro_sha256 text NOT NULL,
  call_id uuid,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS pub_forum_prediction_idx ON pub_forum_prediction(proposal_id, created_at DESC);

-- The benchmark texts and labels are not for the discovery agents (they read the corpus, not the benchmark).
DO $$
BEGIN
  EXECUTE format('REVOKE ALL ON pub_journal_list, pub_dataset, pub_paper, pub_rejection, pub_split, pub_split_item, pub_config, pub_experiment, pub_experiment_event, pub_candidate, pub_result, pub_call, pub_freeze, pub_holdout_access, pub_test_use, pub_activation, pub_forum_prediction FROM research_agent');
END
$$;
