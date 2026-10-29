-- Phase 3: metadata-only CTO/TIM pre-assignment, yes/no/uncertain with evidence. One row per paper and community.
-- Append-only: a new codebook, model or prompt is a new preassignment_run, never an overwrite.
CREATE TABLE IF NOT EXISTS preassignment_run (
  preassignment_run_id uuid PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES catalog_run(run_id),
  source text NOT NULL CHECK (source IN ('batch_v2','v1_suggestion')),
  model text NOT NULL,
  prompt_version text NOT NULL,
  scope_version text NOT NULL,
  codebook_sha256 text NOT NULL,
  manifest jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS community_preassignment (
  preassignment_run_id uuid NOT NULL REFERENCES preassignment_run(preassignment_run_id),
  canonical_id uuid NOT NULL REFERENCES canonical_paper(canonical_id),
  community text NOT NULL CHECK (community IN ('CTO','TIM')),
  label text NOT NULL CHECK (label IN ('yes','no','uncertain')),
  confidence double precision CHECK (confidence >= 0 AND confidence <= 1),
  evidence_field text CHECK (evidence_field IN ('title','abstract','keywords')),
  evidence_quote text,
  rationale text NOT NULL,
  text_basis text NOT NULL CHECK (text_basis IN ('abstract','title_keywords_only','insufficient')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (preassignment_run_id, canonical_id, community)
);
CREATE INDEX IF NOT EXISTS community_preassignment_label_idx ON community_preassignment(preassignment_run_id, community, label);
