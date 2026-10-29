-- Alignment with the research proposal (Proposta_ORG_SI_Extended_CTO_TIM.docx). Additive: tables created by 008-014 keep their data.
-- Constraints that listed allowed values are replaced by wider ones; nothing is dropped from a table.

-- Embedding models: provider/model, dimension, normalization, parameters, validity window (Appendix D, embedding_model).
ALTER TABLE embedding_model ADD COLUMN IF NOT EXISTS normalized boolean NOT NULL DEFAULT true;
ALTER TABLE embedding_model ADD COLUMN IF NOT EXISTS parameters jsonb NOT NULL DEFAULT '{}';
ALTER TABLE embedding_model ADD COLUMN IF NOT EXISTS valid_from timestamptz NOT NULL DEFAULT now();
ALTER TABLE embedding_model ADD COLUMN IF NOT EXISTS valid_to timestamptz;

-- paper_dig of the proposal = community_preassignment: add the stage (metadata or full text) and, for the full-text stage, the span.
ALTER TABLE community_preassignment ADD COLUMN IF NOT EXISTS stage text NOT NULL DEFAULT 'metadata';
ALTER TABLE community_preassignment DROP CONSTRAINT IF EXISTS community_preassignment_stage_check;
ALTER TABLE community_preassignment ADD CONSTRAINT community_preassignment_stage_check CHECK (stage IN ('metadata','full_text'));
ALTER TABLE community_preassignment ADD COLUMN IF NOT EXISTS version_id uuid REFERENCES paper_version(version_id);
ALTER TABLE community_preassignment ADD COLUMN IF NOT EXISTS evidence_page integer;
ALTER TABLE community_preassignment ADD COLUMN IF NOT EXISTS evidence_char_start integer;
ALTER TABLE community_preassignment ADD COLUMN IF NOT EXISTS evidence_char_end integer;
ALTER TABLE community_preassignment DROP CONSTRAINT IF EXISTS community_preassignment_text_basis_check;
ALTER TABLE community_preassignment ADD CONSTRAINT community_preassignment_text_basis_check CHECK (text_basis IN ('abstract','title_keywords_only','insufficient','full_text'));
ALTER TABLE preassignment_run DROP CONSTRAINT IF EXISTS preassignment_run_source_check;
ALTER TABLE preassignment_run ADD CONSTRAINT preassignment_run_source_check CHECK (source IN ('batch_v2','v1_suggestion','fulltext_v1'));

-- Extraction: what the text says, what is inferred and what is missing, each with its page or span (Appendix A, step 05).
CREATE TABLE IF NOT EXISTS extraction_attribute (
  attribute_id uuid PRIMARY KEY,
  mention_id uuid NOT NULL REFERENCES construct_mention(mention_id),
  kind text NOT NULL CHECK (kind IN ('synonym','acronym','level','dimensionality','informant','reliability','validity','mechanism','boundary_condition','context','time_window','population')),
  basis text NOT NULL CHECK (basis IN ('extracted','inferred','missing')),
  value text,
  quote text,
  char_start integer,
  char_end integer,
  page integer,
  CHECK ((basis = 'missing' AND quote IS NULL AND value IS NULL) OR (basis <> 'missing' AND value IS NOT NULL AND quote IS NOT NULL AND page IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS extraction_attribute_mention_idx ON extraction_attribute(mention_id);
ALTER TABLE nomological_claim ADD COLUMN IF NOT EXISTS evidence_status text NOT NULL DEFAULT 'proposed';
ALTER TABLE nomological_claim DROP CONSTRAINT IF EXISTS nomological_claim_evidence_status_check;
ALTER TABLE nomological_claim ADD CONSTRAINT nomological_claim_evidence_status_check CHECK (evidence_status IN ('proposed','tested','supported','not_supported'));
ALTER TABLE nomological_claim ADD COLUMN IF NOT EXISTS confidence double precision CHECK (confidence >= 0 AND confidence <= 1);
-- Mediators and moderators are constructs linked to the claim they qualify (not a free text value).
ALTER TABLE claim_modifier ADD COLUMN IF NOT EXISTS mention_id uuid REFERENCES construct_mention(mention_id);
ALTER TABLE claim_modifier DROP CONSTRAINT IF EXISTS claim_modifier_kind_check;
ALTER TABLE claim_modifier ADD CONSTRAINT claim_modifier_kind_check CHECK (kind IN ('moderator','mediator','boundary_condition','sample','method'));

-- Candidate pairs: the triage categories of the proposal (Jingle, Jangle, Related but distinct, Unresolved).
ALTER TABLE candidate_pair DROP CONSTRAINT IF EXISTS candidate_pair_hint_check;
-- The list is the widest one (migration 025 added the retrieval strata of signals-v4): every migration is re-run in order by `init` and `migrate`, so this line
-- must never be narrower than the data a later migration allows.
ALTER TABLE candidate_pair ADD CONSTRAINT candidate_pair_hint_check CHECK (hint IN ('jingle_candidate','jangle_candidate','related_but_distinct','unresolved','none','partial_candidate',
  'same_label_divergent','different_label_close','close_measures_distinct_meaning','context_shifted','hard_negative'));
ALTER TABLE candidate_pair ADD COLUMN IF NOT EXISTS c_cit_reading text;
ALTER TABLE candidate_pair ADD COLUMN IF NOT EXISTS n_nom_detail jsonb NOT NULL DEFAULT '{}';
-- Decisions: relation and jingle/jangle flag, as in Appendix B (Decision).
ALTER TABLE pair_decision DROP CONSTRAINT IF EXISTS pair_decision_relation_check;
ALTER TABLE pair_decision ADD CONSTRAINT pair_decision_relation_check CHECK (relation IN
  ('equivalent','partially_overlapping','related_but_distinct','distinct','insufficient_evidence','jingle','jangle','partial_overlap','unresolved'));
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS flag text;
ALTER TABLE pair_decision DROP CONSTRAINT IF EXISTS pair_decision_flag_check;
ALTER TABLE pair_decision ADD CONSTRAINT pair_decision_flag_check CHECK (flag IS NULL OR flag IN ('jingle','jangle','none','unresolved'));
-- The language model's interpretive proposal (step 08a) is kept apart from the dossier and from the human decisions.
CREATE TABLE IF NOT EXISTS pair_proposal (
  proposal_id uuid PRIMARY KEY,
  pair_id uuid NOT NULL REFERENCES candidate_pair(pair_id),
  model text NOT NULL,
  prompt_version text NOT NULL,
  flag text NOT NULL CHECK (flag IN ('jingle','jangle','related_but_distinct','unresolved')),
  rationale text NOT NULL,
  contrary_evidence text,
  evidence_needed text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (pair_id, model, prompt_version)
);
-- Pair sampling: a probabilistic sample (to estimate prevalence) and an enriched challenge set (to stress the system), never mixed.
CREATE TABLE IF NOT EXISTS pair_sample (
  pair_sample_id uuid PRIMARY KEY,
  signal_run_id uuid NOT NULL REFERENCES signal_run(signal_run_id),
  seed integer NOT NULL,
  design jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS pair_sample_item (
  pair_sample_id uuid NOT NULL REFERENCES pair_sample(pair_sample_id),
  pair_id uuid NOT NULL REFERENCES candidate_pair(pair_id),
  sample_kind text NOT NULL CHECK (sample_kind IN ('probabilistic','challenge')),
  stratum text NOT NULL,
  population_size integer NOT NULL,
  sample_size integer NOT NULL,
  PRIMARY KEY (pair_sample_id, pair_id)
);

-- Discovery layer: agents and the scope they may read (the questions archive belongs to the separate discovery component).
CREATE TABLE IF NOT EXISTS agent_profile (
  agent_id text PRIMARY KEY,
  community text CHECK (community IN ('CTO','TIM')),
  permitted_topics text[] NOT NULL DEFAULT '{}',
  retrieval_policy jsonb NOT NULL DEFAULT '{}',
  system_prompt_version text,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE retrieval_run ADD COLUMN IF NOT EXISTS scope jsonb NOT NULL DEFAULT '{}';
ALTER TABLE retrieval_hit ADD COLUMN IF NOT EXISTS selected boolean NOT NULL DEFAULT true;

-- The status of each record in the catalog, derived: metadata only, candidate, under verification, full text unavailable, extracted, validated.
CREATE OR REPLACE VIEW paper_status AS
SELECT c.canonical_id, c.run_id,
  CASE
    WHEN EXISTS (SELECT 1 FROM paper_version v JOIN construct_mention m USING (version_id) WHERE v.canonical_id = c.canonical_id
                 AND NOT EXISTS (SELECT 1 FROM decision_event e WHERE e.subject_id = m.mention_id AND e.decision = 'rejected'))
         AND EXISTS (SELECT 1 FROM decision_event e JOIN construct_mention m ON m.mention_id = e.subject_id WHERE m.canonical_id = c.canonical_id AND e.decision = 'validated') THEN 'validated'
    WHEN EXISTS (SELECT 1 FROM construct_mention m WHERE m.canonical_id = c.canonical_id) THEN 'extracted'
    WHEN EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id = c.canonical_id) THEN 'under_verification'
    WHEN EXISTS (SELECT 1 FROM fulltext_attempt a WHERE a.canonical_id = c.canonical_id) THEN 'full_text_unavailable'
    WHEN EXISTS (SELECT 1 FROM community_preassignment p WHERE p.canonical_id = c.canonical_id AND p.label IN ('yes','uncertain')) THEN 'candidate'
    ELSE 'metadata_only'
  END AS status
FROM canonical_paper c;

-- The agent role also reads what this migration added (the grants in 014 only covered the tables that existed then).
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;
