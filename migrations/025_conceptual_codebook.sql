-- Proposal v4 (13 Oct 2026), the pipeline side. Additive: nothing is dropped, and every row written before keeps its meaning.
--
-- 1. Extraction: the meaning profile of a construct use (referent, defining features, exclusions, temporal form, theoretical role, scope) is recorded apart
--    from its operationalization (measures, informant, time window, dimensionality, reliability, validity).
ALTER TABLE extraction_attribute DROP CONSTRAINT IF EXISTS extraction_attribute_kind_check;
ALTER TABLE extraction_attribute ADD CONSTRAINT extraction_attribute_kind_check CHECK (kind IN (
  'synonym','acronym','level','dimensionality','informant','reliability','validity','mechanism','boundary_condition','context','time_window','population',
  'referent','defining_feature','exclusion','temporal_form','theoretical_role','scope_condition'));

-- 2. Signals: S_def (definitions and meaning profiles) and S_op (items and measures) are separate; their disagreement is informative. The hint of a v4 run is
--    a retrieval stratum, never a verdict. The hints of earlier runs stay valid.
ALTER TABLE candidate_pair ADD COLUMN IF NOT EXISTS s_def double precision;
ALTER TABLE candidate_pair ADD COLUMN IF NOT EXISTS s_op double precision;
ALTER TABLE candidate_pair DROP CONSTRAINT IF EXISTS candidate_pair_hint_check;
ALTER TABLE candidate_pair ADD CONSTRAINT candidate_pair_hint_check CHECK (hint IN (
  'jingle_candidate','jangle_candidate','related_but_distinct','unresolved','none','partial_candidate',
  'same_label_divergent','different_label_close','close_measures_distinct_meaning','context_shifted','hard_negative'));

-- 3. Decisions: the codebook of the proposal v4 (Appendix B, Decision) with its comparison checklist. The rows written before (relation + flag) are codebook v1:
--    kept as history, never counted by the v2 readers (the user's decision of 13 Oct 2026: the pairs are decided again).
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS codebook_version text NOT NULL DEFAULT 'pair-codebook-v1';
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS decision_class text;
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS same_referent text;
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS compatible_properties text;
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS substitution_changes text;
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS theoretical_consequence text;
ALTER TABLE pair_decision ADD COLUMN IF NOT EXISTS counterexample text;
ALTER TABLE pair_decision ALTER COLUMN relation DROP NOT NULL;
ALTER TABLE pair_decision DROP CONSTRAINT IF EXISTS pair_decision_codebook_check;
ALTER TABLE pair_decision ADD CONSTRAINT pair_decision_codebook_check CHECK (
  (codebook_version = 'pair-codebook-v1' AND relation IS NOT NULL)
  OR (codebook_version = 'pair-codebook-v2' AND decision_class IS NOT NULL AND same_referent IS NOT NULL AND compatible_properties IS NOT NULL AND substitution_changes IS NOT NULL
      AND decision_class IN ('conceptual_jingle','conceptual_jangle','measurement_divergence','scope_variation','empirical_redundancy',
                                                                  'part_whole','related_but_distinct','distinct','same_meaning','unresolved')
      AND same_referent IN ('yes','no','unclear') AND compatible_properties IN ('yes','no','unclear') AND substitution_changes IN ('yes','no','unclear')));
CREATE INDEX IF NOT EXISTS pair_decision_codebook_idx ON pair_decision(codebook_version, pair_id);

-- 4. Step 08a: the model's proposal names a class of the same codebook and the result of its substitution test.
ALTER TABLE pair_proposal ALTER COLUMN flag DROP NOT NULL;
ALTER TABLE pair_proposal ADD COLUMN IF NOT EXISTS decision_class text;
ALTER TABLE pair_proposal ADD COLUMN IF NOT EXISTS substitution_test text;
ALTER TABLE pair_proposal DROP CONSTRAINT IF EXISTS pair_proposal_class_check;
ALTER TABLE pair_proposal ADD CONSTRAINT pair_proposal_class_check CHECK (flag IS NOT NULL OR decision_class IS NOT NULL);

DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;
