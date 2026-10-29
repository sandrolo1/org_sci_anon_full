-- Proposal v4 (13 Oct 2026): the discovery archive records, for every question, the ideation trigger the episode followed, the numeric seed that reproduces its
-- choices, the opener and the opening theme of an encounter (C), the alternative puzzles that were weighed and discarded, and the conceptual signature (phenomenon,
-- challenged assumption, mechanism, explanatory contrast, level) that lets repeated ideas be detected beyond their wording. The full question stays in `content`;
-- these columns make the fields of Appendix D queryable. Questions written before this migration have NULL here: they had no trigger.
ALTER TABLE question ADD COLUMN IF NOT EXISTS ideation_trigger text;
ALTER TABLE question ADD COLUMN IF NOT EXISTS numeric_seed bigint;
ALTER TABLE question ADD COLUMN IF NOT EXISTS opener text;
ALTER TABLE question ADD COLUMN IF NOT EXISTS opening_theme jsonb;
ALTER TABLE question ADD COLUMN IF NOT EXISTS alternative_puzzles jsonb;
ALTER TABLE question ADD COLUMN IF NOT EXISTS conceptual_signature jsonb;
CREATE INDEX IF NOT EXISTS question_trigger_idx ON question(ideation_trigger);
