-- Staging of an extraction run (13 Oct 2026, the user's request): the output of a new extraction Batch is collected into twin tables and stays there until a person
-- promotes it into the tables the release and the discovery agents read. The twins have the same columns, checks and keys as the real tables (LIKE ... INCLUDING ALL,
-- created after 025 so they carry its wider attribute kinds); they have no foreign keys, which the promotion meets when it copies into the real tables.
-- Nothing is deleted on promotion: the staged rows stay as the record of what came back.
CREATE TABLE IF NOT EXISTS staged_extraction_run (LIKE extraction_run INCLUDING ALL);
CREATE TABLE IF NOT EXISTS staged_construct_mention (LIKE construct_mention INCLUDING ALL);
CREATE TABLE IF NOT EXISTS staged_construct_measure (LIKE construct_measure INCLUDING ALL);
CREATE TABLE IF NOT EXISTS staged_nomological_claim (LIKE nomological_claim INCLUDING ALL);
CREATE TABLE IF NOT EXISTS staged_claim_modifier (LIKE claim_modifier INCLUDING ALL);
CREATE TABLE IF NOT EXISTS staged_extraction_attribute (LIKE extraction_attribute INCLUDING ALL);
-- Who promoted a staged run, when, and how many rows went into each real table. One row per run: a run is promoted once.
CREATE TABLE IF NOT EXISTS extraction_promotion (
  extraction_run_id uuid PRIMARY KEY,
  promoted_by text NOT NULL CHECK (length(btrim(promoted_by)) > 0),
  note text NOT NULL DEFAULT '',
  counts jsonb NOT NULL,
  promoted_at timestamptz NOT NULL DEFAULT now()
);
-- The agents' role must not read what nobody has promoted. The earlier migrations grant SELECT on every table each time they are re-run, so this revoke comes
-- after them; a later migration that grants on all tables again must repeat it.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    EXECUTE 'REVOKE ALL ON staged_extraction_run, staged_construct_mention, staged_construct_measure, staged_nomological_claim, staged_claim_modifier, '
            'staged_extraction_attribute, extraction_promotion FROM research_agent';
  END IF;
END
$$;
