-- Source register of the proposal (Appendix A, "Journal eligibility and source selection"): for each retained journal its exact title, ISSN and eISSN,
-- Scopus Source ID, AJG grade or UTD24 status, the DOI/EID of at least one verified CTO/TIM paper, the community assignment and the date of verification.
-- It sits beside source_list (which stays as it is, with its seven columns) and is append-only per version: a correction is a new version.
CREATE TABLE IF NOT EXISTS source_register (
  version text NOT NULL,
  source_id text NOT NULL CHECK (source_id ~ '^[0-9]+$'),
  exact_title text NOT NULL CHECK (length(btrim(exact_title)) > 0),
  issn text CHECK (issn ~ '^[0-9]{4}-[0-9]{3}[0-9X]$'),
  eissn text CHECK (eissn ~ '^[0-9]{4}-[0-9]{3}[0-9X]$'),
  ajg_grade text CHECK (ajg_grade IN ('1','2','3','4','4*')),
  utd24 boolean NOT NULL DEFAULT false,
  ais_supplement boolean NOT NULL DEFAULT false,
  community text NOT NULL CHECK (community IN ('CTO','TIM','both')),
  verified_doi text,
  verified_eid text,
  verified_on date NOT NULL,
  verified_by text NOT NULL CHECK (length(btrim(verified_by)) > 0),
  title_variants text[] NOT NULL DEFAULT '{}',
  notes text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (version, source_id),
  CHECK (issn IS NOT NULL OR eissn IS NOT NULL),
  CHECK (verified_doi IS NOT NULL OR verified_eid IS NOT NULL)
);
DO $$
BEGIN
  EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO research_agent', current_schema());
END
$$;
