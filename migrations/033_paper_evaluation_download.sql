-- Evaluation of published papers, third step (paper_evaluation.py, 21 Oct 2026, the user's request): the page of the evaluation may now try to DOWNLOAD the open-access PDFs of the
-- 2026 papers that have none, with the resolver of the corpus (fulltext.acquire: only links that Unpaywall, OpenAlex or Semantic Scholar declare open access, from the DOI; principle
-- 8 (a)). It replaces the decision of 20 Oct 2026 that nothing is downloaded for the evaluation. Every attempt is a row of evaluation_fulltext, the failed ones too (principle 5:
-- no PDF = unavailable); the file and its parsed text stay in the local folder inputs/evaluation_2026 (fulltext_download/, outside git).
ALTER TABLE evaluation_fulltext DROP CONSTRAINT IF EXISTS evaluation_fulltext_origin_check;
ALTER TABLE evaluation_fulltext ADD CONSTRAINT evaluation_fulltext_origin_check CHECK (origin IN ('project_archive', 'manual_upload', 'open_access_download'));
