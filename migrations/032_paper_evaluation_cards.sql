-- Evaluation of published papers, second step (paper_evaluation.py, 20 Oct 2026, the user's request): the full evaluation a forum proposal gets - novelty, methodology, impact -
-- reproduced on the 2026 papers. Methodology and impact need the paper's two cards, made as the cards of the corpus papers are: from the full text (the
-- full texts the project already has: the PDFs of the ANVUR package of 17 Oct 2026, or a PDF a person puts in the folder; nothing is downloaded), the same sections, the same
-- instructions and profile versions, the same check of the quotations.
-- The PDFs and the parsed text stay in the local folder inputs/evaluation_2026 (outside git, like the abstracts); the database keeps what happened and the cards.
-- Both new tables are append-only: the newest row of a paper (and kind) counts. The forum agents must not read them.
CREATE TABLE IF NOT EXISTS evaluation_fulltext (
  attempt_id uuid PRIMARY KEY,
  paper_id text NOT NULL REFERENCES evaluation_paper(paper_id),
  origin text NOT NULL,
  outcome text NOT NULL,
  detail text,
  found_by text,
  oa_status text,
  license text,
  version_label text CHECK (version_label IS NULL OR version_label IN ('published', 'accepted', 'submitted')),
  pdf_url text,
  landing_url text,
  sha256 text,
  size_bytes bigint,
  path text,
  parse_status text,
  parse_detail text,
  text_sha256 text,
  page_count integer,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS evaluation_fulltext_idx ON evaluation_fulltext(paper_id, created_at DESC);
-- project_archive = a PDF the project already had (the ANVUR package, git 87c4aaf); manual_upload = a file a person put in the folder.
-- Added only when missing (21 Oct 2026): every init runs every migration again, and 033 widens the list; re-adding this one would refuse the rows 033 allows.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'evaluation_fulltext_origin_check') THEN
    ALTER TABLE evaluation_fulltext ADD CONSTRAINT evaluation_fulltext_origin_check CHECK (origin IN ('project_archive', 'manual_upload'));
  END IF;
END $$;
CREATE TABLE IF NOT EXISTS evaluation_card (
  card_id uuid PRIMARY KEY,
  paper_id text NOT NULL REFERENCES evaluation_paper(paper_id),
  kind text NOT NULL CHECK (kind IN ('method', 'impact')),
  profile_version text NOT NULL,
  model text NOT NULL,
  source_sha256 text NOT NULL,
  source_info jsonb NOT NULL,
  status text NOT NULL CHECK (status IN ('valid', 'rejected')),
  card jsonb,
  rejection text,
  created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  CHECK ((status = 'valid' AND card IS NOT NULL) OR (status = 'rejected' AND card IS NULL))
);
CREATE INDEX IF NOT EXISTS evaluation_card_idx ON evaluation_card(paper_id, kind, created_at DESC);
-- A run now says which criteria it judges: novelty alone, or the three of a proposal.
ALTER TABLE evaluation_run DROP CONSTRAINT IF EXISTS evaluation_run_kind_check;
ALTER TABLE evaluation_run ADD CONSTRAINT evaluation_run_kind_check CHECK (kind IN ('novelty', 'full'));
ALTER TABLE evaluation_score ADD COLUMN IF NOT EXISTS methodology numeric(4,2) CHECK (methodology IS NULL OR (methodology >= 1 AND methodology <= 10));
ALTER TABLE evaluation_score ADD COLUMN IF NOT EXISTS impact numeric(4,2) CHECK (impact IS NULL OR (impact >= 1 AND impact <= 10));
ALTER TABLE evaluation_score DROP CONSTRAINT IF EXISTS evaluation_score_status_check;
ALTER TABLE evaluation_score DROP CONSTRAINT IF EXISTS evaluation_score_check;
ALTER TABLE evaluation_score ADD CONSTRAINT evaluation_score_status_check CHECK (status IN ('ok', 'partial', 'failed'));
ALTER TABLE evaluation_score ADD CONSTRAINT evaluation_score_check CHECK (
  (status = 'failed' AND novelty IS NULL AND novelty_hybrid IS NULL AND methodology IS NULL AND impact IS NULL)
  OR (status IN ('ok', 'partial') AND num_nonnulls(novelty, methodology, impact) >= 1));
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'research_agent') THEN
    EXECUTE 'REVOKE ALL ON evaluation_fulltext, evaluation_card FROM research_agent';
  END IF;
END
$$;
