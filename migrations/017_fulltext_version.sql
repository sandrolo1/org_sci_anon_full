-- Which version of the paper a stored PDF is, when the source says so (Unpaywall and OpenAlex follow the DRIVER vocabulary).
-- NULL means unknown, which is what every hand upload and every file stored before this migration has.
-- A copy from a repository is often an accepted manuscript or a preprint, not the version of record, and extraction must be able to tell.
ALTER TABLE fulltext_file ADD COLUMN IF NOT EXISTS version_label text;
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fulltext_file_version_label_check') THEN
    ALTER TABLE fulltext_file ADD CONSTRAINT fulltext_file_version_label_check CHECK (version_label IN ('published','accepted','submitted'));
  END IF;
END $$;
