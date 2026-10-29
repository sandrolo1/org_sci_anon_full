-- A new attempt outcome: the site answered with a bot-protection wall (Cloudflare, AWS WAF, "Human Verification").
-- It is not a transient failure (download_failed, retried every run) and not a statement about the paper: the person fetches it by hand.
-- fulltext_attempt is append-only; only the allowed outcomes change, no row is touched. Safe to run again.
ALTER TABLE fulltext_attempt DROP CONSTRAINT IF EXISTS fulltext_attempt_outcome_check;
ALTER TABLE fulltext_attempt ADD CONSTRAINT fulltext_attempt_outcome_check
  CHECK (outcome IN ('downloaded','no_doi','not_open_access','no_pdf_link','download_failed','blocked_by_site','not_a_pdf','too_large','lookup_failed'));
