-- Track why article enrichment attempts fail so backlog triage can distinguish
-- recoverable fetch issues from blocked publishers and classifier failures.
ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS enrichment_failure_reason TEXT;
