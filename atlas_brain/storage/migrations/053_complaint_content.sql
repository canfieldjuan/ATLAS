-- Complaint content: Claude-generated sellable content from pain point analysis
-- Content types: comparison_article, forum_post, email_copy, review_summary

CREATE TABLE IF NOT EXISTS complaint_content (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_type        TEXT NOT NULL,
    category            TEXT,
    target_asin         TEXT,
    competitor_asin     TEXT,
    title               TEXT,
    body                TEXT NOT NULL,
    pain_point_summary  TEXT,
    source_report_date  DATE,
    status              TEXT NOT NULL DEFAULT 'draft',
    llm_model           TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_complaint_content_type ON complaint_content(content_type);
CREATE INDEX IF NOT EXISTS idx_complaint_content_status ON complaint_content(status);
CREATE INDEX IF NOT EXISTS idx_complaint_content_asin ON complaint_content(target_asin);
