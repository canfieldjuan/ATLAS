-- Blog posts: data-backed articles generated from review analysis pipelines.
-- Each post includes markdown content with {{chart:id}} placeholders and a
-- JSONB array of chart specifications rendered by the frontend.

CREATE TABLE IF NOT EXISTS blog_posts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                TEXT NOT NULL UNIQUE,
    title               TEXT NOT NULL,
    description         TEXT,
    topic_type          TEXT NOT NULL,
    tags                JSONB NOT NULL DEFAULT '[]',
    content             TEXT NOT NULL,
    charts              JSONB NOT NULL DEFAULT '[]',
    data_context        JSONB,
    status              TEXT NOT NULL DEFAULT 'draft',
    reviewer_notes      TEXT,
    llm_model           TEXT,
    source_report_date  DATE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_blog_posts_status ON blog_posts(status);
CREATE INDEX IF NOT EXISTS idx_blog_posts_slug ON blog_posts(slug);
