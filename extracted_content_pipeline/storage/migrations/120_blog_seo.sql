-- SEO metadata columns for blog posts.
ALTER TABLE blog_posts
    ADD COLUMN IF NOT EXISTS seo_title TEXT,
    ADD COLUMN IF NOT EXISTS seo_description TEXT,
    ADD COLUMN IF NOT EXISTS target_keyword TEXT,
    ADD COLUMN IF NOT EXISTS secondary_keywords JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS faq JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS related_slugs JSONB NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_blog_posts_keyword
    ON blog_posts (target_keyword) WHERE target_keyword IS NOT NULL;
