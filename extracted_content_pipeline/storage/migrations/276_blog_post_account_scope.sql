-- Add host-account scope for extracted AI Content Ops blog-post review paths.
-- Keep the legacy slug uniqueness intact because copied blog writers still
-- use ON CONFLICT (slug).

ALTER TABLE blog_posts
ADD COLUMN IF NOT EXISTS account_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_blog_posts_account_status
    ON blog_posts (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_blog_posts_account_topic
    ON blog_posts (account_id, topic_type, created_at DESC);
