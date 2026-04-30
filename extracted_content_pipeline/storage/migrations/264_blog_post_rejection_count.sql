-- Track how many times a blog post slug has been rejected.
-- Used to cap regeneration attempts and stop wasting LLM tokens
-- on slugs that repeatedly fail the quality gate.
ALTER TABLE blog_posts
    ADD COLUMN IF NOT EXISTS rejection_count integer NOT NULL DEFAULT 0;
