-- Podcast format drafts: per-format generated content for one extracted idea.
-- Five formats are locked by CHECK constraint; adding a sixth requires a
-- follow-on migration to drop and re-add the constraint.

CREATE TABLE IF NOT EXISTS podcast_format_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT,
    idea_id UUID NOT NULL REFERENCES podcast_extracted_ideas(id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL,
    format_type TEXT NOT NULL
        CHECK (format_type IN ('newsletter', 'blog', 'linkedin', 'x_thread', 'shorts')),
    title TEXT,
    body TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_audit JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_podcast_drafts_episode_format
    ON podcast_format_drafts (account_id, episode_id, format_type);

CREATE INDEX IF NOT EXISTS idx_podcast_drafts_idea
    ON podcast_format_drafts (idea_id);
