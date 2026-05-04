-- Podcast extracted ideas: structured output of the idea-extraction step.
-- Each row is one ranked idea drawn from a single episode's transcript.
--
-- No FK to podcast_transcripts on (account_id, episode_id): the BYO
-- FilePodcastIdeaProvider may inject ideas for an episode whose transcript
-- was never persisted to this database. Logical join on (account_id,
-- episode_id) instead.

CREATE TABLE IF NOT EXISTS podcast_extracted_ideas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT,
    episode_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    summary TEXT NOT NULL,
    arguments JSONB NOT NULL DEFAULT '[]'::jsonb,
    hooks JSONB NOT NULL DEFAULT '[]'::jsonb,
    key_quotes JSONB NOT NULL DEFAULT '[]'::jsonb,
    teaching_moments JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, episode_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_podcast_ideas_episode
    ON podcast_extracted_ideas (account_id, episode_id, rank);
