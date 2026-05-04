-- Podcast transcripts: input table for the podcast repurposing pipeline.
-- Each row is one episode's pre-transcribed text plus metadata.

CREATE TABLE IF NOT EXISTS podcast_transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    episode_id TEXT NOT NULL,
    title TEXT,
    transcript_text TEXT NOT NULL,
    duration_seconds INTEGER,
    publish_date DATE,
    host_name TEXT,
    guest_name TEXT,
    source_url TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_podcast_transcripts_account
    ON podcast_transcripts (account_id, status);

CREATE INDEX IF NOT EXISTS idx_podcast_transcripts_episode
    ON podcast_transcripts (episode_id);

CREATE INDEX IF NOT EXISTS idx_podcast_transcripts_raw_payload
    ON podcast_transcripts USING GIN(raw_payload);
