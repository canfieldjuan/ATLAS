-- Call transcripts: post-call transcription and data extraction
-- Stores transcripts, LLM-extracted structured data, and proposed actions

CREATE TABLE IF NOT EXISTS call_transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_sid VARCHAR(128) NOT NULL,
    from_number VARCHAR(32) NOT NULL,
    to_number VARCHAR(32) NOT NULL,
    business_context_id VARCHAR(64) NOT NULL,
    duration_seconds INTEGER,

    -- Transcript
    transcript TEXT,
    transcribed_at TIMESTAMPTZ,

    -- LLM extraction
    summary TEXT,
    extracted_data JSONB DEFAULT '{}'::jsonb,
    proposed_actions JSONB DEFAULT '[]'::jsonb,
    processed_at TIMESTAMPTZ,

    -- Status: pending -> transcribing -> extracting -> ready -> notified
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    notified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_call_transcripts_call_sid ON call_transcripts(call_sid);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_status ON call_transcripts(status);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_created ON call_transcripts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_context ON call_transcripts(business_context_id, created_at DESC);
