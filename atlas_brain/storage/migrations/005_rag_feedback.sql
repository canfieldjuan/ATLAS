-- Migration: 005_rag_feedback.sql
-- RAG source usage tracking for feedback loop

-- Track which RAG sources were used for queries
CREATE TABLE IF NOT EXISTS rag_source_usage (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    source_id VARCHAR(255),
    source_fact TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    was_helpful BOOLEAN DEFAULT NULL,
    feedback_type VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Track aggregate source effectiveness
CREATE TABLE IF NOT EXISTS rag_source_stats (
    id UUID PRIMARY KEY,
    source_id VARCHAR(255) UNIQUE NOT NULL,
    times_retrieved INT DEFAULT 0,
    times_helpful INT DEFAULT 0,
    times_not_helpful INT DEFAULT 0,
    avg_confidence FLOAT DEFAULT 0.0,
    last_retrieved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_rag_source_usage_session
    ON rag_source_usage(session_id);

CREATE INDEX IF NOT EXISTS idx_rag_source_usage_created
    ON rag_source_usage(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rag_source_usage_source
    ON rag_source_usage(source_id);

CREATE INDEX IF NOT EXISTS idx_rag_source_stats_source
    ON rag_source_stats(source_id);
