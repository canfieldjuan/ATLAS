-- 020_presence_and_actions.sql
-- Phase 3: Presence events + Proactive actions tables

-- Presence state transitions (arrival / departure)
CREATE TABLE IF NOT EXISTS presence_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transition VARCHAR(16) NOT NULL,       -- 'arrival', 'departure'
    occupancy_state VARCHAR(16) NOT NULL,  -- 'empty', 'occupied', 'identified'
    occupants TEXT[] DEFAULT '{}',
    person_name VARCHAR(128),
    source_id VARCHAR(128) NOT NULL DEFAULT 'system',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_presence_events_created
    ON presence_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_presence_events_transition
    ON presence_events (transition, created_at DESC);


-- Proactive actions extracted from user conversations
CREATE TABLE IF NOT EXISTS proactive_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_text TEXT NOT NULL,
    action_text_hash VARCHAR(64) NOT NULL,
    action_type VARCHAR(16) NOT NULL DEFAULT 'task',  -- task, reminder, scheduled_task
    source_time TIMESTAMP,
    session_id UUID,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',    -- pending, done, dismissed
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Partial unique index — only one pending action per text
CREATE UNIQUE INDEX IF NOT EXISTS idx_proactive_actions_hash
    ON proactive_actions (action_text_hash) WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_proactive_actions_status
    ON proactive_actions (status, created_at DESC);
