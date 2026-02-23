-- Migration 048: Reasoning Agent Foundation
-- Persisted event bus, entity locks, and reasoning queue for the
-- cross-domain ReasoningAgent.

-- Persisted event bus
CREATE TABLE IF NOT EXISTS atlas_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    processing_result JSONB
);

CREATE INDEX IF NOT EXISTS idx_atlas_events_type
    ON atlas_events (event_type);
CREATE INDEX IF NOT EXISTS idx_atlas_events_entity
    ON atlas_events (entity_type, entity_id)
    WHERE entity_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_atlas_events_unprocessed
    ON atlas_events (created_at)
    WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_atlas_events_created
    ON atlas_events (created_at DESC);

-- Entity locks for sovereignty (AtlasAgent holds lock during voice session)
CREATE TABLE IF NOT EXISTS entity_locks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    holder TEXT NOT NULL,
    session_id TEXT,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_locks_active
    ON entity_locks (entity_type, entity_id)
    WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_entity_locks_session
    ON entity_locks (session_id)
    WHERE released_at IS NULL;

-- Queued decisions for locked entities
CREATE TABLE IF NOT EXISTS reasoning_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES atlas_events(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drained_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_reasoning_queue_entity
    ON reasoning_queue (entity_type, entity_id)
    WHERE drained_at IS NULL;

-- NOTIFY trigger on atlas_events INSERT
CREATE OR REPLACE FUNCTION notify_atlas_event() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('atlas_events', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_atlas_events_notify ON atlas_events;
CREATE TRIGGER trg_atlas_events_notify
    AFTER INSERT ON atlas_events
    FOR EACH ROW EXECUTE FUNCTION notify_atlas_event();
