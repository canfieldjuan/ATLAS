-- Vision events table for storing detection events from atlas_vision nodes
-- Migration: 007_vision_events.sql

CREATE TABLE IF NOT EXISTS vision_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(64) NOT NULL,          -- Original event ID from vision node
    event_type VARCHAR(32) NOT NULL,        -- new_track, track_lost, track_update
    track_id INTEGER NOT NULL,
    class_name VARCHAR(64) NOT NULL,        -- person, car, dog, etc.
    source_id VARCHAR(128) NOT NULL,        -- Camera ID
    node_id VARCHAR(128) NOT NULL,          -- Vision node ID
    bbox_x1 REAL,                           -- Normalized bounding box (0-1)
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    event_timestamp TIMESTAMP NOT NULL,     -- When event occurred on vision node
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When brain received it
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_vision_events_event_timestamp
    ON vision_events(event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_vision_events_source_id
    ON vision_events(source_id, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_vision_events_node_id
    ON vision_events(node_id, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_vision_events_class_name
    ON vision_events(class_name, event_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_vision_events_event_type
    ON vision_events(event_type, event_timestamp DESC);

-- Composite index for common filtering
CREATE INDEX IF NOT EXISTS idx_vision_events_source_class
    ON vision_events(source_id, class_name, event_timestamp DESC);

-- Prevent duplicate events (same event_id from same node)
CREATE UNIQUE INDEX IF NOT EXISTS idx_vision_events_unique_event
    ON vision_events(node_id, event_id);

-- Comment on table
COMMENT ON TABLE vision_events IS 'Detection events from atlas_vision nodes';
