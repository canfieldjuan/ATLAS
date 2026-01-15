-- Vision alerts table for storing triggered alert instances
-- Migration: 008_vision_alerts.sql

CREATE TABLE IF NOT EXISTS vision_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(64) NOT NULL,
    message TEXT NOT NULL,
    source_id VARCHAR(128) NOT NULL,        -- Camera ID
    class_name VARCHAR(64) NOT NULL,        -- Detected class
    node_id VARCHAR(128) NOT NULL,          -- Vision node ID
    event_id VARCHAR(64),                   -- Original event_id from vision node
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(128),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_vision_alerts_triggered_at
    ON vision_alerts(triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_vision_alerts_acknowledged
    ON vision_alerts(acknowledged, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_vision_alerts_rule_name
    ON vision_alerts(rule_name, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_vision_alerts_source_id
    ON vision_alerts(source_id, triggered_at DESC);

-- For finding unacknowledged alerts quickly
CREATE INDEX IF NOT EXISTS idx_vision_alerts_unacked
    ON vision_alerts(triggered_at DESC) WHERE acknowledged = FALSE;

COMMENT ON TABLE vision_alerts IS 'Triggered alerts from vision detection rules';
