-- Unified alerts table for all event types
-- Migration: 009_unified_alerts.sql
--
-- This table supports alerts from multiple sources:
-- - vision: YOLO detection events
-- - audio: YAMNet audio classification
-- - ha_state: Home Assistant state changes
-- - security: Kafka security events

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    message TEXT NOT NULL,
    source_id VARCHAR(128) NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(128),
    event_data JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Primary indexes for common queries
CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at
    ON alerts(triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_event_type
    ON alerts(event_type, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged
    ON alerts(acknowledged, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_rule_name
    ON alerts(rule_name, triggered_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_source_id
    ON alerts(source_id, triggered_at DESC);

-- Partial index for unacknowledged alerts
CREATE INDEX IF NOT EXISTS idx_alerts_unacked
    ON alerts(triggered_at DESC) WHERE acknowledged = FALSE;

-- Composite index for filtered queries
CREATE INDEX IF NOT EXISTS idx_alerts_type_source
    ON alerts(event_type, source_id, triggered_at DESC);

COMMENT ON TABLE alerts IS 'Unified alerts from all event sources (vision, audio, ha_state, security)';
COMMENT ON COLUMN alerts.event_type IS 'Source type: vision, audio, ha_state, security';
COMMENT ON COLUMN alerts.event_data IS 'Full event snapshot as JSON';
