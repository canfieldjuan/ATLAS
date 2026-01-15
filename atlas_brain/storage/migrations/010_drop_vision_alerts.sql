-- Drop deprecated vision_alerts table
-- Migration: 010_drop_vision_alerts.sql
--
-- The vision_alerts table is superseded by the unified alerts table (009).
-- All vision alerts now use the unified alerts table with event_type='vision'.

-- Drop indexes first
DROP INDEX IF EXISTS idx_vision_alerts_triggered_at;
DROP INDEX IF EXISTS idx_vision_alerts_acknowledged;
DROP INDEX IF EXISTS idx_vision_alerts_rule_name;
DROP INDEX IF EXISTS idx_vision_alerts_source_id;
DROP INDEX IF EXISTS idx_vision_alerts_unacked;

-- Drop the table
DROP TABLE IF EXISTS vision_alerts;
