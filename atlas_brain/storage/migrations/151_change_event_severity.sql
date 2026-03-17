-- Change event severity classification via z-score.
-- Migration 151
--
-- z_score: statistical z-score vs vendor's historical distribution.
--   Already computed by temporal anomaly detection but was only embedded
--   in description text. Now a queryable column.
-- severity: derived from z_score magnitude.
--   critical (|z| >= 3), high (>= 2), moderate (>= 1.5), low (< 1.5 or NULL)

ALTER TABLE b2b_change_events
    ADD COLUMN IF NOT EXISTS z_score  REAL,
    ADD COLUMN IF NOT EXISTS severity TEXT;

CREATE INDEX IF NOT EXISTS idx_bce_severity
    ON b2b_change_events (severity, event_date DESC)
    WHERE severity IS NOT NULL;
