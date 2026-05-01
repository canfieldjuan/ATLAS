-- Campaign engagement timing metrics for send optimization.
-- Migration 150
--
-- hours_to_first_open / hours_to_first_click: computed when ESP webhook
-- fires first open/click event. NULL until event arrives.
-- contact_send_count: total emails sent to this recipient across all
-- sequences. Incremented at send time for fatigue detection.

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS hours_to_first_open  REAL,
    ADD COLUMN IF NOT EXISTS hours_to_first_click REAL;

ALTER TABLE campaign_sequences
    ADD COLUMN IF NOT EXISTS contact_send_count INT NOT NULL DEFAULT 0;
