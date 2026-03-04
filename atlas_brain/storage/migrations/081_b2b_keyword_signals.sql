-- 081: B2B keyword search volume signals (Google Trends)
-- Tracks weekly search volume for churn-indicative queries per vendor.

CREATE TABLE IF NOT EXISTS b2b_keyword_signals (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name       TEXT NOT NULL,
    keyword           TEXT NOT NULL,
    query_template    TEXT NOT NULL,
    volume_relative   INT NOT NULL,
    rolling_avg_4w    NUMERIC(5,1),
    volume_change_pct NUMERIC(6,2),
    is_spike          BOOLEAN NOT NULL DEFAULT false,
    snapshot_week     DATE NOT NULL,
    snapshot_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_response      JSONB NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_keyword_signals_dedup
    ON b2b_keyword_signals (vendor_name, keyword, snapshot_week);

CREATE INDEX IF NOT EXISTS idx_b2b_keyword_signals_spike
    ON b2b_keyword_signals (vendor_name, is_spike)
    WHERE is_spike = true;

-- Add keyword columns to b2b_churn_signals
ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS keyword_spike_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS keyword_spike_keywords JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS keyword_trend_summary JSONB NOT NULL DEFAULT '{}';
