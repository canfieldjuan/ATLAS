-- 067: Backfill the timestamp column expected by sequence migrations.
--
-- Migration 066 created b2b_campaigns without updated_at. Migration 068's
-- CREATE TABLE IF NOT EXISTS fallback includes it only for first-time setup,
-- so existing 066-created tables need the column before 309 cancels campaigns.

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
