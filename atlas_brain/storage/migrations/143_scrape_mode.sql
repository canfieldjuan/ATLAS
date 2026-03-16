-- Per-target scrape mode: incremental (shallow, hourly) vs exhaustive (deep, monthly)
-- Migration 143

ALTER TABLE b2b_scrape_targets
    ADD COLUMN IF NOT EXISTS scrape_mode TEXT NOT NULL DEFAULT 'incremental';

-- Guard: ADD CONSTRAINT has no IF NOT EXISTS, so wrap in a DO block
DO $$
BEGIN
    ALTER TABLE b2b_scrape_targets
        ADD CONSTRAINT chk_scrape_mode CHECK (scrape_mode IN ('incremental', 'exhaustive'));
EXCEPTION WHEN duplicate_object THEN
    NULL;
END
$$;

-- Allow same product_slug with different modes (vendor can have both)
DROP INDEX IF EXISTS idx_b2b_scrape_targets_dedup;
CREATE UNIQUE INDEX idx_b2b_scrape_targets_dedup
    ON b2b_scrape_targets(source, product_slug, scrape_mode);
