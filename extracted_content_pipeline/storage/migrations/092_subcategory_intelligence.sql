-- Subcategory tracking on seller category intelligence snapshots.
--
-- This product-owned migration keeps the extracted schema compatible with the
-- current broad/subcategory snapshot conflict target without requiring hosts to
-- install Atlas-only metadata category columns.

-- Add subcategory tracking to existing snapshots table
ALTER TABLE category_intelligence_snapshots
    ADD COLUMN IF NOT EXISTS subcategory TEXT,
    ADD COLUMN IF NOT EXISTS category_path JSONB;

-- Rebuild unique index to include subcategory (existing broad-category rows
-- have subcategory = NULL which COALESCEs to empty string, preserving uniqueness).
DROP INDEX IF EXISTS idx_cat_intel_snapshot_unique;
CREATE UNIQUE INDEX idx_cat_intel_snapshot_unique
    ON category_intelligence_snapshots (category, COALESCE(subcategory, ''), snapshot_date);

-- Fast subcategory lookups
CREATE INDEX IF NOT EXISTS idx_cat_intel_snapshot_subcategory
    ON category_intelligence_snapshots (subcategory) WHERE subcategory IS NOT NULL;
