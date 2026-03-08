-- Subcategory-level intelligence: GIN index for JSONB containment queries
-- on product_metadata.categories, and subcategory tracking on snapshots.

-- Fast containment queries: WHERE pm.categories @> '["Coffee, Tea & Espresso"]'::jsonb
CREATE INDEX IF NOT EXISTS idx_product_metadata_categories_gin
    ON product_metadata USING GIN (categories);

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
