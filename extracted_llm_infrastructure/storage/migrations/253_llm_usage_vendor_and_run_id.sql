-- Add top-level vendor_name and run_id columns to llm_usage
-- for per-vendor cost aggregation and pipeline-run-level rollups.

-- 1. Add nullable columns (backwards-compatible)
ALTER TABLE llm_usage
    ADD COLUMN IF NOT EXISTS vendor_name TEXT,
    ADD COLUMN IF NOT EXISTS run_id TEXT;

-- 2. Backfill vendor_name from direct metadata path
UPDATE llm_usage
SET vendor_name = metadata ->> 'vendor_name'
WHERE vendor_name IS NULL
  AND metadata ->> 'vendor_name' IS NOT NULL
  AND (metadata ->> 'vendor_name') != '';

-- 3. Backfill from nested business.vendor_name path
UPDATE llm_usage
SET vendor_name = metadata #>> '{business,vendor_name}'
WHERE vendor_name IS NULL
  AND metadata #>> '{business,vendor_name}' IS NOT NULL
  AND (metadata #>> '{business,vendor_name}') != '';

-- 4. Backfill run_id from metadata path
UPDATE llm_usage
SET run_id = metadata ->> 'run_id'
WHERE run_id IS NULL
  AND metadata ->> 'run_id' IS NOT NULL
  AND (metadata ->> 'run_id') != '';

-- 5. Partial indexes (only index non-null rows)
CREATE INDEX IF NOT EXISTS idx_llm_usage_vendor_created
    ON llm_usage (vendor_name, created_at DESC)
    WHERE vendor_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_llm_usage_run_id
    ON llm_usage (run_id)
    WHERE run_id IS NOT NULL;
