-- Convert single vendor_name TEXT to vendor_names TEXT[] for multi-vendor filtering.

ALTER TABLE b2b_watchlist_views
    ADD COLUMN IF NOT EXISTS vendor_names TEXT[];

UPDATE b2b_watchlist_views
SET vendor_names = ARRAY[vendor_name]
WHERE vendor_name IS NOT NULL
  AND vendor_names IS NULL;

ALTER TABLE b2b_watchlist_views
    DROP COLUMN IF EXISTS vendor_name;
