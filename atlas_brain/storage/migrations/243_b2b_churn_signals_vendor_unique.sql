-- Migration 243: Deduplicate b2b_churn_signals to one row per vendor
-- The old unique index on (vendor_name, product_category) allowed stale rows
-- with generic categories (e.g. 'B2B Software') to accumulate alongside the
-- correct category row. One row per vendor is the right model.

-- 1. Remove duplicate rows, keeping the row with the highest total_reviews per vendor
DELETE FROM b2b_churn_signals
WHERE id NOT IN (
    SELECT DISTINCT ON (vendor_name) id
    FROM b2b_churn_signals
    ORDER BY vendor_name, total_reviews DESC, last_computed_at DESC
);

-- 2. Drop old composite unique index
DROP INDEX IF EXISTS idx_b2b_churn_signals_vendor_category;

-- 3. Create new unique index on vendor_name only
CREATE UNIQUE INDEX idx_b2b_churn_signals_vendor ON public.b2b_churn_signals (vendor_name);
