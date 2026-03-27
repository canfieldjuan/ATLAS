-- Migration 240: Expand pain-point taxonomy to reduce "other" bucket.
-- Adds common technical and commercial pain points that were previously
-- forced into the "other" category.

BEGIN;

-- 1. Drop the old check constraint from b2b_vendor_pain_points
-- NOTE: We must first find the constraint name if it wasn't explicitly named.
-- In migration 100 it was implicit. PostgreSQL usually names it 
-- 'b2b_vendor_pain_points_pain_category_check'.

ALTER TABLE b2b_vendor_pain_points 
DROP CONSTRAINT IF EXISTS b2b_vendor_pain_points_pain_category_check;

-- 2. Add the expanded check constraint
ALTER TABLE b2b_vendor_pain_points
ADD CONSTRAINT b2b_vendor_pain_points_pain_category_check
CHECK (pain_category IN (
    'pricing', 'support', 'features', 'ux',
    'reliability', 'performance', 'integration', 'security',
    'onboarding', 'technical_debt', 'contract_lock_in', 
    'data_migration', 'api_limitations', 'other'
));

COMMIT;
