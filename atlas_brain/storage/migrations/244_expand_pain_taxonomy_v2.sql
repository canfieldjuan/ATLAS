-- Migration 244: Expand pain taxonomy with 4 new categories.
-- Adds outcome_gap, admin_burden, ai_hallucination, integration_debt to reduce
-- the "other" bucket. These map to complaint patterns that were previously
-- unclassifiable with the old 14-category taxonomy.

BEGIN;

ALTER TABLE b2b_vendor_pain_points
DROP CONSTRAINT IF EXISTS b2b_vendor_pain_points_pain_category_check;

ALTER TABLE b2b_vendor_pain_points
ADD CONSTRAINT b2b_vendor_pain_points_pain_category_check
CHECK (pain_category IN (
    'pricing', 'support', 'features', 'ux',
    'reliability', 'performance', 'integration', 'security',
    'onboarding', 'technical_debt', 'contract_lock_in',
    'data_migration', 'api_limitations',
    'outcome_gap', 'admin_burden', 'ai_hallucination', 'integration_debt',
    'other'
));

COMMIT;
