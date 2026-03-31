-- Migration 250: allow overall_dissatisfaction in vendor pain points.
-- The enrichment/reporting pipeline now uses overall_dissatisfaction as the
-- canonical generic fallback bucket instead of raw "other".

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
    'overall_dissatisfaction',
    'other'
));

COMMIT;
