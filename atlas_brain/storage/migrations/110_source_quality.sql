-- Migration 110: Source quality controls
-- Extend data_corrections to support source-level suppression (suppress all
-- reviews from a given source, optionally scoped to a specific vendor).

-- Extend entity_type CHECK to include 'source'
ALTER TABLE data_corrections DROP CONSTRAINT IF EXISTS chk_entity_type;
ALTER TABLE data_corrections ADD CONSTRAINT chk_entity_type
    CHECK (entity_type IN (
        'review', 'vendor', 'displacement_edge', 'pain_point',
        'churn_signal', 'buyer_profile', 'use_case', 'integration',
        'source'
    ));

-- Extend correction_type CHECK to include 'suppress_source'
ALTER TABLE data_corrections DROP CONSTRAINT IF EXISTS chk_correction_type;
ALTER TABLE data_corrections ADD CONSTRAINT chk_correction_type
    CHECK (correction_type IN (
        'suppress', 'flag', 'override_field', 'merge_vendor', 'reclassify',
        'suppress_source'
    ));

-- Partial index for fast source suppression lookups
CREATE INDEX IF NOT EXISTS idx_data_corrections_source_suppress
    ON data_corrections (entity_type, correction_type, status)
    WHERE entity_type = 'source' AND correction_type = 'suppress_source' AND status = 'applied';
