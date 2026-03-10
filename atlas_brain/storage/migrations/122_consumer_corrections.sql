-- Migration 122: Extend data_corrections for consumer entities
--
-- Adds consumer entity types (product_review, product_pain_point, brand,
-- market_report, complaint_content) to the existing corrections table.

ALTER TABLE data_corrections DROP CONSTRAINT IF EXISTS chk_entity_type;
ALTER TABLE data_corrections ADD CONSTRAINT chk_entity_type
    CHECK (entity_type IN (
        -- B2B entities
        'review', 'vendor', 'displacement_edge', 'pain_point',
        'churn_signal', 'buyer_profile', 'use_case', 'integration',
        'source',
        -- Consumer entities
        'product_review', 'product_pain_point', 'brand',
        'market_report', 'complaint_content'
    ));

-- Add merge_brand to correction_type constraint (B2B has merge_vendor)
ALTER TABLE data_corrections DROP CONSTRAINT IF EXISTS chk_correction_type;
ALTER TABLE data_corrections ADD CONSTRAINT chk_correction_type
    CHECK (correction_type IN (
        'suppress', 'flag', 'override_field',
        'merge_vendor', 'merge_brand', 'reclassify'
    ));
