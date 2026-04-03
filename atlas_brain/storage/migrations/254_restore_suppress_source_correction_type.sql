-- Restore suppress_source support on data_corrections after migration 122
-- accidentally removed it from chk_correction_type.

ALTER TABLE data_corrections DROP CONSTRAINT IF EXISTS chk_correction_type;
ALTER TABLE data_corrections ADD CONSTRAINT chk_correction_type
    CHECK (correction_type IN (
        'suppress', 'flag', 'override_field',
        'merge_vendor', 'merge_brand', 'reclassify',
        'suppress_source'
    ));

CREATE INDEX IF NOT EXISTS idx_data_corrections_source_suppress
    ON data_corrections (entity_type, correction_type, status)
    WHERE entity_type = 'source' AND correction_type = 'suppress_source' AND status = 'applied';
