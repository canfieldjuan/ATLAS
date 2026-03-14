-- Migration 105: Data corrections infrastructure
-- General-purpose table for analyst corrections: suppress, flag, override,
-- merge, reclassify. Covers reviews, vendors, displacement edges, pain points,
-- churn signals, buyer profiles, use cases, and integrations.

CREATE TABLE IF NOT EXISTS data_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- What was corrected
    entity_type TEXT NOT NULL,
    entity_id UUID NOT NULL,

    -- The correction
    correction_type TEXT NOT NULL,
    field_name TEXT,
    old_value TEXT,
    new_value TEXT,

    -- Metadata
    reason TEXT NOT NULL,
    corrected_by TEXT NOT NULL DEFAULT 'analyst',
    status TEXT NOT NULL DEFAULT 'applied',

    -- Affected scope (for bulk corrections like vendor merges)
    affected_count INT DEFAULT 1,
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reverted_at TIMESTAMPTZ,
    reverted_by TEXT
);

-- Constraints (idempotent)
DO $$ BEGIN
    ALTER TABLE data_corrections ADD CONSTRAINT chk_correction_type
        CHECK (correction_type IN ('suppress', 'flag', 'override_field', 'merge_vendor', 'reclassify'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
    ALTER TABLE data_corrections ADD CONSTRAINT chk_correction_status
        CHECK (status IN ('applied', 'reverted', 'pending_review'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
DO $$ BEGIN
    ALTER TABLE data_corrections ADD CONSTRAINT chk_entity_type
        CHECK (entity_type IN (
            'review', 'vendor', 'displacement_edge', 'pain_point',
            'churn_signal', 'buyer_profile', 'use_case', 'integration'
        ));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_data_corrections_entity
    ON data_corrections (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_data_corrections_type
    ON data_corrections (correction_type);
CREATE INDEX IF NOT EXISTS idx_data_corrections_created
    ON data_corrections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_corrections_status
    ON data_corrections (status) WHERE status != 'applied';
