-- Migration 276: Evidence annotations for analyst curation (pin, flag, suppress)
-- Allows analysts to curate which evidence witnesses feed into campaign generation.

CREATE TABLE IF NOT EXISTS b2b_evidence_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    witness_id TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    annotation_type TEXT NOT NULL,
    note_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_annotation_type
        CHECK (annotation_type IN ('pin', 'flag', 'suppress'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_annotations_account_witness
    ON b2b_evidence_annotations (account_id, witness_id);

CREATE INDEX IF NOT EXISTS idx_evidence_annotations_account_vendor
    ON b2b_evidence_annotations (account_id, vendor_name);

CREATE INDEX IF NOT EXISTS idx_evidence_annotations_type
    ON b2b_evidence_annotations (account_id, annotation_type);
