-- Witness governance hardening:
-- - witness specificity metadata
-- - generic-fallback traceability
-- - witness-level hashing for incremental persistence reuse

ALTER TABLE b2b_vendor_witnesses
    ADD COLUMN IF NOT EXISTS specificity_score numeric(8,2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS generic_reason text,
    ADD COLUMN IF NOT EXISTS witness_hash text;

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_witnesses_witness_hash
    ON b2b_vendor_witnesses (witness_hash);
