-- B2B enrichment repair and low-fidelity quarantine metadata.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS enrichment_baseline JSONB,
    ADD COLUMN IF NOT EXISTS enrichment_repair JSONB,
    ADD COLUMN IF NOT EXISTS enrichment_repair_status TEXT,
    ADD COLUMN IF NOT EXISTS enrichment_repair_attempts INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS enrichment_repair_model TEXT,
    ADD COLUMN IF NOT EXISTS enrichment_repaired_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS enrichment_repair_applied_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS low_fidelity BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS low_fidelity_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS low_fidelity_detected_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_low_fidelity
    ON b2b_reviews (low_fidelity)
    WHERE low_fidelity = TRUE;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_repair_status
    ON b2b_reviews (enrichment_repair_status, enriched_at DESC)
    WHERE enrichment_status = 'enriched';
