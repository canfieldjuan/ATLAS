-- Cross-vendor reasoning conclusions (pairwise battles, category councils, resource-asymmetry)
CREATE TABLE IF NOT EXISTS b2b_cross_vendor_conclusions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_type TEXT NOT NULL,            -- pairwise_battle, category_council, resource_asymmetry
    vendors TEXT[] NOT NULL,                -- sorted vendor names
    category TEXT,                          -- for category_council mode
    conclusion JSONB NOT NULL,              -- full LLM output
    confidence NUMERIC(3,2) NOT NULL,
    evidence_hash TEXT NOT NULL,
    tokens_used INT NOT NULL DEFAULT 0,
    cached BOOLEAN NOT NULL DEFAULT false,
    computed_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_xv_conclusions_type_date
    ON b2b_cross_vendor_conclusions (analysis_type, computed_date DESC);
CREATE INDEX IF NOT EXISTS idx_xv_conclusions_vendors
    ON b2b_cross_vendor_conclusions USING GIN (vendors);
