-- Derived review-level company-signal candidates for analyst-assist workflows.
-- This intentionally stays separate from canonical b2b_company_signals so
-- low-trust or not-yet-high-intent rows can be surfaced without polluting
-- downstream named-account truth tables.

CREATE TABLE IF NOT EXISTS b2b_company_signal_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES b2b_reviews(id) ON DELETE CASCADE,
    company_name TEXT NOT NULL,
    company_name_raw TEXT,
    vendor_name TEXT NOT NULL,
    product_category TEXT,
    source TEXT,
    reviewed_at TIMESTAMPTZ,
    urgency_score NUMERIC(3,1),
    relevance_score NUMERIC(4,3),
    pain_category TEXT,
    buyer_role TEXT,
    decision_maker BOOLEAN,
    seat_count INT,
    contract_end TEXT,
    buying_stage TEXT,
    resolution_confidence TEXT,
    confidence_score NUMERIC(3,2),
    confidence_tier TEXT NOT NULL
        CHECK (confidence_tier IN ('low', 'medium', 'high')),
    signal_evidence_present BOOLEAN NOT NULL DEFAULT FALSE,
    canonical_gap_reason TEXT,
    candidate_bucket TEXT NOT NULL
        CHECK (candidate_bucket IN ('canonical_ready', 'analyst_review')),
    materialization_run_id TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (review_id)
);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidates_vendor_bucket
    ON b2b_company_signal_candidates (vendor_name, candidate_bucket, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidates_company
    ON b2b_company_signal_candidates (company_name);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidates_run_id
    ON b2b_company_signal_candidates (materialization_run_id)
    WHERE materialization_run_id IS NOT NULL;
