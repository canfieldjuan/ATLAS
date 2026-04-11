CREATE TABLE IF NOT EXISTS b2b_company_signal_candidate_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT NOT NULL,
    display_company_name TEXT,
    vendor_name TEXT NOT NULL,
    product_category TEXT,
    review_count INTEGER NOT NULL DEFAULT 0,
    distinct_source_count INTEGER NOT NULL DEFAULT 0,
    decision_maker_count INTEGER NOT NULL DEFAULT 0,
    signal_evidence_count INTEGER NOT NULL DEFAULT 0,
    canonical_ready_review_count INTEGER NOT NULL DEFAULT 0,
    avg_urgency_score NUMERIC(4,2),
    max_urgency_score NUMERIC(4,2),
    avg_confidence_score NUMERIC(4,3),
    max_confidence_score NUMERIC(4,3),
    corroborated_confidence_score NUMERIC(4,3),
    confidence_tier TEXT NOT NULL
        CHECK (confidence_tier IN ('low', 'medium', 'high')),
    source_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    gap_reason_distribution JSONB NOT NULL DEFAULT '{}'::jsonb,
    sample_review_ids UUID[] NOT NULL DEFAULT ARRAY[]::uuid[],
    representative_review_id UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL,
    representative_source TEXT,
    representative_pain_category TEXT,
    representative_buyer_role TEXT,
    representative_decision_maker BOOLEAN,
    representative_seat_count INTEGER,
    representative_contract_end TEXT,
    representative_buying_stage TEXT,
    representative_confidence_score NUMERIC(4,3),
    representative_urgency_score NUMERIC(4,2),
    canonical_gap_reason TEXT,
    candidate_bucket TEXT NOT NULL
        CHECK (candidate_bucket IN ('canonical_ready', 'analyst_review')),
    review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'suppressed')),
    review_status_updated_at TIMESTAMPTZ,
    reviewed_by TEXT,
    review_notes TEXT,
    materialization_run_id TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (company_name, vendor_name)
);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidate_groups_vendor_bucket
    ON b2b_company_signal_candidate_groups (
        vendor_name,
        candidate_bucket,
        review_status,
        last_seen_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidate_groups_company
    ON b2b_company_signal_candidate_groups (company_name);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidate_groups_run_id
    ON b2b_company_signal_candidate_groups (materialization_run_id)
    WHERE materialization_run_id IS NOT NULL;
