CREATE TABLE IF NOT EXISTS b2b_company_signal_review_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_batch_id UUID NOT NULL,
    review_scope TEXT NOT NULL CHECK (review_scope IN ('candidate', 'group', 'bulk_group')),
    review_action TEXT NOT NULL CHECK (review_action IN ('approved', 'suppressed')),
    candidate_id UUID REFERENCES b2b_company_signal_candidates(id) ON DELETE SET NULL,
    candidate_group_id UUID REFERENCES b2b_company_signal_candidate_groups(id) ON DELETE SET NULL,
    company_name TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    review_notes TEXT,
    company_signal_id UUID REFERENCES b2b_company_signals(id) ON DELETE SET NULL,
    company_signal_action TEXT NOT NULL CHECK (company_signal_action IN ('created', 'updated', 'deleted', 'none')),
    rebuild_requested BOOLEAN NOT NULL DEFAULT FALSE,
    rebuild_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    rebuild_reason TEXT,
    rebuild_as_of DATE,
    rebuild_persisted_count INT,
    rebuild_total_accounts INT,
    rebuild_vendor_count INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_review_events_created_at
    ON b2b_company_signal_review_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_review_events_batch_vendor
    ON b2b_company_signal_review_events (review_batch_id, vendor_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_review_events_vendor
    ON b2b_company_signal_review_events (vendor_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_review_events_group
    ON b2b_company_signal_review_events (candidate_group_id, created_at DESC);
