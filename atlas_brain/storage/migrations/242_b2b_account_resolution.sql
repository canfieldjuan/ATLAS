-- Migration 242: B2B account resolution layer
-- Resolves anonymous review authors to named companies using multi-signal evidence.
-- Runs post-enrichment. Backfills reviewer_company on b2b_reviews for high/medium confidence.

BEGIN;

CREATE TABLE IF NOT EXISTS b2b_account_resolution (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id               UUID NOT NULL REFERENCES b2b_reviews(id) ON DELETE CASCADE,

    -- Source context
    source                  TEXT NOT NULL,
    source_item_url         TEXT,
    author_handle           TEXT,
    author_profile_url      TEXT,

    -- Resolution input
    reviewer_company_raw    TEXT,

    -- Resolution output
    resolved_company_name   TEXT,
    normalized_company_name TEXT,

    -- Confidence
    confidence_score        NUMERIC(3,2) NOT NULL DEFAULT 0.00,
    confidence_label        TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (confidence_label IN ('high', 'medium', 'low', 'unresolved')),

    -- Method and evidence
    resolution_method       TEXT NOT NULL,
    resolution_evidence     JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Status
    resolution_status       TEXT NOT NULL DEFAULT 'resolved'
        CHECK (resolution_status IN ('resolved', 'unresolved', 'excluded', 'superseded')),

    resolved_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One resolution per review (latest wins)
    UNIQUE (review_id)
);

CREATE INDEX IF NOT EXISTS idx_account_res_normalized
    ON b2b_account_resolution (normalized_company_name)
    WHERE normalized_company_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_account_res_confidence
    ON b2b_account_resolution (confidence_label)
    WHERE confidence_label IN ('high', 'medium');

CREATE INDEX IF NOT EXISTS idx_account_res_status
    ON b2b_account_resolution (resolution_status)
    WHERE resolution_status = 'resolved';

CREATE INDEX IF NOT EXISTS idx_account_res_source
    ON b2b_account_resolution (source);

COMMIT;
