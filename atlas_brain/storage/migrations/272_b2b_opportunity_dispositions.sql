-- Migration 272: Persistent opportunity dispositions for the Campaign Opportunity Workbench
-- Replaces ephemeral client-side hiddenIds with server-persisted save/snooze/dismiss states.
-- No row = active (default). Only non-active dispositions create rows.

CREATE TABLE IF NOT EXISTS b2b_opportunity_dispositions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    opportunity_key TEXT NOT NULL,
    company TEXT NOT NULL,
    vendor TEXT NOT NULL,
    review_id TEXT,
    disposition TEXT NOT NULL DEFAULT 'dismissed',
    snoozed_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_disposition_value
        CHECK (disposition IN ('snoozed', 'dismissed', 'saved')),
    CONSTRAINT chk_snooze_coherence
        CHECK (
            (disposition = 'snoozed' AND snoozed_until IS NOT NULL)
            OR (disposition <> 'snoozed')
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_opp_disp_account_key
    ON b2b_opportunity_dispositions (account_id, opportunity_key);

CREATE INDEX IF NOT EXISTS idx_opp_disp_account_disposition
    ON b2b_opportunity_dispositions (account_id, disposition);

CREATE INDEX IF NOT EXISTS idx_opp_disp_snoozed_until
    ON b2b_opportunity_dispositions (snoozed_until)
    WHERE disposition = 'snoozed' AND snoozed_until IS NOT NULL;
