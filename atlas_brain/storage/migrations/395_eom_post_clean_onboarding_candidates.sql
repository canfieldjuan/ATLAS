-- atlas: atomic-bookkeeping
-- Non-sendable post-clean onboarding candidates derived from durable first-clean
-- completion receipts. This migration creates no email, token, or Stripe effect.
--
-- Rollback: deploy the previous application and retain this additive table as
-- audit/recovery evidence. Old code ignores it; do not delete candidate history.

CREATE TABLE IF NOT EXISTS eom_post_clean_onboarding_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    completion_receipt_id UUID NOT NULL UNIQUE,
    contact_id UUID NOT NULL UNIQUE
        REFERENCES contacts(id) ON DELETE RESTRICT,
    handoff_id UUID NOT NULL UNIQUE,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids'
        CHECK (business_context_id = 'effingham_maids'),
    status VARCHAR(24) NOT NULL DEFAULT 'pending'
        CHECK (status = 'pending'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_eom_post_clean_candidates_pending
    ON eom_post_clean_onboarding_candidates (status, created_at DESC, id DESC);

-- Fresh databases may run ordinary migrations before the separately controlled
-- DBA migration 394. Backfill only when its immutable receipt relation exists;
-- an exact completion retry also heals any receipt deployed before this table.
DO $$
BEGIN
    IF to_regclass('eom_first_clean_completion_receipts') IS NOT NULL THEN
        INSERT INTO eom_post_clean_onboarding_candidates (
            completion_receipt_id,
            contact_id,
            handoff_id
        )
        SELECT receipt.id, receipt.contact_id, receipt.handoff_id
        FROM eom_first_clean_completion_receipts AS receipt
        ON CONFLICT (completion_receipt_id) DO NOTHING;
    END IF;
END;
$$;
