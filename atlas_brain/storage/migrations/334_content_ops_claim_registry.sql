-- Tenant-scoped claim and messaging registry for Content Ops review.
--
-- The extracted claims map owns deterministic status checks. This host table
-- stores the approved wording per account so the review workflow can verify a
-- marketer-provided structured claim list without an MCP transport owning
-- business logic.

CREATE TABLE IF NOT EXISTS content_ops_claim_registry (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id        UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    registry_id       TEXT NOT NULL,
    approved_wording  TEXT NOT NULL,
    risk_tier         TEXT,
    expires_on        DATE,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at       TIMESTAMPTZ,
    CONSTRAINT chk_content_ops_claim_registry_registry_id
        CHECK (btrim(registry_id) <> ''),
    CONSTRAINT chk_content_ops_claim_registry_approved_wording
        CHECK (btrim(approved_wording) <> ''),
    CONSTRAINT chk_content_ops_claim_registry_risk_tier
        CHECK (
            risk_tier IS NULL
            OR risk_tier IN ('low', 'medium', 'high', 'critical')
        )
);

CREATE INDEX IF NOT EXISTS idx_content_ops_claim_registry_account_active
    ON content_ops_claim_registry (account_id, updated_at DESC)
    WHERE archived_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_content_ops_claim_registry_account_registry_id_active
    ON content_ops_claim_registry (account_id, lower(btrim(registry_id)))
    WHERE archived_at IS NULL;
