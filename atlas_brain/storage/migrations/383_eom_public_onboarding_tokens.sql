-- atlas: atomic-bookkeeping
-- Durable, revocable authority for an approved EOM public-onboarding link.
--
-- The raw bearer is intentionally absent. Atlas regenerates
-- eomob1.<token-id>.<HMAC> from this opaque UUID and its Atlas-only secret only
-- while composing the approved email or validating a tracker request. The row
-- remains after redemption/revocation as the audit and idempotency record.
--
-- `handoff_id` deliberately has no cross-table foreign key. Migration 354
-- transfers eom_customer_handoffs to a no-login guard owner and grants the app
-- runtime DML, not REFERENCES; adding an FK here would make a normal additive
-- migration depend on temporary guard-role membership. The application writes
-- it only after the immutable handoff row is returned in the same transaction,
-- and the integration suite proves that relation.

CREATE TABLE IF NOT EXISTS eom_public_onboarding_tokens (
    id UUID NOT NULL,
    draft_id UUID NOT NULL REFERENCES eom_onboarding_email_drafts(id)
        ON DELETE RESTRICT,
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    signing_key_fingerprint VARCHAR(64) NOT NULL
        CHECK (signing_key_fingerprint ~ '^[0-9a-f]{64}$'),
    prefill_full_name VARCHAR(256) NOT NULL,
    prefill_email VARCHAR(256),
    prefill_phone VARCHAR(32),
    prefill_address TEXT,
    prefill_city VARCHAR(128),
    prefill_state VARCHAR(64),
    prefill_zip VARCHAR(16),
    prefill_customer_type VARCHAR(32) NOT NULL,
    approval_key VARCHAR(128) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'issued',
    approved_by_employee_id BIGINT NOT NULL CHECK (approved_by_employee_id > 0),
    approved_by_name VARCHAR(128) NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    redeemed_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    handoff_id UUID,
    CONSTRAINT pk_eom_public_onboarding_tokens PRIMARY KEY (id),
    CONSTRAINT uq_eom_public_onboarding_tokens_draft UNIQUE (draft_id),
    CONSTRAINT uq_eom_public_onboarding_tokens_approval UNIQUE (approval_key),
    CONSTRAINT uq_eom_public_onboarding_tokens_handoff UNIQUE (handoff_id),
    CONSTRAINT ck_eom_public_onboarding_tokens_status
        CHECK (status IN ('issued', 'redeemed', 'revoked')),
    CONSTRAINT ck_eom_public_onboarding_tokens_terminal_state CHECK (
        (status = 'issued'
            AND redeemed_at IS NULL
            AND revoked_at IS NULL
            AND handoff_id IS NULL)
        OR (status = 'redeemed'
            AND redeemed_at IS NOT NULL
            AND revoked_at IS NULL
            AND handoff_id IS NOT NULL)
        OR (status = 'revoked'
            AND redeemed_at IS NULL
            AND revoked_at IS NOT NULL
            AND handoff_id IS NULL)
    )
);

-- One active link fences one contact from a competing office handoff. The row
-- remains after terminal state, so a partial index is both the correct policy
-- and keeps historical email evidence intact.
CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_public_onboarding_tokens_issued_contact
    ON eom_public_onboarding_tokens (contact_id)
    WHERE status = 'issued';

CREATE INDEX IF NOT EXISTS idx_eom_public_onboarding_tokens_status
    ON eom_public_onboarding_tokens (status, issued_at DESC);

COMMENT ON TABLE eom_public_onboarding_tokens IS
    'Atlas-owned opaque public-onboarding token state; one HMAC-signed bearer, signing-key fingerprint, and immutable prefill snapshot per approved email draft, explicitly revoked or redeemed into one immutable EOM customer handoff.';
