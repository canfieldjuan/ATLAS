-- Pending onboarding-email drafts enqueued when a first cleaning is booked.
--
-- The won-completion transaction inserts exactly one 'pending' row per
-- booking operation (UNIQUE operation_key makes replays no-ops; the partial
-- unique index allows at most one live draft per contact). Nothing sends
-- from this table at enqueue time: the approval surface (A3) is the only
-- writer that may advance status.
--
-- Claim ownership is modeled separately from delivery so a crashed or
-- failed send can never be recorded as delivered, and a draft with no
-- usable recipient can never be claimed at all:
--
--   1. Claim (atomic, single winner; readiness predicate built in):
--        UPDATE eom_onboarding_email_drafts
--           SET status = 'sending', claimed_at = NOW(),
--               approved_by_employee_id = $2, approved_by_name = $3
--         WHERE id = $1
--           AND status = 'pending'
--           AND blocker IS NULL
--           AND recipient_email IS NOT NULL
--         RETURNING *;
--      Zero rows returned means another session already claimed it, the
--      draft is blocked (e.g. blocker = 'no_email' -- resolve the blocker
--      and clear it first), or it has no recipient. Never
--      read-check-then-update: that pattern double-sends under two
--      concurrent approvers.
--   2. Send OUTSIDE any open claim transaction, with the draft id as the
--      transport idempotency key (e.g. the Message-ID / provider
--      idempotency header), so a retry of an uncertain send cannot
--      deliver twice.
--   3. Confirm delivery only after the transport accepts:
--        UPDATE eom_onboarding_email_drafts
--           SET status = 'sent', sent_at = NOW()
--         WHERE id = $1 AND status = 'sending';
--   4. Recovery: a row stuck in 'sending' marks a send whose outcome is
--      unknown (worker crashed between claim and confirm). It is operator
--      evidence, not silently retryable -- reconcile against the transport
--      log (query by the draft-id idempotency key), then confirm to 'sent'
--      or revoke. It never re-enters 'pending' automatically.
--
-- blocker records why a draft cannot send as-is (e.g. 'no_email' when the
-- contact had no email address at enqueue time); the claim predicate above
-- refuses blocked rows, so the approval surface must resolve and clear the
-- blocker (and set recipient_email) before claiming.
--
-- Rollback evidence:
--   DROP TABLE IF EXISTS eom_onboarding_email_drafts;
--
-- Roll-forward safety:
--   Purely additive table; no existing reader or writer changes behavior
--   when it exists but is unused.

CREATE TABLE IF NOT EXISTS eom_onboarding_email_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    operation_key VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sending', 'sent', 'revoked')),
    recipient_email TEXT,
    blocker VARCHAR(32),
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    approved_by_employee_id INT,
    approved_by_name TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_onboarding_email_drafts_live_contact
    ON eom_onboarding_email_drafts (contact_id)
    WHERE status IN ('pending', 'sending');

COMMENT ON TABLE eom_onboarding_email_drafts IS
    'Office-approval queue for onboarding welcome emails; enqueued with the won transition, claimed atomically into sending via UPDATE ... WHERE status = pending AND blocker IS NULL RETURNING, confirmed sent only after transport acceptance.';
