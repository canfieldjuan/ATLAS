-- Pending onboarding-email drafts enqueued when a first cleaning is booked.
--
-- The won-completion transaction inserts exactly one 'pending' row per
-- booking operation (UNIQUE operation_key makes replays no-ops; the partial
-- unique index allows at most one live pending draft per contact). Nothing
-- sends from this table at enqueue time: the approval surface (A3) is the
-- only writer that may flip status.
--
-- Single-send claim contract (approval surface MUST use this shape):
--   UPDATE eom_onboarding_email_drafts
--      SET status = 'sent', sent_at = NOW(),
--          approved_by_employee_id = $2, approved_by_name = $3
--    WHERE id = $1 AND status = 'pending'
--    RETURNING *;
-- The WHERE status = 'pending' guard makes concurrent approvals settle to
-- exactly one winner (zero rows returned means another session already
-- claimed it). Never read-check-then-update: that pattern double-sends
-- under two concurrent approvers.
--
-- blocker records why a draft cannot send as-is (e.g. 'no_email' when the
-- contact had no email address at enqueue time); the approval surface must
-- resolve the blocker before claiming.
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
        CHECK (status IN ('pending', 'sent', 'revoked')),
    recipient_email TEXT,
    blocker VARCHAR(32),
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    approved_by_employee_id INT,
    approved_by_name TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_onboarding_email_drafts_pending_contact
    ON eom_onboarding_email_drafts (contact_id)
    WHERE status = 'pending';

COMMENT ON TABLE eom_onboarding_email_drafts IS
    'Office-approval queue for onboarding welcome emails; enqueued with the won transition, claimed atomically via UPDATE ... WHERE status = pending RETURNING.';
