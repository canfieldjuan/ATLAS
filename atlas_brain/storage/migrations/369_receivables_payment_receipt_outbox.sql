-- Durable, non-sending receipt-delivery evidence for residential payments.
--
-- This belongs beside customer_payments in the receivables ledger, not in a
-- mail provider or the separate canonical-CRM pool.  A payment and its receipt
-- enqueue must commit or roll back together; transport delivery happens later,
-- outside that financial transaction.
--
-- One row per payment is the idempotency anchor.  A retry of the same payment
-- returns the original payment before this INSERT is reachable, and this UNIQUE
-- constraint remains the database backstop if a future writer is wrong.
--
-- Delivery state is deliberately closed.  This migration writes no sent-mail
-- evidence: a future sender must move pending/failed rows only after an
-- explicit claim and verifiable transport result.  `skipped/no_email` records
-- that the payment is valid but its canonical customer had no usable address;
-- it must never be mistaken for a send failure or an unrecorded payment.
--
-- Rollback evidence:
--   Revert application code first and retain this additive evidence table.
--   A separately authorized destructive teardown, only after no deployed
--   reader/writer remains, may run:
--       DROP TABLE payment_receipt_deliveries;
--   Never use that as an ordinary rollback: it destroys payment-receipt audit
--   evidence and breaks a mixed-version deployment that reads delivery status.
--
-- Roll-forward safety:
--   Existing payment, allocation, deposit, clearing, return, void, and MCP
--   paths do not reference this table until the receipt-aware EOM release is
--   deployed.  The table is purely additive and has no trigger on legacy
--   financial rows.

CREATE TABLE IF NOT EXISTS payment_receipt_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id UUID NOT NULL REFERENCES customer_payments(id) ON DELETE RESTRICT,
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    receipt_number VARCHAR(64) NOT NULL UNIQUE,
    recipient_email VARCHAR(256),
    delivery_status VARCHAR(16) NOT NULL
        CHECK (delivery_status IN ('pending', 'sent', 'failed', 'skipped')),
    skip_reason VARCHAR(32),
    subject VARCHAR(500) NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT payment_receipt_deliveries_payment_id_key UNIQUE (payment_id),
    CONSTRAINT chk_payment_receipt_deliveries_delivery_shape CHECK (
        (
            delivery_status = 'skipped'
            AND recipient_email IS NULL
            AND skip_reason = 'no_email'
        )
        OR
        (
            delivery_status IN ('pending', 'sent', 'failed')
            AND recipient_email IS NOT NULL
            AND skip_reason IS NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_payment_receipt_deliveries_contact_created
    ON payment_receipt_deliveries (contact_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_payment_receipt_deliveries_status_created
    ON payment_receipt_deliveries (delivery_status, created_at ASC);

COMMENT ON TABLE payment_receipt_deliveries IS
    'One durable non-sending residential payment receipt outbox row per customer payment; pending/sent/failed/skipped tracks delivery separately from financial lifecycle.';
