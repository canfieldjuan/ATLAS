-- Explicit, canonical EOM billing-delivery policy.
--
-- A preference is customer-profile evidence, not a financial ledger and not a
-- delivery action. Existing customers intentionally receive no default row:
-- candidate generation must keep them blocked until an authenticated operator
-- chooses a policy. This migration creates no invoice, PDF, Gmail draft,
-- email, Square record, service marker, or payment mutation.
--
-- Rollback evidence:
--   Revert application code first and retain these audited profile rows.
--   A separately authorized destructive teardown, only after every reader and
--   writer is removed, may run:
--       DROP TABLE eom_billing_delivery_preferences;
--   That is not an ordinary rollback because it destroys operator policy
--   evidence and breaks a mixed-version deployment.

CREATE TABLE IF NOT EXISTS eom_billing_delivery_preferences (
    contact_id UUID PRIMARY KEY
        REFERENCES contacts(id) ON DELETE RESTRICT,
    delivery_method VARCHAR(64) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(128) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT eom_billing_delivery_preferences_method_check
        CHECK (
            delivery_method IN (
                'gmail_pdf',
                'manual_square',
                'no_invoice_residential_receipt'
            )
        ),
    CONSTRAINT eom_billing_delivery_preferences_created_by_check
        CHECK (length(btrim(created_by)) > 0),
    CONSTRAINT eom_billing_delivery_preferences_updated_by_check
        CHECK (length(btrim(updated_by)) > 0)
);

COMMENT ON TABLE eom_billing_delivery_preferences IS
    'Explicit EOM customer billing-delivery policy. Not invoice, payment, or delivery state.';
COMMENT ON COLUMN eom_billing_delivery_preferences.delivery_method IS
    'Closed policy: gmail_pdf, manual_square, or no_invoice_residential_receipt.';
