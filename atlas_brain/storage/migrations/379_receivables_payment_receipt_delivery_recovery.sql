-- atlas: atomic-bookkeeping
-- Recover receipt-delivery schemas where migration 378 was recorded before
-- its later result-shape and reconciliation-evidence DDL was added.
--
-- This migration is intentionally additive. It repairs only schema objects
-- introduced after the original 378 revision and leaves financial, receipt,
-- operation, and audit facts untouched.

ALTER TABLE payment_receipt_delivery_operations
    ADD COLUMN IF NOT EXISTS result_delivery_status VARCHAR(16),
    ADD COLUMN IF NOT EXISTS result_sent_at TIMESTAMPTZ;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'payment_receipt_delivery_operations'::regclass
          AND conname = 'payment_receipt_delivery_operations_result_shape_check'
    ) THEN
        ALTER TABLE payment_receipt_delivery_operations
            ADD CONSTRAINT payment_receipt_delivery_operations_result_shape_check
            CHECK (
                (
                    state <> 'completed'
                    AND result_delivery_status IS NULL
                    AND result_sent_at IS NULL
                )
                OR
                (
                    state = 'completed'
                    AND (
                        (
                            outcome IN ('sent', 'already_sent')
                            AND result_delivery_status = 'sent'
                            AND result_sent_at IS NOT NULL
                        )
                        OR
                        (
                            outcome = 'failed'
                            AND result_delivery_status = 'failed'
                            AND result_sent_at IS NULL
                        )
                    )
                )
            ) NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS payment_receipt_delivery_reconciliation_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_delivery_id UUID NOT NULL
        REFERENCES payment_receipt_deliveries(id) ON DELETE RESTRICT,
    operation_id UUID NOT NULL
        REFERENCES payment_receipt_delivery_operations(id) ON DELETE RESTRICT,
    actor VARCHAR(128) NOT NULL,
    outcome VARCHAR(32) NOT NULL,
    reconciled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT payment_receipt_delivery_reconciliation_events_actor_check
        CHECK (length(btrim(actor)) > 0),
    CONSTRAINT payment_receipt_delivery_reconciliation_events_outcome_check
        CHECK (outcome IN ('sent', 'recovery_required'))
);

CREATE INDEX IF NOT EXISTS idx_payment_receipt_delivery_reconciliation_events_record
    ON payment_receipt_delivery_reconciliation_events (
        receipt_delivery_id, reconciled_at DESC, id DESC
    );

COMMENT ON TABLE payment_receipt_delivery_reconciliation_events IS
    'Append-only actor/timestamp evidence for no-send Gmail Sent-mail reconciliation outcomes.';
