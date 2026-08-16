-- atlas: atomic-bookkeeping
-- Recover receipt-delivery schemas where migration 378 was recorded before
-- its later result-shape and reconciliation-evidence DDL was added.
--
-- This migration is intentionally additive. It repairs only schema objects
-- introduced after the original 378 revision, preserves financial, receipt,
-- and original operation-lifecycle facts, and backfills only the two later
-- immutable result-projection columns where those recorded facts prove them.

ALTER TABLE payment_receipt_delivery_operations
    ADD COLUMN IF NOT EXISTS result_delivery_status VARCHAR(16),
    ADD COLUMN IF NOT EXISTS result_sent_at TIMESTAMPTZ;

-- Completed operations created by the original 378 revision predate the
-- immutable replay-result columns.  A failed outcome proves its result without
-- consulting mutable delivery state.  A sent/already-sent outcome is safe to
-- backfill only when the linked delivery still carries its terminal sent proof.
-- Any contradictory legacy row remains incomplete so semantic readiness stays
-- fail-closed rather than inventing a historical receipt result.
UPDATE payment_receipt_delivery_operations AS operation
SET result_delivery_status = CASE
        WHEN operation.outcome IN ('sent', 'already_sent') THEN 'sent'
        WHEN operation.outcome = 'failed' THEN 'failed'
    END,
    result_sent_at = CASE
        WHEN operation.outcome IN ('sent', 'already_sent') THEN delivery.sent_at
        ELSE NULL
    END
FROM payment_receipt_deliveries AS delivery
WHERE operation.receipt_delivery_id = delivery.id
  AND operation.state = 'completed'
  AND operation.result_delivery_status IS NULL
  AND operation.result_sent_at IS NULL
  AND (
      operation.outcome = 'failed'
      OR (
          operation.outcome IN ('sent', 'already_sent')
          AND delivery.delivery_status = 'sent'
          AND delivery.sent_at IS NOT NULL
      )
  );

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
