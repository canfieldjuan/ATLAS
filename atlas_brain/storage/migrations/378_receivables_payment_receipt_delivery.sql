-- Durable, explicit Gmail delivery/recovery evidence for residential payment
-- receipt outbox rows.
--
-- Migration 369 commits only a financial-transaction-adjacent receipt snapshot.
-- This additive migration records a stable mailbox identity and each authenticated
-- dispatch request separately.  The row is not a payment mutation: a send
-- failure, recovery requirement, or later retry never changes payment amounts,
-- allocations, deposits, clearing, returns, voids, or invoices.
--
-- A dispatch operation moves through prepared -> attempting before the external
-- Gmail request.  If the process dies after attempting, no later caller may send
-- again: it must reconcile the durable RFC Message-ID against Gmail Sent mail.
-- That conservative false-negative is preferable to duplicate customer mail.
--
-- Rollback: stop the dispatch route/service and retain the additive identity and
-- operation evidence.  Do not drop these columns or operation rows during a
-- mixed-version rollback; deleting receipt-delivery history is a separately
-- authorized destructive retention action.

ALTER TABLE payment_receipt_deliveries
    ADD COLUMN IF NOT EXISTS rfc_message_id VARCHAR(320),
    ADD COLUMN IF NOT EXISTS gmail_message_id VARCHAR(256),
    ADD COLUMN IF NOT EXISTS gmail_thread_id VARCHAR(256),
    ADD COLUMN IF NOT EXISTS sent_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_attempt_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_failure_code VARCHAR(64),
    ADD COLUMN IF NOT EXISTS last_failure_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS recovery_required_at TIMESTAMPTZ;

-- UUID receipt-delivery ids are already unique.  Derive a deterministic
-- Message-ID for every historical row before enforcing the non-null invariant.
-- Keep a server-side default as well: mixed-version receipt writers from the
-- already-deployed outbox release do not name this new column, and they must
-- remain able to commit their financial transaction while this provider rolls
-- forward.
UPDATE payment_receipt_deliveries
SET rfc_message_id =
    '<atlas-eom-payment-receipt-' || id::text || '@effinghamofficemaids.com>'
WHERE rfc_message_id IS NULL;

ALTER TABLE payment_receipt_deliveries
    ALTER COLUMN rfc_message_id SET DEFAULT
        ('<atlas-eom-payment-receipt-' || gen_random_uuid()::text
         || '@effinghamofficemaids.com>'),
    ALTER COLUMN rfc_message_id SET NOT NULL;

-- A failed concurrent build leaves an invalid same-named catalog object.  The
-- migration ledger is not recorded until every later statement succeeds, so a
-- retry must remove either an invalid *or* previously-valid partial-run index
-- before rebuilding it.  ``IF NOT EXISTS`` would otherwise silently retain an
-- invalid index and let readiness claim the uniqueness guarantee exists.
DROP INDEX CONCURRENTLY IF EXISTS
    idx_payment_receipt_deliveries_rfc_message_id;

CREATE UNIQUE INDEX CONCURRENTLY
    idx_payment_receipt_deliveries_rfc_message_id
    ON payment_receipt_deliveries (rfc_message_id);

-- The migration runner records this migration only after every statement
-- succeeds.  These guards make a rollback/retry after a later statement fails
-- safe: PostgreSQL otherwise has no ADD CONSTRAINT IF NOT EXISTS syntax.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'payment_receipt_deliveries'::regclass
          AND conname = 'payment_receipt_deliveries_rfc_message_id_check'
    ) THEN
        ALTER TABLE payment_receipt_deliveries
            ADD CONSTRAINT payment_receipt_deliveries_rfc_message_id_check
            CHECK (
                rfc_message_id ~ '^<[^[:space:]<>]+@[^[:space:]<>]+>$'
                AND length(btrim(rfc_message_id)) = length(rfc_message_id)
            ) NOT VALID;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'payment_receipt_deliveries'::regclass
          AND conname = 'payment_receipt_deliveries_sent_identity_check'
    ) THEN
        ALTER TABLE payment_receipt_deliveries
            ADD CONSTRAINT payment_receipt_deliveries_sent_identity_check
            CHECK (
                (
                    delivery_status = 'sent'
                    AND length(btrim(COALESCE(gmail_message_id, ''))) > 0
                    AND length(btrim(COALESCE(gmail_thread_id, ''))) > 0
                    AND sent_at IS NOT NULL
                )
                OR
                (
                    delivery_status <> 'sent'
                    AND gmail_message_id IS NULL
                    AND gmail_thread_id IS NULL
                    AND sent_at IS NULL
                )
            ) NOT VALID;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'payment_receipt_deliveries'::regclass
          AND conname = 'payment_receipt_deliveries_last_attempt_check'
    ) THEN
        ALTER TABLE payment_receipt_deliveries
            ADD CONSTRAINT payment_receipt_deliveries_last_attempt_check
            CHECK (
                (last_attempt_by IS NULL AND last_attempt_at IS NULL)
                OR (
                    length(btrim(COALESCE(last_attempt_by, ''))) > 0
                    AND last_attempt_at IS NOT NULL
                )
            ) NOT VALID;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'payment_receipt_deliveries'::regclass
          AND conname = 'payment_receipt_deliveries_last_failure_check'
    ) THEN
        ALTER TABLE payment_receipt_deliveries
            ADD CONSTRAINT payment_receipt_deliveries_last_failure_check
            CHECK (
                (last_failure_code IS NULL AND last_failure_at IS NULL)
                OR (
                    length(btrim(COALESCE(last_failure_code, ''))) > 0
                    AND last_failure_at IS NOT NULL
                )
            ) NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS payment_receipt_delivery_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_delivery_id UUID NOT NULL
        REFERENCES payment_receipt_deliveries(id) ON DELETE RESTRICT,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    state VARCHAR(32) NOT NULL DEFAULT 'prepared',
    outcome VARCHAR(32),
    requested_by VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attempt_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    recovery_required_at TIMESTAMPTZ,
    result_delivery_status VARCHAR(16),
    result_sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT payment_receipt_delivery_operations_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT payment_receipt_delivery_operations_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT payment_receipt_delivery_operations_actor_check
        CHECK (length(btrim(requested_by)) > 0),
    CONSTRAINT payment_receipt_delivery_operations_state_check
        CHECK (state IN ('prepared', 'attempting', 'completed', 'recovery_required')),
    CONSTRAINT payment_receipt_delivery_operations_shape_check
        CHECK (
            (
                state = 'prepared'
                AND outcome IS NULL
                AND attempt_started_at IS NULL
                AND completed_at IS NULL
                AND recovery_required_at IS NULL
            )
            OR
            (
                state = 'attempting'
                AND outcome IS NULL
                AND attempt_started_at IS NOT NULL
                AND completed_at IS NULL
                AND recovery_required_at IS NULL
            )
            OR
            (
                state = 'completed'
                AND outcome IN ('sent', 'failed', 'already_sent')
                AND completed_at IS NOT NULL
                AND recovery_required_at IS NULL
                AND (
                    outcome = 'already_sent'
                    OR attempt_started_at IS NOT NULL
                )
            )
            OR
            (
                state = 'recovery_required'
                AND outcome IS NULL
                AND attempt_started_at IS NOT NULL
                AND completed_at IS NULL
                AND recovery_required_at IS NOT NULL
            )
        )
);

-- ``CREATE TABLE IF NOT EXISTS`` does not evolve a table that a previously
-- interrupted attempt already created.  Keep these columns additive so the
-- migration runner can safely replay this file before it records migration
-- 378.
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

CREATE INDEX IF NOT EXISTS idx_payment_receipt_delivery_operations_record
    ON payment_receipt_delivery_operations (
        receipt_delivery_id, requested_at DESC
    );

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_payment_receipt_delivery_operations_one_active
    ON payment_receipt_delivery_operations (receipt_delivery_id)
    WHERE state IN ('prepared', 'attempting', 'recovery_required');

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

COMMENT ON TABLE payment_receipt_delivery_operations IS
    'Idempotent actor-attributed Gmail receipt dispatch operations; a recovery-required operation blocks automatic re-send.';

COMMENT ON TABLE payment_receipt_delivery_reconciliation_events IS
    'Append-only actor/timestamp evidence for no-send Gmail Sent-mail reconciliation outcomes.';

COMMENT ON COLUMN payment_receipt_deliveries.rfc_message_id IS
    'Stable server-owned Gmail lookup identity for the exact residential receipt.';
