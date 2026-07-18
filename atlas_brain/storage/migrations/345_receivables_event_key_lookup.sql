-- Support global payment-event idempotency checks added after migration 344.
-- This must remain a separate migration so databases that ran an earlier
-- receivables preview still receive the lookup index.

CREATE INDEX IF NOT EXISTS idx_payment_events_key_lookup
    ON payment_events(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
