-- Optional operational details recorded with a customer payment.
-- Nullable additions preserve every legacy and already-recorded payment.
ALTER TABLE customer_payments
    ADD COLUMN IF NOT EXISTS check_date DATE,
    ADD COLUMN IF NOT EXISTS received_through VARCHAR(128);
