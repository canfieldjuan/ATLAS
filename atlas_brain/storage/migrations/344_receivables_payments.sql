-- First-class customer receipts, invoice allocations, and deposit lifecycle.
-- Existing invoice_payments rows remain the compatibility allocation surface.

CREATE TABLE IF NOT EXISTS customer_payments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id          UUID REFERENCES contacts(id) ON DELETE SET NULL,
    payer_name          VARCHAR(256) NOT NULL,
    total_amount        NUMERIC(12,2) NOT NULL,
    payment_method      VARCHAR(32) NOT NULL,
    reference           VARCHAR(256),
    received_date       DATE NOT NULL,
    status              VARCHAR(16) NOT NULL,
    source              VARCHAR(32) NOT NULL DEFAULT 'manual',
    idempotency_key     VARCHAR(128),
    request_fingerprint VARCHAR(64),
    notes               TEXT,
    recorded_by         VARCHAR(128),
    deposited_at        TIMESTAMPTZ,
    cleared_at          TIMESTAMPTZ,
    returned_at         TIMESTAMPTZ,
    return_reason       TEXT,
    voided_at           TIMESTAMPTZ,
    void_reason         TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT customer_payments_status_check
        CHECK (status IN ('legacy', 'received', 'deposited', 'cleared', 'returned', 'voided')),
    CONSTRAINT customer_payments_positive_new_amount_check
        CHECK (source = 'legacy' OR total_amount > 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_customer_payments_idempotency
    ON customer_payments(source, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_customer_payments_contact_received
    ON customer_payments(contact_id, received_date DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_customer_payments_status_received
    ON customer_payments(status, received_date DESC);

ALTER TABLE invoice_payments
    ADD COLUMN IF NOT EXISTS payment_id UUID,
    ADD COLUMN IF NOT EXISTS reversed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reversed_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS reversal_reason TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'invoice_payments_payment_id_fkey'
          AND conrelid = 'invoice_payments'::regclass
    ) THEN
        ALTER TABLE invoice_payments
            ADD CONSTRAINT invoice_payments_payment_id_fkey
            FOREIGN KEY (payment_id) REFERENCES customer_payments(id) ON DELETE RESTRICT;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_invoice_payments_payment_id
    ON invoice_payments(payment_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_payments_active_payment_invoice
    ON invoice_payments(payment_id, invoice_id)
    WHERE payment_id IS NOT NULL AND reversed_at IS NULL;

-- Preserve history conservatively: one parent per existing allocation. Matching
-- references are not proof that two rows represent the same physical receipt.
INSERT INTO customer_payments (
    id, contact_id, payer_name, total_amount, payment_method, reference,
    received_date, status, source, notes, recorded_by, metadata,
    created_at, updated_at
)
SELECT
    ip.id,
    i.contact_id,
    i.customer_name,
    ip.amount,
    ip.payment_method,
    ip.reference,
    ip.payment_date,
    'legacy',
    'legacy',
    ip.notes,
    ip.recorded_by,
    COALESCE(ip.metadata, '{}'::jsonb),
    ip.created_at,
    ip.created_at
FROM invoice_payments ip
JOIN invoices i ON i.id = ip.invoice_id
WHERE ip.payment_id IS NULL
ON CONFLICT (id) DO NOTHING;

UPDATE invoice_payments ip
SET payment_id = ip.id
WHERE ip.payment_id IS NULL
  AND EXISTS (SELECT 1 FROM customer_payments cp WHERE cp.id = ip.id);

CREATE TABLE IF NOT EXISTS payment_deposit_batches (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deposit_date        DATE NOT NULL,
    bank_reference      VARCHAR(256),
    status              VARCHAR(16) NOT NULL DEFAULT 'deposited',
    idempotency_key     VARCHAR(128),
    request_fingerprint VARCHAR(64),
    clear_idempotency_key VARCHAR(128),
    clear_request_fingerprint VARCHAR(64),
    created_by          VARCHAR(128),
    cleared_at          TIMESTAMPTZ,
    cleared_by          VARCHAR(128),
    voided_at           TIMESTAMPTZ,
    voided_by           VARCHAR(128),
    void_reason         TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT payment_deposit_batches_status_check
        CHECK (status IN ('deposited', 'cleared', 'voided'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_payment_deposit_batches_idempotency
    ON payment_deposit_batches(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_payment_deposit_batches_clear_idempotency
    ON payment_deposit_batches(clear_idempotency_key)
    WHERE clear_idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_payment_deposit_batches_status_date
    ON payment_deposit_batches(status, deposit_date DESC);

CREATE TABLE IF NOT EXISTS payment_deposit_items (
    batch_id    UUID NOT NULL REFERENCES payment_deposit_batches(id) ON DELETE RESTRICT,
    payment_id  UUID NOT NULL REFERENCES customer_payments(id) ON DELETE RESTRICT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (batch_id, payment_id),
    UNIQUE (payment_id)
);

CREATE TABLE IF NOT EXISTS payment_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id          UUID NOT NULL REFERENCES customer_payments(id) ON DELETE RESTRICT,
    event_type          VARCHAR(32) NOT NULL,
    previous_status     VARCHAR(16),
    new_status          VARCHAR(16),
    effective_date      DATE,
    actor               VARCHAR(128),
    reason              TEXT,
    idempotency_key     VARCHAR(128),
    request_fingerprint VARCHAR(64),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_payment_events_idempotency
    ON payment_events(payment_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_payment_events_payment_created
    ON payment_events(payment_id, created_at DESC);

-- If an older process writes the legacy invoice_payments shape before the
-- deployment drain completes, adopt that row in the same transaction so it is
-- immediately visible in global payment history. This is a visibility bridge,
-- not permission to overlap old and new finance writers: the old process's
-- separate balance refresh is not lock-safe. A late check is operationally
-- "received" and can enter a deposit batch; other old methods remain
-- conservatively classified as legacy.
CREATE OR REPLACE FUNCTION adopt_legacy_invoice_payment()
RETURNS TRIGGER AS $$
DECLARE
    invoice_row RECORD;
    adopted_at TIMESTAMPTZ;
BEGIN
    IF NEW.payment_id IS NOT NULL THEN
        RETURN NEW;
    END IF;

    SELECT contact_id, customer_name
    INTO invoice_row
    FROM invoices
    WHERE id = NEW.invoice_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Invoice % not found for legacy payment adoption', NEW.invoice_id;
    END IF;

    adopted_at := COALESCE(NEW.created_at, CURRENT_TIMESTAMP);
    INSERT INTO customer_payments (
        id, contact_id, payer_name, total_amount, payment_method, reference,
        received_date, status, source, notes, recorded_by, metadata,
        created_at, updated_at
    ) VALUES (
        NEW.id,
        invoice_row.contact_id,
        invoice_row.customer_name,
        NEW.amount,
        COALESCE(NULLIF(lower(btrim(NEW.payment_method)), ''), 'other'),
        NEW.reference,
        NEW.payment_date,
        CASE
            WHEN NEW.amount > 0 AND lower(btrim(NEW.payment_method)) = 'check'
            THEN 'received'
            ELSE 'legacy'
        END,
        'legacy',
        NEW.notes,
        NEW.recorded_by,
        COALESCE(NEW.metadata, '{}'::jsonb)
            || jsonb_build_object('adopted_rolling_writer', true),
        adopted_at,
        adopted_at
    )
    ON CONFLICT (id) DO NOTHING;

    NEW.payment_id := NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_adopt_legacy_invoice_payment ON invoice_payments;
CREATE TRIGGER trg_adopt_legacy_invoice_payment
BEFORE INSERT ON invoice_payments
FOR EACH ROW
WHEN (NEW.payment_id IS NULL)
EXECUTE FUNCTION adopt_legacy_invoice_payment();
