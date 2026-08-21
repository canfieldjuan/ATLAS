-- atlas: atomic-bookkeeping
-- Repair the legacy public-onboarding token relation created before migration
-- 383 supplied the immutable signing and prefill projection. Migration 383 is
-- CREATE TABLE IF NOT EXISTS, so it cannot alter that already-existing table.
--
-- An immutable token snapshot cannot be reconstructed from a legacy row. The
-- repair is therefore deliberately limited to an empty relation; a nonempty
-- incomplete relation fails atomically and requires an operator-approved,
-- record-specific recovery instead of fabricated values.

DO $$
DECLARE
    missing_immutable_columns TEXT[];
BEGIN
    SELECT array_agg(required.column_name ORDER BY required.column_name)
    INTO missing_immutable_columns
    FROM unnest(ARRAY[
        'signing_key_fingerprint',
        'prefill_full_name',
        'prefill_email',
        'prefill_phone',
        'prefill_address',
        'prefill_city',
        'prefill_state',
        'prefill_zip',
        'prefill_customer_type'
    ]) AS required(column_name)
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = 'eom_public_onboarding_tokens'::regclass
          AND attname = required.column_name
          AND NOT attisdropped
    );

    IF missing_immutable_columns IS NOT NULL
       AND EXISTS (SELECT 1 FROM eom_public_onboarding_tokens) THEN
        RAISE EXCEPTION
            'cannot safely repair nonempty eom_public_onboarding_tokens; missing immutable columns: %',
            array_to_string(missing_immutable_columns, ', ');
    END IF;
END;
$$;

ALTER TABLE eom_public_onboarding_tokens
    ADD COLUMN IF NOT EXISTS signing_key_fingerprint VARCHAR(64) NOT NULL
        CHECK (signing_key_fingerprint ~ '^[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS prefill_full_name VARCHAR(256) NOT NULL,
    ADD COLUMN IF NOT EXISTS prefill_email VARCHAR(256),
    ADD COLUMN IF NOT EXISTS prefill_phone VARCHAR(32),
    ADD COLUMN IF NOT EXISTS prefill_address TEXT,
    ADD COLUMN IF NOT EXISTS prefill_city VARCHAR(128),
    ADD COLUMN IF NOT EXISTS prefill_state VARCHAR(64),
    ADD COLUMN IF NOT EXISTS prefill_zip VARCHAR(16),
    ADD COLUMN IF NOT EXISTS prefill_customer_type VARCHAR(32) NOT NULL;
