-- 400: bill-to identity on the contact (eom-timetracker bill-to identity
-- contract, slice A).
--
-- `contacts.address` already means two things that must not drift apart from
-- the service location: it is the `location` of EOM booking calendar events and
-- the monthly invoice bill-to. A commercial Customer billed at an office that is
-- not where the cleaners go has nowhere to put that office. `billing_address`
-- is the bill-to address only; invoices prefer it and fall back to `address`,
-- and bookings keep reading `address`.
--
-- `bill_to_revision` fences the tracker's identity writes. The tracker sends a
-- revision from one monotonic sequence with every bill-to identity edit; the
-- operator mutation applies the edit only when that revision is greater than
-- the stored one, in the same transaction and under the same per-contact lock
-- as the write. A retried stale edit therefore cannot undo a newer one. NULL
-- means no fenced identity write has been applied yet.
--
-- Both columns are nullable with no default, so existing rows and every writer
-- that never names them are unaffected. The CHECKs are the enforcement; the
-- operator boundary validates the same bounds so a bad value is a 422, not a
-- constraint error.
--
-- ROLLBACK: revert the code, keep the columns. An older build never names them,
-- so they are inert. Dropping `billing_address` destroys bill-to addresses that
-- exist nowhere else in Atlas; drop it only after every reader is gone, with
-- `ALTER TABLE contacts DROP COLUMN billing_address, DROP COLUMN bill_to_revision;`
-- (dropping a column drops its CHECK).
ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS billing_address TEXT;

ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS bill_to_revision BIGINT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_contacts_billing_address_length'
          AND conrelid = 'contacts'::regclass
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT chk_contacts_billing_address_length
            CHECK (billing_address IS NULL OR char_length(billing_address) <= 500);
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_contacts_bill_to_revision_positive'
          AND conrelid = 'contacts'::regclass
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT chk_contacts_bill_to_revision_positive
            CHECK (bill_to_revision IS NULL OR bill_to_revision > 0);
    END IF;
END $$;
