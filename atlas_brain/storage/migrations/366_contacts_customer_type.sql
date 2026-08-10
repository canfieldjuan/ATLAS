-- 366: customer-level residential/commercial type (Slice 1 / Req A, website #174).
--
-- Type exists today only at the SITE level (`locations.location_type` in the
-- tracker). The record that IS the account -- an Atlas contact, in a flat model
-- with no account/person/site hierarchy -- carries no type at all, so every
-- consumer has to infer it from site rows, from a name, or from memory. Billing
-- shape is a CONSEQUENCE of customer type, so with no customer-level type there
-- is nothing stopping a residential customer carrying commercial billing data
-- and nothing that detects it.
--
-- Deliberately a separate axis from `contact_type` (lead|customer), which
-- answers a different question and must not be overloaded to carry this one.
--
-- `unknown` is a real member of the set, not a NULL substitute. A contact whose
-- type has never been established is a distinct, honest state -- and the ~650
-- calendar_import / email_backfill rows are not accounts at all, so `unknown`
-- is the correct answer for them rather than a guess. NOT NULL + DEFAULT keeps
-- the tri-state in one column instead of splitting it across a nullable value.
--
-- The CHECK is the point of this migration. Application validation lives in the
-- operator mutation boundary, but the boundary is code that can be bypassed by
-- a future writer; the constraint cannot. An out-of-set value must be rejected
-- by Postgres itself.
--
-- No DO-block atomicity needed here (unlike migration 365's seed+FK pair): the
-- column arrives with a conforming DEFAULT, and no writer sets it until the
-- boundary change ships, so no row can exist that the CHECK would reject in the
-- window between the two statements. If one somehow did, ADD CONSTRAINT
-- validates existing rows and would abort loudly rather than admit it.
ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS customer_type VARCHAR(16) NOT NULL DEFAULT 'unknown';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_contacts_customer_type'
          AND conrelid = 'contacts'::regclass
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT chk_contacts_customer_type
            CHECK (customer_type IN ('residential', 'commercial', 'unknown'));
    END IF;
END $$;
