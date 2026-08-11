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
--
-- ROLLBACK: revert the CODE, keep the COLUMN. In that order, and permanently.
--
-- The column is additive and nullable-free with a default, so an older build
-- that never mentions customer_type reads and writes contacts unchanged --
-- every INSERT it issues simply takes the default. There is therefore no
-- reason to drop it, and two reasons not to:
--
--  * Dropping it destroys the classification. There is no contact history
--    table, so once the column is gone the residential/commercial decisions --
--    the backfill's, and every one an operator has made since -- exist
--    nowhere and must be rebuilt by hand.
--  * Dropping it while ANY instance still runs this release breaks that
--    instance immediately: the provider's contact INSERT names the column
--    explicitly, so every operator contact create raises undefined_column.
--
-- If the column genuinely must go (it should not), take the code out of
-- service first, then `ALTER TABLE contacts DROP CONSTRAINT
-- chk_contacts_customer_type, DROP COLUMN customer_type;` -- never the reverse
-- order. Re-running this migration afterwards restores the column but NOT the
-- classifications; those are only recoverable by re-running the backfill,
-- which reconstructs the tracker-derived values and nothing else.
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
