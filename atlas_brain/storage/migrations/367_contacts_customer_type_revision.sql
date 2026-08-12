-- atlas: atomic-bookkeeping
-- 367: database-owned ordering for contacts.customer_type evidence.
--
-- `updated_at` cannot order customer-type evidence: normal contact writers use
-- both application-host time and database time. A later committed type change
-- can consequently carry an earlier timestamp. The tracker needs a source
-- value whose order Atlas, not a caller clock, owns.
--
-- The sequence is global but the consumer compares it only for one contact.
-- A row lock is acquired before a BEFORE UPDATE trigger runs, so successive
-- customer_type changes for one row consume strictly increasing values. Gaps
-- from rolled-back transactions are harmless; this is an ordering token, not a
-- contiguous event number.
--
-- This migration must be atomic. The initial ALTER TABLE lock is held through
-- the backfill and trigger installation, so an old writer cannot insert or
-- update a contact between them and leave an unversioned type behind.
--
-- ROLLBACK PROCEDURE:
-- A normal runtime rollback is code-first and deliberately leaves this entire
-- additive schema contract in place: customer_type_revision, its positive
-- constraint, trg_stamp_contacts_customer_type_revision,
-- stamp_contacts_customer_type_revision(), and
-- contacts_customer_type_revision_seq. Pre-367 contact writers omit the new
-- column; the trigger fills it for their inserts and preserves it on unrelated
-- updates, so retaining the artifacts neither breaks mixed-version writers nor
-- discards ordering evidence.
--
-- Destructive teardown is NOT an ordinary rollback. Only after every deployed
-- provider and tracker consumer has stopped reading/writing this revision, a
-- separately authorized DBA operation may remove it in this order (in the same
-- schema): DROP TRIGGER trg_stamp_contacts_customer_type_revision ON contacts;
-- DROP FUNCTION stamp_contacts_customer_type_revision(); ALTER SEQUENCE
-- contacts_customer_type_revision_seq OWNED BY NONE; ALTER TABLE contacts DROP
-- COLUMN customer_type_revision; DROP SEQUENCE contacts_customer_type_revision_seq.
-- Dropping the column removes its positive CHECK constraint. Never tear down
-- these artifacts while a revision-aware deployment is still running.

CREATE SEQUENCE IF NOT EXISTS contacts_customer_type_revision_seq;

ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS customer_type_revision BIGINT;

-- A previously interrupted manual application may have left NULL or nonpositive
-- values. Establish a positive baseline before making the column NOT NULL.
UPDATE contacts
   SET customer_type_revision = nextval('contacts_customer_type_revision_seq'::regclass)
 WHERE customer_type_revision IS NULL
    OR customer_type_revision <= 0;

-- If the column was partially installed with manually supplied values, advance
-- the sequence past them before the trigger starts issuing revisions.
SELECT setval(
    'contacts_customer_type_revision_seq'::regclass,
    COALESCE(MAX(customer_type_revision), 1),
    COUNT(*) > 0
)
FROM contacts;

ALTER TABLE contacts
    ALTER COLUMN customer_type_revision SET NOT NULL;

ALTER SEQUENCE contacts_customer_type_revision_seq
    OWNED BY contacts.customer_type_revision;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_contacts_customer_type_revision_positive'
          AND conrelid = 'contacts'::regclass
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT chk_contacts_customer_type_revision_positive
            CHECK (customer_type_revision > 0);
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION stamp_contacts_customer_type_revision()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
BEGIN
    -- Inserts establish the first type evidence, including the deliberate
    -- `unknown` default. A true type change establishes newer evidence. The
    -- trigger overwrites a supplied value, so callers cannot forge an order.
    IF TG_OP = 'INSERT'
       OR NEW.customer_type IS DISTINCT FROM OLD.customer_type THEN
        NEW.customer_type_revision := nextval(
            format(
                '%I.%I', TG_TABLE_SCHEMA, 'contacts_customer_type_revision_seq'
            )::regclass
        );
    ELSIF NEW.customer_type_revision IS DISTINCT FROM OLD.customer_type_revision THEN
        RAISE EXCEPTION 'contacts.customer_type_revision is database-owned';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_stamp_contacts_customer_type_revision ON contacts;
CREATE TRIGGER trg_stamp_contacts_customer_type_revision
BEFORE INSERT OR UPDATE ON contacts
FOR EACH ROW
EXECUTE FUNCTION stamp_contacts_customer_type_revision();

COMMENT ON COLUMN contacts.customer_type_revision IS
    'Database-owned monotonic order for this contact customer_type evidence.';
