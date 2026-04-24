-- Phase 1b: persist normalized-grounding result on witnesses so the API can
-- gate quote-grade rendering without recomputing on every request.
--
-- grounding_status is a tri-state enum:
--   'pending'      = never checked (legacy rows; backfilled by
--                    scripts/backfill_witness_grounding.py)
--   'grounded'     = check ran and the excerpt appears in source text
--                    after normalization -- quote-grade eligible
--   'not_grounded' = check ran and the excerpt does not appear -- signal-
--                    grade only; UI must NOT render it as a quote
--
-- grounding_checked_at is NULL iff grounding_status = 'pending'. After the
-- batch-populate backfill completes, no row should remain in 'pending'.
-- The release sign-off audit query is:
--   SELECT grounding_status, count(*) FROM b2b_vendor_witnesses GROUP BY 1;
--
-- Phase 1b step 6 will update the witness write path so newly-written
-- witnesses populate both fields at write time and do not carry 'pending'
-- on fresh rows. Until that lands, new rows default to 'pending'.

ALTER TABLE b2b_vendor_witnesses
    ADD COLUMN IF NOT EXISTS grounding_status text NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS grounding_checked_at timestamptz NULL;

-- Enforce the enum at the DB level so a typo can never land in this column.
-- DROP first so re-running is idempotent.
ALTER TABLE b2b_vendor_witnesses
    DROP CONSTRAINT IF EXISTS b2b_vendor_witnesses_grounding_status_chk;
ALTER TABLE b2b_vendor_witnesses
    ADD CONSTRAINT b2b_vendor_witnesses_grounding_status_chk
    CHECK (grounding_status IN ('pending', 'grounded', 'not_grounded'));

-- Partial index: the backfill scans for pending rows in a tight loop and
-- the audit query expects fast counts. Once backfill drains all pending
-- rows the index becomes very small (only newly-inserted-but-untouched
-- rows would land there, which by the write-path contract should be zero).
CREATE INDEX IF NOT EXISTS idx_b2b_vendor_witnesses_grounding_pending
    ON b2b_vendor_witnesses (witness_id)
    WHERE grounding_status = 'pending';
