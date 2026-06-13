-- Make the deflection reconciliation dedup NULL-safe (#1462 follow-up).
--
-- Migration 336 created content_ops_deflection_paid_reconciliation with a
-- nullable stripe_session_id and UNIQUE (account_id, request_id,
-- stripe_session_id). Postgres treats NULL as DISTINCT in a UNIQUE constraint,
-- so the ON CONFLICT (account_id, request_id, stripe_session_id) DO NOTHING
-- that record_paid_report_missing relies on does NOT dedup rows whose
-- stripe_session_id is NULL. The billing webhook forwarded `session_id or None`,
-- so a missing/empty Stripe session id became NULL and every retry (Stripe is
-- at-least-once) or sibling event type wrote a duplicate ledger row -- breaking
-- the idempotency the function promises.
--
-- This collapses any existing NULL-equivalent duplicates, backfills NULL -> '',
-- and forbids NULL going forward, so the existing UNIQUE dedups uniformly (the
-- empty string is a normal, conflict-eligible value). Re-runnable.

-- 1. Collapse rows that share (account_id, request_id, COALESCE(stripe_session_id, ''))
--    down to the lowest id. The 336 UNIQUE already prevents non-NULL duplicates,
--    so this only removes the NULL-equivalent duplicates the gap allowed.
DELETE FROM content_ops_deflection_paid_reconciliation a
USING content_ops_deflection_paid_reconciliation b
WHERE a.account_id = b.account_id
  AND a.request_id = b.request_id
  AND COALESCE(a.stripe_session_id, '') = COALESCE(b.stripe_session_id, '')
  AND a.id > b.id;

-- 2. Backfill remaining NULLs to '' (safe after the dedup above: at most one
--    COALESCE='' row per (account_id, request_id) remains).
UPDATE content_ops_deflection_paid_reconciliation
SET stripe_session_id = ''
WHERE stripe_session_id IS NULL;

-- 3. Forbid NULL going forward; default to '' so the existing UNIQUE dedups it.
ALTER TABLE content_ops_deflection_paid_reconciliation
    ALTER COLUMN stripe_session_id SET DEFAULT '';
ALTER TABLE content_ops_deflection_paid_reconciliation
    ALTER COLUMN stripe_session_id SET NOT NULL;
