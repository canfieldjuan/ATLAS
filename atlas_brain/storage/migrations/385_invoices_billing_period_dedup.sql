-- 385: cross-pipeline recurring-invoice dedup for commercial customers.
--
-- Two independent writers auto-create commercial-customer invoices: the legacy
-- monthly cron (atlas_brain/autonomous/tasks/monthly_invoice_generation.py,
-- source='monthly_auto') and the newer commercial-billing approval writer
-- (atlas_brain/services/commercial_billing_approvals.py,
-- source='eom_commercial_billing'). Each already bundles a contact's services
-- into one invoice per contact per period and dedups only against its OWN
-- prior invoices; neither is aware of the other. A customer can be
-- auto-invoiced by both for the same month, producing two real invoices.
-- Documented in ATLAS #2363 and explicitly deferred by H-21 (#2439), H-22
-- (#2441), and H-23 (#2445) as "its own evidence-backed financial slice" --
-- this migration is that slice.
--
-- Root cause: invoices carries no queryable covered-period column -- only
-- issue_date/due_date, which are creation/approval timestamps that drift
-- independently between the two writers (the legacy task issues on the 1st of
-- the month after the covered period; the approval writer issues whenever an
-- admin clicks approve, unbounded) and so cannot be used as a same-period
-- proxy. Both writers already compute the covered period in memory
-- (period_label / _InvoiceDraft.billing_period, both plain "YYYY-MM") but
-- only use it to format the invoice_number string. This migration gives that
-- value a real, persisted, queryable home and a database-enforced guarantee
-- that the two recurring sources cannot both claim the same contact+period.
--
-- The unique index deliberately does NOT include `source` in its column
-- list -- only in the WHERE predicate. A monthly_auto row and an
-- eom_commercial_billing row for the same (contact_id, billing_period) must
-- collide on the SAME index key; putting `source` in the column list would
-- give them different keys and let both insert cleanly, which is the exact
-- failure mode this migration exists to close.
--
-- Deliberately scoped so every other invoice-creation path is untouched:
--   * source='mcp_tool' (atlas_brain/mcp/invoicing_server.py create_invoice)
--     never sets billing_period and is outside the source allowlist below --
--     an ad-hoc same-month invoice (e.g. a damage fee) never collides with a
--     recurring invoice.
--   * status='void' invoices are excluded, so voiding and re-issuing a
--     recurring invoice for the same contact+period remains possible.
--   * Historical rows are NOT backfilled and stay NULL; the partial index's
--     `billing_period IS NOT NULL` clause makes every existing row inert.
--
-- ROLLBACK: revert application code first (both writers keep working with
-- billing_period simply unpersisted/unchecked) and leave this column and
-- index in place; they are inert once nothing writes billing_period.
-- Destructive teardown (DROP INDEX / DROP COLUMN) is a separately authorized
-- operation, not a normal rollback -- it discards the only queryable period
-- ATLAS has ever recorded for these invoices.

ALTER TABLE invoices
    ADD COLUMN IF NOT EXISTS billing_period VARCHAR(7);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'invoices_billing_period_check'
          AND conrelid = 'invoices'::regclass
    ) THEN
        ALTER TABLE invoices
            ADD CONSTRAINT invoices_billing_period_check
            CHECK (billing_period ~ '^[0-9]{4}-(0[1-9]|1[0-2])$');
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoices_recurring_contact_period_source
    ON invoices (contact_id, billing_period)
    WHERE billing_period IS NOT NULL
      AND status <> 'void'
      AND source IN ('monthly_auto', 'eom_commercial_billing');

COMMENT ON COLUMN invoices.billing_period IS
    'YYYY-MM covered billing period, populated going forward only (historical rows are NULL). Source-of-truth for cross-pipeline recurring-invoice dedup; see idx_invoices_recurring_contact_period_source.';
