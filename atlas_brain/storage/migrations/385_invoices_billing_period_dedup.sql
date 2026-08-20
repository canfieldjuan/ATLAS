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
--   * Historical rows ARE backfilled where the period is mechanically
--     derivable from data each writer already wrote (source_ref for
--     monthly_auto, invoice_number for eom_commercial_billing -- see the
--     backfill below). A row whose format predates that convention, or whose
--     derived (contact_id, period) collides with another row's, is left NULL
--     on purpose and flagged for manual reconciliation rather than guessed
--     at -- see the collision-quarantine block below. Backfilling matters
--     because the risk this migration closes is specifically about OLD
--     periods: nothing bounds how far in the past an admin can request and
--     approve a commercial-billing candidate (see plans/PR-EOM-Recurring-Invoice-Period-Dedup.md,
--     "Root cause" -- approval timing is unbounded), so a pre-deploy legacy
--     invoice for that same old period must be visible to the dedup check,
--     not just invoices created after this migration runs.
--
-- A quarantined collision group (Backfill 2/2) leaves EVERY row in the group
-- billing_period = NULL, which is inert against both the partial index
-- (WHERE billing_period IS NOT NULL) and the app-level pre-check (WHERE
-- billing_period = $2, which NULL never matches) -- so without more, a
-- THIRD invoice for that same contact+period would go unblocked, despite the
-- period being known and stored in the quarantine metadata. Verified
-- directly: inserting a collision pair, then a third row with
-- billing_period set explicitly to the shared period, succeeds cleanly with
-- no constraint violation. invoices_billing_period_reservations (below)
-- closes this: one reservation row per quarantined (contact_id,
-- billing_period) group, checked by both writers' pre-checks alongside the
-- invoices table. This is deliberately NOT a third invoice row or a
-- billing_period value written onto one of the ambiguous historical
-- invoices -- either would silently crown one historical row "the real
-- one," which is exactly the guess this migration's whole backfill design
-- exists to avoid. The tradeoff this accepts: the reservation is enforced by
-- the two writers' pre-checks, not by the partial unique index itself (nothing
-- can index-enforce a slot without a real row claiming it) -- any future
-- writer that creates invoices directly, bypassing both pre-checks, would
-- need to be audited against this table too. This narrower guarantee is
-- disclosed here, not silently absent.
--
-- This migration deliberately is NOT marked atomic-bookkeeping. It builds the
-- recurring unique index with CREATE INDEX CONCURRENTLY so invoice reads/writes
-- are not blocked behind one long ACCESS EXCLUSIVE lock while the historical
-- backfill scans accumulated invoice history. Every statement below is
-- idempotent so the runner can safely retry if a later statement fails before
-- schema_migrations is recorded.
--
-- ROLLBACK: revert application code first (both writers keep working with
-- billing_period simply unpersisted/unchecked) and leave this column and
-- index in place; they are inert once nothing writes billing_period.
-- Destructive teardown (DROP INDEX / DROP COLUMN) is a separately authorized
-- operation, not a normal rollback -- it discards the only queryable period
-- ATLAS has ever recorded for these invoices, including the backfilled
-- history below.

ALTER TABLE invoices
    ADD COLUMN IF NOT EXISTS billing_period VARCHAR(7);

ALTER TABLE invoices
    ADD COLUMN IF NOT EXISTS billing_period_legacy_null BOOLEAN NOT NULL DEFAULT false;

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
            CHECK (billing_period ~ '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$')
            NOT VALID;
    END IF;
END $$;

UPDATE invoices
SET billing_period_legacy_null = true
WHERE billing_period IS NULL
  AND status <> 'void'
  AND source IN ('monthly_auto', 'eom_commercial_billing');

ALTER TABLE invoices
    DROP CONSTRAINT IF EXISTS invoices_recurring_billing_period_required_check;

ALTER TABLE invoices
    ADD CONSTRAINT invoices_recurring_billing_period_required_check
    CHECK (
        source NOT IN ('monthly_auto', 'eom_commercial_billing')
        OR status = 'void'
        OR billing_period IS NOT NULL
        OR billing_period_legacy_null
    )
    NOT VALID;

CREATE TABLE IF NOT EXISTS invoices_billing_period_reservations (
    contact_id      UUID NOT NULL,
    billing_period  VARCHAR(7) NOT NULL,
    reason          VARCHAR(32) NOT NULL DEFAULT 'backfill_collision',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (contact_id, billing_period)
);

COMMENT ON TABLE invoices_billing_period_reservations IS
    'One row per (contact_id, billing_period) whose historical backfill was ambiguous (see invoices.metadata.billing_period_backfill_collision). Checked by both recurring writers'' pre-checks alongside invoices, so a third invoice for an already-quarantined period is refused even though the slot cannot be enforced by idx_invoices_recurring_contact_period_source itself (no row claims it). Rows are never deleted by application code -- resolving a reservation is a manual reconciliation action.';

-- Backfill 1/2: derive billing_period for historical rows from data each
-- writer already persisted, so the transition window (invoices created
-- before this migration vs. candidates approved after it) isn't silently
-- unprotected. monthly_auto's source_ref always ends "_{YYYY-MM}" (stable
-- since the writer's first commit); eom_commercial_billing's invoice_number
-- always has the shape "INV-{YYYY-Mon}-{seq}" (every row of this source
-- postdates that format's introduction). A row that doesn't match either
-- shape (e.g. the pre-2026-05-04 "INV-YYYY-NNNN" format) is left NULL --
-- inert, not guessed at. The month-abbreviation set is enumerated literally
-- (not a shape-only regex) because to_date() raises on an unrecognized
-- abbreviation rather than failing safe -- an unconstrained regex would crash
-- this migration on a single malformed legacy row.
--
-- The candidate/collision decision is captured once into a session-local temp
-- table and both backfill passes read that frozen snapshot. This migration is
-- intentionally non-transactional so CREATE INDEX CONCURRENTLY can run; without
-- a single snapshot here, live invoice edits between the two UPDATE statements
-- could change which rows are considered colliding and leave a known historical
-- duplicate neither backfilled nor reserved.
CREATE TEMP TABLE IF NOT EXISTS invoices_billing_period_backfill_candidates (
    id               UUID PRIMARY KEY,
    contact_id       UUID,
    candidate_period VARCHAR(7),
    is_collision     BOOLEAN NOT NULL
) ON COMMIT PRESERVE ROWS;

TRUNCATE invoices_billing_period_backfill_candidates;

INSERT INTO invoices_billing_period_backfill_candidates (
    id, contact_id, candidate_period, is_collision
)
WITH candidates AS (
    SELECT
        id, contact_id,
        CASE
            WHEN source = 'monthly_auto'
                 AND source_ref ~ '_((000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2]))$'
                THEN substring(source_ref FROM '_((?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(?:0[1-9]|1[0-2]))$')
            WHEN source = 'eom_commercial_billing'
                 AND invoice_number ~ '^INV-(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4,}$'
                THEN to_char(
                         to_date(
                             substring(invoice_number FROM '^INV-((?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))-\d{4,}$'),
                             'YYYY-Mon'
                         ),
                         'YYYY-MM'
                     )
            ELSE NULL
        END AS candidate_period
    FROM invoices
    WHERE billing_period IS NULL
      AND status <> 'void'
      AND source IN ('monthly_auto', 'eom_commercial_billing')
),
collisions AS (
    SELECT contact_id, candidate_period
    FROM candidates
    WHERE candidate_period IS NOT NULL
      AND contact_id IS NOT NULL
    GROUP BY contact_id, candidate_period
    HAVING count(*) > 1
)
SELECT
    c.id,
    c.contact_id,
    c.candidate_period,
    EXISTS (
        SELECT 1 FROM collisions x
        WHERE x.contact_id IS NOT DISTINCT FROM c.contact_id
          AND x.candidate_period = c.candidate_period
    ) AS is_collision
FROM candidates AS c;

-- Excludes any (contact_id, candidate period) pair that would itself collide
-- across rows -- see Backfill 2/2 -- so this statement can never leave data
-- that violates the unique index below. contact_id IS NULL rows (a linked
-- CRM contact later deleted, ON DELETE SET NULL) are backfilled individually
-- without collision-checking against each other: the unique index treats
-- every NULL contact_id as distinct from every other, so two such rows can
-- never violate it regardless of period, and grouping them together in the
-- collision CTE above would falsely quarantine unrelated customers' invoices
-- on SQL's "NULL groups with NULL" GROUP BY semantics, which do not match
-- the index's per-row-distinct NULL semantics.
UPDATE invoices AS inv
SET billing_period = c.candidate_period,
    billing_period_legacy_null = false
FROM invoices_billing_period_backfill_candidates AS c
WHERE inv.id = c.id
  AND c.candidate_period IS NOT NULL
  AND NOT c.is_collision;

-- Backfill 2/2: a duplicate this migration exists to prevent may already
-- have happened historically (one monthly_auto row and one
-- eom_commercial_billing row -- or two of the same source -- already
-- covering the same contact+period). Backfilling both would itself violate
-- the unique index below. Rather than delete one, guess which is "real", or
-- abort the migration, both rows are left billing_period = NULL (inert
-- against the partial index) and stamped with a queryable marker so an
-- operator can find and manually reconcile them after deploy:
--   SELECT id, contact_id, source, invoice_number, metadata
--   FROM invoices WHERE metadata->>'billing_period_backfill_collision' = 'true';
-- Only contact_id IS NOT NULL collisions are stamped here, matching Backfill
-- 1/2's grouping -- a NULL contact_id can never actually violate the index,
-- so there is nothing to quarantine or report for those rows.
UPDATE invoices AS inv
SET metadata = COALESCE(inv.metadata, '{}'::jsonb)
        || jsonb_build_object(
             'billing_period_backfill_collision', true,
             'billing_period_backfill_candidate_period', c.candidate_period
           ),
    billing_period_legacy_null = true
FROM invoices_billing_period_backfill_candidates AS c
WHERE inv.id = c.id
  AND c.candidate_period IS NOT NULL
  AND c.is_collision;

UPDATE invoices AS inv
SET metadata = COALESCE(inv.metadata, '{}'::jsonb)
    || jsonb_build_object('billing_period_legacy_null', true),
    billing_period_legacy_null = true
FROM invoices_billing_period_backfill_candidates AS c
WHERE inv.id = c.id
  AND inv.billing_period IS NULL
  AND c.candidate_period IS NULL
  AND inv.status <> 'void'
  AND inv.source IN ('monthly_auto', 'eom_commercial_billing');

INSERT INTO invoices_billing_period_reservations (contact_id, billing_period)
SELECT DISTINCT contact_id, candidate_period
FROM invoices_billing_period_backfill_candidates
WHERE is_collision
  AND contact_id IS NOT NULL
  AND candidate_period IS NOT NULL
ON CONFLICT (contact_id, billing_period) DO NOTHING;

DROP INDEX CONCURRENTLY IF EXISTS idx_invoices_recurring_contact_period_source;

CREATE UNIQUE INDEX CONCURRENTLY idx_invoices_recurring_contact_period_source
    ON invoices (contact_id, billing_period)
    WHERE billing_period IS NOT NULL
      AND status <> 'void'
      AND source IN ('monthly_auto', 'eom_commercial_billing');

COMMENT ON COLUMN invoices.billing_period IS
    'YYYY-MM covered billing period. Backfilled on historical rows where mechanically derivable from source_ref (monthly_auto) or invoice_number (eom_commercial_billing); NULL where unparseable or where two historical rows collided (see metadata.billing_period_backfill_collision). Source-of-truth for cross-pipeline recurring-invoice dedup; see idx_invoices_recurring_contact_period_source.';

COMMENT ON COLUMN invoices.billing_period_legacy_null IS
    'Database-owned exemption for pre-migration recurring invoice rows whose billing_period remains NULL because the historical period was unparseable or collision-quarantined. New recurring writers do not set this column, so invoices_recurring_billing_period_required_check rejects fresh NULL-period recurring invoices while preserving later edits to explicit legacy exceptions.';
