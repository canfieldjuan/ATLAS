-- atlas: atomic-bookkeeping
-- Recover the initial recorded migration-385 catalog without rewriting its
-- historical schema_migrations row. The first revision added only a nullable
-- billing_period plus the recurring partial index. The later #2448 revision
-- added the legacy-null marker, reservation table, admission constraint, and
-- historical backfill that the enabled writer-path readiness fence requires.
--
-- This recovery is deliberately atomic: it reuses the already-valid unique
-- index observed in the recorded initial state, so no CONCURRENTLY DDL is
-- needed. If preflight finds an unrecognized index or an invalid stored period,
-- all recovery DDL/DML and this migration's ledger row roll back together.
--
-- ROLLBACK: the retained pre-#2448 writer binaries do not persist
-- billing_period. If an operator must return to those writers after 387,
-- first stop the current writer and run:
--   ALTER TABLE invoices DROP CONSTRAINT IF EXISTS
--       invoices_recurring_billing_period_required_check;
-- Then deploy the retained runtime. The partial index remains inert for its
-- new NULL-period rows. Do not re-enable #2448 merely by re-adding the check:
-- if the retained runtime created recurring invoices, first reconcile those
-- rows with an evidence-backed forward recovery, because 387 is already
-- recorded and will not run a second time.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index AS index_state
        JOIN pg_class AS table_class
          ON table_class.oid = index_state.indrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        JOIN pg_class AS index_class
          ON index_class.oid = index_state.indexrelid
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'invoices'
          AND index_class.relname = 'idx_invoices_recurring_contact_period_source'
          AND index_state.indisunique
          AND index_state.indisvalid
          AND index_state.indisready
          AND index_state.indnkeyatts = 2
          AND pg_get_indexdef(index_state.indexrelid, 1, true) = 'contact_id'
          AND pg_get_indexdef(index_state.indexrelid, 2, true) = 'billing_period'
          AND btrim(
              regexp_replace(
                  translate(
                      regexp_replace(
                          lower(pg_get_expr(index_state.indpred, index_state.indrelid)),
                          E'::(character varying|varchar|text|name)(\\[\\])?',
                          '',
                          'g'
                      ),
                      '''[](),',
                      '      '
                  ),
                  E'\\s+',
                  ' ',
                  'g'
              )
          ) = 'billing_period is not null and status <> void and source = any array monthly_auto eom_commercial_billing'
    ) THEN
        RAISE EXCEPTION
            'Cannot recover migration 385: recurring billing_period index is missing, invalid, or has an unexpected definition';
    END IF;
END $$;

-- Recovery DML is authorized only for the exact recorded initial-385 catalog.
-- A final 385 catalog already has both of these historical artifacts. Its
-- collision classification is frozen evidence, not a view over current invoice
-- status: a later void must not make a surviving quarantined invoice eligible
-- for a second backfill attempt. Keep 387 ledger-only in that final state.
CREATE TEMP TABLE eom_recurring_invoice_dedup_recovery_scope (
    needs_historical_recovery BOOLEAN NOT NULL
) ON COMMIT DROP;

INSERT INTO eom_recurring_invoice_dedup_recovery_scope (
    needs_historical_recovery
)
SELECT NOT (
    EXISTS (
        SELECT 1
        FROM pg_attribute AS attribute_state
        JOIN pg_class AS table_class
          ON table_class.oid = attribute_state.attrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'invoices'
          AND attribute_state.attname = 'billing_period_legacy_null'
          AND attribute_state.attnum > 0
          AND NOT attribute_state.attisdropped
    )
    AND to_regclass(format(
        '%I.%I', current_schema(), 'invoices_billing_period_reservations'
    )) IS NOT NULL
);

ALTER TABLE invoices
    ADD COLUMN IF NOT EXISTS billing_period_legacy_null BOOLEAN NOT NULL DEFAULT false;

DO $$
DECLARE
    period_check_definition TEXT;
BEGIN
    -- A strict final check cannot safely be added over a preexisting invalid
    -- stored value. Do not coerce an historical period; fail atomically so an
    -- operator can reconcile the unexpected state with its source evidence.
    IF EXISTS (
        SELECT 1
        FROM invoices
        WHERE billing_period IS NOT NULL
          AND billing_period !~
              '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$'
    ) THEN
        RAISE EXCEPTION
            'Cannot recover migration 385: invoices.billing_period contains a value outside the final YYYY-MM grammar';
    END IF;

    SELECT pg_get_expr(constraint_state.conbin, constraint_state.conrelid)
    INTO period_check_definition
    FROM pg_constraint AS constraint_state
    WHERE constraint_state.conrelid = 'invoices'::regclass
      AND constraint_state.conname = 'invoices_billing_period_check';

    -- The recorded initial check accepted 0000-01. Keep a clean final-385
    -- schema untouched, but replace a missing, older, or token-similar weaker
    -- definition with the exact nonzero-year grammar required by
    -- recurring_invoice_dedup_schema_ready.
    IF period_check_definition IS NULL
       OR btrim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(period_check_definition),
                        E'::(character varying|varchar|text|name)(\\[\\])?',
                        '',
                        'g'
                    ),
                    '''',
                    '',
                    'g'
                ),
                E'\\s+',
                ' ',
                'g'
            )
        ) <> '((billing_period) ~ ^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$)' THEN
        IF period_check_definition IS NOT NULL THEN
            ALTER TABLE invoices
                DROP CONSTRAINT invoices_billing_period_check;
        END IF;

        ALTER TABLE invoices
            ADD CONSTRAINT invoices_billing_period_check
            CHECK (
                billing_period ~
                    '^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2])$'
            ) NOT VALID;
    END IF;
END $$;

DO $$
DECLARE
    required_check_definition TEXT;
BEGIN
    SELECT pg_get_expr(constraint_state.conbin, constraint_state.conrelid)
    INTO required_check_definition
    FROM pg_constraint AS constraint_state
    WHERE constraint_state.conrelid = 'invoices'::regclass
      AND constraint_state.conname =
          'invoices_recurring_billing_period_required_check';

    -- A named check with all of the expected words can still be a tautology.
    -- Compare the complete canonical expression before preserving it; this is
    -- the same fail-closed contract the runtime readiness predicate applies.
    IF required_check_definition IS NULL
       OR btrim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(required_check_definition),
                        E'::(character varying|varchar|text|name)(\\[\\])?',
                        '',
                        'g'
                    ),
                    '''',
                    '',
                    'g'
                ),
                E'\\s+',
                ' ',
                'g'
            )
        ) <> '(((source) <> all ((array[monthly_auto, eom_commercial_billing]))) or ((status) = void) or (billing_period is not null) or billing_period_legacy_null)' THEN
        IF required_check_definition IS NOT NULL THEN
            ALTER TABLE invoices
                DROP CONSTRAINT invoices_recurring_billing_period_required_check;
        END IF;

        ALTER TABLE invoices
            ADD CONSTRAINT invoices_recurring_billing_period_required_check
            CHECK (
                source NOT IN ('monthly_auto', 'eom_commercial_billing')
                OR status = 'void'
                OR billing_period IS NOT NULL
                OR billing_period_legacy_null
            ) NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS invoices_billing_period_reservations (
    contact_id      UUID NOT NULL,
    billing_period  VARCHAR(7) NOT NULL,
    reason          VARCHAR(32) NOT NULL DEFAULT 'backfill_collision',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (contact_id, billing_period)
);

-- Freeze each unresolved historical row's candidate once. A candidate is
-- ambiguous if another NULL-period candidate wants the same contact/period OR
-- a partially deployed newer writer already populated that slot. In both
-- cases the NULL row stays NULL: selecting a winner would rewrite history.
CREATE TEMP TABLE eom_recurring_invoice_dedup_recovery_candidates (
    id               UUID PRIMARY KEY,
    contact_id       UUID,
    candidate_period VARCHAR(7),
    is_collision     BOOLEAN NOT NULL
) ON COMMIT DROP;

INSERT INTO eom_recurring_invoice_dedup_recovery_candidates (
    id, contact_id, candidate_period, is_collision
)
WITH candidates AS (
    SELECT
        inv.id,
        inv.contact_id,
        CASE
            WHEN inv.source = 'monthly_auto'
                 AND inv.source_ref ~
                     '_((000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(0[1-9]|1[0-2]))$'
                THEN substring(
                    inv.source_ref FROM
                        '_((?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(?:0[1-9]|1[0-2]))$'
                )
            WHEN inv.source = 'eom_commercial_billing'
                 AND inv.invoice_number ~
                     '^INV-(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4,}$'
                THEN to_char(
                    to_date(
                        substring(
                            inv.invoice_number FROM
                                '^INV-((?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))-\d{4,}$'
                        ),
                        'YYYY-Mon'
                    ),
                    'YYYY-MM'
                )
            ELSE NULL
        END AS candidate_period
    FROM invoices AS inv
    CROSS JOIN eom_recurring_invoice_dedup_recovery_scope AS recovery_scope
    WHERE recovery_scope.needs_historical_recovery
      AND inv.billing_period IS NULL
      AND inv.status <> 'void'
      AND inv.source IN ('monthly_auto', 'eom_commercial_billing')
)
SELECT
    candidate.id,
    candidate.contact_id,
    candidate.candidate_period,
    candidate.contact_id IS NOT NULL
    AND candidate.candidate_period IS NOT NULL
    AND (
        EXISTS (
            SELECT 1
            FROM candidates AS peer
            WHERE peer.id <> candidate.id
              AND peer.contact_id = candidate.contact_id
              AND peer.candidate_period = candidate.candidate_period
        )
        OR EXISTS (
            SELECT 1
            FROM invoices AS populated
            WHERE populated.contact_id = candidate.contact_id
              AND populated.billing_period = candidate.candidate_period
              AND populated.status <> 'void'
              AND populated.source IN ('monthly_auto', 'eom_commercial_billing')
        )
    ) AS is_collision
FROM candidates AS candidate;

-- Migration 377 deliberately rejects every invoice mutation while an approved
-- commercial invoice has a pending Gmail draft replacement. Its trigger is
-- valid safety behavior, so do not bypass it or discover it after partially
-- mutating legacy rows. Materialize the exact recovery candidates first, then
-- preflight only the initial-catalog rows one of the following invoices UPDATE
-- statements will change. A final catalog has no recovery DML, so it must not
-- be coupled to migration 377's trigger or replacement-table catalog.
DO $$
DECLARE
    pending_gmail_replacement BOOLEAN := false;
    needs_historical_recovery BOOLEAN;
BEGIN
    SELECT recovery_scope.needs_historical_recovery
    INTO needs_historical_recovery
    FROM eom_recurring_invoice_dedup_recovery_scope AS recovery_scope;

    IF needs_historical_recovery
       AND EXISTS (
        SELECT 1
        FROM pg_trigger AS trigger_state
        JOIN pg_class AS table_class
          ON table_class.oid = trigger_state.tgrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'invoices'
          AND trigger_state.tgname =
              'commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending'
          AND NOT trigger_state.tgisinternal
          AND trigger_state.tgenabled <> 'D'
    ) THEN
        IF to_regclass(format(
            '%I.%I', current_schema(), 'commercial_billing_candidate_approvals'
        )) IS NULL
           OR to_regclass(format(
            '%I.%I', current_schema(), 'commercial_billing_invoice_gmail_drafts'
        )) IS NULL
           OR to_regclass(format(
            '%I.%I', current_schema(),
            'commercial_billing_invoice_gmail_draft_replacement_events'
        )) IS NULL THEN
            RAISE EXCEPTION
                'Cannot recover migration 385: the enabled Gmail replacement trigger has an incomplete catalog; reconcile migration 377 before retrying';
        END IF;

        EXECUTE $pending_replacement$
            SELECT EXISTS (
                SELECT 1
                FROM invoices AS inv
                JOIN eom_recurring_invoice_dedup_recovery_candidates AS candidate
                  ON candidate.id = inv.id
                JOIN commercial_billing_candidate_approvals AS approval
                  ON approval.invoice_id = inv.id
                JOIN commercial_billing_invoice_gmail_drafts AS draft
                  ON draft.approval_id = approval.id
                JOIN commercial_billing_invoice_gmail_draft_replacement_events AS replacement
                  ON replacement.gmail_draft_record_id = draft.id
                 AND replacement.replacement_generation = draft.draft_generation
                WHERE draft.state IN ('creating', 'retryable', 'recovery_required')
                  AND (
                      inv.billing_period_legacy_null IS DISTINCT FROM true
                      OR (
                          candidate.candidate_period IS NOT NULL
                          AND NOT candidate.is_collision
                          AND (
                              inv.billing_period IS DISTINCT FROM candidate.candidate_period
                              OR inv.billing_period_legacy_null IS DISTINCT FROM false
                          )
                      )
                      OR (
                          candidate.candidate_period IS NOT NULL
                          AND candidate.is_collision
                          AND (
                              inv.billing_period_legacy_null IS DISTINCT FROM true
                              OR inv.metadata->>'billing_period_backfill_collision'
                                  IS DISTINCT FROM 'true'
                              OR inv.metadata->>'billing_period_backfill_candidate_period'
                                  IS DISTINCT FROM candidate.candidate_period
                          )
                      )
                      OR (
                          candidate.candidate_period IS NULL
                          AND inv.billing_period IS NULL
                          AND (
                              inv.billing_period_legacy_null IS DISTINCT FROM true
                              OR inv.metadata->>'billing_period_legacy_null'
                                  IS DISTINCT FROM 'true'
                          )
                      )
                  )
            )
        $pending_replacement$
        INTO pending_gmail_replacement;

        IF pending_gmail_replacement THEN
            RAISE EXCEPTION
                'Cannot recover migration 385: pending Gmail draft replacement blocks a recurring legacy invoice update; complete or reconcile the replacement before retrying';
        END IF;
    END IF;
END $$;

-- Preserve the existing NULL-period historical rows before installing the
-- fresh-write gate only in the observed initial catalog. Rows that later
-- backfill below return to false. The candidate-derived preflight above runs
-- first because this is the first invoices UPDATE in the recovery.
UPDATE invoices AS inv
SET billing_period_legacy_null = true
FROM eom_recurring_invoice_dedup_recovery_scope AS recovery_scope
WHERE recovery_scope.needs_historical_recovery
  AND inv.billing_period IS NULL
  AND inv.status <> 'void'
  AND inv.source IN ('monthly_auto', 'eom_commercial_billing')
  AND inv.billing_period_legacy_null IS DISTINCT FROM true;

UPDATE invoices AS inv
SET billing_period = candidate.candidate_period,
    billing_period_legacy_null = false
FROM eom_recurring_invoice_dedup_recovery_candidates AS candidate
WHERE inv.id = candidate.id
  AND candidate.candidate_period IS NOT NULL
  AND NOT candidate.is_collision
  AND (
      inv.billing_period IS DISTINCT FROM candidate.candidate_period
      OR inv.billing_period_legacy_null IS DISTINCT FROM false
  );

UPDATE invoices AS inv
SET metadata = COALESCE(inv.metadata, '{}'::jsonb)
        || jsonb_build_object(
            'billing_period_backfill_collision', true,
            'billing_period_backfill_candidate_period', candidate.candidate_period
        ),
    billing_period_legacy_null = true
FROM eom_recurring_invoice_dedup_recovery_candidates AS candidate
WHERE inv.id = candidate.id
  AND candidate.candidate_period IS NOT NULL
  AND candidate.is_collision
  AND (
      inv.billing_period_legacy_null IS DISTINCT FROM true
      OR inv.metadata->>'billing_period_backfill_collision' IS DISTINCT FROM 'true'
      OR inv.metadata->>'billing_period_backfill_candidate_period'
          IS DISTINCT FROM candidate.candidate_period
  );

UPDATE invoices AS inv
SET metadata = COALESCE(inv.metadata, '{}'::jsonb)
        || jsonb_build_object('billing_period_legacy_null', true),
    billing_period_legacy_null = true
FROM eom_recurring_invoice_dedup_recovery_candidates AS candidate
WHERE inv.id = candidate.id
  AND candidate.candidate_period IS NULL
  AND inv.billing_period IS NULL
  AND (
      inv.billing_period_legacy_null IS DISTINCT FROM true
      OR inv.metadata->>'billing_period_legacy_null' IS DISTINCT FROM 'true'
  );

INSERT INTO invoices_billing_period_reservations (contact_id, billing_period)
SELECT DISTINCT candidate.contact_id, candidate.candidate_period
FROM eom_recurring_invoice_dedup_recovery_candidates AS candidate
WHERE candidate.is_collision
  AND candidate.contact_id IS NOT NULL
  AND candidate.candidate_period IS NOT NULL
ON CONFLICT (contact_id, billing_period) DO NOTHING;

COMMENT ON COLUMN invoices.billing_period IS
    'YYYY-MM covered billing period. Backfilled on historical rows where mechanically derivable from source_ref (monthly_auto) or invoice_number (eom_commercial_billing); NULL where unparseable or where two historical rows collided (see metadata.billing_period_backfill_collision). Source-of-truth for cross-pipeline recurring-invoice dedup; see idx_invoices_recurring_contact_period_source.';

COMMENT ON COLUMN invoices.billing_period_legacy_null IS
    'Database-owned exemption for pre-migration recurring invoice rows whose billing_period remains NULL because the historical period was unparseable or collision-quarantined. New recurring writers do not set this column, so invoices_recurring_billing_period_required_check rejects fresh NULL-period recurring invoices while preserving later edits to explicit legacy exceptions.';

COMMENT ON TABLE invoices_billing_period_reservations IS
    'One row per (contact_id, billing_period) whose historical backfill was ambiguous (see invoices.metadata.billing_period_backfill_collision). Checked by both recurring writers'' pre-checks alongside invoices, so a third invoice for an already-quarantined period is refused while any matching non-void quarantined invoice remains, even though the slot cannot be enforced by idx_invoices_recurring_contact_period_source itself (no row claims it). Voiding every matching quarantined invoice releases the reservation at read time.';
