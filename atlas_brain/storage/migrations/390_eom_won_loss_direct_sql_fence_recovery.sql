-- atlas: atomic-bookkeeping
-- Forward-only recovery for targets that recorded the original 386 NocoDB-only
-- fence before its direct-SQL boundary was strengthened. The migration runner
-- admits this file before later pending EOM migrations only after the exact
-- historical ledger and weak catalog state are attested.
--
-- This is intentionally additive: it does not rewrite migration 386 or any
-- ledger receipt. Reapplying the function and trigger is safe after a retry.
-- A database administrator must run this recovery: its SECURITY DEFINER
-- function moves from the historical application login to the existing
-- no-login EOM guard. The normal Atlas runtime must never acquire that guard
-- membership merely to recover this fence.
-- Rollback evidence: retain this lifecycle fence and its receipt; an approved
-- destructive rollback must settle cancellation evidence before removing the
-- trigger and then the function.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    executor_is_superuser BOOLEAN;
    trusted_guard_role_ready BOOLEAN;
BEGIN
    SELECT COALESCE(executor_role.rolsuper, FALSE)
    INTO executor_is_superuser
    FROM pg_catalog.pg_roles AS executor_role
    WHERE executor_role.rolname = current_user;
    executor_is_superuser := COALESCE(executor_is_superuser, FALSE);

    IF NOT executor_is_superuser THEN
        RAISE EXCEPTION
            'database administrator must run 390_eom_won_loss_direct_sql_fence_recovery';
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_roles AS guard_role
        WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
          AND NOT guard_role.rolcanlogin
          AND NOT guard_role.rolinherit
          AND NOT guard_role.rolsuper
          AND NOT guard_role.rolcreaterole
          AND NOT guard_role.rolcreatedb
          AND NOT guard_role.rolreplication
          AND NOT guard_role.rolbypassrls
          AND pg_catalog.has_schema_privilege(
              guard_role.oid,
              schema_name,
              'USAGE'
          )
          AND pg_catalog.has_schema_privilege(
              guard_role.oid,
              schema_name,
              'CREATE'
          )
          AND NOT EXISTS (
              SELECT 1
              FROM pg_catalog.pg_roles AS member_role
              WHERE member_role.rolcanlogin
                AND NOT member_role.rolsuper
                AND pg_catalog.pg_has_role(
                    member_role.oid,
                    guard_role.oid,
                    'MEMBER'
                )
          )
    )
    INTO trusted_guard_role_ready;

    IF NOT trusted_guard_role_ready THEN
        RAISE EXCEPTION
            'atlas_eom_handoff_owner must be a no-login, membership-isolated guard role before running 390_eom_won_loss_direct_sql_fence_recovery';
    END IF;

    EXECUTE format(
        $sql$
        CREATE OR REPLACE FUNCTION %1$I.reject_nocodb_eom_won_loss_mutation()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, %1$I
        AS $function$
        BEGIN
            IF OLD.business_context_id = 'effingham_maids'
               AND OLD.contact_type = 'lead'
               AND OLD.lead_stage = 'won'
               AND EXISTS (
                   SELECT 1
                   FROM eom_lead_lifecycle_events AS requested
                   WHERE requested.contact_id = OLD.id
                     AND requested.event_type = 'first_clean_cancellation_requested'
                     AND NOT EXISTS (
                         SELECT 1
                         FROM eom_lead_lifecycle_events AS completed
                         WHERE completed.contact_id = requested.contact_id
                           AND completed.event_type = 'first_clean_cancelled'
                           AND completed.operation_key = requested.operation_key
                     )
               )
            THEN
                RAISE EXCEPTION
                    'EOM won lead loss cancellation requires reconciliation before direct contact mutation';
            END IF;

            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $function$;
        $sql$,
        schema_name
    );

    EXECUTE format(
        'DROP TRIGGER IF EXISTS trg_reject_nocodb_eom_won_loss_mutation ON %I.contacts',
        schema_name
    );
    EXECUTE format(
        'CREATE TRIGGER trg_reject_nocodb_eom_won_loss_mutation '
        || 'BEFORE UPDATE OF status, contact_type OR DELETE ON %I.contacts '
        || 'FOR EACH ROW EXECUTE FUNCTION %I.reject_nocodb_eom_won_loss_mutation()',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.reject_nocodb_eom_won_loss_mutation() FROM PUBLIC',
        schema_name
    );
    -- The guard owner needs only this read to evaluate the trigger predicate;
    -- it receives neither general CRM write access nor a login capability.
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_lead_lifecycle_events TO atlas_eom_handoff_owner',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.reject_nocodb_eom_won_loss_mutation() '
        || 'OWNER TO atlas_eom_handoff_owner',
        schema_name
    );
END;
$$;
