-- Fence direct CRM SQL while a won lead's Calendar cancellation remains
-- unresolved. The provider fence cannot cover operational database clients.
--
-- The function is SECURITY DEFINER because atlas_nocodb intentionally has no
-- direct access to immutable lifecycle evidence. Its fixed schema-qualified
-- search path prevents a caller-controlled object from influencing the check.

DO $$
DECLARE
    schema_name TEXT := current_schema();
BEGIN
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
END;
$$;
