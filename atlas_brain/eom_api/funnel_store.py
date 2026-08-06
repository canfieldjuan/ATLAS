"""Shared datastore readiness guard for the private EOM funnel API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


async def require_eom_funnel_data_store(
    config: object,
    *,
    database_enabled: bool,
    get_db_pool_fn: Callable[[], Any],
) -> None:
    """Fail closed if an enabled handoff cannot use the primary CRM store."""
    if not bool(getattr(config, "api_enabled", False)):
        return
    if not database_enabled:
        raise RuntimeError("EOM funnel requires the authoritative Atlas database")

    pool = get_db_pool_fn()
    if not pool.is_initialized:
        raise RuntimeError("EOM funnel requires an initialized Atlas database pool")
    ready = await pool.fetchval(
        """
        WITH readiness_relations AS (
            SELECT
                to_regclass('contacts') AS contacts_rel,
                to_regclass('eom_lead_lifecycle_events') AS lifecycle_rel,
                to_regclass('eom_customer_handoffs') AS handoff_rel,
                to_regclass('eom_onboarding_email_drafts') AS onboarding_drafts_rel
        ),
        readiness_columns AS (
            SELECT
                readiness_relations.contacts_rel IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.contacts_rel
                      AND attname = 'business_context_id'
                      AND NOT attisdropped
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.contacts_rel
                      AND attname = 'contact_type'
                      AND NOT attisdropped
                )
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.contacts_rel
                      AND attname = 'lead_stage'
                      AND NOT attisdropped
                ) AS contacts_required_columns_ready,
                -- Reopen orders lost-stage evidence by migration 363's
                -- database-owned append sequence. Serving without it would turn
                -- the first reopen request into undefined_column after startup.
                readiness_relations.lifecycle_rel IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.lifecycle_rel
                      AND attname = 'lifecycle_sequence'
                      AND NOT attisdropped
                ) AS lifecycle_required_columns_ready,
                -- First-clean completion inserts the onboarding draft in the
                -- same transaction as the won transition; admitting the funnel
                -- without migration 360 would let Calendar creation succeed
                -- and then wedge the operation ambiguous on undefined_table.
                readiness_relations.onboarding_drafts_rel IS NOT NULL
                AND NOT EXISTS (
                    SELECT required.attname
                    FROM unnest(ARRAY[
                        'contact_id', 'operation_key', 'status',
                        'recipient_email', 'blocker', 'subject', 'body'
                    ]) AS required(attname)
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_attribute
                        WHERE attrelid = readiness_relations.onboarding_drafts_rel
                          AND attname = required.attname
                          AND NOT attisdropped
                    )
                )
                -- The funnel actor boundary admits signed-64 ids; serving
                -- draft approvals against migration 360's INTEGER approver
                -- column would pass the HTTP boundary and then fail in
                -- Postgres after the claim. Migration 361 widens it; the
                -- slim profile only applies receivables migrations, so the
                -- funnel stays fail-closed until the canonical store has
                -- the BIGINT column.
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.onboarding_drafts_rel
                      AND attname = 'approved_by_employee_id'
                      AND NOT attisdropped
                      AND atttypid = 'bigint'::regtype
                ) AS onboarding_drafts_required_columns_ready
            FROM readiness_relations
        )
        SELECT readiness_relations.contacts_rel IS NOT NULL
           AND readiness_relations.lifecycle_rel IS NOT NULL
           AND readiness_relations.handoff_rel IS NOT NULL
           AND readiness_relations.onboarding_drafts_rel IS NOT NULL
           AND readiness_columns.contacts_required_columns_ready
           AND readiness_columns.lifecycle_required_columns_ready
           AND readiness_columns.onboarding_drafts_required_columns_ready
           AND EXISTS (
               SELECT 1
               FROM pg_class AS handoff_table
               JOIN pg_roles AS owner_role ON owner_role.oid = handoff_table.relowner
               WHERE handoff_table.oid = readiness_relations.handoff_rel
                 AND owner_role.rolname = 'atlas_eom_handoff_owner'
           )
           AND EXISTS (
               SELECT 1
               FROM pg_roles AS guard_role
               WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
                 AND has_schema_privilege(guard_role.oid, current_schema(), 'CREATE')
           )
           AND NOT EXISTS (
               SELECT 1
               FROM pg_auth_members AS membership
               JOIN pg_roles AS member_role ON member_role.oid = membership.member
               JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
               WHERE member_role.rolname = current_user
                 AND guard_role.rolname = 'atlas_eom_handoff_owner'
           )
           AND COALESCE((
               SELECT nocodb_role.rolcanlogin
                  AND NOT nocodb_role.rolsuper
                  AND NOT nocodb_role.rolcreaterole
                  AND NOT nocodb_role.rolcreatedb
                  AND NOT nocodb_role.rolreplication
                  AND NOT nocodb_role.rolbypassrls
                  AND NOT nocodb_role.rolinherit
                  AND has_database_privilege(
                      nocodb_role.oid, current_database(), 'CONNECT'
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM pg_auth_members AS nocodb_membership
                      WHERE nocodb_membership.member = nocodb_role.oid
                  )
                  AND NOT has_schema_privilege(nocodb_role.oid, current_schema(), 'CREATE')
                  AND CASE
                      WHEN readiness_relations.handoff_rel IS NULL THEN FALSE
                      ELSE NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.handoff_rel,
                          'INSERT'
                      )
                      AND NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.handoff_rel,
                          'UPDATE'
                      )
                      AND NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.handoff_rel,
                          'DELETE'
                      )
                      AND NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.handoff_rel,
                          'TRUNCATE'
                      )
                  END
                  AND CASE
                      WHEN readiness_relations.lifecycle_rel IS NULL THEN FALSE
                      ELSE NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.lifecycle_rel,
                          'INSERT'
                      )
                      AND NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.lifecycle_rel,
                          'UPDATE'
                      )
                      AND NOT has_table_privilege(
                          nocodb_role.oid,
                          readiness_relations.lifecycle_rel,
                          'DELETE'
                      )
                  END
                  AND CASE
                      WHEN NOT readiness_columns.contacts_required_columns_ready THEN FALSE
                      ELSE NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'business_context_id',
                          'INSERT'
                      )
                      AND NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'business_context_id',
                          'UPDATE'
                      )
                      AND NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'contact_type',
                          'INSERT'
                      )
                      AND NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'contact_type',
                          'UPDATE'
                      )
                      AND NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'lead_stage',
                          'INSERT'
                      )
                      AND NOT has_column_privilege(
                          nocodb_role.oid,
                          readiness_relations.contacts_rel,
                          'lead_stage',
                          'UPDATE'
                      )
                  END
                  AND NOT EXISTS (
                      SELECT 1
                      FROM pg_tables AS non_crm_table
                      WHERE non_crm_table.schemaname = current_schema()
                        AND non_crm_table.tablename NOT IN (
                            'contacts', 'contact_interactions', 'appointments'
                        )
                        AND (
                            has_table_privilege(
                                nocodb_role.oid,
                                format(
                                    '%I.%I',
                                    non_crm_table.schemaname,
                                    non_crm_table.tablename
                                ),
                                'SELECT'
                            )
                            OR has_table_privilege(
                                nocodb_role.oid,
                                format(
                                    '%I.%I',
                                    non_crm_table.schemaname,
                                    non_crm_table.tablename
                                ),
                                'INSERT'
                            )
                            OR has_table_privilege(
                                nocodb_role.oid,
                                format(
                                    '%I.%I',
                                    non_crm_table.schemaname,
                                    non_crm_table.tablename
                                ),
                                'UPDATE'
                            )
                            OR has_table_privilege(
                                nocodb_role.oid,
                                format(
                                    '%I.%I',
                                    non_crm_table.schemaname,
                                    non_crm_table.tablename
                                ),
                                'DELETE'
                            )
                            OR has_table_privilege(
                                nocodb_role.oid,
                                format(
                                    '%I.%I',
                                    non_crm_table.schemaname,
                                    non_crm_table.tablename
                                ),
                                'TRUNCATE'
                            )
                        )
                  )
               FROM pg_roles AS nocodb_role
               WHERE nocodb_role.rolname = 'atlas_nocodb'
           ), FALSE)
        FROM readiness_relations, readiness_columns
        """
    )
    if not ready:
        raise RuntimeError(
            "EOM funnel requires the authoritative CRM lifecycle and handoff schema"
        )
