"""Startup readiness checks for the private EOM funnel API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..storage.database import get_db_pool


async def require_eom_funnel_data_store(
    config: object,
    *,
    database_enabled: bool,
    pool_getter: Callable[[], Any] = get_db_pool,
) -> None:
    """Fail closed if an enabled handoff cannot use the primary CRM store."""
    if not bool(getattr(config, "api_enabled", False)):
        return
    if not database_enabled:
        raise RuntimeError("EOM funnel requires the authoritative Atlas database")

    pool = pool_getter()
    if not pool.is_initialized:
        raise RuntimeError("EOM funnel requires an initialized Atlas database pool")
    ready = await pool.fetchval(
        """
        SELECT to_regclass('contacts') IS NOT NULL
           AND to_regclass('eom_lead_lifecycle_events') IS NOT NULL
           AND to_regclass('eom_customer_handoffs') IS NOT NULL
           AND EXISTS (
               SELECT 1
               FROM pg_class AS handoff_table
               JOIN pg_roles AS owner_role ON owner_role.oid = handoff_table.relowner
               WHERE handoff_table.oid = 'eom_customer_handoffs'::regclass
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
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_customer_handoffs', 'INSERT')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_customer_handoffs', 'UPDATE')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_customer_handoffs', 'DELETE')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_customer_handoffs', 'TRUNCATE')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_lead_lifecycle_events', 'INSERT')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_lead_lifecycle_events', 'UPDATE')
                  AND NOT has_table_privilege(nocodb_role.oid, 'eom_lead_lifecycle_events', 'DELETE')
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'business_context_id', 'INSERT'
                  )
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'business_context_id', 'UPDATE'
                  )
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'contact_type', 'INSERT'
                  )
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'contact_type', 'UPDATE'
                  )
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'lead_stage', 'INSERT'
                  )
                  AND NOT has_column_privilege(
                      nocodb_role.oid, 'contacts', 'lead_stage', 'UPDATE'
                  )
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
        """
    )
    if not ready:
        raise RuntimeError(
            "EOM funnel requires the authoritative CRM lifecycle and handoff schema"
        )
