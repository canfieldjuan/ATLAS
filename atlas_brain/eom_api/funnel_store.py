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
    public_onboarding_enabled = bool(
        getattr(config, "public_onboarding_enabled", False)
    )
    # The value is a locally derived boolean, not caller input. Inlining it
    # retains compatibility with the lightweight readiness-pool test seam while
    # still making the migration requirement part of the one startup query.
    public_onboarding_enabled_sql = (
        "TRUE" if public_onboarding_enabled else "FALSE"
    )
    ready = await pool.fetchval(
        f"""
        WITH readiness_relations AS (
            SELECT
                to_regclass('contacts') AS contacts_rel,
                to_regclass('eom_lead_lifecycle_events') AS lifecycle_rel,
                to_regclass('eom_customer_handoffs') AS handoff_rel,
                to_regclass('eom_onboarding_email_drafts') AS onboarding_drafts_rel,
                to_regclass('eom_public_onboarding_tokens') AS public_onboarding_tokens_rel
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
                )
                -- The operator contact INSERT names customer_type explicitly
                -- (migration 366). Startup catches a failed migration and
                -- continues, so without this the funnel would be admitted
                -- against a contacts table lacking the column and every
                -- operator create would fail at the write instead of at the
                -- readiness gate.
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.contacts_rel
                      AND attname = 'customer_type'
                      AND NOT attisdropped
                )
                -- known-contacts publishes the database-owned source revision
                -- for customer_type (migration 367). A partially migrated
                -- store would otherwise pass startup then fail only when a
                -- tracker refresh asks the provider to read it.
                AND EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = readiness_relations.contacts_rel
                      AND attname = 'customer_type_revision'
                      AND NOT attisdropped
                      AND atttypid = 'bigint'::regtype
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
                ) AS onboarding_drafts_required_columns_ready,
                -- A disabled authority may safely precede migration 383. Once
                -- the relation exists, however, the ordinary office fence and
                -- private revoke-link recovery path still SELECT ... FOR UPDATE
                -- and UPDATE it. Check that smaller unconditional surface even
                -- while issuance is dormant, so those paths cannot first fail
                -- on a partial migration or missing runtime DML privilege.
                (
                    readiness_relations.public_onboarding_tokens_rel IS NULL
                    OR (
                        NOT EXISTS (
                            SELECT required.attname
                            FROM unnest(ARRAY[
                                'id', 'draft_id', 'contact_id', 'status', 'revoked_at'
                            ]) AS required(attname)
                            WHERE NOT EXISTS (
                                SELECT 1
                                FROM pg_attribute
                                WHERE attrelid = readiness_relations.public_onboarding_tokens_rel
                                  AND attname = required.attname
                                  AND NOT attisdropped
                            )
                        )
                        AND has_table_privilege(
                            current_user,
                            readiness_relations.public_onboarding_tokens_rel,
                            'SELECT'
                        )
                        AND has_table_privilege(
                            current_user,
                            readiness_relations.public_onboarding_tokens_rel,
                            'UPDATE'
                        )
                    )
                ) AS public_onboarding_recovery_ready,
                -- Issuance itself remains explicitly enabled-only. It needs the
                -- full immutable projection, durable constraints/index, and
                -- INSERT in addition to the recovery surface above.
                CASE WHEN {public_onboarding_enabled_sql} THEN (
                    readiness_relations.public_onboarding_tokens_rel IS NOT NULL
                    AND NOT EXISTS (
                        SELECT required.attname
                        FROM unnest(ARRAY[
                            'id', 'draft_id', 'contact_id',
                            'signing_key_fingerprint', 'prefill_full_name',
                            'prefill_email', 'prefill_phone', 'prefill_address',
                            'prefill_city', 'prefill_state', 'prefill_zip',
                            'prefill_customer_type', 'approval_key', 'status',
                            'approved_by_employee_id', 'approved_by_name',
                            'issued_at', 'redeemed_at', 'revoked_at', 'handoff_id'
                        ]) AS required(attname)
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM pg_attribute
                            WHERE attrelid = readiness_relations.public_onboarding_tokens_rel
                              AND attname = required.attname
                              AND NOT attisdropped
                        )
                    )
                    AND (
                        SELECT COUNT(DISTINCT token_constraint.conname) = 6
                        FROM pg_constraint AS token_constraint
                        WHERE token_constraint.conrelid = readiness_relations.public_onboarding_tokens_rel
                          AND token_constraint.conname = ANY(ARRAY[
                              'pk_eom_public_onboarding_tokens',
                              'uq_eom_public_onboarding_tokens_draft',
                              'uq_eom_public_onboarding_tokens_approval',
                              'uq_eom_public_onboarding_tokens_handoff',
                              'ck_eom_public_onboarding_tokens_status',
                              'ck_eom_public_onboarding_tokens_terminal_state'
                          ])
                    )
                    AND EXISTS (
                        SELECT 1
                        FROM pg_index AS token_index
                        JOIN pg_class AS token_index_relation
                          ON token_index_relation.oid = token_index.indexrelid
                        WHERE token_index.indrelid = readiness_relations.public_onboarding_tokens_rel
                          AND token_index_relation.relname = 'uq_eom_public_onboarding_tokens_issued_contact'
                          AND token_index.indisunique
                          AND token_index.indpred IS NOT NULL
                    )
                    AND has_table_privilege(
                        current_user,
                        readiness_relations.public_onboarding_tokens_rel,
                        'SELECT'
                    )
                    AND has_table_privilege(
                        current_user,
                        readiness_relations.public_onboarding_tokens_rel,
                        'INSERT'
                    )
                    AND has_table_privilege(
                        current_user,
                        readiness_relations.public_onboarding_tokens_rel,
                        'UPDATE'
                    )
                ) ELSE TRUE END AS public_onboarding_issuance_ready
            FROM readiness_relations
        )
        SELECT readiness_relations.contacts_rel IS NOT NULL
           AND readiness_relations.lifecycle_rel IS NOT NULL
           AND readiness_relations.handoff_rel IS NOT NULL
           AND readiness_relations.onboarding_drafts_rel IS NOT NULL
           AND (NOT {public_onboarding_enabled_sql} OR readiness_relations.public_onboarding_tokens_rel IS NOT NULL)
           AND readiness_columns.contacts_required_columns_ready
           AND readiness_columns.lifecycle_required_columns_ready
           AND readiness_columns.onboarding_drafts_required_columns_ready
           AND readiness_columns.public_onboarding_recovery_ready
           AND readiness_columns.public_onboarding_issuance_ready
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
