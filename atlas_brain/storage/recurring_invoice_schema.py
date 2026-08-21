"""Dependency-light recurring-invoice schema readiness checks.

This predicate is shared by recurring invoice writers and migration provenance
probes. Keeping it outside ``storage.repositories`` prevents an operator's
read-only migration check from importing unrelated optional repository modules.
"""

from __future__ import annotations

import re
from typing import Any


_RECURRING_INVOICE_DEDUP_INDEX = "idx_invoices_recurring_contact_period_source"
_RECURRING_INVOICE_DEDUP_INDEX_KEYS = ("contact_id", "billing_period")
_RECURRING_INVOICE_DEDUP_INDEX_PREDICATE_CLAUSES = frozenset(
    {
        "billing_period is not null",
        "source = any array monthly_auto eom_commercial_billing",
        "status <> void",
    }
)
_RECURRING_INVOICE_DEDUP_CONSTRAINTS = (
    "invoices_billing_period_check",
    "invoices_recurring_billing_period_required_check",
)
_RECURRING_INVOICE_DEDUP_CONSTRAINT_EXPRESSIONS = {
    "invoices_billing_period_check": (
        "((billing_period) ~ "
        "^(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})-"
        "(0[1-9]|1[0-2])$)"
    ),
    "invoices_recurring_billing_period_required_check": (
        "(((source) <> all ((array[monthly_auto, eom_commercial_billing]))) "
        "or ((status) = void) or (billing_period is not null) "
        "or billing_period_legacy_null)"
    ),
}


def _normalize_schema_definition(definition: object) -> str:
    """Return a stable lower-case representation of catalog DDL text."""
    return " ".join(str(definition or "").lower().split())


def _canonicalize_catalog_expression(expression: object) -> str:
    """Return a compact comparable form for Postgres expression text."""
    normalized = _normalize_schema_definition(expression)
    normalized = re.sub(
        r"::(?:character varying|varchar|text|name)(?:\[\])?",
        "",
        normalized,
    )
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[\[\](),]", " ", normalized)
    return " ".join(normalized.split())


def _canonicalize_catalog_constraint_expression(expression: object) -> str:
    """Return an exact comparable form for a PostgreSQL CHECK expression."""
    normalized = _normalize_schema_definition(expression)
    normalized = re.sub(
        r"::(?:character varying|varchar|text|name)(?:\[\])?",
        "",
        normalized,
    )
    return " ".join(normalized.replace("'", "").split())


def _recurring_index_predicate_ready(predicate: object) -> bool:
    clauses = frozenset(
        clause.strip()
        for clause in re.split(r"\s+and\s+", _canonicalize_catalog_expression(predicate))
        if clause.strip()
    )
    return clauses == _RECURRING_INVOICE_DEDUP_INDEX_PREDICATE_CLAUSES


async def recurring_invoice_dedup_schema_ready(conn: Any) -> bool:
    """Return whether recurring invoice writers can safely rely on period dedup.

    This is deliberately separate from receivables/payment readiness. Check,
    ACH, Square, ad-hoc invoice, and EOM funnel surfaces do not create
    ``monthly_auto`` or ``eom_commercial_billing`` invoices and must not be
    blocked by this writer-only schema requirement.
    """
    columns_ready = bool(
        await conn.fetchval(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM (
                    VALUES
                        ('invoices', 'billing_period'),
                        ('invoices', 'billing_period_legacy_null'),
                        ('invoices_billing_period_reservations', 'contact_id'),
                        ('invoices_billing_period_reservations', 'billing_period')
                ) AS required(table_name, column_name)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns AS actual
                    WHERE actual.table_schema = current_schema()
                      AND actual.table_name = required.table_name
                      AND actual.column_name = required.column_name
                )
            )
            """
        )
    )
    if not columns_ready:
        return False

    constraint_rows = await conn.fetch(
        """
        SELECT actual.conname, pg_get_expr(actual.conbin, actual.conrelid) AS definition
        FROM pg_constraint AS actual
        JOIN pg_class AS table_class
          ON table_class.oid = actual.conrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'invoices'
          AND actual.conname = ANY($1::text[])
        """,
        list(_RECURRING_INVOICE_DEDUP_CONSTRAINTS),
    )
    constraint_definitions = {
        row["conname"]: row["definition"]
        for row in constraint_rows
    }
    constraints_ready = all(
        _canonicalize_catalog_constraint_expression(
            constraint_definitions.get(name)
        )
        == expected
        for name, expected in _RECURRING_INVOICE_DEDUP_CONSTRAINT_EXPRESSIONS.items()
    )
    if not constraints_ready:
        return False

    index_row = await conn.fetchrow(
        """
        SELECT
            index_state.indisunique,
            index_state.indisvalid,
            index_state.indisready,
            index_state.indnkeyatts,
            pg_get_indexdef(index_state.indexrelid, 1, true) AS key_column_1,
            pg_get_indexdef(index_state.indexrelid, 2, true) AS key_column_2,
            pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate
        FROM pg_index AS index_state
        JOIN pg_class AS table_class
          ON table_class.oid = index_state.indrelid
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        JOIN pg_class AS index_class
          ON index_class.oid = index_state.indexrelid
        WHERE table_namespace.nspname = current_schema()
          AND table_class.relname = 'invoices'
          AND index_class.relname = $1
        """,
        _RECURRING_INVOICE_DEDUP_INDEX,
    )
    if not index_row:
        return False
    key_columns = (
        str(index_row["key_column_1"] or ""),
        str(index_row["key_column_2"] or ""),
    )
    return (
        bool(index_row["indisunique"])
        and bool(index_row["indisvalid"])
        and bool(index_row["indisready"])
        and int(index_row["indnkeyatts"] or 0) == len(_RECURRING_INVOICE_DEDUP_INDEX_KEYS)
        and key_columns == _RECURRING_INVOICE_DEDUP_INDEX_KEYS
        and _recurring_index_predicate_ready(index_row["predicate"])
    )
