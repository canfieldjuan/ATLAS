"""Tests for issue #2151 Phase 2: tenant stamping + backfill classification.

The six writer stamps are one-line kwargs verified here structurally (the
call sites live deep inside async flows whose full harnesses are out of
scope for this slice); the backfill classification is pure logic and is
tested behaviorally.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parent.parent

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

sys.path.insert(0, str(REPO / "scripts"))
from backfill_business_context import (  # noqa: E402
    B2B_CONTEXT,
    B2B_SOURCES,
    EOM_CONTEXT,
    EOM_SOURCES,
    SQL_COUNT_BY_APPOINTMENT,
    SQL_COUNT_BY_SOURCE,
    SQL_UPDATE_BY_APPOINTMENT,
    SQL_UPDATE_BY_SOURCE,
    classify_source,
)


# ---------------------------------------------------------------------------
# Backfill classification (pure)
# ---------------------------------------------------------------------------


def test_eom_sources_classify_to_effingham():
    for src in ("booking", "web", "email_backfill", "calendar_import"):
        assert classify_source(src) == "effingham_maids"


def test_b2b_sources_classify_to_churnsignals():
    for src in ("briefing_gate", "campaign_reply"):
        assert classify_source(src) == "churnsignals"


def test_unknown_sources_stay_null():
    for src in ("manual", "sms", "call", "", None, "calendar"):
        assert classify_source(src) is None


def test_source_maps_are_disjoint():
    assert not set(EOM_SOURCES) & set(B2B_SOURCES)


def test_backfill_sql_only_touches_null_rows():
    for sql in (SQL_UPDATE_BY_SOURCE, SQL_UPDATE_BY_APPOINTMENT,
                SQL_COUNT_BY_SOURCE, SQL_COUNT_BY_APPOINTMENT):
        assert "business_context_id IS NULL" in sql


def test_count_and_update_share_where_clause():
    """The dry-run count must report exactly what --apply would touch."""
    def where(sql):
        return sql.split("WHERE", 1)[1].strip()
    assert where(SQL_COUNT_BY_SOURCE) == where(SQL_UPDATE_BY_SOURCE)
    assert where(SQL_COUNT_BY_APPOINTMENT) == where(SQL_UPDATE_BY_APPOINTMENT)


def test_appointment_backfill_requires_tenant_stamped_appointment():
    assert "a.business_context_id = $1" in SQL_UPDATE_BY_APPOINTMENT
    assert "a.business_context_id = $1" in SQL_COUNT_BY_APPOINTMENT


def test_source_tier_is_opt_in():
    """Tier-2 source classification must be gated behind an explicit flag —
    contacts.source is free text (settable via the MCP tool without a
    context), so it is operator-attested, never automatic."""
    src = (REPO / "scripts/backfill_business_context.py").read_text(encoding="utf-8")
    assert "--classify-by-source" in src
    assert 'classify_by_source' in src


# ---------------------------------------------------------------------------
# Writer stamps: direct writers carry a business_context_id argument. A writer
# that delegates to the EOM inbound resolver is verified as a delegation instead:
# the resolver, not the email-digest caller, owns the tenant/type/stage stamp.
# AST checks make either guarantee fail loudly on a refactor.
# ---------------------------------------------------------------------------

WRITER_SITES = [
    # (file, callee substring, expected context expression substring)
    ("atlas_brain/tools/scheduling.py", "find_or_create_contact", "context.id"),
    ("atlas_brain/autonomous/tasks/email_backfill.py", "find_or_create_contact", "effingham_maids"),
    ("atlas_brain/api/b2b_vendor_briefing.py", "find_or_create_contact", "churnsignals"),
    ("extracted_competitive_intelligence/api/b2b_vendor_briefing.py",
     "find_or_create_contact", "churnsignals"),
]

EOM_INBOUND_DELEGATES = [
    "atlas_brain/autonomous/tasks/gmail_digest.py",
]


def _calls_named(path: str, callee: str):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr == callee)
                or (isinstance(node.func, ast.Name) and node.func.id == callee)
            )
        ):
            yield node


def test_every_stamped_writer_passes_business_context_id():
    for path, callee, expected in WRITER_SITES:
        calls = list(_calls_named(path, callee))
        assert calls, f"{path}: no {callee} call found"
        for call in calls:
            kw = {k.arg: k for k in call.keywords}
            assert "business_context_id" in kw, f"{path}: {callee} missing tenant stamp"
            assert expected in ast.unparse(kw["business_context_id"].value), path


def test_eom_inbound_writers_delegate_to_the_lead_safe_resolver():
    for path in EOM_INBOUND_DELEGATES:
        calls = [
            *list(_calls_named(path, "resolve_or_create_eom_inbound_lead")),
            *list(_calls_named(path, "resolve_or_create_eom_inbound_lead_and_log_interaction")),
        ]
        assert calls, f"{path}: no EOM inbound resolver call found"
        for call in calls:
            kwargs = {keyword.arg: keyword for keyword in call.keywords}
            assert {"source", "source_ref"} <= kwargs.keys(), path


def test_dict_style_writers_carry_context_key():
    # email_intake + calendar import build dict payloads for create_contact
    intake = (REPO / "atlas_brain/autonomous/tasks/email_intake.py").read_text(encoding="utf-8")
    assert '"business_context_id": "churnsignals"' in intake
    calendar = (REPO / "scripts/import_calendar_contacts.py").read_text(encoding="utf-8")
    assert '"business_context_id": "effingham_maids"' in calendar


@pytest.mark.asyncio
async def test_generic_contact_phone_search_uses_exact_last10_for_full_phone(
):
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class RecordingPool:
        query = ""
        params = ()

        async def fetch(self, query, *params):
            self.query = query
            self.params = params
            return []

    pool = RecordingPool()

    await DatabaseCRMProvider(pool=pool).search_contacts(phone="(217) 555-0100")

    assert " LIKE " not in pool.query
    assert (
        "RIGHT(REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g'), 10)"
        in pool.query
    )
    assert pool.params[0] == "2175550100"


@pytest.mark.asyncio
async def test_generic_contact_phone_search_keeps_partial_phone_substring_lookup(
):
    from atlas_brain.services.crm_provider import DatabaseCRMProvider

    class RecordingPool:
        query = ""
        params = ()

        async def fetch(self, query, *params):
            self.query = query
            self.params = params
            return []

    pool = RecordingPool()

    await DatabaseCRMProvider(pool=pool).search_contacts(phone="5550100")

    assert "REGEXP_REPLACE(COALESCE(phone, ''), '[^0-9]', '', 'g') LIKE" in pool.query
    assert pool.params[0] == "%5550100%"
