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
    SQL_BACKFILL_EOM_BY_APPOINTMENT,
    SQL_BACKFILL_EOM_BY_SOURCE,
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
    for sql in (SQL_BACKFILL_EOM_BY_SOURCE, SQL_BACKFILL_EOM_BY_APPOINTMENT):
        assert "business_context_id IS NULL" in sql


def test_appointment_backfill_requires_tenant_stamped_appointment():
    assert "a.business_context_id = $1" in SQL_BACKFILL_EOM_BY_APPOINTMENT


# ---------------------------------------------------------------------------
# Writer stamps: every previously NULL-context CRM write now carries a
# business_context_id argument. AST-verified so refactors that drop the
# kwarg fail loudly (and so this test can't pass on comment text alone).
# ---------------------------------------------------------------------------

WRITER_SITES = [
    # (file, callee substring, expected context expression substring)
    ("atlas_brain/tools/scheduling.py", "find_or_create_contact", "context.id"),
    ("atlas_brain/autonomous/tasks/gmail_digest.py", "find_or_create_contact", "effingham_maids"),
    ("atlas_brain/autonomous/tasks/email_backfill.py", "find_or_create_contact", "effingham_maids"),
    ("atlas_brain/api/b2b_vendor_briefing.py", "find_or_create_contact", "churnsignals"),
]


def _calls_with_kwarg(path: str, callee: str):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == callee
        ):
            yield node


def test_every_stamped_writer_passes_business_context_id():
    for path, callee, expected in WRITER_SITES:
        calls = list(_calls_with_kwarg(path, callee))
        assert calls, f"{path}: no {callee} call found"
        for call in calls:
            kw = {k.arg: k for k in call.keywords}
            assert "business_context_id" in kw, f"{path}: {callee} missing tenant stamp"
            assert expected in ast.unparse(kw["business_context_id"].value), path


def test_dict_style_writers_carry_context_key():
    # email_intake + calendar import build dict payloads for create_contact
    intake = (REPO / "atlas_brain/autonomous/tasks/email_intake.py").read_text(encoding="utf-8")
    assert '"business_context_id": "churnsignals"' in intake
    calendar = (REPO / "scripts/import_calendar_contacts.py").read_text(encoding="utf-8")
    assert '"business_context_id": "effingham_maids"' in calendar
