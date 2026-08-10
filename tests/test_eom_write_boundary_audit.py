"""Slice 0F: the write-boundary monitor must fire, and must stay quiet.

Every signal is asserted in BOTH directions. A monitor exercised only against
clean data proves nothing about the thing it exists to catch, and would ship as
false assurance -- which is exactly how the defect this slice watches for
survived in the first place.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import quote

import asyncpg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "eom_write_boundary_audit", REPO_ROOT / "scripts" / "eom_write_boundary_audit.py"
)
audit = importlib.util.module_from_spec(_SPEC)
sys.modules["eom_write_boundary_audit"] = audit
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit)

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_eom_lead_conversion_integration import (  # noqa: E402
    _database_url_or_skip,
    _prepare_schema,
)


# --- signal evaluation -------------------------------------------------------


def _signals(atlas=(0, 0, 0), tracker=(0, 0), atlas_error=None, tracker_error=None):
    return audit.build_signals(
        (list(atlas) if atlas is not None else None, atlas_error),
        (list(tracker) if tracker is not None else None, tracker_error),
    )


def test_a_clean_reading_reports_ok():
    result = _signals()
    assert result.ok
    assert result.breaches == []


@pytest.mark.parametrize(
    ("atlas", "tracker", "expected"),
    [
        ((1, 0, 0), (0, 0), "atlas_unknown_source"),
        ((0, 1, 0), (0, 0), "atlas_null_tenant"),
        ((0, 0, 1), (0, 0), "atlas_operator_provenance_without_event"),
        ((0, 0, 0), (1, 0), "tracker_unlinked_customers"),
        ((0, 0, 0), (0, 1), "tracker_stale_pending_reservations"),
    ],
)
def test_each_signal_breaches_on_its_own_violation(atlas, tracker, expected):
    result = _signals(atlas=atlas, tracker=tracker)
    assert not result.ok
    assert [signal.name for signal in result.breaches] == [expected]


def test_an_unreadable_source_breaches_rather_than_reporting_clean():
    """Losing a data source must alert, not go quiet.

    A monitor that reports clean while seeing nothing is worse than none: it
    converts an outage into false assurance.
    """
    result = _signals(tracker=None, tracker_error="render CLI exited 1")
    assert not result.ok
    names = [signal.name for signal in result.breaches]
    assert names == ["tracker_unlinked_customers", "tracker_stale_pending_reservations"]
    assert all(signal.unmeasured for signal in result.breaches)
    assert "render CLI exited 1" in result.report()

    atlas_down = _signals(atlas=None, atlas_error="psql not found")
    assert len(atlas_down.breaches) == 3


# --- output parsing ----------------------------------------------------------


def test_partial_output_is_refused_rather_than_read_as_a_low_count():
    """A short row must not become a healthy-looking number."""
    counts, error = audit._parse_counts("3|4\n", 3)
    assert counts is None
    assert error

    counts, error = audit._parse_counts("not|a|number\n", 3)
    assert counts is None
    assert error


def test_a_well_formed_row_parses():
    counts, error = audit._parse_counts("\n 1 | 2 | 3 \n", 3)
    assert error is None
    assert counts == [1, 2, 3]


# --- alert state machine -----------------------------------------------------


def test_first_breach_alerts_then_stays_quiet_until_the_reminder():
    state, alert = audit.decide_alert({}, breached=True, realert_every=3)
    assert alert == "breach"
    assert state == {"breached": True, "consecutive": 1}

    state, alert = audit.decide_alert(state, breached=True, realert_every=3)
    assert alert is None
    state, alert = audit.decide_alert(state, breached=True, realert_every=3)
    assert alert == "reminder"
    assert state["consecutive"] == 3


def test_recovery_notifies_exactly_once():
    state, _ = audit.decide_alert({}, breached=True, realert_every=3)
    state, alert = audit.decide_alert(state, breached=False, realert_every=3)
    assert alert == "recovered"
    state, alert = audit.decide_alert(state, breached=False, realert_every=3)
    assert alert is None


def test_a_configuration_that_could_never_alert_is_refused():
    """A mistyped interval or blank topic must not leave a mute monitor running."""
    with pytest.raises(ValueError, match="negative"):
        audit.validate_settings(-1, "topic")
    with pytest.raises(ValueError, match="blank"):
        audit.validate_settings(24, "   ")
    audit.validate_settings(0, "topic")  # 0 disables the reminder, deliberately


def test_a_corrupt_state_file_is_reported_not_hidden(tmp_path):
    """Losing alert memory is safe but must be said out loud."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{not json", encoding="utf-8")
    previous, warning = audit.read_state(state_path)
    assert previous == {}
    assert warning and "corrupt" in warning

    state_path.write_text('["not", "an", "object"]', encoding="utf-8")
    previous, warning = audit.read_state(state_path)
    assert previous == {}
    assert warning

    missing, warning = audit.read_state(tmp_path / "absent.json")
    assert missing == {} and warning is None, "a first run is not a warning"


def test_a_clean_run_from_cold_state_says_nothing():
    state, alert = audit.decide_alert({}, breached=False, realert_every=3)
    assert alert is None
    assert state == {"breached": False, "consecutive": 0}


def test_main_alerts_on_breach_and_exits_non_zero(monkeypatch, tmp_path):
    sent: list[tuple] = []
    monkeypatch.setattr(audit, "query_atlas", lambda *a, **k: ([1, 0, 0], None))
    monkeypatch.setattr(audit, "query_tracker", lambda *a, **k: ([0, 0], None))

    code = audit.main(
        ["--state-dir", str(tmp_path)],
        notifier=lambda *args: sent.append(args),
    )
    assert code == 1
    assert len(sent) == 1
    assert "breached" in sent[0][2].lower()
    assert "atlas_unknown_source" in sent[0][3]


def test_main_stays_silent_and_exits_zero_when_clean(monkeypatch, tmp_path):
    sent: list[tuple] = []
    monkeypatch.setattr(audit, "query_atlas", lambda *a, **k: ([0, 0, 0], None))
    monkeypatch.setattr(audit, "query_tracker", lambda *a, **k: ([0, 0], None))

    code = audit.main(
        ["--state-dir", str(tmp_path)],
        notifier=lambda *args: sent.append(args),
    )
    assert code == 0
    assert sent == []


# --- the SQL itself, against the real schema ---------------------------------


async def _seed_contact(conn, **kwargs) -> uuid.UUID:
    contact_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO contacts (
            id, full_name, business_context_id, contact_type, status, source, metadata
        ) VALUES ($1, $2, $3, 'customer', 'active', $4, $5::jsonb)
        """,
        contact_id,
        kwargs.get("full_name", "Seeded"),
        kwargs.get("business_context_id", "effingham_maids"),
        kwargs.get("source", "web"),
        kwargs.get("metadata", "{}"),
    )
    return contact_id


@pytest.mark.asyncio
async def test_the_atlas_query_detects_each_violation_and_ignores_clean_rows():
    """Both directions on the real schema, not a hand-built stand-in."""
    database_url = _database_url_or_skip()
    schema = f"eom_write_boundary_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(
            conn, schema, apply_privilege_migration=False
        )

        # A well-formed contact: known source, tenant set, and -- because it
        # carries operator provenance -- a matching lifecycle event.
        clean_id = await _seed_contact(
            conn,
            source="manual",
            metadata='{"eom_operator_contact_sources": {"time_tracker:x": {}}}',
        )
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key, metadata
            ) VALUES ($1, 'contact_created', 'employee:1:Juan', 'eom_office', $2, '{}'::jsonb)
            """,
            clean_id,
            f"op-{uuid.uuid4().hex}",
        )

        dsn = (
            f"{database_url}?options={quote(f'-csearch_path={schema},public')}"
            if "?" not in database_url
            else database_url
        )
        counts, error = audit.query_atlas(os.environ.get("EOM_AUDIT_PSQL_BIN", "psql"), dsn)
        assert error is None, error
        assert counts == [0, 0, 0], "a well-formed contact must not trip any signal"

        # Now one violation of each kind.
        await _seed_contact(conn, source="rogue_writer")
        await _seed_contact(conn, business_context_id=None)
        await _seed_contact(
            conn,
            source="manual",
            metadata='{"eom_operator_contact_sources": {"time_tracker:y": {}}}',
        )

        counts, error = audit.query_atlas(os.environ.get("EOM_AUDIT_PSQL_BIN", "psql"), dsn)
        assert error is None, error
        assert counts == [1, 1, 1]

        result = audit.build_signals((counts, None), ([0, 0], None))
        assert {signal.name for signal in result.breaches} == {
            "atlas_unknown_source",
            "atlas_null_tenant",
            "atlas_operator_provenance_without_event",
        }
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()
