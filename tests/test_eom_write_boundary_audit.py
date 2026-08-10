"""Slice 0F: the write-boundary monitor must fire, and must stay quiet.

Every signal is asserted in BOTH directions. A monitor exercised only against
clean data proves nothing about the thing it exists to catch, and would ship as
false assurance -- which is exactly how the defect this slice watches for
survived in the first place.
"""
from __future__ import annotations

import importlib.util
import itertools
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
        ["--state-dir", str(tmp_path), "--ntfy-topic", "t"],
        notifier=lambda *args: (sent.append(args), True)[1],
    )
    assert code == audit.EXIT_BREACH == 2, "a breach must not share exit 1 with a crash"
    assert len(sent) == 1
    assert "breached" in sent[0][2].lower()
    assert "atlas_unknown_source" in sent[0][3]


def test_main_stays_silent_and_exits_zero_when_clean(monkeypatch, tmp_path):
    sent: list[tuple] = []
    monkeypatch.setattr(audit, "query_atlas", lambda *a, **k: ([0, 0, 0], None))
    monkeypatch.setattr(audit, "query_tracker", lambda *a, **k: ([0, 0], None))

    code = audit.main(
        ["--state-dir", str(tmp_path), "--ntfy-topic", "t"],
        notifier=lambda *args: (sent.append(args), True)[1],
    )
    assert code == 0
    assert sent == []


def test_an_undelivered_alert_does_not_advance_state(monkeypatch, tmp_path):
    """A failed push must not record the breach as notified.

    Otherwise the re-alert interval swallows every following run and the
    monitor has silently stopped alerting -- the failure this slice exists to
    make impossible.
    """
    monkeypatch.setattr(audit, "query_atlas", lambda *a, **k: ([1, 0, 0], None))
    monkeypatch.setattr(audit, "query_tracker", lambda *a, **k: ([0, 0], None))
    attempts: list[tuple] = []

    def _failing(*args):
        attempts.append(args)
        return False

    for _ in range(3):
        audit.main(["--state-dir", str(tmp_path), "--ntfy-topic", "t"], notifier=_failing)

    # Every run retried the first-breach alert rather than falling silent.
    assert len(attempts) == 3
    assert not (tmp_path / "state.json").exists()

    delivered: list[tuple] = []
    audit.main(
        ["--state-dir", str(tmp_path), "--ntfy-topic", "t"],
        notifier=lambda *args: (delivered.append(args), True)[1],
    )
    assert len(delivered) == 1
    assert audit.read_state(tmp_path / "state.json")[0]["consecutive"] == 1


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


@pytest.mark.asyncio
async def test_an_unrelated_lifecycle_row_does_not_excuse_a_bypass():
    """A lead_created row must not be mistaken for operator-tier evidence.

    Migration 351 gives every EOM lead a lifecycle row, so correlating on "any
    row" would let a bypass that adds operator provenance to an existing
    contact report clean -- the audit would be measuring the wrong invariant.
    """
    database_url = _database_url_or_skip()
    schema = f"eom_wb_lifecycle_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await _prepare_schema(conn, schema, apply_privilege_migration=False)
        contact_id = await _seed_contact(
            conn,
            source="manual",
            metadata='{"eom_operator_contact_sources": {"time_tracker:z": {}}}',
        )
        # The kind of row an ordinary lead already has, and which says nothing
        # about the operator boundary.
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key, metadata
            ) VALUES ($1, 'lead_created', 'system', 'eom_office', $2, '{}'::jsonb)
            """,
            contact_id,
            f"op-{uuid.uuid4().hex}",
        )

        dsn = f"{database_url}?options={quote(f'-csearch_path={schema},public')}"
        counts, error = audit.query_atlas(os.environ.get("EOM_AUDIT_PSQL_BIN", "psql"), dsn)
        assert error is None, error
        assert counts[2] == 1, "an unrelated lifecycle row must not excuse the bypass"

        # Adding the operator event clears it.
        await conn.execute(
            """
            INSERT INTO eom_lead_lifecycle_events (
                contact_id, event_type, actor, source, operation_key, metadata
            ) VALUES ($1, 'contact_updated', 'employee:1:Juan', 'eom_office', $2, '{}'::jsonb)
            """,
            contact_id,
            f"op-{uuid.uuid4().hex}",
        )
        counts, error = audit.query_atlas(os.environ.get("EOM_AUDIT_PSQL_BIN", "psql"), dsn)
        assert error is None, error
        assert counts[2] == 0
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


@pytest.mark.asyncio
async def test_the_tracker_sql_detects_each_violation_against_the_tracker_schema():
    """The tracker predicates run against tracker-shaped tables, not injected counts.

    Injecting `[1, 0]` into build_signals proves the plumbing, not the SQL: a
    wrong column, state value, or stale-time predicate would ship undetected.
    """
    database_url = _database_url_or_skip()
    schema = f"eom_wb_tracker_{uuid.uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        # The columns the audit predicates read, matching eom-timetracker's
        # backend/time_tracker_api.py schema for these two tables.
        await conn.execute(
            """
            CREATE TABLE customers (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                atlas_contact_id UUID,
                active BOOLEAN NOT NULL DEFAULT TRUE
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE eom_customer_atlas_reservations (
                id UUID PRIMARY KEY,
                state VARCHAR(16) NOT NULL DEFAULT 'pending',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        async def counts() -> list[int]:
            row = await conn.fetchrow(audit._tracker_sql())
            return [row[0], row[1]]

        # A linked customer and a fresh pending reservation are both fine.
        await conn.execute(
            "INSERT INTO customers (name, atlas_contact_id) VALUES ('Linked', $1)",
            uuid.uuid4(),
        )
        await conn.execute(
            "INSERT INTO eom_customer_atlas_reservations (id, state, updated_at) "
            "VALUES ($1, 'pending', NOW())",
            uuid.uuid4(),
        )
        assert await counts() == [0, 0], "healthy rows must not trip either signal"

        # A finalized reservation, however old, is also fine.
        await conn.execute(
            "INSERT INTO eom_customer_atlas_reservations (id, state, updated_at) "
            "VALUES ($1, 'finalized', NOW() - INTERVAL '3 days')",
            uuid.uuid4(),
        )
        assert await counts() == [0, 0]

        # Now one of each violation, planted independently.
        await conn.execute("INSERT INTO customers (name) VALUES ('Atlas has never heard of me')")
        assert await counts() == [1, 0]

        await conn.execute(
            "INSERT INTO eom_customer_atlas_reservations (id, state, updated_at) "
            "VALUES ($1, 'pending', NOW() - INTERVAL '3 days')",
            uuid.uuid4(),
        )
        assert await counts() == [1, 1]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


# --- guard class-closure: generative property test over open input -----------
#
# `_parse_counts` is a guard over an OPEN input space: it admits or refuses
# arbitrary text produced by an external command. Fixture cases would only prove
# the listed inputs; the class stays open (docs/GUARD_CLASS_CLOSURE.md req 3).
# So the test is derived from the grammar of that input -- tokens x containers x
# families -- and checked against an oracle written from the SPEC, independently
# of the implementation.

_TOKENS = (
    "0", "7", "-3", "+5", " 12 ",      # parse as int
    "x", "", "1.5", "0x10", "None",    # do not
)
_PADDINGS = ("", " ", "\t")
_PREFIXES = ("", "\n", "   \n", "junk\n", "1|2|3|4|5\n")
_SUFFIXES = ("", "\n", "\ntrailing", "\n9|9")


def _oracle(raw: str, expected: int) -> tuple[list[int] | None, bool]:
    """The specification, restated independently of the implementation.

    Spec: scan lines in order; the FIRST non-blank line whose '|'-split yields
    exactly `expected` parts decides the outcome. If every part of that row
    parses as an integer the counts are returned, otherwise the read fails.
    A row of the wrong arity is not that row. No matching row is a failure.
    """
    for line in (raw or "").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        parts = [part.strip() for part in candidate.split("|")]
        if len(parts) != expected:
            continue
        try:
            return [int(part) for part in parts], False
        except ValueError:
            return None, True
    return None, True


@pytest.mark.parametrize("expected", (2, 3))
def test_parse_counts_matches_its_spec_across_the_input_grammar(expected):
    """Walk the product of the grammar axes: tokens x containers x families."""
    arities = [n for n in (expected - 1, expected, expected + 1) if n >= 1]
    checked = 0
    for arity, token, pad, prefix, suffix in itertools.product(
        arities, _TOKENS, _PADDINGS, _PREFIXES, _SUFFIXES
    ):
        row = "|".join(f"{pad}{token}{pad}" for _ in range(arity))
        raw = f"{prefix}{row}{suffix}"
        counts, error = audit._parse_counts(raw, expected)
        want_counts, want_error = _oracle(raw, expected)

        assert counts == want_counts, raw
        assert bool(error) == want_error, raw
        # Invariants that must hold for every input in the class: a returned
        # reading is always complete, and a refusal always says why. A short
        # row must never become counts.
        assert counts is None or len(counts) == expected
        assert (counts is None) == bool(error)
        checked += 1
    assert checked > 500, "the grammar product should be broad, not a fixture list"
