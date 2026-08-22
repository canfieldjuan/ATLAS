"""Durable EOM missed-call recovery proof.

All addresses in this module use the reserved ``example.test`` domain.  The
worker receives fake gateways only; no test can call a live email provider or
schedule a real appointment.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi import FastAPI

asyncpg = pytest.importorskip("asyncpg")

from atlas_brain.eom_api import funnel as funnel_mod  # noqa: E402
from atlas_brain.eom_api import funnel_auth as funnel_auth_mod  # noqa: E402
from atlas_brain.eom_api.config import EOMFunnelConfig  # noqa: E402
from atlas_brain.services import eom_missed_call_recovery as recovery_mod  # noqa: E402
from atlas_brain.services.eom_missed_call_recovery import (  # noqa: E402
    EOMMissedCallRecoveryConflictError,
    EOMMissedCallRecoveryService,
    EOMMissedCallRecoveryUnavailableError,
    ResendMissedCallRecoveryGateway,
    _AmbiguousDeliveryError,
    _DefiniteDeliveryError,
    prepare_eom_missed_call_recovery_worker,
    _third_step_due,
    next_business_day_due,
)
from atlas_brain.templates.email.estimate_confirmation import (  # noqa: E402
    BUSINESS_NAME,
    BUSINESS_PHONE,
)
from atlas_brain.templates.email.missed_call_recovery import (  # noqa: E402
    render_missed_call_recovery_email,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"
DATABASE_URL_ENV = "ATLAS_MIGRATION_TEST_DATABASE_URL"
_EOM_CONTEXT = "effingham_maids"
_BOOKING_LINK = "https://calendar.app.google/eom-test-booking"
_NOW = datetime(2026, 8, 21, 15, 0, tzinfo=timezone.utc)


class _ConnectionPool:
    """Give a real asyncpg connection the small pool API this service needs."""

    is_initialized = True

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @asynccontextmanager
    async def transaction(self):
        async with self._connection.transaction():
            yield self._connection

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        return await self._connection.fetch(*args, **kwargs)

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:
        return await self._connection.fetchrow(*args, **kwargs)

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        return await self._connection.fetchval(*args, **kwargs)


class _FakeGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def send(self, **kwargs: str) -> str:
        self.calls.append(dict(kwargs))
        return f"test-resend-{len(self.calls)}"


class _BlockingGateway(_FakeGateway):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send(self, **kwargs: str) -> str:
        self.calls.append(dict(kwargs))
        self.started.set()
        await self.release.wait()
        return f"test-resend-{len(self.calls)}"


class _RetryingGateway(_FakeGateway):
    async def send(self, **kwargs: str) -> str:
        self.calls.append(dict(kwargs))
        raise _DefiniteDeliveryError("test_transport_rejected", retryable=True)


class _History:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return object()


def _database_url_or_skip() -> str:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} is not configured")
    return database_url


async def _apply_schema(connection: Any, schema: str) -> None:
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    # 035 keeps the FK additive for existing Atlas schema, but a focused empty
    # migration schema needs the referenced base relation before it runs.
    await connection.execute("CREATE TABLE appointments (id UUID PRIMARY KEY)")
    for name in (
        "035_contacts.sql",
        "256_contact_interaction_dedupe.sql",
        "346_contact_lead_pipeline.sql",
        "351_eom_lead_lifecycle_events.sql",
        "363_eom_lead_lifecycle_sequence.sql",
        "366_contacts_customer_type.sql",
        "389_eom_missed_call_recovery.sql",
    ):
        await connection.execute((MIGRATIONS / name).read_text())


@asynccontextmanager
async def _test_store():
    database_url = _database_url_or_skip()
    schema = f"atlas_eom_missed_call_{uuid4().hex}"
    connection = await asyncpg.connect(database_url)
    try:
        await _apply_schema(connection, schema)
        yield _ConnectionPool(connection), schema
    finally:
        try:
            await connection.execute("RESET search_path")
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await connection.close()


async def _schema_connection(database_url: str, schema: str) -> _ConnectionPool:
    connection = await asyncpg.connect(database_url)
    await connection.execute(f'SET search_path TO "{schema}", public')
    return _ConnectionPool(connection)


async def _close_schema_connection(pool: _ConnectionPool) -> None:
    await pool._connection.close()


def _config(
    *, enabled: bool = True, booking_link: str = _BOOKING_LINK
) -> EOMFunnelConfig:
    return EOMFunnelConfig(
        api_enabled=True,
        missed_call_recovery_enabled=enabled,
        missed_call_booking_link=booking_link,
        missed_call_timezone="America/Chicago",
        missed_call_poll_interval_seconds=30,
        missed_call_max_delivery_attempts=3,
        missed_call_delivery_timeout_seconds=5,
    )


def _operation_key(prefix: str = "missed-call") -> str:
    return f"{prefix}-{uuid4().hex}"


async def _insert_estimate_lead(
    pool: _ConnectionPool,
    *,
    contact_id: UUID | None = None,
    ack_variant: str = "residential",
    email: str = "lead@example.test",
) -> UUID:
    contact_id = contact_id or uuid4()
    now = _NOW - timedelta(minutes=2)
    await pool._connection.execute(
        """
        INSERT INTO contacts (
            id, full_name, email, business_context_id, contact_type, status,
            lead_stage, source, customer_type, created_at
        ) VALUES ($1, 'Recovery Test Lead', $2, $3, 'lead', 'active', 'new',
                  'web', 'unknown', $4)
        """,
        contact_id,
        email,
        _EOM_CONTEXT,
        now,
    )
    await pool._connection.execute(
        """
        INSERT INTO contact_interactions (
            id, contact_id, interaction_type, summary, intent, occurred_at,
            metadata, interaction_dedupe_key
        ) VALUES ($1, $2, 'web_form', 'Controlled estimate request',
                  'estimate_request', $3, $4::jsonb, $5)
        """,
        uuid4(),
        contact_id,
        now,
        json.dumps(
            {
                "submitted_email": email,
                "ack_variant": ack_variant,
                "service": "residential cleaning",
            }
        ),
        f"estimate-{contact_id}",
    )
    return contact_id


def _service(
    pool: _ConnectionPool,
    *,
    gateway: Any | None = None,
    now_box: dict[str, datetime] | None = None,
    history: Any | None = None,
    config: EOMFunnelConfig | None = None,
) -> EOMMissedCallRecoveryService:
    clock = now_box if now_box is not None else {"now": _NOW}
    return EOMMissedCallRecoveryService(
        pool=pool,
        config=config or _config(),
        # Tests always use an explicit local gateway; never fall through to
        # process-level email credentials even if a developer has them loaded.
        gateway=gateway if gateway is not None else _FakeGateway(),
        now=lambda: clock["now"],
        email_history=history,
    )


@pytest.mark.asyncio
async def test_real_resend_gateway_preserves_provider_idempotency_and_outcome_classes(
    monkeypatch,
) -> None:
    """Exercise the real adapter with a transport seam, never Resend itself."""

    from atlas_brain.config import settings

    monkeypatch.setattr(settings.email, "enabled", True)
    monkeypatch.setattr(settings.email, "api_key", "test-resend-key")
    request_key = _operation_key("provider-key")
    seen: list[httpx.Request] = []

    async def accepted(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json={"id": "resend-message-test"})

    gateway = ResendMissedCallRecoveryGateway(
        timeout_seconds=5,
        transport=httpx.MockTransport(accepted),
    )
    assert (
        await gateway.send(
            recipient_email="lead@example.test",
            subject="Controlled test",
            body="Controlled test body",
            idempotency_key=request_key,
        )
        == "resend-message-test"
    )
    assert len(seen) == 1
    assert seen[0].url == httpx.URL("https://api.resend.com/emails")
    assert seen[0].headers["Idempotency-Key"] == request_key

    async def rejected(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"name": "validation_error"})

    with pytest.raises(_DefiniteDeliveryError) as rejected_error:
        await ResendMissedCallRecoveryGateway(
            timeout_seconds=5,
            transport=httpx.MockTransport(rejected),
        ).send(
            recipient_email="lead@example.test",
            subject="Controlled test",
            body="Controlled test body",
            idempotency_key=_operation_key("provider-rejected"),
        )
    assert rejected_error.value.code == "resend_rejected"
    assert rejected_error.value.retryable is False

    async def ambiguous(_request: httpx.Request) -> httpx.Response:
        # A 2xx without Resend's stable message identity cannot prove delivery.
        return httpx.Response(200, json={})

    with pytest.raises(
        _AmbiguousDeliveryError, match="resend_success_identity_unknown"
    ):
        await ResendMissedCallRecoveryGateway(
            timeout_seconds=5,
            transport=httpx.MockTransport(ambiguous),
        ).send(
            recipient_email="lead@example.test",
            subject="Controlled test",
            body="Controlled test body",
            idempotency_key=_operation_key("provider-ambiguous"),
        )


@pytest.mark.asyncio
async def test_worker_startup_fences_enabled_disabled_and_blocked_configurations(
    monkeypatch,
) -> None:
    """One shared startup boundary is explicit about every safe branch."""

    events: list[str] = []
    ready = {"value": True}
    worker = object()

    class _StartupRecovery:
        def __init__(self, *, pool: object, config: EOMFunnelConfig) -> None:
            self._config = config

        def delivery_block_reason(self) -> str | None:
            return self._config.missed_call_recovery_delivery_block_reason

        async def block_active_sequences_for_configuration(self, *, reason: str) -> int:
            events.append(f"block:{reason}")
            return 2

    async def schema_ready(_pool: object) -> bool:
        return ready["value"]

    def start_worker(*, pool: object, config: EOMFunnelConfig) -> object:
        assert pool is pool_token
        assert config.missed_call_recovery_enabled is True
        events.append("start")
        return worker

    pool_token = object()
    monkeypatch.setattr(recovery_mod, "EOMMissedCallRecoveryService", _StartupRecovery)
    monkeypatch.setattr(recovery_mod, "missed_call_recovery_schema_ready", schema_ready)
    monkeypatch.setattr(
        recovery_mod, "start_eom_missed_call_recovery_worker", start_worker
    )

    assert (
        await prepare_eom_missed_call_recovery_worker(
            pool=pool_token, config=_config(enabled=False)
        )
        is None
    )
    assert (
        await prepare_eom_missed_call_recovery_worker(
            pool=pool_token, config=_config(booking_link="")
        )
        is None
    )
    assert (
        await prepare_eom_missed_call_recovery_worker(pool=pool_token, config=_config())
        is worker
    )
    assert events == [
        "block:recovery_disabled",
        "block:booking_link_unavailable",
        "start",
    ]

    ready["value"] = False
    with pytest.raises(
        EOMMissedCallRecoveryUnavailableError, match="schema is unavailable"
    ):
        await prepare_eom_missed_call_recovery_worker(pool=pool_token, config=_config())


@pytest.mark.asyncio
async def test_slim_eom_lifespan_wires_and_stops_the_missed_call_worker(
    monkeypatch,
) -> None:
    """The Render EOM entrypoint reaches the shared startup boundary."""

    from atlas_brain import main_eom

    events: list[str] = []
    worker = object()
    pool = SimpleNamespace(is_initialized=True)

    async def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    async def prepare(*, pool: object, config: EOMFunnelConfig) -> object:
        assert pool is not None
        assert config.missed_call_recovery_enabled is True
        events.append("prepare")
        return worker

    async def stop(value: object) -> None:
        assert value is worker
        events.append("stop")

    monkeypatch.setattr(main_eom, "funnel_settings", _config())
    monkeypatch.setattr(main_eom, "invoicing_settings", SimpleNamespace())
    monkeypatch.setattr(
        main_eom,
        "eom_profile_settings",
        SimpleNamespace(run_migrations=False, canonical_crm_database_confirmed=True),
    )
    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=False))
    monkeypatch.setattr(
        main_eom, "validate_receivables_api_config", lambda _config: None
    )
    monkeypatch.setattr(
        main_eom, "validate_eom_funnel_api_config", lambda _config: None
    )
    monkeypatch.setattr(
        main_eom,
        "validate_eom_funnel_canonical_crm_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(main_eom, "init_eom_funnel_database", no_op)
    monkeypatch.setattr(main_eom, "close_eom_funnel_database", no_op)
    monkeypatch.setattr(main_eom, "_validate_eom_funnel_startup", no_op)
    monkeypatch.setattr(main_eom, "get_eom_funnel_db_pool", lambda: pool)
    monkeypatch.setattr(
        recovery_mod, "prepare_eom_missed_call_recovery_worker", prepare
    )
    monkeypatch.setattr(recovery_mod, "stop_eom_missed_call_recovery_worker", stop)

    async with main_eom.lifespan(FastAPI()):
        assert events == ["prepare"]
    assert events == ["prepare", "stop"]


@pytest.mark.asyncio
async def test_full_atlas_lifespan_wires_and_stops_the_missed_call_worker(
    monkeypatch,
) -> None:
    """The deployed aggregate entrypoint reaches the same startup boundary."""

    from atlas_brain import main
    from atlas_brain.eom_api import config as funnel_config_mod

    events: list[str] = []
    worker = object()
    pool = SimpleNamespace(is_initialized=True)

    async def no_op(*_args: object, **_kwargs: object) -> None:
        return None

    async def prepare(*, pool: object, config: EOMFunnelConfig) -> object:
        assert pool is not None
        assert config.missed_call_recovery_enabled is True
        events.append("prepare")
        return worker

    async def stop(value: object) -> None:
        assert value is worker
        events.append("stop")

    runtime_settings = main.settings.model_copy(deep=True)
    runtime_settings.load_llm_on_startup = False
    runtime_settings.llm.model_swap_enabled = False
    runtime_settings.llm.cloud_enabled = False
    runtime_settings.intent_router.llm_fallback_enabled = False
    runtime_settings.email_draft.enabled = False
    runtime_settings.email_draft.triage_enabled = False
    runtime_settings.reasoning.enabled = False
    runtime_settings.discovery.enabled = False
    runtime_settings.alerts.enabled = False
    runtime_settings.reminder.enabled = False
    runtime_settings.autonomous.enabled = False
    runtime_settings.mqtt.enabled = False
    runtime_settings.tools.calendar_enabled = False
    runtime_settings.mcp.client_enabled = False
    runtime_settings.voice.enabled = False
    runtime_settings.invoicing.enabled = False
    runtime_settings.invoicing.receivables_api_enabled = False
    runtime_settings.invoicing.auto_invoice_enabled = False
    runtime_settings.invoicing.receivables_service_token = ""

    monkeypatch.setattr(main, "settings", runtime_settings)
    monkeypatch.setattr(main, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(
        main, "_enforce_paid_funnel_alert_channel", lambda _settings: None
    )
    monkeypatch.setattr(main, "init_database", no_op)
    monkeypatch.setattr(main, "close_database", no_op)
    monkeypatch.setattr(main, "get_db_pool", lambda: pool)
    monkeypatch.setattr(main, "_run_database_migration_check", no_op)
    monkeypatch.setattr(main, "_validate_eom_funnel_startup", no_op)
    monkeypatch.setattr(funnel_config_mod, "funnel_settings", _config())
    monkeypatch.setattr(
        recovery_mod, "prepare_eom_missed_call_recovery_worker", prepare
    )
    monkeypatch.setattr(recovery_mod, "stop_eom_missed_call_recovery_worker", stop)

    async with main.lifespan(FastAPI()):
        assert events == ["prepare"]
    assert events == ["prepare", "stop"]


def test_recovery_templates_match_approved_copy_exactly() -> None:
    first = render_missed_call_recovery_email(
        step_number=1,
        full_name="Ada Lovelace",
        booking_link=_BOOKING_LINK,
    )
    assert first.subject == "I tried reaching you about your cleaning estimate"
    assert (
        first.body
        == f"""Hi Ada,

I just tried calling you about your residential cleaning estimate.

Whenever you have a moment, you can call or text me at {BUSINESS_PHONE}. You can also reply to this email or request a convenient estimate time here:

{_BOOKING_LINK}

Requested times are not confirmed until we verify the appointment with you by phone or text.

Thanks,
Juan
{BUSINESS_NAME}
{BUSINESS_PHONE}
"""
    )

    second = render_missed_call_recovery_email(
        step_number=2,
        full_name="Ada Lovelace",
        booking_link=_BOOKING_LINK,
    )
    assert second.subject == "Still interested in a cleaning estimate?"
    assert (
        second.body
        == f"""Hi Ada,

I wanted to follow up in case we missed each other.

I’d be happy to answer your questions and provide a residential cleaning estimate. You can call or text me at {BUSINESS_PHONE}, reply to this email, or request a time here:

{_BOOKING_LINK}

We’ll confirm the appointment by phone or text before coming out.

Thanks,
Juan
"""
    )

    third = render_missed_call_recovery_email(
        step_number=3,
        full_name="Ada Lovelace",
        booking_link=_BOOKING_LINK,
    )
    assert third.subject == "Should I keep your estimate request open?"
    assert (
        third.body
        == f"""Hi Ada,

I haven’t been able to catch you by phone, so I wanted to check in one last time.

If you’re still looking for cleaning service, reply to this email, call or text {BUSINESS_PHONE}, or request an estimate time here:

{_BOOKING_LINK}

If now isn’t the right time, no problem. You’re welcome to reach out whenever you’re ready.

Requested times are not confirmed until we verify them by phone or text.

Thanks,
Juan
"""
    )


@pytest.mark.parametrize(
    "from_time, expected_date",
    (
        (datetime(2026, 8, 21, 15, tzinfo=timezone.utc), "2026-08-24"),
        (datetime(2026, 8, 22, 15, tzinfo=timezone.utc), "2026-08-24"),
        (datetime(2026, 8, 23, 15, tzinfo=timezone.utc), "2026-08-24"),
    ),
)
def test_next_business_day_is_local_weekday_nine_am(
    from_time: datetime,
    expected_date: str,
) -> None:
    due = next_business_day_due(from_time, timezone_name="America/Chicago")
    local = due.astimezone(ZoneInfo("America/Chicago"))
    assert local.date().isoformat() == expected_date
    assert (local.hour, local.minute, local.weekday()) == (9, 0, 0)


def test_third_step_preserves_eom_local_clock_time_across_dst() -> None:
    # Friday 09:00 CDT becomes Monday 09:00 CST, not Monday 08:00 CST from a
    # raw UTC +72-hour calculation.
    second_step_sent_at = datetime(2026, 10, 30, 14, 0, tzinfo=timezone.utc)

    due = _third_step_due(
        second_step_sent_at,
        timezone_name="America/Chicago",
    )

    assert due == datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)


def test_missing_booking_link_is_a_supported_fail_closed_configuration() -> None:
    config = _config(booking_link="")
    assert config.missed_call_recovery_delivery_is_configured is False
    assert (
        config.missed_call_recovery_delivery_block_reason == "booking_link_unavailable"
    )
    paused = _config(enabled=False)
    assert paused.missed_call_recovery_delivery_is_configured is False
    assert paused.missed_call_recovery_delivery_block_reason == "recovery_disabled"
    with pytest.raises(ValueError, match="Google Calendar"):
        _config(booking_link="https://example.test/not-a-calendar-link")
    with pytest.raises(ValueError, match="Google Calendar"):
        _config(booking_link="https://calendar.google.com")
    with pytest.raises(ValueError, match="Google Calendar"):
        _config(booking_link="https://calendar.google.com:444/appointments")
    with pytest.raises(ValueError, match="requires ATLAS_EOM_FUNNEL_API_ENABLED"):
        EOMFunnelConfig(missed_call_recovery_enabled=True)


def test_booking_link_validator_evidence_gates_the_url_grammar() -> None:
    """Exercise the URL grammar instead of blessing a short allow-list sample.

    The source-of-truth policy is deliberately narrow: only an HTTPS URL at one
    of the two public Google Calendar hosts, with no credentials or explicit
    port and a non-root path, can enter a customer-email template.  Query and
    fragment are permitted because valid public Google Calendar invitation
    links use them in the wild. The independent expected-admission oracle spans
    token/value families (scheme and host), URL authority/path containers, and
    query/fragment keys rather than deriving its verdict from the validator.
    """

    allowed_hosts = {"calendar.google.com", "calendar.app.google"}
    schemes = ("https", "http", "ftp")
    hosts = (
        "calendar.google.com",
        "CALENDAR.APP.GOOGLE",
        "calendar.google.com.evil.test",
        "calendar.google.com.",
    )
    credentials = ("", "operator@", "operator:secret@")
    ports = ("", ":443")
    paths = ("/appointment-request", "/", "")
    suffixes = ("", "?source=eom", "#request")

    for scheme, host, credentials_part, port, path, suffix in product(
        schemes,
        hosts,
        credentials,
        ports,
        paths,
        suffixes,
    ):
        candidate = f"{scheme}://{credentials_part}{host}{port}{path}{suffix}"
        expected_admission = (
            scheme == "https"
            and host.casefold() in allowed_hosts
            and not credentials_part
            and not port
            and bool(path.strip("/"))
        )
        if expected_admission:
            config = _config(booking_link=candidate)
            assert config.missed_call_booking_link == candidate
        else:
            with pytest.raises(ValueError, match="Google Calendar"):
                _config(booking_link=candidate)

    for unsafe_character_variant in (
        "https://calendar.google.com/appointment request",
        "https://calendar.google.com/appointment\\request",
        "https://calendar.google.com/appointment\x00request",
    ):
        with pytest.raises(ValueError, match="control characters|Google Calendar"):
            _config(booking_link=unsafe_character_variant)


@pytest.mark.asyncio
async def test_form_submission_evidence_alone_never_creates_a_recovery_sequence() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)

        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 0
        )
        assert (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM contact_interactions
            WHERE contact_id = $1 AND interaction_type = 'web_form'
            """,
                contact_id,
            )
            == 1
        )


@pytest.mark.asyncio
async def test_qualifying_no_answer_creates_one_sequence_and_replays_idempotently() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        service = _service(pool)
        operation_key = _operation_key()

        created = await service.record_no_answer(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=7,
            actor_name="Juan",
        )
        replay = await service.record_no_answer(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=7,
            actor_name="Juan",
        )

        assert created["idempotent"] is False
        assert replay["idempotent"] is True
        assert replay["attemptId"] == created["attemptId"]
        assert created["sequence"]["state"] == "active"
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        interaction = await pool.fetchrow(
            """
            SELECT interaction_type, intent, metadata
            FROM contact_interactions
            WHERE contact_id = $1 AND interaction_type = 'call'
            """,
            contact_id,
        )
        assert interaction is not None
        assert interaction["intent"] == "no_answer"
        assert interaction["metadata"] is not None
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        assert (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM eom_missed_call_sequence_steps AS step
            JOIN eom_missed_call_sequences AS sequence ON sequence.id = step.sequence_id
            WHERE sequence.contact_id = $1
            """,
                contact_id,
            )
            == 3
        )

        # A later real call is separate immutable evidence, but it cannot make
        # overlapping follow-up mail for the same lead.
        later = await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key("later-missed-call"),
            actor_id=7,
            actor_name="Juan",
        )
        assert later["idempotent"] is False
        assert later["sequence"]["sequenceId"] == created["sequence"]["sequenceId"]
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 2
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )


@pytest.mark.asyncio
async def test_concurrent_same_call_attempt_key_creates_one_attempt_and_sequence() -> (
    None
):
    database_url = _database_url_or_skip()
    async with _test_store() as (pool, schema):
        contact_id = await _insert_estimate_lead(pool)
        first_pool = await _schema_connection(database_url, schema)
        second_pool = await _schema_connection(database_url, schema)
        operation_key = _operation_key("concurrent-no-answer")
        try:
            first, second = await asyncio.gather(
                _service(first_pool).record_no_answer(
                    contact_id=contact_id,
                    operation_key=operation_key,
                    actor_id=7,
                    actor_name="Juan",
                ),
                _service(second_pool).record_no_answer(
                    contact_id=contact_id,
                    operation_key=operation_key,
                    actor_id=7,
                    actor_name="Juan",
                ),
            )
        finally:
            await _close_schema_connection(first_pool)
            await _close_schema_connection(second_pool)

        assert sorted((first["idempotent"], second["idempotent"])) == [False, True]
        assert first["attemptId"] == second["attemptId"]
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )


@pytest.mark.asyncio
async def test_recovery_operation_key_cannot_move_a_retry_to_another_contact() -> None:
    async with _test_store() as (pool, _schema):
        first_contact_id = await _insert_estimate_lead(pool)
        second_contact_id = await _insert_estimate_lead(pool)
        service = _service(pool)
        operation_key = _operation_key("cross-contact")
        await service.record_no_answer(
            contact_id=first_contact_id,
            operation_key=operation_key,
            actor_id=7,
            actor_name="Juan",
        )

        with pytest.raises(
            EOMMissedCallRecoveryConflictError,
            match="different missed-call recovery operation",
        ):
            await service.record_no_answer(
                contact_id=second_contact_id,
                operation_key=operation_key,
                actor_id=7,
                actor_name="Juan",
            )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                second_contact_id,
            )
            == 0
        )


@pytest.mark.asyncio
async def test_non_residential_estimate_records_call_but_never_starts_recovery() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool, ack_variant="commercial")
        result = await _service(pool).record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        assert result["sequence"] is None
        assert result["notStartedReason"] == "not_residential_estimate"
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 0
        )


@pytest.mark.asyncio
async def test_response_after_estimate_but_before_no_answer_never_starts_recovery() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        await pool._connection.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at,
                metadata, interaction_dedupe_key
            ) VALUES ($1, $2, 'sms', 'Controlled inbound response', 'reply', $3,
                      $4::jsonb, $5)
            """,
            uuid4(),
            contact_id,
            _NOW - timedelta(minutes=1),
            json.dumps({"crm_event_id": f"sms:{uuid4().hex}"}),
            f"prior-response-{uuid4().hex}",
        )

        result = await _service(pool).record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        assert result["sequence"] is None
        assert result["notStartedReason"] == "tracked_response_or_new_request"
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == 0
        )


@pytest.mark.asyncio
async def test_missing_booking_configuration_creates_visible_block_without_gateway_call() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway, config=_config(booking_link=""))

        result = await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        delivered = await service.dispatch_due_steps()

        assert result["sequence"]["state"] == "blocked_configuration"
        assert result["sequence"]["blockedReason"] == "booking_link_unavailable"
        assert delivered == 0
        assert gateway.calls == []
        assert (
            await pool.fetchval("SELECT COUNT(*) FROM eom_missed_call_sequence_steps")
            == 0
        )


@pytest.mark.asyncio
async def test_disabled_recovery_creates_an_honest_visible_block_without_gateway_call() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway, config=_config(enabled=False))

        result = await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        assert result["sequence"]["state"] == "blocked_configuration"
        assert result["sequence"]["blockedReason"] == "recovery_disabled"
        assert await service.dispatch_due_steps() == 0
        assert gateway.calls == []


@pytest.mark.asyncio
async def test_configuration_pause_blocks_existing_active_sequence_until_explicit_resume() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        active_gateway = _FakeGateway()
        active = _service(pool, gateway=active_gateway)
        await active.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        paused = _service(pool, gateway=active_gateway, config=_config(enabled=False))
        assert await paused.dispatch_due_steps() == 0
        sequence = await pool.fetchrow(
            """
            SELECT state, blocked_reason FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert dict(sequence) == {
            "state": "blocked_configuration",
            "blocked_reason": "recovery_disabled",
        }
        assert active_gateway.calls == []

        restored_gateway = _FakeGateway()
        restored = _service(pool, gateway=restored_gateway)
        # Restoring deploy-time configuration alone cannot revive overdue mail.
        assert await restored.dispatch_due_steps() == 0
        assert restored_gateway.calls == []

        resumed = await restored.resume_blocked_sequence(
            contact_id=contact_id,
            operation_key=_operation_key("resume-after-pause"),
            actor_id=7,
            actor_name="Juan",
        )
        assert resumed["sequence"]["state"] == "active"
        assert await restored.dispatch_due_steps() == 1
        assert len(restored_gateway.calls) == 1


@pytest.mark.asyncio
async def test_configuration_pause_preserves_a_pre_send_claim_until_resume() -> None:
    """A deployment pause must not turn an unproven claim into a skipped email."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        clock = {"now": _NOW}
        gateway = _FakeGateway()
        active = _service(pool, gateway=gateway, now_box=clock)
        await active.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        claim_result = await active._claim_one_due_step()
        assert claim_result.claim is not None

        paused = _service(
            pool,
            gateway=gateway,
            now_box=clock,
            config=_config(enabled=False),
        )
        assert await paused.dispatch_due_steps() == 0
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE id = $1",
                claim_result.claim.step_id,
            )
            == "attempting"
        )

        # Once the lease has expired, normal claim-recovery is safe only after
        # an operator deliberately resumes the still-eligible sequence.
        clock["now"] += timedelta(minutes=6)
        restored = _service(pool, gateway=gateway, now_box=clock)
        assert await restored.dispatch_due_steps() == 0
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE id = $1",
                claim_result.claim.step_id,
            )
            == "attempting"
        )
        await restored.resume_blocked_sequence(
            contact_id=contact_id,
            operation_key=_operation_key("resume-preserved-claim"),
            actor_id=7,
            actor_name="Juan",
        )
        assert await restored.dispatch_due_steps() == 1
        assert len(gateway.calls) == 1
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE id = $1",
                claim_result.claim.step_id,
            )
            == "sent"
        )


@pytest.mark.asyncio
async def test_blocked_configuration_sequence_can_still_stop_for_a_lead_advance() -> (
    None
):
    """Terminalizing a blocked sequence must clear its blocked-only fields."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        service = _service(pool, config=_config(booking_link=""))
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        await pool._connection.execute(
            "UPDATE contacts SET lead_stage = 'estimate_booked' WHERE id = $1",
            contact_id,
        )
        sequence = await pool.fetchrow(
            """
            SELECT state, blocked_reason, cancellation_reason
            FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert dict(sequence) == {
            "state": "cancelled",
            "blocked_reason": None,
            "cancellation_reason": "lead_advanced",
        }


@pytest.mark.asyncio
async def test_worker_sends_each_step_once_and_records_tenant_scoped_history() -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        clock = {"now": _NOW}
        gateway = _FakeGateway()
        history = _History()
        service = _service(pool, gateway=gateway, now_box=clock, history=history)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        assert await service.dispatch_due_steps() == 1
        assert await service.dispatch_due_steps() == 0
        assert len(gateway.calls) == 1
        assert gateway.calls[0]["recipient_email"] == "lead@example.test"
        assert len(history.calls) == 1
        history_call = history.calls[0]
        assert history_call["business_context_id"] == _EOM_CONTEXT
        assert history_call["template_type"] == "eom_missed_call_recovery"
        assert history_call["metadata"]["contact_id"] == str(contact_id)
        assert "lead@example.test" not in json.dumps(history_call["metadata"])

        sequence_id = UUID(history_call["metadata"]["sequence_id"])
        second_due = await pool.fetchval(
            """
            SELECT due_at FROM eom_missed_call_sequence_steps
            WHERE sequence_id = $1 AND step_number = 2
            """,
            sequence_id,
        )
        assert second_due == next_business_day_due(
            _NOW, timezone_name="America/Chicago"
        )
        # A late worker must preserve Email 3's full three-day interval from
        # actual Email 2 delivery, rather than compressing it from the old due.
        clock["now"] = second_due + timedelta(hours=4)
        assert await service.dispatch_due_steps() == 1
        second_sent_at = await pool.fetchval(
            """
            SELECT sent_at FROM eom_missed_call_sequence_steps
            WHERE sequence_id = $1 AND step_number = 2
            """,
            sequence_id,
        )
        third_due = await pool.fetchval(
            """
            SELECT due_at FROM eom_missed_call_sequence_steps
            WHERE sequence_id = $1 AND step_number = 3
            """,
            sequence_id,
        )
        assert third_due == second_sent_at + timedelta(days=3)
        clock["now"] = third_due
        assert await service.dispatch_due_steps() == 1
        assert await service.dispatch_due_steps() == 0
        assert len(gateway.calls) == 3
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequences WHERE id = $1", sequence_id
            )
            == "completed"
        )


@pytest.mark.asyncio
async def test_two_workers_cannot_send_the_same_due_step_twice() -> None:
    database_url = _database_url_or_skip()
    async with _test_store() as (pool, schema):
        contact_id = await _insert_estimate_lead(pool)
        await _service(pool).record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        first_pool = await _schema_connection(database_url, schema)
        second_pool = await _schema_connection(database_url, schema)
        try:
            gateway = _BlockingGateway()
            first = _service(first_pool, gateway=gateway)
            second = _service(second_pool, gateway=gateway)
            first_task = asyncio.create_task(first.dispatch_due_steps())
            await asyncio.wait_for(gateway.started.wait(), timeout=5)
            assert await second.dispatch_due_steps() == 0
            gateway.release.set()
            assert await asyncio.wait_for(first_task, timeout=5) == 1
        finally:
            await _close_schema_connection(first_pool)
            await _close_schema_connection(second_pool)

        assert len(gateway.calls) == 1
        assert (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM eom_missed_call_sequence_steps
            WHERE state = 'sent'
            """
            )
            == 1
        )


@pytest.mark.asyncio
async def test_state_change_between_durable_claim_and_delivery_stops_the_send() -> None:
    """The second state read is after the durable claim, not just before it."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        claimed = await service._claim_one_due_step()
        assert claimed.claim is not None

        # An appointment advance after the worker claimed the step but before
        # its provider call must terminalize the sequence. The delivery phase
        # then re-reads the claim/current lead state and cannot call the gateway.
        await pool._connection.execute(
            "UPDATE contacts SET lead_stage = 'estimate_booked' WHERE id = $1",
            contact_id,
        )

        assert await service._deliver_claim(claimed.claim) is None
        assert gateway.calls == []
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == "cancelled"
        )


@pytest.mark.asyncio
async def test_crash_after_durable_claim_never_reuses_an_expired_provider_key() -> None:
    """A restart after an unconfirmed provider call fails visible, not duplicate."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        clock = {"now": _NOW}
        original = _service(pool, now_box=clock)
        await original.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        # Claim is its own committed transaction. Simulate the process dying
        # after the provider request could have left the process but before it
        # recorded acceptance; no `deliver_claim` call is made here.
        claimed = await original._claim_one_due_step()
        assert claimed.processed is True
        assert claimed.claim is not None
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE id = $1",
                claimed.claim.step_id,
            )
            == "attempting"
        )

        clock["now"] = _NOW + timedelta(hours=24)
        restarted_gateway = _FakeGateway()
        restarted = _service(
            pool,
            gateway=restarted_gateway,
            now_box=clock,
        )
        assert await restarted.dispatch_due_steps() == 0
        assert restarted_gateway.calls == []
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == "recovery_required"
        )
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE id = $1",
                claimed.claim.step_id,
            )
            == "recovery_required"
        )


@pytest.mark.asyncio
async def test_retry_is_bounded_and_never_erases_call_evidence() -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        clock = {"now": _NOW}
        gateway = _RetryingGateway()
        service = _service(pool, gateway=gateway, now_box=clock)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        for _ in range(3):
            assert await service.dispatch_due_steps() == 1
            next_attempt = await pool.fetchval(
                """
                SELECT next_attempt_at FROM eom_missed_call_sequence_steps
                WHERE step_number = 1
                """
            )
            if next_attempt is not None:
                status = await service.statuses(contact_ids=[contact_id])
                assert status[0]["nextFollowUpAt"] == next_attempt.isoformat()
                clock["now"] = next_attempt

        assert len(gateway.calls) == 3
        assert (
            await pool.fetchval("SELECT state FROM eom_missed_call_sequences")
            == "failed"
        )
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequence_steps WHERE step_number = 1"
            )
            == "failed"
        )
        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )
        assert await service.dispatch_due_steps() == 0
        assert len(gateway.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("column", "value", "expected_reason"),
    (
        ("lead_stage", "estimate_booked", "lead_advanced"),
        ("lead_stage", "completed", "lead_advanced"),
        ("contact_type", "customer", "became_customer"),
        ("status", "lost", "contact_inactive"),
        ("status", "closed", "contact_inactive"),
        ("customer_type", "commercial", "non_residential"),
    ),
)
async def test_lifecycle_stop_conditions_cancel_before_later_delivery(
    column: str,
    value: str,
    expected_reason: str,
) -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        # Values are parametrized from this closed test tuple, never request
        # input, so the identifier interpolation cannot turn this into SQL.
        await pool._connection.execute(
            f"UPDATE contacts SET {column} = $2 WHERE id = $1",
            contact_id,
            value,
        )

        assert await service.dispatch_due_steps() == 0
        assert gateway.calls == []
        sequence = await pool.fetchrow(
            """
            SELECT state, cancellation_reason FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert dict(sequence) == {
            "state": "cancelled",
            "cancellation_reason": expected_reason,
        }


@pytest.mark.asyncio
async def test_contact_email_edit_keeps_sequence_when_form_recipient_is_unchanged() -> (
    None
):
    """The form submission owns the effective recipient, not a shadow field."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        await pool._connection.execute(
            "UPDATE contacts SET email = $2 WHERE id = $1",
            contact_id,
            "directory-correction@example.test",
        )

        assert await service.dispatch_due_steps() == 1
        assert len(gateway.calls) == 1
        sequence = await pool.fetchrow(
            """
            SELECT state, cancellation_reason FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert dict(sequence) == {"state": "active", "cancellation_reason": None}


@pytest.mark.asyncio
async def test_effective_form_recipient_change_cancels_before_delivery() -> None:
    """A repair that changes the actual snapshotted recipient must stop mail."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        await pool._connection.execute(
            """
            UPDATE contact_interactions
               SET metadata = jsonb_set(
                   metadata, '{submitted_email}', to_jsonb($2::text), true
               )
             WHERE contact_id = $1
               AND interaction_type = 'web_form'
               AND intent = 'estimate_request'
            """,
            contact_id,
            "corrected-recipient@example.test",
        )

        assert await service.dispatch_due_steps() == 0
        assert gateway.calls == []
        assert (
            await pool.fetchval(
                """
            SELECT cancellation_reason FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
                contact_id,
            )
            == "recipient_changed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interaction_type", "intent", "expected_reason"),
    (
        ("sms", "reply", "tracked_inbound_response"),
        ("email_inbound", "reply", "email_inbound"),
        ("lead_response", "reply", "lead_response"),
        ("callback_completed", "phone", "callback_completed"),
        ("conversation_completed", "phone", "conversation_completed"),
        ("opt_out", "unsubscribe", "opt_out"),
        ("web_form", "estimate_request", "new_estimate_request"),
    ),
)
async def test_recorded_response_stop_conditions_cancel_before_later_delivery(
    interaction_type: str,
    intent: str,
    expected_reason: str,
) -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        await pool._connection.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at,
                metadata, interaction_dedupe_key
            ) VALUES ($1, $2, $3, 'Controlled stop-condition evidence', $4, $5,
                      $6::jsonb, $7)
            """,
            uuid4(),
            contact_id,
            interaction_type,
            intent,
            _NOW + timedelta(minutes=1),
            json.dumps(
                # The current EOM inbound SMS producer uses this provider
                # marker. A bare CRM `sms` row can also represent an outgoing
                # reminder, and must not be mistaken for a lead response.
                {"crm_event_id": f"sms:{uuid4().hex}"}
                if interaction_type == "sms"
                else {}
            ),
            f"stop-{interaction_type}-{uuid4().hex}",
        )

        assert await service.dispatch_due_steps() == 0
        assert gateway.calls == []
        sequence = await pool.fetchrow(
            """
            SELECT state, cancellation_reason FROM eom_missed_call_sequences
            WHERE contact_id = $1
            """,
            contact_id,
        )
        assert dict(sequence) == {
            "state": "cancelled",
            "cancellation_reason": expected_reason,
        }
        if interaction_type == "opt_out":
            assert (
                await pool.fetchval(
                    """
                SELECT EXISTS (
                    SELECT 1 FROM eom_missed_call_contact_suppressions
                    WHERE contact_id = $1
                )
                """,
                    contact_id,
                )
                is True
            )


@pytest.mark.asyncio
async def test_backfilled_response_before_sequence_does_not_cancel_current_follow_up() -> (
    None
):
    """Trigger evaluation uses the evidence timeline, not insertion order."""

    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        await pool._connection.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at,
                metadata, interaction_dedupe_key
            ) VALUES ($1, $2, 'lead_response', 'Historical controlled response',
                      'reply', $3, '{}'::jsonb, $4)
            """,
            uuid4(),
            contact_id,
            _NOW - timedelta(minutes=1),
            f"backfilled-response-{uuid4().hex}",
        )

        assert await service.dispatch_due_steps() == 1
        assert len(gateway.calls) == 1
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == "active"
        )


@pytest.mark.asyncio
async def test_generic_sms_without_inbound_evidence_does_not_claim_a_lead_replied() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        gateway = _FakeGateway()
        service = _service(pool, gateway=gateway)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )

        await pool._connection.execute(
            """
            INSERT INTO contact_interactions (
                id, contact_id, interaction_type, summary, intent, occurred_at,
                metadata, interaction_dedupe_key
            ) VALUES ($1, $2, 'sms', 'Controlled outbound reminder', 'follow_up',
                      $3, '{}'::jsonb, $4)
            """,
            uuid4(),
            contact_id,
            _NOW + timedelta(minutes=1),
            f"outbound-sms-{uuid4().hex}",
        )

        assert await service.dispatch_due_steps() == 1
        assert len(gateway.calls) == 1
        assert (
            await pool.fetchval(
                "SELECT state FROM eom_missed_call_sequences WHERE contact_id = $1",
                contact_id,
            )
            == "active"
        )


@pytest.mark.asyncio
async def test_explicit_resume_after_missing_booking_configuration_is_the_only_recovery_path() -> (
    None
):
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        blocked = _service(pool, config=_config(booking_link=""))
        created = await blocked.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        assert created["sequence"]["state"] == "blocked_configuration"

        gateway = _FakeGateway()
        resumed = _service(pool, gateway=gateway)
        result = await resumed.resume_blocked_sequence(
            contact_id=contact_id,
            operation_key=_operation_key("resume"),
            actor_id=7,
            actor_name="Juan",
        )
        assert result["idempotent"] is False
        assert result["sequence"]["state"] == "active"
        assert await resumed.dispatch_due_steps() == 1
        assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_explicit_cancellation_is_idempotent_only_for_the_same_reason() -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        service = _service(pool)
        await service.record_no_answer(
            contact_id=contact_id,
            operation_key=_operation_key(),
            actor_id=7,
            actor_name="Juan",
        )
        operation_key = _operation_key("cancel")

        cancelled = await service.cancel_sequence(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=7,
            actor_name="Juan",
            reason="response_recorded",
        )
        replay = await service.cancel_sequence(
            contact_id=contact_id,
            operation_key=operation_key,
            actor_id=7,
            actor_name="Juan",
            reason="response_recorded",
        )
        assert cancelled["idempotent"] is False
        assert replay["idempotent"] is True
        assert cancelled["sequence"]["cancellationReason"] == "response_recorded"
        cancellation_event = await pool.fetchrow(
            """
            SELECT actor_id, actor_name, source, reason_code
            FROM eom_missed_call_sequence_events
            WHERE sequence_id = $1 AND event_type = 'sequence_cancelled'
            """,
            UUID(cancelled["sequence"]["sequenceId"]),
        )
        assert dict(cancellation_event) == {
            "actor_id": 7,
            "actor_name": "Juan",
            "source": "time_tracker",
            "reason_code": "response_recorded",
        }
        with pytest.raises(
            EOMMissedCallRecoveryConflictError,
            match="different missed-call recovery operation",
        ):
            await service.cancel_sequence(
                contact_id=contact_id,
                operation_key=operation_key,
                actor_id=7,
                actor_name="Juan",
                reason="manual",
            )


@pytest.mark.asyncio
async def test_resume_and_cancel_keys_are_globally_bound_to_their_original_contact() -> (
    None
):
    async with _test_store() as (pool, _schema):
        first_contact_id = await _insert_estimate_lead(pool)
        second_contact_id = await _insert_estimate_lead(pool)
        blocked = _service(pool, config=_config(booking_link=""))
        for contact_id in (first_contact_id, second_contact_id):
            await blocked.record_no_answer(
                contact_id=contact_id,
                operation_key=_operation_key("blocked-sequence"),
                actor_id=7,
                actor_name="Juan",
            )

        service = _service(pool)
        resume_key = _operation_key("cross-contact-resume")
        await service.resume_blocked_sequence(
            contact_id=first_contact_id,
            operation_key=resume_key,
            actor_id=7,
            actor_name="Juan",
        )
        with pytest.raises(
            EOMMissedCallRecoveryConflictError,
            match="different missed-call recovery operation",
        ):
            await service.resume_blocked_sequence(
                contact_id=second_contact_id,
                operation_key=resume_key,
                actor_id=7,
                actor_name="Juan",
            )

        cancel_key = _operation_key("cross-contact-cancel")
        await service.cancel_sequence(
            contact_id=second_contact_id,
            operation_key=cancel_key,
            actor_id=7,
            actor_name="Juan",
            reason="manual",
        )
        with pytest.raises(
            EOMMissedCallRecoveryConflictError,
            match="different missed-call recovery operation",
        ):
            await service.cancel_sequence(
                contact_id=first_contact_id,
                operation_key=cancel_key,
                actor_id=7,
                actor_name="Juan",
                reason="manual",
            )


@pytest.mark.asyncio
async def test_operator_route_reaches_persisted_call_and_status_state() -> None:
    async with _test_store() as (pool, _schema):
        contact_id = await _insert_estimate_lead(pool)
        service = _service(pool, gateway=_FakeGateway())
        app = FastAPI()
        app.include_router(funnel_mod.router, prefix="/api/v1")
        app.dependency_overrides[funnel_auth_mod.require_eom_funnel_api] = lambda: None
        app.dependency_overrides[funnel_auth_mod.require_eom_funnel_actor] = lambda: {
            "id": 7,
            "name": "Juan",
        }
        app.dependency_overrides[funnel_mod._missed_call_recovery_dependency] = lambda: (
            service
        )
        key = _operation_key()
        headers = {"Idempotency-Key": key}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                f"/api/v1/eom-funnel/leads/{contact_id}/missed-call-attempts",
                headers=headers,
            )
            assert created.status_code == 201
            assert created.json()["sequence"]["state"] == "active"

            replay = await client.post(
                f"/api/v1/eom-funnel/leads/{contact_id}/missed-call-attempts",
                headers=headers,
            )
            assert replay.status_code == 200
            assert replay.json()["idempotent"] is True

            status_response = await client.get(
                "/api/v1/eom-funnel/missed-call-recovery-status",
                params=[("contact_id", str(contact_id))],
            )
            assert status_response.status_code == 200
            payload = status_response.json()
            assert payload["checked"] == 1
            assert payload["sequences"][0]["contactId"] == str(contact_id)
            assert payload["sequences"][0]["nextStepNumber"] == 1

        assert (
            await pool.fetchval(
                "SELECT COUNT(*) FROM eom_missed_call_attempts WHERE contact_id = $1",
                contact_id,
            )
            == 1
        )


def test_funnel_advertises_the_additive_missed_call_contract_only_when_routes_exist() -> (
    None
):
    expected = {
        "lead.missed_call_attempt.record",
        "lead.missed_call_recovery.status",
        "lead.missed_call_recovery.resume",
        "lead.missed_call_recovery.cancel",
    }
    assert expected <= set(funnel_mod.served_capabilities())
    registered = {
        (method, route.path)
        for route in funnel_mod.router.routes
        for method in (route.methods or ())
    }
    assert {
        ("POST", "/eom-funnel/leads/{contact_id}/missed-call-attempts"),
        ("GET", "/eom-funnel/missed-call-recovery-status"),
        ("POST", "/eom-funnel/leads/{contact_id}/missed-call-recovery/resume"),
        ("POST", "/eom-funnel/leads/{contact_id}/missed-call-recovery/cancel"),
    } <= registered
