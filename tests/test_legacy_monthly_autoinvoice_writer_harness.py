"""Isolated, opt-in integration proof for the legacy monthly invoice writer.

This module deliberately does not call ``init_database()``.  Its writer tests
admit only a separately armed loopback test database, then confine every real
repository write to a UUID-named schema that is dropped in ``finally``.
"""

from __future__ import annotations

import itertools
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Iterator
from urllib.parse import urlsplit
from uuid import uuid4

import pytest


_HARNESS_ENABLED_ENV = "ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_HARNESS"
_HARNESS_DATABASE_URL_ENV = "ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_TEST_DATABASE_URL"
_HARNESS_DATABASE_NAME = "atlas_receivables_test"
_HARNESS_PORT = 5432
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_HARNESS_MIGRATIONS = (
    "045_invoices.sql",
    "047_invoice_extra_fields.sql",
    "048_customer_services.sql",
)


def _is_harness_armed(value: str | None) -> bool:
    """Accept only the explicit marker that authorizes disposable test writes."""
    return value == "1"


def _writer_harness_invoicing_config() -> Any:
    """Read the test-only harness controls through the canonical config model."""
    from atlas_brain.config import InvoicingConfig

    return InvoicingConfig()


def _validate_writer_harness_database_url(database_url: object) -> str:
    """Return only the deliberately narrow local PostgreSQL harness target."""
    if not isinstance(database_url, str) or not database_url:
        raise ValueError("legacy monthly writer harness database URL is required")

    parsed = urlsplit(database_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "legacy monthly writer harness database URL has an invalid port"
        ) from exc

    if (
        parsed.scheme != "postgresql"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port != _HARNESS_PORT
        or parsed.path != f"/{_HARNESS_DATABASE_NAME}"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "legacy monthly writer harness must use the exact loopback "
            "atlas_receivables_test PostgreSQL target"
        )
    return database_url


def _require_writer_harness_database_url() -> str:
    """Skip ordinary local runs; reject an unsafe explicit target before import."""
    harness_config = _writer_harness_invoicing_config()
    if not _is_harness_armed(harness_config.legacy_monthly_writer_harness_opt_in):
        pytest.skip(f"{_HARNESS_ENABLED_ENV}=1 is required for writer-harness tests")
    return _validate_writer_harness_database_url(
        harness_config.legacy_monthly_writer_harness_database_url
    )


def _load_writer_harness_asyncpg(
    importer: Callable[[str], Any] = import_module,
) -> Any:
    """Fail an armed proof if its required PostgreSQL driver is unavailable."""
    try:
        return importer("asyncpg")
    except ImportError:
        harness_config = _writer_harness_invoicing_config()
        if _is_harness_armed(harness_config.legacy_monthly_writer_harness_opt_in):
            pytest.fail(
                "explicitly armed legacy monthly writer harness requires asyncpg"
            )
        pytest.skip("asyncpg is required for legacy monthly writer harness tests")


def _url_from_grammar(
    scheme: str,
    host: str,
    port: int | None,
    path: str,
    suffix: str,
) -> str:
    host_text = f"[{host}]" if ":" in host else host
    port_text = "" if port is None else f":{port}"
    return f"{scheme}://postgres:postgres@{host_text}{port_text}{path}{suffix}"


def test_writer_harness_url_grammar_is_fail_closed() -> None:
    """Generate URL terminals and admit exactly the deliberate safe grammar."""
    schemes = ("postgresql", "postgres", "https", "")
    hosts = tuple(sorted(_LOOPBACK_HOSTS)) + ("db.internal", "localhost.", "0.0.0.0")
    ports = (_HARNESS_PORT, 5433, None)
    paths = (f"/{_HARNESS_DATABASE_NAME}", "/atlas", f"/{_HARNESS_DATABASE_NAME}/extra")
    suffixes = ("", "?sslmode=disable", "#fragment")

    admitted = 0
    for scheme, host, port, path, suffix in itertools.product(
        schemes, hosts, ports, paths, suffixes
    ):
        candidate = _url_from_grammar(scheme, host, port, path, suffix)
        expected_safe = (
            scheme == "postgresql"
            and host in _LOOPBACK_HOSTS
            and port == _HARNESS_PORT
            and path == f"/{_HARNESS_DATABASE_NAME}"
            and suffix == ""
        )
        if expected_safe:
            assert _validate_writer_harness_database_url(candidate) == candidate
            admitted += 1
        else:
            with pytest.raises(ValueError):
                _validate_writer_harness_database_url(candidate)

    assert admitted == len(_LOOPBACK_HOSTS)


def test_writer_harness_opt_in_is_exact() -> None:
    assert _is_harness_armed("1")
    for value in (None, "", "0", "true", "True", "yes", "1 "):
        assert not _is_harness_armed(value)


@pytest.mark.asyncio
async def test_unarmed_harness_stops_before_asyncpg_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unarmed context must skip before it can load a database driver."""
    monkeypatch.delenv(_HARNESS_ENABLED_ENV, raising=False)
    monkeypatch.delenv(_HARNESS_DATABASE_URL_ENV, raising=False)
    asyncpg_loader_calls = 0

    def forbidden_asyncpg_loader() -> Any:
        nonlocal asyncpg_loader_calls
        asyncpg_loader_calls += 1
        raise AssertionError("unarmed harness reached asyncpg import")

    with pytest.raises(pytest.skip.Exception, match=rf"{_HARNESS_ENABLED_ENV}=1"):
        async with _writer_harness_database(asyncpg_loader=forbidden_asyncpg_loader):
            raise AssertionError("unarmed harness opened a context")

    assert asyncpg_loader_calls == 0


def test_writer_harness_settings_use_typed_invoicing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas_brain.config import InvoicingConfig

    database_url = (
        "postgresql://postgres:postgres@127.0.0.1:5432/atlas_receivables_test"
    )
    monkeypatch.delenv(_HARNESS_ENABLED_ENV, raising=False)
    monkeypatch.delenv(_HARNESS_DATABASE_URL_ENV, raising=False)

    inactive_config = InvoicingConfig(_env_file=None)

    assert inactive_config.legacy_monthly_writer_harness_opt_in == ""
    assert inactive_config.legacy_monthly_writer_harness_database_url == ""

    monkeypatch.setenv(_HARNESS_ENABLED_ENV, "1")
    monkeypatch.setenv(_HARNESS_DATABASE_URL_ENV, database_url)

    config = InvoicingConfig(_env_file=None)

    assert config.legacy_monthly_writer_harness_opt_in == "1"
    assert config.legacy_monthly_writer_harness_database_url == database_url


def test_writer_harness_workflow_enrolls_autonomous_notification_config() -> None:
    """Both workflow triggers must exercise the task's notification config gate."""
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/atlas_invoicing_checks.yml"
    ).read_text(encoding="utf-8")
    entry = '      - "atlas_brain/autonomous/config.py"\n'
    pull_request_paths, separator, push_paths = workflow.partition("  push:\n")

    assert separator == "  push:\n"
    assert entry in pull_request_paths
    assert entry in push_paths


@pytest.mark.asyncio
async def test_unsafe_harness_target_stops_before_asyncpg_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe explicit URL cannot progress to the connection-import seam."""
    monkeypatch.setenv(_HARNESS_ENABLED_ENV, "1")
    monkeypatch.setenv(
        _HARNESS_DATABASE_URL_ENV,
        "postgresql://postgres:postgres@db.internal:5432/atlas_receivables_test",
    )

    def forbidden_asyncpg_loader() -> Any:
        raise AssertionError("unsafe harness target reached asyncpg import")

    with pytest.raises(ValueError):
        async with _writer_harness_database(asyncpg_loader=forbidden_asyncpg_loader):
            raise AssertionError("unsafe harness target opened a context")


def test_armed_harness_missing_asyncpg_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_HARNESS_ENABLED_ENV, "1")
    monkeypatch.setenv(
        _HARNESS_DATABASE_URL_ENV,
        "postgresql://postgres:postgres@127.0.0.1:5432/atlas_receivables_test",
    )

    def missing_asyncpg(_module_name: str) -> Any:
        raise ImportError("simulated missing asyncpg")

    with pytest.raises(
        pytest.fail.Exception, match="explicitly armed.*requires asyncpg"
    ):
        _load_writer_harness_asyncpg(missing_asyncpg)


class _SchemaPool:
    """Make real repositories use one disposable schema-scoped connection."""

    is_initialized = True

    def __init__(self, conn: Any, schema: str) -> None:
        self.conn = conn
        self.schema = schema

    async def _in_schema(self, method: str, query: str, *args: object) -> Any:
        async with self.conn.transaction():
            await self.conn.execute(f'SET LOCAL search_path TO "{self.schema}", public')
            return await getattr(self.conn, method)(query, *args)

    async def execute(self, query: str, *args: object) -> str:
        return await self._in_schema("execute", query, *args)

    async def fetch(self, query: str, *args: object) -> list[Any]:
        return await self._in_schema("fetch", query, *args)

    async def fetchrow(self, query: str, *args: object) -> Any:
        return await self._in_schema("fetchrow", query, *args)


@contextmanager
def _bind_harness_database_pool(pool: _SchemaPool) -> Iterator[None]:
    """Bind the real repository singleton to the isolated schema pool."""
    from atlas_brain.storage import database

    previous_pool = database._db_pool
    database._db_pool = pool
    try:
        yield
    finally:
        database._db_pool = previous_pool


@dataclass(frozen=True)
class _WriterHarnessDatabase:
    conn: Any
    schema: str
    pool: _SchemaPool
    database_url: str


@asynccontextmanager
async def _writer_harness_database(
    *, asyncpg_loader: Callable[[], Any] = _load_writer_harness_asyncpg
) -> AsyncIterator[_WriterHarnessDatabase]:
    """Create and always drop one randomly named schema in the safe test DB."""
    database_url = _require_writer_harness_database_url()
    asyncpg = asyncpg_loader()
    schema = f"legacy_monthly_writer_{uuid4().hex}"
    conn = await asyncpg.connect(database_url)
    migrations = Path(__file__).parents[1] / "atlas_brain/storage/migrations"

    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        await conn.execute("CREATE TABLE contacts (id UUID PRIMARY KEY)")
        for migration_name in _HARNESS_MIGRATIONS:
            await conn.execute(
                (migrations / migration_name).read_text(encoding="utf-8")
            )
        yield _WriterHarnessDatabase(
            conn=conn,
            schema=schema,
            pool=_SchemaPool(conn, schema),
            database_url=database_url,
        )
    finally:
        try:
            await conn.execute("SET search_path TO public")
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()


async def _schema_exists(database_url: str, schema: str) -> bool:
    asyncpg = _load_writer_harness_asyncpg()
    conn = await asyncpg.connect(database_url)
    try:
        return bool(
            await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)", schema
            )
        )
    finally:
        await conn.close()


class _HarnessCalendar:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self.events = events
        self.calls: list[tuple[datetime, datetime, str | None]] = []

    async def list_events(
        self,
        start: datetime,
        end: datetime,
        *,
        calendar_id: str | None = None,
    ) -> list[SimpleNamespace]:
        self.calls.append((start, end, calendar_id))
        return list(self.events)


class _HarnessCRM:
    def __init__(self) -> None:
        self.interactions: list[dict[str, object]] = []

    async def get_contact(self, _contact_id: str) -> dict[str, str]:
        return {
            "full_name": "Horizon Fixture LLC",
            "email": "billing-harness@example.test",
            "phone": "555-0100",
            "address": "1 Fixture Way",
        }

    async def log_interaction(self, **kwargs: object) -> None:
        self.interactions.append(kwargs)


def _confirmed_event(summary: str, event_date: date) -> SimpleNamespace:
    return SimpleNamespace(
        summary=summary,
        start=datetime(
            event_date.year, event_date.month, event_date.day, 16, tzinfo=timezone.utc
        ),
        status="confirmed",
    )


def _scheduled_monthly_task():
    from atlas_brain.storage.models import ScheduledTask

    return ScheduledTask(
        id=uuid4(),
        name="monthly_invoice_generation",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 1 * *",
        metadata={"billing_month": "2026-04", "notify": False},
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_isolated_legacy_writer_creates_one_draft_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """Exercise the real writer/repositories without ambient data or delivery."""
    database_url = _require_writer_harness_database_url()
    schema: str | None = None

    async with _writer_harness_database() as harness:
        schema = harness.schema
        from atlas_brain.autonomous.tasks import (
            monthly_invoice_generation as task_module,
        )
        from atlas_brain.autonomous.config import autonomous_config
        from atlas_brain.config import settings
        from atlas_brain.services import (
            calendar_provider,
            crm_provider,
            email_provider,
            invoice_pdf,
        )
        from atlas_brain.storage.repositories.customer_service import (
            CustomerServiceRepository,
        )
        from atlas_brain.storage.repositories.invoice import InvoiceRepository
        import httpx

        contact_id = uuid4()
        await harness.conn.execute("INSERT INTO contacts (id) VALUES ($1)", contact_id)

        cleanup = ExitStack()
        request.addfinalizer(cleanup.close)
        cleanup.enter_context(_bind_harness_database_pool(harness.pool))

        service_repo = CustomerServiceRepository()
        invoice_repo = InvoiceRepository()
        calendar = _HarnessCalendar(
            [
                _confirmed_event("Horizon Fixture cleaning", date(2026, 4, 8)),
                _confirmed_event("Horizon Fixture cleaning", date(2026, 4, 8)),
            ]
        )
        crm = _HarnessCRM()
        rendered_invoice_numbers: list[str] = []
        email_factory_calls: list[None] = []
        notification_transport_posts: list[dict[str, Any]] = []

        def render_harness_pdf(rendered_invoice: dict) -> bytes:
            rendered_invoice_numbers.append(rendered_invoice["invoice_number"])
            return b"%PDF-harness"

        def forbidden_email_provider() -> None:
            email_factory_calls.append(None)
            raise AssertionError(
                "review-mode writer attempted to construct an email provider"
            )

        async def forbidden_notification_transport_post(
            *args: Any, **kwargs: Any
        ) -> Any:
            notification_transport_posts.append({"args": args, "kwargs": kwargs})
            raise AssertionError(
                "review-mode writer attempted to post an ntfy notification"
            )

        monkeypatch.setattr(
            calendar_provider, "get_calendar_provider", lambda: calendar
        )
        monkeypatch.setattr(crm_provider, "get_crm_provider", lambda: crm)
        monkeypatch.setattr(invoice_pdf, "render_invoice_pdf", render_harness_pdf)
        monkeypatch.setattr(
            email_provider, "get_email_provider", forbidden_email_provider
        )
        assert autonomous_config.notify_results is True
        monkeypatch.setattr(settings.alerts, "ntfy_enabled", True)
        monkeypatch.setattr(
            httpx.AsyncClient, "post", forbidden_notification_transport_post
        )
        monkeypatch.setattr(settings.invoicing, "enabled", True)
        monkeypatch.setattr(settings.invoicing, "auto_invoice_enabled", True)
        monkeypatch.setattr(settings.invoicing, "auto_invoice_review_mode", True)
        monkeypatch.setattr(settings.invoicing, "auto_invoice_send_email", True)
        monkeypatch.setattr(settings.invoicing, "auto_invoice_due_days", 30)
        monkeypatch.setattr(
            settings.invoicing, "auto_invoice_calendar_id", "harness-calendar"
        )
        monkeypatch.setattr(settings.invoicing, "auto_invoice_save_path", str(tmp_path))

        service = await service_repo.create(
            contact_id=contact_id,
            service_name="Fixture Cleaning",
            rate=125.0,
            calendar_keyword="Horizon Fixture",
            rate_label="Per Visit",
            auto_invoice=True,
        )
        source_ref = f"{service['id']}_2026-04"
        task = _scheduled_monthly_task()

        first = await task_module.run(task)

        assert first["period"] == "2026-04"
        assert first["review_mode"] is True
        assert first["invoices_created"] == 1
        assert first["invoices_sent"] == 0
        assert first["invoices_skipped_dedup"] == 0
        assert first["total_amount"] == 250.0
        assert calendar.calls == [
            (
                datetime(2026, 4, 1, tzinfo=timezone.utc),
                datetime(2026, 5, 1, 6, tzinfo=timezone.utc),
                "harness-calendar",
            )
        ]

        persisted_invoice = await invoice_repo.get_by_source_ref(source_ref)
        assert persisted_invoice is not None
        assert persisted_invoice["status"] == "draft"
        assert persisted_invoice["total_amount"] == 250.0
        assert persisted_invoice["sent_at"] is None
        assert persisted_invoice["sent_via"] is None
        assert persisted_invoice["line_items"] == [
            {
                "date": "04/08/2026",
                "description": "Fixture Cleaning",
                "quantity": 2,
                "unit_price": 125.0,
                "amount": 250.0,
            }
        ]
        assert (
            await harness.conn.fetchval(
                "SELECT COUNT(*) FROM invoices WHERE source_ref = $1", source_ref
            )
            == 1
        )

        service_state = await harness.conn.fetchrow(
            "SELECT last_invoiced_at, next_invoice_date FROM customer_services WHERE id = $1",
            service["id"],
        )
        assert service_state["last_invoiced_at"] == date(2026, 4, 30)
        assert service_state["next_invoice_date"] == date(2026, 5, 1)

        pdf_path = (
            tmp_path
            / "2026"
            / "Horizon Fixture LLC"
            / f"{persisted_invoice['invoice_number']}.pdf"
        )
        assert pdf_path.read_bytes() == b"%PDF-harness"
        assert rendered_invoice_numbers == [persisted_invoice["invoice_number"]]
        assert email_factory_calls == []
        assert notification_transport_posts == []
        assert len(crm.interactions) == 1

        second = await task_module.run(task)

        assert second["invoices_created"] == 0
        assert second["invoices_sent"] == 0
        assert second["invoices_skipped_dedup"] == 1
        assert (
            await harness.conn.fetchval(
                "SELECT COUNT(*) FROM invoices WHERE source_ref = $1", source_ref
            )
            == 1
        )
        assert rendered_invoice_numbers == [persisted_invoice["invoice_number"]]
        assert email_factory_calls == []
        assert notification_transport_posts == []
        assert len(crm.interactions) == 1

        # Restore before the schema context tears down; the finalizer covers failures.
        cleanup.close()

    assert schema is not None
    assert not await _schema_exists(database_url, schema)


class _SyntheticHarnessFailure(Exception):
    """Used only to prove the schema cleanup path executes during a failure."""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_isolated_writer_schema_is_dropped_after_context_failure() -> None:
    database_url = _require_writer_harness_database_url()
    schema: str | None = None

    with pytest.raises(_SyntheticHarnessFailure):
        async with _writer_harness_database() as harness:
            schema = harness.schema
            raise _SyntheticHarnessFailure("force harness cleanup")

    assert schema is not None
    assert not await _schema_exists(database_url, schema)
