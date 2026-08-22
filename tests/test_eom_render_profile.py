import asyncio
import hashlib
import json
import os
import string
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

_RAW_RECEIVABLES_SERVICE_TOKEN_ENV = "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"
_RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV = "ATLAS_EOM_FUNNEL_SERVICE_TOKEN"
_GENERATED_RECEIVABLES_TOKEN_PREFIX = "eomrx_v1_"
_GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH = 43
_GENERATED_RECEIVABLES_TOKEN_PAYLOAD_CHARS = (
    string.ascii_letters + string.digits + "_-"
)


def _alternating_case(value: str, *, starts_upper: bool) -> str:
    uppercase = starts_upper
    chars: list[str] = []
    for char in value:
        if char.isalpha():
            chars.append(char.upper() if uppercase else char.lower())
            uppercase = not uppercase
        else:
            chars.append(char)
    return "".join(chars)


_RAW_RECEIVABLES_SERVICE_TOKEN_KEY_CASES = tuple(
    dict.fromkeys(
        (
            _RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
            _RAW_RECEIVABLES_SERVICE_TOKEN_ENV.lower(),
            _alternating_case(
                _RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
                starts_upper=True,
            ),
            _alternating_case(
                _RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
                starts_upper=False,
            ),
        )
    )
)
_RAW_EOM_FUNNEL_SERVICE_TOKEN_KEY_CASES = tuple(
    dict.fromkeys(
        (
            _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV,
            _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV.lower(),
            _alternating_case(
                _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV,
                starts_upper=True,
            ),
            _alternating_case(
                _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV,
                starts_upper=False,
            ),
        )
    )
)


def _generated_receivables_token_oracle(token: str) -> bool:
    if not token.startswith(_GENERATED_RECEIVABLES_TOKEN_PREFIX):
        return False
    payload = token.removeprefix(_GENERATED_RECEIVABLES_TOKEN_PREFIX)
    return len(payload) == _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH and all(
        char in _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_CHARS for char in payload
    )


def _mixed_generated_payload(length: int) -> str:
    return "".join(
        _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_CHARS[
            (index * 7 + 3) % len(_GENERATED_RECEIVABLES_TOKEN_PAYLOAD_CHARS)
        ]
        for index in range(length)
    )


def _generated_payload_with_replacement(
    *,
    length: int,
    replacement: str,
    position: int,
) -> str:
    payload = list(_mixed_generated_payload(length))
    payload[position] = replacement
    return "".join(payload)


def _sha256_ascii(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _configure_billing_recipient_readiness(monkeypatch, receivables) -> None:
    """Make authentication-only readiness tests satisfy the payment dependency."""

    class _CanonicalPool:
        is_initialized = True

    class _CanonicalCRM:
        async def billing_recipients_schema_ready(self):
            return True

    monkeypatch.setattr(
        receivables.funnel_settings,
        "db_connection_string",
        "postgresql://canonical.test/eom",
    )
    monkeypatch.setattr(
        receivables,
        "get_eom_funnel_db_pool",
        lambda: _CanonicalPool(),
    )
    monkeypatch.setattr(
        receivables,
        "get_eom_funnel_crm_provider",
        lambda: _CanonicalCRM(),
    )


def _route_paths_for_app_expr(app_expr: str) -> str:
    return f"""
def _route_paths(app):
    paths = []

    def visit(routes, prefix=""):
        for route in routes:
            route_path = getattr(route, "path", None)
            if isinstance(route_path, str):
                paths.append(f"{{prefix}}{{route_path}}")
            original_router = getattr(route, "original_router", None)
            include_context = getattr(route, "include_context", None)
            route_prefix = getattr(include_context, "prefix", "")
            if original_router is not None:
                visit(original_router.routes, f"{{prefix}}{{route_prefix}}")

    visit({app_expr}.routes)
    return sorted(set(paths))
"""


def _isolated_eom_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith(("ATLAS_INVOICING_", "ATLAS_EOM_FUNNEL_")):
            env.pop(key, None)
    env[_RAW_RECEIVABLES_SERVICE_TOKEN_ENV] = ""
    env[_RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV] = ""
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "false"
    env["ATLAS_EOM_FUNNEL_API_ENABLED"] = "false"
    env["ATLAS_DB_ENABLED"] = "false"
    return env


def _atlas_subprocess_env() -> dict[str, str]:
    env = _isolated_eom_subprocess_env()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    return env


def test_route_path_probe_follows_included_router_context():
    nested_route = SimpleNamespace(path="/leads/intake")
    included_router = SimpleNamespace(routes=[nested_route])
    included_route = SimpleNamespace(
        path=None,
        original_router=included_router,
        include_context=SimpleNamespace(prefix="/api/v1"),
    )
    namespace: dict[str, object] = {"app": SimpleNamespace(routes=[included_route])}

    exec(_route_paths_for_app_expr("app"), namespace)

    assert namespace["_route_paths"](namespace["app"]) == ["/api/v1/leads/intake"]


def test_eom_profile_import_does_not_load_full_api_package():
    probe = """
import importlib
import json
import sys

module = importlib.import_module("atlas_brain.main_eom")
paths = []
for route in module.app.routes:
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str):
        paths.append(route_path)
    original_router = getattr(route, "original_router", None)
    include_context = getattr(route, "include_context", None)
    route_prefix = getattr(include_context, "prefix", "")
    if original_router is not None:
        for child_route in original_router.routes:
            child_path = getattr(child_route, "path", None)
            if isinstance(child_path, str):
                paths.append(f"{route_prefix}{child_path}")
paths = sorted(set(paths))
print(json.dumps({
    "loaded": {
        "atlas_brain.api": "atlas_brain.api" in sys.modules,
        "atlas_brain.config": "atlas_brain.config" in sys.modules,
        "atlas_brain.reasoning": "atlas_brain.reasoning" in sys.modules,
        "atlas_brain.services.llm": "atlas_brain.services.llm" in sys.modules,
        "atlas_brain.services.embedding": "atlas_brain.services.embedding" in sys.modules,
        "torch": "torch" in sys.modules,
        "pynvml": "pynvml" in sys.modules,
    },
    "paths": paths,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        env=_isolated_eom_subprocess_env(),
        text=True,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])

    assert observed["loaded"] == {
        "atlas_brain.api": False,
        "atlas_brain.config": False,
        "atlas_brain.reasoning": False,
        "atlas_brain.services.llm": False,
        "atlas_brain.services.embedding": False,
        "torch": False,
        "pynvml": False,
    }
    paths = set(observed["paths"])
    assert "/api/v1/ping" in paths
    assert "/api/v1/receivables/ready" in paths
    assert "/api/v1/eom-funnel/leads" in paths
    assert "/api/v1/eom-funnel/leads/{contact_id}/missed-call-attempts" in paths
    assert "/api/v1/eom-funnel/missed-call-recovery-status" in paths
    assert "/api/v1/eom-funnel/leads/{contact_id}/missed-call-recovery/resume" in paths
    assert "/api/v1/eom-funnel/leads/{contact_id}/missed-call-recovery/cancel" in paths
    assert "/api/v1/eom-funnel/leads/{contact_id}/estimate-bookings" in paths
    assert "/api/v1/eom-funnel/leads/{contact_id}/first-clean-bookings" in paths
    assert "/api/v1/eom-funnel/customer-handoffs" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts/{draft_id}" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts/{draft_id}/approve-send" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts/{draft_id}/revoke" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts/{draft_id}/revoke-link" in paths
    assert "/api/v1/eom-funnel/onboarding-drafts/{draft_id}/confirm-sent" in paths
    assert "/api/v1/eom-funnel/public-onboarding/session" in paths
    assert "/api/v1/eom-funnel/public-onboarding/finalize" in paths
    assert "/openapi.json" not in paths
    assert "/docs" not in paths
    assert "/docs/oauth2-redirect" not in paths
    assert "/redoc" not in paths
    assert not any(path.startswith("/api/v1/b2b") for path in paths)
    assert not any(path.startswith("/api/v1/content-ops") for path in paths)


def test_services_package_keeps_llm_registry_compatibility():
    from atlas_brain.services import llm_registry

    assert llm_registry is not None
    assert "openrouter" in llm_registry.list_available()


def test_direct_registry_import_preserves_registered_llm_backends():
    probe = """
import json
import sys

for module_name in list(sys.modules):
    if (
        module_name == "atlas_brain.services.registry"
        or module_name == "atlas_brain.services.llm"
        or module_name.startswith("atlas_brain.services.llm.")
    ):
        sys.modules.pop(module_name, None)

from atlas_brain.services.registry import llm_registry

print(json.dumps({"available": sorted(llm_registry.list_available())}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    available = set(json.loads(result.stdout.strip().splitlines()[-1])["available"])
    assert {"cloud", "ollama", "openrouter"}.issubset(available)


def test_deferred_registry_registration_waits_for_concurrent_first_callers():
    from atlas_brain.services.registry import ServiceRegistry

    class _FakeService:
        pass

    start = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def load_registry():
        nonlocal calls
        with calls_lock:
            calls += 1
        start.set()
        assert release.wait(timeout=2)
        registry.register("fake", _FakeService)

    registry = ServiceRegistry("Fake", registration_loader=load_registry)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(registry.list_available)
        assert start.wait(timeout=2)
        second = executor.submit(registry.list_available)
        release.set()
        assert first.result(timeout=2) == ["fake"]
        assert second.result(timeout=2) == ["fake"]
    assert calls == 1


def test_eom_env_loader_preserves_process_env_and_local_precedence(
    monkeypatch, tmp_path
):
    from atlas_brain import main_eom

    base_key = "ATLAS_EOM_TEST_BASE_ONLY"
    local_key = "ATLAS_EOM_TEST_LOCAL_WINS"
    process_key = "ATLAS_EOM_TEST_PROCESS_WINS"
    local_only_key = "ATLAS_EOM_TEST_LOCAL_ONLY"
    base_secret_key = "ATLAS_EOM_TEST_BASE_SECRET"
    process_secret_key = "ATLAS_EOM_TEST_PROCESS_SECRET"
    base_interpolated_key = "ATLAS_EOM_TEST_BASE_INTERPOLATED"
    process_interpolated_key = "ATLAS_EOM_TEST_PROCESS_INTERPOLATED"
    for key in (
        base_key,
        local_key,
        process_key,
        local_only_key,
        base_secret_key,
        process_secret_key,
        base_interpolated_key,
        process_interpolated_key,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(process_key, "process")
    monkeypatch.setenv(process_secret_key, "process-secret")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"{base_key}=base",
                f"{local_key}=base",
                f"{process_key}=base",
                f"{base_secret_key}=base-secret",
                f"{process_secret_key}=base-secret",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                f"{local_key}=local",
                f"{process_key}=local",
                f"{local_only_key}=local-only",
                f"{base_interpolated_key}=${{{base_secret_key}}}",
                f"{process_interpolated_key}=${{{process_secret_key}}}",
            ]
        ),
        encoding="utf-8",
    )

    main_eom._load_local_env_files(tmp_path)

    assert os.environ[base_key] == "base"
    assert os.environ[local_key] == "local"
    assert os.environ[local_only_key] == "local-only"
    assert os.environ[process_key] == "process"
    assert os.environ[base_interpolated_key] == "base-secret"
    assert os.environ[process_interpolated_key] == "process-secret"


def test_eom_requirements_cover_estimate_booking_calendar_runtime():
    """The booking route loads tools/calendar.py, whose module-level
    `import httpx` must be satisfiable in the slim EOM runtime: the Render
    build installs only requirements.eom.txt, so the httpx pin has to live
    there and match the main requirements pin."""
    eom_requirements = Path("requirements.eom.txt").read_text(encoding="utf-8")
    main_requirements = Path("requirements.txt").read_text(encoding="utf-8")

    eom_pins = {
        line.strip().split("==")[0].split("[")[0].lower(): line.strip()
        for line in eom_requirements.splitlines()
        if "==" in line
    }
    assert "httpx" in eom_pins

    main_httpx_pin = next(
        line.strip()
        for line in main_requirements.splitlines()
        if line.strip().startswith("httpx==")
    )
    assert eom_pins["httpx"] == main_httpx_pin


def test_eom_render_blueprint_maps_database_and_receivables_auth():
    blueprint = yaml.safe_load(Path("render.eom.yaml").read_text(encoding="utf-8"))

    assert blueprint["databases"] == [
        {
            "name": "atlas-eom-postgres",
            "databaseName": "atlas_eom",
            "user": "atlas_eom",
            "ipAllowList": [],
        }
    ]

    [service] = blueprint["services"]
    assert service["type"] == "pserv"
    assert service["name"] == "atlas-eom-api"
    assert service["startCommand"] == (
        "uvicorn atlas_brain.main_eom:app --host 0.0.0.0 --port $PORT"
    )

    env_vars = {item["key"]: item for item in service["envVars"]}
    assert env_vars["ATLAS_DB_ENABLED"] == {
        "key": "ATLAS_DB_ENABLED",
        "value": "true",
    }
    assert env_vars["ATLAS_DB_CONNECTION_STRING"] == {
        "key": "ATLAS_DB_CONNECTION_STRING",
        "fromDatabase": {
            "name": "atlas-eom-postgres",
            "property": "connectionString",
        },
    }
    assert env_vars["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] == {
        "key": "ATLAS_INVOICING_RECEIVABLES_API_ENABLED",
        "value": "false",
    }
    assert env_vars["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"] == {
        "key": "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256",
        "sync": False,
    }
    assert env_vars["ATLAS_EOM_FUNNEL_API_ENABLED"] == {
        "key": "ATLAS_EOM_FUNNEL_API_ENABLED",
        "value": "false",
    }
    assert env_vars["ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256"] == {
        "key": "ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256",
        "sync": False,
    }
    assert env_vars["ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"] == {
        "key": "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING",
        "sync": False,
    }
    assert env_vars["ATLAS_EOM_FUNNEL_MISSED_CALL_RECOVERY_ENABLED"] == {
        "key": "ATLAS_EOM_FUNNEL_MISSED_CALL_RECOVERY_ENABLED",
        "value": "false",
    }
    assert env_vars["ATLAS_EOM_FUNNEL_MISSED_CALL_BOOKING_LINK"] == {
        "key": "ATLAS_EOM_FUNNEL_MISSED_CALL_BOOKING_LINK",
        "sync": False,
    }
    assert env_vars["ATLAS_EOM_FUNNEL_MISSED_CALL_TIMEZONE"] == {
        "key": "ATLAS_EOM_FUNNEL_MISSED_CALL_TIMEZONE",
        "value": "America/Chicago",
    }
    assert env_vars["ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED"] == {
        "key": "ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED",
        "value": "false",
    }
    assert env_vars["ATLAS_EOM_RUN_MIGRATIONS"] == {
        "key": "ATLAS_EOM_RUN_MIGRATIONS",
        "value": "true",
    }
    assert env_vars["ATLAS_TOOLS_CALENDAR_ENABLED"] == {
        "key": "ATLAS_TOOLS_CALENDAR_ENABLED",
        "value": "true",
    }
    for calendar_secret_key in (
        "ATLAS_TOOLS_CALENDAR_CLIENT_ID",
        "ATLAS_TOOLS_CALENDAR_CLIENT_SECRET",
        "ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN",
        "ATLAS_TOOLS_CALENDAR_ID",
    ):
        assert env_vars[calendar_secret_key] == {
            "key": calendar_secret_key,
            "sync": False,
        }
    assert [
        key
        for key in env_vars
        if key.startswith("ATLAS_INVOICING_") and "TOKEN" in key
    ] == ["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"]
    assert [
        key
        for key in env_vars
        if key.startswith("ATLAS_EOM_FUNNEL_") and "TOKEN" in key
    ] == ["ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256"]
    for split_key in (
        "ATLAS_DB_HOST",
        "ATLAS_DB_PORT",
        "ATLAS_DB_DATABASE",
        "ATLAS_DB_USER",
        "ATLAS_DB_PASSWORD",
    ):
        assert split_key not in env_vars


def test_eom_startup_migrations_are_curated_for_receivables_readiness():
    from atlas_brain import main_eom

    assert main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS == (
        "012_appointments",
        "035_contacts",
        "045_invoices",
        "344_receivables_payments",
        "345_receivables_event_key_lookup",
        "368_receivables_payment_check_metadata",
        "369_receivables_payment_receipt_outbox",
        "378_receivables_payment_receipt_delivery",
        "379_receivables_payment_receipt_delivery_recovery",
    )
    assert not any(
        migration.startswith(("066_", "068_", "074_", "076_", "083_", "095_"))
        for migration in main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS
    )


def test_eom_funnel_canonical_crm_config_defaults_fail_closed(monkeypatch):
    for key in (
        "ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED",
        "ATLAS_EOM_FUNNEL_API_ENABLED",
        "ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256",
        "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING",
        "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_ENABLED",
        "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_URL",
        "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_HMAC_SECRET",
        "ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_PREVIOUS_HMAC_SECRET",
        _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV,
    ):
        monkeypatch.delenv(key, raising=False)

    from atlas_brain.eom_api import funnel_auth
    from atlas_brain.eom_api.config import EOMFunnelConfig, EOMProfileConfig
    from atlas_brain.eom_api.funnel_database import (
        validate_eom_funnel_canonical_crm_config,
    )

    profile_defaults = EOMProfileConfig()
    funnel_defaults = EOMFunnelConfig()
    assert profile_defaults.canonical_crm_database_confirmed is False
    assert funnel_defaults.api_enabled is False
    assert funnel_defaults.db_connection_string == ""
    assert funnel_defaults.public_onboarding_enabled is False
    assert funnel_defaults.public_onboarding_url == ""
    assert funnel_defaults.public_onboarding_hmac_secret.get_secret_value() == ""
    assert funnel_defaults.public_onboarding_previous_hmac_secret.get_secret_value() == ""
    validate_eom_funnel_canonical_crm_config(
        funnel_defaults,
        canonical_crm_database_confirmed=False,
    )

    generated = funnel_auth.generate_eom_funnel_service_token()
    enabled_without_confirmation = EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=generated.sha256,
        db_connection_string="postgresql://atlas:secret@crm.internal/atlas",
    )
    with pytest.raises(
        RuntimeError,
        match="ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true",
    ):
        validate_eom_funnel_canonical_crm_config(
            enabled_without_confirmation,
            canonical_crm_database_confirmed=False,
        )

    enabled_without_dsn = EOMFunnelConfig(
        api_enabled=True,
        service_token_sha256=generated.sha256,
        db_connection_string="",
    )
    with pytest.raises(RuntimeError, match="ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"):
        validate_eom_funnel_canonical_crm_config(
            enabled_without_dsn,
            canonical_crm_database_confirmed=True,
        )

    validate_eom_funnel_canonical_crm_config(
        enabled_without_confirmation,
        canonical_crm_database_confirmed=True,
    )


def test_eom_funnel_database_pool_uses_dedicated_canonical_dsn(monkeypatch):
    from atlas_brain.eom_api import funnel_database

    dsn = "postgresql://atlas_funnel:secret@canonical-crm.internal:5432/atlas"
    calls: dict[str, object] = {}

    class _FakePool:
        async def close(self):
            calls["closed"] = True

    async def create_pool(**kwargs):
        calls["create_pool"] = kwargs
        return _FakePool()

    monkeypatch.setattr(funnel_database.asyncpg, "create_pool", create_pool)

    async def drive_pool():
        pool = funnel_database.EOMFunnelDatabasePool(dsn=f" {dsn} ")
        await pool.initialize()
        await pool.close()
        return pool.target_label

    target_label = asyncio.run(drive_pool())

    assert calls["create_pool"] == {
        "dsn": dsn,
        "min_size": 1,
        "max_size": 5,
        "command_timeout": 30,
    }
    assert calls["closed"] is True
    assert "secret" not in target_label
    assert target_label == "dsn=canonical-crm.internal:5432/atlas"


def test_eom_migration_helper_uses_curated_set():
    from atlas_brain import main_eom

    pool = SimpleNamespace(is_initialized=True)
    calls = []

    async def run_migrations(observed_pool, *, only=None):
        calls.append((observed_pool, tuple(only or ())))

    asyncio.run(
        main_eom._apply_eom_receivables_migrations(
            pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert calls == [(pool, main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS)]


def test_eom_missed_call_recovery_migration_helper_uses_funnel_curated_set():
    from atlas_brain import main_eom

    pool = SimpleNamespace(is_initialized=True)
    calls = []

    async def run_migrations(observed_pool, *, only=None):
        calls.append((observed_pool, tuple(only or ())))

    asyncio.run(
        main_eom._apply_eom_missed_call_recovery_migrations(
            pool,
            run_migrations_fn=run_migrations,
        )
    )

    assert calls == [(pool, main_eom.EOM_MISSED_CALL_RECOVERY_READINESS_MIGRATIONS)]
    assert main_eom.EOM_MISSED_CALL_RECOVERY_READINESS_MIGRATIONS == (
        "035_contacts",
        "256_contact_interaction_dedupe",
        "346_contact_lead_pipeline",
        "351_eom_lead_lifecycle_events",
        "366_contacts_customer_type",
        "389_eom_missed_call_recovery",
    )


def test_eom_startup_migration_runner_skips_uninitialized_pool(monkeypatch):
    from atlas_brain import main_eom

    async def fail_migrations(*_args, **_kwargs):
        raise AssertionError("uninitialized EOM DB pool must not run migrations")

    monkeypatch.setattr(
        main_eom,
        "get_db_pool",
        lambda: SimpleNamespace(is_initialized=False),
    )
    monkeypatch.setattr(
        main_eom,
        "_apply_eom_receivables_migrations",
        fail_migrations,
    )

    asyncio.run(main_eom._run_startup_migrations())


def test_eom_funnel_startup_guard_requires_authoritative_datastore(monkeypatch):
    from atlas_brain import main_eom

    class _Pool:
        def __init__(self, *, initialized: bool, schema_ready: bool) -> None:
            self.is_initialized = initialized
            self._schema_ready = schema_ready
            self.queries: list[str] = []

        async def fetchval(self, query: str) -> bool:
            self.queries.append(query)
            return self._schema_ready

    disabled = SimpleNamespace(api_enabled=False)

    def fail_if_looked_up():
        pytest.fail("disabled EOM funnel must not require a database pool")

    monkeypatch.setattr(main_eom, "get_eom_funnel_db_pool", fail_if_looked_up)
    asyncio.run(
        main_eom._require_eom_funnel_data_store(
            disabled,
            database_enabled=False,
        )
    )

    enabled = SimpleNamespace(api_enabled=True)
    ready_pool = _Pool(initialized=True, schema_ready=True)
    monkeypatch.setattr(main_eom, "get_eom_funnel_db_pool", lambda: ready_pool)

    asyncio.run(
        main_eom._require_eom_funnel_data_store(
            enabled,
            database_enabled=True,
        )
    )
    assert "atlas_eom_handoff_owner" in ready_pool.queries[0]
    assert "atlas_nocodb" in ready_pool.queries[0]
    assert "pg_auth_members" in ready_pool.queries[0]
    assert "has_table_privilege" in ready_pool.queries[0]
    assert "has_column_privilege" in ready_pool.queries[0]

    with pytest.raises(RuntimeError, match="authoritative Atlas database"):
        asyncio.run(
            main_eom._require_eom_funnel_data_store(
                enabled,
                database_enabled=False,
            )
        )

    monkeypatch.setattr(
        main_eom,
        "get_eom_funnel_db_pool",
        lambda: _Pool(initialized=False, schema_ready=True),
    )
    with pytest.raises(RuntimeError, match="initialized Atlas database pool"):
        asyncio.run(
            main_eom._require_eom_funnel_data_store(
                enabled,
                database_enabled=True,
            )
        )

    monkeypatch.setattr(
        main_eom,
        "get_eom_funnel_db_pool",
        lambda: _Pool(initialized=True, schema_ready=False),
    )
    with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
        asyncio.run(
            main_eom._require_eom_funnel_data_store(
                enabled,
                database_enabled=True,
            )
        )


def test_shared_eom_funnel_datastore_guard_accepts_injected_pool():
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    class _Pool:
        is_initialized = True

        def __init__(self) -> None:
            self.queries: list[str] = []

        async def fetchval(self, query: str) -> bool:
            self.queries.append(query)
            return True

    pool = _Pool()
    asyncio.run(
        require_eom_funnel_data_store(
            SimpleNamespace(api_enabled=True),
            database_enabled=True,
            get_db_pool_fn=lambda: pool,
        )
    )

    assert len(pool.queries) == 1
    assert "eom_customer_handoffs" in pool.queries[0]
    assert "atlas_eom_handoff_owner" in pool.queries[0]
    assert "atlas_nocodb" in pool.queries[0]


def test_shared_eom_funnel_datastore_guard_keeps_missing_relations_in_verdict():
    from atlas_brain.eom_api.funnel_store import require_eom_funnel_data_store

    class _Pool:
        is_initialized = True

        def __init__(self) -> None:
            self.query = ""

        async def fetchval(self, query: str) -> bool:
            self.query = query
            return False

    pool = _Pool()
    with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
        asyncio.run(
            require_eom_funnel_data_store(
                SimpleNamespace(api_enabled=True),
                database_enabled=True,
                get_db_pool_fn=lambda: pool,
            )
        )

    assert "::regclass" not in pool.query
    assert "WITH readiness_relations AS" in pool.query
    assert "readiness_columns AS" in pool.query
    assert "to_regclass('eom_customer_handoffs') AS handoff_rel" in pool.query
    assert (
        "to_regclass('eom_onboarding_email_drafts') AS onboarding_drafts_rel"
        in pool.query
    )
    assert "contacts_required_columns_ready" in pool.query
    assert "lifecycle_required_columns_ready" in pool.query
    assert "onboarding_drafts_required_columns_ready" in pool.query
    assert "public_onboarding_recovery_ready" in pool.query
    assert "public_onboarding_issuance_ready" in pool.query
    assert "'id', 'draft_id', 'contact_id', 'status', 'revoked_at'" in pool.query
    assert "WHEN NOT readiness_columns.contacts_required_columns_ready THEN FALSE" in pool.query
    assert "AND readiness_columns.lifecycle_required_columns_ready" in pool.query
    assert "WHEN readiness_relations.handoff_rel IS NULL THEN FALSE" in pool.query


def test_database_config_prefers_connection_string_for_asyncpg_kwargs():
    from atlas_brain.storage.config import DatabaseConfig

    dsn = "postgresql://atlas_eom:secret@atlas-eom-postgres:5432/atlas_eom"
    config = DatabaseConfig(
        connection_string=f" {dsn} ",
        host="localhost",
        port=5433,
        database="atlas",
        user="atlas",
        password="split-secret",
        connect_timeout=7.0,
        command_timeout=11.0,
    )

    assert config.dsn == dsn
    assert config.connection_kwargs() == {
        "dsn": dsn,
        "timeout": 7.0,
        "command_timeout": 11.0,
    }
    assert config.connection_kwargs(command_timeout=60) == {
        "dsn": dsn,
        "timeout": 7.0,
        "command_timeout": 60,
    }
    assert "secret" not in config.target_label
    assert config.target_label == "dsn=atlas-eom-postgres:5432/atlas_eom"


def test_database_config_preserves_split_host_port_kwargs_without_connection_string():
    from atlas_brain.storage.config import DatabaseConfig

    config = DatabaseConfig(
        connection_string=" ",
        host="postgres.internal",
        port=6543,
        database="atlas_prod",
        user="atlas_user",
        password="atlas_pass",
        connect_timeout=5.0,
        command_timeout=17.0,
    )

    assert (
        config.dsn
        == "postgresql://atlas_user:atlas_pass@postgres.internal:6543/atlas_prod"
    )
    assert config.connection_kwargs() == {
        "host": "postgres.internal",
        "port": 6543,
        "database": "atlas_prod",
        "user": "atlas_user",
        "password": "atlas_pass",
        "timeout": 5.0,
        "command_timeout": 17.0,
    }
    assert config.connection_kwargs(command_timeout=60)["command_timeout"] == 60


def test_database_pool_uses_configured_connection_kwargs(monkeypatch):
    from atlas_brain.storage.config import DatabaseConfig
    from atlas_brain.storage import database

    dsn = "postgresql://atlas_eom:secret@atlas-eom-postgres:5432/atlas_eom"
    calls: dict[str, dict[str, object]] = {}

    class _FakePool:
        async def close(self):
            calls["closed"] = {}

    async def create_pool(**kwargs):
        calls["create_pool"] = kwargs
        return _FakePool()

    async def connect(**kwargs):
        calls["connect"] = kwargs
        return object()

    config = DatabaseConfig(
        enabled=True,
        connection_string=dsn,
        min_pool_size=1,
        max_pool_size=3,
        connect_timeout=4.0,
        command_timeout=9.0,
    )
    monkeypatch.setattr(database.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(database.asyncpg, "connect", connect)

    async def drive_pool():
        pool = database.DatabasePool()
        await pool.initialize()
        await pool.acquire_raw()
        await pool.close()

    original_settings = database.db_settings
    database.db_settings = config
    try:
        asyncio.run(drive_pool())
    finally:
        database.db_settings = original_settings

    assert calls["create_pool"] == {
        "dsn": dsn,
        "timeout": 4.0,
        "command_timeout": 9.0,
        "min_size": 1,
        "max_size": 3,
    }
    assert calls["connect"] == {
        "dsn": dsn,
        "timeout": 4.0,
        "command_timeout": 60,
    }
    assert calls["closed"] == {}


def test_eom_receivables_runtime_config_accepts_generated_digest_only():
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api.config import EOMInvoicingConfig

    generated = auth.generate_receivables_service_token()
    config = EOMInvoicingConfig(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
    )

    auth.validate_receivables_api_config(config)
    assert (
        auth.receivables_service_token_sha256(generated.token)
        == config.receivables_service_token_sha256
    )
    assert "receivables_service_token" not in EOMInvoicingConfig.model_fields


def test_eom_receivables_token_generation_ignores_legacy_raw_env():
    probe = """
import json
import sys

from atlas_brain.eom_api import auth

generated = auth.generate_receivables_service_token()
print(json.dumps({
    "token": generated.token,
    "sha256": generated.sha256,
    "config_loaded": "atlas_brain.eom_api.config" in sys.modules,
}))
"""
    env = _isolated_eom_subprocess_env()
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"] = (
        "legacy-raw-token-still-present"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])

    assert _generated_receivables_token_oracle(observed["token"])
    assert observed["sha256"] == _sha256_ascii(observed["token"])
    assert observed["config_loaded"] is False


def test_eom_receivables_raw_token_source_admission_matches_casefold_oracle(
    tmp_path,
):
    from atlas_brain.eom_api import config as eom_config

    raw_values = ("", "   ", "caller-side-raw-token")
    unrelated_keys = (
        "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256",
        "ATLAS_EOM_RECEIVABLES_SERVICE_TOKEN",
    )

    case_index = 0
    for source, key, value in product(
        ("process-env", "dotenv"),
        (*_RAW_RECEIVABLES_SERVICE_TOKEN_KEY_CASES, *unrelated_keys),
        raw_values,
    ):
        expected = (
            key.casefold() == _RAW_RECEIVABLES_SERVICE_TOKEN_ENV.casefold()
            and bool(value.strip())
        )
        if source == "process-env":
            observed = eom_config.raw_receivables_service_token_configured(
                environ={key: value},
                env_files=(),
            )
        else:
            case_index += 1
            env_file = tmp_path / f"raw-token-source-{case_index}.env"
            env_file.write_text(f"{key}={value}\n", encoding="utf-8")
            observed = eom_config.raw_receivables_service_token_configured(
                environ={},
                env_files=(env_file,),
            )

        assert observed is expected, (source, key, repr(value))


def test_raw_token_source_admission_preserves_explicit_empty_environ(monkeypatch):
    from atlas_brain.eom_api import config as eom_config

    monkeypatch.setenv(
        _RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
        "operator-receivables-token",
    )
    monkeypatch.setenv(_RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV, "operator-funnel-token")

    assert not eom_config.raw_receivables_service_token_configured(
        environ={},
        env_files=(),
    )
    assert not eom_config.raw_eom_funnel_service_token_configured(
        environ={},
        env_files=(),
    )
    assert eom_config.raw_receivables_service_token_configured(
        environ=None,
        env_files=(),
    )
    assert eom_config.raw_eom_funnel_service_token_configured(
        environ=None,
        env_files=(),
    )


@pytest.mark.parametrize(
    ("raw_token_key", "api_enabled"),
    (
        ("ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN", "true"),
        ("atlas_invoicing_receivables_service_token", "true"),
        ("ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN", "false"),
        (_RAW_RECEIVABLES_SERVICE_TOKEN_KEY_CASES[-1], "false"),
    ),
)
def test_eom_receivables_runtime_config_rejects_raw_token_env_before_projection(
    monkeypatch,
    raw_token_key,
    api_enabled,
):
    from pydantic import ValidationError

    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api.config import EOMInvoicingConfig

    generated = auth.generate_receivables_service_token()
    monkeypatch.setenv("ATLAS_INVOICING_RECEIVABLES_API_ENABLED", api_enabled)
    monkeypatch.setenv(
        "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256",
        generated.sha256,
    )
    monkeypatch.setenv(raw_token_key, generated.token)

    with pytest.raises(
        ValidationError,
        match="Raw EOM receivables bearer token",
    ):
        EOMInvoicingConfig()
    assert "receivables_service_token" not in EOMInvoicingConfig.model_fields


@pytest.mark.parametrize(
    ("raw_token_key", "api_enabled"),
    (
        ("ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN", "true"),
        ("atlas_invoicing_receivables_service_token", "true"),
        ("ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN", "false"),
        (_RAW_RECEIVABLES_SERVICE_TOKEN_KEY_CASES[-1], "false"),
    ),
)
def test_eom_profile_rejects_raw_receivables_token_from_dotenv_before_projection(
    tmp_path,
    raw_token_key,
    api_enabled,
):
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api.config import RAW_RECEIVABLES_SERVICE_TOKEN_ENV

    generated = auth.generate_receivables_service_token()
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"ATLAS_INVOICING_RECEIVABLES_API_ENABLED={api_enabled}",
                f"ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256={generated.sha256}",
                f"{raw_token_key}={generated.token}",
            ]
        ),
        encoding="utf-8",
    )
    env = _isolated_eom_subprocess_env()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", "import atlas_brain.main_eom"],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "Raw EOM receivables bearer token material" in result.stderr
    assert RAW_RECEIVABLES_SERVICE_TOKEN_ENV in result.stderr


def test_eom_receivables_startup_rejects_unsafe_enabled_runtime_config():
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api.config import EOMInvoicingConfig

    generated = auth.generate_receivables_service_token()
    cases = (
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token=generated.token,
                receivables_service_token_sha256=generated.sha256,
            ),
            "Raw EOM receivables bearer token",
        ),
        (
            EOMInvoicingConfig(receivables_api_enabled=True),
            "digest is required",
        ),
        (
            EOMInvoicingConfig(
                receivables_api_enabled=True,
                receivables_service_token_sha256="0" * 63,
            ),
            "lowercase SHA-256",
        ),
        (
            EOMInvoicingConfig(
                receivables_api_enabled=True,
                receivables_service_token_sha256="0" * 64,
            ),
            "placeholder",
        ),
        (
            EOMInvoicingConfig(
                receivables_api_enabled=True,
                receivables_service_token_sha256=auth._token_sha256("change-me"),
            ),
            "placeholder",
        ),
    )
    for config, message in cases:
        with pytest.raises(RuntimeError, match=message):
            auth.validate_receivables_api_config(config)


def test_eom_funnel_raw_token_source_admission_matches_casefold_oracle(
    tmp_path,
):
    from atlas_brain.eom_api import config as eom_config

    raw_values = ("", "   ", "caller-side-funnel-token")
    unrelated_keys = (
        "ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256",
        "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN",
    )

    case_index = 0
    for source, key, value in product(
        ("process-env", ".env", ".env.local"),
        (*_RAW_EOM_FUNNEL_SERVICE_TOKEN_KEY_CASES, *unrelated_keys),
        raw_values,
    ):
        expected = (
            key.casefold() == _RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV.casefold()
            and bool(value.strip())
        )
        if source == "process-env":
            observed = eom_config.raw_eom_funnel_service_token_configured(
                environ={key: value},
                env_files=(),
            )
        else:
            case_index += 1
            env_file = tmp_path / f"raw-funnel-token-source-{case_index}{source}"
            env_file.write_text(f"{key}={value}\n", encoding="utf-8")
            observed = eom_config.raw_eom_funnel_service_token_configured(
                environ={},
                env_files=(env_file,),
            )

        assert observed is expected, (source, key, repr(value))


@pytest.mark.parametrize(
    ("raw_token_key", "api_enabled"),
    (
        ("ATLAS_EOM_FUNNEL_SERVICE_TOKEN", "true"),
        ("atlas_eom_funnel_service_token", "true"),
        ("ATLAS_EOM_FUNNEL_SERVICE_TOKEN", "false"),
        (_RAW_EOM_FUNNEL_SERVICE_TOKEN_KEY_CASES[-1], "false"),
    ),
)
def test_eom_funnel_runtime_config_rejects_raw_token_env_before_projection(
    monkeypatch,
    raw_token_key,
    api_enabled,
):
    from pydantic import ValidationError

    from atlas_brain.eom_api import funnel_auth
    from atlas_brain.eom_api.config import (
        EOMFunnelConfig,
        RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV,
    )

    generated = funnel_auth.generate_eom_funnel_service_token()
    monkeypatch.setenv("ATLAS_EOM_FUNNEL_API_ENABLED", api_enabled)
    monkeypatch.setenv("ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256", generated.sha256)
    monkeypatch.setenv(raw_token_key, generated.token)

    with pytest.raises(
        ValidationError,
        match="Raw EOM funnel bearer token",
    ):
        EOMFunnelConfig()
    assert "service_token" not in EOMFunnelConfig.model_fields
    assert RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV not in EOMFunnelConfig.model_fields


@pytest.mark.parametrize(
    ("env_file_name", "raw_token_key", "api_enabled"),
    (
        (".env", "ATLAS_EOM_FUNNEL_SERVICE_TOKEN", "true"),
        (".env.local", "atlas_eom_funnel_service_token", "true"),
        (".env", "ATLAS_EOM_FUNNEL_SERVICE_TOKEN", "false"),
        (".env.local", _RAW_EOM_FUNNEL_SERVICE_TOKEN_KEY_CASES[-1], "false"),
    ),
)
def test_eom_profile_rejects_raw_funnel_token_from_dotenv_before_projection(
    tmp_path,
    env_file_name,
    raw_token_key,
    api_enabled,
):
    from atlas_brain.eom_api import funnel_auth
    from atlas_brain.eom_api.config import RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV

    generated = funnel_auth.generate_eom_funnel_service_token()
    (tmp_path / env_file_name).write_text(
        "\n".join(
            [
                f"ATLAS_EOM_FUNNEL_API_ENABLED={api_enabled}",
                f"ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256={generated.sha256}",
                f"{raw_token_key}={generated.token}",
            ]
        ),
        encoding="utf-8",
    )
    env = _isolated_eom_subprocess_env()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", "import atlas_brain.main_eom"],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "Raw EOM funnel bearer token material" in result.stderr
    assert RAW_EOM_FUNNEL_SERVICE_TOKEN_ENV in result.stderr


def test_eom_receivables_trusted_config_rejects_bad_token_digests():
    from atlas_brain.eom_api import auth

    for token_digest, message in (
        ("", "digest is required"),
        ("0" * 63, "lowercase SHA-256"),
        ("0" * 64, "placeholder"),
        ("F" * 64, "lowercase SHA-256"),
        ("z" * 64, "lowercase SHA-256"),
        (auth._token_sha256("change-me"), "placeholder"),
    ):
        config = auth.TrustedReceivablesApiConfig(
            receivables_api_enabled=True,
            receivables_service_token_sha256=token_digest,
        )
        with pytest.raises(RuntimeError, match=message):
            auth.validate_receivables_api_config(config)


def test_eom_receivables_token_digest_helper_rejects_legacy_or_short_tokens():
    from atlas_brain.eom_api import auth

    for token in (
        "x" * 23,
        "x" * 24,
        "a" * 24,
        "eomrx_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopq",
        "eomrx_v1_" + ("a" * 42),
        "eomrx_v1_" + ("a" * 44),
        "eomrx_v1_" + ("*" * 43),
    ):
        with pytest.raises(RuntimeError, match="generated|wrong length|invalid"):
            auth.receivables_service_token_sha256(token)


def test_eom_receivables_startup_accepts_generated_service_token():
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    config = auth.trusted_receivables_api_config(generated)

    auth.validate_receivables_api_config(config)
    assert auth.receivables_service_token_sha256(generated.token) == generated.sha256


def test_eom_receivables_request_auth_does_not_rescan_settings_sources(
    monkeypatch,
):
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api import config as eom_config
    from atlas_brain.eom_api.config import EOMInvoicingConfig

    generated = auth.generate_receivables_service_token()
    runtime_config = EOMInvoicingConfig(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
    )

    def fail_settings_rescan(*args, **kwargs):
        raise AssertionError("request auth must not rescan dotenv settings sources")

    monkeypatch.setattr(
        eom_config,
        "raw_receivables_service_token_configured",
        fail_settings_rescan,
    )

    asyncio.run(
        auth.require_receivables_api(
            f"Bearer {generated.token}",
            config=runtime_config,
        )
    )


def test_eom_receivables_bearer_admission_matches_generated_token_grammar(
    monkeypatch,
):
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api import receivables

    class _ReadyService:
        async def is_ready(self):
            return True

        async def is_receipt_delivery_ready(self):
            return True

    app = FastAPI()
    app.include_router(receivables.router)
    runtime_config = {
        "value": auth.TrustedReceivablesApiConfig(
            receivables_api_enabled=True,
            receivables_service_token_sha256=_sha256_ascii(
                f"{_GENERATED_RECEIVABLES_TOKEN_PREFIX}"
                f"{'a' * _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH}"
            ),
        )
    }
    app.dependency_overrides[auth.get_receivables_api_config] = (
        lambda: runtime_config["value"]
    )
    monkeypatch.setattr(
        receivables,
        "get_receivables_service",
        lambda: _ReadyService(),
    )
    _configure_billing_recipient_readiness(monkeypatch, receivables)

    token_prefixes = (
        _GENERATED_RECEIVABLES_TOKEN_PREFIX,
        _GENERATED_RECEIVABLES_TOKEN_PREFIX.upper(),
        "eomrx_v1x",
        "eomrx_v2_",
        "",
    )
    payload_lengths = (0, 1, 42, 43, 44)
    payload_chars = ("A", "z", "0", "_", "-", "*", ".", "=")
    invalid_payload_positions = (
        0,
        _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH // 2,
        _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH - 1,
    )
    homogeneous_payloads = tuple(
        payload_char * payload_length
        for payload_length, payload_char in product(payload_lengths, payload_chars)
    )
    mixed_allowed_payloads = tuple(
        _mixed_generated_payload(payload_length) for payload_length in payload_lengths
    )
    mixed_invalid_payloads = tuple(
        _generated_payload_with_replacement(
            length=_GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH,
            replacement=invalid_char,
            position=position,
        )
        for invalid_char, position in product(
            ("*", ".", "="),
            invalid_payload_positions,
        )
    )
    payloads = tuple(
        dict.fromkeys(
            (
                *homogeneous_payloads,
                *mixed_allowed_payloads,
                *mixed_invalid_payloads,
            )
        )
    )
    accepted_cases = 0

    with TestClient(app) as client:
        for token_prefix, payload in product(
            token_prefixes,
            payloads,
        ):
            token = f"{token_prefix}{payload}"
            should_accept = _generated_receivables_token_oracle(token)
            runtime_config["value"] = auth.TrustedReceivablesApiConfig(
                receivables_api_enabled=True,
                receivables_service_token_sha256=_sha256_ascii(token),
            )

            response = client.get(
                "/receivables/ready",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == (200 if should_accept else 401), (
                token_prefix,
                len(payload),
                payload,
            )
            accepted_cases += int(should_accept)

    expected_accepted_cases = sum(
        1
        for token_prefix, payload in product(token_prefixes, payloads)
        if _generated_receivables_token_oracle(f"{token_prefix}{payload}")
    )
    assert accepted_cases == expected_accepted_cases
    assert any(
        _generated_receivables_token_oracle(
            f"{_GENERATED_RECEIVABLES_TOKEN_PREFIX}{payload}"
        )
        and len(set(payload)) > 1
        for payload in payloads
    )
    assert any(
        len(payload) == _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_LENGTH
        and any(
            char not in _GENERATED_RECEIVABLES_TOKEN_PAYLOAD_CHARS
            for char in payload
        )
        and not _generated_receivables_token_oracle(
            f"{_GENERATED_RECEIVABLES_TOKEN_PREFIX}{payload}"
        )
        for payload in payloads
    )


def test_all_eom_receivables_routes_require_service_auth():
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api import receivables

    for route in receivables.router.routes:
        dependency_calls = [
            dependency.call for dependency in route.dependant.dependencies
        ]
        assert auth.require_receivables_api in dependency_calls


def test_eom_receivables_ready_route_is_fail_closed(monkeypatch):
    from atlas_brain.eom_api import auth
    from atlas_brain.eom_api import receivables
    from atlas_brain.eom_api.config import EOMInvoicingConfig

    class _ReadyService:
        async def is_ready(self):
            return True

        async def is_receipt_delivery_ready(self):
            return True

    app = FastAPI()
    app.include_router(receivables.router)
    generated = auth.generate_receivables_service_token()
    config = EOMInvoicingConfig(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
    )
    valid_token = generated.token
    app.dependency_overrides[auth.get_receivables_api_config] = lambda: config
    monkeypatch.setattr(
        receivables,
        "get_receivables_service",
        lambda: _ReadyService(),
    )
    _configure_billing_recipient_readiness(monkeypatch, receivables)

    client = TestClient(app)
    assert client.get("/receivables/ready").status_code == 401
    assert (
        client.get(
            "/receivables/ready",
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/receivables/ready",
            headers={b"authorization": b"Bearer \xff"},
        ).status_code
        == 401
    )
    nongenerated_token = "operator-supplied-low-entropy-secret"
    app.dependency_overrides[auth.get_receivables_api_config] = (
        lambda: EOMInvoicingConfig(
            receivables_api_enabled=True,
            receivables_service_token_sha256=auth._token_sha256(nongenerated_token),
        )
    )
    assert (
        client.get(
            "/receivables/ready",
            headers={"Authorization": f"Bearer {nongenerated_token}"},
        ).status_code
        == 401
    )
    too_long_generated_format_token = "eomrx_v1_" + ("a" * 44)
    app.dependency_overrides[auth.get_receivables_api_config] = (
        lambda: EOMInvoicingConfig(
            receivables_api_enabled=True,
            receivables_service_token_sha256=auth._token_sha256(
                too_long_generated_format_token
            ),
        )
    )
    assert (
        client.get(
            "/receivables/ready",
            headers={
                "Authorization": f"Bearer {too_long_generated_format_token}"
            },
        ).status_code
        == 401
    )
    app.dependency_overrides[auth.get_receivables_api_config] = lambda: config
    assert (
        client.get(
            "/receivables/ready",
            headers={"Authorization": f"Bearer {valid_token}"},
        ).status_code
        == 200
    )

    app.dependency_overrides[auth.get_receivables_api_config] = lambda: SimpleNamespace(
        receivables_api_enabled=False
    )
    assert (
        client.get(
            "/receivables/ready",
            headers={"Authorization": f"Bearer {valid_token}"},
        ).status_code
        == 503
    )


def test_eom_lifespan_closes_database_when_migration_startup_fails(monkeypatch):
    from atlas_brain import main_eom

    events: list[str] = []

    async def init_database():
        events.append("init")

    async def fail_migrations():
        events.append("migrations")
        raise RuntimeError("migration failed")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            pass

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(main_eom, "_run_startup_migrations", fail_migrations)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", True)
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", False)

    with pytest.raises(RuntimeError, match="migration failed"):
        asyncio.run(drive_lifespan())

    assert events == ["init", "migrations", "close"]


def test_eom_lifespan_initializes_database_without_running_migrations(monkeypatch):
    from atlas_brain import main_eom

    events: list[str] = []

    async def init_database():
        events.append("init")

    async def fail_migrations():
        raise AssertionError("migrations should stay disabled")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            events.append("inside")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(main_eom, "_run_startup_migrations", fail_migrations)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", False)
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", False)

    asyncio.run(drive_lifespan())

    assert events == ["init", "inside", "close"]


def test_eom_lifespan_rejects_enabled_receivables_without_ready_schema(monkeypatch):
    """Independent readiness fence for finding #5: this profile does not
    always run migrations on startup (run_migrations defaults to False), so
    an enabled receivables API must still fail closed on an unready schema
    rather than silently serving requests against it."""
    from atlas_brain import main_eom
    from atlas_brain.eom_api import auth as receivables_auth
    from atlas_brain.services import receivables as receivables_module

    events: list[str] = []
    generated = receivables_auth.generate_receivables_service_token()

    async def init_database():
        events.append("init")

    async def close_database():
        events.append("close")

    async def fake_is_ready(self, conn=None):
        events.append("readiness-check")
        return False

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            events.append("inside")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "get_db_pool", lambda: SimpleNamespace(is_initialized=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", False)
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", True)
    monkeypatch.setattr(
        main_eom.invoicing_settings,
        "receivables_service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", False)
    monkeypatch.setattr(receivables_module.ReceivablesService, "is_ready", fake_is_ready)

    with pytest.raises(main_eom.ReceivablesSchemaUnavailableError):
        asyncio.run(drive_lifespan())

    assert events == ["init", "readiness-check", "close"]
    assert "inside" not in events


def test_eom_lifespan_accepts_enabled_receivables_with_ready_schema(monkeypatch):
    """Negative control for the above: a genuinely ready schema must not be
    blocked by the new fence."""
    from atlas_brain import main_eom
    from atlas_brain.eom_api import auth as receivables_auth
    from atlas_brain.services import receivables as receivables_module

    events: list[str] = []
    generated = receivables_auth.generate_receivables_service_token()

    async def init_database():
        events.append("init")

    async def close_database():
        events.append("close")

    async def fake_is_ready(self, conn=None):
        events.append("readiness-check")
        return True

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            events.append("inside")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "get_db_pool", lambda: SimpleNamespace(is_initialized=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", False)
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", True)
    monkeypatch.setattr(
        main_eom.invoicing_settings,
        "receivables_service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", False)
    monkeypatch.setattr(receivables_module.ReceivablesService, "is_ready", fake_is_ready)

    asyncio.run(drive_lifespan())

    assert events == ["init", "readiness-check", "inside", "close"]


def test_eom_lifespan_closes_generic_database_when_funnel_close_fails(monkeypatch):
    from atlas_brain import main_eom
    from atlas_brain.eom_api import funnel_auth

    events: list[str] = []
    generated = funnel_auth.generate_eom_funnel_service_token()

    class _Pool:
        is_initialized = True

        async def fetchval(self, query: str) -> bool:
            events.append("datastore-check")
            return True

        def transaction(self):
            return _PoolTransaction(self)

        async def fetch(self, *_args):
            return []

    class _PoolTransaction:
        def __init__(self, pool) -> None:
            self._pool = pool

        async def __aenter__(self):
            return self._pool

        async def __aexit__(self, *_args) -> bool:
            return False

    async def init_database():
        events.append("init")

    async def init_eom_funnel_database(config=None):
        events.append("funnel-init")

    async def close_eom_funnel_database():
        events.append("funnel-close")
        raise RuntimeError("funnel close failed")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            events.append("inside")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "get_eom_funnel_db_pool", lambda: _Pool())
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(
        main_eom,
        "init_eom_funnel_database",
        init_eom_funnel_database,
    )
    monkeypatch.setattr(
        main_eom,
        "close_eom_funnel_database",
        close_eom_funnel_database,
    )
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", False)
    monkeypatch.setattr(
        main_eom.eom_profile_settings,
        "canonical_crm_database_confirmed",
        True,
    )
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", True)
    monkeypatch.setattr(
        main_eom.funnel_settings,
        "service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(
        main_eom.funnel_settings,
        "db_connection_string",
        "postgresql://atlas_funnel:secret@canonical-crm.internal/atlas",
    )

    with pytest.raises(RuntimeError, match="funnel close failed"):
        asyncio.run(drive_lifespan())

    assert events == [
        "init",
        "funnel-init",
        "datastore-check",
        "datastore-check",
        "inside",
        "funnel-close",
        "close",
    ]


def test_eom_lifespan_rejects_enabled_funnel_missing_digest_before_db_work(
    monkeypatch,
):
    from atlas_brain import main_eom

    events: list[str] = []

    async def init_database():
        events.append("init")

    async def init_eom_funnel_database(config=None):
        events.append("funnel-init")

    async def run_migrations():
        events.append("migrations")

    async def close_eom_funnel_database():
        events.append("funnel-close")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            raise AssertionError("enabled funnel without digest must not serve")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(main_eom, "_run_startup_migrations", run_migrations)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", True)
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", True)
    monkeypatch.setattr(main_eom.funnel_settings, "service_token_sha256", "")

    with pytest.raises(RuntimeError, match="ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256"):
        asyncio.run(drive_lifespan())
    assert events == []


@pytest.mark.parametrize(
    ("canonical_confirmed", "dsn", "message"),
    (
        (
            False,
            "postgresql://atlas_funnel:secret@canonical-crm.internal/atlas",
            "ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true",
        ),
        (True, " ", "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"),
    ),
)
def test_eom_lifespan_rejects_enabled_funnel_canonical_crm_config_before_db_work(
    monkeypatch,
    canonical_confirmed,
    dsn,
    message,
):
    from atlas_brain import main_eom
    from atlas_brain.eom_api import funnel_auth

    events: list[str] = []
    generated = funnel_auth.generate_eom_funnel_service_token()

    async def init_database():
        events.append("init")

    async def init_eom_funnel_database(config=None):
        events.append("funnel-init")

    async def run_migrations():
        events.append("migrations")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            raise AssertionError("enabled funnel with bad canonical config must not serve")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(
        main_eom,
        "init_eom_funnel_database",
        init_eom_funnel_database,
    )
    monkeypatch.setattr(main_eom, "_run_startup_migrations", run_migrations)
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", True)
    monkeypatch.setattr(
        main_eom.eom_profile_settings,
        "canonical_crm_database_confirmed",
        canonical_confirmed,
    )
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", True)
    monkeypatch.setattr(
        main_eom.funnel_settings,
        "service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(main_eom.funnel_settings, "db_connection_string", dsn)

    with pytest.raises(RuntimeError, match=message):
        asyncio.run(drive_lifespan())
    assert events == []


def test_eom_lifespan_rejects_funnel_datastore_before_migrations(
    monkeypatch,
):
    from atlas_brain import main_eom
    from atlas_brain.eom_api import funnel_auth

    events: list[str] = []
    generated = funnel_auth.generate_eom_funnel_service_token()

    class _Pool:
        is_initialized = True

        async def fetchval(self, query: str) -> bool:
            events.append("datastore-check")
            assert "eom_customer_handoffs" in query
            assert "atlas_eom_handoff_owner" in query
            return False

    async def init_database():
        events.append("init")

    async def init_eom_funnel_database(config=None):
        events.append("funnel-init")

    async def run_migrations():
        events.append("migrations")

    async def close_eom_funnel_database():
        events.append("funnel-close")

    async def close_database():
        events.append("close")

    async def drive_lifespan():
        async with main_eom.lifespan(FastAPI()):
            raise AssertionError("enabled funnel without datastore must not serve")

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=True))
    monkeypatch.setattr(main_eom, "get_eom_funnel_db_pool", lambda: _Pool())
    monkeypatch.setattr(main_eom, "init_database", init_database)
    monkeypatch.setattr(
        main_eom,
        "init_eom_funnel_database",
        init_eom_funnel_database,
    )
    monkeypatch.setattr(main_eom, "_run_startup_migrations", run_migrations)
    monkeypatch.setattr(
        main_eom,
        "close_eom_funnel_database",
        close_eom_funnel_database,
    )
    monkeypatch.setattr(main_eom, "close_database", close_database)
    monkeypatch.setattr(main_eom.eom_profile_settings, "run_migrations", True)
    monkeypatch.setattr(
        main_eom.eom_profile_settings,
        "canonical_crm_database_confirmed",
        True,
    )
    monkeypatch.setattr(main_eom.invoicing_settings, "receivables_api_enabled", False)
    monkeypatch.setattr(main_eom.funnel_settings, "api_enabled", True)
    monkeypatch.setattr(
        main_eom.funnel_settings,
        "service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(
        main_eom.funnel_settings,
        "db_connection_string",
        "postgresql://atlas_funnel:secret@canonical-crm.internal/atlas",
    )

    with pytest.raises(RuntimeError, match="CRM lifecycle and handoff schema"):
        asyncio.run(drive_lifespan())
    assert events == ["init", "funnel-init", "datastore-check", "funnel-close", "close"]


def test_eom_profile_ping_is_database_independent(monkeypatch):
    from atlas_brain import main_eom
    from atlas_brain.main_eom import app

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=False))
    original_enabled = main_eom.invoicing_settings.receivables_api_enabled
    original_funnel_enabled = main_eom.funnel_settings.api_enabled
    main_eom.invoicing_settings.receivables_api_enabled = False
    main_eom.funnel_settings.api_enabled = False

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/ping")
    finally:
        main_eom.invoicing_settings.receivables_api_enabled = original_enabled
        main_eom.funnel_settings.api_enabled = original_funnel_enabled

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "profile": "eom"}


def test_eom_profile_reaches_receivables_ready_through_real_app(tmp_path):
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    probe = """
import json
import os

from fastapi.testclient import TestClient

from atlas_brain import main_eom
from atlas_brain.eom_api import receivables


class _ReadyService:
    async def is_ready(self):
        return True

    async def is_receipt_delivery_ready(self):
        return True


receivables.get_receivables_service = lambda: _ReadyService()
if main_eom.app.dependency_overrides:
    raise AssertionError("auth dependency must not be overridden")

with TestClient(main_eom.app) as client:
    response = client.get(
        "/api/v1/receivables/ready",
        headers={
            "Authorization": f"Bearer {os.environ['EOM_TEST_CALLER_TOKEN']}"
        },
    )

print(json.dumps({
    "status_code": response.status_code,
    "body": response.json(),
    "env_projected_enabled": main_eom.invoicing_settings.receivables_api_enabled,
    "env_projected_digest": (
        main_eom.invoicing_settings.receivables_service_token_sha256
        == os.environ["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"]
    ),
    "dependency_overrides": len(main_eom.app.dependency_overrides),
}))
"""
    env = _isolated_eom_subprocess_env()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "true"
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"] = generated.sha256
    env["EOM_TEST_CALLER_TOKEN"] = generated.token

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "status_code": 503,
        # A receipt-aware payment route cannot accept a new payment without
        # its canonical-contact database, so an unconfigured profile must
        # advertise that operationally rather than report readiness.
        "body": {
            "detail": {
                "code": "billing_recipients_unavailable",
                "message": (
                    "The canonical EOM contact database is not configured; set "
                    "ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING to use billing "
                    "recipients."
                ),
            }
        },
        "env_projected_enabled": True,
        "env_projected_digest": True,
        "dependency_overrides": 0,
    }


def test_eom_profile_reaches_private_funnel_handoff_through_real_app(tmp_path):
    from atlas_brain.eom_api import funnel_auth

    generated = funnel_auth.generate_eom_funnel_service_token()
    approval_key = "office-handoff-real-app-0001"
    probe = """
import json
import os

from fastapi.testclient import TestClient

from atlas_brain import main_eom
from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_database as funnel_db


class _ReadyPool:
    is_initialized = True

    def __init__(self):
        self.queries = []
        self.closed = 0

    async def fetchval(self, query):
        self.queries.append(query)
        return True

    def transaction(self):
        return _PoolTransaction(self)

    async def fetch(self, *_args):
        return []

    async def close(self):
        self.closed += 1


class _PoolTransaction:
    def __init__(self, pool):
        self._pool = pool

    async def __aenter__(self):
        return self._pool

    async def __aexit__(self, *_args):
        return False


pool = _ReadyPool()

async def init_database():
    return None


async def init_eom_funnel_database(config=None):
    return None


async def close_database():
    return None


async def finalize_eom_customer_handoff(crm, handoff):
    if getattr(crm, "_pool_override", None) is not pool:
        raise AssertionError("funnel route must use the dedicated funnel CRM pool")
    return {
        "handoff_id": "b4fef3b3-a2bd-44e5-aac4-67176270c173",
        "contact_id": str(handoff.contact_id),
        "tracker_customer_id": handoff.tracker_customer_id,
        "tracker_site_id": handoff.tracker_site_id,
        "approval_key": handoff.approval_key,
        "idempotent": False,
    }


funnel_db._eom_funnel_db_pool = pool
main_eom.get_db_pool = lambda: pool
main_eom.get_eom_funnel_db_pool = lambda: pool
main_eom.init_database = init_database
main_eom.init_eom_funnel_database = init_eom_funnel_database
main_eom.close_database = close_database
main_eom.eom_profile_settings.run_migrations = False
funnel_mod.finalize_eom_customer_handoff = finalize_eom_customer_handoff

if main_eom.app.dependency_overrides:
    raise AssertionError("auth dependency must not be overridden")

with TestClient(main_eom.app) as client:
    response = client.post(
        "/api/v1/eom-funnel/customer-handoffs",
        headers={
            "Authorization": f"Bearer {os.environ['EOM_TEST_CALLER_TOKEN']}",
            "X-EOM-Actor": "Juan Canfield",
            "X-EOM-Actor-ID": "1",
            "Idempotency-Key": os.environ["EOM_TEST_APPROVAL_KEY"],
        },
        json={
            "contact_id": "11111111-1111-1111-1111-111111111111",
            "tracker_customer_id": 12,
            "tracker_site_id": 24,
        },
    )

print(json.dumps({
    "status_code": response.status_code,
    "body": response.json(),
    "env_projected_enabled": main_eom.funnel_settings.api_enabled,
    "env_projected_digest": (
        main_eom.funnel_settings.service_token_sha256
        == os.environ["ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256"]
    ),
    "dependency_overrides": len(main_eom.app.dependency_overrides),
    "startup_guard_queries": len(pool.queries),
    "funnel_pool_closed": pool.closed,
}))
"""
    env = _isolated_eom_subprocess_env()
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    env["ATLAS_DB_ENABLED"] = "true"
    env["ATLAS_EOM_RUN_MIGRATIONS"] = "false"
    env["ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED"] = "true"
    env["ATLAS_EOM_FUNNEL_API_ENABLED"] = "true"
    env["ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256"] = generated.sha256
    env["ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING"] = (
        "postgresql://atlas_funnel:secret@canonical-crm.internal/atlas"
    )
    env["EOM_TEST_CALLER_TOKEN"] = generated.token
    env["EOM_TEST_APPROVAL_KEY"] = approval_key

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "status_code": 201,
        "body": {
            "success": True,
            "handoff_id": "b4fef3b3-a2bd-44e5-aac4-67176270c173",
            "contact_id": "11111111-1111-1111-1111-111111111111",
            "tracker_customer_id": 12,
            "tracker_site_id": 24,
            "approval_key": approval_key,
            "idempotent": False,
        },
        "env_projected_enabled": True,
        "env_projected_digest": True,
        "dependency_overrides": 0,
        "startup_guard_queries": 2,
        "funnel_pool_closed": 1,
    }


def test_full_app_starts_with_receivables_digest_only(tmp_path):
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    probe = (
        """
import json
import os

from fastapi.testclient import TestClient

from atlas_brain import main

"""
        + _route_paths_for_app_expr("main.app")
        + """
paths = _route_paths(main.app)

with TestClient(main.app) as client:
    response = client.get("/.well-known/security.txt")

print(json.dumps({
    "status_code": response.status_code,
    "lead_intake_route_mounted": "/api/v1/leads/intake" in paths,
    "env_projected_enabled": main.settings.invoicing.receivables_api_enabled,
    "env_projected_digest": (
        main.settings.invoicing.receivables_service_token_sha256
        == os.environ["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"]
    ),
}))
"""
    )
    env = _atlas_subprocess_env()
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "true"
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"] = generated.sha256

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "status_code": 200,
        "lead_intake_route_mounted": True,
        "env_projected_enabled": True,
        "env_projected_digest": True,
    }


def test_full_app_rejects_raw_receivables_token_with_single_digest_message(tmp_path):
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    env = _atlas_subprocess_env()
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "true"
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"] = generated.sha256
    env["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN"] = generated.token

    result = subprocess.run(
        [sys.executable, "-c", "import atlas_brain.main"],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "Raw EOM receivables bearer token material" in result.stderr
    assert _RAW_RECEIVABLES_SERVICE_TOKEN_ENV in result.stderr
    assert "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN is required" not in result.stderr


def test_full_app_rejects_enabled_receivables_without_digest_once(tmp_path):
    probe = """
from fastapi.testclient import TestClient

from atlas_brain import main


with TestClient(main.app):
    pass
"""
    env = _atlas_subprocess_env()
    env["ATLAS_INVOICING_RECEIVABLES_API_ENABLED"] = "true"

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )

    assert result.returncode != 0
    assert "Receivables service token digest is required" in result.stderr
    assert "ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN is required" not in result.stderr
