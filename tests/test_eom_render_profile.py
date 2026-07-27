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


def _isolated_eom_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith("ATLAS_INVOICING_"):
            env.pop(key, None)
    env["ATLAS_DB_ENABLED"] = "false"
    return env


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
    assert env_vars["ATLAS_EOM_RUN_MIGRATIONS"] == {
        "key": "ATLAS_EOM_RUN_MIGRATIONS",
        "value": "true",
    }
    assert [
        key
        for key in env_vars
        if key.startswith("ATLAS_INVOICING_") and "TOKEN" in key
    ] == ["ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256"]
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
    )
    assert not any(
        migration.startswith(("066_", "068_", "074_", "076_", "083_", "095_"))
        for migration in main_eom.EOM_RECEIVABLES_READINESS_MIGRATIONS
    )


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
    from atlas_brain.eom_api.config import (
        EOMInvoicingConfig,
        RAW_RECEIVABLES_SERVICE_TOKEN_ENV,
    )

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

    asyncio.run(drive_lifespan())

    assert events == ["init", "inside", "close"]


def test_eom_profile_ping_is_database_independent(monkeypatch):
    from atlas_brain import main_eom
    from atlas_brain.main_eom import app

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=False))
    original_enabled = main_eom.invoicing_settings.receivables_api_enabled
    main_eom.invoicing_settings.receivables_api_enabled = False

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/ping")
    finally:
        main_eom.invoicing_settings.receivables_api_enabled = original_enabled

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
        "status_code": 200,
        "body": {"status": "ready"},
        "env_projected_enabled": True,
        "env_projected_digest": True,
        "dependency_overrides": 0,
    }
