import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _isolated_eom_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ATLAS_INVOICING_"):
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


def test_eom_receivables_startup_rejects_raw_or_missing_token_digest():
    from atlas_brain.eom_api import auth

    for token_digest, message in (
        ("", "SERVICE_TOKEN_SHA256"),
        ("0" * 63, "lowercase SHA-256"),
        ("F" * 64, "lowercase SHA-256"),
        ("z" * 64, "lowercase SHA-256"),
        (auth._token_sha256("change-me"), "placeholder"),
    ):
        config = SimpleNamespace(
            receivables_api_enabled=True,
            receivables_service_token="eomrx_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopq",
            receivables_service_token_sha256=token_digest,
            receivables_service_token_provenance_key="",
            receivables_service_token_provenance="",
        )
        with pytest.raises(RuntimeError, match=message):
            auth.validate_receivables_api_config(config)


def test_eom_receivables_startup_rejects_digest_without_generated_provenance():
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    for config, message in (
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token_sha256=generated.sha256,
                receivables_service_token_provenance_key="",
                receivables_service_token_provenance=generated.provenance,
            ),
            "PROVENANCE_KEY",
        ),
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token_sha256=generated.sha256,
                receivables_service_token_provenance_key=generated.provenance_key,
                receivables_service_token_provenance="",
            ),
            "PROVENANCE",
        ),
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token_sha256=generated.sha256,
                receivables_service_token_provenance_key="not-generated",
                receivables_service_token_provenance=generated.provenance,
            ),
            "provenance key",
        ),
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token_sha256=generated.sha256,
                receivables_service_token_provenance_key=generated.provenance_key,
                receivables_service_token_provenance="0" * 64,
            ),
            "provenance",
        ),
        (
            SimpleNamespace(
                receivables_api_enabled=True,
                receivables_service_token_sha256=auth._token_sha256("password123"),
                receivables_service_token_provenance_key=generated.provenance_key,
                receivables_service_token_provenance=generated.provenance,
            ),
            "not bound",
        ),
    ):
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
        "eomrx_v1_" + ("*" * 43),
    ):
        with pytest.raises(RuntimeError, match="generated|too short|invalid"):
            auth.receivables_service_token_sha256(token)


def test_eom_receivables_startup_accepts_generated_service_token():
    from atlas_brain.eom_api import auth

    generated = auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
        receivables_service_token_provenance_key=generated.provenance_key,
        receivables_service_token_provenance=generated.provenance,
    )

    auth.validate_receivables_api_config(config)
    assert auth.receivables_service_token_sha256(generated.token) == generated.sha256


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

    class _ReadyService:
        async def is_ready(self):
            return True

    app = FastAPI()
    app.include_router(receivables.router)
    generated = auth.generate_receivables_service_token()
    config = SimpleNamespace(
        receivables_api_enabled=True,
        receivables_service_token_sha256=generated.sha256,
        receivables_service_token_provenance_key=generated.provenance_key,
        receivables_service_token_provenance=generated.provenance,
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
    assert (
        client.get(
            "/receivables/ready",
            headers={"Authorization": f"Bearer {valid_token}"},
        ).status_code
        == 200
    )

    config.receivables_api_enabled = False
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


def test_eom_profile_ping_is_database_independent(monkeypatch):
    from atlas_brain import main_eom
    from atlas_brain.main_eom import app
    from atlas_brain.eom_api.config import invoicing_settings

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=False))
    monkeypatch.setattr(invoicing_settings, "receivables_api_enabled", False)

    with TestClient(app) as client:
        response = client.get("/api/v1/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "profile": "eom"}


def test_eom_profile_reaches_receivables_ready_through_real_app(monkeypatch):
    from atlas_brain.eom_api import auth
    from atlas_brain import main_eom
    from atlas_brain.main_eom import app
    from atlas_brain.eom_api import receivables
    from atlas_brain.eom_api.config import invoicing_settings

    class _ReadyService:
        async def is_ready(self):
            return True

    monkeypatch.setattr(main_eom, "db_settings", SimpleNamespace(enabled=False))
    monkeypatch.setattr(invoicing_settings, "receivables_api_enabled", True)
    generated = auth.generate_receivables_service_token()
    valid_token = generated.token
    monkeypatch.setattr(
        invoicing_settings,
        "receivables_service_token_sha256",
        generated.sha256,
    )
    monkeypatch.setattr(
        invoicing_settings,
        "receivables_service_token_provenance_key",
        generated.provenance_key,
    )
    monkeypatch.setattr(
        invoicing_settings,
        "receivables_service_token_provenance",
        generated.provenance,
    )
    monkeypatch.setattr(
        receivables,
        "get_receivables_service",
        lambda: _ReadyService(),
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/receivables/ready",
            headers={"Authorization": f"Bearer {valid_token}"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
