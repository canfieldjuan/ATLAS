from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import uuid

import pytest

from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "setup_content_ops_zendesk_credentials.py"
)
ACCOUNT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CREDENTIAL_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
SECRET_TOKEN = "secret-token-value"
DATABASE_URL = "postgres://user:database-secret@example.test/db"
script = None


def setup_module() -> None:
    global script
    script = _load_script_module()


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "setup_content_ops_zendesk_credentials_for_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    values = {
        "database_url": DATABASE_URL,
        "account_id": str(ACCOUNT_ID),
        "label": "Primary",
        "expected_zendesk_base_url": "",
        "dry_run": False,
        "confirm_write": False,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _credentials(**overrides) -> ZendeskMacroCredentials:
    values = {
        "email": "agent@example.com",
        "api_token": SECRET_TOKEN,
        "subdomain": "acme",
        "base_url": "",
    }
    values.update(overrides)
    return ZendeskMacroCredentials(**values)


class _Pool:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.closed = False
        self.close_error = close_error

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Record:
    id = CREDENTIAL_ID
    account_id = ACCOUNT_ID
    api_token_prefix = SECRET_TOKEN[:8]
    subdomain = "acme"
    base_url = "https://acme.zendesk.com"
    label = "Primary"


@pytest.mark.asyncio
async def test_dry_run_validates_config_without_database_access() -> None:
    pool_calls = 0

    async def pool_factory(database_url: str):
        nonlocal pool_calls
        pool_calls += 1
        raise AssertionError("dry-run must not open a database pool")

    code, payload = await script.run_setup(
        _args(
            database_url="",
            dry_run=True,
            expected_zendesk_base_url="https://acme.zendesk.com",
        ),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
    )

    assert code == 0
    assert payload == {
        "ok": True,
        "dry_run": True,
        "would_write": True,
        "account_id": str(ACCOUNT_ID),
        "zendesk_base_url": "https://acme.zendesk.com",
        "label": "Primary",
        "api_token_prefix": SECRET_TOKEN[:8],
    }
    assert pool_calls == 0
    assert SECRET_TOKEN not in _payload_text(payload)
    assert DATABASE_URL not in _payload_text(payload)


@pytest.mark.asyncio
async def test_dry_run_suppresses_complete_short_api_token() -> None:
    credentials = _credentials(api_token="short")

    code, payload = await script.run_setup(
        _args(database_url="", dry_run=True),
        credentials_provider=lambda: credentials,
    )

    assert code == 0
    assert payload["api_token_prefix"] == ""
    assert "short" not in _payload_text(payload)


@pytest.mark.asyncio
async def test_missing_config_fails_before_database_access() -> None:
    async def pool_factory(database_url: str):
        raise AssertionError("missing config must fail before database access")

    code, payload = await script.run_setup(
        _args(dry_run=True),
        pool_factory=pool_factory,
        credentials_provider=lambda: None,
    )

    assert code == 1
    assert payload["reason"] == "zendesk_credentials_incomplete"
    assert payload["account_id"] == str(ACCOUNT_ID)
    assert SECRET_TOKEN not in _payload_text(payload)
    assert DATABASE_URL not in _payload_text(payload)


@pytest.mark.asyncio
async def test_expected_zendesk_guard_fails_before_database_access() -> None:
    async def pool_factory(database_url: str):
        raise AssertionError("endpoint mismatch must fail before database access")

    code, payload = await script.run_setup(
        _args(
            dry_run=True,
            expected_zendesk_base_url="https://other.zendesk.com",
        ),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
    )

    assert code == 1
    assert payload["reason"] == "unexpected_zendesk_base_url"
    assert payload["expected_zendesk_base_url"] == "https://other.zendesk.com"
    assert payload["observed_zendesk_base_url"] == "https://acme.zendesk.com"
    assert SECRET_TOKEN not in _payload_text(payload)


@pytest.mark.asyncio
async def test_write_requires_confirmation_before_database_access() -> None:
    async def pool_factory(database_url: str):
        raise AssertionError("missing confirmation must fail before database access")

    code, payload = await script.run_setup(
        _args(confirm_write=False),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
    )

    assert code == script.SKIPPED_EXIT
    assert payload["skipped"] is True
    assert payload["not_run_reason"] == "missing_confirm_write"
    assert SECRET_TOKEN not in _payload_text(payload)
    assert DATABASE_URL not in _payload_text(payload)


@pytest.mark.asyncio
async def test_write_delegates_to_credential_service_and_closes_pool() -> None:
    pool = _Pool()
    calls: list[dict] = []

    async def pool_factory(database_url: str):
        calls.append({"database_url": database_url, "stage": "pool"})
        return pool

    async def upsert(pool_arg, **kwargs):
        calls.append({"pool": pool_arg, **kwargs})
        return _Record()

    code, payload = await script.run_setup(
        _args(confirm_write=True, label=" Primary "),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
        upsert_func=upsert,
    )

    assert code == 0
    assert calls[0] == {"database_url": DATABASE_URL, "stage": "pool"}
    assert calls[1] == {
        "pool": pool,
        "account_id": ACCOUNT_ID,
        "email": "agent@example.com",
        "api_token": SECRET_TOKEN,
        "subdomain": "acme",
        "base_url": "",
        "label": "Primary",
    }
    assert pool.closed is True
    assert payload["credential_id"] == str(CREDENTIAL_ID)
    assert payload["api_token_prefix"] == SECRET_TOKEN[:8]
    assert SECRET_TOKEN not in _payload_text(payload)
    assert DATABASE_URL not in _payload_text(payload)
    assert "agent@example.com" not in _payload_text(payload)


@pytest.mark.asyncio
async def test_close_failure_after_success_keeps_success_payload_sanitized() -> None:
    pool = _Pool(close_error=RuntimeError(f"close leaked {DATABASE_URL}"))

    async def pool_factory(database_url: str):
        return pool

    async def upsert(pool_arg, **kwargs):
        return _Record()

    code, payload = await script.run_setup(
        _args(confirm_write=True),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
        upsert_func=upsert,
    )

    text = _payload_text(payload)
    assert code == 0
    assert payload["ok"] is True
    assert payload["warnings"] == ["database_pool_close_failed"]
    assert pool.closed is True
    assert DATABASE_URL not in text
    assert SECRET_TOKEN not in text


@pytest.mark.asyncio
async def test_pool_creation_failure_output_is_sanitized() -> None:
    async def pool_factory(database_url: str):
        raise RuntimeError(f"could not connect to {DATABASE_URL}")

    code, payload = await script.run_setup(
        _args(confirm_write=True),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
    )

    text = _payload_text(payload)
    assert code == 1
    assert payload["reason"] == "database_pool_unavailable"
    assert DATABASE_URL not in text
    assert SECRET_TOKEN not in text


@pytest.mark.asyncio
async def test_validation_error_output_is_sanitized_and_pool_is_closed() -> None:
    pool = _Pool()

    async def pool_factory(database_url: str):
        return pool

    async def upsert(pool_arg, **kwargs):
        raise ValueError(f"bad token {SECRET_TOKEN} for {DATABASE_URL}")

    code, payload = await script.run_setup(
        _args(confirm_write=True),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
        upsert_func=upsert,
    )

    text = _payload_text(payload)
    assert code == 1
    assert payload["reason"] == "zendesk_credentials_invalid"
    assert pool.closed is True
    assert SECRET_TOKEN not in text
    assert DATABASE_URL not in text


@pytest.mark.asyncio
async def test_encryption_setup_error_is_distinct_and_sanitized() -> None:
    pool = _Pool()

    async def pool_factory(database_url: str):
        return pool

    async def upsert(pool_arg, **kwargs):
        raise ValueError(
            f"ATLAS_SAAS_BYOK_ENCRYPTION_KEK missing for {DATABASE_URL} {SECRET_TOKEN}"
        )

    code, payload = await script.run_setup(
        _args(confirm_write=True),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
        upsert_func=upsert,
    )

    text = _payload_text(payload)
    assert code == 1
    assert payload["reason"] == "credential_encryption_unavailable"
    assert payload["message"] == "Credential encryption is not configured correctly."
    assert pool.closed is True
    assert "ATLAS_SAAS_BYOK_ENCRYPTION_KEK" not in text
    assert SECRET_TOKEN not in text
    assert DATABASE_URL not in text


@pytest.mark.asyncio
async def test_write_failure_output_is_sanitized_and_pool_is_closed() -> None:
    pool = _Pool(close_error=RuntimeError(f"close leaked {DATABASE_URL}"))

    async def pool_factory(database_url: str):
        return pool

    async def upsert(pool_arg, **kwargs):
        raise RuntimeError(
            f"failed with {SECRET_TOKEN} {DATABASE_URL} ciphertext kid-1"
        )

    code, payload = await script.run_setup(
        _args(confirm_write=True),
        pool_factory=pool_factory,
        credentials_provider=_credentials,
        upsert_func=upsert,
    )

    text = _payload_text(payload)
    assert code == 1
    assert payload["reason"] == "zendesk_credentials_write_failed"
    assert payload["warnings"] == ["database_pool_close_failed"]
    assert pool.closed is True
    assert SECRET_TOKEN not in text
    assert DATABASE_URL not in text
    assert "ciphertext" not in text
    assert "kid-1" not in text


def test_cli_import_does_not_require_asyncpg() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        import importlib.util
        from pathlib import Path

        class BlockAsyncpg:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'asyncpg' or fullname.startswith('asyncpg.'):
                    raise ModuleNotFoundError("No module named 'asyncpg'")
                return None

        sys.meta_path.insert(0, BlockAsyncpg())
        path = Path({str(SCRIPT_PATH)!r})
        spec = importlib.util.spec_from_file_location(
            'setup_content_ops_zendesk_credentials_block_asyncpg_test',
            path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        print(module.SKIPPED_EXIT)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(script.SKIPPED_EXIT) in result.stdout


def _payload_text(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)
