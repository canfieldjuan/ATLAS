from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import sys
from typing import Any
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_content_ops_deflection_env.py"
SPEC = importlib.util.spec_from_file_location("prepare_content_ops_deflection_env", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_env = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_env
SPEC.loader.exec_module(prepare_env)


ACCOUNT_ID = "11111111-1111-4111-8111-111111111111"


class FakeResponse:
    def __init__(self, status: int, payload: Any):
        self.status = status
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


def _fake_open(sequence: list[tuple[int, Any]], calls: list[dict[str, Any]]):
    def _open(request, *, timeout):
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        calls.append({
            "url": request.full_url,
            "method": request.get_method(),
            "headers": dict(request.header_items()),
            "body": body,
            "timeout": timeout,
        })
        status, payload = sequence.pop(0)
        return FakeResponse(status, payload)

    return _open


def _login_payload(token: str = "jwt.secret.value") -> dict[str, Any]:
    return {
        "access_token": token,
        "refresh_token": "refresh.secret.value",
        "token_type": "bearer",
    }


def _me_payload(
    *,
    account_id: str = ACCOUNT_ID,
    product: str = "b2b_retention",
    plan: str = "b2b_growth",
    plan_status: str = "active",
) -> dict[str, Any]:
    return {
        "user_id": "22222222-2222-4222-8222-222222222222",
        "email": "growth@example.com",
        "full_name": "Growth User",
        "role": "owner",
        "account_id": account_id,
        "account_name": "B2B Growth",
        "plan": plan,
        "plan_status": plan_status,
        "asin_limit": 0,
        "trial_ends_at": None,
        "product": product,
        "vendor_limit": 1,
    }


def _base_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--base-url",
        "https://atlas.example.com",
        "--email",
        "growth@example.com",
        "--password",
        "password-secret",
        "--env-file",
        str(tmp_path / ".env"),
        *extra,
    ]


def test_run_writes_redacted_env_after_login_and_auth_me(monkeypatch, tmp_path, capsys) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prepare_env,
        "_open_http_request",
        _fake_open(
            [
                (200, _login_payload()),
                (200, _me_payload()),
            ],
            calls,
        ),
    )

    exit_code = prepare_env.main([*_base_args(tmp_path), "--json"])

    captured = capsys.readouterr()
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["ok"] is True
    assert summary["account_id"] == ACCOUNT_ID
    assert "ATLAS_API_BASE_URL=https://atlas.example.com" in env_text
    assert "ATLAS_B2B_JWT=jwt.secret.value" in env_text
    assert f"ATLAS_ACCOUNT_ID={ACCOUNT_ID}" in env_text
    assert [call["method"] for call in calls] == ["POST", "GET"]
    assert calls[0]["url"] == "https://atlas.example.com/api/v1/auth/login"
    assert calls[0]["body"] == {
        "email": "growth@example.com",
        "password": "password-secret",
    }
    assert calls[1]["url"] == "https://atlas.example.com/api/v1/auth/me"
    assert calls[1]["headers"]["Authorization"] == "Bearer jwt.secret.value"
    serialized = captured.out + captured.err
    assert "jwt.secret.value" not in serialized
    assert "password-secret" not in serialized
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_run_preserves_unrelated_env_lines_and_forces_existing_keys(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# existing\nOTHER=value\nATLAS_API_BASE_URL=https://old.example.com\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prepare_env,
        "_open_http_request",
        _fake_open([(200, _login_payload()), (200, _me_payload())], calls),
    )
    args = prepare_env._build_parser().parse_args(
        _base_args(tmp_path, "--force")
    )

    summary = prepare_env.run(args)

    assert summary["ok"] is True
    assert env_file.read_text(encoding="utf-8") == (
        "# existing\n"
        "OTHER=value\n"
        "ATLAS_API_BASE_URL=https://atlas.example.com\n"
        "\n"
        "ATLAS_B2B_JWT=jwt.secret.value\n"
        f"ATLAS_ACCOUNT_ID={ACCOUNT_ID}\n"
    )


def test_write_env_file_replaces_with_private_temp_file(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ATLAS_B2B_JWT=old\n", encoding="utf-8")
    env_file.chmod(0o644)
    real_replace = prepare_env.os.replace
    observed_temp_modes: list[int] = []

    def _replace(src, dst):
        observed_temp_modes.append(stat.S_IMODE(Path(src).stat().st_mode))
        assert Path(src).read_text(encoding="utf-8") == (
            "ATLAS_B2B_JWT=new-secret\n"
            "\n"
            "ATLAS_API_BASE_URL=https://atlas.example.com\n"
            f"ATLAS_ACCOUNT_ID={ACCOUNT_ID}\n"
        )
        real_replace(src, dst)

    monkeypatch.setattr(prepare_env.os, "replace", _replace)

    prepare_env._write_env_file(
        env_file,
        {
            "ATLAS_API_BASE_URL": "https://atlas.example.com",
            "ATLAS_B2B_JWT": "new-secret",
            "ATLAS_ACCOUNT_ID": ACCOUNT_ID,
        },
        force=True,
    )

    assert observed_temp_modes == [0o600]
    assert "ATLAS_B2B_JWT=new-secret" in env_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_run_refuses_to_overwrite_existing_env_keys_without_force(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ATLAS_ACCOUNT_ID=old-account\n", encoding="utf-8")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prepare_env,
        "_open_http_request",
        _fake_open([(200, _login_payload()), (200, _me_payload())], calls),
    )
    args = prepare_env._build_parser().parse_args(_base_args(tmp_path))

    summary = prepare_env.run(args)

    assert summary["ok"] is False
    assert summary["stage"] == "env_write"
    assert summary["errors"] == [
        "ATLAS_ACCOUNT_ID already exist in env file; rerun with --force to replace"
    ]
    assert env_file.read_text(encoding="utf-8") == "ATLAS_ACCOUNT_ID=old-account\n"


def test_validate_args_rejects_localhost_and_missing_credentials(tmp_path) -> None:
    args = prepare_env._build_parser().parse_args([
        "--base-url",
        "http://127.0.0.1:8000",
        "--email",
        "",
        "--env-file",
        str(tmp_path / ".env"),
        "--timeout",
        "0",
    ])

    assert prepare_env._validate_args(args) == [
        "--base-url must be an absolute HTTPS URL for hosted proof",
        "ATLAS_LOGIN_EMAIL or --email is required",
        "--timeout must be positive",
    ]


def test_main_fails_closed_when_password_is_missing(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("ATLAS_LOGIN_PASSWORD", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    exit_code = prepare_env.main([
        "--base-url",
        "https://atlas.example.com",
        "--email",
        "growth@example.com",
        "--env-file",
        str(tmp_path / ".env"),
        "--json",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["ok"] is False
    assert payload["stage"] == "preflight"
    assert "ATLAS_LOGIN_PASSWORD or --password is required" in payload["errors"]


def test_login_rejects_missing_access_token(monkeypatch) -> None:
    monkeypatch.setattr(
        prepare_env,
        "_open_http_request",
        _fake_open([(200, {"refresh_token": "refresh"})], []),
    )

    token, errors = prepare_env._login(
        "https://atlas.example.com",
        "growth@example.com",
        "password-secret",
        timeout=1,
    )

    assert token == ""
    assert "login response must include access_token" in errors


def test_login_preserves_http_error_status(monkeypatch) -> None:
    def _raise_http_error(_request, *, timeout):
        raise urllib.error.HTTPError(
            "https://atlas.example.com/api/v1/auth/login",
            401,
            "Unauthorized",
            {},
            None,
        )

    monkeypatch.setattr(prepare_env, "_open_http_request", _raise_http_error)

    token, errors = prepare_env._login(
        "https://atlas.example.com",
        "growth@example.com",
        "password-secret",
        timeout=1,
    )

    assert token == ""
    assert "login status must be 200, got 401" in errors


def test_auth_me_rejects_missing_account_id() -> None:
    account, errors = prepare_env._validate_me_payload(_me_payload(account_id=""))

    assert account["account_id"] == ""
    assert "auth/me response must include account_id" in errors


def test_auth_me_rejects_non_b2b_product() -> None:
    _account, errors = prepare_env._validate_me_payload(_me_payload(product="consumer"))

    assert "auth/me product must be b2b_retention or b2b_challenger" in errors


def test_auth_me_rejects_underplan() -> None:
    _account, errors = prepare_env._validate_me_payload(_me_payload(plan="b2b_starter"))

    assert "auth/me plan must be b2b_growth or higher" in errors


def test_auth_me_rejects_bad_plan_status() -> None:
    _account, errors = prepare_env._validate_me_payload(_me_payload(plan_status="past_due"))

    assert "auth/me plan_status must not be past_due or canceled" in errors


def test_auth_me_rejects_non_object_payload() -> None:
    account, errors = prepare_env._validate_me_payload([])

    assert account == {}
    assert errors == ["auth/me response must be an object"]


def test_env_value_rejects_newlines() -> None:
    try:
        prepare_env._merge_env_text("", {"ATLAS_API_BASE_URL": "https://atlas.example.com\nbad"}, force=True)
    except prepare_env.HandoffError as exc:
        assert str(exc) == "ATLAS_API_BASE_URL cannot contain newlines"
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected HandoffError")
