"""Tests for the Reddit Listening token-mint helper's deterministic logic.

The live parts (praw, the localhost socket, the browser approval) are the
external boundary and are exercised by the operator, not here; these cover
the pure logic: dotenv parsing, credential resolution + B2B fallback, the
redirect-param parser, and the fail-closed CLI exits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "mint_reddit_listening_token.py"
_spec = importlib.util.spec_from_file_location("mint_reddit_listening_token", _MODULE_PATH)
mint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mint)


# -- load_env ---------------------------------------------------------------


def test_load_env_parses_and_strips_quotes(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nATLAS_REDDIT_CLIENT_ID="abc123"\n'
        "ATLAS_REDDIT_CLIENT_SECRET='sek=ret'\n\nBARE=1\n",
        encoding="utf-8",
    )
    env = mint.load_env(str(env_file))
    assert env["ATLAS_REDDIT_CLIENT_ID"] == "abc123"
    assert env["ATLAS_REDDIT_CLIENT_SECRET"] == "sek=ret"  # only outer quotes stripped, '=' kept
    assert env["BARE"] == "1"
    assert "# comment" not in env


def test_load_env_missing_file_is_empty(tmp_path) -> None:
    assert mint.load_env(str(tmp_path / "nope.env")) == {}


# -- resolve_credentials ----------------------------------------------------


def test_resolve_prefers_listening_keys() -> None:
    env = {
        "ATLAS_REDDIT_CLIENT_ID": "listen-id",
        "ATLAS_REDDIT_CLIENT_SECRET": "listen-secret",
        "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID": "b2b-id",
        "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET": "b2b-secret",
    }
    assert mint.resolve_credentials(env) == ("listen-id", "listen-secret")


def test_resolve_falls_back_to_b2b_app() -> None:
    env = {
        "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID": "b2b-id",
        "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET": "b2b-secret",
    }
    assert mint.resolve_credentials(env) == ("b2b-id", "b2b-secret")


def test_resolve_explicit_args_win() -> None:
    env = {"ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID": "b2b-id",
           "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET": "b2b-secret"}
    assert mint.resolve_credentials(env, client_id="cli-id", client_secret="cli-sec") == (
        "cli-id", "cli-sec"
    )


def test_resolve_fails_closed_when_missing() -> None:
    with pytest.raises(mint.MintConfigError, match="no client_id/secret"):
        mint.resolve_credentials({})
    # a blank value is not a set value
    with pytest.raises(mint.MintConfigError):
        mint.resolve_credentials({"ATLAS_REDDIT_CLIENT_ID": "  ", "ATLAS_REDDIT_CLIENT_SECRET": "x"})


# -- parse_redirect_params --------------------------------------------------


def test_parse_redirect_params_extracts_code_and_state() -> None:
    line = "GET /?state=42&code=the-code HTTP/1.1\r\nHost: localhost:8080"
    params = mint.parse_redirect_params(line)
    assert params == {"state": "42", "code": "the-code"}


def test_parse_redirect_params_captures_error() -> None:
    line = "GET /?state=42&error=access_denied HTTP/1.1"
    assert mint.parse_redirect_params(line)["error"] == "access_denied"


# -- CLI fail-closed exits --------------------------------------------------


def test_cli_missing_creds_exits_two(tmp_path, capsys) -> None:
    empty = tmp_path / ".env"
    empty.write_text("", encoding="utf-8")
    code = mint.main(["--username", "someuser", "--env-file", str(empty)])
    assert code == 2
    assert "client_id/secret" in capsys.readouterr().err


def test_cli_missing_username_exits_two(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_REDDIT_USERNAME", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID=id\nATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET=sec\n",
        encoding="utf-8",
    )
    code = mint.main(["--env-file", str(env_file)])
    assert code == 2
    assert "username" in capsys.readouterr().err
