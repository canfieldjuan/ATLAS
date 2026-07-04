"""Tests for the Reddit Listening token-mint helper's deterministic logic.

The live parts (praw, the localhost socket, the browser approval) are the
external boundary and are exercised by the operator, not here; these cover
the pure logic: dotenv parsing, atomic credential-pair resolution + B2B
fallback, the .env/shell merge, the redirect-param parser, username
validation, and the fail-closed CLI exits.

Named test_atlas_reddit_* so the atlas_reddit CI workflow's glob runs it.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "mint_reddit_listening_token.py"
_spec = importlib.util.spec_from_file_location("mint_reddit_listening_token", _MODULE_PATH)
mint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mint)

LISTEN = ("ATLAS_REDDIT_CLIENT_ID", "ATLAS_REDDIT_CLIENT_SECRET")
B2B = ("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID", "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET")


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Clear ambient ATLAS_* so main()'s os.environ merge is deterministic."""
    for key in list(os.environ):
        if key.startswith("ATLAS_"):
            monkeypatch.delenv(key, raising=False)


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
    assert env["ATLAS_REDDIT_CLIENT_SECRET"] == "sek=ret"  # only outer quotes stripped
    assert env["BARE"] == "1"


def test_load_env_missing_or_unreadable_is_empty(tmp_path) -> None:
    assert mint.load_env(str(tmp_path / "nope.env")) == {}
    assert mint.load_env(str(tmp_path)) == {}  # a directory -> OSError -> {}


# -- build_cred_source (.env + shell merge) ---------------------------------


def test_build_cred_source_shell_overrides_env_and_filters() -> None:
    env = {LISTEN[0]: "from-dotenv", "OTHER": "keep-out"}
    environ = {LISTEN[0]: "from-shell", "PATH": "/bin", "ATLAS_X": "y"}
    src = mint.build_cred_source(env, environ)
    assert src[LISTEN[0]] == "from-shell"  # shell wins
    assert src["ATLAS_X"] == "y"           # ATLAS_* shell vars merged
    assert "PATH" not in src               # non-ATLAS shell vars filtered out
    assert src["OTHER"] == "keep-out"      # dotenv non-ATLAS kept


# -- resolve_credentials: atomic pairs --------------------------------------


def test_resolve_prefers_listening_pair() -> None:
    env = {LISTEN[0]: "lid", LISTEN[1]: "lsec", B2B[0]: "bid", B2B[1]: "bsec"}
    assert mint.resolve_credentials(env) == ("lid", "lsec")


def test_resolve_falls_back_to_complete_b2b_pair() -> None:
    assert mint.resolve_credentials({B2B[0]: "bid", B2B[1]: "bsec"}) == ("bid", "bsec")


def test_resolve_never_mixes_across_namespaces() -> None:
    """listening id + B2B secret only = no COMPLETE pair -> fail closed,
    NOT a mismatched cross-app pair."""
    env = {LISTEN[0]: "lid", B2B[1]: "bsec"}
    with pytest.raises(mint.MintConfigError, match="no complete"):
        mint.resolve_credentials(env)


def test_resolve_partial_listening_falls_back_to_complete_b2b() -> None:
    # listening secret missing, but B2B is a complete pair -> use B2B whole
    env = {LISTEN[0]: "lid", B2B[0]: "bid", B2B[1]: "bsec"}
    assert mint.resolve_credentials(env) == ("bid", "bsec")


def test_resolve_explicit_args_require_both() -> None:
    assert mint.resolve_credentials({}, client_id="x", client_secret="y") == ("x", "y")
    with pytest.raises(mint.MintConfigError, match="BOTH"):
        mint.resolve_credentials({B2B[0]: "bid", B2B[1]: "bsec"}, client_id="x")


def test_resolve_fails_closed_when_missing() -> None:
    with pytest.raises(mint.MintConfigError, match="no complete"):
        mint.resolve_credentials({})
    with pytest.raises(mint.MintConfigError):  # blank is not "set"
        mint.resolve_credentials({LISTEN[0]: "  ", LISTEN[1]: "x"})


# -- parse_redirect_params --------------------------------------------------


def test_parse_redirect_params_extracts_code_and_state() -> None:
    line = "GET /?state=42&code=the-code HTTP/1.1\r\nHost: localhost:8080"
    assert mint.parse_redirect_params(line) == {"state": "42", "code": "the-code"}


def test_parse_redirect_params_captures_error() -> None:
    line = "GET /?state=42&error=access_denied HTTP/1.1"
    assert mint.parse_redirect_params(line)["error"] == "access_denied"


def test_parse_redirect_params_malformed_raises_valueerror() -> None:
    for bad in ("GET / HTTP/1.1", "garbage", ""):
        with pytest.raises(ValueError, match="malformed redirect"):
            mint.parse_redirect_params(bad)


# -- CLI fail-closed exits --------------------------------------------------


def test_cli_missing_creds_exits_two(tmp_path, capsys) -> None:
    empty = tmp_path / ".env"
    empty.write_text("", encoding="utf-8")
    assert mint.main(["--username", "someuser", "--env-file", str(empty)]) == 2
    assert "complete client_id/secret" in capsys.readouterr().err


def test_cli_reads_creds_from_shell_env_not_just_dotenv(tmp_path, capsys, monkeypatch) -> None:
    """Exported ATLAS_* creds (no .env) are honored -> resolution passes and
    the run proceeds to username validation, not a missing-creds exit."""
    monkeypatch.setenv(B2B[0], "bid")
    monkeypatch.setenv(B2B[1], "bsec")
    empty = tmp_path / ".env"
    empty.write_text("", encoding="utf-8")
    # invalid username -> exits at the username gate (2), proving creds resolved
    code = mint.main(["--username", "!!bad!!", "--env-file", str(empty)])
    assert code == 2
    assert "USERNAME" in capsys.readouterr().err.upper()


def test_cli_invalid_username_exits_two(tmp_path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{B2B[0]}=bid\n{B2B[1]}=bsec\n", encoding="utf-8")
    # a space is an invalid Reddit username character
    assert mint.main(["--username", "a b", "--env-file", str(env_file)]) == 2
    assert "USERNAME" in capsys.readouterr().err.upper()
