"""Google OAuth credential resolution and failure reporting.

Regression cover for a real five-day production outage (2026-08-05 to
2026-08-10). The API runs as a systemd unit whose `WorkingDirectory` is
whichever git worktree is deployed. The token path defaulted to the RELATIVE
`data/google_tokens.json`, so when the runtime moved to a new worktree that had
no `data/` directory, the file lookup silently missed, `get_credentials` fell
back to an OLDER refresh token in `.env`, and Google rejected it. The log said
"Calendar refresh token is INVALID ... Re-run setup" — which would have
rewritten the perfectly valid token FILE and fixed nothing, because the stale
`.env` value was what kept winning.

Three defects, three properties asserted here:
  1. a relative token path must not depend on the process CWD;
  2. an absent token file must be announced, not silent;
  3. a credential taken from the `.env` fallback must be reported as such, so
     the remedy names the right file.

What this does NOT claim: anchoring is relative to the checkout containing the
module, so when the service is deployed FROM a worktree a relative path still
moves with that worktree. Only an ABSOLUTE configured path is deployment
stable. `test_relative_path_still_moves_with_the_deployed_checkout` asserts that
limit rather than leaving it implied.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from atlas_brain.services.google_oauth import (
    GoogleTokenStore,
    resolve_token_file_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bound_settings():
    """Return the exact `settings` object `google_oauth` reads at call time."""
    from atlas_brain.services import google_oauth

    return google_oauth.settings



def _write_token_file(path: Path, *, calendar: str = "", gmail: str = "") -> None:
    services = {}
    if calendar:
        services["calendar"] = {"refresh_token": calendar}
    if gmail:
        services["gmail"] = {"refresh_token": gmail}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "client_id": "file-client-id",
                "client_secret": "file-client-secret",
                "services": services,
            }
        )
    )


# --- 1. path resolution must not follow the CWD ---------------------------


def test_relative_path_anchors_to_the_repo_root(tmp_path, monkeypatch):
    """The defect: `Path("data/x.json")` resolved against whatever CWD."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_token_file_path("data/google_tokens.json")

    assert resolved.is_absolute()
    assert resolved == REPO_ROOT / "data" / "google_tokens.json"
    # The failure mode itself: resolution must NOT land under the CWD.
    assert tmp_path not in resolved.parents


def test_relative_path_is_identical_from_two_different_cwds(tmp_path, monkeypatch):
    """Same config + different CWD must yield the same path.

    Note the limit: this covers a differing `WorkingDirectory`, NOT a differing
    CHECKOUT. See `test_relative_path_still_moves_with_the_deployed_checkout`.
    """
    first_cwd = tmp_path / "worktree-a" / "nested"
    second_cwd = tmp_path / "worktree-b"
    first_cwd.mkdir(parents=True)
    second_cwd.mkdir(parents=True)

    monkeypatch.chdir(first_cwd)
    from_first = resolve_token_file_path("data/google_tokens.json")
    monkeypatch.chdir(second_cwd)
    from_second = resolve_token_file_path("data/google_tokens.json")

    assert from_first == from_second


def test_absolute_path_is_honoured_unchanged(tmp_path, monkeypatch):
    """Explicit deployments and tests keep full control."""
    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / "elsewhere" / "tokens.json"

    assert resolve_token_file_path(str(explicit)) == explicit


def test_user_home_shorthand_expands(tmp_path):
    resolved = resolve_token_file_path("~/atlas-tokens.json")

    assert resolved.is_absolute()
    assert "~" not in str(resolved)


def test_store_exposes_the_resolved_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = GoogleTokenStore("data/google_tokens.json")

    assert store.token_file_path == REPO_ROOT / "data" / "google_tokens.json"


# --- 2. an absent file must be announced ----------------------------------


def test_missing_token_file_is_logged_with_its_resolved_path(tmp_path, caplog):
    missing = tmp_path / "nope" / "google_tokens.json"
    store = GoogleTokenStore(str(missing))

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        store.get_credentials("calendar")

    messages = [r.getMessage() for r in caplog.records]
    assert any("token file not found" in m.lower() for m in messages), messages
    # Naming the path is the point: it is how an operator sees WHICH file the
    # deployed process actually looked for.
    assert any(str(missing) in m for m in messages), messages


def test_present_token_file_does_not_warn_about_absence(tmp_path, caplog):
    present = tmp_path / "google_tokens.json"
    _write_token_file(present, calendar="file-token")
    store = GoogleTokenStore(str(present))

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        store.get_credentials("calendar")

    assert not any(
        "token file not found" in r.getMessage().lower() for r in caplog.records
    )


# --- 3. the credential SOURCE must be reported ----------------------------


def test_file_token_is_reported_as_coming_from_the_file(tmp_path):
    present = tmp_path / "google_tokens.json"
    _write_token_file(present, calendar="file-token")
    store = GoogleTokenStore(str(present))

    creds = store.get_credentials("calendar")

    assert creds is not None
    assert creds.refresh_token == "file-token"
    assert creds.refresh_token_source == "file"


def test_env_fallback_is_reported_and_warned(tmp_path, caplog, monkeypatch):
    """The production defect, reproduced.

    File absent -> a DIFFERENT, older `.env` token is substituted. Previously
    silent; the operator only saw "refresh token is INVALID".
    """
    # Patch the `settings` object google_oauth BOUND AT IMPORT
    # (`from ..config import settings`). Patching `atlas_brain.config.settings`
    # instead is unreliable in a full-suite run: if any other module rebinds or
    # reloads config, the test and the code under test hold different objects
    # and the patch silently lands on the wrong one.
    settings = _bound_settings()

    monkeypatch.setattr(settings.tools, "calendar_client_id", "env-client-id")
    monkeypatch.setattr(settings.tools, "calendar_client_secret", "env-secret")
    monkeypatch.setattr(settings.tools, "calendar_refresh_token", "stale-env-token")

    store = GoogleTokenStore(str(tmp_path / "absent" / "google_tokens.json"))

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        creds = store.get_credentials("calendar")

    assert creds is not None
    assert creds.refresh_token == "stale-env-token"
    assert creds.refresh_token_source == "env"
    assert any(
        "fallback" in r.getMessage().lower() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_file_token_wins_over_a_different_env_token(tmp_path, monkeypatch):
    """Precedence is unchanged by this slice -- only the reporting is new."""
    # Patch the `settings` object google_oauth BOUND AT IMPORT
    # (`from ..config import settings`). Patching `atlas_brain.config.settings`
    # instead is unreliable in a full-suite run: if any other module rebinds or
    # reloads config, the test and the code under test hold different objects
    # and the patch silently lands on the wrong one.
    settings = _bound_settings()

    monkeypatch.setattr(settings.tools, "calendar_client_id", "env-client-id")
    monkeypatch.setattr(settings.tools, "calendar_client_secret", "env-secret")
    monkeypatch.setattr(settings.tools, "calendar_refresh_token", "stale-env-token")

    present = tmp_path / "google_tokens.json"
    _write_token_file(present, calendar="good-file-token")
    store = GoogleTokenStore(str(present))

    creds = store.get_credentials("calendar")

    assert creds.refresh_token == "good-file-token"
    assert creds.refresh_token_source == "file"


def test_no_credential_anywhere_returns_none(tmp_path, monkeypatch):
    # Patch the `settings` object google_oauth BOUND AT IMPORT
    # (`from ..config import settings`). Patching `atlas_brain.config.settings`
    # instead is unreliable in a full-suite run: if any other module rebinds or
    # reloads config, the test and the code under test hold different objects
    # and the patch silently lands on the wrong one.
    settings = _bound_settings()

    monkeypatch.setattr(settings.tools, "calendar_client_id", None)
    monkeypatch.setattr(settings.tools, "calendar_client_secret", None)
    monkeypatch.setattr(settings.tools, "calendar_refresh_token", None)

    store = GoogleTokenStore(str(tmp_path / "absent.json"))

    assert store.get_credentials("calendar") is None


@pytest.mark.parametrize("service", ["calendar", "gmail"])
def test_both_services_report_their_source(tmp_path, service):
    """Gmail shares the file and was equally broken -- it just never logged.

    Only Calendar surfaced a CRITICAL during the outage, because Gmail had no
    `.env` fallback and so returned None instead of a rejected credential. Both
    must report their source.
    """
    present = tmp_path / "google_tokens.json"
    _write_token_file(present, calendar="cal-token", gmail="gmail-token")
    store = GoogleTokenStore(str(present))

    creds = store.get_credentials(service)

    assert creds is not None
    assert creds.refresh_token_source == "file"


def test_unknown_service_is_rejected(tmp_path):
    present = tmp_path / "google_tokens.json"
    _write_token_file(present, calendar="cal-token")
    store = GoogleTokenStore(str(present))

    assert store.get_credentials("drive") is None


# --- the outage, end to end -----------------------------------------------


def test_cwd_change_no_longer_moves_the_credential(tmp_path, monkeypatch):
    """What this slice DOES fix: resolution no longer follows the CWD.

    A process that chdirs -- or a systemd `WorkingDirectory` that differs from
    the checkout -- previously moved the resolved token path with it.
    """
    old_runtime = tmp_path / "worktrees" / "old-runtime"
    new_runtime = tmp_path / "worktrees" / "new-runtime"
    old_runtime.mkdir(parents=True)
    new_runtime.mkdir(parents=True)

    monkeypatch.chdir(old_runtime)
    before = GoogleTokenStore("data/google_tokens.json").token_file_path
    monkeypatch.chdir(new_runtime)
    after = GoogleTokenStore("data/google_tokens.json").token_file_path

    assert before == after
    assert not str(before).startswith(str(tmp_path))


def test_relative_path_still_moves_with_the_deployed_checkout(tmp_path):
    """What this slice does NOT fix, asserted so the limit is recorded.

    Anchoring is relative to the checkout containing the module. When the
    service is deployed FROM a git worktree, the module lives inside it, so a
    relative path still resolves under whichever worktree is deployed — the
    exact 2026-08-05 trigger. Only an ABSOLUTE configured path is deployment
    stable. Stating this as a test keeps the residual hazard honest instead of
    letting the plan imply a guarantee the code does not provide.
    """
    worktree_a = tmp_path / "worktree-a"
    worktree_b = tmp_path / "worktree-b"
    for wt in (worktree_a, worktree_b):
        (wt / "atlas_brain" / "services").mkdir(parents=True)

    def anchor_for(checkout: Path) -> Path:
        module = checkout / "atlas_brain" / "services" / "google_oauth.py"
        return module.resolve().parents[2] / "data" / "google_tokens.json"

    assert anchor_for(worktree_a) != anchor_for(worktree_b)

    # ...whereas an absolute configured path is identical from either checkout.
    shared = tmp_path / "shared-state" / "google_tokens.json"
    assert resolve_token_file_path(str(shared)) == shared


def test_missing_file_warning_names_the_absolute_path_remedy(tmp_path, caplog):
    """The residual hazard must be actionable, not merely reported."""
    store = GoogleTokenStore(str(tmp_path / "gone" / "google_tokens.json"))

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        store.get_credentials("calendar")

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "ATLAS_TOOLS_GOOGLE_TOKEN_FILE" in joined
    assert "absolute" in joined.lower()


def test_repo_root_derivation_points_at_a_real_checkout():
    """Guard the `parents[2]` hop: a file move would silently retarget it."""
    from atlas_brain.services import google_oauth

    root = google_oauth._REPO_ROOT
    assert (root / "atlas_brain").is_dir()
    assert (root / "atlas_brain" / "services" / "google_oauth.py").is_file()
    assert os.path.samefile(root, REPO_ROOT)
