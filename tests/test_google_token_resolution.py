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

The fix is that `DEFAULT_TOKEN_FILE` lives OUTSIDE the repo. Anchoring a
relative path "better" cannot work, because the deployed module lives inside the
deployed worktree — `test_relative_path_still_moves_with_the_deployed_checkout`
asserts that, so the reasoning stays in the suite.

Deliberate non-goal: credentials stranded in a DIFFERENT (old) worktree are not
auto-discovered. Scanning sibling checkouts could authenticate as an arbitrary
stale Google account, which is the same hazard as borrowing a legacy credential
for an explicit override. The missing-file warning enumerates every path
searched and gives an explicit `cp -L` migration command instead.
"""

import json
import logging
import os
from pathlib import Path

import pytest

from atlas_brain.services.google_oauth import (
    DEFAULT_TOKEN_FILE,
    GoogleTokenStore,
    locate_token_file,
    resolve_token_file_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):  # pragma: no cover - marker registration
    config.addinivalue_line(
        "markers", "real_provenance: exercise the real provenance function"
    )


def _bound_settings():
    """Return the exact `settings` object `google_oauth` reads at call time."""
    from atlas_brain.services import google_oauth

    return google_oauth.settings




@pytest.fixture(autouse=True)
def _isolate_credential_discovery(monkeypatch, tmp_path_factory, request):
    """Never let a test reach a REAL credential on this machine.

    Legacy discovery deliberately searches the shared repo root, so without
    this fixture a test passing an absent tmp path falls through to the
    operator's live `data/google_tokens.json` — which both makes assertions
    depend on machine state and prints real refresh tokens into failure output.
    Tests that want legacy behaviour opt back in via `_default_with_legacy`.
    """
    from atlas_brain.services import google_oauth

    nowhere = tmp_path_factory.mktemp("no-legacy") / "google_tokens.json"
    monkeypatch.setattr(google_oauth, "LEGACY_TOKEN_FILES", (nowhere,))
    # A test that exercises the provenance function itself must see the real
    # one, not this stub.
    if request.node.get_closest_marker("real_provenance") is None:
        monkeypatch.setattr(
            google_oauth, "token_path_was_explicitly_configured", lambda: False
        )
    monkeypatch.setattr(
        google_oauth.settings.tools, "calendar_refresh_token", None, raising=False
    )
    monkeypatch.setattr(
        google_oauth.settings.tools, "gmail_refresh_token", None, raising=False
    )


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

    assert store.token_file_path == resolve_token_file_path(
        "data/google_tokens.json"
    )


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
    """WHY the default had to leave the repo, stated as a test.

    A relative path is anchored to the checkout containing the module, and the
    deployed module lives inside the runtime worktree — so a relative path
    still moves with the deployment. This is the reason `DEFAULT_TOKEN_FILE`
    points outside the repo; see
    `test_default_token_file_is_outside_every_worktree`.
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


def test_missing_file_warning_names_the_stable_default(tmp_path, caplog):
    """A missing file must be actionable: say where it SHOULD live."""
    store = GoogleTokenStore(str(tmp_path / "gone" / "google_tokens.json"))

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        store.get_credentials("calendar")

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert DEFAULT_TOKEN_FILE in joined
    assert "outside any git worktree" in joined
    # The operator must be able to see WHERE it looked, and be told that a
    # credential stranded in a different worktree is deliberately not borrowed.
    assert "Legacy locations searched" in joined


def test_repo_root_derivation_points_at_a_real_checkout():
    """Guard the `parents[2]` hop: a file move would silently retarget it."""
    from atlas_brain.services import google_oauth

    root = google_oauth._REPO_ROOT
    assert (root / "atlas_brain").is_dir()
    assert (root / "atlas_brain" / "services" / "google_oauth.py").is_file()
    assert os.path.samefile(root, REPO_ROOT)


# --- what actually prevents recurrence -----------------------------------


def test_default_token_file_is_outside_every_worktree():
    """The fix. The default must not live under any checkout.

    Every in-repo anchor — CWD, repo root, the setup script's PROJECT_ROOT —
    moves when the deployed worktree moves, because the deployed code lives
    inside that worktree. Only a path outside the repo survives a deploy.
    """
    resolved = resolve_token_file_path(DEFAULT_TOKEN_FILE)

    assert resolved.is_absolute()
    assert REPO_ROOT not in resolved.parents, resolved
    assert "worktree" not in str(resolved)
    assert resolved == Path.home() / ".config" / "atlas" / "google_tokens.json"


def test_default_is_identical_from_any_checkout(tmp_path, monkeypatch):
    """The 2026-08-05 scenario, now actually prevented.

    Two different deployed checkouts, same config: the credential must resolve
    to the SAME file. This is the assertion the earlier draft of this slice
    could not honestly make.
    """
    for checkout in ("worktrees/old-runtime", "worktrees/new-runtime"):
        d = tmp_path / checkout
        d.mkdir(parents=True)
        monkeypatch.chdir(d)
        assert resolve_token_file_path(DEFAULT_TOKEN_FILE) == (
            Path.home() / ".config" / "atlas" / "google_tokens.json"
        )


def _default_with_legacy(monkeypatch, tmp_path, *, token="legacy-token", explicit=False):
    """Point the DEFAULT at an absent path and stage a legacy file.

    `explicit` drives PROVENANCE (was the setting actually supplied?), which is
    what gates legacy discovery — not the path's value.
    """
    from atlas_brain.services import google_oauth

    primary = tmp_path / "config" / "google_tokens.json"
    legacy = tmp_path / "checkout" / "data" / "google_tokens.json"
    _write_token_file(legacy, calendar=token)
    monkeypatch.setattr(google_oauth, "DEFAULT_TOKEN_FILE", str(primary))
    monkeypatch.setattr(google_oauth, "LEGACY_TOKEN_FILES", (legacy,))
    monkeypatch.setattr(
        google_oauth, "token_path_was_explicitly_configured", lambda: explicit
    )
    return primary, legacy


def test_legacy_in_repo_file_is_still_found_and_warns(tmp_path, caplog, monkeypatch):
    """Upgrades must not break: an existing in-repo file keeps working."""
    from atlas_brain.services import google_oauth

    primary, legacy = _default_with_legacy(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        chosen = google_oauth.locate_token_file(str(primary))

    assert chosen == legacy
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "LEGACY" in joined
    # The warning names the stable default (the PATCHED one this test installed,
    # not the module constant) and points at the tracked procedure rather than
    # advertising an inline shell command (see ATLAS #2359).
    assert str(primary) in joined
    assert "#2359" in joined


def test_primary_wins_over_legacy_when_both_exist(tmp_path, monkeypatch):
    """Once migrated, the stable copy is authoritative."""
    from atlas_brain.services import google_oauth

    legacy = tmp_path / "checkout" / "data" / "google_tokens.json"
    primary = tmp_path / "config" / "google_tokens.json"
    _write_token_file(legacy, calendar="legacy-token")
    _write_token_file(primary, calendar="stable-token")
    # MUST patch the plural set the resolver actually reads. Patching the
    # singular alias left `legacy` out of the candidate list entirely, so this
    # test passed even when the resolver preferred legacy over an existing
    # primary -- i.e. it proved nothing (Codex #2355 R2, round 7).
    monkeypatch.setattr(google_oauth, "LEGACY_TOKEN_FILES", (legacy,))

    assert locate_token_file(str(primary)) == primary
    assert GoogleTokenStore(str(primary)).get_credentials("calendar").refresh_token == (
        "stable-token"
    )


def test_first_write_lands_in_the_stable_location(tmp_path, monkeypatch):
    """With nothing on disk anywhere, a fresh install must not recreate the
    legacy in-repo file."""
    from atlas_brain.services import google_oauth

    monkeypatch.setattr(
        google_oauth,
        "LEGACY_TOKEN_FILES",
        (tmp_path / "checkout" / "data" / "nope.json",),
    )
    monkeypatch.setattr(
        google_oauth, "token_path_was_explicitly_configured", lambda: False
    )
    primary = tmp_path / "config" / "google_tokens.json"

    assert google_oauth.locate_token_file(str(primary)) == primary


def test_store_itself_uses_the_legacy_fallback(tmp_path, monkeypatch):
    """Asserted THROUGH the store, not just the helper.

    A test that only calls `locate_token_file` directly cannot notice the store
    quietly going back to `resolve_token_file_path` and losing upgrade support.
    """
    primary, legacy = _default_with_legacy(monkeypatch, tmp_path)

    store = GoogleTokenStore(str(primary))

    assert store.token_file_path == legacy
    creds = store.get_credentials("calendar")
    assert creds is not None and creds.refresh_token == "legacy-token"


def test_explicit_override_never_borrows_a_legacy_credential(tmp_path, monkeypatch):
    """An absent EXPLICIT path must not silently load an unrelated account.

    Legacy discovery is an upgrade aid for the default path. If an operator
    points ATLAS_TOOLS_GOOGLE_TOKEN_FILE at account B and that file is briefly
    missing, borrowing account A's legacy credential would send Calendar and
    Gmail at the wrong Google account (Codex #2355 R11).
    """
    from atlas_brain.services import google_oauth

    legacy = tmp_path / "checkout" / "data" / "google_tokens.json"
    _write_token_file(legacy, calendar="account-a-token")
    monkeypatch.setattr(google_oauth, "LEGACY_TOKEN_FILES", (legacy,))
    monkeypatch.setattr(
        google_oauth, "token_path_was_explicitly_configured", lambda: True
    )

    explicit_absent = tmp_path / "account-b" / "google_tokens.json"
    chosen = google_oauth.locate_token_file(str(explicit_absent))

    assert chosen == explicit_absent
    assert chosen != legacy
    assert GoogleTokenStore(str(explicit_absent)).token_file_path == explicit_absent


def test_explicit_override_equal_to_the_default_is_still_an_override(
    tmp_path, monkeypatch
):
    """Provenance, not value equality (Codex #2355 R11, round 2).

    Setting ATLAS_TOOLS_GOOGLE_TOKEN_FILE to the same absolute path the default
    expands to is still an explicit choice of credential; classifying it as
    "no override" would permit borrowing an unrelated legacy account.
    """
    from atlas_brain.services import google_oauth

    legacy = tmp_path / "checkout" / "data" / "google_tokens.json"
    _write_token_file(legacy, calendar="account-a-token")
    same_as_default = tmp_path / "config" / "google_tokens.json"
    monkeypatch.setattr(google_oauth, "DEFAULT_TOKEN_FILE", str(same_as_default))
    monkeypatch.setattr(google_oauth, "LEGACY_TOKEN_FILES", (legacy,))
    monkeypatch.setattr(
        google_oauth, "token_path_was_explicitly_configured", lambda: True
    )

    assert google_oauth.locate_token_file(str(same_as_default)) == same_as_default


@pytest.mark.real_provenance
def test_provenance_comes_from_pydantic_fields_set(monkeypatch):
    """The provenance signal is the settings object's own record.

    `model_fields_set` is a read-only pydantic property, so the settings object
    is stubbed rather than mutated.
    """
    from atlas_brain.services import google_oauth

    class _Tools:
        def __init__(self, fields):
            self.model_fields_set = fields

    class _Settings:
        def __init__(self, fields):
            self.tools = _Tools(fields)

    monkeypatch.setattr(google_oauth, "settings", _Settings({"google_token_file"}))
    assert google_oauth.token_path_was_explicitly_configured() is True

    monkeypatch.setattr(google_oauth, "settings", _Settings(set()))
    assert google_oauth.token_path_was_explicitly_configured() is False


def test_legacy_anchor_includes_the_shared_repo_root(tmp_path):
    """An upgrade that deploys a NEW worktree must still find the old file.

    A legacy path anchored only to the current checkout looks inside the new
    worktree, which has never held the credential — exactly the upgrade that
    caused the outage (Codex #2355 R12).
    """
    from atlas_brain.services import google_oauth

    # Use the derivation, not the module constant: the autouse isolation
    # fixture neutralises the constant so tests never reach a live credential.
    roots = {p.parent.parent for p in google_oauth.legacy_token_candidates()}
    assert google_oauth._shared_repo_root() in roots


def test_shared_repo_root_resolves_through_a_worktree_gitfile(tmp_path):
    """`.git` as a FILE (a linked worktree) must resolve to the shared root."""
    from atlas_brain.services import google_oauth

    shared = tmp_path / "MainRepo"
    worktree = shared / "worktrees" / "runtime"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(
        f"gitdir: {shared}/.git/worktrees/runtime\n"
    )

    monkeypatch_root = worktree
    original = google_oauth._REPO_ROOT
    try:
        google_oauth._REPO_ROOT = monkeypatch_root
        assert google_oauth._shared_repo_root() == shared
    finally:
        google_oauth._REPO_ROOT = original


def test_both_calendar_callers_use_the_shared_remedy():
    """A remedy fixed in one refresh path leaves the other misdirecting."""
    tool = (REPO_ROOT / "atlas_brain" / "tools" / "calendar.py").read_text()
    provider = (
        REPO_ROOT / "atlas_brain" / "services" / "calendar_provider.py"
    ).read_text()

    for source in (tool, provider):
        assert "describe_credential_remedy" in source
        assert "Re-run: python scripts/setup_google_oauth.py" not in source


# --- rotation must survive a migrated (symlinked) path --------------------


def test_every_recovery_message_requires_a_restart(tmp_path, caplog, monkeypatch):
    """Fixing the credential does nothing until the process restarts.

    `_load()` sets `_loaded=True` permanently and the settings object captures
    `.env` at import, so an operator who follows the guidance and restores the
    file (or edits `.env`) sees no change and reasonably concludes the fix
    failed — prolonging the outage (Codex #2355 R6).
    """
    from atlas_brain.services import google_oauth

    # missing-file path
    store = GoogleTokenStore(str(tmp_path / "gone" / "google_tokens.json"))
    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        store.get_credentials("calendar")
    assert "RESTART" in " ".join(r.getMessage() for r in caplog.records)

    # .env-fallback path
    caplog.clear()
    monkeypatch.setattr(google_oauth.settings.tools, "calendar_client_id", "cid")
    monkeypatch.setattr(google_oauth.settings.tools, "calendar_client_secret", "sec")
    monkeypatch.setattr(google_oauth.settings.tools, "calendar_refresh_token", "envtok")
    store2 = GoogleTokenStore(str(tmp_path / "also-gone" / "google_tokens.json"))
    with caplog.at_level(logging.WARNING, logger="atlas.services.google_oauth"):
        creds = store2.get_credentials("calendar")
    assert creds is not None and creds.refresh_token_source == "env"
    assert "RESTART" in " ".join(r.getMessage() for r in caplog.records)

    # both rejection remedies
    from atlas_brain.services.google_oauth import (
        GoogleCredentials,
        describe_credential_remedy,
    )

    for source in ("file", "env"):
        remedy = describe_credential_remedy(
            GoogleCredentials("i", "s", "t", refresh_token_source=source),
            tmp_path / "tokens.json",
        )
        assert "RESTART" in remedy, source
