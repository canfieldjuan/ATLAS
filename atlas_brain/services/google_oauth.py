"""
Persistent Google OAuth token store.

Loads tokens from a JSON file (``~/.config/atlas/google_tokens.json`` by
default — deliberately OUTSIDE the repo, see DEFAULT_TOKEN_FILE), falls back to
the legacy in-repo path and then to .env config fields. Automatically persists rotated
refresh tokens so the user never has to re-run the setup script
after Google rotates a token.

Usage:
    store = get_google_token_store()
    creds = store.get_credentials("calendar")
    # creds.client_id, creds.client_secret, creds.refresh_token

    # After detecting rotation in a token refresh response:
    store.persist_refresh_token("calendar", new_token)
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger("atlas.services.google_oauth")

# Repo root, derived from this file's location rather than the process CWD.
# `atlas_brain/services/google_oauth.py` -> parents[2] is the checkout root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The credential lives OUTSIDE the repo. This is the whole fix.
#
# Anything anchored to a checkout — CWD, repo root, `PROJECT_ROOT` — moves when
# the deployed git worktree moves, because the deployed code lives inside that
# worktree. That is what severed Calendar for five days on 2026-08-05: the
# runtime switched worktrees and the new one had no `data/`. Anchoring "better"
# inside the repo cannot fix it; the credential has to stop living in the repo.
#
# `~/.config/atlas/` is already where this deployment keeps its other service
# credentials (`eom-funnel.token`, `eom-receivables.token`,
# `eom-settings-admin.token`), so this is the existing convention rather than a
# new one.
DEFAULT_TOKEN_FILE = "~/.config/atlas/google_tokens.json"

def _shared_repo_root() -> Path:
    """Return the root shared by every git worktree of this checkout.

    For a linked worktree, `<checkout>/.git` is a FILE containing
    `gitdir: <common>/.git/worktrees/<name>`; the shared root is that common
    dir's parent. For an ordinary checkout `.git` is a directory and the
    checkout IS the shared root.

    This matters for legacy discovery: an upgrade typically means deploying a
    NEW worktree, so a legacy path anchored to the new checkout looks in a
    directory that has never held the credential. Anchoring to the shared root
    finds the file the previous deployment actually wrote.
    """
    git_marker = _REPO_ROOT / ".git"
    try:
        if git_marker.is_file():
            text = git_marker.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                # <common>/.git/worktrees/<name> -> <common>
                if gitdir.parent.name == "worktrees":
                    common = gitdir.parent.parent
                    return common.parent if common.name == ".git" else common
    except OSError:
        pass
    return _REPO_ROOT


# Where installs created before this change put the file. Both anchors are
# checked: the current checkout (an in-place upgrade) and the shared repo root
# (an upgrade that deploys a NEW worktree, which is the common case here and
# the one that caused the outage). Using either emits a migration warning.
def legacy_token_candidates() -> tuple[Path, ...]:
    """Legacy locations to search, in order. Exposed so tests can assert the
    real derivation without depending on the module constant, which test
    fixtures deliberately neutralise to avoid touching live credentials."""
    return tuple(
        dict.fromkeys(
            [
                _REPO_ROOT / "data" / "google_tokens.json",
                _shared_repo_root() / "data" / "google_tokens.json",
            ]
        )
    )


LEGACY_TOKEN_FILES = legacy_token_candidates()
# Retained for callers/tests that referenced the single-path name.
LEGACY_TOKEN_FILE = LEGACY_TOKEN_FILES[0]


def resolve_token_file_path(token_file_path: str) -> Path:
    """Resolve the configured token-file path deterministically.

    A RELATIVE path is anchored to the checkout containing this module, never
    to the process working directory, so a `cd` (or a systemd `WorkingDirectory`
    that differs from the checkout) can no longer move the credential.

    NOTE ON SCOPE. This makes resolution deterministic, but a relative path is
    still anchored INSIDE the checkout — and when the service is deployed from a
    git worktree the module lives in that worktree, so a relative path still
    moves with the deployment. Determinism alone therefore would NOT have
    prevented the 2026-08-05 outage. What prevents it is `DEFAULT_TOKEN_FILE`
    living outside the repo entirely; this function's job is only to honour
    absolute and `~` paths verbatim so that default works.
    """
    path = Path(token_file_path).expanduser()
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def token_path_was_explicitly_configured() -> bool:
    """True when the operator actually supplied the token path.

    Provenance, not value equality. An operator who explicitly sets
    ATLAS_TOOLS_GOOGLE_TOKEN_FILE to the same absolute path the default happens
    to expand to has still named a specific credential, and a value comparison
    would misread that as "no override" and permit borrowing an unrelated
    legacy file (Codex #2355 R11, round 2). pydantic records which fields were
    supplied in `model_fields_set`, which is the actual provenance signal.
    """
    try:
        return "google_token_file" in settings.tools.model_fields_set
    except Exception:  # pragma: no cover - defensive: never block auth on this
        return False


def describe_credential_remedy(creds: Optional["GoogleCredentials"], path: Path) -> str:
    """Return the operator remedy for a credential Google rejected.

    Centralised because EVERY Calendar refresh path needs it: the tool client
    and the MCP provider both refresh, and a remedy fixed in only one of them
    leaves the original misdirection reachable (Codex #2355 R13).
    """
    source = getattr(creds, "refresh_token_source", "file")
    if source == "env":
        return (
            "The token came from the .env fallback "
            "(ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN), NOT from "
            f"{path}. Re-running the setup script will not help: remove or "
            "update that .env value, or restore the token file."
        )
    return (
        f"The token came from {path}. "
        "Re-run: python scripts/setup_google_oauth.py"
    )


def locate_token_file(token_file_path: str, *, explicit: bool | None = None) -> Path:
    """Return the token file to use, preferring the deployment-stable location.

    Order:
      1. the configured/default path (``~/.config/atlas/google_tokens.json``
         unless the operator overrode it) — outside every git worktree, so no
         deploy can move it;
      2. the legacy in-repo ``<checkout>/data/google_tokens.json``, so an
         install created before this change keeps working on upgrade. Using it
         warns, because that is the location a worktree switch severs.

    When neither exists, the primary path is returned so a first write lands in
    the stable location rather than re-creating the legacy one.
    """
    primary = resolve_token_file_path(token_file_path)
    if primary.exists():
        return primary

    # Legacy discovery is an UPGRADE aid for the UNCONFIGURED default only. An
    # explicit operator override names a specific credential -- possibly a
    # different Google account -- and a temporarily absent override must never
    # silently borrow an unrelated legacy file, which could point Calendar and
    # Gmail at the wrong account. Decided on provenance, so an override that
    # merely equals the default value is still treated as an override.
    if explicit is None:
        explicit = token_path_was_explicitly_configured()
    if explicit:
        return primary

    for legacy in LEGACY_TOKEN_FILES:
        if legacy == primary or not legacy.exists():
            continue
        logger.warning(
            "Google token file found at the LEGACY in-repo path %s, not at the "
            "stable default %s. That path lives inside a git checkout, so a "
            "deploy that switches worktrees can sever it (this caused a "
            "five-day Calendar outage on 2026-08-05). Moving it is an operator "
            "action -- see ATLAS #2359 for the supported procedure.",
            legacy,
            DEFAULT_TOKEN_FILE,
        )
        return legacy

    return primary


@dataclass
class GoogleCredentials:
    """OAuth credentials for a Google service."""

    client_id: str
    client_secret: str
    refresh_token: str
    # Which source supplied `refresh_token`: "file" or "env". Carried so the
    # caller can say WHICH credential Google rejected instead of blaming the
    # token file the operator would then pointlessly regenerate.
    refresh_token_source: str = "file"


class GoogleTokenStore:
    """
    Persistent store for Google OAuth tokens.

    Priority: token file > .env config.
    On token rotation, auto-persists to file.
    """

    def __init__(self, token_file_path: str) -> None:
        self._path = locate_token_file(token_file_path)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._loaded = False
        self._file_present = False

    @property
    def token_file_path(self) -> Path:
        """The resolved absolute path this store reads and writes."""
        return self._path

    def _load(self) -> None:
        """Load tokens from file if it exists."""
        if self._loaded:
            return
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                self._file_present = True
                logger.info("Loaded Google tokens from %s", self._path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read token file %s: %s", self._path, e)
                self._data = {}
        else:
            # Previously silent. An absent token file is the single most likely
            # cause of a Google auth failure in this deployment, and saying so
            # here is what distinguishes "the file is missing" from "Google
            # rejected the credential" -- two problems with opposite fixes.
            logger.warning(
                "Google token file not found at %s; falling back to .env "
                "config fields. If Google auth fails, the .env fallback is "
                "what is being used, not this file. The stable default is %s, "
                "outside any git worktree. Legacy locations searched: %s.",
                self._path,
                DEFAULT_TOKEN_FILE,
                ", ".join(str(p) for p in LEGACY_TOKEN_FILES) or "(none)",
            )
        self._loaded = True


    def _save(self) -> None:
        """Write current token data to file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._data, f, indent=2)
            tmp_path.replace(self._path)
            logger.info("Saved Google tokens to %s", self._path)
        except OSError as e:
            logger.error("Failed to write token file %s: %s", self._path, e)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def get_credentials(self, service: str) -> Optional[GoogleCredentials]:
        """
        Get OAuth credentials for a service.

        Args:
            service: "calendar" or "gmail"

        Returns:
            GoogleCredentials or None if not configured.
        """
        with self._lock:
            self._load()
            cfg = settings.tools

            # Try token file first
            file_token = (
                self._data.get("services", {})
                .get(service, {})
                .get("refresh_token")
            )
            file_client_id = self._data.get("client_id")
            file_client_secret = self._data.get("client_secret")

            # Resolve with .env fallback per field
            if service == "calendar":
                client_id = file_client_id or cfg.calendar_client_id
                client_secret = file_client_secret or cfg.calendar_client_secret
                refresh_token = file_token or cfg.calendar_refresh_token
            elif service == "gmail":
                client_id = file_client_id or cfg.gmail_client_id
                client_secret = file_client_secret or cfg.gmail_client_secret
                refresh_token = file_token or cfg.gmail_refresh_token
            else:
                logger.warning("Unknown Google service: %s", service)
                return None

            if not all([client_id, client_secret, refresh_token]):
                return None

            source = "file" if file_token else "env"
            if source == "env":
                # A cross-source substitution is exactly what turned a missing
                # symlink into five days of "refresh token is INVALID": the
                # file token was fine, but an older .env token silently took
                # its place. Name the substitution when it happens.
                logger.warning(
                    "Google %s refresh token came from the .env fallback, not "
                    "the token file %s. These can be DIFFERENT credentials; a "
                    "stale fallback will be rejected by Google even when the "
                    "token file is valid.",
                    service,
                    self._path,
                )

            return GoogleCredentials(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
                refresh_token_source=source,
            )

    def persist_refresh_token(self, service: str, new_token: str) -> None:
        """
        Persist a rotated refresh token to the token file.

        Called when Google returns a new refresh_token during
        an access token refresh.
        """
        with self._lock:
            self._load()

            if "services" not in self._data:
                self._data["services"] = {}
            if service not in self._data["services"]:
                self._data["services"][service] = {}

            old_token = self._data["services"][service].get("refresh_token")
            self._data["services"][service]["refresh_token"] = new_token
            self._data["updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

            self._save()
            logger.info(
                "Persisted rotated %s refresh token (changed=%s)",
                service,
                old_token != new_token,
            )

    def get_status(self) -> dict:
        """
        Get token configuration status for health checks.

        Returns dict with per-service status.
        """
        with self._lock:
            self._load()
            result = {"token_file": str(self._path), "file_exists": self._path.exists()}

            for svc in ("calendar", "gmail"):
                creds = None
                # Release lock briefly to call get_credentials
                # (it re-acquires). Use internal data instead.
                cfg = settings.tools
                file_token = (
                    self._data.get("services", {})
                    .get(svc, {})
                    .get("refresh_token")
                )

                if svc == "calendar":
                    env_token = cfg.calendar_refresh_token
                else:
                    env_token = cfg.gmail_refresh_token

                token = file_token or env_token
                source = "file" if file_token else ("env" if env_token else None)

                result[svc] = {
                    "configured": token is not None,
                    "source": source,
                }

            updated = self._data.get("updated_at")
            if updated:
                result["last_updated"] = updated

            return result


# Module-level singleton
_store: Optional[GoogleTokenStore] = None
_store_lock = threading.Lock()


def get_google_token_store() -> GoogleTokenStore:
    """Get or create the global GoogleTokenStore instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = GoogleTokenStore(settings.tools.google_token_file)
    return _store
