"""
Persistent Google OAuth token store.

Loads tokens from a JSON file (data/google_tokens.json by default),
falls back to .env config fields. Automatically persists rotated
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


def resolve_token_file_path(token_file_path: str) -> Path:
    """Resolve the configured token-file path deterministically.

    A RELATIVE path is anchored to the checkout containing this module, never
    to the process working directory, so a `cd` (or a systemd `WorkingDirectory`
    that differs from the checkout) can no longer move the credential.

    SCOPE OF THIS GUARANTEE — read before relying on it. Anchoring is relative
    to *this checkout*. When the service is deployed from a git worktree, the
    module lives inside that worktree, so the anchor moves with the deployment:
    a relative path still resolves under whichever worktree is deployed. That
    is exactly how Calendar broke on 2026-08-05, when the runtime moved from
    `worktrees/eom-receivables-runtime` (which had `data/google_tokens.json`
    symlinked) to `worktrees/atlas-runtime-main` (which had no `data/` at all).

    So this function makes resolution DETERMINISTIC, not deployment-stable. For
    a multi-worktree deployment the durable fix is an ABSOLUTE
    `ATLAS_TOOLS_GOOGLE_TOKEN_FILE` pointing at shared state outside any
    worktree; absolute paths are honoured verbatim. `_load` warns with the
    resolved path and that remedy whenever the file is absent, so the residual
    hazard is loud and correctly diagnosed instead of silent.
    """
    path = Path(token_file_path).expanduser()
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


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
        self._path = resolve_token_file_path(token_file_path)
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
                "what is being used, not this file. If this path sits inside a "
                "git worktree, a deploy that switches worktrees will keep "
                "moving it -- set an ABSOLUTE ATLAS_TOOLS_GOOGLE_TOKEN_FILE "
                "pointing at shared state outside any worktree.",
                self._path,
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
