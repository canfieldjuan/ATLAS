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


@dataclass
class GoogleCredentials:
    """OAuth credentials for a Google service."""

    client_id: str
    client_secret: str
    refresh_token: str


class GoogleTokenStore:
    """
    Persistent store for Google OAuth tokens.

    Priority: token file > .env config.
    On token rotation, auto-persists to file.
    """

    def __init__(self, token_file_path: str) -> None:
        self._path = Path(token_file_path)
        self._data: dict = {}
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        """Load tokens from file if it exists."""
        if self._loaded:
            return
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                logger.info("Loaded Google tokens from %s", self._path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read token file %s: %s", self._path, e)
                self._data = {}
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

            return GoogleCredentials(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
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
