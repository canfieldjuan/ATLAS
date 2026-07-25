"""Encrypted, exact-context credential state for scoped Gmail inbox reads."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from typing import AsyncContextManager, AsyncIterator

from ...auth.encryption import decrypt_secret, encrypt_secret
from ..database import get_db_pool

logger = logging.getLogger("atlas.storage.scoped_mailbox_credentials")

GMAIL_PROVIDER = "gmail"


class ScopedMailboxCredentialUnavailable(RuntimeError):
    """No active, decryptable credential exists for the exact context."""


@dataclass(frozen=True)
class ScopedGmailCredentials:
    client_id: str
    client_secret: str
    refresh_token: str
    generation: int


@dataclass
class LockedScopedGmailCredentials:
    """Credential row held under ``FOR UPDATE`` in one transaction."""

    _conn: object
    business_context_id: str
    credentials: ScopedGmailCredentials

    async def persist_refresh_token(self, new_refresh_token: str) -> int:
        """Replace a rotated token on the locked row and advance generation."""
        token = _required_secret(new_refresh_token, "refresh_token")
        ciphertext, kid = _encrypt_bundle(
            client_id=self.credentials.client_id,
            client_secret=self.credentials.client_secret,
            refresh_token=token,
        )
        row = await self._conn.fetchrow(
            """
            UPDATE scoped_mailbox_credentials
            SET encrypted_credentials = $4,
                encryption_kid = $5,
                generation = generation + 1,
                updated_at = NOW()
            WHERE business_context_id = $1
              AND provider = $2
              AND generation = $3
              AND revoked_at IS NULL
            RETURNING generation
            """,
            self.business_context_id,
            GMAIL_PROVIDER,
            self.credentials.generation,
            ciphertext,
            kid,
        )
        if row is None:
            raise ScopedMailboxCredentialUnavailable(
                "scoped_gmail_rotation_lost_generation"
            )
        generation = int(row["generation"])
        self.credentials = ScopedGmailCredentials(
            client_id=self.credentials.client_id,
            client_secret=self.credentials.client_secret,
            refresh_token=token,
            generation=generation,
        )
        return generation


class ScopedMailboxCredentialRepository:
    """Narrow repository; no list API or broad secret projection exists."""

    async def bind_gmail(
        self,
        *,
        business_context_id: str,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> int:
        context = _exact_context(business_context_id)
        ciphertext, kid = _encrypt_bundle(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
        )
        row = await get_db_pool().fetchrow(
            """
            INSERT INTO scoped_mailbox_credentials (
                business_context_id,
                provider,
                encrypted_credentials,
                encryption_kid
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (business_context_id, provider) DO UPDATE
            SET encrypted_credentials = EXCLUDED.encrypted_credentials,
                encryption_kid = EXCLUDED.encryption_kid,
                generation = scoped_mailbox_credentials.generation + 1,
                updated_at = NOW(),
                revoked_at = NULL
            RETURNING generation
            """,
            context,
            GMAIL_PROVIDER,
            ciphertext,
            kid,
        )
        return int(row["generation"])

    async def get_active_gmail(
        self,
        business_context_id: str,
    ) -> ScopedGmailCredentials | None:
        context = _exact_context(business_context_id)
        row = await get_db_pool().fetchrow(
            """
            SELECT encrypted_credentials, encryption_kid, generation
            FROM scoped_mailbox_credentials
            WHERE business_context_id = $1
              AND provider = $2
              AND revoked_at IS NULL
            """,
            context,
            GMAIL_PROVIDER,
        )
        return _decrypt_row(row, context)

    async def revoke_gmail(self, business_context_id: str) -> int | None:
        context = _exact_context(business_context_id)
        row = await get_db_pool().fetchrow(
            """
            UPDATE scoped_mailbox_credentials
            SET revoked_at = NOW(),
                generation = generation + 1,
                updated_at = NOW()
            WHERE business_context_id = $1
              AND provider = $2
              AND revoked_at IS NULL
            RETURNING generation
            """,
            context,
            GMAIL_PROVIDER,
        )
        return int(row["generation"]) if row is not None else None

    @asynccontextmanager
    async def locked_gmail(
        self,
        business_context_id: str,
    ) -> AsyncIterator[LockedScopedGmailCredentials]:
        """Serialize token refresh for one exact context across processes."""
        context = _exact_context(business_context_id)
        async with get_db_pool().transaction() as conn:
            row = await conn.fetchrow(
                """
                SELECT encrypted_credentials, encryption_kid, generation
                FROM scoped_mailbox_credentials
                WHERE business_context_id = $1
                  AND provider = $2
                  AND revoked_at IS NULL
                FOR UPDATE
                """,
                context,
                GMAIL_PROVIDER,
            )
            credentials = _decrypt_row(row, context)
            if credentials is None:
                raise ScopedMailboxCredentialUnavailable(
                    "scoped_gmail_credentials_unavailable"
                )
            yield LockedScopedGmailCredentials(
                _conn=conn,
                business_context_id=context,
                credentials=credentials,
            )


class ScopedGmailCredentialSource:
    """Gmail-client credential port bound to one exact business context."""

    def __init__(
        self,
        business_context_id: str,
        repository: ScopedMailboxCredentialRepository | None = None,
    ) -> None:
        self.business_context_id = _exact_context(business_context_id)
        self.repository = repository or ScopedMailboxCredentialRepository()

    async def is_available(self) -> bool:
        return (
            await self.repository.get_active_gmail(self.business_context_id)
            is not None
        )

    def locked_credentials(
        self,
    ) -> AsyncContextManager[LockedScopedGmailCredentials]:
        return self.repository.locked_gmail(self.business_context_id)


def _exact_context(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError("business_context_id must be a nonblank string of at most 64 characters")
    return value


def _required_secret(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _encrypt_bundle(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> tuple[bytes, str]:
    payload = json.dumps(
        {
            "client_id": _required_secret(client_id, "client_id"),
            "client_secret": _required_secret(client_secret, "client_secret"),
            "refresh_token": _required_secret(refresh_token, "refresh_token"),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return encrypt_secret(payload)


def _decrypt_row(row, business_context_id: str) -> ScopedGmailCredentials | None:
    if row is None:
        return None
    try:
        plaintext = decrypt_secret(
            bytes(row["encrypted_credentials"]),
            str(row["encryption_kid"]),
        )
    except Exception:
        # KEK parser errors can contain the malformed configuration value.
        # Contain them at this credential boundary rather than allowing a
        # caller to interpolate secret-bearing exception text into its logs.
        logger.warning(
            "Scoped Gmail credential configuration invalid for context=%r",
            business_context_id,
        )
        return None
    if not plaintext:
        logger.warning(
            "Scoped Gmail credential decrypt failed for context=%r",
            business_context_id,
        )
        return None
    try:
        payload = json.loads(plaintext)
        return ScopedGmailCredentials(
            client_id=_required_secret(payload["client_id"], "client_id"),
            client_secret=_required_secret(
                payload["client_secret"],
                "client_secret",
            ),
            refresh_token=_required_secret(
                payload["refresh_token"],
                "refresh_token",
            ),
            generation=int(row["generation"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning(
            "Scoped Gmail credential payload invalid for context=%r",
            business_context_id,
        )
        return None
