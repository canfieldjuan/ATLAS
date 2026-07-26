"""Encrypted, exact-context credential state for scoped Gmail inbox reads."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
from typing import AsyncContextManager, AsyncIterator

from ...auth.encryption import decrypt_secret, encrypt_secret
from ..config import db_settings
from ..database import get_db_pool

logger = logging.getLogger("atlas.storage.scoped_mailbox_credentials")

# In-process refresh gate, one per (event loop, exact context). The row-level
# FOR UPDATE in locked_gmail() serializes refreshes across processes, but every
# in-process waiter would otherwise hold a pool connection while blocked on the
# same row -- ten concurrent scoped reads for one context could pin the entire
# default ten-connection pool behind one token exchange. Waiters queue here
# WITHOUT a connection; each process then holds at most one connection per
# context inside the locked section. Keyed by running loop because an
# asyncio.Lock is loop-bound; growth is bounded by loops x contexts (one loop in
# production, and contexts come from the validated binding config).
_REFRESH_GATES: dict[tuple[int, str], asyncio.Lock] = {}


def _refresh_gate(context: str) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), context)
    gate = _REFRESH_GATES.get(key)
    if gate is None:
        gate = _REFRESH_GATES[key] = asyncio.Lock()
    return gate


# The per-context gate bounds concurrency WITHIN one context; it does nothing
# ACROSS contexts. Ten distinct contexts refreshing at once each hold their own
# connection through Google's token exchange, which is the whole default pool --
# so unrelated work (invoicing, CRM reads) starves for the length of an external
# HTTP call. This semaphore caps how many connection-holding refresh sections run
# at once, portfolio-wide, leaving the rest of the pool for everything else.
# Waiters queue here WITHOUT a connection, same as the per-context gate.
# Keyed by (loop, budget) so a configuration reload that changes the pool size
# yields a semaphore sized to the NEW pool rather than pinning the first budget
# this process ever computed.
_REFRESH_SLOTS: dict[tuple[int, int], asyncio.Semaphore] = {}


def _refresh_budget() -> int:
    """Connections refreshes may occupy at once; the rest stay available."""
    return max(1, db_settings.max_pool_size // 2)


def _refresh_slot(budget: int | None = None) -> asyncio.Semaphore:
    if budget is None:
        budget = _refresh_budget()
    key = (id(asyncio.get_running_loop()), budget)
    slot = _REFRESH_SLOTS.get(key)
    if slot is None:
        slot = _REFRESH_SLOTS[key] = asyncio.Semaphore(budget)
    return slot

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

    def __init__(self, pool=None, refresh_budget: int | None = None) -> None:
        # Edge seam: tests hand in a pool double; production resolves the
        # process pool lazily so import order stays unconstrained.
        self._pool = pool
        # Same seam for the refresh budget: production derives it from the
        # configured pool size, tests state it outright. The semaphore stays
        # process-global (keyed by budget), so every instance deriving the same
        # budget shares one -- an instance cannot mint itself extra headroom.
        self._refresh_budget = refresh_budget

    def _db(self):
        return self._pool if self._pool is not None else get_db_pool()

    def _slot(self):
        return _refresh_slot(self._refresh_budget)

    async def bind_gmail(
        self,
        *,
        business_context_id: str,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> int:
        context = _exact_context(business_context_id)
        # Same-context mutations share the refresh gate: a rebind or revoke
        # arriving while a refresh holds the row would otherwise occupy a pool
        # connection blocked on the row lock -- the profile the gate exists to
        # prevent. Cross-context mutations are unaffected.
        async with _refresh_gate(context), self._slot():
            ciphertext, kid = _encrypt_bundle(
                client_id=client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
            )
            row = await self._db().fetchrow(
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
        row = await self._db().fetchrow(
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
        async with _refresh_gate(context), self._slot():
            row = await self._db().fetchrow(
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
        async with _refresh_gate(context), self._slot():
            async with self._db().transaction() as conn:
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
