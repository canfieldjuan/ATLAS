"""BYOK (Bring Your Own Keys) provider key resolution + DB-backed storage.

Customers configure their own provider API keys (Anthropic / OpenRouter
/ etc.) in the dashboard. The LLM Gateway router (PR-D4) calls the
resolver to fetch the right key per request, then proxies through to
the provider with the customer's credentials.

Storage model (PR-D5):
  - Encrypted at rest using ``atlas_brain.auth.encryption`` (Fernet).
  - Account-scoped via FK + WHERE clauses: account A cannot read or
    revoke B's keys.
  - Soft-delete (``revoked_at`` timestamp) preserves audit trail.
  - One active row per (account_id, provider) -- rotating just adds
    a new row and revokes the old.

Resolver order (in ``lookup_provider_key_async``):
  1. DB row WHERE account_id=$1 AND provider=$2 AND revoked_at IS NULL
     -- decrypts via ``atlas_brain.auth.encryption.decrypt_secret``.
  2. Env-var fallback -- ``ATLAS_BYOK_<PROVIDER>_<UNDERSCORED_UUID>``.
     Stays for local dev so the same env-var plumbing PR-D4 used
     keeps working when no DB row is present.

The LLM Gateway router (``api/llm_gateway.py``) calls the async
resolver so DB-stored keys ARE honored. The sync
``lookup_provider_key`` helper is kept for backward-compat with any
external caller that imports the legacy name; it only checks the
env-var fallback (no DB lookup).
"""

from __future__ import annotations

import logging
import os
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..auth.encryption import decrypt_secret, encrypt_secret

logger = logging.getLogger("atlas.services.byok_keys")


SUPPORTED_PROVIDERS = ("anthropic", "openrouter", "together", "groq")

# Display the first N characters of the raw provider key so the
# customer dashboard can hint which row is which without exposing
# the secret. The full key never leaves the encrypt/decrypt boundary.
KEY_PREFIX_LEN = 8


@dataclass(frozen=True)
class BYOKKeyRecord:
    """Display-safe view of a byok_keys row. Never carries the
    encrypted ciphertext or the decrypted plaintext."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    provider: str
    key_prefix: str
    label: str
    added_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]


def _env_var_name(provider: str, account_id: str) -> str:
    """Compose the env-var fallback name for a (provider, account)
    pair. Hyphens in the UUID become underscores so the name is
    portable across shells.
    """
    safe_account = account_id.replace("-", "_")
    return f"ATLAS_BYOK_{provider.upper()}_{safe_account}"


def _validate_provider(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported BYOK provider {provider!r}. "
            f"Expected one of {sorted(SUPPORTED_PROVIDERS)}."
        )


# ---- DB-backed CRUD -----------------------------------------------------


class BYOKKeyLimitExceeded(Exception):
    """Raised by ``insert_provider_key`` when the caller's plan caps
    the number of active BYOK providers and adding a new one would
    exceed the cap. The router translates this to 403."""


async def insert_provider_key(
    pool,
    *,
    account_id: _uuid.UUID,
    provider: str,
    raw_key: str,
    label: str = "",
    max_keys: Optional[int] = None,
) -> BYOKKeyRecord:
    """Encrypt + persist a customer's provider key.

    If an active row already exists for (account_id, provider), it is
    revoked first (transactionally) so the unique-active partial
    index holds. The customer can rotate keys without explicit revoke.

    When ``max_keys`` is supplied (and != -1), enforces a per-account
    cap on the number of active BYOK providers. The check + insert
    run inside a single transaction with ``SELECT ... FOR UPDATE``
    on the ``saas_accounts`` row, which serializes concurrent BYOK
    writes per account and closes the COUNT-then-INSERT race that
    would otherwise let two concurrent submissions both pass the
    count check. Raises ``BYOKKeyLimitExceeded`` when the cap is
    reached -- excludes the provider being rotated from the count.
    """
    _validate_provider(provider)
    if not raw_key or not str(raw_key).strip():
        raise ValueError("insert_provider_key: raw_key is required")
    raw_key = str(raw_key).strip()
    ciphertext, kid = encrypt_secret(raw_key)
    prefix = raw_key[:KEY_PREFIX_LEN]

    async with pool.transaction() as conn:
        # Serialize concurrent BYOK writes per account so the
        # plan-limit count check below cannot race. Other accounts'
        # writes proceed in parallel.
        await conn.execute(
            "SELECT id FROM saas_accounts WHERE id = $1 FOR UPDATE",
            account_id,
        )

        if max_keys is not None and max_keys != -1:
            count_row = await conn.fetchrow(
                """
                SELECT COUNT(*)::int AS active_count
                FROM byok_keys
                WHERE account_id = $1
                  AND provider != $2
                  AND revoked_at IS NULL
                """,
                account_id,
                provider,
            )
            active_count = int(count_row["active_count"]) if count_row else 0
            if active_count >= max_keys:
                raise BYOKKeyLimitExceeded(
                    f"Account already has {active_count} active BYOK "
                    f"providers; plan cap is {max_keys}."
                )

        await conn.execute(
            """
            UPDATE byok_keys
            SET revoked_at = NOW()
            WHERE account_id = $1 AND provider = $2 AND revoked_at IS NULL
            """,
            account_id,
            provider,
        )
        row = await conn.fetchrow(
            """
            INSERT INTO byok_keys (
                account_id, provider, encrypted_key, encryption_kid,
                key_prefix, label
            ) VALUES (
                $1, $2, $3, $4, $5, $6
            )
            RETURNING id, account_id, provider, key_prefix, label,
                      added_at, last_used_at, revoked_at
            """,
            account_id,
            provider,
            ciphertext,
            kid,
            prefix,
            label,
        )

    return BYOKKeyRecord(
        id=row["id"],
        account_id=row["account_id"],
        provider=row["provider"],
        key_prefix=row["key_prefix"],
        label=row["label"],
        added_at=row["added_at"],
        last_used_at=row["last_used_at"],
        revoked_at=row["revoked_at"],
    )


async def list_provider_keys(
    pool,
    *,
    account_id: _uuid.UUID,
) -> list[BYOKKeyRecord]:
    """Return all non-revoked BYOK keys for an account.

    Display-safe (never the ciphertext or plaintext)."""
    rows = await pool.fetch(
        """
        SELECT id, account_id, provider, key_prefix, label,
               added_at, last_used_at, revoked_at
        FROM byok_keys
        WHERE account_id = $1 AND revoked_at IS NULL
        ORDER BY added_at DESC
        """,
        account_id,
    )
    return [
        BYOKKeyRecord(
            id=row["id"],
            account_id=row["account_id"],
            provider=row["provider"],
            key_prefix=row["key_prefix"],
            label=row["label"],
            added_at=row["added_at"],
            last_used_at=row["last_used_at"],
            revoked_at=row["revoked_at"],
        )
        for row in rows
    ]


async def revoke_provider_key(
    pool,
    *,
    key_id: _uuid.UUID,
    account_id: _uuid.UUID,
) -> bool:
    """Soft-delete a BYOK key. Account-scoped: A cannot revoke B's
    keys. Returns True when a row was revoked, False otherwise."""
    row = await pool.fetchrow(
        """
        UPDATE byok_keys
        SET revoked_at = NOW()
        WHERE id = $1 AND account_id = $2 AND revoked_at IS NULL
        RETURNING id
        """,
        key_id,
        account_id,
    )
    return row is not None


async def touch_provider_key(
    pool,
    *,
    key_id: _uuid.UUID,
) -> None:
    """Update last_used_at on a successful lookup. Best-effort: a
    failure here does not block the gateway request."""
    try:
        await pool.execute(
            """
            UPDATE byok_keys
            SET last_used_at = NOW()
            WHERE id = $1
            """,
            key_id,
        )
    except Exception:
        logger.exception("byok_keys.touch_failed key_id=%s", key_id)


# ---- Resolver -----------------------------------------------------------


def _env_var_fallback(provider: str, account_id: str) -> Optional[str]:
    """Local-dev fallback only. Returns None when SaaS auth is
    enabled (prod) so a DB outage / unconfigured pool can never
    silently route customer traffic through atlas's process-level
    ``ATLAS_BYOK_*`` keys.

    PR-D5 review fix on top of the prior fail-closed work: the
    earlier patch addressed exceptions during DB query / decrypt,
    but ``pool.is_initialized=False`` still skipped the DB block
    entirely and reached this fallback. Tying the fallback to the
    SaaS-auth flag closes that path -- prod always has saas_auth
    enabled; local dev runs with it off.
    """
    from .. import config as _config

    if _config.settings.saas_auth.enabled:
        # Production: no env fallback. Customers must configure their
        # provider keys via /api/v1/byok-keys.
        return None
    raw = os.environ.get(_env_var_name(provider, account_id), "").strip()
    return raw or None


async def lookup_provider_key_async(
    pool,
    provider: str,
    account_id: str,
) -> Optional[str]:
    """Async resolver -- DB lookup first, env-var fallback second.

    Returns the raw plaintext key or None. None becomes 503 in the
    gateway router. Bumps ``last_used_at`` on a successful DB hit
    (best-effort).

    Fail-closed semantics:
      - DB query EXCEPTION (transient outage / connection error):
        return None. We do NOT fall back to env-var because that
        would silently route customer traffic through atlas's
        process-level keys during an outage and bill the wrong
        credentials.
      - DB row exists but DECRYPT FAILS (KEK rotation drift):
        return None. Same reason -- env fallback would mask a real
        configuration breakage.
      - DB row simply NOT PRESENT: legitimate "not configured" --
        env-var fallback fires (covers local dev where no DB row
        was inserted yet).
    """
    if provider not in SUPPORTED_PROVIDERS:
        logger.warning("BYOK lookup: unsupported provider %r", provider)
        return None
    try:
        acct_uuid = _uuid.UUID(account_id)
    except (ValueError, TypeError):
        logger.warning("BYOK lookup: invalid account_id %r", account_id)
        return None

    if pool is not None and getattr(pool, "is_initialized", True):
        try:
            row = await pool.fetchrow(
                """
                SELECT id, encrypted_key, encryption_kid
                FROM byok_keys
                WHERE account_id = $1 AND provider = $2
                  AND revoked_at IS NULL
                """,
                acct_uuid,
                provider,
            )
        except Exception:
            logger.exception("BYOK lookup: DB query failed -- failing closed")
            return None

        if row is not None:
            plaintext = decrypt_secret(bytes(row["encrypted_key"]), row["encryption_kid"])
            if plaintext is None:
                logger.warning(
                    "BYOK lookup: decrypt failed for provider=%s account=%s "
                    "(KEK rotation drift?) -- failing closed",
                    provider,
                    account_id,
                )
                return None
            await touch_provider_key(pool, key_id=row["id"])
            return plaintext

    return _env_var_fallback(provider, account_id)


def lookup_provider_key(provider: str, account_id: str) -> Optional[str]:
    """Sync resolver -- env-var fallback only.

    Kept for backward-compat with PR-D4's existing call site. PR-D4's
    gateway router calls this sync helper; the async resolver above
    is the new path that production callers should switch to so
    DB-stored keys are honored. PR-D4b will migrate the router over.
    """
    if provider not in SUPPORTED_PROVIDERS:
        logger.warning("BYOK lookup: unsupported provider %r", provider)
        return None
    return _env_var_fallback(provider, account_id)
