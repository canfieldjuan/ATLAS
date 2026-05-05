"""API key generation, hashing, and lookup for the LLM Gateway product.

API keys are long-lived bearer tokens that customer scripts use to call
the ``/api/v1/llm/*`` endpoints. They sit alongside JWT auth (for the
dashboard) -- both resolve to the same ``AuthUser`` shape so downstream
endpoints are auth-method agnostic.

Hashing model: HMAC-SHA256(server_pepper, raw_key). The raw key is 32
random base32 characters (~160 bits entropy) so a KDF (bcrypt / PBKDF2)
would only add latency without security benefit. Pepper is a single
server-wide secret in ``SaaSAuthConfig.api_key_pepper``.

Format: ``atls_live_<32-base32-chars>`` so customers can pattern-match
in their secret-scanners (GitGuardian, TruffleHog, etc.).

Verification path:
  1. Caller presents ``Authorization: Bearer atls_live_xxxxx``.
  2. ``lookup_api_key`` narrows candidates by ``key_prefix`` (the first
     ``KEY_PREFIX_LEN`` chars of the raw key) -- typically 0-2 rows.
  3. HMAC-compare the candidate hash against ``HMAC-SHA256(pepper, raw)``.
  4. Constant-time compare the two hex digests.
  5. On match: bump ``last_used_at`` / ``last_used_ip`` and return the row.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid as _uuid
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger("atlas.auth.api_keys")


KEY_LIVE_PREFIX = "atls_live_"
KEY_BODY_LENGTH = 32  # base32 characters; 32 * 5 = 160 bits of entropy.
KEY_PREFIX_LEN = len(KEY_LIVE_PREFIX) + 8  # store ``atls_live_`` + 8 body chars.
DEFAULT_SCOPES = ("llm:*",)


@dataclass(frozen=True)
class APIKeyRecord:
    """Subset of the api_keys row that callers handle. Never carries
    the raw key -- raw is returned exactly once at creation time.
    """

    id: _uuid.UUID
    account_id: _uuid.UUID
    user_id: Optional[_uuid.UUID]
    name: str
    key_prefix: str
    scopes: tuple[str, ...]
    last_used_at: Optional[object]  # datetime; typed loose to avoid import
    created_at: object  # datetime
    revoked_at: Optional[object]  # datetime


@dataclass(frozen=True)
class APIKeyMintResult:
    """Returned at creation time. ``raw_key`` is the value the customer
    must store -- it is NOT recoverable after this moment."""

    raw_key: str
    record: APIKeyRecord


# ---- Generation + hashing ------------------------------------------------


def _base32_body(length: int) -> str:
    """Generate ``length`` random base32 characters (lowercase a-z2-7).

    Uses ``secrets.choice`` over a 32-char alphabet so each character is
    5 bits of entropy. We strip the standard padding ``=`` because we
    are not encoding bytes, just sampling characters.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz234567"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_api_key() -> tuple[str, str]:
    """Generate a fresh raw key + its prefix.

    Returns ``(raw_key, key_prefix)``. Caller is responsible for
    hashing the raw key before persisting and surfacing the raw key
    to the customer exactly once.
    """
    body = _base32_body(KEY_BODY_LENGTH)
    raw = f"{KEY_LIVE_PREFIX}{body}"
    prefix = raw[:KEY_PREFIX_LEN]
    return raw, prefix


def _pepper() -> bytes:
    """Read the API-key hashing pepper from SaaSAuthConfig.

    The pepper is server-wide; a per-key salt is unnecessary because
    the raw key already has ~160 bits of entropy. The pepper guards
    against an attacker with read-only DB access reconstructing keys
    -- without the pepper, the stored hash is useless to them.

    Read via module-attribute access on ``..config`` so that test code
    that reloads the config module sees the new pepper without also
    reloading this module.
    """
    from .. import config as _config

    pepper = getattr(_config.settings.saas_auth, "api_key_pepper", "") or ""
    return pepper.encode("utf-8")


def hash_api_key(raw_key: str) -> str:
    """Hash a raw key for storage. Returns hex-encoded HMAC-SHA256."""
    return hmac.new(_pepper(), raw_key.encode("utf-8"), hashlib.sha256).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    """Wrap ``hmac.compare_digest`` for str inputs. Public helper so
    test code can assert the verify path uses constant-time comparison.
    """
    return hmac.compare_digest(a, b)


def split_prefix(raw_key: str) -> str:
    """Extract the ``key_prefix`` (lookup hint) from a raw key.

    Returns the empty string when the input is too short or does not
    start with the live prefix; callers should treat that as "no
    candidate keys" rather than raising.
    """
    if not raw_key.startswith(KEY_LIVE_PREFIX):
        return ""
    if len(raw_key) < KEY_PREFIX_LEN:
        return ""
    return raw_key[:KEY_PREFIX_LEN]


# ---- DB-backed lookup / mutation ----------------------------------------


async def lookup_api_key(pool, raw_key: str) -> Optional[dict]:
    """Look up the active api_keys row that matches ``raw_key``.

    Returns the row as a dict (asyncpg.Record fields) when verified;
    returns None when the key is unrecognized, revoked, or the hash
    does not match.

    The lookup is split into a prefix narrow + an HMAC compare so
    high-throughput verification stays O(1) per request even at
    100k keys.
    """
    prefix = split_prefix(raw_key)
    if not prefix:
        return None

    rows = await pool.fetch(
        """
        SELECT id, account_id, user_id, name, key_prefix, key_hash, scopes,
               last_used_at, created_at, revoked_at
        FROM api_keys
        WHERE key_prefix = $1
          AND revoked_at IS NULL
        """,
        prefix,
    )
    if not rows:
        return None

    candidate_hash = hash_api_key(raw_key)
    for row in rows:
        if constant_time_equals(str(row["key_hash"]), candidate_hash):
            return dict(row)
    return None


async def insert_api_key(
    pool,
    *,
    account_id: _uuid.UUID,
    user_id: Optional[_uuid.UUID],
    name: str,
    scopes: Optional[Sequence[str]] = None,
) -> APIKeyMintResult:
    """Create a new api_keys row and return the raw key (one-time)
    plus the persisted record."""
    raw_key, prefix = generate_api_key()
    key_hash = hash_api_key(raw_key)
    scope_values = tuple(scopes) if scopes else DEFAULT_SCOPES

    row = await pool.fetchrow(
        """
        INSERT INTO api_keys (account_id, user_id, name, key_prefix, key_hash, scopes)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, account_id, user_id, name, key_prefix, scopes,
                  last_used_at, created_at, revoked_at
        """,
        account_id,
        user_id,
        name,
        prefix,
        key_hash,
        list(scope_values),
    )
    record = APIKeyRecord(
        id=row["id"],
        account_id=row["account_id"],
        user_id=row["user_id"],
        name=row["name"],
        key_prefix=row["key_prefix"],
        scopes=tuple(row["scopes"]),
        last_used_at=row["last_used_at"],
        created_at=row["created_at"],
        revoked_at=row["revoked_at"],
    )
    return APIKeyMintResult(raw_key=raw_key, record=record)


async def list_api_keys(pool, *, account_id: _uuid.UUID) -> list[APIKeyRecord]:
    """Return all non-revoked keys for an account.

    Never includes raw keys or hashes -- only display-safe fields.
    Callers that need to show revoked keys (audit views) can filter
    server-side; this list path is for the customer dashboard.
    """
    rows = await pool.fetch(
        """
        SELECT id, account_id, user_id, name, key_prefix, scopes,
               last_used_at, created_at, revoked_at
        FROM api_keys
        WHERE account_id = $1 AND revoked_at IS NULL
        ORDER BY created_at DESC
        """,
        account_id,
    )
    return [
        APIKeyRecord(
            id=row["id"],
            account_id=row["account_id"],
            user_id=row["user_id"],
            name=row["name"],
            key_prefix=row["key_prefix"],
            scopes=tuple(row["scopes"]),
            last_used_at=row["last_used_at"],
            created_at=row["created_at"],
            revoked_at=row["revoked_at"],
        )
        for row in rows
    ]


async def revoke_api_key(
    pool,
    *,
    key_id: _uuid.UUID,
    account_id: _uuid.UUID,
) -> bool:
    """Soft-delete a key by setting revoked_at = NOW().

    Scoped to ``account_id`` so account A cannot revoke account B's
    keys. Returns True when a row was revoked, False when no matching
    active key existed.
    """
    row = await pool.fetchrow(
        """
        UPDATE api_keys
        SET revoked_at = NOW()
        WHERE id = $1 AND account_id = $2 AND revoked_at IS NULL
        RETURNING id
        """,
        key_id,
        account_id,
    )
    return row is not None


async def touch_api_key(
    pool,
    *,
    key_id: _uuid.UUID,
    client_ip: Optional[str],
) -> None:
    """Update last_used_at + last_used_ip on a successful auth.

    Best-effort: a failure here does not block the caller. We surface
    via logger.exception so observability catches any DB regression.
    """
    try:
        await pool.execute(
            """
            UPDATE api_keys
            SET last_used_at = NOW(), last_used_ip = $2
            WHERE id = $1
            """,
            key_id,
            client_ip,
        )
    except Exception:
        logger.exception("api_keys.touch_failed key_id=%s", key_id)
