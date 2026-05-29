"""Encrypted tenant Zendesk credentials for Content Ops macro writeback."""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from extracted_content_pipeline.faq_macro_writeback_zendesk import (
    ZendeskMacroCredentials,
)

from ..auth.encryption import decrypt_secret, encrypt_secret


logger = logging.getLogger("atlas.services.content_ops_zendesk_credentials")

TOKEN_PREFIX_LEN = 8


@dataclass(frozen=True)
class ContentOpsZendeskCredentialRecord:
    """Display-safe active Zendesk credential row."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    email: str
    api_token_prefix: str
    subdomain: str
    base_url: str
    label: str
    added_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]


async def upsert_zendesk_credentials(
    pool,
    *,
    account_id: _uuid.UUID,
    email: str,
    api_token: str,
    subdomain: str = "",
    base_url: str = "",
    label: str = "",
) -> ContentOpsZendeskCredentialRecord:
    """Encrypt and store one active Zendesk credential row for an account."""

    credentials = _validated_credentials(
        email=email,
        api_token=api_token,
        subdomain=subdomain,
        base_url=base_url,
    )
    cleaned_token = str(api_token or "").strip()
    ciphertext, kid = encrypt_secret(cleaned_token)
    token_prefix = cleaned_token[:TOKEN_PREFIX_LEN]

    async with pool.transaction() as conn:
        await conn.execute(
            "SELECT id FROM saas_accounts WHERE id = $1 FOR UPDATE",
            account_id,
        )
        await conn.execute(
            """
            UPDATE content_ops_zendesk_credentials
            SET revoked_at = NOW()
            WHERE account_id = $1 AND revoked_at IS NULL
            """,
            account_id,
        )
        row = await conn.fetchrow(
            """
            INSERT INTO content_ops_zendesk_credentials (
                account_id, email, encrypted_api_token, encryption_kid,
                api_token_prefix, subdomain, base_url, label
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            RETURNING id, account_id, email, api_token_prefix, subdomain,
                      base_url, label, added_at, last_used_at, revoked_at
            """,
            account_id,
            credentials.email,
            ciphertext,
            kid,
            token_prefix,
            credentials.subdomain,
            credentials.base_url,
            str(label or "").strip(),
        )

    return _display_record(row)


async def list_zendesk_credentials(
    pool,
    *,
    account_id: _uuid.UUID,
) -> list[ContentOpsZendeskCredentialRecord]:
    """Return display-safe active Zendesk credentials for one tenant."""

    rows = await pool.fetch(
        """
        SELECT id, account_id, email, api_token_prefix, subdomain,
               base_url, label, added_at, last_used_at, revoked_at
        FROM content_ops_zendesk_credentials
        WHERE account_id = $1 AND revoked_at IS NULL
        ORDER BY added_at DESC
        """,
        account_id,
    )
    return [_display_record(row) for row in rows]


async def revoke_zendesk_credentials(
    pool,
    *,
    account_id: _uuid.UUID,
    credential_id: _uuid.UUID,
) -> bool:
    """Soft-delete one tenant Zendesk credential row."""

    row = await pool.fetchrow(
        """
        UPDATE content_ops_zendesk_credentials
        SET revoked_at = NOW()
        WHERE id = $1 AND account_id = $2 AND revoked_at IS NULL
        RETURNING id
        """,
        credential_id,
        account_id,
    )
    return row is not None


async def lookup_zendesk_credentials(
    pool,
    *,
    account_id: str | _uuid.UUID,
) -> ZendeskMacroCredentials | None:
    """Resolve decrypted Zendesk credentials for one tenant."""

    try:
        account_uuid = _uuid.UUID(str(account_id))
    except (TypeError, ValueError):
        logger.warning("zendesk credential lookup: invalid account_id=%r", account_id)
        return None
    try:
        row = await pool.fetchrow(
            """
            SELECT id, email, encrypted_api_token, encryption_kid, subdomain, base_url
            FROM content_ops_zendesk_credentials
            WHERE account_id = $1 AND revoked_at IS NULL
            ORDER BY added_at DESC
            LIMIT 1
            """,
            account_uuid,
        )
    except Exception:
        logger.exception("zendesk credential lookup failed account_id=%s", account_uuid)
        return None
    if row is None:
        return None
    token = decrypt_secret(bytes(row["encrypted_api_token"]), str(row["encryption_kid"]))
    if not token:
        logger.warning(
            "zendesk credential decrypt failed account_id=%s credential_id=%s",
            account_uuid,
            row["id"],
        )
        return None
    credentials = ZendeskMacroCredentials(
        email=str(row["email"] or "").strip(),
        api_token=token,
        subdomain=str(row["subdomain"] or "").strip(),
        base_url=str(row["base_url"] or "").strip(),
    )
    if not credentials.is_complete():
        logger.warning(
            "zendesk credential row incomplete account_id=%s credential_id=%s",
            account_uuid,
            row["id"],
        )
        return None
    await _touch_credentials(pool, credential_id=row["id"])
    return credentials


def _validated_credentials(
    *,
    email: str,
    api_token: str,
    subdomain: str,
    base_url: str,
) -> ZendeskMacroCredentials:
    credentials = ZendeskMacroCredentials(
        email=str(email or "").strip(),
        api_token=str(api_token or "").strip(),
        subdomain=str(subdomain or "").strip(),
        base_url=str(base_url or "").strip(),
    )
    if not credentials.is_complete():
        raise ValueError("Complete Zendesk email, API token, and endpoint are required")
    return credentials


def _display_record(row) -> ContentOpsZendeskCredentialRecord:
    return ContentOpsZendeskCredentialRecord(
        id=row["id"],
        account_id=row["account_id"],
        email=str(row["email"] or ""),
        api_token_prefix=str(row["api_token_prefix"] or ""),
        subdomain=str(row["subdomain"] or ""),
        base_url=str(row["base_url"] or ""),
        label=str(row["label"] or ""),
        added_at=row["added_at"],
        last_used_at=row["last_used_at"],
        revoked_at=row["revoked_at"],
    )


async def _touch_credentials(pool, *, credential_id: _uuid.UUID) -> None:
    try:
        await pool.execute(
            """
            UPDATE content_ops_zendesk_credentials
            SET last_used_at = NOW()
            WHERE id = $1
            """,
            credential_id,
        )
    except Exception:
        logger.exception(
            "zendesk credential touch failed credential_id=%s",
            credential_id,
        )


__all__ = [
    "ContentOpsZendeskCredentialRecord",
    "TOKEN_PREFIX_LEN",
    "list_zendesk_credentials",
    "lookup_zendesk_credentials",
    "revoke_zendesk_credentials",
    "upsert_zendesk_credentials",
]
