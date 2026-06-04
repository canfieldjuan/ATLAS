"""Tenant brand voice profiles for Content Ops generation."""

from __future__ import annotations

import json
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from extracted_content_pipeline.brand_voice import (
    BrandVoiceProfile,
    brand_voice_profile_from_mapping,
)
from extracted_content_pipeline.campaign_ports import TenantScope

logger = logging.getLogger("atlas.content_ops_brand_voice_profiles")


@dataclass(frozen=True)
class ContentOpsBrandVoiceProfileRecord:
    """Display-safe saved brand voice profile row."""

    id: _uuid.UUID
    account_id: _uuid.UUID
    name: str
    descriptors: tuple[str, ...]
    exemplars: tuple[str, ...]
    banned_terms: tuple[str, ...]
    preferred_pov: Optional[str]
    reading_level: Optional[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime]

    def as_profile(self) -> BrandVoiceProfile:
        return BrandVoiceProfile(
            id=str(self.id),
            account_id=str(self.account_id),
            name=self.name,
            descriptors=self.descriptors,
            exemplars=self.exemplars,
            banned_terms=self.banned_terms,
            preferred_pov=self.preferred_pov,
            reading_level=self.reading_level,
            metadata=dict(self.metadata),
        )


async def create_brand_voice_profile(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsBrandVoiceProfileRecord:
    """Create one active brand voice profile for an account."""

    profile = _validated_profile(account_id=account_id, payload=payload)
    row = await pool.fetchrow(
        """
        INSERT INTO content_ops_brand_voice_profiles (
            account_id, name, descriptors, exemplars, banned_terms,
            preferred_pov, reading_level, metadata
        )
        VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6, $7, $8::jsonb)
        RETURNING id, account_id, name, descriptors, exemplars, banned_terms,
                  preferred_pov, reading_level, metadata,
                  created_at, updated_at, archived_at
        """,
        account_id,
        profile.name,
        _json_dump(profile.descriptors),
        _json_dump(profile.exemplars),
        _json_dump(profile.banned_terms),
        profile.preferred_pov,
        profile.reading_level,
        _json_dump(profile.metadata),
    )
    return _display_record(row)


async def update_brand_voice_profile(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    profile_id: _uuid.UUID,
    payload: Mapping[str, Any],
) -> ContentOpsBrandVoiceProfileRecord | None:
    """Update one active tenant profile. Missing/cross-tenant rows return None."""

    profile = _validated_profile(
        account_id=account_id,
        profile_id=profile_id,
        payload=payload,
    )
    row = await pool.fetchrow(
        """
        UPDATE content_ops_brand_voice_profiles
           SET name = $3,
               descriptors = $4::jsonb,
               exemplars = $5::jsonb,
               banned_terms = $6::jsonb,
               preferred_pov = $7,
               reading_level = $8,
               metadata = $9::jsonb,
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id, account_id, name, descriptors, exemplars, banned_terms,
                  preferred_pov, reading_level, metadata,
                  created_at, updated_at, archived_at
        """,
        profile_id,
        account_id,
        profile.name,
        _json_dump(profile.descriptors),
        _json_dump(profile.exemplars),
        _json_dump(profile.banned_terms),
        profile.preferred_pov,
        profile.reading_level,
        _json_dump(profile.metadata),
    )
    return _display_record(row) if row is not None else None


async def list_brand_voice_profiles(
    pool: Any,
    *,
    account_id: _uuid.UUID,
) -> list[ContentOpsBrandVoiceProfileRecord]:
    """Return active brand voice profiles for one tenant."""

    rows = await pool.fetch(
        """
        SELECT id, account_id, name, descriptors, exemplars, banned_terms,
               preferred_pov, reading_level, metadata,
               created_at, updated_at, archived_at
          FROM content_ops_brand_voice_profiles
         WHERE account_id = $1
           AND archived_at IS NULL
         ORDER BY updated_at DESC
        """,
        account_id,
    )
    return [_display_record(row) for row in rows]


async def get_brand_voice_profile_record(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    profile_id: _uuid.UUID,
) -> ContentOpsBrandVoiceProfileRecord | None:
    """Return one active tenant profile row."""

    row = await pool.fetchrow(
        """
        SELECT id, account_id, name, descriptors, exemplars, banned_terms,
               preferred_pov, reading_level, metadata,
               created_at, updated_at, archived_at
          FROM content_ops_brand_voice_profiles
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        """,
        profile_id,
        account_id,
    )
    return _display_record(row) if row is not None else None


async def lookup_brand_voice_profile(
    pool: Any,
    *,
    account_id: str | _uuid.UUID,
    profile_id: str | _uuid.UUID,
) -> BrandVoiceProfile | None:
    """Resolve a full stored brand voice profile for one tenant."""

    try:
        account_uuid = _uuid.UUID(str(account_id))
        profile_uuid = _uuid.UUID(str(profile_id))
    except (TypeError, ValueError):
        logger.warning(
            "brand voice profile lookup: invalid account_id/profile_id",
        )
        return None
    record = await get_brand_voice_profile_record(
        pool,
        account_id=account_uuid,
        profile_id=profile_uuid,
    )
    return record.as_profile() if record is not None else None


async def archive_brand_voice_profile(
    pool: Any,
    *,
    account_id: _uuid.UUID,
    profile_id: _uuid.UUID,
) -> bool:
    """Soft-delete one tenant brand voice profile."""

    row = await pool.fetchrow(
        """
        UPDATE content_ops_brand_voice_profiles
           SET archived_at = NOW(),
               updated_at = NOW()
         WHERE id = $1
           AND account_id = $2
           AND archived_at IS NULL
        RETURNING id
        """,
        profile_id,
        account_id,
    )
    return row is not None


def _validated_profile(
    *,
    account_id: _uuid.UUID,
    payload: Mapping[str, Any],
    profile_id: _uuid.UUID | None = None,
) -> BrandVoiceProfile:
    if not isinstance(payload, Mapping):
        raise ValueError("Brand voice profile payload must be an object")
    name = _clean(payload.get("name") or payload.get("label"))
    if not name:
        raise ValueError("Brand voice profile name is required")

    expected_id = str(profile_id) if profile_id is not None else None
    normalized = brand_voice_profile_from_mapping(
        {
            **dict(payload),
            "id": expected_id or payload.get("id") or payload.get("profile_id"),
            "account_id": str(account_id),
            "name": name,
        },
        scope=TenantScope(account_id=str(account_id)),
        profile_id=expected_id,
    )
    if normalized is None or not normalized.has_guidance():
        raise ValueError(
            "Brand voice profile requires at least one descriptor, exemplar, "
            "banned term, preferred POV, or reading level"
        )
    return BrandVoiceProfile(
        id=expected_id or normalized.id,
        account_id=str(account_id),
        name=name,
        descriptors=normalized.descriptors,
        exemplars=normalized.exemplars,
        banned_terms=normalized.banned_terms,
        preferred_pov=normalized.preferred_pov,
        reading_level=normalized.reading_level,
        metadata=dict(normalized.metadata or {}),
    )


def _display_record(row: Any) -> ContentOpsBrandVoiceProfileRecord:
    return ContentOpsBrandVoiceProfileRecord(
        id=row["id"],
        account_id=row["account_id"],
        name=str(row["name"] or ""),
        descriptors=_json_string_sequence(row["descriptors"]),
        exemplars=_json_string_sequence(row["exemplars"]),
        banned_terms=_json_string_sequence(row["banned_terms"]),
        preferred_pov=_optional_text(row["preferred_pov"]),
        reading_level=_optional_text(row["reading_level"]),
        metadata=_json_mapping(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row["archived_at"],
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_mapping(value: Any) -> dict[str, Any]:
    decoded = _decode_json(value, default={})
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _json_string_sequence(value: Any) -> tuple[str, ...]:
    decoded = _decode_json(value, default=[])
    if isinstance(decoded, (str, bytes, bytearray)) or not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded if str(item).strip())


def _decode_json(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _optional_text(value: Any) -> str | None:
    return _clean(value) or None


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


__all__ = [
    "ContentOpsBrandVoiceProfileRecord",
    "archive_brand_voice_profile",
    "create_brand_voice_profile",
    "get_brand_voice_profile_record",
    "list_brand_voice_profiles",
    "lookup_brand_voice_profile",
    "update_brand_voice_profile",
]
