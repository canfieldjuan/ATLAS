"""Exact-match Postgres cache for B2B/reporting LLM responses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger("atlas.services.b2b.llm_exact_cache")

CACHE_VERSION = "b2b-llm-exact-v1"


class CacheUnavailable(RuntimeError):
    """Raised when a cache request depends on a missing skill."""


class B2BLLMExactCacheHit(dict):
    """Simple dict wrapper for cached response metadata."""


def is_b2b_llm_exact_cache_enabled() -> bool:
    """Return whether the feature-flagged exact cache is enabled."""
    from ...config import settings

    return bool(getattr(settings.b2b_churn, "llm_exact_cache_enabled", False))


def _normalize_maybe_json_string(value: str) -> Any:
    stripped = value.strip()
    if stripped and stripped[0] in "{[":
        try:
            return _normalize_value(json.loads(stripped))
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _normalize_message(value: Any) -> dict[str, Any] | None:
    role = getattr(value, "role", None)
    content = getattr(value, "content", None)
    if role is None and content is None:
        return None

    payload: dict[str, Any] = {
        "role": str(role or ""),
        "content": _normalize_value(content or ""),
    }
    tool_calls = getattr(value, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _normalize_value(tool_calls)
    tool_call_id = getattr(value, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = str(tool_call_id)
    return payload


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)

    msg_payload = _normalize_message(value)
    if msg_payload is not None:
        return msg_payload

    if isinstance(value, str):
        return _normalize_maybe_json_string(value)

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _normalize_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, set):
        normalized = [_normalize_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize_value(item) for item in value]

    return str(value)


def canonicalize_for_cache(value: Any) -> str:
    """Return deterministic JSON for hashing exact cache requests."""
    normalized = _normalize_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def build_request_envelope(
    *,
    provider: str,
    model: str,
    messages: Any,
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized request envelope that drives the cache key."""
    envelope: dict[str, Any] = {
        "provider": str(provider or ""),
        "model": str(model or ""),
        "messages": _normalize_value(messages),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "temperature": float(temperature) if temperature is not None else None,
    }
    if response_format is not None:
        envelope["response_format"] = _normalize_value(response_format)
    if guided_json is not None:
        envelope["guided_json"] = _normalize_value(guided_json)
    if extra:
        envelope["extra"] = _normalize_value(extra)
    return envelope


def llm_identity(llm: Any) -> tuple[str, str]:
    """Return provider/model identifiers for a resolved LLM instance."""
    if llm is None:
        return "", ""
    provider = str(getattr(llm, "name", "") or "")
    model = str(
        getattr(llm, "model", "")
        or getattr(llm, "model_id", "")
        or getattr(llm, "model_name", "")
        or ""
    )
    return provider, model


def compute_cache_key(namespace: str, request_envelope: dict[str, Any]) -> str:
    """Hash cache version + namespace + canonical request envelope."""
    raw = f"{CACHE_VERSION}:{namespace}:{canonicalize_for_cache(request_envelope)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_skill_messages(skill_name: str, payload: Any) -> list[dict[str, str]]:
    """Build the exact message envelope used by call_llm_with_skill()."""
    from ...skills import get_skill_registry

    skill = get_skill_registry().get(skill_name)
    if skill is None:
        raise CacheUnavailable(f"Skill '{skill_name}' not found")

    return [
        {"role": "system", "content": skill.content},
        {
            "role": "user",
            "content": json.dumps(payload, separators=(",", ":"), default=str),
        },
    ]


def build_skill_request_envelope(
    *,
    namespace: str,
    skill_name: str,
    payload: Any,
    provider: str,
    model: str,
    max_tokens: int | None,
    temperature: float | None,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    """Return cache key + normalized envelope + exact skill messages."""
    messages = build_skill_messages(skill_name, payload)
    request_envelope = build_request_envelope(
        provider=provider,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
        extra={"skill_name": skill_name, **(extra or {})},
    )
    return compute_cache_key(namespace, request_envelope), request_envelope, messages


def _resolve_pool(pool: Any | None) -> Any | None:
    if pool is not None:
        return pool
    from ...storage.database import get_db_pool

    db_pool = get_db_pool()
    if not db_pool.is_initialized:
        return None
    return db_pool


def _json_field_to_mapping(value: Any) -> dict[str, Any]:
    """Normalize JSON/JSONB driver return values into a plain mapping."""
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, Mapping):
            return {str(key): item for key, item in parsed.items()}
    return {}


# Sentinel account UUID for atlas's internal pipeline calls (PR-D3).
# Atlas's existing pipeline writes to this cache 100k+ times/month
# without knowing about accounts; the sentinel marks those rows so
# customer cache hits cannot leak across tenants.
SENTINEL_ACCOUNT_ID = "00000000-0000-0000-0000-000000000000"


async def lookup_cached_text(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    pool: Any | None = None,
    account_id: str = SENTINEL_ACCOUNT_ID,
) -> B2BLLMExactCacheHit | None:
    """Return the cached text for an exact request envelope.

    ``account_id`` defaults to the SENTINEL so atlas's existing
    pipeline keeps working unmodified. Customer-facing callers
    (PR-D4 LLM Gateway router) pass the requesting account's UUID
    so cross-tenant hits are impossible -- the (cache_key, account_id)
    composite PK guarantees isolation at the storage layer.
    """
    if not namespace or not is_b2b_llm_exact_cache_enabled():
        return None

    db_pool = _resolve_pool(pool)
    if db_pool is None:
        return None

    cache_key = compute_cache_key(namespace, request_envelope)
    row = await db_pool.fetchrow(
        """
        WITH hit AS (
            UPDATE b2b_llm_exact_cache
            SET last_hit_at = NOW(),
                hit_count = hit_count + 1
            WHERE cache_key = $1 AND account_id = $2
            RETURNING cache_key, namespace, provider, model,
                      response_text, usage_json, metadata,
                      created_at, last_hit_at, hit_count
        )
        SELECT * FROM hit
        """,
        cache_key,
        account_id,
    )
    if row is None:
        return None

    logger.info("Exact LLM cache hit: %s", namespace)
    return B2BLLMExactCacheHit(
        cache_key=row["cache_key"],
        namespace=row["namespace"],
        provider=row["provider"],
        model=row["model"],
        response_text=row["response_text"],
        usage=_json_field_to_mapping(row["usage_json"]),
        metadata=_json_field_to_mapping(row["metadata"]),
        created_at=row["created_at"],
        last_hit_at=row["last_hit_at"],
        hit_count=row["hit_count"],
    )


async def store_cached_text(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    provider: str,
    model: str,
    response_text: str,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    pool: Any | None = None,
    account_id: str = SENTINEL_ACCOUNT_ID,
) -> bool:
    """Store a successful exact-match response.

    ``account_id`` defaults to the SENTINEL so atlas's existing
    pipeline keeps writing to its own cache namespace. Customer-
    facing callers pass the requesting account's UUID.
    """
    if (
        not namespace
        or not response_text
        or not str(response_text).strip()
        or not is_b2b_llm_exact_cache_enabled()
    ):
        return False

    db_pool = _resolve_pool(pool)
    if db_pool is None:
        return False

    cache_key = compute_cache_key(namespace, request_envelope)
    metadata_json = {
        "cache_version": CACHE_VERSION,
        **(_normalize_value(metadata or {}) or {}),
    }
    await db_pool.execute(
        """
        INSERT INTO b2b_llm_exact_cache (
            cache_key, account_id, namespace, provider, model,
            response_text, usage_json, metadata
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb
        )
        ON CONFLICT (cache_key, account_id) DO UPDATE SET
            provider = EXCLUDED.provider,
            model = EXCLUDED.model,
            response_text = EXCLUDED.response_text,
            usage_json = EXCLUDED.usage_json,
            metadata = EXCLUDED.metadata
        """,
        cache_key,
        account_id,
        namespace,
        str(provider or ""),
        str(model or ""),
        str(response_text).strip(),
        canonicalize_for_cache(usage or {}),
        canonicalize_for_cache(metadata_json),
    )
    logger.info("Stored exact LLM cache entry: %s", namespace)
    return True


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Synchronous exact-cache helpers cannot run inside an active event loop"
    )


def lookup_cached_text_sync(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    pool: Any | None = None,
) -> B2BLLMExactCacheHit | None:
    """Sync wrapper for exact-cache lookup."""
    return _run_coro_sync(
        lookup_cached_text(namespace, request_envelope, pool=pool)
    )


def store_cached_text_sync(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    provider: str,
    model: str,
    response_text: str,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    pool: Any | None = None,
) -> bool:
    """Sync wrapper for exact-cache store."""
    return bool(
        _run_coro_sync(
            store_cached_text(
                namespace,
                request_envelope,
                provider=provider,
                model=model,
                response_text=response_text,
                usage=usage,
                metadata=metadata,
                pool=pool,
            )
        )
    )
