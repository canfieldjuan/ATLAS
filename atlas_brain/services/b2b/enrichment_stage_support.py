from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def enrichment_batch_custom_id(stage: str, review_id: Any) -> str:
    normalized_stage = re.sub(r"[^A-Za-z0-9_-]+", "_", str(stage or "").strip()).strip("_") or "stage"
    normalized_review = re.sub(r"[^A-Za-z0-9_-]+", "_", str(review_id or "").strip()).strip("_") or "review"
    return f"{normalized_stage}_{normalized_review}"[:64]


def unpack_cached_lookup_result(
    result: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    if len(result) == 3:
        cached, request_envelope, cache_hit = result
        return cached, request_envelope, bool(cache_hit)
    if len(result) == 2:
        cached, request_envelope = result
        return cached, request_envelope, cached is not None
    raise ValueError(f"Unexpected cached lookup result shape: {len(result)}")


def unpack_stage_result(
    result: tuple[Any, ...],
) -> tuple[dict[str, Any] | None, str | None, bool]:
    if len(result) == 3:
        parsed, model_id, cache_hit = result
        return parsed, model_id, bool(cache_hit)
    if len(result) == 2:
        parsed, model_id = result
        return parsed, model_id, False
    raise ValueError(f"Unexpected stage result shape: {len(result)}")


def pack_stage_result(
    parsed: dict[str, Any] | None,
    model_id: str | None,
    cache_hit: bool,
    *,
    include_cache_hit: bool,
) -> tuple[dict[str, Any] | None, str | None] | tuple[dict[str, Any] | None, str | None, bool]:
    if include_cache_hit:
        return parsed, model_id, cache_hit
    return parsed, model_id


def prepare_stage_request(
    stage_id: str,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
):
    from ...services.b2b.cache_runner import prepare_b2b_exact_stage_request
    from ...autonomous.tasks._b2b_batch_utils import exact_stage_request_fingerprint

    work_fingerprint_payload = {
        "stage_id": str(stage_id or ""),
        "system_prompt": str(system_prompt or ""),
        "user_content": str(user_content or ""),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    request = prepare_b2b_exact_stage_request(
        stage_id,
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
        guided_json=guided_json,
    )
    work_fingerprint = hashlib.sha256(
        json.dumps(
            work_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return request, exact_stage_request_fingerprint(request), work_fingerprint


def stage_result_text(parsed: dict[str, Any] | None) -> str | None:
    if parsed is None:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def empty_exact_cache_usage() -> dict[str, int]:
    return {
        "exact_cache_hits": 0,
        "tier1_exact_cache_hits": 0,
        "tier2_exact_cache_hits": 0,
        "generated": 0,
        "tier1_generated_calls": 0,
        "tier2_generated_calls": 0,
        "witness_rows": 0,
        "witness_count": 0,
        "secondary_write_hits": 0,
    }


def stage_usage_snapshot(*, tier: int, cache_hit: bool, generated: bool) -> dict[str, int]:
    usage = empty_exact_cache_usage()
    if cache_hit:
        usage[f"tier{tier}_exact_cache_hits"] += 1
        usage["exact_cache_hits"] += 1
    elif generated:
        usage[f"tier{tier}_generated_calls"] += 1
        usage["generated"] += 1
    return usage


def stage_usage_from_row(row: dict[str, Any] | None, *, tier: int) -> dict[str, int]:
    usage = empty_exact_cache_usage()
    if not isinstance(row, dict):
        return usage
    stored = row.get("usage_json")
    if isinstance(stored, dict):
        for key in usage:
            usage[key] = int(stored.get(key) or 0)
        if any(usage.values()):
            return usage
    result_source = str(row.get("result_source") or "").strip().lower()
    if result_source == "exact_cache":
        return stage_usage_snapshot(tier=tier, cache_hit=True, generated=False)
    if result_source in {"generated", "batch_reuse"}:
        return stage_usage_snapshot(tier=tier, cache_hit=False, generated=True)
    return usage


def parse_stage_row_result(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    text = str(row.get("response_text") or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def accumulate_exact_cache_usage(
    totals: dict[str, int],
    usage: dict[str, Any] | None,
) -> None:
    if not usage:
        return
    for key in (
        "exact_cache_hits",
        "tier1_exact_cache_hits",
        "tier2_exact_cache_hits",
        "generated",
        "tier1_generated_calls",
        "tier2_generated_calls",
        "witness_rows",
        "witness_count",
        "secondary_write_hits",
    ):
        totals[key] = int(totals.get(key, 0) or 0) + int(usage.get(key, 0) or 0)


def row_usage_result(status: Any, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = {"status": status}
    usage_dict = usage or {}
    for key in empty_exact_cache_usage():
        normalized[key] = int(usage_dict.get(key, 0) or 0)
    return normalized
