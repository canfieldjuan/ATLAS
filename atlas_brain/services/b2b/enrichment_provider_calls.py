from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks.b2b_enrichment")


@dataclass(frozen=True)
class EnrichmentProviderCallDeps:
    unpack_cached_lookup_result: Any
    pack_stage_result: Any
    maybe_anthropic_cache: Any
    trace_enrichment_llm_call: Any
    build_classify_payload: Any
    tier2_system_prompt_for_content_type: Any
    lookup_cached_json_response: Any
    store_cached_json_response: Any
    tier1_json_schema: dict[str, Any]


async def lookup_cached_json_response(
    namespace: str,
    *,
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None = None,
    guided_json: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool]:
    from ...pipelines.llm import clean_llm_output, parse_json_response
    from ...services.b2b.cache_runner import (
        lookup_b2b_exact_stage_text,
        prepare_b2b_exact_stage_request,
    )

    request = prepare_b2b_exact_stage_request(
        namespace,
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
    cached = await lookup_b2b_exact_stage_text(request)
    if cached is None:
        return None, request.request_envelope, False

    text = clean_llm_output(cached["response_text"])
    parsed = parse_json_response(text, recover_truncated=True)
    if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
        return parsed, request.request_envelope, True
    return None, request.request_envelope, False


async def store_cached_json_response(
    namespace: str,
    request_envelope: dict[str, Any],
    *,
    provider: str,
    model: str,
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    from ...services.b2b.cache_runner import (
        bind_b2b_exact_stage_request,
        store_b2b_exact_stage_text,
    )

    request = bind_b2b_exact_stage_request(
        namespace,
        provider=provider,
        model=model,
        request_envelope=request_envelope,
    )
    await store_b2b_exact_stage_text(
        request,
        response_text=response_text,
        metadata=metadata,
    )


async def call_vllm_tier1(
    payload_json: str,
    cfg: Any,
    client: Any,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
    deps: EnrichmentProviderCallDeps,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = cfg.enrichment_tier1_model
    request_envelope: dict[str, Any] | None = None
    messages = [
        {"role": "system", "content": skill.content},
        {"role": "user", "content": payload_json},
    ]
    try:
        cached, request_envelope, cache_hit = deps.unpack_cached_lookup_result(
            await deps.lookup_cached_json_response(
                "b2b_enrichment.tier1",
                provider="vllm",
                model=model_id,
                system_prompt=skill.content,
                user_content=payload_json,
                max_tokens=cfg.enrichment_tier1_max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
                guided_json=deps.tier1_json_schema,
            )
        )
        if cached is not None:
            return deps.pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        call_started = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": cfg.enrichment_tier1_model,
                "messages": messages,
                "max_tokens": cfg.enrichment_tier1_max_tokens,
                "temperature": 0.0,
                "guided_json": deps.tier1_json_schema,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        deps.trace_enrichment_llm_call(
            "task.b2b_enrichment.tier1",
            provider="vllm",
            model=model_id,
            messages=messages,
            usage=body.get("usage", {}),
            metadata=trace_metadata,
            duration_ms=(time.monotonic() - call_started) * 1000,
            api_endpoint=f"{str(getattr(client, 'base_url', '')).rstrip('/')}/v1/chat/completions",
        )
        text = body["choices"][0]["message"]["content"].strip()
        if not text:
            return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)

        from ...pipelines.llm import clean_llm_output, parse_json_response

        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            if request_envelope is not None:
                await deps.store_cached_json_response(
                    "b2b_enrichment.tier1",
                    request_envelope,
                    provider="vllm",
                    model=model_id,
                    response_text=text,
                    metadata={"tier": 1, "backend": "vllm"},
                )
            return deps.pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
        return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 1 vLLM returned invalid JSON")
        return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.exception("Tier 1 vLLM call failed")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


async def call_openrouter_tier1(
    payload_json: str,
    cfg: Any,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
    deps: EnrichmentProviderCallDeps,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    import httpx
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier1")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier1' not found")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for enrichment")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = cfg.enrichment_openrouter_model or "anthropic/claude-haiku-4-5"
    request_envelope: dict[str, Any] | None = None
    messages = [
        {"role": "system", "content": skill.content},
        {"role": "user", "content": payload_json},
    ]
    try:
        cached, request_envelope, cache_hit = deps.unpack_cached_lookup_result(
            await deps.lookup_cached_json_response(
                "b2b_enrichment.tier1",
                provider="openrouter",
                model=model_id,
                system_prompt=skill.content,
                user_content=payload_json,
                max_tokens=max(cfg.enrichment_tier1_max_tokens, 4096),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return deps.pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as client:
            call_started = time.monotonic()
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": deps.maybe_anthropic_cache(model_id, messages),
                    "max_tokens": max(cfg.enrichment_tier1_max_tokens, 4096),
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            deps.trace_enrichment_llm_call(
                "task.b2b_enrichment.tier1",
                provider="openrouter",
                model=model_id,
                messages=messages,
                usage=body.get("usage", {}),
                metadata=trace_metadata,
                duration_ms=(time.monotonic() - call_started) * 1000,
                api_endpoint="https://openrouter.ai/api/v1/chat/completions",
                provider_request_id=resp.headers.get("x-request-id") or body.get("id"),
            )
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter returned no choices")
                return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            if not text and msg.get("reasoning"):
                reasoning = msg["reasoning"]
                import re as _re

                json_match = _re.search(r"\{[\s\S]*\}", reasoning)
                if json_match:
                    text = json_match.group(0)
            text = text.strip()
            if not text:
                logger.warning("OpenRouter returned empty content")
                return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)

            from ...pipelines.llm import clean_llm_output, parse_json_response

            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                if request_envelope is not None:
                    await deps.store_cached_json_response(
                        "b2b_enrichment.tier1",
                        request_envelope,
                        provider="openrouter",
                        model=model_id,
                        response_text=text,
                        metadata={"tier": 1, "backend": "openrouter"},
                    )
                return deps.pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
            logger.warning("OpenRouter tier 1 returned unparseable JSON")
            return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.exception("OpenRouter tier 1 call failed")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


async def call_vllm_tier2(
    tier1_result: dict,
    row: dict,
    cfg: Any,
    client: Any,
    truncate_length: int,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
    deps: EnrichmentProviderCallDeps,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    tier2_model = cfg.enrichment_tier2_model or cfg.enrichment_tier1_model
    payload = deps.build_classify_payload(row, truncate_length)
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)
    system_prompt = deps.tier2_system_prompt_for_content_type(
        skill.content,
        payload.get("content_type"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload_json},
    ]

    try:
        request_envelope: dict[str, Any] | None
        cached, request_envelope, cache_hit = deps.unpack_cached_lookup_result(
            await deps.lookup_cached_json_response(
                "b2b_enrichment.tier2",
                provider="vllm",
                model=tier2_model,
                system_prompt=system_prompt,
                user_content=payload_json,
                max_tokens=cfg.enrichment_tier2_max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return deps.pack_stage_result(cached, tier2_model, cache_hit, include_cache_hit=include_cache_hit)

        call_started = time.monotonic()
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": tier2_model,
                "messages": messages,
                "max_tokens": cfg.enrichment_tier2_max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        deps.trace_enrichment_llm_call(
            "task.b2b_enrichment.tier2",
            provider="vllm",
            model=tier2_model,
            messages=messages,
            usage=body.get("usage", {}),
            metadata=trace_metadata,
            duration_ms=(time.monotonic() - call_started) * 1000,
            api_endpoint=f"{str(getattr(client, 'base_url', '')).rstrip('/')}/v1/chat/completions",
        )
        text = body["choices"][0]["message"]["content"].strip()
        if not text:
            return deps.pack_stage_result(None, tier2_model, False, include_cache_hit=include_cache_hit)

        from ...pipelines.llm import clean_llm_output, parse_json_response

        text = clean_llm_output(text)
        parsed = parse_json_response(text, recover_truncated=True)
        if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
            if request_envelope is not None:
                await deps.store_cached_json_response(
                    "b2b_enrichment.tier2",
                    request_envelope,
                    provider="vllm",
                    model=tier2_model,
                    response_text=text,
                    metadata={"tier": 2, "backend": "vllm"},
                )
            return deps.pack_stage_result(parsed, tier2_model, False, include_cache_hit=include_cache_hit)
        return deps.pack_stage_result(None, tier2_model, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.warning("Tier 2 vLLM call failed", exc_info=True)
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)


async def call_openrouter_tier2(
    tier1_result: dict,
    row: dict,
    cfg: Any,
    truncate_length: int,
    *,
    include_cache_hit: bool = False,
    trace_metadata: dict[str, Any] | None = None,
    deps: EnrichmentProviderCallDeps,
) -> tuple[dict | None, str | None] | tuple[dict | None, str | None, bool]:
    import httpx
    from ...skills import get_skill_registry

    skill = get_skill_registry().get("digest/b2b_churn_extraction_tier2")
    if not skill:
        logger.warning("Skill 'digest/b2b_churn_extraction_tier2' not found for OpenRouter tier 2")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    api_key = cfg.openrouter_api_key
    if not api_key:
        logger.warning("OpenRouter API key not configured for tier 2 enrichment")
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)

    model_id = (
        cfg.enrichment_tier2_openrouter_model
        or cfg.enrichment_openrouter_model
        or "anthropic/claude-haiku-4-5"
    )
    payload = deps.build_classify_payload(row, truncate_length)
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)
    system_prompt = deps.tier2_system_prompt_for_content_type(
        skill.content,
        payload.get("content_type"),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload_json},
    ]

    try:
        request_envelope: dict[str, Any] | None
        cached, request_envelope, cache_hit = deps.unpack_cached_lookup_result(
            await deps.lookup_cached_json_response(
                "b2b_enrichment.tier2",
                provider="openrouter",
                model=model_id,
                system_prompt=system_prompt,
                user_content=payload_json,
                max_tokens=cfg.enrichment_tier2_max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        if cached is not None:
            return deps.pack_stage_result(cached, model_id, cache_hit, include_cache_hit=include_cache_hit)

        async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.enrichment_tier2_timeout_seconds, connect=10.0)) as client:
            call_started = time.monotonic()
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": deps.maybe_anthropic_cache(model_id, messages),
                    "max_tokens": cfg.enrichment_tier2_max_tokens,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            deps.trace_enrichment_llm_call(
                "task.b2b_enrichment.tier2",
                provider="openrouter",
                model=model_id,
                messages=messages,
                usage=body.get("usage", {}),
                metadata=trace_metadata,
                duration_ms=(time.monotonic() - call_started) * 1000,
                api_endpoint="https://openrouter.ai/api/v1/chat/completions",
                provider_request_id=resp.headers.get("x-request-id") or body.get("id"),
            )
            choices = body.get("choices") or []
            if not choices:
                logger.warning("OpenRouter tier 2 returned no choices")
                return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
            text = (choices[0].get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                logger.warning("OpenRouter tier 2 returned empty content")
                return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)

            from ...pipelines.llm import clean_llm_output, parse_json_response

            text = clean_llm_output(text)
            parsed = parse_json_response(text, recover_truncated=True)
            if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                if request_envelope is not None:
                    await deps.store_cached_json_response(
                        "b2b_enrichment.tier2",
                        request_envelope,
                        provider="openrouter",
                        model=model_id,
                        response_text=text,
                        metadata={"tier": 2, "backend": "openrouter"},
                    )
                return deps.pack_stage_result(parsed, model_id, False, include_cache_hit=include_cache_hit)
            logger.warning("OpenRouter tier 2 returned unparseable JSON")
            return deps.pack_stage_result(None, model_id, False, include_cache_hit=include_cache_hit)
    except Exception:
        logger.warning("OpenRouter tier 2 call failed", exc_info=True)
        return deps.pack_stage_result(None, None, False, include_cache_hit=include_cache_hit)
