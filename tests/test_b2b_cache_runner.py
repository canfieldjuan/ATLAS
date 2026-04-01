import json

import pytest

from atlas_brain.services.b2b.cache_runner import (
    lookup_b2b_exact_stage_text,
    prepare_b2b_exact_stage_request,
    run_b2b_exact_stage,
    store_b2b_exact_stage_text,
)


class _FakeLLM:
    name = "openrouter"
    model = "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_prepare_exact_stage_request_uses_declared_namespace():
    request = prepare_b2b_exact_stage_request(
        "b2b_campaign_generation.content",
        llm=_FakeLLM(),
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
        temperature=0.1,
    )
    assert request.namespace == "b2b_campaign_generation.content"
    assert request.provider == "openrouter"
    assert request.model == "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_lookup_and_store_stage_text_delegate_to_exact_cache(monkeypatch):
    captured = {}

    async def _fake_lookup(namespace, envelope, pool=None):
        captured["lookup"] = (namespace, envelope, pool)
        return {"response_text": "{\"ok\":true}", "usage": {"input_tokens": 1}}

    async def _fake_store(namespace, envelope, **kwargs):
        captured["store"] = (namespace, envelope, kwargs)
        return True

    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.llm_exact_cache.lookup_cached_text",
        _fake_lookup,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.llm_exact_cache.store_cached_text",
        _fake_store,
    )

    request = prepare_b2b_exact_stage_request(
        "b2b_blog_post_generation.content",
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
        temperature=0.7,
    )
    hit = await lookup_b2b_exact_stage_text(request)
    stored = await store_b2b_exact_stage_text(
        request,
        response_text='{"title":"x"}',
        metadata={"slug": "demo"},
    )

    assert hit["response_text"] == '{"ok":true}'
    assert stored is True
    assert captured["lookup"][0] == "b2b_blog_post_generation.content"
    assert captured["store"][0] == "b2b_blog_post_generation.content"


@pytest.mark.asyncio
async def test_run_b2b_exact_stage_caches_normalized_response(monkeypatch):
    cache = {}

    async def _fake_lookup(namespace, envelope, pool=None):
        return cache.get((namespace, json.dumps(envelope, sort_keys=True)))

    async def _fake_store(namespace, envelope, **kwargs):
        cache[(namespace, json.dumps(envelope, sort_keys=True))] = {
            "response_text": kwargs["response_text"],
            "usage": kwargs.get("usage") or {},
        }
        return True

    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.llm_exact_cache.lookup_cached_text",
        _fake_lookup,
    )
    monkeypatch.setattr(
        "atlas_brain.services.b2b.cache_runner.llm_exact_cache.store_cached_text",
        _fake_store,
    )

    def _parse_response(text: str) -> dict[str, str]:
        return json.loads(text)

    def _normalize_parsed(payload: dict[str, str]) -> dict[str, str]:
        payload = dict(payload)
        payload["expert_take"] = " ".join(payload["expert_take"].split()[:3])
        return payload

    first = await run_b2b_exact_stage(
        "b2b_churn_reports.scorecard_narrative",
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "demo"}],
        max_tokens=64,
        temperature=0.3,
        invoke=lambda: _completed('{"expert_take":"one two three four"}'),
        parse_response=_parse_response,
        normalize_parsed=_normalize_parsed,
        serialize_parsed=lambda parsed: json.dumps(parsed, separators=(",", ":")),
        should_store_parsed=lambda parsed: True,
        metadata={"task": "test"},
    )
    second = await run_b2b_exact_stage(
        "b2b_churn_reports.scorecard_narrative",
        provider="openrouter",
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "demo"}],
        max_tokens=64,
        temperature=0.3,
        invoke=lambda: _completed('{"expert_take":"should not run"}'),
        parse_response=_parse_response,
    )

    assert first.cached is False
    assert first.parsed == {"expert_take": "one two three"}
    assert second.cached is True
    assert second.parsed == {"expert_take": "one two three"}


async def _completed(value):
    return value
