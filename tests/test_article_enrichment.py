import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.autonomous.tasks import article_enrichment, news_intake
from atlas_brain.autonomous.tasks._google_news import resolve_google_news_url


class _FakeResponse:
    def __init__(self, text: str, url: str):
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, **kwargs):
        self.get_calls.append(url)
        return self._responses.pop(0)

    async def post(self, url, **kwargs):
        self.post_calls.append(url)
        return self._responses.pop(0)

    async def aclose(self):
        return None


class _Pool:
    def __init__(self, rows):
        self.is_initialized = True
        self.fetch = AsyncMock(return_value=rows)
        self.execute = AsyncMock(return_value="UPDATE 1")


@pytest.mark.asyncio
async def test_resolve_google_news_url_decodes_publisher_url():
    wrapper_url = (
        "https://news.google.com/rss/articles/test-id"
        "?oc=5&hl=en-US&gl=US&ceid=US:en"
    )
    html = (
        '<div data-n-a-id="test-id" data-n-a-ts="1774297014" '
        'data-n-a-sg="sig-value"></div>'
    )
    batched = (
        ")]}'\n\n"
        '[["wrb.fr","Fbv4je",'
        '"[\\"garturlres\\",\\"https://example.com/article?x\\\\u003d1\\\\u0026gaa_at\\\\u003dtoken\\",1]",'
        'null,null,null,"generic"]]'
    )
    client = _FakeAsyncClient([
        _FakeResponse(html, wrapper_url),
        _FakeResponse(batched, "https://news.google.com/_/DotsSplashUi/data/batchexecute"),
    ])

    resolved = await resolve_google_news_url(wrapper_url, client=client)

    assert resolved == "https://example.com/article?x=1"
    assert client.get_calls == [wrapper_url]
    assert client.post_calls == [
        "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je"
    ]


@pytest.mark.asyncio
async def test_fetch_google_rss_resolves_wrapper_urls(monkeypatch):
    wrapper_url = "https://news.google.com/rss/articles/test-id?oc=5"
    entry = {
        "title": "Example headline",
        "summary": "Example summary",
        "link": wrapper_url,
        "published": "Mon, 23 Mar 2026 20:04:31 GMT",
        "source": {"title": "Example Source"},
    }
    fake_feedparser = SimpleNamespace(parse=lambda url: SimpleNamespace(entries=[entry]))
    monkeypatch.setitem(sys.modules, "feedparser", fake_feedparser)
    monkeypatch.setattr(
        news_intake,
        "resolve_google_news_url",
        AsyncMock(return_value="https://example.com/article"),
    )

    articles = await news_intake._fetch_google_rss(
        ["crowdstrike"],
        max_articles=5,
        max_feeds=1,
        timeout=1.0,
    )

    assert len(articles) == 1
    assert articles[0]["url"] == "https://example.com/article"
    assert articles[0]["source_name"] == "Example Source"


@pytest.mark.asyncio
async def test_fetch_article_content_resolves_wrapper_and_returns_canonical_url(monkeypatch):
    import httpx

    seen_urls: list[str] = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            seen_urls.append(url)
            return _FakeResponse("<html><body>publisher page</body></html>", url)

    monkeypatch.setattr(
        article_enrichment,
        "resolve_google_news_url",
        AsyncMock(return_value="https://example.com/article"),
    )
    monkeypatch.setattr(
        article_enrichment,
        "html_to_text",
        lambda html, max_chars=30000: "A" * 120,
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client())

    cfg = SimpleNamespace(
        enrichment_fetch_timeout=5.0,
        enrichment_content_max_chars=1000,
    )
    content, final_url, reason = await article_enrichment._fetch_article_content(
        "https://news.google.com/rss/articles/test-id?oc=5",
        cfg,
    )

    assert final_url == "https://example.com/article"
    assert content == "A" * 120
    assert reason is None
    assert seen_urls == ["https://example.com/article"]


@pytest.mark.asyncio
async def test_fetch_article_content_returns_resolved_url_on_empty_extract(monkeypatch):
    import httpx

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            return _FakeResponse("<html><body>short</body></html>", url)

    monkeypatch.setattr(
        article_enrichment,
        "resolve_google_news_url",
        AsyncMock(return_value="https://example.com/article"),
    )
    monkeypatch.setattr(article_enrichment, "html_to_text", lambda html, max_chars=30000: "short")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client())

    cfg = SimpleNamespace(
        enrichment_fetch_timeout=5.0,
        enrichment_content_max_chars=1000,
    )
    content, final_url, reason = await article_enrichment._fetch_article_content(
        "https://news.google.com/rss/articles/test-id?oc=5",
        cfg,
    )

    assert content is None
    assert final_url == "https://example.com/article"
    assert reason == "fetch_extraction_empty"


@pytest.mark.asyncio
async def test_classify_soram_uses_default_triage_routing(monkeypatch):
    import atlas_brain.pipelines.llm as llm_mod

    captured: dict[str, object] = {}

    def _fake_call(skill_name, payload, **kwargs):
        captured["skill_name"] = skill_name
        captured["kwargs"] = kwargs
        return '{"soram_channels":{"societal":0.1,"operational":0.2,"regulatory":0.3,"alignment":0.4,"media":0.5},"linguistic_indicators":{"permission_shift":false,"certainty_spike":true,"linguistic_dissociation":false,"hedging_withdrawal":false,"urgency_escalation":true},"entities":["Example"],"pressure_direction":"building"}'

    monkeypatch.setattr(llm_mod, "call_llm_with_skill", _fake_call)
    monkeypatch.setattr(llm_mod, "parse_json_response", lambda text, recover_truncated=True: llm_mod.json.loads(text))

    result, reason = await article_enrichment._classify_soram(
        "Example title",
        "Example content",
        ["example"],
    )

    assert result["pressure_direction"] == "building"
    assert reason is None
    assert captured["skill_name"] == "digest/soram_classification"
    assert captured["kwargs"]["max_tokens"] == 1200
    assert captured["kwargs"]["workload"] == "vllm"
    assert "try_openrouter" not in captured["kwargs"]
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["guided_json"]["title"] == "soram_classification"


def test_validate_classification_defaults_pressure_direction():
    result = article_enrichment._validate_classification(
        {
            "soram_channels": {"societal": 0.2},
            "linguistic_indicators": {},
            "entities": ["Example"],
        }
    )

    assert result["pressure_direction"] == "unclear"


@pytest.mark.asyncio
async def test_classify_soram_returns_invalid_json_reason(monkeypatch):
    import atlas_brain.pipelines.llm as llm_mod

    monkeypatch.setattr(llm_mod, "call_llm_with_skill", lambda *args, **kwargs: '{"oops":true}')
    monkeypatch.setattr(llm_mod, "parse_json_response", lambda text, recover_truncated=True: llm_mod.json.loads(text))

    result, reason = await article_enrichment._classify_soram(
        "Example title",
        "Example content",
        ["example"],
    )

    assert result is None
    assert reason == "classify_invalid_json"


@pytest.mark.asyncio
async def test_run_marks_terminal_fetch_failures_with_reason(monkeypatch):
    row = {
        "id": "article-1",
        "title": "Example title",
        "url": "https://example.com/article",
        "summary": "Example summary",
        "matched_keywords": [],
        "enrichment_status": "pending",
        "enrichment_attempts": 2,
        "content": None,
    }
    pool = _Pool([row])
    cfg = article_enrichment.settings.external_data

    monkeypatch.setattr(article_enrichment, "get_db_pool", lambda: pool)
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "enrichment_enabled", True)
    monkeypatch.setattr(cfg, "enrichment_max_per_batch", 10)
    monkeypatch.setattr(cfg, "enrichment_max_attempts", 3)
    monkeypatch.setattr(
        article_enrichment,
        "_fetch_article_content",
        AsyncMock(return_value=(None, row["url"], "fetch_blocked")),
    )

    result = await article_enrichment.run(SimpleNamespace())

    assert result["failed"] == 1
    assert result["failure_reasons"] == {"fetch_blocked": 1}
    query, url, attempts, reason, status, article_id = pool.execute.await_args.args
    assert "enrichment_failure_reason = $3" in query
    assert url == row["url"]
    assert attempts == 3
    assert reason == "fetch_blocked"
    assert status == "failed"
    assert article_id == "article-1"
