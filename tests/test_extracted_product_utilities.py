from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from extracted_content_pipeline.autonomous.tasks import _blog_deploy, _google_news
from extracted_content_pipeline.autonomous.tasks import _blog_ts as blog_ts_mod
from extracted_content_pipeline.autonomous.tasks._blog_ts import (
    build_post_ts,
    slug_to_var_name,
    update_blog_index,
)
from extracted_content_pipeline.autonomous.tasks._execution_progress import (
    _update_execution_progress,
    task_run_id,
)
from extracted_content_pipeline.storage.models import ScheduledTask
from extracted_content_pipeline.storage.repositories import scheduled_task as repo_mod


class _Response:
    def __init__(self, text: str, url: str):
        self.text = text
        self.url = url

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    async def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.responses.pop(0)

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.responses.pop(0)

    async def aclose(self):
        self.closed = True


class _Repo:
    def __init__(self):
        self.calls = []

    async def update_execution_metadata(self, exec_id, metadata):
        self.calls.append((exec_id, metadata))


def test_task_run_id_prefers_scheduler_execution_id():
    execution_id = uuid4()
    task = ScheduledTask(
        metadata={
            "_execution_id": str(execution_id),
            "run_id": "manual-run",
        }
    )

    assert task_run_id(task) == str(execution_id)


def test_task_run_id_falls_back_to_metadata_and_task_id():
    assert task_run_id(ScheduledTask(metadata={"run_id": "manual-run"})) == "manual-run"

    task = ScheduledTask(metadata={})
    assert task_run_id(task) == str(task.id)


@pytest.mark.asyncio
async def test_update_execution_progress_writes_metadata_to_configured_repo(monkeypatch):
    execution_id = uuid4()
    task = ScheduledTask(metadata={"_execution_id": str(execution_id)})
    repo = _Repo()
    monkeypatch.setattr(repo_mod, "_scheduled_task_repo", repo)

    await _update_execution_progress(
        task,
        stage="rendering",
        progress_current=2,
        progress_total=5,
        progress_message="Building draft",
        drafts_created=1,
    )

    assert repo.calls == [
        (
            execution_id,
            {
                "stage": "rendering",
                "progress_current": 2,
                "progress_total": 5,
                "progress_message": "Building draft",
                "drafts_created": 1,
            },
        )
    ]


def test_slug_to_var_name_handles_invalid_identifier_starts():
    assert slug_to_var_name("2026-report") == "post2026Report"
    assert slug_to_var_name("vendor.comparison") == "vendorComparison"


def test_build_post_ts_escapes_content_and_emits_seo_fields():
    _, content = build_post_ts(
        slug="pricing-guide",
        title="Pricing's Guide",
        description="Dollar ${value}",
        date_str="2026-05-01",
        author="Canfield AI",
        tags=["pricing"],
        topic_type="guide",
        charts_json=[],
        content="## Intro\nUse ${spend} wisely.",
        seo_title="AI pricing guide",
        seo_description="Plan LLM spend",
        target_keyword="AI cost monitoring",
        secondary_keywords=["LLM cost admin"],
        faq=[{"q": "Why?", "a": "Control spend."}],
        related_slugs=["llm-cost-admin"],
        cta={"label": "Audit costs"},
    )

    assert "seo_title: 'AI pricing guide'" in content
    assert "target_keyword: 'AI cost monitoring'" in content
    assert "secondary_keywords" in content
    assert "content: `" in content
    assert "\\${" in content
    assert "Pricing\\'s Guide" in content


def test_blog_ts_markdown_fallback_emits_safe_html(monkeypatch):
    monkeypatch.setattr(blog_ts_mod, "_md_converter", None)

    _, content = blog_ts_mod.build_post_ts(
        slug="fallback",
        title="Fallback",
        description="Fallback",
        date_str="2026-05-01",
        author="Canfield AI",
        tags=[],
        topic_type="guide",
        charts_json=[],
        content="# Title\n<script>alert(1)</script>",
    )

    assert "<h1>Title</h1>" in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content


def test_update_blog_index_skips_dynamic_glob_index(tmp_path):
    index = tmp_path / "index.ts"
    original = "export const posts = import.meta.glob('./*.ts')\n"
    index.write_text(original)

    assert update_blog_index(index, "pricing-guide", "pricingGuide") is False
    assert index.read_text() == original


@pytest.mark.asyncio
async def test_resolve_google_news_url_decodes_publisher_url():
    wrapper_url = "https://news.google.com/rss/articles/test-id?hl=en-US&gl=US&ceid=US:en"
    html = (
        '<div data-n-a-id="test-id" data-n-a-ts="1774297014" '
        'data-n-a-sg="sig-value"></div>'
    )
    batched = (
        ")]}'\n"
        '[["wrb.fr","Fbv4je",'
        '"[\\"garturlres\\",\\"https://example.com/article?x\\\\u003d1\\\\u0026gaa_at\\\\u003dtoken\\",1]",'
        'null,null,null,"generic"]]'
    )
    client = _Client([
        _Response(html, wrapper_url),
        _Response(batched, "https://news.google.com/_/DotsSplashUi/data/batchexecute"),
    ])

    resolved = await _google_news.resolve_google_news_url(wrapper_url, client=client)

    assert resolved == "https://example.com/article?x=1"
    assert client.get_calls[0][0] == wrapper_url
    assert client.post_calls[0][0].endswith("rpcids=Fbv4je")


@pytest.mark.asyncio
async def test_resolve_google_news_url_returns_non_wrapper_unchanged():
    url = "https://example.com/article"

    assert await _google_news.resolve_google_news_url(url, client=_Client([])) == url


@pytest.mark.asyncio
async def test_auto_deploy_blog_disabled_is_safe_noop(tmp_path):
    result = await _blog_deploy.auto_deploy_blog(
        str(tmp_path),
        "pricing-guide",
        enabled=False,
    )

    assert result == {"deployed": False, "skipped": "auto_deploy disabled"}


def test_auto_deploy_find_repo_root_returns_none_outside_git(tmp_path):
    assert _blog_deploy._find_repo_root(str(tmp_path)) is None
