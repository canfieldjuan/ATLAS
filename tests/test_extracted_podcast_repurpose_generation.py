from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.podcast_example import DeterministicPodcastLLM
from extracted_content_pipeline.podcast_ports import (
    LLMResponse,
    PodcastFormatDraft,
    PodcastIdea,
    PodcastTranscript,
)
from extracted_content_pipeline.podcast_repurpose_generation import (
    SUPPORTED_FORMATS,
    PodcastRepurposeConfig,
    PodcastRepurposeService,
    parse_podcast_format_response,
)
from extracted_content_pipeline.skills.registry import get_skill_registry


class _IdeaRepo:
    def __init__(self, ideas):
        self.saved = list(ideas)

    async def save_ideas(self, ideas, *, scope):
        return tuple(f"id-{i}" for i, _ in enumerate(ideas))

    async def read_ideas(self, *, scope, episode_id, limit=10):
        return tuple(i for i in self.saved if i.episode_id == episode_id)[:limit]


class _TranscriptRepo:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    async def read_transcripts(self, *, scope, episode_id=None, limit=20, filters=None):
        out = list(self._transcripts)
        if episode_id is not None:
            out = [t for t in out if t.episode_id == episode_id]
        return out[: max(0, limit)]


class _DraftRepo:
    def __init__(self):
        self.saved: list[PodcastFormatDraft] = []

    async def save_drafts(self, drafts, *, scope):
        ids: list[str] = []
        for idx, draft in enumerate(drafts):
            self.saved.append(draft)
            ids.append(f"d-{len(self.saved)}")
        return tuple(ids)


class _CountingLLM:
    def __init__(self, content_provider):
        self.calls: list[dict] = []
        self._provider = content_provider

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        meta = dict(metadata or {})
        self.calls.append({"max_tokens": max_tokens, "metadata": meta})
        return LLMResponse(content=self._provider(meta), model="stub", usage={})


def _good_payload(format_type: str) -> str:
    body = "Body for " + format_type + " " + ("word " * 600)
    return json.dumps({
        "title": f"T-{format_type}",
        "body": body,
        "format_type": format_type,
        "metadata": {"word_count": 600},
    })


def _make_idea(rank: int) -> PodcastIdea:
    return PodcastIdea(
        idea_id=f"id-{rank}",
        episode_id="ep-77",
        rank=rank,
        summary=f"summary {rank}",
        arguments=("a", "b"),
        hooks=("h1",),
        key_quotes=("a verbatim quote longer than eight words for this test",),
        teaching_moments=("the takeaway",),
    )


def test_parse_extracts_object() -> None:
    text = '{"title":"t","body":"b","format_type":"newsletter","metadata":{"word_count":1}}'
    parsed = parse_podcast_format_response(text)
    assert parsed and parsed["title"] == "t" and parsed["body"] == "b"


def test_parse_handles_fences_and_think() -> None:
    text = "<think>x</think>```json\n{\"body\":\"hi\",\"title\":\"t\"}\n```"
    parsed = parse_podcast_format_response(text)
    assert parsed and parsed["body"] == "hi"


@pytest.mark.asyncio
async def test_three_ideas_times_five_formats_yields_fifteen_calls() -> None:
    ideas = [_make_idea(r) for r in (1, 2, 3)]
    llm = _CountingLLM(lambda meta: _good_payload(meta["format_type"]))
    drafts_repo = _DraftRepo()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=3),
    )

    result = await service.repurpose(
        scope=TenantScope(account_id="a"),
        episode_id="ep-77",
    )

    assert len(llm.calls) == 15
    assert result.drafts_generated == 15
    assert {d.format_type for d in drafts_repo.saved} == set(SUPPORTED_FORMATS)


@pytest.mark.asyncio
async def test_format_filter_limits_calls() -> None:
    ideas = [_make_idea(r) for r in (1, 2, 3)]
    llm = _CountingLLM(lambda meta: _good_payload(meta["format_type"]))
    drafts_repo = _DraftRepo()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=3),
    )

    result = await service.repurpose(
        scope=TenantScope(account_id="a"),
        episode_id="ep-77",
        formats=("x_thread",),
    )

    assert len(llm.calls) == 3
    assert {c["metadata"]["format_type"] for c in llm.calls} == {"x_thread"}
    assert result.drafts_generated == 3


@pytest.mark.asyncio
async def test_per_format_max_tokens_used() -> None:
    ideas = [_make_idea(1)]
    llm = _CountingLLM(lambda meta: _good_payload(meta["format_type"]))
    drafts_repo = _DraftRepo()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=1),
    )

    await service.repurpose(scope=TenantScope(account_id="a"), episode_id="ep-77")

    by_format = {c["metadata"]["format_type"]: c["max_tokens"] for c in llm.calls}
    assert by_format["blog"] == 6000
    assert by_format["newsletter"] == 3500
    assert by_format["linkedin"] == 800
    assert by_format["shorts"] == 800
    assert by_format["x_thread"] == 1500


@pytest.mark.asyncio
async def test_unparseable_response_records_error_and_continues() -> None:
    ideas = [_make_idea(1)]

    def content(meta):
        if meta["format_type"] == "blog":
            return "garbage that is not json"
        return _good_payload(meta["format_type"])

    llm = _CountingLLM(content)
    drafts_repo = _DraftRepo()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=1),
    )

    result = await service.repurpose(scope=TenantScope(account_id="a"), episode_id="ep-77")

    assert result.drafts_generated == 4
    assert result.skipped == 1
    assert any(err.get("format_type") == "blog" for err in result.errors)
    assert "blog" not in {d.format_type for d in drafts_repo.saved}


@pytest.mark.asyncio
async def test_quality_audit_attached_to_metadata() -> None:
    ideas = [_make_idea(1)]
    llm = _CountingLLM(lambda meta: _good_payload(meta["format_type"]))
    drafts_repo = _DraftRepo()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=1, formats=("linkedin",)),
    )

    await service.repurpose(scope=TenantScope(account_id="a"), episode_id="ep-77")

    assert len(drafts_repo.saved) == 1
    draft = drafts_repo.saved[0]
    assert "quality_audit" in draft.metadata
    assert draft.quality_audit.get("format_type") == "linkedin"


@pytest.mark.asyncio
async def test_offline_llm_produces_passing_drafts_for_all_five_formats() -> None:
    ideas = [_make_idea(1)]
    drafts_repo = _DraftRepo()
    llm = DeterministicPodcastLLM()

    service = PodcastRepurposeService(
        ideas=_IdeaRepo(ideas),
        transcripts=_TranscriptRepo([PodcastTranscript(episode_id="ep-77")]),
        drafts=drafts_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastRepurposeConfig(idea_limit=1),
    )

    await service.repurpose(scope=TenantScope(account_id="a"), episode_id="ep-77")

    assert len(drafts_repo.saved) == 5
    statuses = {d.format_type: d.quality_audit.get("status") for d in drafts_repo.saved}
    assert statuses == {fmt: "pass" for fmt in SUPPORTED_FORMATS}
