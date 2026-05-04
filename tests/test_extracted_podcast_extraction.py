from __future__ import annotations

import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.podcast_example import DeterministicPodcastLLM
from extracted_content_pipeline.podcast_extraction import (
    PodcastExtractionConfig,
    PodcastExtractionService,
    parse_podcast_idea_response,
)
from extracted_content_pipeline.podcast_idea_data import FilePodcastIdeaProvider
from extracted_content_pipeline.podcast_ports import (
    LLMResponse,
    PodcastIdea,
    PodcastTranscript,
)
from extracted_content_pipeline.skills.registry import get_skill_registry


class _StubTranscriptRepo:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    async def read_transcripts(self, *, scope, episode_id=None, limit=20, filters=None):
        out = list(self._transcripts)
        if episode_id is not None:
            out = [t for t in out if t.episode_id == episode_id]
        return out[: max(0, limit)]


class _StubIdeaRepo:
    def __init__(self):
        self.saved: list[PodcastIdea] = []

    async def save_ideas(self, ideas, *, scope):
        ids: list[str] = []
        for idx, idea in enumerate(ideas):
            sid = f"id-{len(self.saved) + idx + 1}"
            self.saved.append(idea)
            ids.append(sid)
        return tuple(ids)

    async def read_ideas(self, *, scope, episode_id, limit=10):
        return tuple(i for i in self.saved if i.episode_id == episode_id)[:limit]


class _CountingLLM:
    def __init__(self, content: str = "[]"):
        self.calls = 0
        self.content = content

    async def complete(self, messages, *, max_tokens, temperature, metadata=None):
        self.calls += 1
        return LLMResponse(content=self.content, model="stub", usage={})


class _BYOExtractor:
    def __init__(self, ideas):
        self.ideas = list(ideas)
        self.calls = 0

    async def extract_ideas(self, *, scope, transcript, target_idea_count):
        self.calls += 1
        return self.ideas[:target_idea_count]


def test_parse_strips_think_and_fences() -> None:
    text = (
        "<think>internal</think>\n"
        "```json\n"
        "[{\"rank\": 1, \"summary\": \"hello\"}]\n"
        "```\n"
    )
    items = parse_podcast_idea_response(text)
    assert items == [{"rank": 1, "summary": "hello"}]


def test_parse_extracts_balanced_array_after_prose() -> None:
    text = "Here you go:\n\n[{\"rank\":1,\"summary\":\"x\"}]\nthanks!"
    items = parse_podcast_idea_response(text)
    assert items and items[0]["rank"] == 1


def test_parse_returns_none_on_garbage() -> None:
    assert parse_podcast_idea_response("not json at all") is None
    assert parse_podcast_idea_response("") is None


@pytest.mark.asyncio
async def test_extraction_calls_llm_once_per_transcript() -> None:
    transcript = PodcastTranscript(
        episode_id="ep-1",
        title="t",
        transcript_text="x" * 300,
    )
    llm_payload = json.dumps([
        {
            "rank": 1,
            "summary": "first",
            "arguments": ["a", "b", "c"],
            "key_quotes": ["a verbatim quote longer than eight words for this test"],
        },
        {"rank": 2, "summary": "second"},
        {"rank": 3, "summary": "third"},
    ])
    llm = _CountingLLM(content=llm_payload)
    ideas_repo = _StubIdeaRepo()

    service = PodcastExtractionService(
        transcripts=_StubTranscriptRepo([transcript]),
        ideas=ideas_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastExtractionConfig(target_idea_count=3),
    )
    result = await service.extract(scope=TenantScope(account_id="a"), episode_id="ep-1")

    assert llm.calls == 1
    assert result.ideas_extracted == 3
    assert [i.rank for i in ideas_repo.saved] == [1, 2, 3]


@pytest.mark.asyncio
async def test_extraction_with_byo_extractor_skips_llm() -> None:
    transcript = PodcastTranscript(episode_id="ep-9", transcript_text="x" * 300)
    seeded = [
        PodcastIdea(episode_id="ep-9", rank=1, summary="byo-1"),
        PodcastIdea(episode_id="ep-9", rank=2, summary="byo-2"),
    ]
    llm = _CountingLLM(content="garbage")
    ideas_repo = _StubIdeaRepo()
    extractor = _BYOExtractor(seeded)

    service = PodcastExtractionService(
        transcripts=_StubTranscriptRepo([transcript]),
        ideas=ideas_repo,
        llm=llm,
        skills=get_skill_registry(),
        extractor=extractor,
        config=PodcastExtractionConfig(target_idea_count=2),
    )
    result = await service.extract(scope=TenantScope(account_id="a"), episode_id="ep-9")

    assert llm.calls == 0
    assert extractor.calls == 1
    assert result.ideas_extracted == 2
    assert {i.summary for i in ideas_repo.saved} == {"byo-1", "byo-2"}


@pytest.mark.asyncio
async def test_extraction_handles_unparseable_response() -> None:
    transcript = PodcastTranscript(episode_id="ep-1", transcript_text="x" * 300)
    llm = _CountingLLM(content="no json here")
    ideas_repo = _StubIdeaRepo()

    service = PodcastExtractionService(
        transcripts=_StubTranscriptRepo([transcript]),
        ideas=ideas_repo,
        llm=llm,
        skills=get_skill_registry(),
    )
    result = await service.extract(scope=TenantScope(account_id="a"), episode_id="ep-1")

    assert result.ideas_extracted == 0
    assert result.skipped == 1
    assert any(err.get("reason") == "no_ideas_extracted" for err in result.errors)


@pytest.mark.asyncio
async def test_extraction_offline_llm_produces_target_count() -> None:
    """End-to-end smoke with the deterministic offline LLM."""

    transcript = PodcastTranscript(episode_id="ep-77", title="t", transcript_text="x" * 300)
    llm = DeterministicPodcastLLM()
    ideas_repo = _StubIdeaRepo()
    service = PodcastExtractionService(
        transcripts=_StubTranscriptRepo([transcript]),
        ideas=ideas_repo,
        llm=llm,
        skills=get_skill_registry(),
        config=PodcastExtractionConfig(target_idea_count=3),
    )

    result = await service.extract(scope=TenantScope(account_id="a"), episode_id="ep-77")

    assert result.ideas_extracted == 3
    assert len(llm.calls) == 1
    assert [i.rank for i in ideas_repo.saved] == [1, 2, 3]


def test_file_podcast_idea_provider_indexes_by_episode(tmp_path: Path) -> None:
    file = tmp_path / "ideas.json"
    file.write_text(json.dumps({
        "ideas": [
            {"episode_id": "ep-a", "rank": 1, "summary": "a1"},
            {"episode_id": "ep-a", "rank": 2, "summary": "a2"},
            {"episode_id": "ep-b", "rank": 1, "summary": "b1"},
        ],
    }), encoding="utf-8")

    provider = FilePodcastIdeaProvider.from_file(file)

    assert "ep-a" in provider.ideas_by_episode
    assert "ep-b" in provider.ideas_by_episode
    assert len(provider.ideas_by_episode["ep-a"]) == 2
    assert provider.ideas_by_episode["ep-a"][0].rank == 1
