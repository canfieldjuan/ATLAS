"""
Tests for edge voice pipeline escalation behavior.
"""

import asyncio
from types import SimpleNamespace

from atlas_edge.pipeline.voice_pipeline import EdgeVoicePipeline, PipelineResult


def test_offline_escalation_fallback_to_skill_clears_escalated_flag():
    """If offline escalation falls back to a local skill, result should not remain escalated."""
    pipeline = EdgeVoicePipeline()

    class MockEscalation:
        async def escalate(self, query, session_id=None, speaker_id=None):
            return SimpleNamespace(
                response_text="Brain is offline.",
                action_type="fallback",
                success=False,
                was_offline=True,
            )

    async def fake_try_skill(_query: str):
        return SimpleNamespace(
            success=True,
            response_text="Local skill response.",
            action_type="skill",
        )

    pipeline._escalation = MockEscalation()
    pipeline._try_skill = fake_try_skill

    result = asyncio.run(
        pipeline._escalate_to_brain("status", None, None, PipelineResult(success=False))
    )

    assert result.handled_locally is True
    assert result.success is True
    assert result.action_type == "skill"
    assert result.escalated is False


def test_offline_escalation_without_skill_stays_escalated():
    """If offline escalation has no local skill fallback, it should stay escalated."""
    pipeline = EdgeVoicePipeline()

    class MockEscalation:
        async def escalate(self, query, session_id=None, speaker_id=None):
            return SimpleNamespace(
                response_text="Brain is offline.",
                action_type="fallback",
                success=False,
                was_offline=True,
            )

    async def fake_try_skill(_query: str):
        return None

    pipeline._escalation = MockEscalation()
    pipeline._try_skill = fake_try_skill

    result = asyncio.run(
        pipeline._escalate_to_brain("status", None, None, PipelineResult(success=False))
    )

    assert result.handled_locally is False
    assert result.success is False
    assert result.action_type == "fallback"
    assert result.escalated is True
