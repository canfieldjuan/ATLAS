"""
Tests for LLM synthesis post-processing in HeadlessRunner.

Covers: skill loading, think-tag stripping, config toggle,
missing skill / no LLM fallback, and _run_builtin integration.
"""

import re
from dataclasses import field
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from atlas_brain.autonomous.runner import HeadlessRunner
from atlas_brain.skills import get_skill_registry
from atlas_brain.storage.models import ScheduledTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builtin_task(
    handler: str = "gmail_digest",
    synthesis_skill: str | None = None,
) -> ScheduledTask:
    metadata = {"builtin_handler": handler}
    if synthesis_skill:
        metadata["synthesis_skill"] = synthesis_skill
    return ScheduledTask(
        id=uuid4(),
        name=f"test_{handler}",
        task_type="builtin",
        schedule_type="cron",
        cron_expression="0 8 * * *",
        enabled=True,
        metadata=metadata,
    )


SAMPLE_RAW_RESULT = {
    "total_emails": 5,
    "emails": [
        {"from": "boss@example.com", "subject": "Q3 Review", "snippet": "Please review..."},
        {"from": "github@github.com", "subject": "PR #42 merged", "snippet": "Your pull request..."},
    ],
}


# ---------------------------------------------------------------------------
# Skill loading
# ---------------------------------------------------------------------------

class TestSkillLoading:
    """Verify the digest/email_triage skill loads correctly."""

    def test_email_triage_skill_exists(self):
        registry = get_skill_registry()
        registry.reload()
        skill = registry.get("digest/email_triage")
        assert skill is not None

    def test_email_triage_skill_metadata(self):
        registry = get_skill_registry()
        skill = registry.get("digest/email_triage")
        assert skill.domain == "digest"
        assert "email" in skill.tags
        assert "triage" in skill.tags
        assert "autonomous" in skill.tags
        assert skill.version == 1

    def test_email_triage_skill_has_content(self):
        registry = get_skill_registry()
        skill = registry.get("digest/email_triage")
        assert len(skill.content) > 100
        assert "Action Required" in skill.content


# ---------------------------------------------------------------------------
# Think-tag stripping (same regex used in runner)
# ---------------------------------------------------------------------------

class TestThinkTagStripping:
    """Verify <think> tags are stripped from synthesis output."""

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    def test_strips_single_think_block(self):
        text = "<think>reasoning here</think>The actual summary."
        cleaned = self._THINK_RE.sub("", text).strip()
        assert cleaned == "The actual summary."

    def test_strips_multiline_think_block(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nHere is your digest."
        cleaned = self._THINK_RE.sub("", text).strip()
        assert cleaned == "Here is your digest."

    def test_no_think_tags_unchanged(self):
        text = "Just a normal summary."
        cleaned = self._THINK_RE.sub("", text).strip()
        assert cleaned == text


# ---------------------------------------------------------------------------
# _synthesize_with_skill unit tests
# ---------------------------------------------------------------------------

class TestSynthesizeWithSkill:
    """Test _synthesize_with_skill in isolation."""

    @pytest.fixture
    def runner(self):
        r = HeadlessRunner()
        return r

    @pytest.mark.asyncio
    async def test_returns_none_when_synthesis_disabled(self, runner):
        with patch("atlas_brain.autonomous.runner.autonomous_config") as cfg:
            cfg.synthesis_enabled = False
            result = await runner._synthesize_with_skill(
                SAMPLE_RAW_RESULT, "digest/email_triage", "test",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_skill(self, runner):
        with patch("atlas_brain.autonomous.runner.autonomous_config") as cfg:
            cfg.synthesis_enabled = True
            result = await runner._synthesize_with_skill(
                SAMPLE_RAW_RESULT, "nonexistent/skill", "test",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_llm(self, runner):
        with (
            patch("atlas_brain.autonomous.runner.autonomous_config") as cfg,
            patch("atlas_brain.services.llm_registry") as mock_reg,
        ):
            cfg.synthesis_enabled = True
            mock_reg.get_active.return_value = None
            result = await runner._synthesize_with_skill(
                SAMPLE_RAW_RESULT, "digest/email_triage", "test",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_synthesized_text(self, runner):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"response": "5 emails -- 1 needs attention."}

        with (
            patch("atlas_brain.autonomous.runner.autonomous_config") as cfg,
            patch("atlas_brain.services.llm_registry") as mock_reg,
        ):
            cfg.synthesis_enabled = True
            cfg.synthesis_max_tokens = 1024
            cfg.synthesis_temperature = 0.4
            mock_reg.get_active.return_value = mock_llm
            result = await runner._synthesize_with_skill(
                SAMPLE_RAW_RESULT, "digest/email_triage", "test",
            )

        assert result == "5 emails -- 1 needs attention."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_strips_think_tags_from_response(self, runner):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {
            "response": "<think>let me analyze</think>Here is the digest."
        }

        with (
            patch("atlas_brain.autonomous.runner.autonomous_config") as cfg,
            patch("atlas_brain.services.llm_registry") as mock_reg,
        ):
            cfg.synthesis_enabled = True
            cfg.synthesis_max_tokens = 1024
            cfg.synthesis_temperature = 0.4
            mock_reg.get_active.return_value = mock_llm
            result = await runner._synthesize_with_skill(
                SAMPLE_RAW_RESULT, "digest/email_triage", "test",
            )

        assert result == "Here is the digest."


# ---------------------------------------------------------------------------
# _run_builtin integration
# ---------------------------------------------------------------------------

class TestRunBuiltinSynthesis:
    """Test _run_builtin with and without synthesis_skill."""

    @pytest.fixture
    def runner(self):
        r = HeadlessRunner()
        # Register a fake builtin handler
        async def fake_handler(task):
            return SAMPLE_RAW_RESULT
        r.register_builtin("gmail_digest", fake_handler)
        return r

    @pytest.mark.asyncio
    async def test_without_synthesis_returns_str_result(self, runner):
        task = _builtin_task(handler="gmail_digest", synthesis_skill=None)
        result = await runner._run_builtin(task)
        assert result.success is True
        assert result.response_text == str(SAMPLE_RAW_RESULT)

    @pytest.mark.asyncio
    async def test_with_synthesis_returns_natural_language(self, runner):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"response": "You have 5 emails today."}

        task = _builtin_task(handler="gmail_digest", synthesis_skill="digest/email_triage")

        with (
            patch("atlas_brain.autonomous.runner.autonomous_config") as cfg,
            patch("atlas_brain.services.llm_registry") as mock_reg,
        ):
            cfg.synthesis_enabled = True
            cfg.synthesis_max_tokens = 1024
            cfg.synthesis_temperature = 0.4
            mock_reg.get_active.return_value = mock_llm
            result = await runner._run_builtin(task)

        assert result.success is True
        assert result.response_text == "You have 5 emails today."
        assert result.metadata["raw_result"] == SAMPLE_RAW_RESULT
        assert result.metadata["synthesis_skill"] == "digest/email_triage"

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_back_to_str(self, runner):
        """When synthesis fails (e.g., LLM unavailable), falls back to str(result)."""
        task = _builtin_task(handler="gmail_digest", synthesis_skill="digest/email_triage")

        with (
            patch("atlas_brain.autonomous.runner.autonomous_config") as cfg,
            patch("atlas_brain.services.llm_registry") as mock_reg,
        ):
            cfg.synthesis_enabled = True
            mock_reg.get_active.return_value = None
            result = await runner._run_builtin(task)

        assert result.success is True
        assert result.response_text == str(SAMPLE_RAW_RESULT)
