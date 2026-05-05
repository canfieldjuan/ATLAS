"""Verify email_graph_sync stage 1 routes through llm_router (not direct Ollama).

This guards against regressions where the email-to-graph pipeline gets
hardcoded to a specific local LLM, which silently breaks when the
production setup uses cloud LLMs (OpenRouter, Anthropic, etc.).
"""

from unittest.mock import MagicMock, patch

import pytest


def test_get_llm_uses_router_email_triage_workflow():
    """EmailGraphSync._get_llm must call llm_router.get_llm('email_triage')."""
    from atlas_brain.jobs.email_graph_sync import EmailGraphSync

    fake_llm = MagicMock(name="triage_llm")
    fake_llm.model_name = "anthropic/claude-haiku-4-5"

    with patch("atlas_brain.services.llm_router.get_llm", return_value=fake_llm) as mock_router:
        job = EmailGraphSync()
        result = job._get_llm()

    assert result is fake_llm
    mock_router.assert_called_once_with("email_triage")


def test_get_llm_raises_when_router_returns_none():
    """If the router has no LLM configured for email_triage and no local
    fallback, raise a clear error instead of silently returning None.
    """
    from atlas_brain.jobs.email_graph_sync import EmailGraphSync

    with patch("atlas_brain.services.llm_router.get_llm", return_value=None):
        job = EmailGraphSync()
        with pytest.raises(RuntimeError, match="No LLM available"):
            job._get_llm()


def test_get_llm_caches_for_subsequent_calls():
    """The router lookup is cached on the instance for the run."""
    from atlas_brain.jobs.email_graph_sync import EmailGraphSync

    fake_llm = MagicMock(name="triage_llm")

    with patch("atlas_brain.services.llm_router.get_llm", return_value=fake_llm) as mock_router:
        job = EmailGraphSync()
        job._get_llm()
        job._get_llm()
        job._get_llm()

    assert mock_router.call_count == 1
