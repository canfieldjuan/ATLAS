"""
Tests for post-call transcription and data extraction pipeline.

Covers: recording download, ASR transcription, LLM extraction,
notification, config guards, error handling, and JSON parsing.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from atlas_brain.comms.call_intelligence import (
    _download_recording,
    _extract_call_data,
    _find_matching_brace,
    _notify_call_summary,
    _parse_extraction,
    _transcribe_audio,
    process_call_recording,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECORDING_URL = "https://test.signalwire.com/api/recordings/rec-123"
FAKE_WAV = b"RIFF" + b"\x00" * 200  # >100 bytes to pass auth loop content check


def _make_business_context():
    ctx = MagicMock()
    ctx.id = "test-biz"
    ctx.name = "Test Cleaning Co"
    ctx.business_type = "cleaning"
    ctx.services = ["deep cleaning", "move-out clean"]
    ctx.persona = "Friendly receptionist"
    ctx.voice_name = "Sarah"
    return ctx


def _mock_comms(project_id="", api_token="",
                account_sid="acct-test", recording_token="rec-tok"):
    """Create a mock comms_settings with all required SignalWire attributes."""
    mock = MagicMock()
    mock.signalwire_project_id = project_id
    mock.signalwire_api_token = api_token
    mock.signalwire_account_sid = account_sid
    mock.signalwire_recording_token = recording_token
    return mock


def _mock_httpx_client(response=None, side_effect=None):
    """Create a mock httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    if side_effect:
        mock_client.get.side_effect = side_effect
        mock_client.post.side_effect = side_effect
    else:
        mock_client.get.return_value = response
        mock_client.post.return_value = response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# Recording Download
# ---------------------------------------------------------------------------

class TestDownloadRecording:
    @pytest.mark.asyncio
    async def test_downloads_with_auth(self):
        mock_resp = MagicMock()
        mock_resp.content = FAKE_WAV
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        mock_comms = _mock_comms("proj-123", "token-456")

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch("atlas_brain.comms.call_intelligence.comms_settings", mock_comms, create=True), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            result = await _download_recording(RECORDING_URL)

        assert result == FAKE_WAV
        call_args = mock_client.get.call_args
        assert call_args.args[0].endswith(".wav")
        assert call_args.kwargs.get("auth") is not None

    @pytest.mark.asyncio
    async def test_appends_wav_extension(self):
        mock_resp = MagicMock()
        mock_resp.content = FAKE_WAV
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            await _download_recording("https://example.com/recording/123")

        url_called = mock_client.get.call_args.args[0]
        assert url_called == "https://example.com/recording/123.wav"

    @pytest.mark.asyncio
    async def test_skips_wav_if_already_present(self):
        mock_resp = MagicMock()
        mock_resp.content = FAKE_WAV
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            await _download_recording("https://example.com/recording/123.wav")

        url_called = mock_client.get.call_args.args[0]
        assert url_called == "https://example.com/recording/123.wav"

    @pytest.mark.asyncio
    async def test_download_failure_raises(self):
        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("refused"))

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            with pytest.raises(httpx.ConnectError):
                await _download_recording(RECORDING_URL)


# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------

class TestParseExtraction:
    def test_valid_json_with_actions(self):
        text = (
            '{"customer_name": "John", "intent": "estimate_request", '
            '"services_mentioned": ["deep cleaning"], "notes": "Referred by neighbor"}\n\n'
            '[{"type": "book_estimate", "label": "Book Estimate", "reason": "Customer requested"}]'
        )
        summary, data, actions = _parse_extraction(text, "fallback")
        assert data["customer_name"] == "John"
        assert data["intent"] == "estimate_request"
        assert len(actions) == 1
        assert actions[0]["type"] == "book_estimate"
        assert "John" in summary

    def test_json_only_no_actions(self):
        text = '{"customer_name": "Jane", "intent": "inquiry", "notes": "Asked about pricing"}'
        summary, data, actions = _parse_extraction(text, "fallback")
        assert data["customer_name"] == "Jane"
        assert actions == []

    def test_malformed_json_fallback(self):
        text = "This is not JSON at all"
        summary, data, actions = _parse_extraction(text, "fallback transcript")
        assert data == {}
        assert actions == []
        assert summary == "fallback transcript"

    def test_think_tags_stripped_before_parse(self):
        """Extraction handles text after think tag stripping."""
        text = '{"customer_name": "Bob", "intent": "booking", "notes": ""}'
        summary, data, actions = _parse_extraction(text, "fallback")
        assert data["customer_name"] == "Bob"


class TestFindMatchingBrace:
    def test_simple_object(self):
        assert _find_matching_brace('{"a": 1}', 0) == 7

    def test_nested_object(self):
        assert _find_matching_brace('{"a": {"b": 2}}', 0) == 14

    def test_string_with_braces(self):
        assert _find_matching_brace('{"a": "}{"}', 0) == 10

    def test_no_opening_brace(self):
        assert _find_matching_brace("no brace", 0) == -1

    def test_negative_start(self):
        assert _find_matching_brace("{}", -1) == -1


# ---------------------------------------------------------------------------
# Transcription (mocked)
# ---------------------------------------------------------------------------

class TestTranscription:
    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Hello, I need an estimate."}
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            result = await _transcribe_audio(b"fake-wav-data")

        assert result == "Hello, I need an estimate."
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "   "}
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            result = await _transcribe_audio(b"fake-wav-data")

        assert result is None

    @pytest.mark.asyncio
    async def test_asr_timeout_raises(self):
        mock_client = _mock_httpx_client(side_effect=httpx.TimeoutException("timeout"))

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.TimeoutException):
                await _transcribe_audio(b"fake-wav-data")


# ---------------------------------------------------------------------------
# LLM Extraction (mocked)
# ---------------------------------------------------------------------------

class TestExtraction:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        llm_response = {
            "response": json.dumps({
                "customer_name": "John Smith",
                "intent": "estimate_request",
                "services_mentioned": ["deep cleaning"],
                "notes": "Referred by neighbor",
            }) + "\n\n" + json.dumps([
                {"type": "book_estimate", "label": "Book Estimate", "reason": "Requested"}
            ])
        }

        mock_llm = MagicMock()
        mock_llm.chat.return_value = llm_response

        with patch("atlas_brain.services.llm_registry.get_active", return_value=mock_llm), \
             patch("atlas_brain.comms.call_intelligence.get_skill_registry") as mock_skills:
            mock_skill = MagicMock()
            mock_skill.content = "Extract data.\n\n{business_context}"
            mock_skills.return_value.get.return_value = mock_skill

            summary, data, actions = await _extract_call_data(
                "Hi, I need a deep cleaning estimate.",
                _make_business_context(),
            )

        assert data["customer_name"] == "John Smith"
        assert len(actions) == 1
        assert "John Smith" in summary

    @pytest.mark.asyncio
    async def test_no_llm_returns_truncated_transcript(self):
        with patch("atlas_brain.services.llm_registry.get_active", return_value=None):
            summary, data, actions = await _extract_call_data("Some transcript", None)

        assert summary == "Some transcript"
        assert data == {}
        assert actions == []

    @pytest.mark.asyncio
    async def test_think_tag_stripping(self):
        llm_response = {
            "response": '<think>Let me analyze...</think>{"customer_name": "Jane", "intent": "inquiry", "notes": ""}'
        }

        mock_llm = MagicMock()
        mock_llm.chat.return_value = llm_response

        with patch("atlas_brain.services.llm_registry.get_active", return_value=mock_llm), \
             patch("atlas_brain.comms.call_intelligence.get_skill_registry") as mock_skills:
            mock_skills.return_value.get.return_value = None

            summary, data, actions = await _extract_call_data("transcript", None)

        assert data["customer_name"] == "Jane"


# ---------------------------------------------------------------------------
# Notification (mocked)
# ---------------------------------------------------------------------------

class TestNotification:
    @pytest.mark.asyncio
    async def test_sends_ntfy(self):
        repo = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            mock_settings.call_intelligence.notify_enabled = True
            mock_settings.alerts.ntfy_enabled = True
            mock_settings.alerts.ntfy_url = "https://ntfy.example.com"
            mock_settings.alerts.ntfy_topic = "test"

            tid = uuid4()
            await _notify_call_summary(
                repo, tid, "call-123",
                "+16185551234", 222,
                "Customer: John. Intent: estimate",
                {"customer_name": "John", "intent": "estimate_request",
                 "services_mentioned": ["deep cleaning"]},
                [{"type": "book_estimate", "label": "Book Estimate", "reason": "Requested"}],
                _make_business_context(),
            )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "Test Cleaning Co" in call_args.kwargs.get("headers", {}).get("Title", "")
        repo.mark_notified.assert_awaited_once_with(tid)

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        repo = AsyncMock()

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.notify_enabled = False

            await _notify_call_summary(
                repo, uuid4(), "call-123",
                "+1234", 60, "summary", {}, [], None,
            )

        repo.mark_notified.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_ntfy_disabled(self):
        repo = AsyncMock()

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.notify_enabled = True
            mock_settings.alerts.ntfy_enabled = False

            await _notify_call_summary(
                repo, uuid4(), "call-123",
                "+1234", 60, "summary", {}, [], None,
            )

        repo.mark_notified.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_formats_duration(self):
        repo = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = _mock_httpx_client(response=mock_resp)

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            mock_settings.call_intelligence.notify_enabled = True
            mock_settings.alerts.ntfy_enabled = True
            mock_settings.alerts.ntfy_url = "https://ntfy.example.com"
            mock_settings.alerts.ntfy_topic = "test"

            await _notify_call_summary(
                repo, uuid4(), "call-123",
                "+1234", 185,
                "summary", {}, [], None,
            )

        body = mock_client.post.call_args.kwargs.get("content", "")
        assert "3m 5s" in body


# ---------------------------------------------------------------------------
# Full Pipeline (mocked)
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.mark.asyncio
    async def test_disabled_config_early_return(self):
        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.enabled = False

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )
            # Should return without creating any DB records

    @pytest.mark.asyncio
    async def test_short_call_skipped(self):
        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 10

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 5,
            )
            # Should return without creating any DB records (duration < min)

    @pytest.mark.asyncio
    async def test_end_to_end_success(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        # Download response
        download_resp = MagicMock()
        download_resp.content = b"fake-wav-audio"
        download_resp.raise_for_status = MagicMock()

        # ASR response
        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": "Hi, I need a cleaning estimate."}
        asr_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = download_resp
        mock_client.post.return_value = asr_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        llm_response = {
            "response": json.dumps({
                "customer_name": "John",
                "intent": "estimate_request",
                "services_mentioned": ["cleaning"],
                "notes": "",
            }) + "\n\n" + json.dumps([
                {"type": "book_estimate", "label": "Book Estimate", "reason": "Asked"}
            ])
        }
        mock_llm = MagicMock()
        mock_llm.chat.return_value = llm_response

        mock_comms = _mock_comms("proj", "tok")

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch("atlas_brain.services.llm_registry.get_active", return_value=mock_llm), \
             patch("atlas_brain.comms.call_intelligence.get_skill_registry") as mock_skills, \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.llm_max_tokens = 1024
            mock_settings.call_intelligence.llm_temperature = 0.3
            mock_settings.call_intelligence.llm_timeout = 30.0
            mock_settings.call_intelligence.notify_enabled = False
            mock_settings.alerts.ntfy_enabled = False
            mock_skills.return_value.get.return_value = None

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )

        mock_repo.create.assert_awaited_once()
        mock_repo.update_transcript.assert_awaited_once()
        mock_repo.update_extraction.assert_awaited_once()
        # Status: transcribing, extracting, ready
        assert mock_repo.update_status.await_count >= 2

    @pytest.mark.asyncio
    async def test_download_failure_stops_gracefully(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        mock_client = _mock_httpx_client(side_effect=httpx.ConnectError("refused"))

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_timeout = 60

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )

        # Should have set error status
        status_calls = [
            (c.args[1] if len(c.args) > 1 else c.kwargs.get("status"))
            for c in mock_repo.update_status.await_args_list
        ]
        assert "error" in status_calls
        # Should NOT have called transcription or extraction
        mock_repo.update_transcript.assert_not_awaited()
        mock_repo.update_extraction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_asr_failure_stops_gracefully(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        # Download succeeds, ASR fails
        download_resp = MagicMock()
        download_resp.content = b"wav-data"
        download_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = download_resp
        mock_client.post.side_effect = httpx.ConnectError("ASR down")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_timeout = 60

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )

        status_calls = [
            c.args[1] if len(c.args) > 1 else c.kwargs.get("status")
            for c in mock_repo.update_status.await_args_list
        ]
        assert "error" in status_calls
        mock_repo.update_extraction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_extraction(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        # Download succeeds
        download_resp = MagicMock()
        download_resp.content = b"wav-data"
        download_resp.raise_for_status = MagicMock()

        # ASR returns empty
        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": ""}
        asr_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = download_resp
        mock_client.post.return_value = asr_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_comms = _mock_comms()

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.notify_enabled = False
            mock_settings.alerts.ntfy_enabled = False

            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )

        mock_repo.update_transcript.assert_awaited_once()
        # "No speech detected" stored
        extraction_call = mock_repo.update_extraction.call_args
        assert extraction_call.kwargs.get("summary") == "No speech detected"
        # Status should be "ready"
        status_calls = [
            c.args[1] if len(c.args) > 1 else c.kwargs.get("status")
            for c in mock_repo.update_status.call_args_list
        ]
        assert "ready" in status_calls

    @pytest.mark.asyncio
    async def test_db_create_failure_returns_early(self):
        mock_repo = AsyncMock()
        mock_repo.create.side_effect = Exception("DB down")

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5

            # Should not raise
            await process_call_recording(
                "call-123", RECORDING_URL, "+1234", "+5678", "ctx", 30,
            )

        mock_repo.update_status.assert_not_awaited()
