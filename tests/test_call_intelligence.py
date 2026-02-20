"""
Tests for post-call transcription and data extraction pipeline.

Covers: audio conversion, ASR transcription, LLM extraction,
notification, config guards, error handling, and repository ops.
"""

import asyncio
import audioop
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from atlas_brain.comms.call_intelligence import (
    _convert_audio_to_wav,
    _extract_call_data,
    _find_matching_brace,
    _notify_call_summary,
    _parse_extraction,
    process_call_recording,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mulaw_chunks(duration_seconds: float = 2.0, num_chunks: int = 10) -> list[bytes]:
    """Generate synthetic mulaw audio chunks (silence)."""
    total_bytes = int(8000 * duration_seconds)
    chunk_size = total_bytes // num_chunks
    # mulaw silence is 0xFF
    return [b"\xff" * chunk_size for _ in range(num_chunks)]


def _make_business_context():
    ctx = MagicMock()
    ctx.id = "test-biz"
    ctx.name = "Test Cleaning Co"
    ctx.business_type = "cleaning"
    ctx.services = ["deep cleaning", "move-out clean"]
    ctx.persona = "Friendly receptionist"
    ctx.voice_name = "Sarah"
    return ctx


# ---------------------------------------------------------------------------
# Audio Conversion
# ---------------------------------------------------------------------------

class TestAudioConversion:
    def test_valid_wav_output(self):
        chunks = _make_mulaw_chunks(duration_seconds=1.0)
        wav = _convert_audio_to_wav(chunks)

        # Check RIFF header
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "

        # Parse WAV header
        fmt_size = struct.unpack_from("<I", wav, 16)[0]
        assert fmt_size == 16
        audio_fmt = struct.unpack_from("<H", wav, 20)[0]
        assert audio_fmt == 1  # PCM
        channels = struct.unpack_from("<H", wav, 22)[0]
        assert channels == 1
        sample_rate = struct.unpack_from("<I", wav, 24)[0]
        assert sample_rate == 16000
        bits = struct.unpack_from("<H", wav, 34)[0]
        assert bits == 16

    def test_resamples_to_16khz(self):
        chunks = _make_mulaw_chunks(duration_seconds=1.0)
        wav = _convert_audio_to_wav(chunks)

        # Data chunk starts at byte 44
        data_size = struct.unpack_from("<I", wav, 40)[0]
        # 1 second at 16kHz, 16-bit, mono ~ 32000 bytes (ratecv may be off by a few)
        assert abs(data_size - 32000) < 100

    def test_empty_audio_raises(self):
        with pytest.raises(ValueError, match="No audio data"):
            _convert_audio_to_wav([])

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="No audio data"):
            _convert_audio_to_wav([b""])


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
        # _extract_call_data strips think tags, _parse_extraction gets clean text
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
        from atlas_brain.comms.call_intelligence import _transcribe_wav

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "Hello, I need an estimate."}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            result = await _transcribe_wav(b"fake-wav-data")

        assert result == "Hello, I need an estimate."
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_none(self):
        from atlas_brain.comms.call_intelligence import _transcribe_wav

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"text": "   "}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            result = await _transcribe_wav(b"fake-wav-data")

        assert result is None

    @pytest.mark.asyncio
    async def test_asr_timeout_raises(self):
        from atlas_brain.comms.call_intelligence import _transcribe_wav

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.TimeoutException):
                await _transcribe_wav(b"fake-wav-data")


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

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

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


# ---------------------------------------------------------------------------
# Full Pipeline (mocked)
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.mark.asyncio
    async def test_disabled_config_early_return(self):
        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.enabled = False

            await process_call_recording(
                "call-123", _make_mulaw_chunks(), "+1234", "+5678", "ctx", 30,
            )
            # Should return without doing anything

    @pytest.mark.asyncio
    async def test_short_call_skipped(self):
        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings:
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 10

            await process_call_recording(
                "call-123", _make_mulaw_chunks(), "+1234", "+5678", "ctx", 5,
            )
            # Should return without doing anything (duration < min)

    @pytest.mark.asyncio
    async def test_end_to_end_success(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": "Hi, I need a cleaning estimate."}
        asr_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
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

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch("atlas_brain.services.llm_registry.get_active", return_value=mock_llm), \
             patch("atlas_brain.comms.call_intelligence.get_skill_registry") as mock_skills:
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.llm_max_tokens = 1024
            mock_settings.call_intelligence.llm_temperature = 0.3
            mock_settings.call_intelligence.notify_enabled = False
            mock_settings.alerts.ntfy_enabled = False
            mock_skills.return_value.get.return_value = None

            await process_call_recording(
                "call-123", _make_mulaw_chunks(2.0), "+1234", "+5678", "ctx", 30,
            )

        mock_repo.create.assert_awaited_once()
        mock_repo.update_transcript.assert_awaited_once()
        mock_repo.update_extraction.assert_awaited_once()
        # Status should have been updated through the pipeline
        assert mock_repo.update_status.await_count >= 2

    @pytest.mark.asyncio
    async def test_asr_failure_stops_gracefully(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60

            await process_call_recording(
                "call-123", _make_mulaw_chunks(2.0), "+1234", "+5678", "ctx", 30,
            )

        # Should have set error status
        status_calls = [
            (c.args[1] if len(c.args) > 1 else c.kwargs.get("status"))
            for c in mock_repo.update_status.await_args_list
        ]
        assert "error" in status_calls
        # Should NOT have called extraction
        mock_repo.update_extraction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_transcript_skips_extraction(self):
        mock_repo = AsyncMock()
        mock_repo.create.return_value = {"id": uuid4(), "call_sid": "call-123"}

        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": ""}
        asr_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = asr_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=mock_repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.notify_enabled = False
            mock_settings.alerts.ntfy_enabled = False

            await process_call_recording(
                "call-123", _make_mulaw_chunks(2.0), "+1234", "+5678", "ctx", 30,
            )

        mock_repo.update_transcript.assert_awaited_once()
        # Should have stored "No speech detected" summary
        extraction_call = mock_repo.update_extraction.call_args
        assert extraction_call.kwargs.get("summary") == "No speech detected"
        # Status should be "ready", not "error"
        status_calls = [
            c.args[1] if len(c.args) > 1 else c.kwargs.get("status")
            for c in mock_repo.update_status.call_args_list
        ]
        assert "ready" in status_calls
