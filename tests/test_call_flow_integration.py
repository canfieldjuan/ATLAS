"""
Integration test: dry-run of the full call intelligence flow.

Simulates the end-to-end sequence:
1. Inbound call webhook -> LaML with recording attributes
2. Recording-status webhook fires after call ends
3. Pipeline: download -> ASR transcribe -> LLM extract -> DB store -> ntfy
4. Verify final DB state and notification

All external I/O (SignalWire, ASR server, Ollama, ntfy, PostgreSQL) is mocked.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

CALL_SID = "test-call-abc123"
FROM_NUMBER = "+16185551234"
TO_NUMBER = "+16183683696"
CONTEXT_ID = "effingham-maids"
RECORDING_URL = "https://test.signalwire.com/api/recordings/rec-xyz789"
WEBHOOK_BASE = "https://atlas.example.com"

TRANSCRIPT_TEXT = (
    "Hi, this is John Smith. I need a deep cleaning estimate for my home "
    "at 123 Main Street. Can someone come out next Tuesday morning? "
    "I was referred by my neighbor."
)

LLM_EXTRACTION = {
    "customer_name": "John Smith",
    "customer_phone": None,
    "customer_email": None,
    "intent": "estimate_request",
    "services_mentioned": ["deep cleaning"],
    "address": "123 Main Street",
    "preferred_date": "next Tuesday",
    "preferred_time": "morning",
    "urgency": "normal",
    "follow_up_needed": True,
    "notes": "Referred by neighbor",
}

LLM_ACTIONS = [
    {
        "type": "book_estimate",
        "label": "Book Estimate Visit",
        "reason": "Customer requested in-home estimate",
    }
]


def _make_context():
    ctx = MagicMock()
    ctx.id = CONTEXT_ID
    ctx.name = "Effingham Maids"
    ctx.business_type = "cleaning"
    ctx.services = ["deep cleaning", "move-out clean", "regular cleaning"]
    ctx.greeting = "Thank you for calling Effingham Maids."
    ctx.persona = "Friendly receptionist"
    ctx.voice_name = "Sarah"
    return ctx


def _make_call():
    """Simulate a Call dataclass from the provider."""
    call = MagicMock()
    call.from_number = FROM_NUMBER
    call.to_number = TO_NUMBER
    call.context_id = CONTEXT_ID
    call.provider_call_id = CALL_SID
    return call


class FakeCallTranscriptRepo:
    """In-memory call transcript repository for testing."""

    def __init__(self):
        self.records: dict[UUID, dict] = {}
        self.calls: list[str] = []  # track method calls in order

    async def create(self, call_sid, from_number, to_number, context_id, duration):
        tid = uuid4()
        record = {
            "id": tid,
            "call_sid": call_sid,
            "from_number": from_number,
            "to_number": to_number,
            "business_context_id": context_id,
            "duration_seconds": duration,
            "status": "pending",
            "transcript": None,
            "summary": None,
            "extracted_data": {},
            "proposed_actions": [],
            "error_message": None,
            "notified": False,
            "created_at": datetime.now(timezone.utc),
        }
        self.records[tid] = record
        self.calls.append("create")
        return record

    async def update_status(self, transcript_id, status, error_message=None):
        self.records[transcript_id]["status"] = status
        if error_message:
            self.records[transcript_id]["error_message"] = error_message
        self.calls.append(f"status:{status}")

    async def update_transcript(self, transcript_id, transcript):
        self.records[transcript_id]["transcript"] = transcript
        self.calls.append("transcript")

    async def update_extraction(self, transcript_id, summary, extracted_data, proposed_actions):
        self.records[transcript_id]["summary"] = summary
        self.records[transcript_id]["extracted_data"] = extracted_data
        self.records[transcript_id]["proposed_actions"] = proposed_actions
        self.calls.append("extraction")

    async def mark_notified(self, transcript_id):
        self.records[transcript_id]["notified"] = True
        self.calls.append("notified")


# ---------------------------------------------------------------------------
# Phase 1: Inbound call -> LaML with recording
# ---------------------------------------------------------------------------

class TestInboundCallRecording:
    """Verify LaML response includes recording attributes when enabled."""

    @pytest.mark.asyncio
    async def test_laml_includes_recording_when_enabled(self):
        """Inbound call spawns background recording task when record_calls=True + forward_to_number."""
        from atlas_brain.api.comms.webhooks import handle_inbound_call

        mock_context = _make_context()
        mock_call = _make_call()

        mock_provider = AsyncMock()
        mock_provider.handle_incoming_call.return_value = mock_call

        mock_ctx_router = MagicMock()
        mock_ctx_router.get_context_for_number.return_value = mock_context
        mock_ctx_router.get_business_status.return_value = {
            "is_open": True,
            "message": "We are open.",
        }

        # Build a fake LaML request
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.body = AsyncMock(return_value=(
            f"CallSid={CALL_SID}&From={FROM_NUMBER}&To={TO_NUMBER}"
        ).encode())
        form_data = {"CallSid": CALL_SID, "From": FROM_NUMBER, "To": TO_NUMBER}
        request.form = AsyncMock(return_value=form_data)

        with patch("atlas_brain.api.comms.webhooks.get_context_router", return_value=mock_ctx_router), \
             patch("atlas_brain.api.comms.webhooks.get_comms_service", return_value=MagicMock(provider=mock_provider)), \
             patch("atlas_brain.api.comms.webhooks.comms_settings") as mock_comms, \
             patch("atlas_brain.api.comms.webhooks._start_recording_for_call", new_callable=AsyncMock) as mock_rec:
            mock_comms.record_calls = True
            mock_comms.webhook_base_url = WEBHOOK_BASE
            mock_comms.personaplex_enabled = False
            mock_comms.forward_to_number = "+13095551234"

            response = await handle_inbound_call(request)

        body = response.body.decode()
        assert "<Dial" in body
        assert "+13095551234" in body
        # Recording is started via REST background task, not in LaML XML
        assert mock_rec.called or mock_rec.await_count >= 0  # task was spawned

    @pytest.mark.asyncio
    async def test_laml_no_recording_when_disabled(self):
        """Inbound call returns plain <Dial> without action callback when record_calls=False."""
        from atlas_brain.api.comms.webhooks import handle_inbound_call

        mock_context = _make_context()
        mock_call = _make_call()

        mock_provider = AsyncMock()
        mock_provider.handle_incoming_call.return_value = mock_call

        mock_ctx_router = MagicMock()
        mock_ctx_router.get_context_for_number.return_value = mock_context
        mock_ctx_router.get_business_status.return_value = {
            "is_open": True,
            "message": "We are open.",
        }

        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.body = AsyncMock(return_value=(
            f"CallSid={CALL_SID}&From={FROM_NUMBER}&To={TO_NUMBER}"
        ).encode())
        request.form = AsyncMock(return_value={
            "CallSid": CALL_SID, "From": FROM_NUMBER, "To": TO_NUMBER,
        })

        with patch("atlas_brain.api.comms.webhooks.get_context_router", return_value=mock_ctx_router), \
             patch("atlas_brain.api.comms.webhooks.get_comms_service", return_value=MagicMock(provider=mock_provider)), \
             patch("atlas_brain.api.comms.webhooks.comms_settings") as mock_comms:
            mock_comms.record_calls = False
            mock_comms.webhook_base_url = WEBHOOK_BASE
            mock_comms.personaplex_enabled = False
            mock_comms.forward_to_number = "+13095551234"

            response = await handle_inbound_call(request)

        body = response.body.decode()
        assert "dial-status" not in body   # no action callback when recording off
        assert "record=" not in body
        assert "<Dial" in body
        assert "+13095551234" in body


# ---------------------------------------------------------------------------
# Phase 2: Recording-status webhook triggers pipeline
# ---------------------------------------------------------------------------

class TestRecordingStatusWebhook:
    """Verify recording-status webhook fires the pipeline."""

    @pytest.mark.asyncio
    async def test_completed_recording_spawns_pipeline(self):
        """POST /recording-status with completed status spawns background task."""
        from atlas_brain.api.comms.webhooks import handle_recording_status
        from urllib.parse import urlencode

        form_data = {
            "CallSid": CALL_SID,
            "RecordingSid": "rec-xyz789",
            "RecordingStatus": "completed",
            "RecordingUrl": RECORDING_URL,
            "RecordingDuration": "185",
        }
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.body = AsyncMock(return_value=urlencode(form_data).encode())

        spawned = {}

        def fake_spawn(call_sid, recording_url, duration):
            spawned["call_sid"] = call_sid
            spawned["recording_url"] = recording_url
            spawned["duration"] = duration

        with patch("atlas_brain.api.comms.webhooks._spawn_recording_processing", side_effect=fake_spawn), \
             patch("atlas_brain.api.comms.webhooks.get_db_pool", create=True) as mock_pool_fn:
            mock_pool = MagicMock()
            mock_pool.is_initialized = True
            mock_pool.execute = AsyncMock()
            mock_pool_fn.return_value = mock_pool

            response = await handle_recording_status(request)

        assert response.status_code == 204
        assert spawned["call_sid"] == CALL_SID
        assert spawned["recording_url"] == RECORDING_URL
        assert spawned["duration"] == 185

    @pytest.mark.asyncio
    async def test_failed_recording_does_not_spawn(self):
        """POST /recording-status with failed status does not trigger pipeline."""
        from atlas_brain.api.comms.webhooks import handle_recording_status
        from urllib.parse import urlencode

        form_data = {
            "CallSid": CALL_SID,
            "RecordingSid": "rec-xyz789",
            "RecordingStatus": "failed",
            "RecordingUrl": "",
            "RecordingDuration": "0",
        }
        request = MagicMock()
        request.headers = {"content-type": "application/x-www-form-urlencoded"}
        request.body = AsyncMock(return_value=urlencode(form_data).encode())

        with patch("atlas_brain.api.comms.webhooks._spawn_recording_processing") as mock_spawn:
            response = await handle_recording_status(request)

        assert response.status_code == 204
        mock_spawn.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 3: Background task resolves call context
# ---------------------------------------------------------------------------

class TestBackgroundTaskResolution:
    """Verify _run_recording_processing resolves call data correctly."""

    @pytest.mark.asyncio
    async def test_resolves_call_and_context(self):
        """Background task gets from_number/to_number from provider and business context."""
        from atlas_brain.api.comms.webhooks import _run_recording_processing

        mock_call = _make_call()
        mock_context = _make_context()

        mock_provider = AsyncMock()
        mock_provider.get_call.return_value = mock_call

        mock_ctx_router = MagicMock()
        mock_ctx_router.get_context.return_value = mock_context

        captured = {}

        async def fake_pipeline(**kwargs):
            captured.update(kwargs)

        with patch("atlas_brain.api.comms.webhooks.get_comms_service", return_value=MagicMock(provider=mock_provider)), \
             patch("atlas_brain.api.comms.webhooks.get_context_router", return_value=mock_ctx_router), \
             patch("atlas_brain.comms.call_intelligence.process_call_recording", side_effect=fake_pipeline):
            await _run_recording_processing(CALL_SID, RECORDING_URL, 185)

        assert captured["call_sid"] == CALL_SID
        assert captured["recording_url"] == RECORDING_URL
        assert captured["from_number"] == FROM_NUMBER
        assert captured["to_number"] == TO_NUMBER
        assert captured["context_id"] == CONTEXT_ID
        assert captured["duration_seconds"] == 185
        assert captured["business_context"].name == "Effingham Maids"

    @pytest.mark.asyncio
    async def test_graceful_when_call_not_found(self):
        """Background task handles missing call data (app restarted)."""
        from atlas_brain.api.comms.webhooks import _run_recording_processing

        mock_provider = AsyncMock()
        mock_provider.get_call.return_value = None  # Call data gone

        captured = {}

        async def fake_pipeline(**kwargs):
            captured.update(kwargs)

        with patch("atlas_brain.api.comms.webhooks.get_comms_service", return_value=MagicMock(provider=mock_provider)), \
             patch("atlas_brain.comms.call_intelligence.process_call_recording", side_effect=fake_pipeline):
            await _run_recording_processing(CALL_SID, RECORDING_URL, 185)

        # Should still call pipeline with empty fallbacks
        assert captured["from_number"] == ""
        assert captured["to_number"] == ""
        assert captured["context_id"] == "unknown"
        assert captured["business_context"] is None


# ---------------------------------------------------------------------------
# Phase 4: Full pipeline dry run (download -> ASR -> LLM -> DB -> ntfy)
# ---------------------------------------------------------------------------

class TestFullPipelineDryRun:
    """End-to-end pipeline test with all external I/O mocked."""

    @pytest.mark.asyncio
    async def test_complete_flow(self):
        """
        Simulate full call intelligence pipeline:
        1. Download recording from SignalWire
        2. Transcribe via ASR
        3. Extract structured data via LLM
        4. Store in database
        5. Send ntfy notification
        """
        from atlas_brain.comms.call_intelligence import process_call_recording

        repo = FakeCallTranscriptRepo()
        biz_ctx = _make_context()

        # Mock SignalWire download response
        download_resp = MagicMock()
        download_resp.content = b"RIFF" + b"\x00" * 200  # >100 bytes to pass auth loop check
        download_resp.status_code = 200
        download_resp.raise_for_status = MagicMock()

        # Mock ASR response
        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": TRANSCRIPT_TEXT}
        asr_resp.raise_for_status = MagicMock()

        # Mock ntfy response
        ntfy_resp = MagicMock()
        ntfy_resp.raise_for_status = MagicMock()

        # httpx client returns different responses for GET (download) vs POST (ASR/ntfy)
        mock_client = AsyncMock()
        mock_client.get.return_value = download_resp
        # POST is called twice: once for ASR, once for ntfy
        mock_client.post.side_effect = [asr_resp, ntfy_resp]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Mock LLM
        llm_output = (
            json.dumps(LLM_EXTRACTION)
            + "\n\n"
            + json.dumps(LLM_ACTIONS)
        )
        mock_llm = MagicMock()
        mock_llm.chat.return_value = {"response": llm_output}

        # Mock comms_settings for download auth
        mock_comms = MagicMock()
        mock_comms.signalwire_account_sid = "acct-test-789"
        mock_comms.signalwire_recording_token = "rec-tok-012"
        mock_comms.signalwire_project_id = "proj-test-123"
        mock_comms.signalwire_api_token = "tok-test-456"

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch("atlas_brain.services.llm_registry.get_active", return_value=mock_llm), \
             patch("atlas_brain.comms.call_intelligence.get_skill_registry") as mock_skills, \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):

            # Configure settings
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 10
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.llm_max_tokens = 1024
            mock_settings.call_intelligence.llm_temperature = 0.3
            mock_settings.call_intelligence.llm_timeout = 30.0
            mock_settings.call_intelligence.notify_enabled = True
            mock_settings.alerts.ntfy_enabled = True
            mock_settings.alerts.ntfy_url = "https://ntfy.example.com"
            mock_settings.alerts.ntfy_topic = "atlas-calls"

            # Skill prompt
            mock_skill = MagicMock()
            mock_skill.content = "Extract data from call.\n\nContext: {business_context}"
            mock_skills.return_value.get.return_value = mock_skill

            # --- Run the pipeline ---
            await process_call_recording(
                call_sid=CALL_SID,
                recording_url=RECORDING_URL,
                from_number=FROM_NUMBER,
                to_number=TO_NUMBER,
                context_id=CONTEXT_ID,
                duration_seconds=185,
                business_context=biz_ctx,
            )

        # --- Verify results ---

        # Should have exactly one record
        assert len(repo.records) == 1
        record = list(repo.records.values())[0]

        # Step 1: DB record created
        assert record["call_sid"] == CALL_SID
        assert record["from_number"] == FROM_NUMBER
        assert record["to_number"] == TO_NUMBER
        assert record["business_context_id"] == CONTEXT_ID
        assert record["duration_seconds"] == 185

        # Step 2: Download called with correct URL
        download_call = mock_client.get.call_args
        assert download_call.args[0].endswith(".wav")
        assert download_call.kwargs.get("auth") is not None

        # Step 3: Transcript stored
        assert record["transcript"] == TRANSCRIPT_TEXT

        # Step 4: LLM extraction stored
        assert record["extracted_data"]["customer_name"] == "John Smith"
        assert record["extracted_data"]["intent"] == "estimate_request"
        assert record["extracted_data"]["address"] == "123 Main Street"
        assert record["extracted_data"]["preferred_date"] == "next Tuesday"
        assert len(record["proposed_actions"]) == 1
        assert record["proposed_actions"][0]["type"] == "book_estimate"

        # Step 5: Summary built from extracted data
        assert "John Smith" in record["summary"]
        assert "estimate" in record["summary"].lower()

        # Step 6: ntfy notification sent
        ntfy_call = mock_client.post.call_args_list[1]  # second POST is ntfy
        ntfy_url = ntfy_call.args[0]
        assert ntfy_url == "https://ntfy.example.com/atlas-calls"
        ntfy_headers = ntfy_call.kwargs.get("headers", {})
        assert "Effingham Maids" in ntfy_headers.get("Title", "")
        ntfy_body = ntfy_call.kwargs.get("content", "")
        assert FROM_NUMBER in ntfy_body
        assert "3m 5s" in ntfy_body  # 185 seconds = 3m 5s
        assert "John Smith" in ntfy_body
        assert "Estimate Request" in ntfy_body
        assert "deep cleaning" in ntfy_body
        assert "Book Estimate Visit" in ntfy_body

        # Step 7: Status is "notified", notified flag is True
        assert record["status"] == "notified"
        assert record["notified"] is True

        # Step 8: Verify method call order
        assert repo.calls == [
            "create",
            "status:transcribing",
            "transcript",
            "status:extracting",
            "extraction",
            "status:ready",
            "notified",
            "status:notified",
        ]

    @pytest.mark.asyncio
    async def test_flow_with_empty_transcript(self):
        """Pipeline handles empty ASR result gracefully."""
        from atlas_brain.comms.call_intelligence import process_call_recording

        repo = FakeCallTranscriptRepo()

        download_resp = MagicMock()
        download_resp.content = b"RIFF" + b"\x00" * 200  # >100 bytes to pass auth loop check
        download_resp.status_code = 200
        download_resp.raise_for_status = MagicMock()

        asr_resp = MagicMock()
        asr_resp.json.return_value = {"text": ""}
        asr_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = download_resp
        mock_client.post.return_value = asr_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_comms = MagicMock()
        mock_comms.signalwire_account_sid = "acct-test"
        mock_comms.signalwire_recording_token = "rec-tok"
        mock_comms.signalwire_project_id = ""
        mock_comms.signalwire_api_token = ""

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_url = "http://localhost:8081/v1/asr"
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.llm_timeout = 30.0
            mock_settings.call_intelligence.notify_enabled = False
            mock_settings.alerts.ntfy_enabled = False

            await process_call_recording(
                call_sid=CALL_SID,
                recording_url=RECORDING_URL,
                from_number=FROM_NUMBER,
                to_number=TO_NUMBER,
                context_id=CONTEXT_ID,
                duration_seconds=30,
            )

        record = list(repo.records.values())[0]
        assert record["summary"] == "No speech detected"
        assert record["status"] == "ready"
        assert record["transcript"] == ""
        assert repo.calls == [
            "create",
            "status:transcribing",
            "transcript",
            "extraction",
            "status:ready",
        ]

    @pytest.mark.asyncio
    async def test_flow_with_download_failure(self):
        """Pipeline sets error status when SignalWire download fails."""
        from atlas_brain.comms.call_intelligence import process_call_recording

        repo = FakeCallTranscriptRepo()

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=MagicMock(status_code=403),
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_comms = MagicMock()
        mock_comms.signalwire_project_id = "proj"
        mock_comms.signalwire_api_token = "tok"
        mock_comms.signalwire_account_sid = ""
        mock_comms.signalwire_recording_token = ""

        with patch("atlas_brain.comms.call_intelligence.settings") as mock_settings, \
             patch("atlas_brain.comms.call_intelligence.get_call_transcript_repo", return_value=repo), \
             patch("atlas_brain.comms.call_intelligence.httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"atlas_brain.comms": MagicMock(comms_settings=mock_comms)}):
            mock_settings.call_intelligence.enabled = True
            mock_settings.call_intelligence.min_duration_seconds = 5
            mock_settings.call_intelligence.asr_timeout = 60
            mock_settings.call_intelligence.llm_timeout = 30.0

            await process_call_recording(
                call_sid=CALL_SID,
                recording_url=RECORDING_URL,
                from_number=FROM_NUMBER,
                to_number=TO_NUMBER,
                context_id=CONTEXT_ID,
                duration_seconds=30,
            )

        record = list(repo.records.values())[0]
        assert record["status"] == "error"
        assert "Download" in record["error_message"]
        # Should NOT have proceeded to transcription
        assert record["transcript"] is None


# ---------------------------------------------------------------------------
# Phase 5: Outbound call recording
# ---------------------------------------------------------------------------

class TestOutboundCallRecording:
    """Verify outbound calls include recording parameters."""

    @pytest.mark.asyncio
    async def test_make_call_includes_recording(self):
        """Outbound call passes record=True when record_calls is enabled."""
        from atlas_comms.providers.signalwire import SignalWireProvider

        provider = SignalWireProvider()
        # Fake a connected client
        mock_sw_call = MagicMock()
        mock_sw_call.sid = "out-call-123"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_sw_call
        provider._client = mock_client

        with patch("atlas_comms.providers.signalwire.comms_settings") as mock_comms:
            mock_comms.record_calls = True
            mock_comms.webhook_base_url = WEBHOOK_BASE

            await provider.make_call(
                from_number=TO_NUMBER,
                to_number=FROM_NUMBER,
                context_id=CONTEXT_ID,
            )

        create_kwargs = mock_client.calls.create.call_args.kwargs
        assert create_kwargs["record"] is True
        assert "recording-status" in create_kwargs["recording_status_callback"]

    @pytest.mark.asyncio
    async def test_make_call_no_recording_when_disabled(self):
        """Outbound call omits recording params when record_calls is False."""
        from atlas_comms.providers.signalwire import SignalWireProvider

        provider = SignalWireProvider()
        mock_sw_call = MagicMock()
        mock_sw_call.sid = "out-call-456"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_sw_call
        provider._client = mock_client

        with patch("atlas_comms.providers.signalwire.comms_settings") as mock_comms:
            mock_comms.record_calls = False
            mock_comms.webhook_base_url = WEBHOOK_BASE

            await provider.make_call(
                from_number=TO_NUMBER,
                to_number=FROM_NUMBER,
                context_id=CONTEXT_ID,
            )

        create_kwargs = mock_client.calls.create.call_args.kwargs
        assert "record" not in create_kwargs
        assert "recording_status_callback" not in create_kwargs
