"""
Webhook endpoints for telephony providers.

Handles incoming calls, SMS, and status updates from SignalWire.
Supports both SWML (new) and LaML (legacy) formats.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse

from ...comms import comms_settings, get_comms_service
from ...comms.context import get_context_router

logger = logging.getLogger("atlas.api.comms.webhooks")

router = APIRouter(prefix="/voice")


def swml_response(sections: dict) -> JSONResponse:
    """Return a SWML JSON response."""
    return JSONResponse(
        content={
            "version": "1.0.0",
            "sections": sections,
        }
    )


def laml_response(content: str) -> Response:
    """Return a LaML/TwiML XML response (legacy)."""
    return Response(
        content=content,
        media_type="application/xml",
    )


def _normalize_to_e164(raw_to: str) -> str:
    """Normalize SIP URI or phone input into E.164 when possible."""
    if not raw_to:
        return ""

    # Strip SIP URI envelope first (sip:user@domain -> user)
    clean = re.sub(r"^sip:", "", raw_to, flags=re.IGNORECASE)
    clean = re.sub(r"@.*", "", clean)

    # Keep digits and optional leading plus
    had_plus = clean.strip().startswith("+")
    digits = re.sub(r"\D", "", clean)

    if not digits:
        return clean.strip()

    if had_plus:
        return f"+{digits}"

    # US defaults
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"

    # Fallback: prefix plus for international/non-standard lengths
    return f"+{digits}"


def _fallback_outbound_caller_id() -> str:
    """Best-effort E.164 caller ID for outbound PSTN bridging."""
    try:
        context_router = get_context_router()
        for ctx in context_router.list_contexts():
            for number in getattr(ctx, "phone_numbers", []) or []:
                normalized = _normalize_to_e164(number)
                if normalized.startswith("+"):
                    return normalized
    except Exception:
        pass
    return ""


# Track which calls already have recording started (avoid duplicates).
# In-memory set for fast path; DB check via call_transcripts for restart resilience.
_recording_started: set[str] = set()


async def _is_recording_already_started(call_sid: str) -> bool:
    """Check if recording was already started for this call.

    Fast path: in-memory set (covers current process lifetime).
    Slow path: DB lookup (covers restart scenarios where the set was lost).
    """
    if call_sid in _recording_started:
        return True
    try:
        from ...storage.repositories.call_transcript import get_call_transcript_repo
        existing = await get_call_transcript_repo().get_by_call_sid(call_sid)
        if existing:
            _recording_started.add(call_sid)  # cache for future checks
            return True
    except Exception:
        pass  # DB unavailable -- rely on in-memory only
    return False


async def _start_recording_for_call(call_sid: str):
    """Start recording via REST API, retrying until the call is active.

    Tries at 2s, 5s, 10s after the webhook. The first success wins;
    the /status handler may also start recording on in-progress -- the
    _recording_started set + DB check prevents duplicates.
    """
    for delay in (2, 5, 10):
        if await _is_recording_already_started(call_sid):
            return
        await asyncio.sleep(delay)
        if await _is_recording_already_started(call_sid):
            return
        try:
            provider = get_comms_service().provider
            # Don't attempt recording if call already ended/failed.
            call = await provider.get_call(call_sid)
            if call is not None:
                state_value = getattr(getattr(call, "state", None), "value", "")
                if state_value in ("failed", "ended"):
                    logger.info(
                        "Skipping recording for %s because call state is %s",
                        call_sid,
                        state_value,
                    )
                    return

            cb_url = (
                f"{comms_settings.webhook_base_url}"
                "/api/v1/comms/voice/recording-status"
            )
            await provider.start_recording(
                call_sid=call_sid,
                recording_status_callback=cb_url,
            )
            _recording_started.add(call_sid)
            logger.info("Recording started for %s (background, delay=%ds)", call_sid, delay)
            return
        except Exception as e:
            logger.warning("Recording attempt for %s at %ds failed: %s", call_sid, delay, e)
            # SignalWire's Twilio-compatible SDK can raise JSON parse errors
            # even when the REST API returned 200. Treat that specific case as
            # success to avoid duplicate recording attempts.
            if "Expecting value" in str(e):
                _recording_started.add(call_sid)
                logger.info(
                    "Recording likely started for %s (HTTP 200 but SDK parse error), stopping retries",
                    call_sid,
                )
                return
    logger.warning("All recording attempts failed for %s (call may have ended)", call_sid)


async def prewarm_llm():
    """Prewarm LLM in background while greeting plays."""
    try:
        from ...services import llm_registry
        llm = llm_registry.get_active()
        if llm:
            from ...services.protocols import Message
            messages = [Message(role="user", content="Hello")]
            llm.chat(messages=messages, max_tokens=5)
            logger.info("LLM prewarmed successfully")
    except Exception as e:
        logger.warning("LLM prewarm failed: %s", e)


async def prewarm_personaplex(call_sid: str, from_number: str, to_number: str, context):
    """Pre-connect to PersonaPlex while SignalWire sets up the stream."""
    from ...comms.personaplex_processor import (
        create_personaplex_processor,
        get_personaplex_processor,
        remove_personaplex_processor,
    )

    try:
        # Check if already exists
        if get_personaplex_processor(call_sid):
            return

        t0 = time.time()
        logger.info(
            "Pre-warming PersonaPlex for call %s with context %s",
            call_sid,
            context.name if hasattr(context, 'name') else 'unknown',
        )

        # Create processor without audio callback (will be set later)
        processor = create_personaplex_processor(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            context_id=context.id,
            business_context=context,
            on_audio_ready=None,  # Set later when stream connects
        )

        # Connect to PersonaPlex server
        connected = await processor.connect()
        t1 = time.time()

        if connected:
            logger.info(
                "PersonaPlex pre-warmed in %.2fs for %s",
                t1 - t0,
                call_sid,
            )
        else:
            logger.error("PersonaPlex prewarm connection failed for %s", call_sid)
            await remove_personaplex_processor(call_sid)
    except Exception as e:
        logger.error("PersonaPlex prewarm error for %s: %s", call_sid, e)
        # Clean up on error
        try:
            await remove_personaplex_processor(call_sid)
        except Exception:
            pass


def is_laml_request(request: Request) -> bool:
    """Check if request is LaML (form data) vs SWML (JSON)."""
    content_type = request.headers.get("content-type", "")
    return "application/x-www-form-urlencoded" in content_type


@router.post("/inbound")
async def handle_inbound_call(request: Request):
    """
    Handle incoming voice call webhook.

    Supports both LaML (form data) and SWML (JSON) formats.
    LaML returns XML with Connect/Stream for bidirectional audio.
    SWML returns JSON with AI agent.
    """
    import asyncio

    raw_body = await request.body()
    logger.info("Raw webhook body: %s", raw_body.decode()[:500])

    # Determine format and parse accordingly
    use_laml = is_laml_request(request)

    if use_laml:
        # LaML format - form data
        form = await request.form()
        call_id = form.get("CallSid", "unknown")
        from_number = form.get("From", "")
        to_number = form.get("To", "")
        logger.info("LaML request: call=%s from=%s to=%s", call_id, from_number, to_number)
    else:
        # SWML format - JSON
        try:
            body = json.loads(raw_body)
            if "call" in body:
                call_data = body["call"]
                call_id = call_data.get("call_id", "unknown")
                from_number = call_data.get("from_number") or call_data.get("from", "")
                to_number = call_data.get("to_number") or call_data.get("to", "")
            else:
                call_id = "unknown"
                from_number = ""
                to_number = ""
            logger.info("SWML request: call=%s from=%s to=%s", call_id, from_number, to_number)
        except Exception as e:
            logger.error("JSON parse error: %s", e)
            call_id = "unknown"
            from_number = ""
            to_number = ""

    logger.info("Inbound call: %s from %s to %s", call_id, from_number, to_number)

    # Get the business context for this phone number
    context_router = get_context_router()
    context = context_router.get_context_for_number(to_number)

    logger.info("Routing to context: %s (%s)", context.id, context.name)

    # Check if within business hours
    status = context_router.get_business_status(context)

    if not status["is_open"]:
        # After hours - play message and prompt to leave voicemail
        if use_laml:
            return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{status['message']}</Say>
    <Record maxLength="120" finishOnKey="#" />
    <Hangup />
</Response>""")
        else:
            return swml_response({
                "main": [
                    {"play": {"url": f"say:{status['message']}", "say_voice": "en-US-Neural2-F"}},
                    {"record": {"stereo": False, "max_length": 120, "terminators": "#"}},
                    {"hangup": {}}
                ]
            })

    # During business hours - greet and start AI conversation
    try:
        provider = get_comms_service().provider

        # Track the call
        call = await provider.handle_incoming_call(
            call_sid=call_id,
            from_number=from_number,
            to_number=to_number,
        )
        call.context_id = context.id

        logger.info("Starting AI conversation for call %s (laml=%s)", call_id, use_laml)

        if use_laml:
            # If a forward number is configured, skip AI and dial directly.
            if comms_settings.forward_to_number:
                logger.info(
                    "Forwarding call %s to %s",
                    call_id,
                    comms_settings.forward_to_number,
                )
                # Recording via REST API. The record= attr on <Dial> is
                # broken on SignalWire. We use two paths:
                # 1) Background task retries at 2s/5s/10s
                # 2) /status handler fires on in-progress
                # Whichever succeeds first wins (_recording_started guard).
                if comms_settings.record_calls:
                    asyncio.create_task(
                        _start_recording_for_call(call_id)
                    )
                return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial timeout="30" answerOnBridge="true">
        {comms_settings.forward_to_number}
    </Dial>
</Response>""")

            # No forward number - use Atlas AI via bidirectional WebSocket stream
            asyncio.create_task(prewarm_llm())

            # Pre-connect PersonaPlex in background (takes ~6s)
            if comms_settings.personaplex_enabled:
                asyncio.create_task(
                    prewarm_personaplex(call_id, from_number, to_number, context)
                )

            # Build WebSocket URL for audio streaming
            ws_url = comms_settings.webhook_base_url.replace(
                "https://", "wss://"
            ).replace("http://", "ws://")
            stream_url = f"{ws_url}/api/v1/comms/voice/stream/{call_id}"

            logger.info("LaML stream URL: %s", stream_url)

            recording_attrs = ""
            if comms_settings.record_calls:
                cb_url = (
                    f"{comms_settings.webhook_base_url}"
                    "/api/v1/comms/voice/recording-status"
                )
                recording_attrs = (
                    f' record="record-from-answer-dual"'
                    f' recordingStatusCallback="{cb_url}"'
                    f' recordingStatusCallbackEvent="completed"'
                )

            return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect{recording_attrs}>
        <Stream url="{stream_url}" />
    </Connect>
</Response>""")

        # SWML: Use Atlas models via WebSocket tap
        # Prewarm LLM in background while greeting plays
        asyncio.create_task(prewarm_llm())

        # Build WebSocket URL for audio streaming
        ws_url = comms_settings.webhook_base_url.replace(
            "https://", "wss://"
        ).replace("http://", "ws://")
        stream_url = f"{ws_url}/api/v1/comms/voice/stream/{call_id}"

        logger.info("SWML tap stream URL: %s", stream_url)

        # Return SWML with tap for bidirectional WebSocket
        swml_sections = {
            "main": [
                {"answer": {}},
            ]
        }

        # Enable call recording if configured
        if comms_settings.record_calls:
            cb_url = (
                f"{comms_settings.webhook_base_url}"
                "/api/v1/comms/voice/recording-status"
            )
            swml_sections["main"].append({
                "record_call": {
                    "format": "wav",
                    "stereo": True,
                    "status_callback": cb_url,
                }
            })

        swml_sections["main"].extend([
            {
                "play": {
                    "url": f"say:{context.greeting}",
                    "say_voice": "en-US-Neural2-F"
                }
            },
            {
                "tap": {
                    "uri": stream_url,
                    "direction": "both"
                }
            },
            {
                "play": {
                    "url": "silence:300",
                }
            }
        ])

        return swml_response(swml_sections)

    except Exception as e:
        logger.error("Error handling inbound call: %s", e)
        return swml_response({
            "main": [
                {
                    "play": {
                        "url": "say:I'm sorry, I'm having trouble right now. Please try again later.",
                        "say_voice": "en-US-Neural2-F"
                    }
                },
                {"hangup": {}}
            ]
        })


# Store conversation history per call
_call_conversations: dict[str, list[dict]] = {}


@router.post("/conversation")
async def handle_conversation(request: Request):
    """
    Handle conversation callback from SWML prompt.

    Receives speech recognition result, processes with ReceptionistAgent,
    returns SWML to play the response.
    """
    raw_body = await request.body()
    logger.info("Conversation callback: %s", raw_body.decode()[:500])

    try:
        body = json.loads(raw_body)

        # Extract speech result from SWML execute callback
        # Format: {"vars": {"prompt_result": "...", "prompt_value": "..."}, ...}
        vars_data = body.get("vars", {})
        speech_text = vars_data.get("prompt_value", "")

        # Also check for speech in other locations
        if not speech_text:
            speech_text = vars_data.get("prompt_result", "")
        if not speech_text:
            speech_text = body.get("speech", {}).get("text", "")

        # Get call info from params
        params = body.get("params", {})
        call_id = params.get("call_id", "unknown")
        context_id = params.get("context_id", "")
        from_number = params.get("from_number", "")

        logger.info("Speech from %s: %s", call_id, speech_text)

        if not speech_text or not speech_text.strip():
            # No speech detected, prompt again
            return swml_response({
                "main": [
                    {
                        "play": {
                            "url": "say:I didn't catch that. How can I help you?",
                            "say_voice": "en-US-Neural2-F"
                        }
                    },
                    {"return": {}}
                ]
            })

        # Get or create conversation history
        if call_id not in _call_conversations:
            _call_conversations[call_id] = []

        # Get business context
        context_router = get_context_router()
        business_context = None
        if context_id:
            business_context = context_router.get_context(context_id)

        # Normalize call_id to a valid UUID for session persistence
        from ...utils.session_id import normalize_session_id, ensure_session_row

        norm_session_id = normalize_session_id(call_id)
        await ensure_session_row(norm_session_id)

        # Process with ReceptionistAgent via unified interface
        from ...agents.interface import get_agent

        agent = get_agent(
            agent_type="receptionist",
            session_id=norm_session_id,
            business_context=business_context,
        )

        agent_result = await agent.process(
            input_text=speech_text,
            input_type="voice",
            session_id=norm_session_id,
            runtime_context={
                "conversation_history": _call_conversations[call_id],
            },
        )
        response_text = agent_result.response_text or "I'm not sure how to help with that."

        logger.info("Agent response: %s", response_text)

        # Store in conversation history
        _call_conversations[call_id].append({
            "role": "user",
            "content": speech_text,
        })
        _call_conversations[call_id].append({
            "role": "assistant",
            "content": response_text,
        })

        # Check for goodbye/end phrases
        goodbye_phrases = ["goodbye", "bye", "hang up", "end call", "that's all"]
        if any(phrase in speech_text.lower() for phrase in goodbye_phrases):
            # Clean up and end call
            if call_id in _call_conversations:
                del _call_conversations[call_id]

            return swml_response({
                "main": [
                    {
                        "play": {
                            "url": f"say:{response_text}",
                            "say_voice": "en-US-Neural2-F"
                        }
                    },
                    {"hangup": {}}
                ]
            })

        # Return response and continue conversation
        return swml_response({
            "main": [
                {
                    "play": {
                        "url": f"say:{response_text}",
                        "say_voice": "en-US-Neural2-F"
                    }
                },
                {"return": {}}
            ]
        })

    except Exception as e:
        logger.exception("Conversation error: %s", e)
        return swml_response({
            "main": [
                {
                    "play": {
                        "url": "say:I'm sorry, I'm having some trouble. Let me try again.",
                        "say_voice": "en-US-Neural2-F"
                    }
                },
                {"return": {}}
            ]
        })


@router.api_route("/sip-outbound", methods=["GET", "POST"])
async def handle_sip_outbound(request: Request):
    """
    Handle outbound calls from SIP endpoint (Zoiper).

    SignalWire hits this webhook when the SIP client dials a number.
    Returns LaML to connect the call to the PSTN destination, with
    optional recording.
    """
    raw_body = await request.body()
    logger.info("SIP outbound webhook: %s", raw_body.decode()[:500])

    # SignalWire can send SIP endpoint webhook as GET query params.
    if request.method == "GET":
        query = request.query_params
        call_id = query.get("CallSid", "unknown")
        to_number = query.get("To", "")
        from_number = query.get("From", "")
    # Parse form data or JSON for POST
    elif is_laml_request(request):
        form = await request.form()
        call_id = form.get("CallSid", "unknown")
        to_number = form.get("To", "")
        from_number = form.get("From", "")
    else:
        body = json.loads(raw_body) if raw_body else {}
        call_data = body.get("call", body)
        call_id = call_data.get("call_id") or call_data.get("CallSid", "unknown")
        to_number = call_data.get("to_number") or call_data.get("To", "")
        from_number = call_data.get("from_number") or call_data.get("From", "")

    # Clean SIP URI/phone to E.164: sip:8135855363@domain -> +18135855363
    clean = _normalize_to_e164(to_number)

    if not clean or not clean.startswith("+"):
        logger.warning(
            "SIP outbound %s invalid destination after normalization: raw=%s normalized=%s",
            call_id,
            to_number,
            clean,
        )
        return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, that number format is invalid.</Say>
    <Hangup />
</Response>""")

    logger.info("SIP outbound: %s dialing %s -> %s (from %s)", call_id, to_number, clean, from_number)

    # Track SIP-originated parent leg so /status and recording pipeline can
    # resolve context and numbers for this call.
    try:
        provider = get_comms_service().provider
        await provider.handle_incoming_call(
            call_sid=call_id,
            from_number=from_number,
            to_number=clean or to_number,
        )
    except Exception as e:
        logger.debug("Could not pre-track SIP outbound parent leg %s: %s", call_id, e)

    dial_action = f"{comms_settings.webhook_base_url}/api/v1/comms/voice/dial-status"

    # For SIP-originated calls, provider default caller ID may remain a SIP URI,
    # which many PSTN routes reject before creating a dial leg. Prefer E.164.
    caller_id_attr = ""
    if from_number and not from_number.lower().startswith("sip:"):
        caller_id_attr = f' callerId="{from_number}"'
    else:
        fallback_caller_id = _fallback_outbound_caller_id()
        if fallback_caller_id:
            caller_id_attr = f' callerId="{fallback_caller_id}"'

    logger.info(
        "SIP outbound %s dial config: target=%s caller_id_attr=%s",
        call_id,
        clean,
        caller_id_attr or "<provider-default>",
    )

    # Recording: use record= attribute on <Dial> instead of REST API.
    # SignalWire's recordings.create() REST endpoint returns HTTP 200 with
    # an unparseable body for SIP-originated calls — the recording is never
    # actually created. The <Dial record=...> attribute works reliably and
    # fires recordingStatusCallback when the recording is ready.
    record_attr = ""
    recording_cb_attr = ""
    if comms_settings.record_calls:
        recording_cb = (
            f"{comms_settings.webhook_base_url}"
            "/api/v1/comms/voice/recording-status"
        )
        record_attr = ' record="record-from-answer-dual"'
        recording_cb_attr = f' recordingStatusCallback="{recording_cb}"'

    # Answer the SIP leg immediately so Zoiper shows the call as active
    # and the user hears ringing while the customer's phone rings.
    # Without this, answerOnBridge keeps the SIP leg silent until the
    # customer picks up — which feels like a dead call from Zoiper.
    return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connecting your call.</Say>
    <Dial timeout="60"{caller_id_attr}{record_attr}{recording_cb_attr} action="{dial_action}" method="POST">
        <Number>{clean}</Number>
    </Dial>
</Response>""")


@router.post("/sip-outbound-swml")
async def handle_sip_outbound_swml(request: Request):
    """
    Handle outbound SIP calls via SWML execute.

    Called by SWML execute with call variables as params.
    Returns SWML JSON with connect to the cleaned PSTN number
    and record_call for the intelligence pipeline.
    """
    raw_body = await request.body()
    logger.info("SIP outbound SWML: %s", raw_body.decode()[:500])

    try:
        body = json.loads(raw_body)
    except Exception:
        body = {}

    # SWML execute sends the full call context under "call"
    call_data = body.get("call", {})
    call_to = call_data.get("to") or body.get("call_to", "")
    call_id = call_data.get("call_id") or body.get("call_id", "unknown")

    # Clean SIP URI to E.164: sip:8135855363@domain -> +18135855363
    import re
    clean = re.sub(r'^sip:', '', call_to, flags=re.IGNORECASE)
    clean = re.sub(r'@.*', '', clean)
    if clean and not clean.startswith('+'):
        clean = f'+1{clean}' if len(clean) == 10 else f'+{clean}'

    logger.info("SIP outbound SWML: %s dialing %s -> %s", call_id, call_to, clean)

    recording_url = (
        f"{comms_settings.webhook_base_url}"
        "/api/v1/comms/voice/recording-status"
    )

    # Return SWML that just sets the cleaned number as a variable.
    # The calling SWML script uses %{dest_number} in its connect verb.
    sections = {"main": [
        {"set": {"dest_number": clean}},
    ]}

    return swml_response(sections)


@router.post("/outbound")
async def handle_outbound_call(
    request: Request,
    CallSid: str = Form(...),
    To: str = Form(...),
    From: str = Form(...),
):
    """
    Handle programmatic outbound calls (from make_call MCP tool).

    This is NOT the primary outbound path — SIP outbound via Zoiper uses
    /sip-outbound instead. This endpoint handles calls initiated via the
    REST API (e.g., the make_call MCP tool).

    Flow: bridges the call to forward_to_number so the owner can talk
    to the customer directly. Recording is handled by /status on in-progress.
    AI does NOT talk to the customer.
    """
    logger.info("Outbound call answered: %s (customer=%s, from=%s)", CallSid, To, From)

    try:
        bridge_to = comms_settings.forward_to_number
        if not bridge_to:
            logger.error("No forward_to_number configured — cannot bridge outbound call %s", CallSid)
            return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, we are unable to connect your call right now.</Say>
    <Hangup />
</Response>""")

        dial_action = f"{comms_settings.webhook_base_url}/api/v1/comms/voice/dial-status"

        # Recording is started from /status when the call goes in-progress.
        # No need to start it here — same pattern as inbound/SIP outbound.

        return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Please hold while we connect you.</Say>
    <Dial timeout="30" answerOnBridge="true" action="{dial_action}" method="POST">
        {bridge_to}
    </Dial>
</Response>""")

    except Exception as e:
        logger.error("Error handling outbound call: %s", e)
        return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, something went wrong. Goodbye.</Say>
    <Hangup />
</Response>""")


@router.post("/dial-status")
async def handle_dial_status(
    request: Request,
    CallSid: str = Form(...),
    DialCallStatus: str = Form(...),
    DialCallSid: Optional[str] = Form(None),
    DialCallDuration: Optional[str] = Form(None),
):
    """
    Action callback fired by SignalWire when a <Dial> leg completes.

    When call forwarding is used with record_calls=True, this starts a
    recording on the call via the REST API (more reliable than <Dial record=...>).
    Fires the call intelligence pipeline once the recording is ready.
    """
    logger.info(
        "Dial status for %s: %s (leg=%s, duration=%s)",
        CallSid, DialCallStatus, DialCallSid, DialCallDuration,
    )

    # Capture full provider payload for troubleshooting SIP leg failures.
    try:
        form = await request.form()
        payload = dict(form)
        if DialCallStatus in ("failed", "busy", "no-answer", "canceled"):
            logger.warning("Dial failed payload for %s: %s", CallSid, payload)
        else:
            logger.info("Dial payload for %s: %s", CallSid, payload)
    except Exception as e:
        logger.debug("Unable to parse dial-status payload for %s: %s", CallSid, e)

    if DialCallStatus == "completed" and comms_settings.record_calls:
        try:
            cb_url = (
                f"{comms_settings.webhook_base_url}"
                "/api/v1/comms/voice/recording-status"
            )
            provider = get_comms_service().provider
            await provider.start_recording(
                call_sid=CallSid,
                recording_status_callback=cb_url,
            )
            logger.info("Started REST recording for call %s", CallSid)
        except Exception as e:
            logger.warning("Failed to start recording for %s: %s", CallSid, e)

    return Response(status_code=204)


@router.post("/status")
async def handle_call_status(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    Duration: Optional[str] = Form(None),
    ParentCallSid: Optional[str] = Form(None),
    To: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    Direction: Optional[str] = Form(None),
):
    """
    Handle call status updates.

    Called when call state changes (ringing, in-progress, completed, etc.).
    Starts recording via REST API when the call goes in-progress, since
    the record= attribute on <Dial> is broken on SignalWire.
    """
    logger.info(
        "Call %s status: %s (duration: %s, parent: %s, direction: %s)",
        CallSid,
        CallStatus,
        Duration,
        ParentCallSid,
        Direction,
    )

    # Capture provider-specific failure hints for SIP debugging.
    if CallStatus in ("failed", "busy", "no-answer", "canceled"):
        try:
            form = await request.form()
            failure_details = {
                "ErrorCode": form.get("ErrorCode"),
                "ErrorMessage": form.get("ErrorMessage"),
                "SipResponseCode": form.get("SipResponseCode"),
                "SipErrorCode": form.get("SipErrorCode"),
                "DialCallStatus": form.get("DialCallStatus"),
                "DialCallSid": form.get("DialCallSid"),
                "ParentCallSid": form.get("ParentCallSid"),
                "To": form.get("To"),
                "From": form.get("From"),
            }
            logger.warning("Call %s failed status details: %s", CallSid, failure_details)
        except Exception as e:
            logger.debug("Unable to parse status failure details for %s: %s", CallSid, e)

    try:
        provider = get_comms_service().provider

        # For <Dial> scenarios SignalWire sends child-leg statuses with a
        # different CallSid. Ensure child leg is tracked before status update.
        # This avoids "unknown call" warnings and preserves recording context.
        existing = await provider.get_call(CallSid)
        if existing is None:
            parent_call = await provider.get_call(ParentCallSid) if ParentCallSid else None
            from_number = parent_call.from_number if parent_call else (From or "")
            to_number = parent_call.to_number if parent_call else (To or "")
            child = await provider.handle_incoming_call(
                call_sid=CallSid,
                from_number=from_number,
                to_number=to_number,
            )
            if parent_call and parent_call.context_id:
                child.context_id = parent_call.context_id

        await provider.handle_call_status(CallSid, CallStatus)
    except Exception as e:
        logger.error("Error handling call status: %s", e)

    # Start recording as soon as the call is active (fast path).
    # Only record the parent leg -- child legs (dial-out) have a ParentCallSid
    # and recording them would duplicate the pipeline run + ntfy notification.
    if CallStatus == "in-progress" and comms_settings.record_calls:
        if ParentCallSid:
            logger.debug("Skipping recording for child leg %s (parent=%s)", CallSid, ParentCallSid)
        elif not await _is_recording_already_started(CallSid):
            try:
                provider = get_comms_service().provider
                cb_url = (
                    f"{comms_settings.webhook_base_url}"
                    "/api/v1/comms/voice/recording-status"
                )
                await provider.start_recording(
                    call_sid=CallSid,
                    recording_status_callback=cb_url,
                )
                _recording_started.add(CallSid)
                logger.info("Recording started for %s on in-progress", CallSid)
            except Exception as e:
                logger.warning("Failed to start recording for %s: %s", CallSid, e)

    # Clean up tracking when call ends
    if CallStatus in ("completed", "failed", "canceled", "busy", "no-answer"):
        _recording_started.discard(CallSid)

    return Response(status_code=204)


@router.api_route("/whisper", methods=["GET", "POST"])
async def call_whisper(
    request: Request,
    from_: str = Query("", alias="from"),
    context: str = Query(""),
):
    """
    Whisper endpoint played to the answering party (you) before the call bridges.

    SignalWire fetches this URL when you pick up the forwarded call.
    The caller hears nothing until you press a key or the whisper finishes.
    """
    # Format caller number for speech: +16185551234 -> 618-555-1234
    spoken_number = from_
    if from_.startswith("+1") and len(from_) == 12:
        digits = from_[2:]
        spoken_number = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif from_.startswith("+") and len(from_) > 1:
        spoken_number = from_[1:]

    biz_name = context or "Business"

    laml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{biz_name} call from {spoken_number}.</Say>
    <Gather numDigits="1" timeout="5" />
</Response>"""
    return Response(content=laml, media_type="application/xml")


@router.post("/voicemail")
async def handle_voicemail(
    request: Request,
    CallSid: str = Form(...),
    RecordingUrl: str = Form(...),
    RecordingDuration: str = Form(...),
    From: str = Form(""),
    To: str = Form(""),
    context: str = "",
):
    """
    Handle voicemail recording completion.

    Saves voicemail to the appointment_messages table and sends
    a push notification to the business owner.
    """
    logger.info(
        "Voicemail received: %s (%ss) from %s for context %s",
        RecordingUrl,
        RecordingDuration,
        From,
        context,
    )

    # Resolve business context for this number
    context_router = get_context_router()
    biz_context = context_router.get_context_for_number(To)
    context_id = biz_context.id if biz_context else (context or "unknown")

    # Persist voicemail as an appointment message
    try:
        from ...storage.repositories.appointment import get_appointment_repo
        repo = get_appointment_repo()
        await repo.create_message(
            caller_phone=From,
            message_text=f"Voicemail ({RecordingDuration}s)",
            business_context_id=context_id,
            metadata={
                "type": "voicemail",
                "recording_url": RecordingUrl,
                "recording_duration": int(RecordingDuration or 0),
                "call_sid": CallSid,
            },
        )
        logger.info("Voicemail saved for call %s", CallSid)
    except Exception as e:
        logger.error("Failed to save voicemail: %s", e)

    # Send push notification to business owner
    try:
        from ...tools.notify import get_notify_tool
        notify = get_notify_tool()
        duration = int(RecordingDuration or 0)
        biz_name = biz_context.name if biz_context else "Atlas"
        await notify._send_notification(
            title=f"{biz_name}: New Voicemail",
            message=f"Voicemail from {From} ({duration}s)",
            priority="high",
        )
    except Exception as e:
        logger.warning("Failed to send voicemail notification: %s", e)

    return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for your message. We will get back to you soon. Goodbye.</Say>
    <Hangup />
</Response>""")


@router.post("/recording-status")
async def handle_recording_status(request: Request):
    """
    Handle recording status updates.

    Updates the voicemail message metadata with recording status
    (completed, failed, etc.) and final recording URL.

    SignalWire field names vary between REST-initiated and SWML-initiated
    recordings, so we parse form data dynamically.
    """
    # SignalWire may send form-encoded or JSON depending on how
    # the recording was initiated (REST API vs SWML record_call).
    raw_body = await request.body()
    content_type = request.headers.get("content-type", "")
    logger.info("Recording status webhook content-type=%s body=%s", content_type, raw_body.decode()[:500])

    params: dict = {}
    if "application/json" in content_type:
        try:
            params = json.loads(raw_body)
        except Exception:
            pass
    elif "application/x-www-form-urlencoded" in content_type:
        from urllib.parse import parse_qs
        parsed = parse_qs(raw_body.decode())
        params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
    else:
        # Try JSON first, then form
        try:
            params = json.loads(raw_body)
        except Exception:
            from urllib.parse import parse_qs
            try:
                parsed = parse_qs(raw_body.decode())
                params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
            except Exception:
                pass

    # SWML record_call sends JSON with fields nested under "params".
    # REST-initiated recordings send form-encoded with top-level fields.
    inner = params.get("params", {})
    merged = {**params, **inner}  # inner overrides top-level

    logger.info("Recording status merged: %s", {
        k: v for k, v in merged.items() if k != "params"
    })

    # Normalize field names across both formats
    CallSid = (merged.get("CallSid") or merged.get("call_sid")
               or merged.get("call_id") or "unknown")
    RecordingSid = (merged.get("RecordingSid") or merged.get("recording_sid")
                    or merged.get("recording_id") or "unknown")
    RecordingUrl = (merged.get("RecordingUrl") or merged.get("recording_url")
                    or merged.get("url"))
    RecordingDuration = (merged.get("RecordingDuration")
                         or merged.get("recording_duration")
                         or merged.get("duration"))

    # SWML uses state: "recording"/"finished"; REST uses RecordingStatus: "completed"
    raw_state = (merged.get("RecordingStatus") or merged.get("recording_status")
                 or merged.get("state") or "")
    # Map SWML state names to the ones our pipeline expects
    RecordingStatus = {"finished": "completed", "recording": "in-progress"}.get(
        raw_state, raw_state
    )

    logger.info(
        "Recording %s for call %s: %s (url=%s, duration=%s)",
        RecordingSid,
        CallSid,
        RecordingStatus,
        RecordingUrl,
        RecordingDuration,
    )

    if RecordingStatus == "completed" and RecordingUrl:
        try:
            from ...storage.database import get_db_pool
            pool = get_db_pool()
            if pool.is_initialized:
                await pool.execute(
                    """
                    UPDATE appointment_messages
                    SET metadata = metadata || $1::jsonb
                    WHERE metadata->>'call_sid' = $2
                    """,
                    json.dumps({
                        "recording_status": RecordingStatus,
                        "recording_sid": RecordingSid,
                        "final_recording_url": RecordingUrl,
                        "final_duration": int(RecordingDuration or 0),
                    }),
                    CallSid,
                )
                logger.info("Recording status updated for call %s", CallSid)
        except Exception as e:
            logger.warning("Failed to update recording status: %s", e)

        # Trigger call intelligence pipeline (skip if already processed)
        duration = int(RecordingDuration or 0)
        already_processed = False
        try:
            from ...storage.repositories.call_transcript import get_call_transcript_repo
            existing = await get_call_transcript_repo().get_by_call_sid(CallSid)
            if existing:
                already_processed = True
                logger.info("Skipping duplicate recording webhook for %s (already in DB)", CallSid)
        except Exception:
            pass  # DB unavailable -- proceed with processing
        if not already_processed:
            _spawn_recording_processing(
                CallSid, RecordingUrl, duration,
            )

    return Response(status_code=204)


@router.post("/swml-debug")
async def swml_debug(request: Request):
    """
    Debug endpoint called from SWML request step.
    Logs all variables SignalWire injects into the SWML execution context.
    Returns 200 with no body so SWML continues to the next step.
    """
    try:
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            import json as _json
            params = _json.loads(body)
        else:
            from urllib.parse import parse_qs
            parsed = parse_qs(body.decode("utf-8", errors="replace"))
            params = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        logger.info("SWML debug variables: %s", params)
    except Exception as e:
        logger.warning("SWML debug parse error: %s", e)
    return Response(status_code=200)


@router.websocket("/stream/{call_sid}")
async def handle_audio_stream(websocket: WebSocket, call_sid: str):
    """
    Handle bidirectional audio streaming for a call.

    This WebSocket receives audio from the caller and sends AI responses back.
    Audio format: 8kHz mulaw (base64 encoded in JSON messages).
    Uses PersonaPlex for direct speech-to-speech processing.
    """
    from ...comms.personaplex_processor import (
        get_personaplex_processor,
        create_personaplex_processor,
        remove_personaplex_processor,
    )

    await websocket.accept()
    logger.info("Audio stream connected for call %s", call_sid)

    # Register in the global media stream registry so the Twilio provider's
    # stream_audio_to_call() / set_audio_callback() can reach this WebSocket.
    from ...comms.core.media_streams import get_media_stream_registry

    media_registry = get_media_stream_registry()
    media_stream = media_registry.register(call_sid)

    stream_sid = None
    processor = None
    from_number = ""
    to_number = ""
    context = None

    try:
        provider = get_comms_service().provider
        call = await provider.get_call(call_sid)

        # Get context - try from call first, then default to first context
        context_router = get_context_router()

        if call and call.context_id:
            context = context_router.get_context(call.context_id)

        if context is None:
            # Fallback to first registered context
            contexts = context_router.list_contexts()
            if contexts:
                context = contexts[0]
                logger.info("Using fallback context: %s", context.id)

        if context is None:
            logger.error("No context available for call %s", call_sid)
            await websocket.close()
            return

        from_number = call.from_number if call else ""
        to_number = call.to_number if call else ""

        # Track if we need to set callback after stream starts (pre-warm case)
        needs_callback_setup = False

        processor = get_personaplex_processor(call_sid)

        async def send_audio(audio_b64: str):
            """Callback to send audio back to caller."""
            logger.info(
                "send_audio callback: stream_sid=%s, audio_len=%d",
                stream_sid,
                len(audio_b64),
            )
            if stream_sid:
                await websocket.send_json({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": audio_b64},
                })
                logger.info("Audio sent to SignalWire")
            else:
                logger.warning("No stream_sid, cannot send audio")

        if processor is not None and processor.state.is_connected:
            # Pre-warmed and connected - wait for stream_sid before setting callback
            logger.info("Using pre-warmed PersonaPlex for %s", call_sid)
            needs_callback_setup = True
        elif processor is not None and processor.state.is_connecting:
            # Processor is connecting - wait for it
            logger.info("Waiting for PersonaPlex connection for %s", call_sid)
            for _ in range(300):  # Wait up to 30 seconds (handshake can take ~21s)
                await asyncio.sleep(0.1)
                if processor.state.is_connected:
                    logger.info("PersonaPlex connection completed for %s", call_sid)
                    needs_callback_setup = True
                    break
                if not processor.state.is_connecting:
                    logger.error("PersonaPlex connection failed for %s", call_sid)
                    break
            else:
                logger.error("PersonaPlex connection timeout for %s", call_sid)

            if not processor.state.is_connected:
                await remove_personaplex_processor(call_sid)
                await websocket.close()
                return
        else:
            # No pre-warmed processor - create new one
            if processor is not None:
                logger.info("Pre-warm failed for %s, recreating", call_sid)
                await remove_personaplex_processor(call_sid)

            t0 = time.time()
            logger.info("Starting PersonaPlex connection for %s", call_sid)
            processor = create_personaplex_processor(
                call_sid=call_sid,
                from_number=from_number,
                to_number=to_number,
                context_id=context.id,
                business_context=context,
                on_audio_ready=lambda b64: asyncio.create_task(send_audio(b64)),
            )
            connected = await processor.connect()
            t1 = time.time()
            logger.info(
                "PersonaPlex connect took %.2fs for %s",
                t1 - t0,
                call_sid,
            )
            if not connected:
                logger.error("Failed to connect PersonaPlex for %s", call_sid)
                await remove_personaplex_processor(call_sid)
                await websocket.close()
                return
            logger.info("Created PersonaPlex processor for call %s", call_sid)

        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "connected":
                logger.info("Stream connected data: %s", data)

            elif event == "start":
                # SignalWire/Twilio nests streamSid inside "start" object
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                logger.info("Stream started: %s (format: %s)",
                           stream_sid, start_data.get("mediaFormat", {}))
                if processor:
                    processor._state.stream_sid = stream_sid

                # Wire the media stream registry so the Twilio provider
                # can push audio to this WebSocket via stream_audio_to_call().
                media_stream.stream_sid = stream_sid

                async def _registry_send(payload_b64: str):
                    if stream_sid:
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload_b64},
                        })

                media_stream.set_send_func(_registry_send)
                await media_stream.flush_buffer()

                # For pre-warmed PersonaPlex, now set callback and drain buffer
                if needs_callback_setup and processor:
                    logger.info(
                        "Setting PersonaPlex callback and draining buffer for %s",
                        call_sid,
                    )
                    # Set callback and get any buffered audio
                    buffered_audio = processor.set_audio_callback(
                        lambda b64: asyncio.create_task(send_audio(b64))
                    )
                    # Send buffered audio (greeting from PersonaPlex)
                    if buffered_audio:
                        logger.info(
                            "Sending %d buffered audio chunks for %s",
                            len(buffered_audio),
                            call_sid,
                        )
                        for audio_chunk in buffered_audio:
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": audio_chunk},
                            })
                        logger.info("Buffered audio sent to SignalWire")
                    needs_callback_setup = False

                if not needs_callback_setup:
                    logger.info("PersonaPlex greeting sent from buffer")

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    # Feed audio to PersonaPlex processor if active
                    if processor:
                        await processor.process_audio_chunk(payload)
                    # Also feed to media stream registry callback (for
                    # provider-level set_audio_callback consumers)
                    if media_stream._audio_callback is not None:
                        import base64 as _b64
                        raw = _b64.b64decode(payload)
                        await media_stream.receive_audio(raw)

            elif event == "stop":
                logger.info("Stream stopped for call %s", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("Audio stream disconnected for call %s", call_sid)
    except Exception as e:
        logger.error("Audio stream error for call %s: %s", call_sid, e)
    finally:
        media_registry.unregister(call_sid)
        if processor:
            await remove_personaplex_processor(call_sid)
        logger.info("Audio stream ended for call %s", call_sid)


def _spawn_recording_processing(
    call_sid: str, recording_url: str, duration: int,
):
    """Spawn background task to process a completed call recording."""
    logger.info(
        "Spawning call intelligence pipeline: call=%s url=%s duration=%d",
        call_sid, recording_url, duration,
    )
    asyncio.create_task(_run_recording_processing(
        call_sid, recording_url, duration,
    ))


async def _run_recording_processing(
    call_sid: str, recording_url: str, duration: int,
):
    """Run call intelligence pipeline on a SignalWire recording."""
    try:
        provider = get_comms_service().provider
        call = await provider.get_call(call_sid)
        from_number = call.from_number if call else ""
        to_number = call.to_number if call else ""
        context_id = (call.context_id if call else None) or "unknown"

        # Resolve business context
        biz_ctx = None
        if context_id and context_id != "unknown":
            ctx_router = get_context_router()
            biz_ctx = ctx_router.get_context(context_id)

        from ...comms.call_intelligence import process_call_recording
        await process_call_recording(
            call_sid=call_sid,
            recording_url=recording_url,
            from_number=from_number,
            to_number=to_number,
            context_id=context_id,
            duration_seconds=duration,
            business_context=biz_ctx,
        )
    except Exception as e:
        logger.error("Call recording processing failed for %s: %s", call_sid, e)


async def _process_inbound_sms(
    sms_id, from_number: str, to_number: str, body: str, context, media_urls: list,
) -> None:
    """Background task: CRM link, intelligence pipeline, ntfy, auto-reply.

    Each step is fail-open -- partial results are better than no results.
    Matches the pattern of _run_recording_processing for calls.
    """
    from ...storage.repositories.sms_message import get_sms_message_repo

    sms_repo = get_sms_message_repo()

    # Step 1: Run SMS intelligence pipeline (classify + CRM + action plan + ntfy)
    try:
        from ...comms.sms_intelligence import process_inbound_sms as run_intelligence
        await run_intelligence(
            sms_id=sms_id,
            from_number=from_number,
            to_number=to_number,
            body=body,
            business_context_id=context.id,
            business_context=context,
            media_urls=media_urls,
        )
    except Exception as e:
        logger.error("SMS intelligence pipeline failed for %s: %s", sms_id, e)

        # Fallback: basic CRM + ntfy if intelligence pipeline is unavailable
        try:
            await _sms_fallback_crm_and_notify(
                sms_id, sms_repo, from_number, body, context,
            )
        except Exception as e2:
            logger.error("SMS fallback CRM+ntfy also failed: %s", e2)

    # Step 2: Auto-reply via LLM if enabled
    if context.sms_auto_reply and context.sms_enabled and body.strip():
        try:
            reply = await _generate_sms_reply(body, context)
            if reply:
                provider = get_comms_service().provider
                msg = await provider.send_sms(
                    to_number=from_number,
                    from_number=to_number,
                    body=reply,
                    context_id=context.id,
                )
                logger.info("SMS auto-reply sent to %s", from_number)

                # Persist outbound auto-reply
                try:
                    from uuid import uuid4 as _uuid4
                    await sms_repo.create(
                        message_sid=getattr(msg, "provider_message_id", "") or f"auto_reply_{_uuid4().hex[:12]}",
                        from_number=to_number,
                        to_number=from_number,
                        direction="outbound",
                        body=reply,
                        business_context_id=context.id,
                        status="sent",
                        source="auto_reply",
                        source_ref=str(sms_id) if sms_id else None,
                    )
                except Exception as e:
                    logger.warning("Failed to persist auto-reply SMS: %s", e)
        except Exception as e:
            logger.warning("SMS auto-reply failed: %s", e)


async def _sms_fallback_crm_and_notify(
    sms_id, sms_repo, from_number: str, body: str, context,
) -> None:
    """Fallback CRM link + ntfy when full intelligence pipeline is unavailable."""
    from ...services.crm_provider import get_crm_provider
    from ...config import settings

    # CRM lookup
    contact_id = None
    contact_name = from_number
    is_new_lead = False
    try:
        crm = get_crm_provider()
        contact = await crm.find_or_create_contact(
            full_name=from_number,
            phone=from_number,
            contact_type="customer",
            source="sms",
            business_context_id=context.id,
        )
        if contact.get("id"):
            contact_id = str(contact["id"])
            contact_name = contact.get("full_name") or from_number
            is_new_lead = contact.get("_was_created", False)
            if sms_id:
                await sms_repo.link_contact(sms_id, contact_id)
            await crm.log_interaction(
                contact_id=contact_id,
                interaction_type="sms",
                summary=f"Inbound SMS: {body[:100]}",
            )
    except Exception as e:
        logger.warning("Fallback CRM link failed: %s", e)

    # ntfy notification
    try:
        if not settings.alerts.ntfy_enabled or not settings.alerts.ntfy_url:
            return

        import httpx

        ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
        biz_name = context.name if hasattr(context, "name") else "Business"

        title = f"NEW LEAD SMS: {biz_name}" if is_new_lead else f"SMS: {biz_name}"
        message = f"From: {from_number}\nCustomer: {contact_name}\nMessage: {body[:200]}"

        headers = {
            "Title": title,
            "Priority": "high",
            "Tags": "speech_balloon",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(ntfy_url, content=message, headers=headers)

        if sms_id:
            await sms_repo.mark_notified(sms_id)
    except Exception as e:
        logger.warning("Fallback ntfy notification failed: %s", e)


async def _generate_sms_reply(body: str, context) -> Optional[str]:
    """Generate an SMS reply using the LLM with business context persona."""
    from ...services import llm_registry
    from ...services.protocols import Message

    llm = llm_registry.get_active()
    if not llm:
        logger.warning("No active LLM for SMS auto-reply")
        return None

    system_prompt = (
        f"You are {context.voice_name}, responding to an SMS for {context.name}."
    )
    if context.persona:
        system_prompt += f" {context.persona}"
    if context.business_type:
        system_prompt += f" Business type: {context.business_type}."
    if context.services:
        system_prompt += f" Services: {', '.join(context.services)}."
    system_prompt += (
        " Keep replies concise (under 160 characters if possible)."
        " Be helpful and professional."
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=body),
    ]

    loop = asyncio.get_event_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(messages=messages, max_tokens=200, temperature=0.7),
        ),
        timeout=15.0,
    )

    text = result.get("response", "").strip()
    if not text:
        return None

    # Strip <think> tags (Qwen3 models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text if text else None


# SMS webhooks
sms_router = APIRouter(prefix="/sms")


@sms_router.post("/inbound")
async def handle_inbound_sms(
    request: Request,
    MessageSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...),
    Body: str = Form(""),
    NumMedia: str = Form("0"),
):
    """
    Handle incoming SMS webhook.

    Persists to DB immediately, then spawns a background task for
    CRM linking, intelligence pipeline, ntfy, and auto-reply.
    Webhook returns TwiML in <1s.
    """
    logger.info("Inbound SMS from %s to %s: %s", From, To, Body[:50])

    # Get business context
    context_router = get_context_router()
    context = context_router.get_context_for_number(To)

    # Collect media URLs if any
    media_urls = []
    num_media = int(NumMedia)
    form_data = await request.form()
    for i in range(num_media):
        url = form_data.get(f"MediaUrl{i}")
        if url:
            media_urls.append(str(url))

    try:
        provider = get_comms_service().provider
        message = await provider.handle_incoming_sms(
            message_sid=MessageSid,
            from_number=From,
            to_number=To,
            body=Body,
            media_urls=media_urls,
        )
        message.context_id = context.id
    except Exception as e:
        logger.error("Error handling inbound SMS: %s", e)

    # Persist inbound SMS to DB (dedup on message_sid unique constraint)
    sms_id = None
    try:
        from ...storage.repositories.sms_message import get_sms_message_repo
        sms_repo = get_sms_message_repo()

        # Check for duplicate (webhook retry)
        existing = await sms_repo.get_by_message_sid(MessageSid)
        if existing:
            logger.info("Duplicate inbound SMS webhook for %s, skipping", MessageSid)
            return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response></Response>""")

        record = await sms_repo.create(
            message_sid=MessageSid,
            from_number=From,
            to_number=To,
            direction="inbound",
            body=Body,
            media_urls=media_urls,
            business_context_id=context.id,
            source="webhook",
        )
        sms_id = record["id"]
        logger.info("Inbound SMS persisted: id=%s sid=%s", sms_id, MessageSid)
    except Exception as e:
        logger.warning("Failed to persist inbound SMS %s: %s", MessageSid, e)

    # Spawn background task for CRM + intelligence + ntfy + auto-reply
    asyncio.create_task(
        _process_inbound_sms(sms_id, From, To, Body, context, media_urls)
    )

    # Return empty TwiML immediately
    return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response></Response>""")


@sms_router.post("/status")
async def handle_sms_status(
    request: Request,
    MessageSid: str = Form(...),
    MessageStatus: str = Form(...),
    To: str = Form(...),
):
    """Handle SMS delivery status updates.

    Updates the sms_messages row with the new status and delivery timestamp.
    """
    logger.info("SMS %s to %s: %s", MessageSid, To, MessageStatus)

    try:
        from ...storage.repositories.sms_message import get_sms_message_repo
        from datetime import datetime, timezone

        sms_repo = get_sms_message_repo()
        record = await sms_repo.get_by_message_sid(MessageSid)
        if record:
            if MessageStatus == "delivered":
                await sms_repo.update_delivery(record["id"], datetime.now(timezone.utc))
            else:
                error_msg = None
                if MessageStatus in ("failed", "undelivered"):
                    error_msg = f"Delivery failed: {MessageStatus}"
                await sms_repo.update_status(record["id"], MessageStatus, error_msg)
            logger.info("SMS status updated: %s -> %s", MessageSid, MessageStatus)
    except Exception as e:
        logger.warning("Failed to update SMS status for %s: %s", MessageSid, e)

    return Response(status_code=204)


# Include SMS router
router.include_router(sms_router)
