"""
Webhook endpoints for telephony providers.

Handles incoming calls, SMS, and status updates from SignalWire.
Supports both SWML (new) and LaML (legacy) formats.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse

from ...comms import comms_settings
from ...comms.context import get_context_router
from ...comms.providers import get_provider

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
        provider = get_provider()

        # Track the call
        call = await provider.handle_incoming_call(
            call_sid=call_id,
            from_number=from_number,
            to_number=to_number,
        )
        call.context_id = context.id

        logger.info("Starting AI conversation for call %s (laml=%s)", call_id, use_laml)

        if use_laml:
            # LaML: Use Atlas models via bidirectional WebSocket stream
            # Prewarm LLM in background
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

            # Return LaML - connect directly to Atlas for unified voice
            # Greeting will be played via Atlas TTS (Kokoro) when stream starts
            return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
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
        return swml_response({
            "main": [
                {"answer": {}},
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
            ]
        })

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

        # Process with ReceptionistAgent via unified interface
        from ...agents.interface import get_agent

        agent = get_agent(
            agent_type="receptionist",
            session_id=call_id,
            business_context=business_context,
        )

        agent_result = await agent.process(
            input_text=speech_text,
            input_type="voice",
            session_id=call_id,
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


@router.post("/outbound")
async def handle_outbound_call(
    request: Request,
    CallSid: str = Form(...),
    To: str = Form(...),
    From: str = Form(...),
):
    """
    Handle outbound call connection.

    Called when an outbound call is answered.
    """
    logger.info("Outbound call connected: %s to %s", CallSid, To)

    try:
        provider = get_provider()
        call = await provider.get_call(CallSid)

        if call and call.context_id:
            context_router = get_context_router()
            context = context_router.get_context(call.context_id)
            greeting = context.greeting if context else "Hello, this is Atlas."
        else:
            greeting = "Hello, this is Atlas calling."

        ws_url = comms_settings.webhook_base_url.replace(
            "https://", "wss://"
        ).replace("http://", "ws://")
        stream_url = f"{ws_url}/api/v1/comms/voice/stream/{CallSid}"

        return laml_response(f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{greeting}</Say>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>""")

    except Exception as e:
        logger.error("Error handling outbound call: %s", e)
        return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm sorry, I'm having trouble. Goodbye.</Say>
    <Hangup />
</Response>""")


@router.post("/status")
async def handle_call_status(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    Duration: Optional[str] = Form(None),
):
    """
    Handle call status updates.

    Called when call state changes (ringing, answered, completed, etc.)
    """
    logger.info("Call %s status: %s (duration: %s)", CallSid, CallStatus, Duration)

    try:
        provider = get_provider()
        await provider.handle_call_status(CallSid, CallStatus)
    except Exception as e:
        logger.error("Error handling call status: %s", e)

    return Response(status_code=204)


@router.post("/voicemail")
async def handle_voicemail(
    request: Request,
    CallSid: str = Form(...),
    RecordingUrl: str = Form(...),
    RecordingDuration: str = Form(...),
    context: str = "",
):
    """
    Handle voicemail recording completion.
    """
    logger.info(
        "Voicemail received: %s (%ss) for context %s",
        RecordingUrl,
        RecordingDuration,
        context,
    )

    # TODO: Save voicemail to database, send notification

    return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for your message. We will get back to you soon. Goodbye.</Say>
    <Hangup />
</Response>""")


@router.post("/recording-status")
async def handle_recording_status(
    request: Request,
    CallSid: str = Form(...),
    RecordingSid: str = Form(...),
    RecordingStatus: str = Form(...),
    RecordingUrl: Optional[str] = Form(None),
):
    """Handle recording status updates."""
    logger.info(
        "Recording %s for call %s: %s",
        RecordingSid,
        CallSid,
        RecordingStatus,
    )

    # TODO: Update recording status in database

    return Response(status_code=204)


@router.websocket("/stream/{call_sid}")
async def handle_audio_stream(websocket: WebSocket, call_sid: str):
    """
    Handle bidirectional audio streaming for a call.

    This WebSocket receives audio from the caller and sends AI responses back.
    Audio format: 8kHz mulaw (base64 encoded in JSON messages)

    Supports two modes:
    - Legacy: STT -> LLM -> TTS (PhoneCallProcessor)
    - PersonaPlex: Direct speech-to-speech (PersonaPlexProcessor)
    """
    use_personaplex = comms_settings.personaplex_enabled

    if use_personaplex:
        from ...comms.personaplex_processor import (
            get_personaplex_processor,
            create_personaplex_processor,
            remove_personaplex_processor,
        )
    else:
        from ...comms.phone_processor import (
            get_call_processor,
            create_call_processor,
            remove_call_processor,
        )

    await websocket.accept()
    logger.info(
        "Audio stream connected for call %s (mode=%s)",
        call_sid,
        "personaplex" if use_personaplex else "legacy"
    )

    stream_sid = None
    processor = None

    try:
        provider = get_provider()
        call = await provider.get_call(call_sid)

        # Get context - try from call first, then default to first context
        context_router = get_context_router()
        context = None

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

        # Get or create call processor based on mode
        from_number = call.from_number if call else ""
        to_number = call.to_number if call else ""

        # Track if we need to set callback after stream starts (pre-warm case)
        needs_callback_setup = False

        if use_personaplex:
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
                        # Connection failed
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
        else:
            processor = get_call_processor(call_sid)
            if processor is None:
                processor = create_call_processor(
                    call_sid=call_sid,
                    from_number=from_number,
                    to_number=to_number,
                    context_id=context.id,
                    business_context=context,
                )
                logger.info("Created legacy processor for call %s", call_sid)

        while True:
            data = await websocket.receive_json()
            event = data.get("event")

            if event == "connected":
                logger.info("Stream connected data: %s", data)

            elif event == "start":
                # SignalWire nests streamSid inside "start" object
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                logger.info("Stream started: %s (format: %s)",
                           stream_sid, start_data.get("mediaFormat", {}))
                if processor:
                    processor._state.stream_sid = stream_sid

                # For pre-warmed PersonaPlex, now set callback and drain buffer
                if use_personaplex and needs_callback_setup and processor:
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

                # Play greeting - PersonaPlex generates its own, legacy uses TTS
                if processor and context and stream_sid and not use_personaplex:
                    try:
                        # Try cached greeting first for instant playback
                        from ...comms.phone_processor import get_cached_greeting
                        greeting_audio = get_cached_greeting(context.id)

                        if greeting_audio:
                            logger.info("Using cached greeting for %s", context.id)
                        else:
                            # Fall back to synthesis
                            logger.info("Synthesizing greeting for call %s", call_sid)
                            greeting_audio = await processor.synthesize_greeting(
                                context.greeting
                            )

                        if greeting_audio:
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": greeting_audio,
                                },
                            })
                            logger.info("Greeting sent (%d bytes)", len(greeting_audio))
                        else:
                            logger.warning("No greeting audio generated")
                    except Exception as e:
                        logger.error("Failed to send greeting: %s", e, exc_info=True)
                elif use_personaplex and not needs_callback_setup:
                    logger.info("PersonaPlex greeting sent from buffer")

            elif event == "media":
                # Audio data from caller
                payload = data.get("media", {}).get("payload")
                if payload and processor:
                    # Process audio - PersonaPlex sends responses via callback
                    response_audio = await processor.process_audio_chunk(payload)

                    # Only legacy mode returns audio here
                    if not use_personaplex and response_audio and stream_sid:
                        logger.info(
                            "Sending response audio to caller (%d bytes)",
                            len(response_audio)
                        )
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": response_audio,
                            },
                        })
                        logger.info("Response audio sent successfully")
                    elif not use_personaplex and response_audio:
                        logger.warning(
                            "Response audio generated but no stream_sid"
                        )

            elif event == "stop":
                logger.info("Stream stopped for call %s", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("Audio stream disconnected for call %s", call_sid)
    except Exception as e:
        logger.error("Audio stream error for call %s: %s", call_sid, e)
    finally:
        if processor:
            if use_personaplex:
                await remove_personaplex_processor(call_sid)
            else:
                remove_call_processor(call_sid)
        logger.info("Audio stream ended for call %s", call_sid)


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
        provider = get_provider()
        message = await provider.handle_incoming_sms(
            message_sid=MessageSid,
            from_number=From,
            to_number=To,
            body=Body,
            media_urls=media_urls,
        )
        message.context_id = context.id

        # TODO: Process with LLM and send auto-reply if enabled
        if context.sms_auto_reply and context.sms_enabled:
            # Generate response using LLM
            # await send_sms_response(message, context)
            pass

    except Exception as e:
        logger.error("Error handling inbound SMS: %s", e)

    # Return empty TwiML to acknowledge
    return laml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response></Response>""")


@sms_router.post("/status")
async def handle_sms_status(
    request: Request,
    MessageSid: str = Form(...),
    MessageStatus: str = Form(...),
    To: str = Form(...),
):
    """Handle SMS delivery status updates."""
    logger.info("SMS %s to %s: %s", MessageSid, To, MessageStatus)
    return Response(status_code=204)


# Include SMS router
router.include_router(sms_router)
