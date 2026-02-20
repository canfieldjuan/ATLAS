"""
Post-call transcription and data extraction pipeline.

After a call ends, processes accumulated audio chunks:
1. Convert mulaw 8kHz -> PCM 16kHz WAV
2. Transcribe via local ASR server (Nemotron)
3. Extract structured data via LLM (qwen3:14b)
4. Store results in call_transcripts table
5. Push ntfy notification with summary
"""

import asyncio
import audioop
import json
import logging
import re
import struct
from typing import Optional
from uuid import UUID

import httpx

from ..config import settings
from ..skills import get_skill_registry
from ..storage.repositories.call_transcript import get_call_transcript_repo

logger = logging.getLogger("atlas.comms.call_intelligence")


async def process_call_recording(
    call_sid: str,
    audio_chunks: list[bytes],
    from_number: str,
    to_number: str,
    context_id: str,
    duration_seconds: int,
    ai_audio_chunks: list[bytes] | None = None,
    business_context=None,
) -> None:
    """
    Process a completed call recording through the intelligence pipeline.

    audio_chunks: caller's raw mulaw audio from SignalWire media events
    ai_audio_chunks: AI receptionist's mulaw audio from send_audio callback

    Each step is wrapped in try/except to fail-open -- partial results
    are better than no results.
    """
    cfg = settings.call_intelligence

    if not cfg.enabled:
        logger.debug("Call intelligence disabled, skipping %s", call_sid)
        return

    if duration_seconds < cfg.min_duration_seconds:
        logger.debug(
            "Call %s too short (%ds < %ds), skipping",
            call_sid, duration_seconds, cfg.min_duration_seconds,
        )
        return

    repo = get_call_transcript_repo()

    # Step 1: Create DB row
    try:
        record = await repo.create(
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number,
            context_id=context_id,
            duration=duration_seconds,
        )
        transcript_id = record["id"]
    except Exception as e:
        logger.warning("Call intelligence: failed to create record for %s: %s", call_sid, e)
        return

    # Step 2: Convert caller audio to WAV
    try:
        await repo.update_status(transcript_id, "transcribing")
        caller_wav = _convert_audio_to_wav(audio_chunks)
    except Exception as e:
        logger.warning("Call intelligence: audio conversion failed for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Audio conversion: {e}")
        return

    # Step 2b: Convert AI audio to WAV (best-effort, don't block on failure)
    ai_wav = None
    if ai_audio_chunks:
        try:
            ai_wav = _convert_audio_to_wav(ai_audio_chunks)
        except Exception as e:
            logger.debug("Call intelligence: AI audio conversion failed for %s: %s", call_sid, e)

    # Step 3: Transcribe caller audio
    try:
        caller_transcript = await _transcribe_wav(caller_wav)
    except Exception as e:
        logger.warning("Call intelligence: transcription failed for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Transcription: {e}")
        return

    # Step 3b: Transcribe AI audio (best-effort)
    ai_transcript = None
    if ai_wav:
        try:
            ai_transcript = await _transcribe_wav(ai_wav)
        except Exception as e:
            logger.debug("Call intelligence: AI transcription failed for %s: %s", call_sid, e)

    # Build combined transcript
    transcript = _build_transcript(caller_transcript, ai_transcript)
    if not transcript:
        logger.info("Call intelligence: empty transcript for %s", call_sid)
        await repo.update_transcript(transcript_id, "")
        await repo.update_extraction(
            transcript_id,
            summary="No speech detected",
            extracted_data={},
            proposed_actions=[],
        )
        await repo.update_status(transcript_id, "ready")
        return
    await repo.update_transcript(transcript_id, transcript)

    # Step 4: LLM extraction
    try:
        await repo.update_status(transcript_id, "extracting")
        summary, extracted_data, proposed_actions = await _extract_call_data(
            transcript, business_context,
        )
        await repo.update_extraction(transcript_id, summary, extracted_data, proposed_actions)
        await repo.update_status(transcript_id, "ready")
    except Exception as e:
        logger.warning("Call intelligence: extraction failed for %s: %s", call_sid, e)
        await _safe_update_status(repo, transcript_id, "error", f"Extraction: {e}")
        return

    # Step 5: Notify
    try:
        await _notify_call_summary(
            repo, transcript_id, call_sid,
            from_number, duration_seconds,
            summary, extracted_data, proposed_actions,
            business_context,
        )
    except Exception as e:
        logger.warning("Call intelligence: notification failed for %s: %s", call_sid, e)


def _convert_audio_to_wav(audio_chunks: list[bytes]) -> bytes:
    """Convert mulaw 8kHz audio chunks to a 16kHz 16-bit mono WAV."""
    raw_mulaw = b"".join(audio_chunks)
    if not raw_mulaw:
        raise ValueError("No audio data")

    # mulaw -> PCM 16-bit at 8kHz
    pcm_8k = audioop.ulaw2lin(raw_mulaw, 2)

    # Resample 8kHz -> 16kHz
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

    # Build WAV file
    data_size = len(pcm_16k)
    sample_rate = 16000
    bits_per_sample = 16
    channels = 1
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,             # fmt chunk size
        1,              # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_16k


def _build_transcript(
    caller_transcript: Optional[str],
    ai_transcript: Optional[str],
) -> Optional[str]:
    """Combine caller and AI transcripts into a labeled conversation.

    Note: Each side is transcribed as a single block, so turn-by-turn
    ordering is lost. The LLM extraction prompt handles this gracefully.
    """
    if not caller_transcript and not ai_transcript:
        return None
    if not ai_transcript:
        return caller_transcript
    if not caller_transcript:
        return f"[AI Receptionist]: {ai_transcript}"
    return f"[Caller]: {caller_transcript}\n\n[AI Receptionist]: {ai_transcript}"


async def _transcribe_wav(wav_bytes: bytes) -> Optional[str]:
    """Send WAV to the local ASR server and return transcript text."""
    cfg = settings.call_intelligence

    async with httpx.AsyncClient(timeout=cfg.asr_timeout) as client:
        resp = await client.post(
            cfg.asr_url,
            files={"file": ("call.wav", wav_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        result = resp.json()

    text = result.get("text", "").strip()
    return text if text else None


async def _extract_call_data(
    transcript: str,
    business_context=None,
) -> tuple[str, dict, list]:
    """Use LLM to extract structured data from the transcript."""
    from ..services import llm_registry
    from ..services.protocols import Message

    llm = llm_registry.get_active()
    if not llm:
        logger.warning("No active LLM for call extraction")
        return transcript[:200], {}, []

    # Build business context string
    ctx_parts = []
    if business_context:
        if hasattr(business_context, "name"):
            ctx_parts.append(f"Business: {business_context.name}")
        if hasattr(business_context, "business_type") and business_context.business_type:
            ctx_parts.append(f"Type: {business_context.business_type}")
        if hasattr(business_context, "services") and business_context.services:
            ctx_parts.append(f"Services: {', '.join(business_context.services)}")
    business_context_str = "\n".join(ctx_parts) if ctx_parts else "General business"

    # Load skill prompt
    skill = get_skill_registry().get("call/call_extraction")
    if skill:
        system_prompt = skill.content.replace("{business_context}", business_context_str)
    else:
        system_prompt = (
            "Extract customer info, intent, and proposed actions from this call transcript. "
            f"Business context: {business_context_str}\n"
            "Return a JSON object with: customer_name, customer_phone, customer_email, "
            "intent, services_mentioned, address, preferred_date, preferred_time, "
            "urgency, follow_up_needed, notes. "
            "Then a blank line and a JSON array of proposed actions."
        )

    cfg = settings.call_intelligence
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=transcript),
    ]

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ),
        ),
        timeout=30.0,
    )

    text = result.get("response", "").strip()
    if not text:
        return transcript[:200], {}, []

    # Strip <think> tags (Qwen3 models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return _parse_extraction(text, transcript)


def _parse_extraction(text: str, transcript: str) -> tuple[str, dict, list]:
    """Parse LLM output into (summary, extracted_data, proposed_actions)."""
    # Try splitting into two JSON blocks (data + actions)
    extracted_data = {}
    proposed_actions = []

    # Find the first JSON object
    obj_start = text.find("{")
    obj_end = _find_matching_brace(text, obj_start) if obj_start >= 0 else -1

    if obj_start >= 0 and obj_end > obj_start:
        try:
            extracted_data = json.loads(text[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass

        # Find the actions array after the object
        remainder = text[obj_end + 1 :].strip()
        arr_start = remainder.find("[")
        arr_end = remainder.rfind("]")

        if arr_start >= 0 and arr_end > arr_start:
            try:
                proposed_actions = json.loads(remainder[arr_start : arr_end + 1])
            except json.JSONDecodeError:
                pass

    # Build summary from extracted data
    parts = []
    name = extracted_data.get("customer_name")
    if name:
        parts.append(f"Customer: {name}")
    intent = extracted_data.get("intent")
    if intent:
        parts.append(f"Intent: {intent.replace('_', ' ')}")
    services = extracted_data.get("services_mentioned")
    if services:
        parts.append(f"Services: {', '.join(services)}")
    notes = extracted_data.get("notes")
    if notes:
        parts.append(notes)

    summary = ". ".join(parts) if parts else transcript[:200]

    return summary, extracted_data, proposed_actions


def _find_matching_brace(text: str, start: int) -> int:
    """Find the closing brace matching the opening brace at `start`."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


async def _notify_call_summary(
    repo,
    transcript_id: UUID,
    call_sid: str,
    from_number: str,
    duration_seconds: int,
    summary: str,
    extracted_data: dict,
    proposed_actions: list,
    business_context=None,
) -> None:
    """Send ntfy notification with the call summary."""
    if not settings.call_intelligence.notify_enabled:
        return
    if not settings.alerts.ntfy_enabled:
        return
    if not settings.alerts.ntfy_url or not settings.alerts.ntfy_topic:
        return

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"

    # Format duration
    mins, secs = divmod(duration_seconds, 60)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    # Build message
    biz_name = business_context.name if business_context and hasattr(business_context, "name") else "Business"
    lines = [f"From: {from_number} ({dur_str})"]

    name = extracted_data.get("customer_name")
    if name:
        lines.append(f"Customer: {name}")

    intent = extracted_data.get("intent")
    if intent:
        lines.append(f"Intent: {intent.replace('_', ' ').title()}")

    services = extracted_data.get("services_mentioned")
    if services:
        lines.append(f"Services: {', '.join(services)}")

    if proposed_actions:
        actions_str = "; ".join(
            a.get("label", a.get("type", "")) for a in proposed_actions
            if a.get("type") != "none"
        )
        if actions_str:
            lines.append(f"Follow-up: {actions_str}")

    message = "\n".join(lines)

    headers = {
        "Title": f"{biz_name}: Call Summary",
        "Priority": "default",
        "Tags": "phone,call",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
        await repo.mark_notified(transcript_id)
        await repo.update_status(transcript_id, "notified")
        logger.info("Call summary notification sent for %s", call_sid)
    except Exception as e:
        logger.warning("Failed to send call summary notification: %s", e)


async def _safe_update_status(repo, transcript_id, status, error_message=None):
    """Update status without raising on failure."""
    try:
        await repo.update_status(transcript_id, status, error_message)
    except Exception as e:
        logger.warning("Failed to update status for %s: %s", transcript_id, e)
