"""
Voice pipeline launcher for Atlas Brain.

Integrates the voice pipeline with AtlasAgent.
"""

import asyncio
import concurrent.futures
import logging
import re
import signal
import sys
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from ..agents.interface import get_agent, process_with_fallback
from ..agents.graphs.workflow_state import get_workflow_state_manager
from ..config import settings
from .pipeline import (
    ErrorPhrase,
    NemotronAsrHttpClient,
    NemotronAsrStreamingClient,
    PiperTTS,
    SentenceBuffer,
    VoicePipeline,
)

logger = logging.getLogger("atlas.voice.launcher")

# Minimum facts to retrieve when an entity is detected via graph traversal
_ENTITY_MIN_FACTS = 5

_voice_pipeline: Optional[VoicePipeline] = None
_voice_thread: Optional[threading.Thread] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_free_mode_manager = None  # FreeModeManager instance if enabled

# Early preparation cache for overlapped prefill during conversation silence
import time as _time

_early_prep_cache: dict = {}
_early_prep_lock = threading.Lock()


def _notify_ui(state: str, **kwargs) -> None:
    """Broadcast voice pipeline state to connected UI WebSocket sessions."""
    if _event_loop is None:
        return
    try:
        from ..api.orchestrated.websocket import broadcast_from_thread
        broadcast_from_thread(_event_loop, state, **kwargs)
    except Exception as e:
        logger.debug("UI broadcast failed: %s", e)


def _create_agent_runner():
    """Create a sync wrapper for the async agent via unified interface."""
    agent = get_agent("atlas")

    def runner(transcript: str, context_dict: Dict[str, Any]) -> str:
        """Run the agent synchronously."""
        if _event_loop is None:
            logger.error("No event loop available for agent runner")
            return ""

        _notify_ui("transcript", text=transcript)
        _notify_ui("processing")

        session_id = context_dict.get("session_id")
        node_id = context_dict.get("node_id")
        speaker_id = context_dict.get("speaker_id")
        speaker_name = context_dict.get("speaker_name")

        try:
            future = asyncio.run_coroutine_threadsafe(
                agent.process(
                    input_text=transcript,
                    session_id=session_id,
                    speaker_id=speaker_name,
                    input_type="voice",
                    runtime_context={"node_id": node_id, "speaker_uuid": speaker_id},
                ),
                _event_loop,
            )
            result = future.result(timeout=settings.voice.agent_timeout)
            _signal_workflow_state(result)
            response = result.response_text or ""
            logger.info("Agent runner response: %s", response[:100] if response else "(empty)")
            _notify_ui("response", text=response, audio_base64="")
            _notify_ui("idle")
            return response
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.error("Agent runner timed out after %.1fs", settings.voice.agent_timeout)
            return ErrorPhrase(settings.voice.error_agent_timeout)
        except Exception as e:
            logger.error("Agent runner failed: %s", e, exc_info=True)
            return ErrorPhrase(settings.voice.error_agent_failed)

    return runner


def _create_streaming_agent_runner():
    """Create a streaming agent runner that yields sentences to TTS."""
    from ..services import llm_registry
    from ..services.protocols import Message
    from ..services.intent_router import route_query
    from ..config import settings

    # Store last route result for intent gating
    _last_route_result = {"result": None}

    def runner(
        transcript: str,
        context_dict: Dict[str, Any],
        on_sentence: Callable[[str], None],
    ) -> None:
        """
        Run agent with streaming LLM for conversations, regular agent for tools.

        Args:
            transcript: User input text
            context_dict: Session context
            on_sentence: Callback for each complete sentence
        """
        if _event_loop is None:
            logger.error("No event loop for streaming agent")
            return

        _notify_ui("transcript", text=transcript)
        _notify_ui("processing")

        # Track sentences emitted so we can discard partial output on fallback
        _emitted = []

        def _tracked_on_sentence(sentence: str) -> None:
            _emitted.append(sentence)
            on_sentence(sentence)

        # First, check intent to decide streaming vs regular path
        async def check_and_run():
            use_streaming = False
            route_result = None
            has_active_workflow = False
            _classify_ms: float = 0.0

            # Check for active workflow - if present, must use agent path
            session_id = context_dict.get("session_id")
            if session_id:
                manager = get_workflow_state_manager()
                workflow = await manager.restore_workflow_state(session_id)
                if workflow is not None:
                    has_active_workflow = True
                    logger.info(
                        "Active workflow detected: %s, forcing agent path",
                        workflow.workflow_type
                    )

            # Check early preparation cache (from conversation silence)
            cached_prep = None
            with _early_prep_lock:
                if _early_prep_cache and _time.monotonic() - _early_prep_cache.get("ts", 0) < 10:
                    cached_partial = _early_prep_cache.get("partial", "")
                    # Reuse if final transcript is similar to cached partial
                    if (transcript.lower().startswith(cached_partial.lower()[:20])
                            or cached_partial.lower().startswith(transcript.lower()[:20])):
                        cached_prep = _early_prep_cache.copy()
                        logger.info("Using early prep cache (partial=%s)", cached_partial[:30])
                _early_prep_cache.clear()  # Always clear after check

            if cached_prep and not has_active_workflow:
                route_result = cached_prep["route_result"]
                _last_route_result["result"] = route_result
                threshold = settings.intent_router.confidence_threshold
                if route_result.action_category == "conversation" and route_result.confidence >= threshold:
                    use_streaming = True
                    logger.info(
                        "Streaming path: conversation via early prep (conf=%.2f)",
                        route_result.confidence
                    )
                elif route_result.action_category != "conversation":
                    logger.info(
                        "Non-streaming path via early prep: %s/%s (conf=%.2f)",
                        route_result.action_category,
                        route_result.tool_name or "none",
                        route_result.confidence
                    )
                # Prefill already triggered by early silence

                # Conversation mode workflow breakout: if cached prep returned
                # "conversation" but the full transcript might be a workflow trigger,
                # re-classify fresh to avoid masking workflow routes.
                if (use_streaming
                        and _voice_pipeline is not None
                        and _voice_pipeline.frame_processor.state == "conversing"):
                    from ..services.intent_router import ROUTE_TO_WORKFLOW
                    fresh_route = await route_query(transcript)
                    if fresh_route.raw_label in ROUTE_TO_WORKFLOW:
                        route_result = fresh_route
                        _last_route_result["result"] = route_result
                        use_streaming = False
                        logger.info(
                            "Workflow breakout: %s detected over cached conversation "
                            "(conf=%.2f)",
                            fresh_route.raw_label, fresh_route.confidence,
                        )
            elif settings.intent_router.enabled and not has_active_workflow:
                _classify_t0 = _time.perf_counter()
                route_result = await route_query(transcript)
                _classify_ms = (_time.perf_counter() - _classify_t0) * 1000
                _last_route_result["result"] = route_result
                threshold = settings.intent_router.confidence_threshold

                if route_result.action_category == "conversation":
                    if route_result.confidence >= threshold:
                        use_streaming = True
                        logger.info(
                            "Streaming path: conversation (conf=%.2f)",
                            route_result.confidence
                        )
                    # Conversation needs LLM -- trigger prefill now
                    if _voice_pipeline is not None:
                        _voice_pipeline.trigger_prefill()
                    # Note: if the semantic router misclassifies a workflow
                    # command as "conversation" here, the streaming path has
                    # no tool access, so no unauthorized actions occur -- the
                    # user just gets a wrong text reply. The fix for this is
                    # improving the router's training data, not re-routing.
                else:
                    logger.info(
                        "Non-streaming path: %s/%s (conf=%.2f, skipping prefill)",
                        route_result.action_category,
                        route_result.tool_name or "none",
                        route_result.confidence
                    )
            elif has_active_workflow:
                # Still run intent router for gating, but don't use streaming
                if settings.intent_router.enabled:
                    route_result = await route_query(transcript)
                    _last_route_result["result"] = route_result

                    # If the new route is a different workflow or a clear
                    # non-conversation intent, the user is starting something
                    # new — clear the stale workflow so it doesn't hijack.
                    from ..services.intent_router import ROUTE_TO_WORKFLOW
                    new_wf = ROUTE_TO_WORKFLOW.get(route_result.raw_label)
                    if (route_result.confidence >= settings.intent_router.confidence_threshold
                            and route_result.action_category != "conversation"
                            and new_wf != workflow.workflow_type):
                        logger.info(
                            "New intent %s (conf=%.2f) overrides active %s workflow — clearing",
                            route_result.raw_label, route_result.confidence,
                            workflow.workflow_type,
                        )
                        await manager.clear_workflow_state(session_id)
                        has_active_workflow = False

            # Conversation mode: use agent path for full tool access.
            # Streaming has no tools — the LLM can't search emails, check
            # calendars, or execute any MCP tools in streaming mode.
            if (use_streaming
                    and settings.voice.conversation_agent_enabled
                    and _voice_pipeline is not None
                    and _voice_pipeline.frame_processor.state == "conversing"):
                use_streaming = False
                logger.info("Conversation mode: using agent path for tool access")

            if use_streaming:
                _entity = getattr(route_result, "entity_name", None) if route_result else None
                success = await _stream_llm_response(
                    transcript, context_dict, _tracked_on_sentence,
                    cached_mem_ctx=cached_prep.get("mem_ctx") if cached_prep else None,
                    entity_name=_entity,
                    classify_ms=_classify_ms,
                )
                if not success:
                    # Discard partial sentences before running fallback agent
                    if _emitted:
                        logger.info(
                            "Discarding %d partial sentences before fallback",
                            len(_emitted),
                        )
                        on_sentence(None)  # sentinel: clear accumulated sentences
                        _emitted.clear()
                    await _run_agent_fallback(transcript, context_dict, _tracked_on_sentence, pre_route_result=route_result)
            else:
                await _run_agent_fallback(transcript, context_dict, _tracked_on_sentence, pre_route_result=route_result)

            # Check intent gating after processing
            if route_result and settings.voice_filter.intent_gating_enabled:
                _check_intent_gating(route_result)

        try:
            future = asyncio.run_coroutine_threadsafe(check_and_run(), _event_loop)
            future.result(timeout=settings.voice.agent_timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.error("Streaming agent runner timed out after %.1fs", settings.voice.agent_timeout)
            on_sentence(ErrorPhrase(settings.voice.error_agent_timeout))
        except Exception as e:
            logger.error("Streaming agent runner failed: %s", e)
            on_sentence(ErrorPhrase(settings.voice.error_agent_failed))

        # Broadcast full response to UI
        if _emitted:
            full_response = " ".join(s for s in _emitted if s)
            _notify_ui("response", text=full_response, audio_base64="")
            _notify_ui("idle")

    return runner


def _check_intent_gating(route_result) -> None:
    """Check intent confidence and exit conversation mode if needed.

    Called after processing a command to determine if conversation should continue.
    If confidence is below threshold or category is not in the continuation list,
    exits conversation mode.

    Args:
        route_result: Result from intent router
    """
    from ..config import settings

    if not settings.voice_filter.intent_gating_enabled:
        return

    # In conversation mode, let natural exit mechanisms handle termination
    # (silence timeout, turn limit) rather than gating on intent confidence.
    if (_voice_pipeline is not None
            and _voice_pipeline.frame_processor.state == "conversing"):
        logger.debug(
            "Intent gating: skipping in conversation mode (category=%s, conf=%.2f)",
            route_result.action_category, route_result.confidence
        )
        return

    filter_cfg = settings.voice_filter
    confidence = route_result.confidence
    category = route_result.action_category

    # Check if category allows continuation
    categories_continue = filter_cfg.intent_categories_continue
    category_ok = category in categories_continue

    # Check if confidence is above threshold
    confidence_ok = confidence >= filter_cfg.intent_continuation_threshold

    if not category_ok or not confidence_ok:
        logger.info(
            "Intent gating: exiting conversation (category=%s ok=%s, conf=%.2f ok=%s)",
            category, category_ok, confidence, confidence_ok
        )
        # Exit conversation mode
        if _voice_pipeline is not None:
            _voice_pipeline.frame_processor.exit_conversation_mode("intent_gating")
    else:
        logger.debug(
            "Intent gating: continuing conversation (category=%s, conf=%.2f)",
            category, confidence
        )


async def _stream_llm_response(
    transcript: str,
    context_dict: Dict[str, Any],
    on_sentence: Callable[[str], None],
    cached_mem_ctx: Optional[Any] = None,
    entity_name: Optional[str] = None,
    classify_ms: float = 0.0,
) -> bool:
    """Stream LLM response for conversation queries.

    Returns True if successful, False if fallback needed.
    """
    from ..services import llm_registry
    from ..services.protocols import Message
    from ..memory.service import get_memory_service

    llm = llm_registry.get_active()
    if llm is None or not hasattr(llm, "chat_stream_async"):
        logger.warning("LLM does not support streaming")
        return False

    buffer = SentenceBuffer()

    # Gather unified context via MemoryService (history + profile + token budget)
    session_id = context_dict.get("session_id")
    speaker_name = context_dict.get("speaker_name")

    # FTL trace span — covers entire streaming turn (all turns, not just first)
    from ..services.tracing import tracer as _tracer
    _stream_span = _tracer.start_span(
        span_name="agent.process",
        operation_type="llm_call",
        model_name=getattr(llm, "model", None),
        model_provider="ollama",
        session_id=session_id,
        metadata={"input_type": "voice", "speaker_id": speaker_name, "action_type": "conversation"},
    )
    _stream_stats: dict = {}
    _span_start = _time.perf_counter()

    # Pre-pop previous turn's stashed RAG usage_ids before gather_context
    # can overwrite the stash with this turn's sources.
    prev_usage_ids: list = []
    if session_id:
        try:
            from ..memory.feedback import get_feedback_service
            prev_usage_ids = get_feedback_service().pop_session_usage(session_id)
        except Exception as e:
            logger.debug("Pop session usage failed: %s", e)

    svc = get_memory_service()
    _memory_t0 = _time.perf_counter()
    if cached_mem_ctx is not None:
        mem_ctx = cached_mem_ctx
        logger.info("Using cached context from early prep")
    else:
        # Run entity-aware search (parallel vector + graph traversal)
        pre_fetched = None
        try:
            from ..memory.rag_client import get_rag_client
            client = get_rag_client()
            max_facts = settings.memory.context_results
            if entity_name:
                logger.info("Voice entity detected: %r -- running parallel search + traversal", entity_name)
                max_facts = max(max_facts, _ENTITY_MIN_FACTS)
            result = await client.search_with_traversal(
                query=transcript,
                entity_name=entity_name,
                max_facts=max_facts,
            )
            pre_fetched = result.facts
        except Exception as e:
            logger.warning("Voice entity search failed: %s", e)

        mem_ctx = await svc.gather_context(
            query=transcript,
            session_id=session_id,
            user_id=context_dict.get("speaker_id"),
            include_rag=True,
            pre_fetched_sources=pre_fetched,
            include_history=True,
            include_physical=False,
            max_history=6,
        )
    _memory_ms = (_time.perf_counter() - _memory_t0) * 1000

    # Stash this turn's RAG usage_ids for the next turn's correction detection
    if mem_ctx.feedback_context and mem_ctx.feedback_context.usage_ids and session_id:
        try:
            from ..memory.feedback import get_feedback_service
            get_feedback_service().stash_session_usage(
                session_id, mem_ctx.feedback_context.usage_ids,
            )
        except Exception as e:
            logger.debug("Stash session usage failed: %s", e)

    # Build system prompt with physical awareness context
    prompt_parts = [_get_system_prompt()]
    try:
        from ..orchestration.context import get_context
        awareness = get_context().build_context_string()
        if awareness:
            prompt_parts.append(f"\nCurrent awareness:\n{awareness}")
    except Exception as e:
        logger.debug("Physical context unavailable: %s", e)

    # Append learned temporal patterns (routines)
    try:
        from ..orchestration.temporal import get_temporal_context
        temporal = await get_temporal_context()
        if temporal:
            prompt_parts.append(temporal)
    except Exception as e:
        logger.debug("Temporal context unavailable: %s", e)

    # Add user profile from MemoryService context
    if mem_ctx.user_name:
        prompt_parts.append(f"The user's name is {mem_ctx.user_name}.")
    if mem_ctx.response_style == "brief":
        prompt_parts.append("Preference: Keep responses short and concise.")
    elif mem_ctx.response_style == "detailed":
        prompt_parts.append("Preference: Provide thorough explanations.")
    if mem_ctx.expertise_level == "beginner":
        prompt_parts.append("Level: Explain concepts simply.")
    elif mem_ctx.expertise_level == "expert":
        prompt_parts.append("Level: Use technical language freely.")

    if speaker_name and speaker_name != "unknown":
        prompt_parts.append(f"The speaker is {speaker_name}.")

    # Add RAG context from GraphRAG knowledge graph
    if mem_ctx.rag_context_used and mem_ctx.rag_result and mem_ctx.rag_result.sources:
        entity_facts = []
        search_facts = []
        for s in mem_ctx.rag_result.sources:
            if not s.fact:
                continue
            if s.source_type == "entity_edge":
                entity_facts.append(s.fact)
            else:
                search_facts.append(s.fact)

        if entity_facts:
            label = f"Known facts about {entity_name}" if entity_name else "Known facts"
            prompt_parts.append(f"\n{label}:\n" + "\n".join(f"- {f}" for f in entity_facts))
        if search_facts:
            prompt_parts.append("\nRelated context:\n" + "\n".join(f"- {f}" for f in search_facts))

    # Inject entity context (recent turns + current room for disambiguation)
    try:
        from .entity_context import collect_recent_entities, format_entity_context
        entity_dicts = list(mem_ctx.recent_entities)
        try:
            from ..orchestration.context import get_context
            _room = get_context().build_context_dict().get("location")
            if _room:
                entity_dicts.insert(0, {
                    "type": "location",
                    "name": _room,
                    "action": "current room",
                    "source": "context",
                })
        except Exception:
            pass
        if entity_dicts:
            refs = collect_recent_entities(entity_dicts)
            entity_str = format_entity_context(refs)
            if entity_str:
                prompt_parts.append(f"\n{entity_str}")
    except Exception as e:
        logger.debug("Entity context injection failed: %s", e)

    messages = [Message(role="system", content=" ".join(prompt_parts))]

    # History from MemoryContext as separate messages (already chronological order)
    for h in mem_ctx.conversation_history:
        messages.append(Message(role=h["role"], content=h["content"]))
    if mem_ctx.conversation_history:
        logger.info("Added %d conversation turns to streaming context", len(mem_ctx.conversation_history))

    # Add current user message
    messages.append(Message(role="user", content=transcript))

    sentence_count = 0
    collected_sentences = []
    max_tokens = settings.voice.streaming_max_tokens
    try:
        _loop_t0 = _time.perf_counter()
        async for token in llm.chat_stream_async(messages, max_tokens=max_tokens, stats=_stream_stats):
            sentence = buffer.add_token(token)
            if sentence:
                sentence_count += 1
                collected_sentences.append(sentence)
                logger.info("Streaming sentence ready: %s", sentence[:60])
                on_sentence(sentence)
        remaining = buffer.flush()
        if remaining:
            sentence_count += 1
            collected_sentences.append(remaining)
            logger.info("Streaming final chunk: %s", remaining[:60])
            on_sentence(remaining)
        _loop_ms = (_time.perf_counter() - _loop_t0) * 1000

        # Persist conversation turns in background (don't block TTS)
        if sentence_count > 0 and session_id:
            full_response = " ".join(collected_sentences)
            speaker_uuid = context_dict.get("speaker_id")  # actual UUID
            _total_ms = (_time.perf_counter() - _span_start) * 1000
            _tracer.end_span(
                _stream_span,
                status="completed",
                input_tokens=_stream_stats.get("input_tokens"),
                output_tokens=_stream_stats.get("output_tokens"),
                input_data={"user_message": transcript, "input_type": "voice"},
                output_data={"response": full_response[:500], "action_type": "conversation"},
                ttft_ms=_stream_stats.get("prompt_eval_duration_ms"),
                inference_time_ms=_stream_stats.get("eval_duration_ms"),
                metadata={
                    "timing": {
                        "classify": round(classify_ms, 1),
                        "memory": round(_memory_ms, 1),
                        "respond": round(_loop_ms, 1),
                        "total": round(_total_ms, 1),
                    },
                    "action_type": "conversation",
                    "speaker_id": speaker_name,
                    "llm": {
                        "has_llm_response": True,
                        "input_tokens": _stream_stats.get("input_tokens"),
                        "output_tokens": _stream_stats.get("output_tokens"),
                        "eval_duration_ms": _stream_stats.get("eval_duration_ms"),
                        "total_duration_ms": _stream_stats.get("total_duration_ms"),
                    },
                },
            )
            # Emit phase-level child spans to match LangGraph trace structure
            try:
                _span_dt = datetime.fromisoformat(_stream_span.start_iso)
                _offset = 0.0
                if classify_ms > 0:
                    _cs = _span_dt + timedelta(milliseconds=_offset)
                    _tracer.emit_child_span(
                        parent=_stream_span,
                        span_name="agent.classify",
                        operation_type="classification",
                        start_iso=_cs.isoformat(),
                        end_iso=(_cs + timedelta(milliseconds=classify_ms)).isoformat(),
                        duration_ms=classify_ms,
                        metadata={"phase": "classify", "action_type": "conversation"},
                    )
                    _offset += classify_ms
                if _memory_ms > 0:
                    _ms = _span_dt + timedelta(milliseconds=_offset)
                    _tracer.emit_child_span(
                        parent=_stream_span,
                        span_name="agent.memory",
                        operation_type="retrieval",
                        start_iso=_ms.isoformat(),
                        end_iso=(_ms + timedelta(milliseconds=_memory_ms)).isoformat(),
                        duration_ms=_memory_ms,
                        metadata={"phase": "memory", "action_type": "conversation"},
                    )
                    _offset += _memory_ms
                _rs = _span_dt + timedelta(milliseconds=_offset)
                _tracer.emit_child_span(
                    parent=_stream_span,
                    span_name="agent.respond",
                    operation_type="llm_call",
                    start_iso=_rs.isoformat(),
                    end_iso=(_rs + timedelta(milliseconds=_loop_ms)).isoformat(),
                    duration_ms=_loop_ms,
                    input_tokens=_stream_stats.get("input_tokens"),
                    output_tokens=_stream_stats.get("output_tokens"),
                    metadata={
                        "phase": "respond",
                        "action_type": "conversation",
                        "ttft_ms": round(_stream_stats.get("prompt_eval_duration_ms") or 0),
                        "inference_time_ms": round(_stream_stats.get("eval_duration_ms") or 0),
                    },
                )
            except Exception:
                pass  # never crash pipeline for child span emission
            asyncio.ensure_future(_persist_streaming_turns(
                session_id, transcript, full_response, speaker_name,
                speaker_uuid=speaker_uuid,
                prev_usage_ids=prev_usage_ids,
            ))
            return True

        # Streaming produced no output -- re-stash prev_usage_ids so the
        # fallback agent's pre-pop can find them for correction feedback.
        _tracer.end_span(_stream_span, status="failed", error_message="streaming LLM produced no output")
        if prev_usage_ids and session_id:
            try:
                from ..memory.feedback import get_feedback_service
                get_feedback_service().stash_session_usage(session_id, prev_usage_ids)
            except Exception as e:
                logger.debug("Re-stash usage_ids failed: %s", e)
        return False
    except Exception as e:
        logger.error("Streaming LLM error: %s", e)
        _tracer.end_span(_stream_span, status="failed", error_message=str(e), error_type=type(e).__name__)
        # Re-stash prev_usage_ids so fallback agent can use them
        if prev_usage_ids and session_id:
            try:
                from ..memory.feedback import get_feedback_service
                get_feedback_service().stash_session_usage(session_id, prev_usage_ids)
            except Exception as e2:
                logger.debug("Re-stash usage_ids failed: %s", e2)
        return False


async def _persist_streaming_turns(
    session_id: str,
    user_text: str,
    assistant_text: str,
    speaker_name: Optional[str] = None,
    speaker_uuid: Optional[str] = None,
    prev_usage_ids: Optional[list] = None,
) -> None:
    """Persist conversation turns from streaming LLM to database and GraphRAG."""
    from ..memory.service import get_memory_service

    # Detect memory quality signals (fail-open)
    user_metadata: Optional[dict] = None
    assistant_metadata: Optional[dict] = None
    try:
        from ..memory.quality import get_quality_detector
        signal = get_quality_detector().detect(
            session_id=session_id,
            user_content=user_text,
            turn_type="conversation",
        )
        user_metadata = signal.to_metadata() or None
        if signal.correction:
            assistant_metadata = {"memory_quality": {"preceded_by_correction": True}}
            # Downvote RAG sources from the turn being corrected.
            # Uses pre-popped ids passed from _stream_llm_response to avoid
            # the timing bug where gather_context overwrites the stash.
            if prev_usage_ids:
                try:
                    from ..memory.feedback import get_feedback_service
                    await get_feedback_service().record_not_helpful(
                        prev_usage_ids, feedback_type="correction",
                    )
                    logger.info(
                        "Downvoted %d RAG sources on correction for session %s",
                        len(prev_usage_ids), session_id,
                    )
                except Exception as e:
                    logger.debug("Correction feedback failed: %s", e)
    except Exception as e:
        logger.debug("Quality detection skipped in streaming: %s", e)

    # Store entity context for streaming turns (no intent object available)
    # Extract location from transcript then response text; extract topic from transcript.
    try:
        from .entity_context import extract_location_from_text, extract_topic_from_text
        for _text in (user_text, assistant_text):
            if _text:
                _loc = extract_location_from_text(_text)
                if _loc:
                    if assistant_metadata is None:
                        assistant_metadata = {}
                    assistant_metadata.setdefault("entities", []).append(
                        {"type": "location", "name": _loc, "source": "text"}
                    )
                    break  # one location per turn is enough
        if user_text:
            _topic = extract_topic_from_text(user_text)
            if _topic:
                if assistant_metadata is None:
                    assistant_metadata = {}
                assistant_metadata.setdefault("entities", []).append(
                    {"type": "topic", "name": _topic, "source": "text"}
                )
    except Exception:
        pass

    if speaker_name:
        if assistant_metadata is None:
            assistant_metadata = {}
        assistant_metadata.setdefault("entities", []).append(
            {"type": "person", "name": speaker_name, "source": "speaker"}
        )

    try:
        svc = get_memory_service()
        await svc.store_conversation(
            session_id=session_id,
            user_content=user_text,
            assistant_content=assistant_text,
            speaker_id=speaker_name,
            speaker_uuid=speaker_uuid,
            turn_type="conversation",
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
        )
        logger.debug("Persisted streaming conversation turns for session %s", session_id)
    except Exception as e:
        logger.warning("Failed to persist streaming turns: %s", e)


def _signal_workflow_state(result) -> None:
    """Adjust voice pipeline segmenter thresholds based on workflow state."""
    if _voice_pipeline is None:
        return
    awaiting = result.metadata.get("awaiting_user_input", False)
    if awaiting:
        _voice_pipeline.set_workflow_active()
    else:
        _voice_pipeline.clear_workflow_active()


async def _run_agent_fallback(
    transcript: str,
    context_dict: Dict[str, Any],
    on_sentence: Callable[[str], None],
    pre_route_result: Optional[Any] = None,
) -> None:
    """Run regular agent for tool/device queries via unified interface."""
    session_id = context_dict.get("session_id")
    node_id = context_dict.get("node_id")
    speaker_id = context_dict.get("speaker_id")
    speaker_name = context_dict.get("speaker_name")
    runtime_ctx: Dict[str, Any] = {"node_id": node_id, "speaker_uuid": speaker_id}
    if (_voice_pipeline is not None
            and _voice_pipeline.frame_processor.state == "conversing"):
        runtime_ctx["in_conversation_mode"] = True
        runtime_ctx["conversation_max_tokens"] = settings.voice.conversation_agent_max_tokens
    if pre_route_result is not None:
        runtime_ctx["pre_route_result"] = pre_route_result
    try:
        result = await process_with_fallback(
            input_text=transcript,
            agent_type="atlas",
            session_id=session_id,
            speaker_id=speaker_name,
            input_type="voice",
            runtime_context=runtime_ctx,
        )
        _signal_workflow_state(result)
        if result.response_text:
            sentences = re.split(r'(?<=[.!?])\s+', result.response_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    on_sentence(sentence)
    except Exception as e:
        logger.error("Agent fallback failed: %s", e)
        on_sentence(ErrorPhrase(settings.voice.error_agent_failed))


def _get_system_prompt() -> str:
    """Get system prompt from centralized persona config."""
    return settings.persona.system_prompt


def _create_prefill_runner():
    """Create a prefill runner to warm up LLM KV cache on wake word detection."""
    from ..services import llm_registry
    from ..services.protocols import Message
    import time

    def runner() -> None:
        """Prefill the LLM system prompt."""
        start_time = time.perf_counter()
        logger.info("LLM prefill STARTING...")

        if _event_loop is None:
            logger.warning("No event loop available for prefill")
            return

        llm = llm_registry.get_active()
        if llm is None:
            logger.warning("No active LLM for prefill")
            return

        # Check if LLM supports prefill
        if not hasattr(llm, "prefill_async"):
            logger.warning("LLM does not support prefill_async")
            return

        messages = [Message(role="system", content=_get_system_prompt())]
        logger.info("Prefill sending system prompt (%d chars)", len(_get_system_prompt()))

        try:
            future = asyncio.run_coroutine_threadsafe(
                llm.prefill_async(messages),
                _event_loop,
            )
            result = future.result(timeout=settings.voice.prefill_timeout)
            prefill_ms = result.get("prefill_time_ms", 0)
            prompt_tokens = result.get("prompt_tokens", 0)
            total_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM prefill DONE: ollama_time=%.1fms, total=%.1fms, tokens=%d",
                prefill_ms, total_ms, prompt_tokens
            )
        except Exception as e:
            logger.warning("Prefill failed after %.1fms: %s",
                          (time.perf_counter() - start_time) * 1000, e)

    return runner


def _create_early_prep_runner():
    """Create runner that pre-gathers context during conversation silence."""

    def runner(partial_transcript: str, session_id: Optional[str] = None):
        if not partial_transcript or _event_loop is None:
            return

        async def _gather():
            from ..memory.service import get_memory_service
            from ..services.intent_router import route_query

            # Intent routing on partial transcript
            route_result = await route_query(partial_transcript)

            # Entity-aware search (parallel vector + graph traversal)
            pre_fetched = None
            try:
                from ..memory.rag_client import get_rag_client
                _entity = getattr(route_result, "entity_name", None)
                if _entity:
                    logger.info("Early prep entity detected: %r -- running parallel search + traversal", _entity)
                result = await get_rag_client().search_with_traversal(
                    query=partial_transcript,
                    entity_name=_entity,
                    max_facts=settings.memory.context_results,
                )
                pre_fetched = result.facts
            except Exception as e:
                logger.warning("Early prep entity search failed: %s", e)

            # Gather context using partial transcript
            svc = get_memory_service()
            mem_ctx = await svc.gather_context(
                query=partial_transcript,
                session_id=session_id,
                include_rag=True,
                pre_fetched_sources=pre_fetched,
                include_history=True,
                include_physical=False,
                max_history=6,
            )
            with _early_prep_lock:
                _early_prep_cache.clear()
                _early_prep_cache.update({
                    "partial": partial_transcript,
                    "route_result": route_result,
                    "mem_ctx": mem_ctx,
                    "ts": _time.monotonic(),
                })
            logger.info("Early prep cached (partial=%s)", partial_transcript[:40])

        try:
            future = asyncio.run_coroutine_threadsafe(_gather(), _event_loop)
            future.result(timeout=5.0)
        except Exception as e:
            logger.debug("Early prep gather failed: %s", e)

    return runner


def _create_kokoro_tts(tts_cfg, voice_cfg):
    """Create Kokoro TTS engine with Piper fallback."""
    try:
        from .tts_kokoro import KokoroTTS
        tts = KokoroTTS(
            voice=tts_cfg.voice,
            speed=tts_cfg.speed,
            lang=tts_cfg.kokoro_lang,
        )
        logger.info(
            "Using Kokoro TTS (voice=%s, speed=%.2f, lang=%s)",
            tts_cfg.voice, tts_cfg.speed, tts_cfg.kokoro_lang,
        )
        try:
            tts._ensure_loaded()
            logger.info("Kokoro TTS model preloaded")
        except Exception as preload_err:
            logger.warning("Kokoro TTS preload failed: %s", preload_err)
        return tts
    except Exception as e:
        logger.warning("Kokoro TTS init failed (%s), falling back to Piper", e)
        return _create_piper_tts(voice_cfg)


def _create_piper_tts(voice_cfg):
    """Create Piper TTS engine."""
    tts = PiperTTS(
        binary_path=voice_cfg.piper_binary,
        model_path=voice_cfg.piper_model,
        speaker=voice_cfg.piper_speaker,
        length_scale=voice_cfg.piper_length_scale,
        noise_scale=voice_cfg.piper_noise_scale,
        noise_w=voice_cfg.piper_noise_w,
        sample_rate=voice_cfg.piper_sample_rate,
    )
    tts.warm_up()
    return tts


def create_voice_pipeline(event_loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[VoicePipeline]:
    """Create the voice pipeline from config."""
    cfg = settings.voice

    if not cfg.enabled:
        logger.info("Voice pipeline disabled in config")
        return None

    # Log full configuration at startup for debugging
    logger.info("=== Voice Pipeline Configuration ===")
    logger.info("  enabled=%s", cfg.enabled)
    logger.info("  sample_rate=%d, block_size=%d", cfg.sample_rate, cfg.block_size)
    logger.info("  use_arecord=%s, arecord_device=%s", cfg.use_arecord, cfg.arecord_device)
    logger.info("  input_device=%s", cfg.input_device)
    logger.info("  audio_gain=%.2f", cfg.audio_gain)
    logger.info("  wake_threshold=%.3f", cfg.wake_threshold)
    logger.info("  wakeword_model_paths=%s", cfg.wakeword_model_paths)
    logger.info("  asr_url=%s", cfg.asr_url)
    logger.info("  asr_streaming_enabled=%s, asr_ws_url=%s", cfg.asr_streaming_enabled, cfg.asr_ws_url)
    logger.info("  tts_backend=%s", settings.tts.default_model)
    logger.info("  piper_binary=%s", cfg.piper_binary)
    logger.info("  piper_model=%s", cfg.piper_model)
    logger.info("  vad_aggressiveness=%d", cfg.vad_aggressiveness)
    logger.info("  silence_ms=%d, hangover_ms=%d", cfg.silence_ms, cfg.hangover_ms)
    logger.info("  debug_logging=%s, log_interval_frames=%d", cfg.debug_logging, cfg.log_interval_frames)
    logger.info("  conversation_mode=%s, timeout=%dms", cfg.conversation_mode_enabled, cfg.conversation_timeout_ms)
    logger.info("  node_id=%s, node_name=%s", cfg.node_id, cfg.node_name)
    # Voice filter configuration
    vf = settings.voice_filter
    logger.info("=== Voice Filter Configuration ===")
    logger.info("  enabled=%s, vad_backend=%s", vf.enabled, vf.vad_backend)
    logger.info("  silero_threshold=%.2f", vf.silero_threshold)
    logger.info("  rms_min=%.4f, adaptive=%s, above_ambient=%.1fx",
                vf.rms_min_threshold, vf.rms_adaptive, vf.rms_above_ambient_factor)
    logger.info("  turn_limit=%s (max=%d)", vf.turn_limit_enabled, vf.max_conversation_turns)
    logger.info("  intent_gating=%s (threshold=%.2f)", vf.intent_gating_enabled, vf.intent_continuation_threshold)
    logger.info("  speaker_continuity=%s", vf.speaker_continuity_enabled)
    logger.info("====================================")

    if not cfg.wakeword_model_paths:
        logger.warning("No wake word models configured")

    if settings.tts.default_model != "kokoro" and (not cfg.piper_binary or not cfg.piper_model):
        logger.warning("Piper TTS not fully configured")

    # Create ASR client (streaming or HTTP based on config)
    if cfg.asr_streaming_enabled and cfg.asr_ws_url:
        logger.info("Using streaming ASR: %s", cfg.asr_ws_url)
        try:
            asr_client = NemotronAsrStreamingClient(
                url=cfg.asr_ws_url,
                timeout=cfg.asr_timeout,
                sample_rate=cfg.sample_rate,
            )
        except ImportError as e:
            logger.warning("Streaming ASR unavailable (%s), falling back to HTTP", e)
            asr_client = NemotronAsrHttpClient(
                url=cfg.asr_url,
                api_key=cfg.asr_api_key,
                timeout=cfg.asr_timeout,
            )
    else:
        if not cfg.asr_url:
            logger.warning("No ASR URL configured")
        logger.info("Using HTTP batch ASR: %s", cfg.asr_url)
        asr_client = NemotronAsrHttpClient(
            url=cfg.asr_url,
            api_key=cfg.asr_api_key,
            timeout=cfg.asr_timeout,
        )

    # Create TTS engine based on config
    tts_cfg = settings.tts
    if tts_cfg.default_model == "kokoro":
        tts = _create_kokoro_tts(tts_cfg, cfg)
    else:
        tts = _create_piper_tts(cfg)
        logger.info("Using Piper TTS")

    agent_runner = _create_agent_runner()
    streaming_agent_runner = _create_streaming_agent_runner()
    prefill_runner = _create_prefill_runner()
    early_prep_runner = _create_early_prep_runner()

    # Initialize speaker ID service if enabled
    speaker_id_service = None
    speaker_cfg = settings.speaker_id
    if speaker_cfg.enabled:
        from ..services.speaker_id import get_speaker_id_service
        speaker_id_service = get_speaker_id_service()
        logger.info("Speaker ID enabled (require_known=%s, threshold=%.2f)",
                    speaker_cfg.require_known_speaker, speaker_cfg.confidence_threshold)
        # Preload the encoder to avoid first-command latency
        try:
            _ = speaker_id_service.embedder.encoder
            logger.info("Speaker ID encoder preloaded")
        except Exception as e:
            logger.warning("Failed to preload speaker ID encoder: %s", e)

    # Preload intent router if enabled
    if settings.intent_router.enabled:
        try:
            from ..services.intent_router import get_intent_router
            logger.info("Preloading semantic intent router...")
            router = get_intent_router()
            # Use sync loader -- async load() deadlocks here because
            # the event loop is blocked by the lifespan context.
            router.load_sync()
            logger.info("Semantic intent router preloaded successfully")
        except Exception as e:
            logger.warning(
                "Failed to preload semantic intent router: %s. "
                "Router will load on first query (may cause delay).",
                e
            )

    # Preload agent tools to avoid registration delay on first command
    try:
        from ..agents.tools import get_agent_tools
        logger.info("Preloading agent tools...")
        tools = get_agent_tools()
        logger.info("Agent tools preloaded: %d tools registered", len(tools.list_tools()))
    except Exception as e:
        logger.warning("Failed to preload agent tools: %s", e)

    # Get voice filter config
    filter_cfg = settings.voice_filter

    # Preload Silero VAD if using it
    silero_vad = None
    if filter_cfg.vad_backend == "silero":
        try:
            from .vad import SileroVAD
            logger.info("Preloading Silero VAD model...")
            silero_vad = SileroVAD(threshold=filter_cfg.silero_threshold)
            silero_vad.preload()
            logger.info("Silero VAD model preloaded successfully")
        except Exception as e:
            logger.warning(
                "Failed to preload Silero VAD: %s. "
                "Falling back to WebRTC VAD.",
                e
            )
            filter_cfg = settings.voice_filter.__class__(vad_backend="webrtc")

    pipeline = VoicePipeline(
        wakeword_model_paths=cfg.wakeword_model_paths,
        wake_threshold=cfg.wake_threshold,
        asr_client=asr_client,
        tts=tts,
        agent_runner=agent_runner,
        streaming_agent_runner=streaming_agent_runner,
        streaming_llm_enabled=cfg.streaming_llm_enabled,
        sample_rate=cfg.sample_rate,
        block_size=cfg.block_size,
        silence_ms=cfg.silence_ms,
        max_command_seconds=cfg.max_command_seconds,
        min_command_ms=cfg.min_command_ms,
        min_speech_frames=cfg.min_speech_frames,
        wake_buffer_frames=cfg.wake_buffer_frames,
        vad_aggressiveness=cfg.vad_aggressiveness,
        hangover_ms=cfg.hangover_ms,
        use_arecord=cfg.use_arecord,
        arecord_device=cfg.arecord_device,
        input_device=cfg.input_device,
        stop_hotkey=cfg.stop_hotkey,
        allow_wake_barge_in=cfg.allow_wake_barge_in,
        interrupt_on_speech=cfg.interrupt_on_speech,
        interrupt_speech_frames=cfg.interrupt_speech_frames,
        interrupt_rms_threshold=cfg.interrupt_rms_threshold,
        interrupt_wake_models=cfg.interrupt_wake_models,
        interrupt_wake_threshold=cfg.interrupt_wake_threshold,
        command_workers=cfg.command_workers,
        audio_gain=cfg.audio_gain,
        prefill_runner=prefill_runner,
        prefill_cache_ttl=cfg.prefill_cache_ttl,
        filler_enabled=cfg.filler_enabled,
        filler_delay_ms=cfg.filler_delay_ms,
        filler_phrases=cfg.filler_phrases,
        filler_followup_delay_ms=cfg.filler_followup_delay_ms,
        filler_followup_phrases=cfg.filler_followup_phrases,
        debug_logging=cfg.debug_logging,
        log_interval_frames=cfg.log_interval_frames,
        conversation_mode_enabled=cfg.conversation_mode_enabled,
        conversation_timeout_ms=cfg.conversation_timeout_ms,
        conversation_start_delay_ms=cfg.conversation_start_delay_ms,
        conversation_speech_frames=cfg.conversation_speech_frames,
        conversation_speech_tolerance=cfg.conversation_speech_tolerance,
        conversation_rms_threshold=cfg.conversation_rms_threshold,
        speaker_id_enabled=speaker_cfg.enabled,
        speaker_id_service=speaker_id_service,
        require_known_speaker=speaker_cfg.require_known_speaker,
        unknown_speaker_response=speaker_cfg.unknown_speaker_response,
        error_asr_empty=cfg.error_asr_empty,
        error_agent_timeout=cfg.error_agent_timeout,
        error_agent_failed=cfg.error_agent_failed,
        speaker_id_timeout=cfg.speaker_id_timeout,
        node_id=cfg.node_id,
        event_loop=event_loop,
        # Voice filter settings
        vad_backend=filter_cfg.vad_backend,
        silero_threshold=filter_cfg.silero_threshold,
        rms_min_threshold=filter_cfg.rms_min_threshold,
        rms_adaptive=filter_cfg.rms_adaptive,
        rms_above_ambient_factor=filter_cfg.rms_above_ambient_factor,
        turn_limit_enabled=filter_cfg.turn_limit_enabled,
        max_conversation_turns=filter_cfg.max_conversation_turns,
        intent_gating_enabled=filter_cfg.intent_gating_enabled,
        intent_continuation_threshold=filter_cfg.intent_continuation_threshold,
        intent_categories_continue=filter_cfg.intent_categories_continue,
        speaker_continuity_enabled=filter_cfg.speaker_continuity_enabled,
        speaker_continuity_threshold=filter_cfg.speaker_continuity_threshold,
        # Conversation-mode recording thresholds
        conversation_silence_ms=cfg.conversation_silence_ms,
        conversation_hangover_ms=cfg.conversation_hangover_ms,
        conversation_max_command_seconds=cfg.conversation_max_command_seconds,
        # Sliding window segmentation
        conversation_window_frames=cfg.conversation_window_frames,
        conversation_silence_ratio=cfg.conversation_silence_ratio,
        conversation_asr_holdoff_ms=cfg.conversation_asr_holdoff_ms,
        asr_quiet_limit=cfg.asr_quiet_limit,
        # Workflow-aware thresholds
        workflow_silence_ms=cfg.workflow_silence_ms,
        workflow_hangover_ms=cfg.workflow_hangover_ms,
        workflow_max_command_seconds=cfg.workflow_max_command_seconds,
        workflow_conversation_timeout_ms=cfg.workflow_conversation_timeout_ms,
        # Turn limit warning
        conversation_turn_limit_phrase=cfg.conversation_turn_limit_phrase,
        # Wake confirmation tone
        wake_confirmation_enabled=cfg.wake_confirmation_enabled,
        wake_confirmation_freq=cfg.wake_confirmation_freq,
        wake_confirmation_duration_ms=cfg.wake_confirmation_duration_ms,
        # Early preparation during conversation silence
        early_preparation_runner=early_prep_runner,
        conversation_early_silence_ms=cfg.conversation_early_silence_ms,
    )

    return pipeline


def start_voice_pipeline(loop: asyncio.AbstractEventLoop) -> bool:
    """
    Start the voice pipeline in a background thread.

    Args:
        loop: The main asyncio event loop for agent calls

    Returns:
        True if started successfully
    """
    global _voice_pipeline, _voice_thread, _event_loop, _free_mode_manager

    if _voice_thread is not None and _voice_thread.is_alive():
        logger.warning("Voice pipeline already running")
        return True

    _event_loop = loop

    try:
        _voice_pipeline = create_voice_pipeline(event_loop=loop)
        if _voice_pipeline is None:
            return False
    except Exception as e:
        logger.error("Failed to create voice pipeline: %s", e)
        return False

    def run_pipeline():
        """Run the pipeline (blocking)."""
        try:
            _voice_pipeline.start()
        except Exception as e:
            logger.error("Voice pipeline crashed: %s", e)

    _voice_thread = threading.Thread(
        target=run_pipeline,
        name="voice-pipeline",
        daemon=True,
    )
    _voice_thread.start()

    logger.info("Voice pipeline started in background thread")

    # Start free mode evaluator if enabled
    if settings.free_mode.enabled:
        try:
            from .free_mode import FreeModeManager
            _free_mode_manager = FreeModeManager(
                pipeline=_voice_pipeline,
                config=settings.free_mode,
            )
            _free_mode_manager.start()
            logger.info("Free conversation mode evaluator started")
        except Exception as e:
            logger.warning("Failed to start free mode evaluator: %s", e)

    return True


def stop_voice_pipeline() -> None:
    """Stop the voice pipeline."""
    global _voice_pipeline, _voice_thread, _free_mode_manager

    # Stop free mode evaluator first
    if _free_mode_manager is not None:
        try:
            _free_mode_manager.stop()
        except Exception as e:
            logger.warning("Error stopping free mode evaluator: %s", e)
        _free_mode_manager = None

    if _voice_pipeline is not None:
        # Release entity locks for the ending session
        session_id = getattr(_voice_pipeline, "session_id", None)
        if session_id:
            try:
                import asyncio
                from ..reasoning.lock_integration import on_voice_session_end
                loop = getattr(_voice_pipeline, "event_loop", None)
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        on_voice_session_end(session_id), loop
                    )
            except Exception:
                pass

        try:
            _voice_pipeline.playback.stop()
            _voice_pipeline.command_executor.shutdown()
        except Exception as e:
            logger.warning("Error stopping voice pipeline: %s", e)

    _voice_pipeline = None
    _voice_thread = None

    logger.info("Voice pipeline stopped")


def get_voice_pipeline() -> Optional[VoicePipeline]:
    """Get the active voice pipeline instance."""
    return _voice_pipeline
