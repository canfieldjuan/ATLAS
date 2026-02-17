"""
Unified Agent Interface.

Provides a common interface for LangGraph agent implementations.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol, runtime_checkable

from .protocols import AgentResult

logger = logging.getLogger("atlas.agents.interface")
_SENSITIVE_HEADER_KEY_PARTS = ("authorization", "api-key", "token", "cookie", "secret", "password")


@runtime_checkable
class AgentInterface(Protocol):
    """Common interface for all agent implementations."""

    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        input_type: str = "text",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Process input and return result.

        Args:
            input_text: User input text
            session_id: Session ID for persistence
            speaker_id: Identified speaker name
            input_type: Input source type (text, voice, vision)
            runtime_context: Additional runtime context

        Returns:
            AgentResult with response and metadata
        """
        ...


class LangGraphAgentAdapter:
    """Adapts LangGraph agent to unified interface."""

    def __init__(self, graph: Any):
        """
        Initialize adapter with a LangGraph agent.

        Args:
            graph: A LangGraph agent (HomeAgentGraph, AtlasAgentGraph, etc.)
        """
        self._graph = graph

    def _get_model_name(self) -> str:
        """Get the active LLM model name for tracing."""
        try:
            from ..services import llm_registry
            active = llm_registry.get_active()
            return getattr(active, "model_name", "unknown") if active else "unknown"
        except Exception:
            return "unknown"

    async def process(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        input_type: str = "text",
        runtime_context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Process input through LangGraph agent."""
        from ..services.tracing import tracer

        runtime_ctx = runtime_context or {}

        span = tracer.start_span(
            span_name="agent.process",
            operation_type="llm_call",
            model_name=self._get_model_name(),
            model_provider="ollama",
            session_id=session_id,
            metadata={"input_type": input_type, "speaker_id": speaker_id},
        )

        try:
            result = await self._graph.run(
                input_text=input_text,
                session_id=session_id,
                speaker_id=speaker_id,
                input_type=input_type,
                runtime_context=runtime_ctx,
            )

            # Convert dict result to AgentResult
            timing = result.get("timing", {})
            agent_result = AgentResult(
                success=result.get("success", False),
                response_text=result.get("response_text"),
                action_type=result.get("action_type", "none"),
                intent=result.get("intent"),
                action_results=result.get("action_results", []),
                error=result.get("error"),
                total_ms=timing.get("total", 0),
                think_ms=timing.get("think", 0) + timing.get("classify", 0),
                act_ms=timing.get("act", 0),
                llm_ms=timing.get("respond", 0),
                metadata={
                    "awaiting_user_input": result.get("awaiting_user_input", False),
                },
            )

            # Extract LLM metadata from result
            llm_meta = result.get("llm_meta", {}) or {}
            has_llm_response = bool(llm_meta.get("has_llm_response"))
            input_tokens = self._to_int_or_none(llm_meta.get("input_tokens"))
            output_tokens = self._to_int_or_none(llm_meta.get("output_tokens"))
            if not has_llm_response and (input_tokens or 0) == 0 and (output_tokens or 0) == 0:
                input_tokens = None
                output_tokens = None

            ttft_ms = self._to_float_or_none(llm_meta.get("prompt_eval_duration_ms"))
            inference_time_ms = self._to_float_or_none(llm_meta.get("eval_duration_ms"))
            provider_total_ms = self._to_float_or_none(llm_meta.get("total_duration_ms"))

            queue_time_ms: Optional[float] = None
            if (
                provider_total_ms is not None
                and ttft_ms is not None
                and inference_time_ms is not None
            ):
                provider_residual_ms = provider_total_ms - ttft_ms - inference_time_ms
                if provider_residual_ms > 0:
                    queue_time_ms = provider_residual_ms

            context_tokens = self._to_int_or_none(llm_meta.get("context_tokens"))
            retrieval_latency_ms = self._to_float_or_none(llm_meta.get("retrieval_latency_ms"))
            if retrieval_latency_ms is None:
                memory_phase_ms = self._safe_ms(timing.get("memory"))
                retrieval_latency_ms = memory_phase_ms if memory_phase_ms > 0 else None

            rag_nodes_retrieved = self._to_int_or_none(llm_meta.get("rag_nodes_retrieved"))
            rag_chunks_used = self._to_int_or_none(llm_meta.get("rag_chunks_used"))
            if rag_chunks_used is None:
                rag_chunks_used = rag_nodes_retrieved

            rag_graph_used = llm_meta.get("rag_graph_used")
            if not isinstance(rag_graph_used, bool):
                rag_graph_used = True if (rag_nodes_retrieved is not None and rag_nodes_retrieved > 0) else None

            provider_request_id = llm_meta.get("provider_request_id")
            if not isinstance(provider_request_id, str) or not provider_request_id.strip():
                provider_request_id = None

            api_endpoint = self._extract_api_endpoint(runtime_ctx)
            request_headers_sanitized = self._extract_request_headers(runtime_ctx)

            # Build rich input_data
            input_data = {
                "user_message": input_text,
                "input_type": input_type,
            }
            if llm_meta.get("system_prompt"):
                input_data["system_prompt"] = llm_meta["system_prompt"][:3000]
            if llm_meta.get("history_count"):
                input_data["history_turns"] = llm_meta["history_count"]

            # Build rich output_data
            output_data = {
                "response": (agent_result.response_text or "")[:5000],
                "action_type": agent_result.action_type,
            }
            if agent_result.action_results:
                output_data["action_results"] = [
                    {"success": r.success, "message": r.message}
                    for r in agent_result.action_results[:5]
                ]

            # Build trace metadata
            trace_meta: dict[str, Any] = {
                "timing": timing,
                "action_type": agent_result.action_type,
            }
            if llm_meta:
                trace_meta["llm"] = llm_meta
            intent = result.get("intent")
            if intent:
                trace_meta["intent"] = intent.action if hasattr(intent, "action") else str(intent)
            tools_executed = result.get("tools_executed")
            if tools_executed:
                trace_meta["tools_executed"] = tools_executed

            # Emit child spans for per-phase timings (classify/think/act/respond/etc.).
            self._emit_timing_child_spans(
                tracer=tracer,
                parent_span=span,
                timing=timing,
                tools_executed=tools_executed,
                llm_meta=llm_meta,
                action_type=agent_result.action_type,
            )

            tracer.end_span(
                span,
                status="completed" if agent_result.success else "failed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_data=input_data,
                output_data=output_data,
                error_message=agent_result.error,
                metadata=trace_meta,
                ttft_ms=ttft_ms,
                inference_time_ms=inference_time_ms,
                queue_time_ms=queue_time_ms,
                context_tokens=context_tokens,
                retrieval_latency_ms=retrieval_latency_ms,
                rag_graph_used=rag_graph_used,
                rag_nodes_retrieved=rag_nodes_retrieved,
                rag_chunks_used=rag_chunks_used,
                api_endpoint=api_endpoint,
                request_headers_sanitized=request_headers_sanitized,
                provider_request_id=provider_request_id,
            )

            return agent_result

        except Exception as e:
            tracer.end_span(
                span,
                status="failed",
                input_data={"user_message": input_text},
                error_message=str(e),
                error_type=type(e).__name__,
            )
            raise

    @staticmethod
    def _safe_ms(value: Any) -> float:
        """Coerce timing values to non-negative milliseconds."""
        try:
            ms = float(value)
        except (TypeError, ValueError):
            return 0.0
        return ms if ms > 0 else 0.0

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        """Coerce value to int when available."""
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float_or_none(value: Any) -> Optional[float]:
        """Coerce value to float when available."""
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_api_endpoint(runtime_context: dict[str, Any]) -> Optional[str]:
        """Extract request endpoint/path from runtime context."""
        request_obj = runtime_context.get("request")
        candidates = [
            runtime_context.get("api_endpoint"),
            runtime_context.get("endpoint"),
            runtime_context.get("path"),
            runtime_context.get("route"),
            request_obj.get("url") if isinstance(request_obj, dict) else None,
            request_obj.get("path") if isinstance(request_obj, dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()[:500]
        return None

    @staticmethod
    def _extract_request_headers(runtime_context: dict[str, Any]) -> Optional[dict[str, str]]:
        """Sanitize inbound headers for observability without leaking secrets."""
        headers = runtime_context.get("request_headers") or runtime_context.get("headers")
        request_obj = runtime_context.get("request")
        if headers is None and isinstance(request_obj, dict):
            headers = request_obj.get("headers")
        if not isinstance(headers, dict):
            return None

        sanitized: dict[str, str] = {}
        for raw_key, raw_value in headers.items():
            key = str(raw_key)
            key_lower = key.lower()
            if any(part in key_lower for part in _SENSITIVE_HEADER_KEY_PARTS):
                sanitized[key] = "[redacted]"
                continue
            if raw_value is None:
                continue
            sanitized[key] = str(raw_value)[:256]
        return sanitized or None

    def _emit_timing_child_spans(
        self,
        tracer: Any,
        parent_span: Any,
        timing: dict[str, Any],
        tools_executed: Any,
        llm_meta: dict[str, Any],
        action_type: str,
    ) -> None:
        """Emit child spans from timing breakdown returned by agent graphs."""
        try:
            root_start = datetime.fromisoformat(parent_span.start_iso)
        except Exception:
            # If timestamp parsing fails, skip child spans rather than risking bad payloads.
            return

        # Keep this order aligned with graph execution flow.
        phase_definitions = [
            ("classify", "agent.classify", "classification"),
            ("memory", "agent.memory", "retrieval"),
            ("think", "agent.think", "reasoning"),
            ("act", "agent.act", "tool_call"),
            ("respond", "agent.respond", "llm_call"),
        ]

        total_ms = self._safe_ms(timing.get("total"))
        offset_ms = 0.0

        for phase_key, span_name, operation_type in phase_definitions:
            phase_ms = self._safe_ms(timing.get(phase_key))
            if phase_ms <= 0:
                continue

            if total_ms > 0 and offset_ms >= total_ms:
                break
            if total_ms > 0 and offset_ms + phase_ms > total_ms:
                phase_ms = max(0.0, total_ms - offset_ms)
                if phase_ms <= 0:
                    break

            phase_start = root_start + timedelta(milliseconds=offset_ms)
            phase_end = phase_start + timedelta(milliseconds=phase_ms)

            phase_metadata: dict[str, Any] = {
                "phase": phase_key,
                "action_type": action_type,
            }
            if phase_key == "act" and tools_executed:
                phase_metadata["tools_executed"] = tools_executed

            phase_input_tokens: Optional[int] = None
            phase_output_tokens: Optional[int] = None
            if phase_key == "respond":
                phase_input_tokens = self._to_int_or_none(llm_meta.get("input_tokens"))
                phase_output_tokens = self._to_int_or_none(llm_meta.get("output_tokens"))
                prompt_eval_ms = self._to_float_or_none(llm_meta.get("prompt_eval_duration_ms"))
                eval_ms = self._to_float_or_none(llm_meta.get("eval_duration_ms"))
                total_ms = self._to_float_or_none(llm_meta.get("total_duration_ms"))
                if prompt_eval_ms is not None:
                    phase_metadata["ttft_ms"] = int(round(prompt_eval_ms))
                if eval_ms is not None:
                    phase_metadata["inference_time_ms"] = int(round(eval_ms))
                if total_ms is not None:
                    phase_metadata["provider_total_duration_ms"] = int(round(total_ms))

            tracer.emit_child_span(
                parent=parent_span,
                span_name=span_name,
                operation_type=operation_type,
                start_iso=phase_start.isoformat(),
                end_iso=phase_end.isoformat(),
                duration_ms=phase_ms,
                status="completed",
                input_tokens=phase_input_tokens,
                output_tokens=phase_output_tokens,
                metadata=phase_metadata,
            )
            offset_ms += phase_ms


# Agent factory singletons
_atlas_agent: Optional[AgentInterface] = None
_home_agent: Optional[AgentInterface] = None


def get_agent(
    agent_type: str = "atlas",
    session_id: Optional[str] = None,
    backend: Optional[str] = None,
    business_context: Optional[Any] = None,
) -> AgentInterface:
    """
    Get an agent instance.

    Args:
        agent_type: Type of agent ("atlas", "home", "receptionist")
        session_id: Session ID for the agent
        backend: Ignored (kept for backwards compatibility)
        business_context: Business context for receptionist agent

    Returns:
        Agent adapter implementing AgentInterface
    """
    if agent_type == "atlas":
        from .graphs import get_atlas_agent_langgraph
        graph = get_atlas_agent_langgraph(session_id=session_id)
        return LangGraphAgentAdapter(graph)

    elif agent_type == "home":
        from .graphs import get_home_agent_langgraph
        graph = get_home_agent_langgraph(session_id=session_id)
        return LangGraphAgentAdapter(graph)

    elif agent_type == "receptionist":
        from .graphs import get_receptionist_agent_langgraph
        graph = get_receptionist_agent_langgraph(
            business_context=business_context,
            session_id=session_id,
        )
        return LangGraphAgentAdapter(graph)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


async def process_with_fallback(
    input_text: str,
    agent_type: str = "atlas",
    session_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
    input_type: str = "text",
    runtime_context: Optional[dict[str, Any]] = None,
) -> AgentResult:
    """
    Process input with error handling.

    Args:
        input_text: User input text
        agent_type: Type of agent to use
        session_id: Session ID
        speaker_id: Speaker ID
        input_type: Input type
        runtime_context: Runtime context

    Returns:
        AgentResult from agent processing
    """
    try:
        agent = get_agent(
            agent_type=agent_type,
            session_id=session_id,
        )
        return await agent.process(
            input_text=input_text,
            session_id=session_id,
            speaker_id=speaker_id,
            input_type=input_type,
            runtime_context=runtime_context,
        )

    except Exception as e:
        logger.exception("Agent processing failed: %s", e)
        return AgentResult(
            success=False,
            error=str(e),
            response_text="I encountered an error processing your request.",
        )


def reset_agent_cache() -> None:
    """Reset cached agent instances."""
    global _atlas_agent, _home_agent
    _atlas_agent = None
    _home_agent = None
