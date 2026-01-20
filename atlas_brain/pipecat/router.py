"""
FunctionGemma Tool Router for fast tool calling.

Uses Google's FunctionGemma-270M model to quickly determine
if a query needs a tool and which tool to use.

Two-tier architecture:
- Simple queries (single tool): Direct execution via FunctionGemma (~200ms)
- Complex queries (multi-tool/reasoning): gpt-oss:20b multi-turn tool calling (~1.5s)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger("atlas.pipecat.router")

# Patterns indicating complex queries that need multi-turn reasoning
COMPLEX_QUERY_PATTERNS = [
    r"\b(and|then|also|after that)\b.*\b(remind|set|check|get)\b",  # Multiple actions
    r"\bplan\s+(my|the|a)\s+(day|week|schedule)\b",  # Planning requests
    r"\b(help me|assist)\b.*\b(with|to)\b",  # Help requests often need reasoning
    r"\bif\s+.*(then|remind|set)\b",  # Conditional logic
    r"\b(first|next|finally)\b",  # Sequential instructions
    r"\bsummarize\b.*\b(calendar|schedule|events)\b",  # Summarization
    r"\bbased on\b",  # Contextual reasoning
]

# Keywords that strongly indicate simple single-tool queries
SIMPLE_QUERY_PATTERNS = [
    r"^what('s|\s+is)\s+(the\s+)?(time|date)\??$",  # "what time is it"
    r"^what('s|\s+is)\s+(the\s+)?weather\b",  # "what's the weather"
    r"^(get|check)\s+(the\s+)?(time|date|weather)\b",  # "get the time"
    r"^tell me the (time|date)\b",  # "tell me the time"
]


@dataclass
class ToolRouteResult:
    """Result from tool routing."""
    needs_tool: bool
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    route_time_ms: float = 0
    confidence: float = 0.0


@dataclass
class ProcessedQueryResult:
    """Full result from processing a query through the router."""
    response: str
    was_tool_query: bool
    latency_ms: float
    tools_executed: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    used_llm_tools: bool = False  # True if routed to gpt-oss multi-turn


class FunctionGemmaRouter:
    """
    Fast tool router using FunctionGemma-270M.

    Determines if a query needs a tool and which tool to use.
    Falls back to main LLM for complex queries.
    """

    # Priority tools for quick routing (subset loaded on first use)
    PRIORITY_TOOL_NAMES = [
        "get_time", "get_weather", "get_calendar", "get_location",
        "set_reminder", "list_reminders", "send_notification",
    ]

    def __init__(
        self,
        model_name: str = "google/functiongemma-270m-it",
        device: str = "cuda",
    ):
        """
        Initialize FunctionGemma router.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cuda" or "cpu")
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._tools = None

    @property
    def TOOLS(self) -> list:
        """Get tool schemas from registry (lazy loaded)."""
        if self._tools is None:
            from ..tools import tool_registry
            self._tools = tool_registry.get_tool_schemas_filtered(
                self.PRIORITY_TOOL_NAMES
            )
        return self._tools

    async def load(self):
        """Load the FunctionGemma model."""
        if self._model is not None:
            return

        logger.info("Loading FunctionGemma router: %s", self._model_name)
        start = time.time()

        try:
            loop = asyncio.get_event_loop()

            def _load():
                from transformers import AutoModelForCausalLM, AutoProcessor

                processor = AutoProcessor.from_pretrained(self._model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self._device,
                )
                model.eval()
                return model, processor

            self._model, self._processor = await loop.run_in_executor(None, _load)

            load_time = time.time() - start
            logger.info("FunctionGemma loaded in %.2fs on %s", load_time, self._device)

        except Exception as e:
            logger.error("Failed to load FunctionGemma: %s", e)
            raise

    async def route(self, query: str) -> ToolRouteResult:
        """
        Route a query to determine if it needs a tool.

        Args:
            query: User query text

        Returns:
            ToolRouteResult with tool info or needs_tool=False
        """
        if self._model is None:
            await self.load()

        start = time.time()

        try:
            # Build message with tool definitions
            messages = [
                {
                    "role": "developer",
                    "content": "You are a model that can do function calling with the following functions"
                },
                {"role": "user", "content": query},
            ]

            loop = asyncio.get_event_loop()

            def _generate():
                inputs = self._processor.apply_chat_template(
                    messages,
                    tools=self.TOOLS,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self._model.device)

                # Get eos_token_id from processor or model config
                eos_token_id = getattr(self._processor, 'eos_token_id', None)
                if eos_token_id is None:
                    eos_token_id = self._model.config.eos_token_id

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=64,
                        pad_token_id=eos_token_id,
                        do_sample=False,
                    )

                # Decode only new tokens
                response = self._processor.decode(
                    outputs[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True,
                )
                return response

            response = await loop.run_in_executor(None, _generate)

            route_time = (time.time() - start) * 1000
            logger.info("FunctionGemma route: '%s' -> '%s' (%.0fms)",
                       query[:30], response[:50], route_time)

            # Parse response for function call
            result = self._parse_response(response)
            result.route_time_ms = route_time
            return result

        except Exception as e:
            logger.error("FunctionGemma routing error: %s", e)
            return ToolRouteResult(
                needs_tool=False,
                route_time_ms=(time.time() - start) * 1000,
            )

    def _parse_response(self, response: str) -> ToolRouteResult:
        """Parse FunctionGemma response for tool calls."""
        response = response.strip()

        # Check for "NO_TOOL" or conversational response
        if "NO_TOOL" in response or not response:
            return ToolRouteResult(needs_tool=False)

        # Parse function call format: <start_function_call>call:tool_name{args}<end_function_call>
        func_pattern = r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
        match = re.search(func_pattern, response)

        if match:
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments (format: key:<escape>value<escape>)
            args = {}
            arg_pattern = r"(\w+):<escape>(.*?)<escape>"
            for arg_match in re.finditer(arg_pattern, args_str):
                args[arg_match.group(1)] = arg_match.group(2)

            return ToolRouteResult(
                needs_tool=True,
                tool_name=tool_name,
                tool_args=args,
                confidence=0.9,
            )

        # Try simpler format: call:tool_name{args}
        simple_pattern = r"call:(\w+)\{(.*?)\}"
        match = re.search(simple_pattern, response)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse simple args
            args = {}
            for part in args_str.split(","):
                if ":" in part:
                    key, val = part.split(":", 1)
                    args[key.strip()] = val.strip().strip("<>").replace("escape", "")

            return ToolRouteResult(
                needs_tool=True,
                tool_name=tool_name,
                tool_args=args,
                confidence=0.7,
            )

        # Check if response looks like it wants to call a known tool
        response_lower = response.lower()
        for tool in self.TOOLS:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name and tool_name in response_lower:
                return ToolRouteResult(
                    needs_tool=True,
                    tool_name=tool_name,
                    tool_args={},
                    confidence=0.5,
                )

        return ToolRouteResult(needs_tool=False)

    async def unload(self):
        """Unload model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
        logger.info("FunctionGemma unloaded")


# Module-level instance
_router: Optional[FunctionGemmaRouter] = None


async def get_router() -> FunctionGemmaRouter:
    """Get or create the FunctionGemma router."""
    global _router
    if _router is None:
        _router = FunctionGemmaRouter()
        await _router.load()
    return _router


async def route_query(query: str) -> ToolRouteResult:
    """Convenience function to route a query."""
    router = await get_router()
    return await router.route(query)


def is_complex_query(text: str) -> bool:
    """
    Determine if a query requires multi-turn reasoning (gpt-oss).

    Complex queries:
    - Multiple tool calls needed
    - Conditional logic
    - Planning/scheduling
    - Summarization requests

    Simple queries:
    - Single tool call with clear parameters
    - Direct questions (time, weather, etc.)
    """
    text_lower = text.lower().strip()

    # Check for simple patterns first (fast path)
    for pattern in SIMPLE_QUERY_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # Check for complex patterns
    for pattern in COMPLEX_QUERY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.debug("Complex query detected: '%s' matched pattern", text[:50])
            return True

    # Heuristic: multiple tool-related keywords suggest complexity
    tool_keywords = ["time", "weather", "calendar", "remind", "schedule", "event"]
    matches = sum(1 for kw in tool_keywords if kw in text_lower)
    if matches >= 2:
        logger.debug("Complex query detected: multiple tool keywords (%d)", matches)
        return True

    return False


class ToolRouterProcessor:
    """
    Two-tier tool router for voice pipeline.

    Routing logic:
    1. Simple tool queries -> FunctionGemma (~120ms) -> Direct execution (~80ms)
    2. Complex queries -> gpt-oss:20b multi-turn tool calling (~1.5s)
    3. Non-tool queries -> Pass to conversational LLM

    Usage in pipeline:
        router_proc = ToolRouterProcessor()
        result = await router_proc.process(text)
    """

    def __init__(self, router: Optional[FunctionGemmaRouter] = None):
        """Initialize with optional pre-loaded router."""
        self._router = router
        self._tool_registry = None
        self._gptoss_service = None

    async def _ensure_router(self):
        """Ensure router is loaded."""
        if self._router is None:
            self._router = await get_router()

    async def _get_tool_registry(self):
        """Lazy load tool registry."""
        if self._tool_registry is None:
            from ..tools import tool_registry
            self._tool_registry = tool_registry
        return self._tool_registry

    async def _get_gptoss_service(self):
        """Lazy load gpt-oss service."""
        if self._gptoss_service is None:
            from .llm import get_gptoss_service
            self._gptoss_service = get_gptoss_service()
        return self._gptoss_service

    async def process(self, text: str) -> ProcessedQueryResult:
        """
        Process a text query through the two-tier router.

        Args:
            text: User query text

        Returns:
            ProcessedQueryResult with response and execution details
        """
        if not text or not text.strip():
            return ProcessedQueryResult(
                response="",
                was_tool_query=False,
                latency_ms=0,
            )

        start_time = time.time()

        # Check if this is a complex query that needs gpt-oss
        if is_complex_query(text):
            return await self._process_complex(text, start_time)

        # Simple query path: FunctionGemma routing
        await self._ensure_router()
        route_result = await self._router.route(text)

        if not route_result.needs_tool:
            # Not a tool query - pass through to conversational LLM
            logger.info("Router: '%s' -> conversation (no tool)", text[:30])
            return ProcessedQueryResult(
                response="",
                was_tool_query=False,
                latency_ms=route_result.route_time_ms,
            )

        # Simple tool query - execute directly (fast path)
        logger.info(
            "Router: '%s' -> %s (simple, %.0fms)",
            text[:30],
            route_result.tool_name,
            route_result.route_time_ms,
        )

        registry = await self._get_tool_registry()
        tool_result = await registry.execute(
            route_result.tool_name,
            route_result.tool_args or {},
        )

        total_latency = (time.time() - start_time) * 1000

        if tool_result.success:
            response_text = tool_result.message
        else:
            response_text = "Sorry, I couldn't get that information."

        return ProcessedQueryResult(
            response=response_text,
            was_tool_query=True,
            latency_ms=total_latency,
            tools_executed=[route_result.tool_name],
            tool_results={route_result.tool_name: tool_result.message},
            used_llm_tools=False,
        )

    async def _process_complex(self, text: str, start_time: float) -> ProcessedQueryResult:
        """
        Process a complex query through gpt-oss multi-turn tool calling.

        Args:
            text: User query text
            start_time: Time when processing started

        Returns:
            ProcessedQueryResult with LLM response and all tools executed
        """
        logger.info("Router: '%s' -> gpt-oss (complex query)", text[:30])

        try:
            gptoss = await self._get_gptoss_service()
            result = await gptoss.process_with_tools(text)

            total_latency = (time.time() - start_time) * 1000

            logger.info(
                "gpt-oss completed: %d tools in %d turns (%.0fms)",
                len(result.tools_executed),
                result.turns,
                total_latency,
            )

            return ProcessedQueryResult(
                response=result.response,
                was_tool_query=True,
                latency_ms=total_latency,
                tools_executed=result.tools_executed,
                tool_results=result.tool_results,
                used_llm_tools=True,
            )

        except Exception as e:
            logger.error("gpt-oss error: %s", e)
            total_latency = (time.time() - start_time) * 1000
            return ProcessedQueryResult(
                response="Sorry, I had trouble processing that request.",
                was_tool_query=True,
                latency_ms=total_latency,
                used_llm_tools=True,
            )

    # Legacy method for backward compatibility
    async def process_legacy(self, text: str) -> tuple[str, bool, float]:
        """
        Legacy process method that returns tuple.

        Deprecated: Use process() which returns ProcessedQueryResult.
        """
        result = await self.process(text)
        return result.response, result.was_tool_query, result.latency_ms
