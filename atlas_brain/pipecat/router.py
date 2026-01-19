"""
FunctionGemma Tool Router for fast tool calling.

Uses Google's FunctionGemma-270M model to quickly determine
if a query needs a tool and which tool to use.

This bypasses the slower LLM for simple tool queries like
"what time is it?" - achieving ~100ms routing vs 1.5s LLM.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger("atlas.pipecat.router")


@dataclass
class ToolRouteResult:
    """Result from tool routing."""
    needs_tool: bool
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    route_time_ms: float = 0
    confidence: float = 0.0


class FunctionGemmaRouter:
    """
    Fast tool router using FunctionGemma-270M.

    Determines if a query needs a tool and which tool to use.
    Falls back to main LLM for complex queries.
    """

    # Tool definitions for common queries - OpenAI function calling format
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current time and date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (optional)",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set a reminder for a specific time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "What to remind about",
                        },
                        "time": {
                            "type": "string",
                            "description": "When to remind (e.g., 'in 10 minutes', '3pm')",
                        },
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_calendar",
                "description": "Get calendar events for today or a date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to check (optional, defaults to today)",
                        }
                    },
                    "required": [],
                },
            },
        },
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


class ToolRouterProcessor:
    """
    Pipecat-compatible processor that routes tool queries via FunctionGemma.

    Intercepts TranscriptionFrames and:
    - Tool queries: Execute tool, return TextFrame directly (bypass LLM)
    - Non-tool queries: Pass through to LLM

    Usage in pipeline:
        router_proc = ToolRouterProcessor()
        # Use router_proc.process() to handle transcriptions
    """

    def __init__(self, router: Optional[FunctionGemmaRouter] = None):
        """Initialize with optional pre-loaded router."""
        self._router = router
        self._tool_registry = None

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

    async def process(self, text: str) -> tuple[str, bool, float]:
        """
        Process a text query through the router.

        Args:
            text: User query text

        Returns:
            Tuple of (response_text, was_tool_query, latency_ms)
        """
        if not text or not text.strip():
            return "", False, 0.0

        await self._ensure_router()

        # Route the query
        route_result = await self._router.route(text)

        if not route_result.needs_tool:
            # Not a tool query - pass through to LLM
            logger.info("Router: '%s' -> LLM (no tool)", text[:30])
            return "", False, route_result.route_time_ms

        # Tool query - execute directly
        logger.info(
            "Router: '%s' -> %s (%.0fms)",
            text[:30],
            route_result.tool_name,
            route_result.route_time_ms,
        )

        registry = await self._get_tool_registry()
        tool_result = await registry.execute(
            route_result.tool_name,
            route_result.tool_args or {},
        )

        # Create response text from tool result
        if tool_result.success:
            response_text = tool_result.message
        else:
            response_text = f"Sorry, I couldn't get that information."

        return response_text, True, route_result.route_time_ms
