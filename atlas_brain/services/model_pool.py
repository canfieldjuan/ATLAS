"""
Model Pool - Multi-model management for simultaneous loading.

Keeps multiple LLM models loaded in VRAM simultaneously,
allowing instant routing without model swap overhead.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("atlas.services.model_pool")


class ModelTier(Enum):
    """Model tiers for routing."""
    ACTION = 0      # No LLM - device commands only
    FAST = 1        # Local small model (1B) - simple queries
    BALANCED = 2    # Local medium model (8B) - reasoning
    POWERFUL = 3    # Local large model (30B+) - complex local
    CLOUD = 4       # Cloud API (OpenAI/Claude) - most capable


@dataclass
class TierConfig:
    """Configuration for a model tier."""
    tier: ModelTier
    name: str
    model_path: Optional[str] = None  # None for ACTION/CLOUD tiers
    model_type: str = "llama-cpp"  # llama-cpp, transformers, openai, anthropic
    n_ctx: int = 2048
    n_gpu_layers: int = -1

    # For cloud APIs
    api_key_env: Optional[str] = None  # e.g., "OPENAI_API_KEY"
    api_model: Optional[str] = None    # e.g., "gpt-4o-mini"
    api_base_url: Optional[str] = None

    # Routing hints
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass
class PoolConfig:
    """Configuration for the model pool."""
    tiers: dict[ModelTier, TierConfig] = field(default_factory=dict)

    # Default tier for fallback
    default_tier: ModelTier = ModelTier.FAST

    # Which tiers to pre-load at startup
    preload_tiers: list[ModelTier] = field(default_factory=lambda: [ModelTier.FAST])


class ModelPool:
    """
    Pool of simultaneously loaded models.

    Unlike model swapping, keeps multiple models in VRAM
    for instant routing with zero swap overhead.
    """

    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or self._default_config()
        self._models: dict[ModelTier, Any] = {}
        self._initialized = False
        self._warmup_state: dict[ModelTier, dict] = {}  # tier -> {warmed: bool, prompt_hash: int}

    def _default_config(self) -> PoolConfig:
        """Create default pool configuration."""
        return PoolConfig(
            tiers={
                ModelTier.ACTION: TierConfig(
                    tier=ModelTier.ACTION,
                    name="action",
                    model_path=None,  # No model needed
                ),
                ModelTier.FAST: TierConfig(
                    tier=ModelTier.FAST,
                    name="llama-1b",
                    model_path="models/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    n_ctx=2048,
                    max_tokens=150,
                ),
                ModelTier.BALANCED: TierConfig(
                    tier=ModelTier.BALANCED,
                    name="hermes-8b",
                    model_path="models/Hermes-3-Llama-3.1-8B-GGUF/Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
                    n_ctx=4096,
                    max_tokens=512,
                ),
                ModelTier.POWERFUL: TierConfig(
                    tier=ModelTier.POWERFUL,
                    name="local-large",
                    model_path=None,  # Configure if you have a 30B+ model
                    n_ctx=4096,
                    max_tokens=1024,
                ),
                ModelTier.CLOUD: TierConfig(
                    tier=ModelTier.CLOUD,
                    name="openai",
                    model_type="openai",
                    api_key_env="OPENAI_API_KEY",
                    api_model="gpt-4o-mini",
                    max_tokens=2048,
                ),
            },
            default_tier=ModelTier.FAST,
            preload_tiers=[ModelTier.FAST, ModelTier.BALANCED],
        )

    async def initialize(self, tiers: Optional[list[ModelTier]] = None) -> None:
        """
        Initialize and load specified model tiers.

        Args:
            tiers: Which tiers to load (defaults to config.preload_tiers)
        """
        tiers_to_load = tiers or self.config.preload_tiers

        for tier in tiers_to_load:
            if tier == ModelTier.ACTION:
                continue  # No model to load

            tier_config = self.config.tiers.get(tier)
            if not tier_config:
                logger.warning("No config for tier %s", tier)
                continue

            try:
                await self._load_tier(tier, tier_config)
            except Exception as e:
                logger.error("Failed to load tier %s: %s", tier, e)

        self._initialized = True
        logger.info("Model pool initialized with tiers: %s",
                   [t.name for t in self._models.keys()])

    async def _load_tier(self, tier: ModelTier, config: TierConfig) -> None:
        """Load a single tier's model."""
        if config.model_type == "llama-cpp":
            await self._load_llama_cpp(tier, config)
        elif config.model_type == "openai":
            await self._load_openai(tier, config)
        elif config.model_type == "anthropic":
            await self._load_anthropic(tier, config)
        else:
            logger.warning("Unknown model type: %s", config.model_type)

    async def _load_llama_cpp(self, tier: ModelTier, config: TierConfig) -> None:
        """Load a llama-cpp model."""
        if not config.model_path:
            logger.warning("No model path for tier %s", tier)
            return

        model_path = Path(config.model_path)
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            return

        logger.info("Loading %s model: %s", tier.name, config.name)

        try:
            from llama_cpp import Llama

            model = Llama(
                model_path=str(model_path),
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                verbose=False,
            )

            self._models[tier] = {
                "type": "llama-cpp",
                "model": model,
                "config": config,
            }

            logger.info("Loaded %s (%s)", tier.name, config.name)

        except Exception as e:
            logger.error("Failed to load %s: %s", tier.name, e)
            raise

    async def _load_openai(self, tier: ModelTier, config: TierConfig) -> None:
        """Set up OpenAI client (lazy - doesn't load anything)."""
        import os

        api_key = os.environ.get(config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found in %s", config.api_key_env)
            return

        self._models[tier] = {
            "type": "openai",
            "api_key": api_key,
            "model": config.api_model,
            "config": config,
        }
        logger.info("OpenAI tier configured: %s", config.api_model)

    async def _load_anthropic(self, tier: ModelTier, config: TierConfig) -> None:
        """Set up Anthropic client (lazy - doesn't load anything)."""
        import os

        api_key = os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("Anthropic API key not found in %s", config.api_key_env)
            return

        self._models[tier] = {
            "type": "anthropic",
            "api_key": api_key,
            "model": config.api_model,
            "config": config,
        }
        logger.info("Anthropic tier configured: %s", config.api_model)

    def get_model(self, tier: ModelTier) -> Optional[dict]:
        """Get a loaded model by tier."""
        return self._models.get(tier)

    def is_tier_available(self, tier: ModelTier) -> bool:
        """Check if a tier is loaded and available."""
        if tier == ModelTier.ACTION:
            return True  # Always available
        return tier in self._models

    def get_available_tiers(self) -> list[ModelTier]:
        """Get list of available tiers."""
        available = [ModelTier.ACTION]  # Always available
        available.extend(self._models.keys())
        return sorted(available, key=lambda t: t.value)

    async def warmup_context(
        self,
        tier: ModelTier,
        system_prompt: str,
    ) -> bool:
        """
        Pre-load system prompt into model KV cache to reduce first-token latency.

        This processes the system prompt tokens without generating output,
        populating the KV cache for faster subsequent generation.

        Args:
            tier: Which model tier to warm up
            system_prompt: The system prompt to pre-load

        Returns:
            True if warmup succeeded, False otherwise
        """
        if tier == ModelTier.ACTION:
            return True  # No warmup needed

        model_info = self._models.get(tier)
        if not model_info:
            logger.warning("Cannot warmup tier %s: not loaded", tier)
            return False

        # Check if already warmed with same prompt
        prompt_hash = hash(system_prompt)
        warmup_info = self._warmup_state.get(tier, {})
        if warmup_info.get("warmed") and warmup_info.get("prompt_hash") == prompt_hash:
            logger.debug("Tier %s already warmed with same prompt", tier)
            return True

        model_type = model_info["type"]

        if model_type == "llama-cpp":
            success = await self._warmup_llama_cpp(model_info, system_prompt)
        elif model_type in ("openai", "anthropic"):
            # Cloud APIs don't benefit from warmup
            success = True
        else:
            logger.warning("Unknown model type for warmup: %s", model_type)
            success = False

        if success:
            self._warmup_state[tier] = {"warmed": True, "prompt_hash": prompt_hash}
            logger.info("Warmed up tier %s with system prompt (%d chars)",
                       tier.name, len(system_prompt))

        return success

    async def _warmup_llama_cpp(
        self,
        model_info: dict,
        system_prompt: str,
    ) -> bool:
        """
        Warm up llama-cpp model by doing a minimal generation.

        This processes the system prompt through the model, populating
        internal caches and warming up CUDA kernels for faster subsequent calls.
        """
        import asyncio

        model = model_info["model"]

        def _warmup():
            try:
                # Do a minimal chat completion to warm up the model
                # This processes the system prompt and warms CUDA kernels
                model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Hi"},
                    ],
                    max_tokens=1,  # Generate minimal tokens
                    temperature=0.0,
                )
                return True

            except Exception as e:
                logger.error("Llama warmup failed: %s", e)
                return False

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _warmup)
        except Exception as e:
            logger.error("Warmup executor failed: %s", e)
            return False

    def is_warmed_up(self, tier: ModelTier) -> bool:
        """Check if a tier has been warmed up."""
        return self._warmup_state.get(tier, {}).get("warmed", False)

    def clear_warmup(self, tier: ModelTier) -> None:
        """Clear warmup state for a tier."""
        if tier in self._warmup_state:
            del self._warmup_state[tier]
            logger.debug("Cleared warmup state for tier %s", tier)

    def clear_all_warmup(self) -> None:
        """Clear warmup state for all tiers."""
        self._warmup_state.clear()
        logger.debug("Cleared all warmup states")

    async def chat(
        self,
        tier: ModelTier,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ):
        """
        Generate a chat response using the specified tier.

        Args:
            tier: Which model tier to use
            messages: Chat messages [{"role": "user", "content": "..."}]
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stream: If True, yield tokens as generated

        Returns:
            Response dict (non-streaming only)
        """
        if stream:
            raise ValueError("Use chat_stream() for streaming responses")

        if tier == ModelTier.ACTION:
            raise ValueError("ACTION tier doesn't support chat")

        model_info = self._models.get(tier)
        if not model_info:
            # Fallback to default tier
            logger.warning("Tier %s not available, using default", tier)
            tier = self.config.default_tier
            model_info = self._models.get(tier)

        if not model_info:
            raise RuntimeError("No models available")

        config = model_info["config"]
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature

        model_type = model_info["type"]

        if model_type == "llama-cpp":
            return await self._chat_llama(model_info, messages, max_tokens, temperature)
        elif model_type == "openai":
            return await self._chat_openai(model_info, messages, max_tokens, temperature)
        elif model_type == "anthropic":
            return await self._chat_anthropic(model_info, messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def chat_stream(
        self,
        tier: ModelTier,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Stream a chat response using the specified tier.

        Args:
            tier: Which model tier to use
            messages: Chat messages [{"role": "user", "content": "..."}]
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Yields:
            String tokens as they are generated
        """
        if tier == ModelTier.ACTION:
            raise ValueError("ACTION tier doesn't support chat")

        model_info = self._models.get(tier)
        if not model_info:
            # Fallback to default tier
            logger.warning("Tier %s not available, using default", tier)
            tier = self.config.default_tier
            model_info = self._models.get(tier)

        if not model_info:
            raise RuntimeError("No models available")

        config = model_info["config"]
        max_tokens = max_tokens or config.max_tokens
        temperature = temperature or config.temperature

        model_type = model_info["type"]

        if model_type == "llama-cpp":
            async for token in self._chat_llama_stream(model_info, messages, max_tokens, temperature):
                yield token
        elif model_type == "openai":
            async for token in self._chat_openai_stream(model_info, messages, max_tokens, temperature):
                yield token
        elif model_type == "anthropic":
            async for token in self._chat_anthropic_stream(model_info, messages, max_tokens, temperature):
                yield token
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def _chat_llama(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat using llama-cpp model."""
        import asyncio

        model = model_info["model"]

        def _generate():
            return model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        result = await asyncio.get_event_loop().run_in_executor(None, _generate)

        return {
            "response": result["choices"][0]["message"]["content"].strip(),
            "usage": result.get("usage", {}),
            "tier": model_info["config"].tier.name,
        }

    async def _chat_llama_stream(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ):
        """Stream chat using llama-cpp model."""
        import asyncio

        model = model_info["model"]
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        done = asyncio.Event()

        def producer():
            try:
                stream = model.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        loop.call_soon_threadsafe(queue.put_nowait, content)
            finally:
                loop.call_soon_threadsafe(done.set)

        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(producer)

        while not done.is_set() or not queue.empty():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                continue

        executor.shutdown(wait=False)

    async def _chat_openai(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. pip install openai")

        client = AsyncOpenAI(api_key=model_info["api_key"])

        response = await client.chat.completions.create(
            model=model_info["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            "tier": "CLOUD",
        }

    async def _chat_openai_stream(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ):
        """Stream chat using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. pip install openai")

        client = AsyncOpenAI(api_key=model_info["api_key"])

        stream = await client.chat.completions.create(
            model=model_info["model"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _chat_anthropic(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Chat using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=model_info["api_key"])

        # Convert messages format (Anthropic uses different format)
        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        response = await client.messages.create(
            model=model_info["model"],
            max_tokens=max_tokens,
            system=system_msg or "",
            messages=anthropic_messages,
        )

        return {
            "response": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "tier": "CLOUD",
        }

    async def _chat_anthropic_stream(
        self,
        model_info: dict,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ):
        """Stream chat using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. pip install anthropic")

        client = anthropic.AsyncAnthropic(api_key=model_info["api_key"])

        system_msg = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                anthropic_messages.append(msg)

        async with client.messages.stream(
            model=model_info["model"],
            max_tokens=max_tokens,
            system=system_msg or "",
            messages=anthropic_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def shutdown(self) -> None:
        """Unload all models and free resources."""
        for tier, model_info in self._models.items():
            if model_info["type"] == "llama-cpp":
                logger.info("Unloading %s", tier.name)
                del model_info["model"]

        self._models.clear()
        self._warmup_state.clear()
        self._initialized = False

        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model pool shutdown complete")


# Global instance
_pool: Optional[ModelPool] = None


def get_model_pool() -> ModelPool:
    """Get or create the global model pool."""
    global _pool
    if _pool is None:
        _pool = ModelPool()
    return _pool


async def initialize_pool(tiers: Optional[list[ModelTier]] = None) -> ModelPool:
    """Initialize the global model pool."""
    pool = get_model_pool()
    await pool.initialize(tiers)
    return pool
