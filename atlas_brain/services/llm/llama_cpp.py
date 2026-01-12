"""
LLM implementation using llama-cpp-python.

Supports any GGUF model (Qwen, Mistral, LLaMA, etc.)
Efficient inference with GPU acceleration.
"""

import os
from pathlib import Path
from typing import Any, Optional

# Ensure CUDA libraries are findable (Ollama installs them here)
_cuda_paths = ["/usr/local/lib/ollama/cuda_v12", "/usr/local/lib/ollama/cuda_v13"]
for _path in _cuda_paths:
    if os.path.exists(_path):
        os.environ["LD_LIBRARY_PATH"] = f"{_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        break

from ..base import BaseModelService, InferenceTimer
from ..protocols import Message, ModelInfo
from ..registry import register_llm


def _get_free_vram_mb() -> int:
    """Get free VRAM in MB. Returns 0 if CUDA not available."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return 0


@register_llm("llama-cpp")
class LlamaCppLLM(BaseModelService):
    """
    LLM implementation using llama-cpp-python.

    Supports GGUF format models with efficient CPU/GPU inference.
    """

    CAPABILITIES = ["text", "chat", "reasoning"]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_id: str = "local-llm",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        cache_path: Optional[Path] = None,
    ):
        super().__init__(
            name="llama-cpp",
            model_id=model_id,
            cache_path=cache_path or Path("models/llm"),
            log_file=Path("logs/atlas_llm.log"),
        )
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._llm = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._llm is not None

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device=self.device,
            capabilities=self.CAPABILITIES,
        )

    def load(self) -> None:
        """Load the LLM model."""
        if self._llm is not None:
            self.logger.info("Model already loaded")
            return

        if self._model_path is None:
            raise ValueError(
                "model_path is required. Download a GGUF model and provide the path."
            )

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        self.logger.info("Loading LLM: %s", self._model_path)

        # Check available VRAM and adjust GPU layers if needed
        n_gpu_layers = self._n_gpu_layers
        free_vram = _get_free_vram_mb()
        model_size_mb = self._model_path.stat().st_size / (1024 * 1024)

        self.logger.info("Free VRAM: %d MB, Model size: %.0f MB", free_vram, model_size_mb)

        # If requesting all layers (-1) but not enough VRAM, estimate safe layers
        if n_gpu_layers == -1 and free_vram > 0:
            # Rough estimate: each layer ~300MB for a 7B model, scale by model size
            estimated_per_layer = model_size_mb / 40  # Assume ~40 layers typical
            safe_layers = int((free_vram * 0.8) / estimated_per_layer)  # Use 80% of free
            if safe_layers < 40:
                n_gpu_layers = max(0, min(safe_layers, 35))  # Cap at 35 layers
                self.logger.warning(
                    "Limited VRAM detected. Using %d GPU layers instead of all",
                    n_gpu_layers
                )

        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            self.logger.info(
                "LLM loaded successfully (ctx=%d, gpu_layers=%d)",
                self._n_ctx, n_gpu_layers
            )

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )
        except ValueError as e:
            if "out of memory" in str(e).lower() or "Failed to load" in str(e):
                # Retry with fewer GPU layers
                self.logger.warning("GPU memory error, retrying with CPU fallback")
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=str(self._model_path),
                    n_ctx=self._n_ctx,
                    n_gpu_layers=0,  # CPU only
                    verbose=False,
                )
                self.logger.info("LLM loaded on CPU (ctx=%d)", self._n_ctx)
            else:
                raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._llm is not None:
            self.logger.info("Unloading LLM: %s", self.name)
            del self._llm
            self._llm = None
            self._clear_gpu_memory()
            self.logger.info("LLM unloaded")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stop: Stop sequences

        Returns:
            Dict with response text and metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        self.logger.info("Generating response for prompt: %s...", prompt[:50])

        with InferenceTimer() as timer:
            result = self._llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                echo=False,
            )

        response_text = result["choices"][0]["text"].strip()
        metrics = self.gather_metrics(timer.duration)

        self.logger.info(
            "Generated %d tokens in %.0fms",
            result["usage"]["completion_tokens"],
            metrics.duration_ms,
        )

        return {
            "prompt": prompt,
            "response": response_text,
            "usage": {
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
            },
            "metrics": metrics.to_dict(),
        }

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate a response in a chat conversation.

        Args:
            messages: List of Message objects (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Dict with response text and metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to llama-cpp format
        llm_messages = [{"role": m.role, "content": m.content} for m in messages]

        self.logger.info("Chat with %d messages", len(messages))

        with InferenceTimer() as timer:
            try:
                result = self._llm.create_chat_completion(
                    messages=llm_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
            except Exception as e:
                if "undefined" in str(e) or "jinja" in str(e).lower():
                    self.logger.warning(
                        "Chat template error, using fallback format: %s", e
                    )
                    result = self._chat_fallback(
                        llm_messages, max_tokens, temperature, stop
                    )
                else:
                    raise

        response_text = result["choices"][0]["message"]["content"].strip()
        metrics = self.gather_metrics(timer.duration)

        self.logger.info(
            "Chat response: %d tokens in %.0fms",
            result["usage"]["completion_tokens"],
            metrics.duration_ms,
        )

        return {
            "response": response_text,
            "message": {"role": "assistant", "content": response_text},
            "usage": result["usage"],
            "metrics": metrics.to_dict(),
        }

    async def chat_stream(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ):
        """
        Stream chat response token by token.

        Yields tokens as they're generated for low-latency responses.

        Args:
            messages: List of Message objects (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Yields:
            String tokens as they're generated
        """
        import asyncio

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        llm_messages = [{"role": m.role, "content": m.content} for m in messages]

        self.logger.info("Streaming chat with %d messages", len(messages))

        def _stream_sync():
            """Synchronous generator for streaming."""
            try:
                stream = self._llm.create_chat_completion(
                    messages=llm_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
            except Exception as e:
                if "undefined" in str(e) or "jinja" in str(e).lower():
                    # Fallback to non-streaming on template error
                    self.logger.warning("Chat template error, falling back: %s", e)
                    result = self._chat_fallback(
                        llm_messages, max_tokens, temperature, stop
                    )
                    yield result["choices"][0]["message"]["content"]
                else:
                    raise

        # Run streaming in executor to avoid blocking
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        done = asyncio.Event()

        def producer():
            try:
                for token in _stream_sync():
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(done.set)

        # Start producer in thread
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(producer)

        # Yield tokens from queue
        while not done.is_set() or not queue.empty():
            try:
                token = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                continue

        executor.shutdown(wait=False)

    def _chat_fallback(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: Optional[list[str]],
    ) -> dict[str, Any]:
        """Fallback chat using manual prompt formatting."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}</s>")

        prompt_parts.append("<|assistant|>\n")
        full_prompt = "\n".join(prompt_parts)

        result = self._llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "<|user|>", "<|system|>"],
            echo=False,
        )

        response_text = result["choices"][0]["text"].strip()

        return {
            "choices": [{"message": {"content": response_text}}],
            "usage": result["usage"],
        }
