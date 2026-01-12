"""
LLM implementation using Hugging Face Transformers with Flash Attention 2.

Optimized for low-latency inference with:
- Flash Attention 2 for faster attention computation
- BF16/FP16 precision for memory efficiency
- KV-cache for faster generation
- Streaming support for real-time responses
"""

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from ..base import BaseModelService, InferenceTimer
from ..protocols import Message, ModelInfo
from ..registry import register_llm

logger = logging.getLogger("atlas.llm.transformers_flash")


def _check_flash_attention_available() -> bool:
    """Check if flash attention 2 is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Check compute capability (need SM 8.0+ for flash attention)
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning(
                "Flash Attention requires compute capability 8.0+ (Ampere), "
                "detected %d.x. Falling back to SDPA.",
                major
            )
            return False
        # Try importing flash_attn
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "flash-attn not installed. Install with: "
                "pip install flash-attn --no-build-isolation"
            )
            return False
    except Exception:
        return False


def _get_gpu_memory_info() -> dict:
    """Get GPU memory information."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            free = total - reserved
            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2),
            }
    except Exception:
        pass
    return {}


@register_llm("transformers-flash")
class TransformersFlashLLM(BaseModelService):
    """
    High-performance LLM using Transformers with Flash Attention 2.

    Features:
    - Flash Attention 2 for 2-4x faster attention
    - BF16 precision for quality + efficiency
    - Streaming generation for low TTFT
    - Automatic GPU memory management

    Recommended models:
    - meta-llama/Llama-3.1-8B-Instruct (best quality/speed balance)
    - meta-llama/Llama-3.2-3B-Instruct (faster, smaller)
    - mistralai/Mistral-7B-Instruct-v0.3
    - NousResearch/Hermes-3-Llama-3.1-8B
    """

    CAPABILITIES = ["text", "chat", "reasoning", "streaming"]

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype: str = "bfloat16",  # bfloat16, float16, or auto
        max_memory_gb: Optional[float] = None,  # Limit GPU memory usage
        use_flash_attention: bool = True,
        cache_path: Optional[Path] = None,
    ):
        super().__init__(
            name="transformers-flash",
            model_id=model_id,
            cache_path=cache_path or Path("models/transformers"),
            log_file=Path("logs/atlas_llm.log"),
        )
        self._torch_dtype_str = torch_dtype
        self._max_memory_gb = max_memory_gb
        self._use_flash_attention = use_flash_attention
        self._tokenizer = None
        self._streamer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device=self.device,
            capabilities=self.CAPABILITIES,
            extra={
                "flash_attention": self._use_flash_attention and _check_flash_attention_available(),
                "dtype": self._torch_dtype_str,
                "gpu_memory": _get_gpu_memory_info(),
            },
        )

    def load(self) -> None:
        """Load the model with flash attention optimization."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading model: %s", self.model_id)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "Required packages not installed. Install with:\n"
                "pip install transformers accelerate bitsandbytes"
            )

        # Determine torch dtype
        if self._torch_dtype_str == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        elif self._torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self._torch_dtype_str == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.logger.info("Using dtype: %s", torch_dtype)

        # Determine attention implementation
        flash_available = _check_flash_attention_available()
        if self._use_flash_attention and flash_available:
            attn_implementation = "flash_attention_2"
            self.logger.info("Using Flash Attention 2")
        else:
            attn_implementation = "sdpa"  # Scaled Dot Product Attention (PyTorch 2.0+)
            self.logger.info("Using SDPA (PyTorch native)")

        # Memory management
        device_map = "auto"
        max_memory = None
        if self._max_memory_gb and torch.cuda.is_available():
            max_memory = {0: f"{self._max_memory_gb}GiB", "cpu": "32GiB"}
            self.logger.info("Limiting GPU memory to %.1f GB", self._max_memory_gb)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_path),
            trust_remote_code=True,
            padding_side="left",  # Better for batch inference
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with optimizations
        self.logger.info("Loading model weights (this may take a moment)...")

        model_kwargs = {
            "cache_dir": str(self.cache_path),
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
        }

        if max_memory:
            model_kwargs["max_memory"] = max_memory

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        # Enable inference optimizations
        self._model.eval()

        # Log memory usage
        mem_info = _get_gpu_memory_info()
        self.logger.info(
            "Model loaded. GPU memory: %.1f/%.1f GB used",
            mem_info.get("allocated_gb", 0),
            mem_info.get("total_gb", 0),
        )

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            self.logger.info("Unloading model: %s", self.model_id)
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._clear_gpu_memory()
            self.logger.info("Model unloaded")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate text from a prompt."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]

        self.logger.info("Generating response for %d input tokens...", input_length)

        with InferenceTimer() as timer:
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                )

        # Decode response (only new tokens)
        response_ids = outputs[0][input_length:]
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        # Handle stop sequences
        if stop:
            for seq in stop:
                if seq in response_text:
                    response_text = response_text.split(seq)[0]

        metrics = self.gather_metrics(timer.duration)
        completion_tokens = len(response_ids)

        self.logger.info(
            "Generated %d tokens in %.0fms (%.1f tok/s)",
            completion_tokens,
            metrics.duration_ms,
            completion_tokens / (timer.duration + 0.001),
        )

        return {
            "prompt": prompt,
            "response": response_text.strip(),
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
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
        """Generate a chat response."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Convert messages
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]

        self.logger.info("Chat with %d messages (%d tokens)", len(messages), input_length)

        with InferenceTimer() as timer:
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                )

        # Decode response
        response_ids = outputs[0][input_length:]
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        # Handle stop sequences
        if stop:
            for seq in stop:
                if seq in response_text:
                    response_text = response_text.split(seq)[0]

        metrics = self.gather_metrics(timer.duration)
        completion_tokens = len(response_ids)

        self.logger.info(
            "Chat response: %d tokens in %.0fms (%.1f tok/s)",
            completion_tokens,
            metrics.duration_ms,
            completion_tokens / (timer.duration + 0.001),
        )

        return {
            "response": response_text.strip(),
            "message": {"role": "assistant", "content": response_text.strip()},
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": completion_tokens,
                "total_tokens": input_length + completion_tokens,
            },
            "metrics": metrics.to_dict(),
        }

    async def chat_stream(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat response token by token.

        Uses a background thread for generation while yielding tokens
        asynchronously for low-latency streaming.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        from transformers import TextIteratorStreamer

        # Convert messages
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        self.logger.info("Streaming chat with %d messages", len(messages))

        # Create streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else None,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }

        # Run generation in background thread
        def generate_in_thread():
            with torch.inference_mode():
                self._model.generate(**gen_kwargs)

        thread = threading.Thread(target=generate_in_thread)
        thread.start()

        # Yield tokens as they're generated
        try:
            for text_chunk in streamer:
                if text_chunk:
                    # Check for stop sequences
                    if stop:
                        for seq in stop:
                            if seq in text_chunk:
                                text_chunk = text_chunk.split(seq)[0]
                                if text_chunk:
                                    yield text_chunk
                                return
                    yield text_chunk
                # Small sleep to allow other async tasks
                await asyncio.sleep(0)
        finally:
            thread.join(timeout=1.0)

    def get_context_length(self) -> int:
        """Get the model's maximum context length."""
        if self._model is not None:
            return getattr(self._model.config, "max_position_embeddings", 4096)
        return 4096

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Rough estimate
        return len(text) // 4
