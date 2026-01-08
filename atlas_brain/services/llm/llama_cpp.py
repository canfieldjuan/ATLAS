"""
LLM implementation using llama-cpp-python.

Supports any GGUF model (Qwen, Mistral, LLaMA, etc.)
Efficient inference with GPU acceleration.
"""

from pathlib import Path
from typing import Any, Optional

from ..base import BaseModelService, InferenceTimer
from ..protocols import Message, ModelInfo
from ..registry import register_llm


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

        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            self.logger.info("LLM loaded successfully (ctx=%d)", self._n_ctx)

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

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
            result = self._llm.create_chat_completion(
                messages=llm_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

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


@register_llm("transformers")
class TransformersLLM(BaseModelService):
    """
    LLM implementation using Hugging Face Transformers.

    Supports any causal LM model from the Hub.
    """

    CAPABILITIES = ["text", "chat", "reasoning"]

    def __init__(
        self,
        model_id: str = "microsoft/phi-3-mini-4k-instruct",
        cache_path: Optional[Path] = None,
        torch_dtype: str = "auto",
    ):
        super().__init__(
            name="transformers",
            model_id=model_id,
            cache_path=cache_path or Path("models/llm"),
            log_file=Path("logs/atlas_llm.log"),
        )
        self._torch_dtype = torch_dtype
        self._tokenizer = None
        self._pipeline = None

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
        """Load the model from Hugging Face."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading model: %s", self.model_id)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            # Determine torch dtype
            if self._torch_dtype == "auto":
                dtype = torch.float16 if self.device == "cuda" else torch.float32
            else:
                dtype = getattr(torch, self._torch_dtype)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_path),
                trust_remote_code=True,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_path),
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )

            self.logger.info("Model loaded on %s", self.device)

        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers accelerate"
            )

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self.logger.info("Unloading model: %s", self.name)
            del self._model
            del self._tokenizer
            del self._pipeline
            self._model = None
            self._tokenizer = None
            self._pipeline = None
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

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        self.logger.info("Generating response...")

        with InferenceTimer() as timer:
            outputs = self._pipeline(
                full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_text = outputs[0]["generated_text"][len(full_prompt):].strip()
        metrics = self.gather_metrics(timer.duration)

        return {
            "prompt": prompt,
            "response": response_text,
            "metrics": metrics.to_dict(),
        }

    def chat(
        self,
        messages: list[Message],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate a response in a chat conversation."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Apply chat template
        chat_messages = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        self.logger.info("Chat with %d messages", len(messages))

        with InferenceTimer() as timer:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_text = outputs[0]["generated_text"][len(prompt):].strip()
        metrics = self.gather_metrics(timer.duration)

        return {
            "response": response_text,
            "message": {"role": "assistant", "content": response_text},
            "metrics": metrics.to_dict(),
        }
