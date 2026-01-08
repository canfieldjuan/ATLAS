"""
Moondream2 VLM implementation.

A lightweight vision-language model for text and image understanding.
"""

import io
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_vlm


@register_vlm("moondream")
class MoondreamVLM(BaseModelService):
    """Moondream2 implementation of VLMService."""

    MODEL_ID = "vikhyatk/moondream2"
    CAPABILITIES = ["text", "vision"]

    def __init__(
        self,
        cache_path: Optional[Path] = None,
    ):
        super().__init__(
            name="moondream",
            model_id=self.MODEL_ID,
            cache_path=cache_path or Path("models/moondream"),
            log_file=Path("logs/atlas_vlm.log"),
        )
        self._tokenizer = None

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
        """Load the moondream2 model and tokenizer."""
        if self._model is not None:
            self.logger.info("Model already loaded")
            return

        self.logger.info("Loading model: %s", self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=str(self.cache_path),
        ).to(device=self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_path),
        )
        self.logger.info("Model loaded successfully on %s", self.device)

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            self.logger.info("Unloading model: %s", self.name)
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._clear_gpu_memory()
            self.logger.info("Model unloaded")

    def process_text(self, query: str) -> dict[str, Any]:
        """Process a text-only query using a dummy image."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.logger.info("Text query received: %s", query[:100])

        with InferenceTimer() as timer:
            # For text-only queries, use a dummy blank image
            dummy_image = Image.new("RGB", (1, 1))
            image_embeds = self._model.encode_image(dummy_image)

            response_text = self._model.answer_question(
                image_embeds=image_embeds,
                question=query,
                tokenizer=self._tokenizer,
            )

        metrics = self.gather_metrics(timer.duration)
        self.logger.info(
            "Text query completed in %.2f ms on %s",
            metrics.duration_ms,
            metrics.device,
        )

        return {
            "query_received": query,
            "response": response_text,
            "metrics": metrics.to_dict(),
        }

    async def process_vision(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Process an image with optional text prompt."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = prompt or "Describe this image."
        self.logger.info("Vision query received: prompt=%s", prompt[:100])

        try:
            image = Image.open(io.BytesIO(image_bytes))

            with InferenceTimer() as timer:
                image_embeds = self._model.encode_image(image)
                response_text = self._model.answer_question(
                    image_embeds=image_embeds,
                    question=prompt,
                    tokenizer=self._tokenizer,
                )

            metrics = self.gather_metrics(timer.duration)
            self.logger.info(
                "Vision query completed in %.2f ms on %s",
                metrics.duration_ms,
                metrics.device,
            )

            return {
                "prompt_received": prompt,
                "response": response_text,
                "metrics": metrics.to_dict(),
            }
        except Exception as exc:
            self.logger.exception("Error processing vision query")
            return {"error": f"Failed to process image: {exc}"}
