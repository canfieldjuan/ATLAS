"""SAM 3 Video Object Segmentation service."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..base import BaseModelService, InferenceTimer
from ..protocols import ModelInfo
from ..registry import register_vos
from ...config import settings

logger = logging.getLogger(__name__)


@register_vos("sam3")
class SAM3Service(BaseModelService):
    """Meta SAM 3 video object segmentation service.

    SAM 3 is Meta's Segment Anything Model 3 (Nov 2025 release).
    - 848M parameters
    - Unified image + video segmentation
    - Text prompt support (open vocabulary)
    - 30ms inference for 100+ objects
    - Real-time video streaming
    """

    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        bpe_path: Optional[str] = None,
        load_from_hf: Optional[bool] = None,
    ):
        """Initialize SAM 3 service.

        Args:
            device: torch device ('cuda', 'cpu') - defaults to config
            dtype: Model dtype ('float16', 'float32', 'bfloat16') - defaults to config
            bpe_path: Path to BPE vocab file - defaults to config
            load_from_hf: Load model from HuggingFace - defaults to config
        """
        super().__init__(
            name="sam3",
            model_id="facebook/sam3",
            cache_path=Path("models/sam3"),
            log_file=Path("logs/atlas_vos.log"),
        )

        vos_config = settings.vos
        self._dtype_str = dtype or vos_config.dtype
        self._dtype = getattr(torch, self._dtype_str)
        self._bpe_path = bpe_path or vos_config.bpe_path
        self._load_from_hf = load_from_hf if load_from_hf is not None else vos_config.load_from_hf

        if device:
            self._device = device

        self._processor = None

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata about the current model."""
        return ModelInfo(
            name=self.name,
            model_id=self.model_id,
            is_loaded=self.is_loaded,
            device=self.device,
            capabilities=["image-segmentation", "video-segmentation", "text-prompts"],
        )

    def load(self) -> None:
        """Load the SAM 3 model into memory."""
        if self._model is not None:
            self.logger.debug("SAM 3 already loaded")
            return

        self.logger.info("Loading SAM 3 model...")
        try:
            self._model, self._processor = self._load_model_sync()
            self.logger.info("SAM 3 loaded successfully on %s", self.device)
        except Exception as e:
            self.logger.error("Failed to load SAM 3: %s", e)
            raise

    def _load_model_sync(self) -> Tuple[Any, Any]:
        """Synchronous model loading."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = self._bpe_path
        if bpe_path is None:
            bpe_path = os.path.expanduser(
                "~/models/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            )

        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            load_from_HF=self._load_from_hf,
            enable_segmentation=True,
        )

        processor = Sam3Processor(model)
        model.eval()
        return model, processor

    def unload(self) -> None:
        """Unload model from memory to free resources."""
        if self._model is None:
            return

        self.logger.info("Unloading SAM 3...")
        self._model = None
        self._processor = None
        self._clear_gpu_memory()
        self.logger.info("SAM 3 unloaded")

    async def segment_image(
        self,
        image: Image.Image,
        prompts: Optional[List[str]] = None,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        box_prompts: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """Segment objects in an image.

        Args:
            image: PIL Image
            prompts: Text prompts (e.g., ["person", "car"])
            point_prompts: Click points [(x1, y1), (x2, y2), ...]
            box_prompts: Bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            Dict with 'masks', 'scores', 'labels', 'boxes', 'metrics'
        """
        if not self.is_loaded:
            self.load()

        self.logger.debug("Segmenting image with prompts: %s", prompts)

        loop = asyncio.get_event_loop()
        with InferenceTimer() as timer:
            result = await loop.run_in_executor(
                None,
                self._segment_image_sync,
                image,
                prompts,
                point_prompts,
                box_prompts,
            )

        result["metrics"] = self.gather_metrics(timer.duration).to_dict()
        return result

    def _segment_image_sync(
        self,
        image: Image.Image,
        prompts: Optional[List[str]],
        point_prompts: Optional[List[Tuple[int, int]]],
        box_prompts: Optional[List[Tuple[int, int, int, int]]],
    ) -> Dict[str, Any]:
        """Synchronous image segmentation using official SAM 3 API."""
        inference_state = self._processor.set_image(image)

        prompt = "object"
        if prompts and len(prompts) > 0:
            prompt = prompts[0]

        output = self._processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

        masks = output["masks"]
        boxes = output.get("boxes", None)
        scores = output.get("scores", None)

        masks_np = masks.cpu().numpy() if torch.is_tensor(masks) else masks
        scores_np = None
        if scores is not None:
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
        boxes_np = None
        if boxes is not None:
            boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes

        num_masks = len(masks_np) if masks_np is not None else 0
        labels = prompts if prompts else [("object_%d" % i) for i in range(num_masks)]

        return {
            "masks": masks_np,
            "scores": scores_np,
            "labels": labels,
            "boxes": boxes_np,
            "image_shape": image.size,
        }

    async def segment_video(
        self,
        video_path: str,
        prompts: Optional[List[str]] = None,
        frame_skip: int = 1,
    ) -> List[Dict[str, Any]]:
        """Segment objects across video frames.

        Args:
            video_path: Path to video file
            prompts: Text prompts for objects to track
            frame_skip: Process every Nth frame (1 = all frames)

        Returns:
            List of frame results with masks, scores, labels
        """
        if not self.is_loaded:
            self.load()

        self.logger.info("Segmenting video: %s", video_path)

        loop = asyncio.get_event_loop()
        with InferenceTimer() as timer:
            results = await loop.run_in_executor(
                None,
                self._segment_video_sync,
                video_path,
                prompts,
                frame_skip,
            )

        metrics = self.gather_metrics(timer.duration).to_dict()
        for r in results:
            r["metrics"] = metrics

        return results

    def _segment_video_sync(
        self,
        video_path: str,
        prompts: Optional[List[str]],
        frame_skip: int,
    ) -> List[Dict[str, Any]]:
        """Synchronous video segmentation."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        frame_results = []
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                result = self._segment_image_sync(image, prompts, None, None)
                result["frame_idx"] = frame_idx
                frame_results.append(result)

                frame_idx += 1

                if frame_idx % 10 == 0:
                    self.logger.debug("Processed %d frames", frame_idx)

        finally:
            cap.release()

        self.logger.info("Segmented %d frames from video", len(frame_results))
        return frame_results
