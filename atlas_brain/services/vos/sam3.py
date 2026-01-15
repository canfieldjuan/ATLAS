"""SAM 3 Video Object Segmentation service."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAM3Service:
    """Meta SAM 3 video object segmentation service.
    
    SAM 3 is Meta's Segment Anything Model 3 (Nov 2025 release).
    - 1.7B parameters
    - Unified image + video segmentation
    - Text prompt support
    - 30ms inference for 100+ objects
    - Real-time video streaming
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float16",
    ):
        """Initialize SAM 3 service.
        
        Args:
            model_path: Path to SAM 3 model (default: HuggingFace cache)
            device: torch device ('cuda', 'cpu')
            dtype: Model dtype ('float16', 'float32', 'bfloat16')
        """
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model_path = model_path or "facebook/sam3"
        
        self.model = None
        self.processor = None
        self._loaded = False
        
        logger.info(f"SAM3Service initialized (device={device}, dtype={dtype})")

    async def load(self) -> None:
        """Load SAM 3 model asynchronously."""
        if self._loaded:
            logger.debug("SAM 3 already loaded")
            return

        logger.info("Loading SAM 3 model...")
        try:
            from transformers import AutoModel, AutoProcessor
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model, self.processor = await loop.run_in_executor(
                None,
                self._load_model_sync,
            )
            
            self._loaded = True
            logger.info("SAM 3 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3: {e}")
            raise

    def _load_model_sync(self) -> Tuple[Any, Any]:
        """Synchronous model loading (runs in thread pool)."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # Build model using official SAM 3 API
        model = build_sam3_image_model(
            config_file=None,  # Will download from HF
            ckpt_path=None,  # Will download from HF
            device=self.device,
        )
        
        processor = Sam3Processor(model)
        
        model.eval()
        return model, processor

    async def unload(self) -> None:
        """Unload model from memory."""
        if not self._loaded:
            return
            
        logger.info("Unloading SAM 3...")
        self.model = None
        self.processor = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SAM 3 unloaded")

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
            Dict with 'masks', 'scores', 'labels'
        """
        if not self._loaded:
            await self.load()

        logger.debug(f"Segmenting image with prompts: {prompts}")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._segment_image_sync,
            image,
            prompts,
            point_prompts,
            box_prompts,
        )
        
        return result

    def _segment_image_sync(
        self,
        image: Image.Image,
        prompts: Optional[List[str]],
        point_prompts: Optional[List[Tuple[int, int]]],
        box_prompts: Optional[List[Tuple[int, int, int, int]]],
    ) -> Dict[str, Any]:
        """Synchronous image segmentation using official SAM 3 API."""
        # Set the image in the processor
        inference_state = self.processor.set_image(image)
        
        # Use text prompt if provided
        if prompts and len(prompts) > 0:
            prompt = prompts[0]  # Take first prompt
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt,
            )
        elif box_prompts and len(box_prompts) > 0:
            # TODO: Add box prompt support when needed
            # For now, use text prompt with generic "object"
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt="object",
            )
        else:
            # Default to detecting all objects
            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt="object",
            )
        
        # Extract results
        masks = output["masks"]  # Shape: [N, H, W]
        boxes = output.get("boxes", None)  # Shape: [N, 4]
        scores = output.get("scores", None)  # Shape: [N]
        
        return {
            "masks": masks.cpu().numpy() if torch.is_tensor(masks) else masks,
            "scores": scores.cpu().numpy() if torch.is_tensor(scores) and scores is not None else scores,
            "labels": prompts if prompts else [f"object_{i}" for i in range(len(masks))],
            "boxes": boxes.cpu().numpy() if torch.is_tensor(boxes) and boxes is not None else boxes,
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
        if not self._loaded:
            await self.load()

        logger.info(f"Segmenting video: {video_path}")
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._segment_video_sync,
            video_path,
            prompts,
            frame_skip,
        )
        
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
                
                # Skip frames if needed
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Segment frame
                result = self._segment_image_sync(image, prompts, None, None)
                result["frame_idx"] = frame_idx
                frame_results.append(result)
                
                frame_idx += 1
                
                if frame_idx % 10 == 0:
                    logger.debug(f"Processed {frame_idx} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Segmented {len(frame_results)} frames from video")
        return frame_results

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        return allocated
