"""Video Object Segmentation API endpoints."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image

from atlas_brain.api.dependencies import get_vos
from atlas_brain.services.protocols import VOSService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/segment/image")
async def segment_image(
    image: UploadFile = File(...),
    prompts: Optional[str] = Form(None),
    vos: VOSService = Depends(get_vos),
):
    """Segment objects in an uploaded image.

    Args:
        image: Image file upload
        prompts: Comma-separated text prompts (e.g., "person,car,tree")
        vos: VOSService dependency

    Returns:
        Segmentation results with masks, scores, labels
    """
    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        prompt_list = None
        if prompts:
            prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

        result = await vos.segment_image(
            image=pil_image,
            prompts=prompt_list,
        )

        if hasattr(result["masks"], "tolist"):
            result["masks"] = result["masks"].tolist()
        if result.get("scores") is not None and hasattr(result["scores"], "tolist"):
            result["scores"] = result["scores"].tolist()
        if result.get("boxes") is not None and hasattr(result["boxes"], "tolist"):
            result["boxes"] = result["boxes"].tolist()

        return {"status": "success", "data": result}

    except Exception as e:
        logger.error("Image segmentation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment/video")
async def segment_video(
    video: UploadFile = File(...),
    prompts: Optional[str] = Form(None),
    frame_skip: int = Form(1),
    vos: VOSService = Depends(get_vos),
):
    """Segment objects across video frames.

    Args:
        video: Video file upload
        prompts: Comma-separated text prompts
        frame_skip: Process every Nth frame (default: 1)
        vos: VOSService dependency

    Returns:
        List of frame segmentation results
    """
    try:
        temp_path = Path("/tmp") / ("atlas_video_%s" % video.filename)
        with open(temp_path, "wb") as f:
            f.write(await video.read())

        prompt_list = None
        if prompts:
            prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

        results = await vos.segment_video(
            video_path=str(temp_path),
            prompts=prompt_list,
            frame_skip=frame_skip,
        )

        for result in results:
            if hasattr(result["masks"], "tolist"):
                result["masks"] = result["masks"].tolist()
            if result.get("scores") is not None and hasattr(result["scores"], "tolist"):
                result["scores"] = result["scores"].tolist()
            if result.get("boxes") is not None and hasattr(result["boxes"], "tolist"):
                result["boxes"] = result["boxes"].tolist()

        temp_path.unlink(missing_ok=True)

        return {
            "status": "success",
            "frame_count": len(results),
            "data": results,
        }

    except Exception as e:
        logger.error("Video segmentation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def vos_status(vos: VOSService = Depends(get_vos)):
    """Get VOS service status."""
    info = vos.model_info
    return {
        "loaded": info.is_loaded,
        "model": info.name,
        "model_id": info.model_id,
        "device": info.device,
        "capabilities": info.capabilities,
    }
