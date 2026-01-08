"""
Vision query endpoint.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...services.protocols import VLMService
from ..dependencies import get_vlm

router = APIRouter()


@router.post("/vision")
async def query_vision(
    image_file: UploadFile = File(...),
    prompt_text: Optional[str] = Form(None),
    vlm: VLMService = Depends(get_vlm),
):
    """
    Process an image with optional text prompt using the active VLM.

    If no prompt is provided, defaults to "Describe this image."
    """
    image_bytes = await image_file.read()
    result = await vlm.process_vision(image_bytes=image_bytes, prompt=prompt_text)

    # Include file info in response
    result["file_received"] = image_file.filename
    return result
