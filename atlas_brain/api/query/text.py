"""
Text query endpoint.
"""

from fastapi import APIRouter, Depends

from ...schemas.query import TextQueryRequest
from ...services.protocols import VLMService
from ..dependencies import get_vlm

router = APIRouter()


@router.post("/text")
async def query_text(
    request: TextQueryRequest,
    vlm: VLMService = Depends(get_vlm),
):
    """
    Process a text query using the active VLM.

    The VLM will generate a response based on the input text.
    """
    return vlm.process_text(query=request.query_text)
