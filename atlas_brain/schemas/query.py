from typing import Optional

from pydantic import BaseModel


class TextQueryRequest(BaseModel):
    """
    The request model for a simple text query.
    """
    query_text: str
    session_id: Optional[str] = None
