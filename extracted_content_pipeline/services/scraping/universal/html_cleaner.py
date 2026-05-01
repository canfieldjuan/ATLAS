from __future__ import annotations

import os
import re
from html import unescape

if os.getenv("EXTRACTED_PIPELINE_STANDALONE", "0") == "1":
    _SCRIPT_STYLE_RE = re.compile(
        r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    _TAG_RE = re.compile(r"<[^>]+>")
    _SPACE_RE = re.compile(r"\s+")

    def html_to_text(html: str | None, max_chars: int = 30000) -> str:
        text = _SCRIPT_STYLE_RE.sub(" ", str(html or ""))
        text = _TAG_RE.sub(" ", text)
        text = _SPACE_RE.sub(" ", unescape(text)).strip()
        if max_chars and len(text) > max_chars:
            return text[:max_chars].rstrip()
        return text
else:
    from atlas_brain.services.scraping.universal.html_cleaner import *
