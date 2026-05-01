from __future__ import annotations

import re


def normalize_company_name(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
