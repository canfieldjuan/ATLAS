"""Shared company-name normalization utilities."""

import re

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|sa|srl|pty|nv|bv)\b\.?",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[,.\-;:]+$")


def normalize_company_name(name: str) -> str:
    """Lowercase, strip legal suffixes, collapse whitespace, strip trailing punctuation."""
    n = name.lower().strip()
    n = _LEGAL_SUFFIXES.sub("", n)
    n = _MULTI_SPACE.sub(" ", n).strip()
    n = _TRAILING_PUNCT.sub("", n).strip()
    return n
