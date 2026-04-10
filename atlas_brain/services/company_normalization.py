"""Shared company-name normalization utilities."""

import re

_LEGAL_SUFFIX_PATTERN = (
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|sa|srl|pty|nv|bv)\b\.?"
)
_LEGAL_SUFFIXES = re.compile(
    _LEGAL_SUFFIX_PATTERN,
    re.IGNORECASE,
)
_MULTI_SPACE_PATTERN = r"\s+"
_MULTI_SPACE = re.compile(_MULTI_SPACE_PATTERN)
_TRAILING_PUNCT_PATTERN = r"[,.\-;:]+$"
_TRAILING_PUNCT = re.compile(_TRAILING_PUNCT_PATTERN)


def normalize_company_name(name: str) -> str:
    """Lowercase, strip legal suffixes, collapse whitespace, strip trailing punctuation."""
    n = name.lower().strip()
    n = _LEGAL_SUFFIXES.sub("", n)
    n = _MULTI_SPACE.sub(" ", n).strip()
    n = _TRAILING_PUNCT.sub("", n).strip()
    return n


def normalized_company_name_sql(column_sql: str) -> str:
    """Return a SQL expression that mirrors normalize_company_name() for a column."""
    # Use $re$...$re$ dollar-quoting (not $$...$$) because _TRAILING_PUNCT_PATTERN
    # ends with a literal $ anchor, which collides with $$ delimiters.
    return (
        "TRIM("
        "REGEXP_REPLACE("
        "REGEXP_REPLACE("
        f"REGEXP_REPLACE(LOWER(COALESCE({column_sql}, '')), $re${_LEGAL_SUFFIX_PATTERN}$re$, '', 'gi'), "
        f"$re${_MULTI_SPACE_PATTERN}$re$, ' ', 'g'), "
        f"$re${_TRAILING_PUNCT_PATTERN}$re$, '', 'g'"
        ")"
        ")"
    )
