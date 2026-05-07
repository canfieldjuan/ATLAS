"""Shared JSONB / asyncpg helpers for the AI Content Ops postgres adapters.

Extracted from the per-asset adapters (``campaign_postgres``,
``report_postgres``, ``landing_page_postgres``, ``sales_brief_postgres``)
where the same handful of helpers had been copy-pasted across four
modules. Surfaced as a coordination follow-up from the PR-#354 review:
duplication was acknowledged across three adapters, a fourth was
imminent, and the helpers had been byte-identical for long enough that
a single landing point was the right next move.

All helpers are framework-neutral: no asyncpg import, no DB connection
state. They handle the boundary between ``Mapping`` host inputs and
JSONB column round-tripping, plus asyncpg's command-tag parsing.

Public surface:

  - ``json_dump_jsonb(value)`` -- serialize an arbitrary value as a
    JSON string suitable for a ``$N::jsonb`` parameter. Matches the
    historical ``_jsonb()`` private helper.
  - ``decode_jsonb_field(raw, *, default)`` -- defensively decode a
    JSONB column. Handles both pre-decoded values (asyncpg with the
    json codec installed delivers Python objects directly) and the
    string form (driver without the codec, or test fakes).
  - ``row_to_dict(row)`` -- coerce an asyncpg.Record-like row to a
    plain ``dict``. Test fakes that hand back plain dicts pass
    through unchanged.
  - ``parse_command_tag(result)`` -- parse asyncpg's ``"UPDATE 1"`` /
    ``"UPDATE 0"`` command tag and return ``True`` on hit, ``False``
    on miss. Test fakes / alternative drivers that return ``None`` or
    ``"OK"`` default to ``True`` so the parse never crashes.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def json_dump_jsonb(value: Any) -> str:
    """Serialize ``value`` as a JSON string for a ``$N::jsonb`` param.

    ``None`` is mapped to an empty object so callers that pass
    ``meta or None`` don't have to special-case the SQL parameter.
    """
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def row_to_dict(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Coerce an asyncpg.Record-like row to a plain ``dict``."""
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


def decode_jsonb_field(raw: Any, *, default: Any) -> Any:
    """Defensively decode a JSONB column value.

    asyncpg with the json codec installed delivers JSONB pre-decoded
    (dict / list); without the codec the value comes through as a JSON
    string. ``None`` rows fall back to ``default``. Malformed JSON
    strings also fall back to ``default`` rather than raising -- the
    list/save path enforces structural validation; this helper is only
    a hardening boundary.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default
    if raw is None:
        return default
    return raw


def parse_command_tag(result: Any) -> bool:
    """Parse an asyncpg command tag (``"UPDATE 1"`` / ``"UPDATE 0"``).

    Returns ``True`` on a hit (rows affected > 0) and ``False`` on a
    miss. Non-string results (test fakes, alternative drivers that
    return ``None`` or ``"OK"``) default to ``True`` so the parse
    never crashes -- callers that need strict miss-detection should
    use a real asyncpg driver.
    """
    if not isinstance(result, str):
        return True
    try:
        return int(result.rsplit(" ", 1)[-1]) > 0
    except (ValueError, IndexError):
        return True


__all__ = [
    "decode_jsonb_field",
    "json_dump_jsonb",
    "parse_command_tag",
    "row_to_dict",
]
