"""Pure semantic-cache primitives (PR-C2 / PR 4).

Reasoning core owns the *pure* parts of semantic-cache identity and
confidence math. Postgres-backed storage stays in
``extracted_llm_infrastructure.reasoning.semantic_cache`` (the
``SemanticCache`` class) behind the ``SemanticCacheStore`` port
declared in ``extracted_reasoning_core.ports``.

Public surface:

  - ``STALE_THRESHOLD`` -- entries with effective confidence below this
    are treated as stale (cache miss) by the storage adapter.
  - ``CacheEntry`` -- the cached reasoning conclusion; pure data. The
    storage adapter is responsible for filling ``created_at`` /
    ``last_validated_at`` / ``effective_confidence``.
  - ``compute_evidence_hash(evidence)`` -- SHA-256 of deterministically
    serialised evidence dict; 16-char fingerprint that callers use to
    invalidate cached conclusions when their underlying evidence
    changes.
  - ``apply_decay(confidence, last_validated, half_life_days)`` --
    exponential decay of confidence since last validation. Reads the
    system clock; callers wanting deterministic time should pass a
    pre-computed decayed value or use a clock-injected wrapper.
  - ``row_to_cache_entry(row)`` -- coerce a Postgres row mapping
    (``asyncpg.Record`` or plain ``dict`` -- anything supporting
    ``Mapping`` access) into a ``CacheEntry``. Handles JSONB columns
    that arrive either pre-decoded or as raw strings, and the
    ``falsification_conditions`` shape drift between dict and list.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


# Entries with effective confidence below this are treated as stale.
# Storage adapters should return ``None`` from ``lookup`` when the
# decayed confidence falls below this threshold.
STALE_THRESHOLD = 0.5


@dataclass
class CacheEntry:
    """A single cached reasoning conclusion.

    ``effective_confidence`` is set by storage adapters after they
    apply ``apply_decay`` to ``confidence`` based on
    ``last_validated_at``; callers reading from cache should consult
    ``effective_confidence`` rather than ``confidence`` directly.
    """

    pattern_sig: str
    pattern_class: str
    conclusion: dict[str, Any]
    confidence: float
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    boundary_conditions: dict[str, Any] = field(default_factory=dict)
    falsification_conditions: list[str] = field(default_factory=list)
    uncertainty_sources: list[str] = field(default_factory=list)
    vendor_name: str | None = None
    product_category: str | None = None
    decay_half_life_days: int = 90
    conclusion_type: str | None = None
    evidence_hash: str | None = None
    created_at: datetime | None = None
    last_validated_at: datetime | None = None
    validation_count: int = 1
    effective_confidence: float | None = None


def compute_evidence_hash(evidence: dict[str, Any]) -> str:
    """SHA-256 of deterministically serialised evidence dict.

    Returns a 16-char hex fingerprint. ``sort_keys=True`` ensures dict
    iteration order doesn't perturb the hash; ``default=str`` keeps
    non-JSON-native values (datetimes, decimals) hashable.
    """
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def apply_decay(
    confidence: float,
    last_validated: datetime,
    half_life_days: int,
) -> float:
    """Return effective confidence after exponential decay.

    Reads the system clock (``datetime.now(timezone.utc)``); callers
    that need deterministic time should compute the decay externally
    rather than passing a fake clock here.

    A naive ``last_validated`` (no ``tzinfo``) is assumed to be UTC.
    Negative or zero elapsed days return ``confidence`` unchanged
    (no decay applied); a non-positive ``half_life_days`` short-circuits
    the same way.
    """
    now = datetime.now(timezone.utc)
    if last_validated.tzinfo is None:
        last_validated = last_validated.replace(tzinfo=timezone.utc)
    days = (now - last_validated).total_seconds() / 86400.0
    if days <= 0 or half_life_days <= 0:
        return confidence
    return confidence * math.pow(2, -(days / half_life_days))


def row_to_cache_entry(row: Mapping[str, Any]) -> CacheEntry:
    """Coerce a Postgres row mapping into a :class:`CacheEntry`.

    Accepts any object supporting ``Mapping`` access -- typically an
    ``asyncpg.Record`` (which subscripts like a dict) but also a plain
    dict, which is what tests + non-asyncpg adapters produce. JSONB
    columns may arrive either pre-decoded (asyncpg with the json codec
    installed) or as raw strings (a vanilla adapter); the body handles
    both shapes defensively.

    Storage adapters should call this on rows from the
    ``reasoning_semantic_cache`` table; the result is the same
    ``CacheEntry`` shape regardless of whether the JSONB columns were
    pre-decoded or not.
    """
    falsification = row["falsification_conditions"]
    if isinstance(falsification, str):
        falsification = json.loads(falsification)
    if isinstance(falsification, dict):
        falsification = list(falsification.values()) if falsification else []

    conclusion = row["conclusion"]
    if not isinstance(conclusion, dict):
        conclusion = json.loads(conclusion)

    reasoning_steps = row["reasoning_steps"]
    if not isinstance(reasoning_steps, list):
        reasoning_steps = json.loads(reasoning_steps)

    boundary_conditions = row["boundary_conditions"]
    if not isinstance(boundary_conditions, dict):
        boundary_conditions = json.loads(boundary_conditions)

    return CacheEntry(
        pattern_sig=row["pattern_sig"],
        pattern_class=row["pattern_class"],
        vendor_name=row["vendor_name"],
        product_category=row["product_category"],
        conclusion=conclusion,
        confidence=row["confidence"],
        reasoning_steps=reasoning_steps,
        boundary_conditions=boundary_conditions,
        falsification_conditions=falsification if isinstance(falsification, list) else [],
        uncertainty_sources=list(row["uncertainty_sources"]) if row["uncertainty_sources"] else [],
        decay_half_life_days=row["decay_half_life_days"],
        conclusion_type=row["conclusion_type"],
        evidence_hash=row["evidence_hash"],
        created_at=row["created_at"],
        last_validated_at=row["last_validated_at"],
        validation_count=row["validation_count"],
    )


__all__ = [
    "CacheEntry",
    "STALE_THRESHOLD",
    "apply_decay",
    "compute_evidence_hash",
    "row_to_cache_entry",
]
