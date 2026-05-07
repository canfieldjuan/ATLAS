"""Stage-2 claim ledger schema for the Evidence-to-Story product.

Pure data + validators. No LLM, no I/O beyond JSON load/write. Stage 2's
extraction half (sources -> claims via an LLM call) is deliberately a
follow-up slice; this module exists so that extraction has a typed
contract to write into and downstream stages have a stable schema to
read from.

The schema follows section 5 of
``docs/evidence_to_story_v0_build_contract.md``. Per-type invariants
are encoded as :func:`validate_claim` and cross-claim invariants as
:func:`validate_ledger`. Both raise :class:`ClaimSchemaError` on
violations so callers get a single exception type to catch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


CLAIM_TYPES: tuple[str, ...] = (
    "factual",
    "timeline",
    "entity",
    "emotional_inference",
    "disputed",
    "reveal",
    "transition",
)
CONFIDENCE_LEVELS: tuple[str, ...] = ("verified", "inferred", "disputed", "unknown")
INSERTED_BY_VALUES: tuple[str, ...] = (
    "extraction_pass",
    "drafter",
    "revision",
    "reviewer",
)
MODE_CONSTRAINTS: tuple[str, ...] = ("strict", "narrative", "dramatized")
DEFAULT_MODE_CONSTRAINT = "narrative"
CLAIMS_FILENAME = "claims.json"

# Claim types that require source_id, quote, and a positional locator.
SOURCED_CLAIM_TYPES: tuple[str, ...] = (
    "factual",
    "timeline",
    "entity",
    "emotional_inference",
    "disputed",
    "reveal",
)

ClaimType = Literal[
    "factual",
    "timeline",
    "entity",
    "emotional_inference",
    "disputed",
    "reveal",
    "transition",
]
Confidence = Literal["verified", "inferred", "disputed", "unknown"]


class ClaimSchemaError(ValueError):
    """Raised when a claim ledger violates the v0 schema or an invariant."""


class InvalidClaimType(ClaimSchemaError):
    """Raised when a claim's ``claim_type`` is outside the v0 taxonomy."""


@dataclass(frozen=True)
class SourceLocator:
    """Structured pointer into a source's text.

    At least one of ``paragraph`` or ``timestamp`` must be set for any
    claim type that requires a locator (see
    :data:`SOURCED_CLAIM_TYPES`). ``quote_offset`` is an additional aid
    for ledger-to-text round-trip but does not satisfy the locator
    requirement on its own.
    """

    url: str = ""
    paragraph: int | None = None
    timestamp: str | None = None
    quote_offset: int | None = None

    def has_position(self) -> bool:
        return self.paragraph is not None or self.timestamp is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "paragraph": self.paragraph,
            "timestamp": self.timestamp,
            "quote_offset": self.quote_offset,
        }


@dataclass(frozen=True)
class ClaimRecord:
    """One row in the claim ledger.

    Field semantics follow section 5 of the v0 build contract. Validation is
    not performed in ``__init__`` to keep construction cheap; call
    :func:`validate_claim` to enforce per-type invariants.
    """

    claim_id: str
    story_id: str
    text: str
    claim_type: ClaimType
    source_id: str = ""
    quote: str = ""
    source_locator: SourceLocator = field(default_factory=SourceLocator)
    confidence: Confidence = "unknown"
    mode_constraint: str = DEFAULT_MODE_CONSTRAINT
    rewrite_applied: bool = False
    original_text: str = ""
    inserted_by: str = "extraction_pass"
    verified_by: str = ""
    verified_at: str = ""
    dispute_group_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "story_id": self.story_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "source_id": self.source_id,
            "quote": self.quote,
            "source_locator": self.source_locator.as_dict(),
            "confidence": self.confidence,
            "mode_constraint": self.mode_constraint,
            "rewrite_applied": self.rewrite_applied,
            "original_text": self.original_text,
            "inserted_by": self.inserted_by,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "dispute_group_id": self.dispute_group_id,
        }


@dataclass(frozen=True)
class ClaimLedger:
    """Full claim ledger for one story package."""

    story_id: str
    claims: tuple[ClaimRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "claims": [claim.as_dict() for claim in self.claims],
        }


def validate_claim(claim: ClaimRecord) -> None:
    """Apply the per-type fact-check invariants from section 5 of the contract.

    Raises :class:`ClaimSchemaError` (or :class:`InvalidClaimType`) on
    the first violation. The exception message identifies the offending
    ``claim_id`` so callers can surface it directly.
    """

    if claim.claim_type not in CLAIM_TYPES:
        raise InvalidClaimType(
            f"claim {claim.claim_id!r} has invalid claim_type "
            f"{claim.claim_type!r}; expected one of {', '.join(CLAIM_TYPES)}"
        )

    if not claim.claim_id:
        raise ClaimSchemaError("claim_id must be a non-empty string")
    if not claim.story_id:
        raise ClaimSchemaError(f"claim {claim.claim_id!r} story_id must be a non-empty string")
    if not claim.text:
        raise ClaimSchemaError(f"claim {claim.claim_id!r} text must be a non-empty string")

    if claim.claim_type == "transition":
        _require_empty(claim, "source_id")
        _require_empty(claim, "quote")
        if claim.source_locator.has_position():
            raise ClaimSchemaError(
                f"claim {claim.claim_id!r} is a transition but carries a source locator"
            )
        return

    # SOURCED_CLAIM_TYPES from here on.
    _require_non_empty(claim, "source_id")
    _require_non_empty(claim, "quote")
    if not claim.source_locator.has_position():
        raise ClaimSchemaError(
            f"claim {claim.claim_id!r} ({claim.claim_type}) requires a source_locator "
            "with at least one of paragraph or timestamp"
        )

    if claim.claim_type == "emotional_inference":
        if not claim.rewrite_applied:
            raise ClaimSchemaError(
                f"claim {claim.claim_id!r} is emotional_inference but rewrite_applied is False; "
                "soft-rewrite must run before the script lands"
            )
        if not claim.original_text:
            raise ClaimSchemaError(
                f"claim {claim.claim_id!r} is emotional_inference with rewrite_applied "
                "but original_text is empty; the audit trail requires the pre-rewrite assertion"
            )

    if claim.claim_type == "disputed" and not claim.dispute_group_id:
        raise ClaimSchemaError(
            f"claim {claim.claim_id!r} is disputed but has no dispute_group_id; "
            "disputed claims must declare which disagreement set they belong to"
        )


def validate_ledger(ledger: ClaimLedger) -> None:
    """Apply cross-claim invariants on top of per-claim validation.

    Raises :class:`ClaimSchemaError` on uniqueness violations or
    insufficient disputed-set coverage.
    """

    if not ledger.story_id:
        raise ClaimSchemaError("ledger.story_id must be a non-empty string")

    seen_ids: set[str] = set()
    for claim in ledger.claims:
        validate_claim(claim)
        if claim.story_id != ledger.story_id:
            raise ClaimSchemaError(
                f"claim {claim.claim_id!r} story_id {claim.story_id!r} "
                f"does not match ledger story_id {ledger.story_id!r}"
            )
        if claim.claim_id in seen_ids:
            raise ClaimSchemaError(f"duplicate claim_id {claim.claim_id!r} in ledger")
        seen_ids.add(claim.claim_id)

    # CITE-004: each disputed group needs >=2 distinct source_ids across
    # its members so both sides of the disagreement land in the ledger.
    dispute_groups: dict[str, set[str]] = {}
    for claim in ledger.claims:
        if claim.claim_type != "disputed":
            continue
        dispute_groups.setdefault(claim.dispute_group_id, set()).add(claim.source_id)
    for group_id, source_ids in dispute_groups.items():
        if len(source_ids) < 2:
            raise ClaimSchemaError(
                f"disputed group {group_id!r} has only {len(source_ids)} distinct source_id(s); "
                "every disputed group must surface at least 2 distinct sources"
            )


def apply_soft_rewrite(claim: ClaimRecord, *, rewritten_text: str) -> ClaimRecord:
    """Stamp a soft-rewrite onto an emotional_inference claim.

    Returns a new frozen :class:`ClaimRecord` with ``original_text``
    pinned to the input claim's ``text``, ``text`` replaced with
    ``rewritten_text``, and ``rewrite_applied`` set to ``True``.

    Rejects non-``emotional_inference`` claims to prevent the rewrite
    flag from being smuggled onto factual claims (which would defeat
    the audit trail).
    """

    if claim.claim_type != "emotional_inference":
        raise ClaimSchemaError(
            f"apply_soft_rewrite is only valid for emotional_inference claims; "
            f"claim {claim.claim_id!r} is {claim.claim_type!r}"
        )
    rewritten = rewritten_text.strip()
    if not rewritten:
        raise ClaimSchemaError(
            f"apply_soft_rewrite requires non-empty rewritten_text for claim {claim.claim_id!r}"
        )

    return ClaimRecord(
        claim_id=claim.claim_id,
        story_id=claim.story_id,
        text=rewritten,
        claim_type=claim.claim_type,
        source_id=claim.source_id,
        quote=claim.quote,
        source_locator=claim.source_locator,
        confidence=claim.confidence,
        mode_constraint=claim.mode_constraint,
        rewrite_applied=True,
        original_text=claim.text,
        inserted_by=claim.inserted_by,
        verified_by=claim.verified_by,
        verified_at=claim.verified_at,
        dispute_group_id=claim.dispute_group_id,
    )


def load_claim_ledger(path: str | Path) -> ClaimLedger:
    """Load ``claims.json`` into a :class:`ClaimLedger` and validate.

    Raises :class:`ClaimSchemaError` if the file cannot be parsed or
    fails any per-claim or cross-claim invariant.
    """

    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClaimSchemaError(f"claim ledger not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ClaimSchemaError(f"claim ledger is not valid JSON: {target}") from exc
    if not isinstance(raw, Mapping):
        raise ClaimSchemaError("claim ledger must be a JSON object")

    story_id = raw.get("story_id")
    if not isinstance(story_id, str) or not story_id:
        raise ClaimSchemaError("claim ledger story_id must be a non-empty string")
    raw_claims = raw.get("claims")
    if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes)):
        raise ClaimSchemaError("claim ledger claims must be an array")

    claims = tuple(_claim_from_mapping(item, index=i) for i, item in enumerate(raw_claims, 1))
    ledger = ClaimLedger(story_id=story_id, claims=claims)
    validate_ledger(ledger)
    return ledger


def write_claim_ledger(ledger: ClaimLedger, output_dir: str | Path) -> Path:
    """Validate then write the ledger to ``output_dir/claims.json``."""

    validate_ledger(ledger)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / CLAIMS_FILENAME
    output_path.write_text(
        f"{json.dumps(ledger.as_dict(), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return output_path


def _claim_from_mapping(raw: Any, *, index: int) -> ClaimRecord:
    if not isinstance(raw, Mapping):
        raise ClaimSchemaError(f"claim entry {index} must be an object")
    locator_raw = raw.get("source_locator") or {}
    if not isinstance(locator_raw, Mapping):
        raise ClaimSchemaError(f"claim entry {index} source_locator must be an object")
    locator = SourceLocator(
        url=str(locator_raw.get("url") or ""),
        paragraph=_optional_int(locator_raw.get("paragraph"), context=f"claim {index} paragraph"),
        timestamp=_optional_str_or_none(locator_raw.get("timestamp")),
        quote_offset=_optional_int(
            locator_raw.get("quote_offset"), context=f"claim {index} quote_offset"
        ),
    )
    return ClaimRecord(
        claim_id=str(raw.get("claim_id") or ""),
        story_id=str(raw.get("story_id") or ""),
        text=str(raw.get("text") or ""),
        claim_type=str(raw.get("claim_type") or ""),  # type: ignore[arg-type]
        source_id=str(raw.get("source_id") or ""),
        quote=str(raw.get("quote") or ""),
        source_locator=locator,
        confidence=str(raw.get("confidence") or "unknown"),  # type: ignore[arg-type]
        mode_constraint=str(raw.get("mode_constraint") or DEFAULT_MODE_CONSTRAINT),
        rewrite_applied=bool(raw.get("rewrite_applied") or False),
        original_text=str(raw.get("original_text") or ""),
        inserted_by=str(raw.get("inserted_by") or "extraction_pass"),
        verified_by=str(raw.get("verified_by") or ""),
        verified_at=str(raw.get("verified_at") or ""),
        dispute_group_id=str(raw.get("dispute_group_id") or ""),
    )


def _optional_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClaimSchemaError(f"{context} must be an integer or null")
    return value


def _optional_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_non_empty(claim: ClaimRecord, field_name: str) -> None:
    if not getattr(claim, field_name):
        raise ClaimSchemaError(
            f"claim {claim.claim_id!r} ({claim.claim_type}) requires "
            f"non-empty {field_name}"
        )


def _require_empty(claim: ClaimRecord, field_name: str) -> None:
    if getattr(claim, field_name):
        raise ClaimSchemaError(
            f"claim {claim.claim_id!r} is a transition but {field_name} is non-empty"
        )


__all__ = [
    "CLAIMS_FILENAME",
    "CLAIM_TYPES",
    "CONFIDENCE_LEVELS",
    "ClaimLedger",
    "ClaimRecord",
    "ClaimSchemaError",
    "Confidence",
    "ClaimType",
    "DEFAULT_MODE_CONSTRAINT",
    "INSERTED_BY_VALUES",
    "InvalidClaimType",
    "MODE_CONSTRAINTS",
    "SOURCED_CLAIM_TYPES",
    "SourceLocator",
    "apply_soft_rewrite",
    "load_claim_ledger",
    "validate_claim",
    "validate_ledger",
    "write_claim_ledger",
]
