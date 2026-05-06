"""Shared synthesis primitives for the reasoning core.

Private module. Provides the LLM-call-with-retry-on-validation-failure loop
and the parsing/validation/extraction helpers used by both
``run_reasoning`` and ``continue_reasoning``. Tracing and event-sink
emission are NOT performed here; public functions own that wrapping so
they can attach their own span names and event names.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SynthesisLoopConfig:
    """Pack-derived policies for the synthesis loop."""

    max_attempts: int = 2
    feedback_limit: int = 5
    max_tokens: int = 16384
    temperature: float = 0.3


@dataclass
class SynthesisLoopResult:
    """Outcome of one full synthesis loop, success or terminal failure."""

    valid_candidate: dict[str, Any] | None = None
    attempts: tuple[Mapping[str, Any], ...] = ()
    total_tokens: int = 0
    failure_reasons: tuple[str, ...] = ()
    last_text: str = ""
    last_candidate: dict[str, Any] | None = None
    error_text: str = ""

    @property
    def succeeded(self) -> bool:
        return self.valid_candidate is not None


def synthesis_config_from_pack(pack: Any) -> SynthesisLoopConfig:
    """Read max_attempts / feedback_limit / max_tokens / temperature from a pack."""

    policies = dict(getattr(pack, "policies", None) or {})
    return SynthesisLoopConfig(
        max_attempts=max(1, int(policies.get("max_attempts", 2))),
        feedback_limit=max(1, int(policies.get("feedback_limit", 5))),
        max_tokens=max(1, int(policies.get("max_tokens", 16384))),
        temperature=float(policies.get("temperature", 0.3)),
    )


async def invoke_synthesis_loop(
    *,
    system_prompt: str,
    payload_text: str,
    llm_metadata: Mapping[str, Any],
    config: SynthesisLoopConfig,
    llm_port: Any,
) -> SynthesisLoopResult:
    """Run system+user prompt through the LLM with validation-feedback retry.

    Pure: no tracing, no event emission. The caller wraps this with whatever
    span/event semantics they need. ``llm_metadata`` is forwarded into each
    LLM call so attempt-level metadata (entity_id, reasoning_mode, etc.) is
    attached at the provider boundary.
    """

    attempts: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    last_text = ""
    last_candidate: dict[str, Any] | None = None
    total_tokens = 0

    for attempt_index in range(config.max_attempts):
        attempt_no = attempt_index + 1
        messages: list[Mapping[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload_text},
        ]
        if attempt_index > 0 and last_text:
            messages.append({"role": "assistant", "content": last_text})
        if attempt_index > 0 and failure_reasons:
            feedback = "\n".join(
                f"- {reason}" for reason in failure_reasons[: config.feedback_limit]
            )
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was rejected. Return a complete "
                    "corrected JSON object only.\nFix these issues:\n"
                    f"{feedback}"
                ),
            })

        response = await llm_port.complete(
            messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            metadata={**dict(llm_metadata), "attempt_no": attempt_no},
        )
        text = _response_text(response)
        cleaned = _clean_reasoning_text(text)
        last_text = cleaned
        usage = dict(response.get("usage") or {}) if isinstance(response, Mapping) else {}
        attempt_tokens = _usage_tokens(usage)
        total_tokens += attempt_tokens

        parsed = _parse_llm_json(cleaned)
        validation = _validate_reasoning_candidate(parsed)
        attempts.append({
            "attempt_no": attempt_no,
            "valid": validation["valid"],
            "errors": tuple(validation["errors"]),
            "warnings": tuple(validation["warnings"]),
            "tokens_used": attempt_tokens,
        })

        if validation["valid"]:
            assert isinstance(parsed, dict)
            return SynthesisLoopResult(
                valid_candidate=parsed,
                attempts=tuple(attempts),
                total_tokens=total_tokens,
                failure_reasons=(),
                last_text=cleaned,
                last_candidate=parsed,
            )

        if isinstance(parsed, dict):
            last_candidate = parsed
        failure_reasons = list(validation["errors"])

    error_text = "; ".join(failure_reasons[:2])[:200] or "validation failed"
    return SynthesisLoopResult(
        valid_candidate=None,
        attempts=tuple(attempts),
        total_tokens=total_tokens,
        failure_reasons=tuple(failure_reasons[: config.feedback_limit]),
        last_text=last_text,
        last_candidate=last_candidate,
        error_text=error_text,
    )


def evidence_to_mapping(item: Any) -> Mapping[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, Mapping):
        return dict(item)
    return {"text": str(item)}


def _response_text(response: Mapping[str, Any]) -> str:
    for key in ("response", "content", "text"):
        value = response.get(key)
        if value is not None:
            return str(value)
    message = response.get("message")
    if isinstance(message, Mapping) and message.get("content") is not None:
        return str(message.get("content"))
    return json.dumps(response, default=str)


def _clean_reasoning_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    if "<scratchpad>" in cleaned:
        cleaned = cleaned.split("</scratchpad>")[-1].strip()
    return cleaned


def _parse_llm_json(text: str) -> Any:
    try:
        from extracted_llm_infrastructure.pipelines.llm import parse_json_response
    except ImportError:
        parse_json_response = None
    if parse_json_response is not None:
        return parse_json_response(text, recover_truncated=True)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_reasoning_candidate(candidate: Any) -> Mapping[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(candidate, dict):
        return {"valid": False, "errors": ("LLM did not return a JSON object",), "warnings": ()}
    if candidate.get("_parse_fallback"):
        return {"valid": False, "errors": ("LLM did not return valid JSON",), "warnings": ()}
    summary = extract_summary(candidate)
    claims = extract_claims(candidate)
    if not summary:
        errors.append("missing_summary")
    if not claims:
        errors.append("missing_claims")
    confidence = extract_confidence(candidate, claims)
    if confidence <= 0.0:
        warnings.append("zero_or_missing_confidence")
    return {"valid": not errors, "errors": tuple(errors), "warnings": tuple(warnings)}


def extract_summary(candidate: Mapping[str, Any]) -> str:
    for key in ("summary", "executive_summary", "causal_narrative"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = extract_summary(value)
            if nested:
                return nested
    contracts = candidate.get("reasoning_contracts")
    if isinstance(contracts, Mapping):
        for value in contracts.values():
            if isinstance(value, Mapping):
                nested = extract_summary(value)
                if nested:
                    return nested
    return ""


def extract_claims(candidate: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = candidate.get("claims")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        claims = tuple(
            dict(item) if isinstance(item, Mapping) else {"claim": str(item)}
            for item in raw
        )
        if claims:
            return claims
    contracts = candidate.get("reasoning_contracts")
    found: list[Mapping[str, Any]] = []
    if isinstance(contracts, Mapping):
        _collect_contract_claims(contracts, found)
    return tuple(found)


def _collect_contract_claims(value: Mapping[str, Any], found: list[Mapping[str, Any]]) -> None:
    for key, item in value.items():
        if not isinstance(item, Mapping):
            continue
        claim_text = item.get("claim") or item.get("summary") or item.get("narrative")
        if claim_text:
            found.append({"claim": str(claim_text), "section": str(key), **dict(item)})
        nested = item.get("claims") or item.get("sections")
        if isinstance(nested, Mapping):
            _collect_contract_claims(nested, found)


def extract_confidence(
    candidate: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
) -> float:
    rank = {"high": 0.9, "medium": 0.6, "low": 0.3, "insufficient": 0.0}
    raw = candidate.get("confidence")
    if raw is not None:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return rank.get(str(raw).strip().lower(), 0.0)
    values: list[float] = []
    for claim in claims:
        value = claim.get("confidence")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(rank.get(str(value or "").strip().lower(), 0.0))
    return max(values) if values else 0.0


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    return int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)


__all__ = [
    "SynthesisLoopConfig",
    "SynthesisLoopResult",
    "synthesis_config_from_pack",
    "invoke_synthesis_loop",
    "evidence_to_mapping",
    "extract_summary",
    "extract_claims",
    "extract_confidence",
]
