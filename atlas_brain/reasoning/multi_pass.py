"""Multi-pass reasoning engine (classify -> verify -> challenge -> ground).

Pass 1 (CLASSIFY): Standard LLM classification from evidence.
Pass 1.5 (VERIFY): Deterministic sufficiency check that caps overconfident thin-evidence outputs.
Pass 2 (CHALLENGE): Deterministic contradiction detection + LLM self-critique.
Pass 3 (GROUND): Forces every key_signal to cite exact evidence field:value.

Skipping rules:
- Pass 2 skipped if no contradictions are found
- Pass 2 skipped when low-confidence conclusions are also low-impact and thinly evidenced
- Pass 3 can run even when challenge is skipped when lightweight grounding is enabled
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("atlas.reasoning.multi_pass")

# Maps dominant pain category to expected archetype
_PAIN_TO_ARCHETYPE: dict[str, str] = {
    "ux": "leadership_redesign",
    "usability": "leadership_redesign",
    "pricing": "pricing_shock",
    "price": "pricing_shock",
    "support": "support_collapse",
    "integration": "integration_break",
    "api": "integration_break",
    "compliance": "compliance_gap",
    "regulatory": "compliance_gap",
    "feature": "feature_gap",
    "functionality": "feature_gap",
}


@dataclass
class PassResult:
    pass_number: int        # 1, 2, or 3
    pass_type: str          # "classify", "challenge", "ground"
    conclusion: dict        # normalized conclusion
    tokens_used: int
    duration_ms: float
    changed: bool           # did this pass change the archetype?
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiPassResult:
    final_conclusion: dict
    passes: list[PassResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    passes_executed: int = 0
    boundary_conditions: dict[str, Any] = field(default_factory=dict)


_CHALLENGE_PROMPT = """\
You classified this vendor's churn pattern. Now CHALLENGE your own conclusion.

YOUR CONCLUSION:
{conclusion_json}

CONTRADICTING EVIDENCE:
{contradictions}

Rules:
- If the contradictions are valid, REVISE the archetype and adjust confidence.
- If you can defend with specific metrics from the evidence, KEEP the archetype but explain why.
- You MUST output the full conclusion JSON (same schema), revised or defended.
- Do NOT default to the same confidence -- re-evaluate based on the challenge.

Output ONLY valid JSON with the same fields as the original conclusion.\
"""

_GROUND_PROMPT = """\
You are grounding a B2B churn classification. Every key_signal MUST cite an exact field:value from the evidence.

CONCLUSION TO GROUND:
{conclusion_json}

AVAILABLE EVIDENCE FIELDS:
{evidence_fields}

Rules:
- Each key_signal must be in format "field_name: value" where field_name exists in the evidence.
- Remove any key_signal that cannot be grounded in the evidence.
- If all signals are removed, set confidence to 0.3 and archetype to "mixed".
- executive_summary sentence 2 must reference at least 2 specific numbers from evidence.
- Do NOT invent data not in the evidence fields above.

Output ONLY valid JSON with the same fields.\
"""


def _find_contradicting_evidence(
    conclusion: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    """Deterministic contradiction detection. Returns up to 3 contradiction strings."""
    contradictions: list[str] = []
    chosen_archetype = conclusion.get("archetype", "")

    # 1. Top pain category maps to a different archetype than chosen
    pain_categories = evidence.get("pain_categories")
    if pain_categories and isinstance(pain_categories, list):
        top_pain = pain_categories[0] if pain_categories else {}
        if isinstance(top_pain, dict):
            top_cat = top_pain.get("category", "").lower()
            expected = _PAIN_TO_ARCHETYPE.get(top_cat)
            if expected and expected != chosen_archetype and chosen_archetype not in ("mixed", "stable"):
                count = top_pain.get("count", "?")
                contradictions.append(
                    f"Top pain category is '{top_cat}' (count={count}), "
                    f"which maps to {expected}, not {chosen_archetype}"
                )

    # 2. Archetype pre-scores show a strong alternative
    arch_scores = evidence.get("archetype_scores")
    if arch_scores and isinstance(arch_scores, list):
        for score_entry in arch_scores:
            if not isinstance(score_entry, dict):
                continue
            arch_name = score_entry.get("archetype", "")
            score_val = score_entry.get("score", 0)
            if (
                arch_name != chosen_archetype
                and isinstance(score_val, (int, float))
                and score_val > 0.4
                and chosen_archetype not in ("mixed", "stable")
            ):
                contradictions.append(
                    f"Archetype pre-score for '{arch_name}' is {score_val:.2f} (> 0.4), "
                    f"suggesting it may be a stronger fit than {chosen_archetype}"
                )
                break  # only report strongest alternative

    # 3. Low review count with high confidence
    total_reviews = evidence.get("total_reviews", 0)
    confidence = conclusion.get("confidence", 0)
    if (
        isinstance(total_reviews, (int, float))
        and total_reviews < 20
        and isinstance(confidence, (int, float))
        and confidence > 0.7
    ):
        contradictions.append(
            f"Only {total_reviews} reviews but confidence is {confidence:.2f} -- "
            f"insufficient sample size for high confidence"
        )

    # 4. High displacement count with non-displacement archetype
    displacement = evidence.get("displacement_mention_count", 0)
    if (
        isinstance(displacement, (int, float))
        and displacement > 5
        and chosen_archetype not in ("category_disruption", "feature_gap", "mixed", "stable")
    ):
        contradictions.append(
            f"displacement_mention_count={displacement} is high, "
            f"suggesting category_disruption or feature_gap, not {chosen_archetype}"
        )

    return contradictions[:3]


def _safe_float(value: Any) -> float | None:
    """Return float(value) when possible."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_grounded_signal(signal: Any, evidence_keys: set[str]) -> bool:
    """Return True when a signal references a known evidence field."""
    if not isinstance(signal, str) or ":" not in signal:
        return False
    metric, value = signal.split(":", 1)
    metric = metric.strip().lower().replace(" ", "_")
    if not metric or not value.strip():
        return False
    if metric in evidence_keys:
        return True
    return any(key.startswith(metric) or metric.startswith(key) for key in evidence_keys)


def _verify_evidence_sufficiency(
    conclusion: dict[str, Any],
    evidence: dict[str, Any],
    *,
    min_reviews: int,
    min_snapshot_days: int,
    min_grounded_signals: int,
    confidence_cap: float,
) -> tuple[dict[str, Any], list[str], bool]:
    """Deterministically cap overconfident thin-evidence classify outputs."""
    updated = dict(conclusion)
    issues: list[str] = []
    changed = False
    evidence_keys = set(evidence.keys())

    raw_signals = list(updated.get("key_signals") or [])
    grounded_signals = [s for s in raw_signals if _is_grounded_signal(s, evidence_keys)]
    if grounded_signals != raw_signals:
        updated["key_signals"] = grounded_signals
        changed = True
    if len(grounded_signals) < min_grounded_signals:
        issues.append(
            f"only {len(grounded_signals)} grounded key_signals (min {min_grounded_signals})"
        )

    total_reviews = _safe_float(evidence.get("total_reviews"))
    if total_reviews is not None and total_reviews < min_reviews:
        issues.append(f"review volume is thin ({int(total_reviews)} < {min_reviews})")

    snapshot_days = _safe_float(evidence.get("snapshot_days"))
    if snapshot_days is not None and snapshot_days < min_snapshot_days:
        issues.append(f"temporal depth is thin ({int(snapshot_days)}d < {min_snapshot_days}d)")

    if issues:
        confidence = _safe_float(updated.get("confidence")) or 0.0
        if confidence > confidence_cap:
            updated["confidence"] = confidence_cap
            changed = True
        uncertainty_sources = [str(item) for item in updated.get("uncertainty_sources", []) if item]
        for issue in issues:
            if issue not in uncertainty_sources:
                uncertainty_sources.append(issue)
                changed = True
        updated["uncertainty_sources"] = uncertainty_sources[:4]

    return updated, issues, changed


def _should_attempt_challenge(
    *,
    conclusion: dict[str, Any],
    evidence: dict[str, Any],
    contradictions: list[str],
    confidence_floor: float,
    min_reviews: int,
    mixed_polarity_min_share: float,
    high_impact_churn_density: float,
    high_impact_avg_urgency: float,
    high_impact_displacement_mentions: int,
) -> tuple[bool, str]:
    """Evidence-aware gate for the challenge pass."""
    if not contradictions:
        return False, "no contradictions found"

    total_reviews = _safe_float(evidence.get("total_reviews"))
    churn_density = _safe_float(evidence.get("churn_density")) or 0.0
    avg_urgency = _safe_float(evidence.get("avg_urgency")) or 0.0
    displacement = _safe_float(evidence.get("displacement_mention_count")) or 0.0
    recommend_yes = _safe_float(evidence.get("recommend_yes")) or 0.0
    recommend_no = _safe_float(evidence.get("recommend_no")) or 0.0
    recommend_total = recommend_yes + recommend_no
    minority_share = (min(recommend_yes, recommend_no) / recommend_total) if recommend_total > 0 else 0.0
    enough_evidence = total_reviews is None or total_reviews >= min_reviews
    mixed_polarity = recommend_total > 0 and minority_share >= mixed_polarity_min_share
    high_impact = (
        churn_density >= high_impact_churn_density
        or avg_urgency >= high_impact_avg_urgency
        or displacement >= high_impact_displacement_mentions
    )
    confidence = _safe_float(conclusion.get("confidence")) or 0.0
    if confidence > confidence_floor:
        return True, "confidence above floor"
    if enough_evidence and mixed_polarity:
        return True, "low-confidence but mixed polarity"
    if enough_evidence and high_impact:
        return True, "low-confidence but high impact"
    return False, "low-confidence and not high-impact enough to challenge"


async def multi_pass_reason(
    *,
    llm: Any,
    system_prompt: str,
    evidence_payload: dict[str, Any],
    json_schema: dict[str, Any],
    max_tokens: int,
    temperature: float,
    enabled: bool = True,
    verify_enabled: bool = True,
    verify_min_reviews: int = 12,
    verify_min_snapshot_days: int = 14,
    verify_min_grounded_signals: int = 2,
    verify_confidence_cap: float = 0.58,
    light_pass_max_tokens: int = 4096,
    challenge_confidence_floor: float = 0.3,
    challenge_min_reviews: int = 8,
    challenge_mixed_polarity_min_share: float = 0.2,
    challenge_high_impact_churn_density: float = 20.0,
    challenge_high_impact_avg_urgency: float = 6.0,
    challenge_high_impact_displacement_mentions: int = 5,
    ground_change_threshold: float = 0.05,
    ground_always: bool = False,
    ground_only: bool = False,
    span_prefix: str = "reasoning.stratified",
    trace_metadata: dict[str, Any] | None = None,
    normalize_fn: Callable[[dict], dict] | None = None,
    contradiction_finder: Callable[[dict, dict], list[str]] | None = None,
    llm_light: Any = None,
) -> MultiPassResult:
    """Run multi-pass reasoning: classify -> challenge -> ground.

    If ``enabled`` is False, runs a single classify pass (backwards compatible).
    If ``ground_only`` is True, skips challenge and runs classify -> ground
    (used by reconstitute to add evidence grounding without self-critique).

    ``llm`` is the Tier 1 (heavy) model used for the classify pass.
    ``llm_light``, when provided, is the Tier 2 (light) model used for
    challenge and ground passes.  Falls back to ``llm`` when None.
    """
    from ..pipelines.llm import parse_json_response, trace_llm_call
    from ..services.protocols import Message

    trace_metadata = dict(trace_metadata or {})
    result = MultiPassResult(final_conclusion={})
    finder = contradiction_finder or _find_contradicting_evidence
    _llm_light = llm_light or llm  # fallback to heavy when no light model
    result.boundary_conditions = {
        "verify_enabled": bool(verify_enabled),
        "ground_always": bool(ground_always),
        "ground_only": bool(ground_only),
    }

    def _normalize(c: dict) -> dict:
        return normalize_fn(c) if normalize_fn else c

    # Deferred trace: LLM calls store trace data here; _flush_trace()
    # emits them once pass_changed is known.
    _pending_trace: dict[str, Any] | None = None  # noqa: F841

    async def _call_llm(
        messages: list[Message],
        pass_num: int,
        pass_type: str,
        *,
        mt: int | None = None,
        pass_changed: bool | None = None,
        use_llm: Any = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict, int, float]:
        """Single LLM call with tracing. Returns (conclusion, tokens, duration_ms).

        ``use_llm`` overrides which LLM instance to call.  Defaults to the
        heavy model (``llm``).

        ``pass_changed`` is written into trace metadata.  For Pass 1 it is
        always None (not applicable).  For Pass 2/3, the caller should call
        ``_emit_trace_changed()`` after comparing conclusions to retroactively
        record the value -- but we also accept it here for the common case
        where we already know.
        """
        _active_llm = use_llm or llm
        pass_meta = {
            **trace_metadata,
            "pass_number": pass_num,
            "pass_type": pass_type,
            "multi_pass": True,
            "model_tier": "light" if _active_llm is _llm_light and _llm_light is not llm else "heavy",
        }
        if pass_changed is not None:
            pass_meta["pass_changed"] = pass_changed
        if extra_metadata:
            pass_meta.update(extra_metadata)
        span = f"{span_prefix}.reason" if pass_num == 1 else f"{span_prefix}.reason.{pass_type}"

        t0 = time.monotonic()
        try:
            llm_result = await asyncio.to_thread(
                _active_llm.chat,
                messages=messages,
                max_tokens=mt or max_tokens,
                temperature=temperature,
                guided_json=json_schema,
                response_format={"type": "json_object"},
            )
            text = llm_result.get("response", "").strip()
            usage = llm_result.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            trace_meta_resp = llm_result.get("_trace_meta", {})
            duration_ms = (time.monotonic() - t0) * 1000

            if not text:
                trace_llm_call(
                    span_name=span,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    model=getattr(_active_llm, "model", getattr(_active_llm, "model_id", "")),
                    provider=getattr(_active_llm, "name", ""),
                    duration_ms=duration_ms,
                    status="failed",
                    metadata={**pass_meta, "pass_changed": False},
                    error_message="llm returned empty structured response",
                    error_type="EmptyStructuredResponse",
                    input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
                    api_endpoint=trace_meta_resp.get("api_endpoint"),
                    provider_request_id=trace_meta_resp.get("provider_request_id"),
                    ttft_ms=trace_meta_resp.get("ttft_ms"),
                    inference_time_ms=trace_meta_resp.get("inference_time_ms"),
                    queue_time_ms=trace_meta_resp.get("queue_time_ms"),
                )
                raise ValueError("llm returned empty structured response")

            # Store trace data for deferred emission (pass_changed not yet known)
            nonlocal _pending_trace
            _pending_trace = {
                "span": span,
                "usage": usage,
                "trace_meta_resp": trace_meta_resp,
                "duration_ms": duration_ms,
                "pass_meta": pass_meta,
                "messages": messages,
                "text": text,
                "llm_ref": _active_llm,
            }

            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            # Strip complexity-gated scratchpad (emitted before JSON on complex cases)
            if "<scratchpad>" in text:
                text = text.split("</scratchpad>")[-1].strip()
            conclusion = parse_json_response(text, recover_truncated=True)
            if isinstance(conclusion, dict) and conclusion.get("_parse_fallback"):
                _pending_trace = None
                trace_llm_call(
                    span_name=span,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    model=getattr(_active_llm, "model", getattr(_active_llm, "model_id", "")),
                    provider=getattr(_active_llm, "name", ""),
                    duration_ms=duration_ms,
                    status="failed",
                    metadata={**pass_meta, "pass_changed": False},
                    error_message="llm returned invalid structured json",
                    error_type="InvalidStructuredJson",
                    input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
                    output_data={"response": text[:2000]},
                    api_endpoint=trace_meta_resp.get("api_endpoint"),
                    provider_request_id=trace_meta_resp.get("provider_request_id"),
                    ttft_ms=trace_meta_resp.get("ttft_ms"),
                    inference_time_ms=trace_meta_resp.get("inference_time_ms"),
                    queue_time_ms=trace_meta_resp.get("queue_time_ms"),
                )
                raise ValueError("llm returned invalid structured json")
            return conclusion, tokens, duration_ms

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            trace_llm_call(
                span_name=span,
                model=getattr(_active_llm, "model", getattr(_active_llm, "model_id", "")),
                provider=getattr(_active_llm, "name", ""),
                duration_ms=duration_ms,
                status="failed",
                metadata={**pass_meta, "pass_changed": False},
                error_message=str(exc)[:500],
                error_type=type(exc).__name__,
                input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
            )
            raise

    def _flush_trace(changed: bool | None = None) -> None:
        """Emit the deferred trace with the now-known pass_changed value."""
        nonlocal _pending_trace
        pt = _pending_trace
        if pt is None:
            return
        _pending_trace = None
        meta = pt["pass_meta"]
        if changed is not None:
            meta["pass_changed"] = changed
        _traced_llm = pt.get("llm_ref") or llm
        trace_llm_call(
            span_name=pt["span"],
            input_tokens=pt["usage"].get("input_tokens", 0),
            output_tokens=pt["usage"].get("output_tokens", 0),
            model=getattr(_traced_llm, "model", getattr(_traced_llm, "model_id", "")),
            provider=getattr(_traced_llm, "name", ""),
            duration_ms=pt["duration_ms"],
            metadata=meta,
            input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in pt["messages"]]},
            output_data={"response": pt["text"][:2000]} if pt["text"] else None,
            api_endpoint=pt["trace_meta_resp"].get("api_endpoint"),
            provider_request_id=pt["trace_meta_resp"].get("provider_request_id"),
            ttft_ms=pt["trace_meta_resp"].get("ttft_ms"),
            inference_time_ms=pt["trace_meta_resp"].get("inference_time_ms"),
            queue_time_ms=pt["trace_meta_resp"].get("queue_time_ms"),
        )

    # ------------------------------------------------------------------
    # Pass 1: CLASSIFY
    # ------------------------------------------------------------------
    payload_str = json.dumps(evidence_payload, separators=(",", ":"), sort_keys=True, default=str)
    p1_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=payload_str),
    ]
    p1_conclusion, p1_tokens, p1_duration = await _call_llm(p1_messages, 1, "classify")
    _flush_trace(changed=False)  # Pass 1 never "changes"
    p1_conclusion = _normalize(p1_conclusion)

    result.passes.append(PassResult(
        pass_number=1, pass_type="classify",
        conclusion=p1_conclusion, tokens_used=p1_tokens,
        duration_ms=p1_duration, changed=False,
    ))
    result.total_tokens += p1_tokens
    result.total_duration_ms += p1_duration
    result.passes_executed = 1
    result.final_conclusion = p1_conclusion

    if not enabled:
        # Still run deterministic verification in single-pass mode
        if verify_enabled:
            raw_evidence = evidence_payload.get("evidence", evidence_payload)
            p1_conclusion, verifier_issues, verifier_changed = _verify_evidence_sufficiency(
                p1_conclusion,
                raw_evidence,
                min_reviews=verify_min_reviews,
                min_snapshot_days=verify_min_snapshot_days,
                min_grounded_signals=verify_min_grounded_signals,
                confidence_cap=verify_confidence_cap,
            )
            if verifier_changed:
                p1_conclusion = _normalize(p1_conclusion)
                result.passes[0].conclusion = p1_conclusion
                result.final_conclusion = p1_conclusion
            result.boundary_conditions["verifier_issues"] = verifier_issues
            result.boundary_conditions["verifier_changed"] = verifier_changed
            result.passes[0].metadata = {
                "verifier_issues": verifier_issues,
                "verifier_changed": verifier_changed,
            }
        return result

    raw_evidence = evidence_payload.get("evidence", evidence_payload)
    if verify_enabled:
        p1_conclusion, verifier_issues, verifier_changed = _verify_evidence_sufficiency(
            p1_conclusion,
            raw_evidence,
            min_reviews=verify_min_reviews,
            min_snapshot_days=verify_min_snapshot_days,
            min_grounded_signals=verify_min_grounded_signals,
            confidence_cap=verify_confidence_cap,
        )
        if verifier_changed:
            p1_conclusion = _normalize(p1_conclusion)
            result.passes[0].conclusion = p1_conclusion
            result.final_conclusion = p1_conclusion
        if verifier_issues:
            logger.info(
                "Verifier adjusted classify output: %s",
                "; ".join(verifier_issues),
            )
        result.boundary_conditions["verifier_issues"] = verifier_issues
        result.boundary_conditions["verifier_changed"] = verifier_changed
        result.passes[0].metadata = {
            "verifier_issues": verifier_issues,
            "verifier_changed": verifier_changed,
        }
    else:
        result.boundary_conditions["verifier_issues"] = []
        result.boundary_conditions["verifier_changed"] = False

    # ------------------------------------------------------------------
    # ground_only mode: skip challenge, go straight to ground (used by
    # reconstitute to add evidence grounding without self-critique).
    # ------------------------------------------------------------------
    if ground_only:
        return await _run_ground_pass(
            result, p1_conclusion, raw_evidence, system_prompt,
            _call_llm, _flush_trace, _normalize, max_tokens, light_pass_max_tokens, 2,
            ground_mode="ground_only",
            use_llm=_llm_light,
        )

    # ------------------------------------------------------------------
    # Pass 2: CHALLENGE
    # ------------------------------------------------------------------
    p1_confidence = p1_conclusion.get("confidence", 0.5)
    challenge_attempted = False
    contradictions = finder(p1_conclusion, raw_evidence)
    should_challenge, challenge_reason = _should_attempt_challenge(
        conclusion=p1_conclusion,
        evidence=raw_evidence,
        contradictions=contradictions,
        confidence_floor=challenge_confidence_floor,
        min_reviews=challenge_min_reviews,
        mixed_polarity_min_share=challenge_mixed_polarity_min_share,
        high_impact_churn_density=challenge_high_impact_churn_density,
        high_impact_avg_urgency=challenge_high_impact_avg_urgency,
        high_impact_displacement_mentions=challenge_high_impact_displacement_mentions,
    )
    result.boundary_conditions["contradiction_count"] = len(contradictions)
    result.boundary_conditions["challenge_gate_reason"] = challenge_reason
    if not should_challenge:
        logger.info("Skipping challenge pass: %s", challenge_reason)
    else:
        challenge_attempted = True
        result.boundary_conditions["challenge_attempted"] = True
        challenge_prompt = _CHALLENGE_PROMPT.format(
            conclusion_json=json.dumps(p1_conclusion, indent=2, default=str),
            contradictions="\n".join(f"- {c}" for c in contradictions),
        )
        p2_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=payload_str),
            Message(role="assistant", content=json.dumps(p1_conclusion, default=str)),
            Message(role="user", content=challenge_prompt),
        ]
        try:
            p2_conclusion, p2_tokens, p2_duration = await _call_llm(
                p2_messages,
                2,
                "challenge",
                mt=min(max_tokens, light_pass_max_tokens),
                use_llm=_llm_light,
                extra_metadata={
                    "challenge_gate_reason": challenge_reason,
                    "contradiction_count": len(contradictions),
                },
            )
        except Exception as exc:
            logger.warning("Challenge pass failed; keeping classify conclusion: %s", exc)
        else:
            p2_conclusion = _normalize(p2_conclusion)

            p2_archetype_changed = p2_conclusion.get("archetype") != p1_conclusion.get("archetype")
            p2_confidence_delta = abs(
                p2_conclusion.get("confidence", 0.5) - p1_conclusion.get("confidence", 0.5)
            )
            p2_changed = p2_archetype_changed or p2_confidence_delta > ground_change_threshold
            _flush_trace(changed=p2_changed)

            result.passes.append(PassResult(
                pass_number=2, pass_type="challenge",
                conclusion=p2_conclusion, tokens_used=p2_tokens,
                duration_ms=p2_duration, changed=p2_changed,
                metadata={
                    "gate_reason": challenge_reason,
                    "contradiction_count": len(contradictions),
                },
            ))
            result.total_tokens += p2_tokens
            result.total_duration_ms += p2_duration
            result.passes_executed = 2
            result.final_conclusion = p2_conclusion
            result.boundary_conditions["challenge_changed"] = p2_changed

            logger.info(
                "Challenge pass: archetype %s->%s, confidence %.2f->%.2f, changed=%s (%s)",
                p1_conclusion.get("archetype"), p2_conclusion.get("archetype"),
                p1_conclusion.get("confidence", 0), p2_conclusion.get("confidence", 0),
                p2_changed, challenge_reason,
            )
    if not challenge_attempted:
        result.boundary_conditions["challenge_attempted"] = False

    # ------------------------------------------------------------------
    # Pass 3: GROUND -- runs after any attempted challenge pass so the
    # final answer cites concrete evidence even when the challenge defends
    # the original conclusion. It can also run as a cheap final check when
    # challenge is skipped and ground_always=True.
    # ------------------------------------------------------------------
    if not challenge_attempted and not ground_always:
        result.boundary_conditions["ground_mode"] = "skipped"
        return result

    pre_ground = result.final_conclusion
    pass_num = result.passes_executed + 1
    ground_mode = "ground_only" if ground_only else (
        "after_challenge" if challenge_attempted else "always"
    )
    return await _run_ground_pass(
        result, pre_ground, raw_evidence, system_prompt,
        _call_llm, _flush_trace, _normalize, max_tokens, light_pass_max_tokens, pass_num,
        ground_mode=ground_mode,
        use_llm=_llm_light,
    )


async def _run_ground_pass(
    result: MultiPassResult,
    pre_ground_conclusion: dict[str, Any],
    raw_evidence: dict[str, Any],
    system_prompt: str,
    _call_llm: Any,
    _flush_trace: Any,
    _normalize: Any,
    max_tokens: int,
    light_pass_max_tokens: int,
    pass_num: int,
    *,
    ground_mode: str,
    use_llm: Any = None,
) -> MultiPassResult:
    """Execute the grounding pass and update the result."""
    from ..services.protocols import Message

    evidence_fields_str = "\n".join(
        f"- {k}: {json.dumps(v, default=str)[:150]}"
        for k, v in raw_evidence.items()
    )
    ground_prompt = _GROUND_PROMPT.format(
        conclusion_json=json.dumps(pre_ground_conclusion, indent=2, default=str),
        evidence_fields=evidence_fields_str,
    )
    pg_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=ground_prompt),
    ]
    try:
        pg_conclusion, pg_tokens, pg_duration = await _call_llm(
            pg_messages,
            pass_num,
            "ground",
            mt=min(max_tokens, light_pass_max_tokens),
            use_llm=use_llm,
            extra_metadata={"ground_mode": ground_mode},
        )
    except Exception as exc:
        logger.warning("Ground pass failed; keeping prior conclusion: %s", exc)
        result.boundary_conditions["ground_mode"] = f"{ground_mode}_failed"
        return result
    pg_conclusion = _normalize(pg_conclusion)

    # Safety: if grounding removed all signals, force low confidence
    if not pg_conclusion.get("key_signals"):
        pg_conclusion["confidence"] = 0.3
        pg_conclusion["archetype"] = "mixed"
        pg_conclusion = _normalize(pg_conclusion)

    pg_changed = pg_conclusion.get("archetype") != pre_ground_conclusion.get("archetype")
    _flush_trace(changed=pg_changed)

    result.passes.append(PassResult(
        pass_number=pass_num, pass_type="ground",
        conclusion=pg_conclusion, tokens_used=pg_tokens,
        duration_ms=pg_duration, changed=pg_changed,
        metadata={
            "mode": ground_mode,
            "grounded_signal_count": len(pg_conclusion.get("key_signals", [])),
        },
    ))
    result.total_tokens += pg_tokens
    result.total_duration_ms += pg_duration
    result.passes_executed = pass_num
    result.final_conclusion = pg_conclusion
    result.boundary_conditions["ground_mode"] = ground_mode
    result.boundary_conditions["ground_changed"] = pg_changed
    result.boundary_conditions["grounded_signal_count"] = len(pg_conclusion.get("key_signals", []))

    logger.info(
        "Ground pass: archetype %s, confidence %.2f, %d signals grounded",
        pg_conclusion.get("archetype"),
        pg_conclusion.get("confidence", 0),
        len(pg_conclusion.get("key_signals", [])),
    )

    return result
