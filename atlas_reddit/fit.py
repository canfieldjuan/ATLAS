"""Fit output contract: FitDecision, the strict parser, and the prompt
builder (v2 S2, #1931).

This module is the runtime half of the ruler the S1 harness shipped. The
parser is the authoritative gate on model output -- the harness's shape
checker now delegates here, so fixture grading and runtime parsing can
never drift. The prompt builder renders its "never do this" boundaries
directly from the fit_rules catalogue messages: the model is told, graded
on, and (from S3) blocked by the same rules.

Pure and deterministic: no I/O, no network, no clock, no randomness, no
model calls. The S5 client will feed raw model text into
``parse_fit_decision`` and translate ``FitParseError.code`` into the
prediction-envelope ``parse_error`` field.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .fit_rules import (
    CLAIM_RULES,
    FIT_RISK_FLAGS,
    FIT_VERDICTS,
    MAX_FIT_ANGLE_CHARS,
    MAX_FIT_REASON_CHARS,
    PII_RULES,
    POSTURE_RULES,
)

PROMPT_VERSION = "fit.v1"

_PREDICTION_KEYS = frozenset({"verdict", "reason", "angle", "risk_flags"})
_WHITESPACE_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)

# Strict-mode-safe wire schema: type/enum/required/additionalProperties
# only. Length caps are deliberately absent -- the parser is the
# authoritative gate, and some OpenAI-compatible backends reject
# unsupported keywords in strict mode.
FIT_OUTPUT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(FIT_VERDICTS)},
        "reason": {"type": "string"},
        "angle": {"type": ["string", "null"]},
        "risk_flags": {
            "type": "array",
            "items": {"type": "string", "enum": list(FIT_RISK_FLAGS)},
        },
    },
    "required": ["verdict", "reason", "angle", "risk_flags"],
    "additionalProperties": False,
}


class FitParseError(Exception):
    """Model output that failed the contract. ``code`` is always one of
    the closed PARSE_ERROR_CODES so it can ride in a prediction envelope;
    ``problems`` carries the per-field shape codes when applicable."""

    def __init__(self, code: str, problems: tuple[str, ...] = ()) -> None:
        detail = f" ({', '.join(problems)})" if problems else ""
        super().__init__(f"{code}{detail}")
        self.code = code
        self.problems = problems


@dataclass(frozen=True)
class FitDecision:
    """One parsed, contract-valid fit judgment. ``angle`` is None exactly
    when the verdict is ``no`` (canonicalized from null or empty)."""

    verdict: str
    reason: str
    angle: str | None
    risk_flags: tuple[str, ...]


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def fit_decision_problems(prediction: dict) -> tuple[str, ...]:
    """Deterministic strict-shape validation of the model-output object.

    The single source of shape truth: the S1 harness's
    ``check_prediction_shape`` delegates here, so the ruler and the
    runtime parser cannot drift. Codes are stable snake identifiers.
    """
    problems: list[str] = []
    keys = set(prediction)
    if keys - _PREDICTION_KEYS:
        problems.append("unknown_keys")
    if _PREDICTION_KEYS - keys:
        problems.append("missing_keys")
    verdict = prediction.get("verdict")
    if "verdict" in prediction and verdict not in FIT_VERDICTS:
        problems.append("verdict_invalid")
    reason = prediction.get("reason")
    if "reason" in prediction:
        if not isinstance(reason, str) or not _collapse(reason):
            problems.append("reason_missing")
        elif len(_collapse(reason)) > MAX_FIT_REASON_CHARS:
            problems.append("reason_too_long")
    angle = prediction.get("angle")
    if "angle" in prediction and "verdict" in prediction and verdict in FIT_VERDICTS:
        if verdict in ("yes", "maybe"):
            if not isinstance(angle, str) or not _collapse(angle):
                problems.append("angle_required")
            elif len(_collapse(angle)) > MAX_FIT_ANGLE_CHARS:
                problems.append("angle_too_long")
        else:  # verdict == "no": advisory text on a rejected thread is
            # exactly where pitch language leaks; require null/empty.
            if angle is not None and _collapse(str(angle)):
                problems.append("angle_forbidden_for_no")
    flags = prediction.get("risk_flags")
    if "risk_flags" in prediction:
        if not isinstance(flags, list):
            problems.append("risk_flags_invalid")
        elif any(not isinstance(flag, str) for flag in flags):
            problems.append("risk_flags_invalid")
        else:
            if any(flag not in FIT_RISK_FLAGS for flag in flags):
                problems.append("risk_flag_unknown")
            if len(set(flags)) != len(flags):
                problems.append("risk_flag_duplicate")
    return tuple(problems)


def parse_fit_decision(raw: str | dict) -> FitDecision:
    """Parse raw model output (a JSON string, optionally fenced, or an
    already-decoded object) into a FitDecision, or raise FitParseError
    with a closed-taxonomy code. This is the authoritative gate: wire
    schemas are best-effort, the parser is not."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise FitParseError("model_empty_response")
        fenced = _FENCE_RE.match(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FitParseError("model_output_invalid_json") from exc
    else:
        decoded = raw
    if not isinstance(decoded, dict):
        raise FitParseError("model_output_schema_mismatch")
    problems = fit_decision_problems(decoded)
    if problems:
        raise FitParseError("model_output_schema_mismatch", problems)
    verdict = decoded["verdict"]
    angle_raw = decoded["angle"]
    angle = _collapse(angle_raw) if isinstance(angle_raw, str) else None
    if verdict == "no" or not angle:
        angle = None
    return FitDecision(
        verdict=verdict,
        reason=_collapse(decoded["reason"]),
        angle=angle,
        risk_flags=tuple(decoded["risk_flags"]),
    )


# -- prompt builder ------------------------------------------------------------

_PRODUCT_TRUTH = (
    "Support-ticket or support-thread evidence shows repeated questions, "
    "customer wording, likely FAQ or help-center opportunities, and "
    "operational gaps. It does NOT prove future ticket reduction, "
    "guaranteed deflection, churn or retention impact, ROI, time savings, "
    "capacity gains, rankings, or resolution improvements, and you must "
    "never imply that it does."
)

_ALLOWED_LANGUAGE = (
    'Safe framing sounds like: "This thread is a fit because it discusses '
    'repeated support questions / unresolved support demand / broken '
    'help-center discovery / cost concern." A useful angle asks about the '
    "evidence in their own tickets and what an audit of that evidence "
    "could identify. Do not pitch. Do not write a reply. Do not promise "
    "outcomes."
)


def _boundary_bullets() -> tuple[str, ...]:
    """One bullet per catalogue rule message: the model is told exactly
    the rules the harness grades and the runtime guard enforces."""
    seen: list[str] = []
    for rule in CLAIM_RULES + POSTURE_RULES + PII_RULES:
        if rule.message not in seen:
            seen.append(rule.message)
    return tuple(f"- {message}" for message in seen)


def build_fit_prompt(
    *,
    title: str,
    subreddit: str,
    body: str = "",
    matched_topics: tuple[str, ...] = (),
    keyword_score: float | None = None,
    reddit_score: int | None = None,
    num_comments: int | None = None,
) -> tuple[dict, ...]:
    """Build the deterministic two-message prompt for one candidate.

    ``body`` may be empty (pre-v5 stores have no body_excerpt); the
    prompt then carries an explicit no-body marker and invites the
    low_context risk flag rather than letting the model invent content.
    """
    boundaries = "\n".join(_boundary_bullets())
    system = (
        "You judge whether one Reddit thread is a FIT for a support-ops "
        "resolution audit conversation. You are an advisory analyst only: "
        "this system is read-only, never posts, comments, votes, "
        "schedules, or contacts anyone, and you never draft reply text.\n\n"
        f"Product truth: {_PRODUCT_TRUTH}\n\n"
        f"{_ALLOWED_LANGUAGE}\n\n"
        "Hard boundaries (violations are rejected by a deterministic "
        "guard):\n"
        f"{boundaries}\n\n"
        "Verdicts: 'yes' = concrete, repeated, unresolved support demand "
        "or operational gap worth a conversation; 'maybe' = "
        "support-adjacent but thin operational detail; 'no' = not a fit "
        "(growth threads, job posts, tool-recommendation bait, anything "
        "where engaging would read as vendor promotion).\n"
        "Risk flags (use any that apply): promo_risk, "
        "unsupported_outcome, vendor_bait, pii_risk, low_context.\n\n"
        "Respond with ONLY a JSON object, no prose, exactly these keys:\n"
        '{"verdict": "yes|maybe|no", "reason": "<one grounded sentence, '
        f'max {MAX_FIT_REASON_CHARS} chars>", "angle": "<for yes/maybe: '
        "one honest question-first angle grounded in their own evidence, "
        f'max {MAX_FIT_ANGLE_CHARS} chars; for no: null>", '
        '"risk_flags": ["..."]}\n'
        "A 'no' verdict must carry angle: null."
    )
    if body.strip():
        body_block = _collapse(body)
    else:
        body_block = (
            "[no body available -- judge from the title only and add the "
            "low_context risk flag if this limits your judgment]"
        )
    details = [
        f"Subreddit: r/{_collapse(subreddit)}",
        f"Title: {_collapse(title)}",
        f"Body: {body_block}",
    ]
    if matched_topics:
        details.append("Matched watchlist topics: " + ", ".join(matched_topics))
    if keyword_score is not None:
        details.append(f"Keyword score: {keyword_score}")
    if reddit_score is not None:
        details.append(f"Reddit score: {reddit_score}")
    if num_comments is not None:
        details.append(f"Comments: {num_comments}")
    user = "Candidate thread:\n" + "\n".join(details)
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )
