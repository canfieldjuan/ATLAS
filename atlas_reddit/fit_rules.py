"""Claim-safety rule catalogue for the Reddit fit pass (v2 S1, #1931).

This module is the single source of truth for what model-written fit
output may never say. It is born with the evaluation harness so the
ruler exists before any model integration: the S1 harness grades fixture
predictions against these rules, the S2 prompt builder renders its
"do not" boundaries from these rule messages, and the S3 runtime guard
enforces exactly this catalogue (a parity test pins that).

The taxonomy consolidates the repo's existing support-ticket product
truth: the six marketing-claim codes from the Content Ops claim audit
script, the landing-page/blog-post skill claim limits, and the fit-pass
specific families (reply drafting, write-action posture, stealth
promotion). PII detectors are simplified copies of the deflection-report
constants. Everything is re-declared locally: atlas_reddit is
deliberately standalone and imports nothing from atlas_brain,
extracted_* packages, or scripts/.

Pure and deterministic: no I/O, no network, no clock, no randomness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# -- fit output contract constants (shared: harness S1, parser S2, guard S3,
# CLI knob checks; the MAX_* pattern mirrors config.py's knob ceilings) -------

FIT_VERDICTS: tuple[str, ...] = ("yes", "maybe", "no")
FIT_RISK_FLAGS: tuple[str, ...] = (
    "promo_risk",
    "unsupported_outcome",
    "vendor_bait",
    "pii_risk",
    "low_context",
)
MAX_FIT_REASON_CHARS = 280
MAX_FIT_ANGLE_CHARS = 280

# Closed parse-error taxonomy for prediction envelopes. Emitters (the S5
# client, S6 runner) may ONLY use these codes; the harness rejects
# anything else at load, because envelope codes flow into summaries and a
# free-form "code" can smuggle model output or PII past privacy stripping.
PARSE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "model_returned_prose",
        "model_output_invalid_json",
        "model_output_schema_mismatch",
        "model_empty_response",
        "model_timeout",
        "model_http_error",
    }
)


@dataclass(frozen=True)
class FitRule:
    """One banned-claim family: stable code, detection pattern, and the
    human message the prompt builder renders as a boundary bullet."""

    code: str
    pattern: str
    message: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class FitFinding:
    """One rule hit. Carries the code and character span only -- never the
    matched text -- so downstream summaries stay privacy-strippable."""

    code: str
    message: str
    start: int
    end: int


_HELP_TARGET = r"(?:help[- ]?cent(?:er|re)|knowledge[- ]?base|docs? site|documentation)"
_HELPDESK_PLATFORMS = (
    r"\b(?:zendesk|intercom|gorgias|freshdesk|help ?scout"
    r"|salesforce(?: service cloud)?|hubspot(?: service hub)?|shopify|front"
    r"|jira(?: service management)?)\b"
)

# Outcome/product-truth families. Uploaded or discussed support-ticket data
# shows repeated questions and operational gaps; it does not prove future
# ticket reduction, deflection, ROI, time savings, churn impact, rankings,
# or integrations. See the consolidated taxonomy in the plan doc.
CLAIM_RULES: tuple[FitRule, ...] = (
    FitRule(
        code="GUARANTEED_DEFLECTION",
        pattern=(
            r"guarantee[ds]?\b[^.]{0,60}\b(?:reduction|deflection|fewer tickets)"
            r"|(?:cut|reduce|deflect)\w*\s+(?:your |their |the )?(?:support )?"
            r"ticket(?:s| volume)?\s+by\s+\d+"
            r"|(?:cut|reduce|deflect)\w*\s+\d+\s?%\s+of\s+"
            r"(?:your\s+|their\s+|the\s+)?(?:support\s+)?tickets"
            r"|\d+\s?%\s+(?:fewer|less)\s+(?:support\s+)?tickets"
        ),
        message="Never guarantee or quantify ticket reduction or deflection.",
    ),
    FitRule(
        code="TICKET_REDUCTION_PROMISE",
        pattern=(
            r"ticket(?:s| volume)?\s+(?:will|would)\s+(?:drop|fall|shrink|decrease)"
            r"|prevent(?:s|ed)?\s+(?:future\s+)?tickets"
            r"|queue\s+will\s+shrink"
            r"|(?:deflect|reduce|cut|lower)s?\s+(?:your\s+|their\s+|the\s+)?"
            r"(?:support\s+)?(?:tickets?(?:\s+volume)?|volume|load)"
            r"|answers?\b[^.]{0,30}\b(?:before|without)\s+"
            r"(?:customers?\s+)?open(?:ing)?\s+(?:a\s+)?tickets?"
        ),
        message="Never promise support volume will drop, even unquantified.",
    ),
    FitRule(
        code="ROI_SAVINGS",
        pattern=(
            r"\bROI\b|return on investment"
            r"|(?:cost|time)\s+savings?|hours?\s+saved"
            r"|save\s+(?:them|you|your\s+team|the\s+team|teams?)\s+"
            r"(?:\d+\s+)?(?:time|money|hours)"
            r"|capacity\s+gains?|free\s+up\s+(?:the\s+)?(?:team|agents|capacity)"
        ),
        message="Never claim ROI, cost/time savings, or capacity gains.",
    ),
    FitRule(
        code="RETENTION_CHURN_OUTCOME",
        pattern=(
            r"churn\s+(?:less|will\s+drop|reduction)"
            r"|(?:reduce|cut|lower)s?\s+churn"
            r"|(?:improve|boost)s?\s+(?:their\s+|your\s+)?retention"
            r"|retention\s+(?:improves?|will\s+improve)"
            r"|more\s+likely\s+to\s+(?:stay|renew)"
            r"|stop\s+cancell?ations"
        ),
        message="Never claim churn or retention impact.",
    ),
    FitRule(
        code="RANKING_SEO_OUTCOME",
        pattern=(
            r"rank\s+(?:higher|better|first)|search\s+ranking"
            r"|seo\s+(?:win|boost|improvement)"
            r"|(?:improve|boost)s?\s+(?:their\s+|your\s+)?seo\b"
            r"|top\s+of\s+(?:google|search)"
        ),
        message="Never claim search-ranking or SEO outcomes.",
    ),
    FitRule(
        code="FIX_RESOLVE_PROMISE",
        pattern=(
            r"\b(?:we|it|this|that|(?:the|this|our)\s+tool)\s+"
            r"(?:can|will|could)\s+(?:fix|resolve|solve|handle|eliminate)\b"
            r"|this\s+(?:can|will)\s+be\s+(?:fixed|resolved|solved)\s+"
            r"(?:by|with)\s+(?:our|the\s+tool)"
        ),
        message="Never promise that we can fix or resolve their problem.",
    ),
    FitRule(
        code="AUTO_PUBLISH",
        pattern=(
            r"auto[- ]?publish|automatically\s+publish(?:es)?"
            r"|publish(?:es)?\s+directly\s+to\s+(?:the\s+|your\s+|their\s+)?"
            + _HELP_TARGET
        ),
        message="Never claim automatic help-center or knowledge-base publishing.",
    ),
    FitRule(
        code="LIVE_HELPDESK_INTEGRATION",
        pattern=(
            r"(?:connect(?:s|ed)?\s+(?:to\s+|with\s+)?"
            r"|(?:native\s+)?integrat(?:es?|ed|ion)\s+with\s+)"
            + _HELPDESK_PLATFORMS
            + r"|(?:native\s+)?" + _HELPDESK_PLATFORMS + r"\s+integration"
        ),
        message="Never claim live help-desk integrations.",
    ),
    FitRule(
        code="SEMANTIC_CLUSTERING",
        pattern=(
            r"semantic(?:ally)?\s+cluster\w*|embedding[- ]based\s+cluster\w*"
            r"|cluster\w*\b[^.]{0,30}\bembeddings?\b"
            r"|embeddings?\b[^.]{0,30}\bcluster\w*"
        ),
        message="Never claim semantic or embedding-based clustering.",
    ),
    FitRule(
        code="COST_RANKING",
        pattern=(
            r"rank(?:s|ed|ing)?\s+(?:\w+\s+){0,2}by\s+cost"
            r"|cost[- ]rank\w*|support\s+cost\s+ranking"
        ),
        message="Never claim cost ranking without imported cost data.",
    ),
    FitRule(
        code="UNBOUNDED_HOSTED_UPLOADS",
        pattern=(
            r"unlimited\s+(?:ticket|upload|row)s?"
            r"|(?:50,?000|fifty\s+thousand)\s+(?:hosted|synchronous)\b"
            r"|(?:hosted|synchronous)\s+uploads?\b[^.]{0,40}"
            r"\b(?:50,?000|fifty\s+thousand)"
            r"|(?:handle|process|support)s?\s+(?:up\s+to\s+)?"
            r"(?:50,?000|fifty\s+thousand)\s+"
            r"(?:tickets?|rows?|uploads?|ticket\s+exports?)"
        ),
        message="Never claim unbounded or 50k hosted uploads.",
    ),
    FitRule(
        code="SELF_PROMO_PITCH",
        pattern=(
            r"\b(?:our|my)\s+(?:tool|product|platform|audit|service|app)\b"
            r"|\b(?:we|i)\s+(?:built|have|offer)\s+a\s+"
            r"(?:tool|product|service|platform|app)\b"
            r"|\b(?:we|i)\s+built\b|\bmy\s+company\b|check\s+out\s+(?:our|my)"
            r"|\bsign\s+up\b|free\s+trial|book\s+a\s+(?:demo|call)"
        ),
        message="Never pitch: no product mentions, trials, demos, or sign-ups.",
    ),
)

# Reply-draft posture: output that reads as a ready-to-paste Reddit reply or
# proposes write actions. The fit pass is advisory only -- no posting,
# commenting, voting, scheduling, or outreach drafting, ever.
POSTURE_RULES: tuple[FitRule, ...] = (
    FitRule(
        code="REPLY_DRAFT",
        pattern=(
            r"^(?:hey|hi|hello)\s+(?:op|there|everyone|u/)"
            r"|\bi(?:'d|\s+would)\s+(?:post|comment|write|say|reply|respond"
            r"|start\s+by\s+saying)\b"
            r"|feel\s+free\s+to\s+(?:dm|pm|reach\s+out)"
            r"|\b(?:dm|pm)\s+(?:me|us)\b|reach\s+out\s+to\s+(?:me|us)"
            r"|happy\s+to\s+(?:chat|help\s+directly|connect)"
        ),
        message="Never draft reply text or invite direct contact.",
    ),
    FitRule(
        code="WRITE_ACTION_POSTURE",
        pattern=(
            r"post\s+this\s+(?:as\s+a\s+)?(?:comment|reply)"
            r"|\b(?:reply|respond)\s+to\s+the\s+(?:thread|post|op)\b"
            r"|leave\s+a\s+comment"
            r"|i\s+(?:will|would|can)\s+(?:comment|post|respond)\s+(?:on|to)\b"
            r"|\b(?:upvote|downvote)\b"
            r"|schedule\s+(?:a\s+)?(?:post|comment|follow[- ]?up)"
        ),
        message="Never propose posting, voting, or scheduling actions.",
    ),
)

# PII detectors for SHORT model output (reason/angle sentences). Simplified
# from the deflection-report constants; detection-and-block, never scrub.
PII_RULES: tuple[FitRule, ...] = (
    FitRule(
        code="PII_EMAIL",
        pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        message="Never echo email addresses from the thread.",
    ),
    FitRule(
        code="PII_PHONE",
        pattern=(
            r"(?<![\d.])(?:\+?1[\s.-]?)?"
            r"(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\d.])"
            r"|(?:call|phone|text|dial|reach|cell|mobile|number)"
            r"(?:\s+\S+){0,3}\s+"
            r"(?<![\d.])\+?1?\d{10}(?![\d.])"
        ),
        message="Never echo phone numbers from the thread.",
    ),
    FitRule(
        code="PII_SSN",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        message="Never echo SSN-shaped identifiers.",
    ),
    FitRule(
        code="PII_PAYMENT_CARD",
        pattern=r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b|\b\d{13,16}\b",
        message="Never echo payment-card-shaped numbers.",
    ),
    FitRule(
        code="PII_PERSON_NAME",
        pattern=(
            r"(?i:(?:customer|user|client|their|his|her)\s+name\s+is\s+)"
            r"(?!(?:Redacted|Unknown|Not|Unavailable|Withheld|Missing|Private)\b)"
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
            r"|(?i:(?:support\s+lead|team\s+lead|agent|manager|rep|owner"
            r"|founder|user|customer|client|colleague|employee)\s+)"
            r"(?!(?:Success|Team|Support|Service|Experience|Operations"
            r"|Redacted|Unknown)\b)"
            r"[A-Z][a-z]+\s+"
            r"(?!(?:Success|Team|Support|Service|Experience|Operations)\b)"
            r"[A-Z][a-z]+"
        ),
        message="Never echo personal names surfaced in the thread.",
        case_sensitive=True,
    ),
    FitRule(
        code="PII_IDENTIFIER",
        pattern=(
            r"(?:account|order|case|ticket|invoice|ref(?:erence)?)\s*"
            r"(?:(?:number|no\.?|id)\s*[:\s]\s*|#\s*|\s)"
            r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{4,}\b"
        ),
        message="Never echo account, order, or ticket identifiers.",
    ),
)

RULES: tuple[FitRule, ...] = CLAIM_RULES + POSTURE_RULES + PII_RULES

CLAIM_CODES: frozenset[str] = frozenset(rule.code for rule in CLAIM_RULES)
POSTURE_CODES: frozenset[str] = frozenset(rule.code for rule in POSTURE_RULES)
PII_CODES: frozenset[str] = frozenset(rule.code for rule in PII_RULES)
ALL_RULE_CODES: frozenset[str] = frozenset(rule.code for rule in RULES)


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[FitRule, re.Pattern[str]], ...]:
    compiled = []
    for rule in RULES:
        flags = re.MULTILINE
        if not rule.case_sensitive:
            flags |= re.IGNORECASE
        compiled.append((rule, re.compile(rule.pattern, flags)))
    return tuple(compiled)


def scan_fit_text(
    text: str, *, pii_allowlist: frozenset[str] = frozenset()
) -> tuple[FitFinding, ...]:
    """Scan one short model-output string against the full catalogue.

    Returns one finding per (rule, match). A PII finding is suppressed only
    when its exact matched text appears in ``pii_allowlist`` (per-fixture
    labels in the harness; the runtime guard passes an empty set).
    """
    findings: list[FitFinding] = []
    for rule, pattern in _compiled():
        for match in pattern.finditer(text):
            if rule.code in PII_CODES and match.group(0) in pii_allowlist:
                continue
            findings.append(
                FitFinding(
                    code=rule.code,
                    message=rule.message,
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(findings)
