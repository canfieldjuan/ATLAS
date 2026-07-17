"""Deterministic copy-verification gate for Content Factory drafts (Phase 4.1).

The #2116 contract gave ``EditorialAudit`` a ``copy_verification`` field and a rule
that a draft may not be promoted unless ``copy_verification.verdict == "pass"`` -- but
nothing PRODUCED that verdict, so the gate was a shape with no teeth. This module is the
producer: a pure, deterministic scan of draft text that fails the verdict when the copy
contains a forbidden marketing claim or raw contact PII.

WHAT THIS GATE IS -- and is not. It is a deterministic BEST-EFFORT BACKSTOP that catches
the common wordings of the operator's promote-blocking claim categories. It is NOT a
complete natural-language classifier: no regex catalogue can enumerate every paraphrase
of "guaranteed savings", so a novel wording can still pass. The real safety guarantee is
the human approval step before publish (nothing leaves the box unapproved); this gate
reduces that reviewer's load and hard-blocks the obvious overclaims and PII. It fails the
verdict conservatively (any hit blocks promotion) but a "pass" means "no known-bad pattern
matched", not "provably clean".

The claim categories and PII patterns originate from the operator's "Resolution Audit
Draft Verifier" tool (authored by Juan Canfield / Codex), which lived only in the Open
WebUI database and was therefore un-versioned. This repo module is now the canonical gate
(the OWUI copy is superseded and should be re-synced from here). Relative to the source it
carries operator-authorized corrections and same-category coverage broadening:

  - the ``%``-boundary gap is closed (``30%`` is caught like ``30 percent``);
  - negation detection is scoped to the words immediately before the claim, so an
    unrelated earlier negation ("No setup fee, and guaranteed savings ...") no longer
    suppresses a real hit;
  - each category matches common inflections/modifiers, not one fixture wording.

Categories (all promote-blocking -- "Do not post yet"): forbidden OUTCOME claims
(guaranteed savings, fixed deflection %, ticket reductions), forbidden AUTOMATION claims
(auto-publishing / auto-answering), REPLACING-AGENTS / avoided-hire claims, and raw
contact PII (email / phone). The source tool's softer "needs human review" layer
(answer/ownership qualifiers, owner-routing coverage, CTA reminder) is a later slice, as
is wiring this producer into the runner / Phase 4.2 Filter.
"""

from __future__ import annotations

import re

from atlas_brain.schemas.content_factory import CopyVerification

# Forbidden-claim catalogue. Each pattern targets a promote-blocking category and matches
# the common inflections/modifiers of that category, not a single fixture wording. Still a
# best-effort backstop (see module docstring): a novel paraphrase can pass.
_RULES: dict[str, list[tuple[str, str]]] = {
    "outcomes": [
        # "guaranteed savings", "guaranteed cost/monthly savings", "guaranteed 30%
        # savings", "guarantees savings". Up to 3 tokens (incl. numeric/percent ones)
        # may sit between the guarantee verb and "savings".
        ("guaranteed-savings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$.,-]+){0,3}\s+savings\b"),
        ("guaranteed-rankings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$.,-]+){0,3}\s+rankings\b"),
        # Fixed deflection percentage, "%" or spelled-out "percent".
        ("fixed-deflection-percent", r"\b\d{1,3}\s*(?:%|percent)\s+deflection\b"),
        # Fixed ticket reductions: "reduce (support) tickets/ticket volume by N%/percent".
        (
            "fixed-ticket-reduction",
            r"\b(?:cut|cuts|reduce|reduces|lower|lowers|drop|drops|shrink|shrinks)\s+(?:your\s+|the\s+)?(?:support\s+)?tickets?(?:\s+volume)?\s+by\s+\d{1,3}\s*(?:%|percent\b)",
        ),
        # "N% fewer/less (support) tickets".
        (
            "fixed-fewer-tickets",
            r"\b\d{1,3}\s*(?:%|percent)\s+(?:fewer|less)\s+(?:support\s+)?tickets?\b",
        ),
    ],
    "automation": [
        ("live-help-center-publishing", r"\blive\s+help[- ]center\s+publishing\b"),
        ("automatic-help-center-updates", r"\bautomatic\s+help[- ]center\s+updates\b"),
        ("automatic-ticket-answering", r"\bautomatic\s+ticket\s+answering\b"),
        # auto-publish / auto-publishes / auto-publishing / auto-published.
        ("auto-publish", r"\bauto[- ]publish(?:es|ing|ed)?\b"),
        (
            "automatically-updates-help-center",
            r"\bautomatically\s+(?:updates?|publish(?:es)?)\s+(?:your\s+)?help[- ]center\b",
        ),
        ("answers-tickets-automatically", r"\banswers?\s+tickets?\s+automatically\b"),
    ],
    "replacing_agents": [
        # "replace (your/our/the) (support) agents" as the direct object. The negative
        # lookahead excludes a possessive ("replace your agents' spreadsheet"), where the
        # thing replaced is not the agents.
        (
            "replace-agents",
            r"\breplac(?:e|es|ing)\s+(?:your\s+|our\s+|the\s+)?(?:support\s+)?agents?\b(?![\u0027\u2019])",
        ),
        # "avoid a support hire", "avoid hiring (another) (support) agent". Anchored on
        # hire/hiring semantics -- NOT any nearby "agents" (which would match benign copy
        # like "avoid distracting your agents").
        (
            "avoid-support-hire",
            r"\bavoid(?:s|ing)?\s+(?:\w+\s+){0,3}(?:hire|hiring)\b",
        ),
    ],
}

# Raw contact PII. A match blocks promotion; the matched value is NEVER copied into the
# verdict's hits (that JSON is persisted in the git-backed job folder), only a redacted
# marker -- otherwise the gate would duplicate the very PII it exists to block.
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")

# A negation directly before the claim (no/not/never/without, or an -n't contraction).
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without)\b|n't\b", re.I)


def _is_negated(text: str, start: int) -> bool:
    """True when the claim at ``start`` is negated by the words IMMEDIATELY before it.

    Only the last two words of the current segment are considered (a segment ends at
    ``.!?;,`` or newline), so a real negation ("no guaranteed savings", "does not promise
    guaranteed savings") suppresses the hit, but an UNRELATED earlier negation in the same
    sentence ("No setup fee, and guaranteed savings ...", "does not delay launch and
    guarantees savings") does not -- the source tool scanned the whole prefix and leaked
    those false negatives."""
    segment_start = max(text.rfind(mark, 0, start) for mark in ".!?;,\n")
    prefix = text[segment_start + 1 : start]
    window = " ".join(prefix.split()[-2:])
    return bool(_NEGATION_RE.search(window))


def _claim_hits(text: str) -> list[str]:
    """"code: evidence" for each non-negated forbidden-claim match. Claim evidence is the
    marketing phrase itself (safe and useful for a human reviewer), not PII."""
    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, pattern in _RULES[category]:
            for match in re.finditer(pattern, text, re.I):
                if not _is_negated(text, match.start()):
                    hits.append(f"{code}: {match.group(0)}")
    return hits


def verify_copy(text: str) -> CopyVerification:
    """Deterministically verify draft copy, producing the ``CopyVerification`` verdict
    the #2116 EditorialAudit contract requires for promotion.

    The verdict is ``"fail"`` when the copy contains any forbidden marketing claim
    (outcome / automation / replacing-agents) or raw contact PII (email / phone), and
    ``"pass"`` otherwise. A ``fail`` verdict makes ``recommendation == "promote"`` invalid,
    so a model cannot self-promote copy that overclaims or leaks PII. PII hits record only a
    redacted marker, so the verdict (persisted in the job folder) never re-exposes the
    contact data. A ``"pass"`` means no known-bad pattern matched, not provably clean -- the
    human approval step remains the real gate (see module docstring)."""
    if not isinstance(text, str):
        raise TypeError("verify_copy requires a string; draft body is text")

    hits = _claim_hits(text)
    # PII: block on a match, but redact the value out of the persisted hit.
    if _EMAIL_RE.search(text):
        hits.append("email: <redacted>")
    if _PHONE_RE.search(text):
        hits.append("phone: <redacted>")

    verdict = "fail" if hits else "pass"
    return CopyVerification(verdict=verdict, hits=hits)
