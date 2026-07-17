"""Deterministic copy-verification gate for Content Factory drafts (Phase 4.1).

The #2116 contract gave ``EditorialAudit`` a ``copy_verification`` field and a rule
that a draft may not be promoted unless ``copy_verification.verdict == "pass"`` -- but
nothing PRODUCED that verdict, so the gate was a shape with no teeth. This module is the
producer: a pure, deterministic scan of draft text that fails the verdict when the copy
contains a forbidden marketing claim or raw contact PII.

The banned-claim catalogue and PII patterns are ported VERBATIM from the operator's
"Resolution Audit Draft Verifier" copy_verification tool (authored by Juan Canfield /
Codex), which lived only in the Open WebUI database and was therefore un-versioned --
with ONE operator-authorized fix, annotated at the ``fixed-ticket-volume-reduction`` rule,
that closes a ``%``-boundary gap so "30%" is caught like "30 percent" (a strict tightening).
This repo module is now the canonical gate; the Open WebUI copy is superseded. This is the
blocker core of that tool -- the categories it marks "Do not post yet":

  - forbidden OUTCOME claims (guaranteed savings, fixed deflection %, ticket reductions),
  - forbidden AUTOMATION claims (auto-publishing / auto-answering the help center),
  - REPLACING-AGENTS / avoided-hire claims,
  - raw contact PII (email addresses, phone-number-shaped strings).

Matching is negation-aware (``no guaranteed savings`` is not a hit), matching the source
tool. The tool's softer "needs human review" layer (answer/ownership qualifiers, owner-
routing coverage, CTA reminder) is intentionally NOT ported here -- those are warnings,
not promote-blockers, and are a later slice. Wiring this producer into the runner /
Phase 4.2 Filter is also a later slice; this is the deterministic core those will call.
"""

from __future__ import annotations

import re

from atlas_brain.schemas.content_factory import CopyVerification

# Forbidden-claim catalogue, ported verbatim from the operator's copy_verification tool.
# A match in any category blocks promotion (verdict "fail").
_RULES: dict[str, list[tuple[str, str]]] = {
    "outcomes": [
        ("guaranteed-savings", r"\bguaranteed\s+savings\b"),
        ("guarantees-savings", r"\bguarantees?\s+savings\b"),
        ("guaranteed-rankings", r"\bguaranteed\s+rankings\b"),
        ("fixed-deflection-percent", r"\b\d{1,3}\s*%\s+deflection\b"),
        (
            # Operator-authorized fix to the one source quirk: the source ended this
            # rule `(?:%|percent)\b`, but `\b` after `%` can never match (`%` before a
            # space is two non-word chars), so "30%" slipped through while "30 percent"
            # did not. Moving `\b` inside the alternation guards only the word "percent"
            # and lets a bare "%" match -- a strict tightening of the gate, not a
            # loosening. (The other percent rules anchor `%` with following text, so
            # they never had this gap.)
            "fixed-ticket-volume-reduction",
            r"\b(?:cut|cuts|reduce|reduces|lower|lowers|drop|drops|shrink|shrinks)\s+ticket\s+volume\s+by\s+\d{1,3}\s*(?:%|percent\b)",
        ),
        (
            "fixed-fewer-tickets",
            r"\b\d{1,3}\s*(?:%|percent)\s+(?:fewer|less)\s+tickets?\b",
        ),
    ],
    "automation": [
        ("live-help-center-publishing", r"\blive\s+help[- ]center\s+publishing\b"),
        ("automatic-help-center-updates", r"\bautomatic\s+help[- ]center\s+updates\b"),
        ("automatic-ticket-answering", r"\bautomatic\s+ticket\s+answering\b"),
        ("auto-published", r"\bauto[- ]published\b"),
        (
            "automatically-updates-help-center",
            r"\bautomatically\s+updates?\s+(?:your\s+)?help[- ]center\b",
        ),
        ("answers-tickets-automatically", r"\banswers?\s+tickets?\s+automatically\b"),
    ],
    "replacing_agents": [
        ("replace-agents", r"\breplac(?:e|es|ing)\s+(support\s+)?agents?\b"),
        ("avoid-support-hire", r"\bavoid\s+a\s+support\s+hire\b"),
    ],
}

# Raw contact PII, ported verbatim. A match blocks promotion.
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")


def _is_negated(text: str, start: int) -> bool:
    """A claim is not a hit when its clause is negated (no/not/never/without/does not
    promise ...). Ported verbatim from the source tool so parity is exact."""
    clause_start = max(text.rfind(mark, 0, start) for mark in ".!?;\n")
    prefix = text[clause_start + 1 : start].lower()
    if re.search(r"\b(?:but|however)\b", prefix):
        prefix = re.split(r"\b(?:but|however)\b", prefix)[-1]
    return bool(
        re.search(
            r"\b(no|not|never|without)\b|\bdo(?:es)?\s+not\s+promise\b|\bdoesn't\s+promise\b",
            prefix,
        )
    )


def _pattern_hits(text: str, rules: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(code, evidence) for each non-negated pattern match. Ported verbatim."""
    hits: list[tuple[str, str]] = []
    for code, pattern in rules:
        for match in re.finditer(pattern, text, re.I):
            if not _is_negated(text, match.start()):
                hits.append((code, match.group(0)))
    return hits


def verify_copy(text: str) -> CopyVerification:
    """Deterministically verify draft copy, producing the ``CopyVerification`` verdict
    the #2116 EditorialAudit contract requires for promotion.

    The verdict is ``"fail"`` when the copy contains any forbidden marketing claim
    (outcome / automation / replacing-agents) or raw contact PII (email / phone), and
    ``"pass"`` otherwise. Each hit is recorded as ``"code: evidence"``. A ``fail`` verdict
    makes ``recommendation == "promote"`` invalid, so a model cannot self-promote copy
    that overclaims or leaks PII."""
    if not isinstance(text, str):
        raise TypeError("verify_copy requires a string; draft body is text")

    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, evidence in _pattern_hits(text, _RULES[category]):
            hits.append(f"{code}: {evidence}")
    for match in _EMAIL_RE.finditer(text):
        hits.append(f"email: {match.group(0)}")
    for match in _PHONE_RE.finditer(text):
        hits.append(f"phone: {match.group(0)}")

    verdict = "fail" if hits else "pass"
    return CopyVerification(verdict=verdict, hits=hits)
