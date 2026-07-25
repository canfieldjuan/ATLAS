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
contact PII (email / phone).

ADVISORY LAYER (#2136 item 2). ``advisory_warnings`` ports the source tool's softer
"needs human review" checks: owner-routing coverage, unqualified answer/ownership
claims, and the honest-CTA reminder. Warnings NEVER affect the verdict or promotion --
they are a reviewer checklist persisted on the audit artifact. Evidence sentences are
PII-redacted before recording, like the gate's claim hits.
"""

from __future__ import annotations

import bisect
import re

from atlas_brain.schemas.content_factory import (
    ADVISORY_CTA_REMINDER,
    ADVISORY_OWNER_ROUTING_WARNING,
    CopyVerification,
)

# Forbidden-claim catalogue. Each pattern targets a promote-blocking category and matches
# the common inflections/modifiers of that category, not a single fixture wording. Still a
# best-effort backstop (see module docstring): a novel paraphrase can pass.
_RULES: dict[str, list[tuple[str, str]]] = {
    "outcomes": [
        # "guaranteed savings", "guaranteed cost/monthly savings", "guaranteed 30%
        # savings", "guarantees savings". Up to 3 tokens (incl. numeric/percent ones)
        # may sit between the guarantee verb and "savings".
        # Gap tokens allow digits/percent (e.g. "30%") but NOT sentence punctuation, so a
        # modifier gap cannot bridge a sentence boundary ("guarantee X. Savings ...").
        ("guaranteed-savings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$-]+){0,3}\s+savings\b"),
        ("guaranteed-rankings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$-]+){0,3}\s+rankings\b"),
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
# International formats: a +country-code prefix followed by 7-12 digits with
# optional separators ("+44 20 7946 0958"). REDACTION-ONLY: the gate's
# blocking scope is a separate operator decision (this slice's contract
# freezes verdict semantics); over-redacting evidence is harmless.
_INTL_PHONE_RE = re.compile(r"\+\d{1,3}(?:[\s().-]?\d){7,12}\b")
# Phone-SHAPED local formats without a country prefix ("020 7946 0958"):
# 9-13 digits in separator-joined groups. Also redaction-only -- a false
# positive costs a `<redacted-phone>` marker in advisory evidence, never a
# verdict change.
_PHONE_SHAPED_RE = re.compile(r"\b\d(?:[\s().-]?\d){8,12}\b")
# Class backstop for evidence (review round 3): ANY run of 5+ digits joined by
# at most one non-word separator each ("020/7946/0958", "10,000.55") is masked
# before persisting. No enumeration of separator styles can be complete, so
# the evidence path masks the whole class; over-redaction is harmless there.
_DIGIT_RUN_RE = re.compile(r"\d(?:\W{0,4}\d){4,}")

# A negation directly before the claim (no/not/never/without/cannot, or an -n't
# contraction). "not only"/"not just" are emphatic, NOT negations, and are excluded.
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|cannot)\b|n't\b", re.I)
_EMPHATIC_RE = re.compile(r"\bnot\s+(?:only|just)\b", re.I)


# --- Advisory-layer patterns (ported from the operator's OWUI verifier tool) ---
_ANSWER_RE = re.compile(
    r"\b(?:drafted\s+answers?|answers?|resolutions?(?!\s+(?:audit|snapshot)))\b",
    re.I,
)
_ANSWER_QUALIFIER_RE = re.compile(
    r"\b(agent resolution|scoped resolution|when (?:that )?evidence exists|if (?:the )?tickets contain|no proven answer)\b",
    re.I,
)
_REPORT_SHAPE_RE = re.compile(
    r"\b(?:resolution\s+audit|resolution\s+snapshot|action\s+queue)\b|"
    r"\b(?:snapshot|report|audit)s?\s+(?:that\s+)?"
    r"(?:ranks?|lists?|shows?|includes?|names?|delivers?|contains?|identifies|surfaces?|highlights?)\b",
    re.I,
)
_OWNER_SUBJECTS = (
    "engineering|product|support|cx|policy|ops|operations|billing|success|"
    "content|docs|documentation|legal|team|teams|owner|owners|department|"
    "departments|group|groups|lane|lanes"
)
_OWNER_TARGETS = (
    _OWNER_SUBJECTS + "|reviewer|reviewers|person|people|staff|manager|"
    "managers|lead|leads|nobody"
)
_OWNER_ROUTING_RE = re.compile(
    r"\b(?:owner\s+lane|"
    r"owned\s+by\s+(?:the\s+|an?\s+)?(?:owning\s+)?(?:" + _OWNER_TARGETS + r")\b|"
    r"assigned\s+to\s+(?:the\s+|an?\s+)?(?:owning\s+)?(?:" + _OWNER_TARGETS + r")\b|"
    r"route[sd]?\s+(?:each\s+\w+\s+)?to\s+(?:the\s+|an?\s+)?(?:owning\s+)?(?:" + _OWNER_TARGETS + r")\b|"
    r"(?:" + _OWNER_SUBJECTS + r")\s+(?:\w+\s+)?owns?\b|"
    r"who\s+needs\s+to\s+(?:fix|review)|needs\s+to\s+(?:fix|review))",
    re.I,
)
# Copular absence directly after a routing relation: "the owner lane is
# unknown", "... remains unassigned". Anchored at the match end so a
# separate trailing modifier is not read as absence.
_ROUTING_SUFFIX_ABSENCE_RE = re.compile(
    r"\s+(?:is|are|was|were|remains?|stays?)\s+(?:\w+\s+)?"
    r"(?:unknown|unassigned|unresolved|undecided|missing|absent|tbd)\b",
    re.I,
)
_ROUTING_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|nobody|nothing|neither|without|cannot|isn|aren|"
    r"unresolved|unknown|unassigned|undecided)\b|n't\b",
    re.I,
)


def _has_affirmative_owner_routing(
    text: str, clause_bounds: "tuple[list[int], list[tuple[int, int]]]"
) -> bool:
    """True only for a routing/ownership relation whose COMPLETE clause is
    free of negation/absence language: 'no one is assigned to them',
    'assigned to nobody', 'not routed to Billing', and 'routing remains
    unresolved' all fail to count as coverage (review rounds 3-4)."""
    for match in _OWNER_ROUTING_RE.finditer(text):
        _index, (start, _end) = _span_for(clause_bounds, match.start())
        # Negation is bound to the RELATION: the clause-bounded words just
        # before the match, the match itself, and a copular absence
        # predicate immediately after it ("the owner lane is unknown").
        # An unrelated trailing modifier ("assigned to Billing with no due
        # date") does not invalidate the routing (review rounds 10-11).
        prefix = text[start : match.start()]
        window = " ".join(prefix.split()[-4:]) + " " + text[match.start() : match.end()]
        if _ROUTING_NEGATION_RE.search(window):
            continue
        if _ROUTING_SUFFIX_ABSENCE_RE.match(text, match.end()):
            continue
        return True
    return False
_OWNERSHIP_RE = re.compile(
    r"\b(?:engineering|product|support|cx|policy|ops|operations|billing|success|content|docs|documentation|legal|team|owner)s?\s+(?:\w+\s+){0,2}(?:owns?|is\s+responsible\s+for|are\s+responsible\s+for|should\s+own|must\s+own)\b|\bowned\s+by\b",
    re.I,
)
_OWNERSHIP_QUALIFIER_RE = re.compile(
    r"\b(probable|probably|may|might|could|likely|often|appears|seems|investigate|route|routing|signal)\b",
    re.I,
)

_CTA_REMINDER = ADVISORY_CTA_REMINDER


def _is_negated(text: str, start: int) -> bool:
    """True when the claim at ``start`` is negated by the words IMMEDIATELY before it.

    Only the last two words of the current segment are considered (a segment ends at
    ``.!?;,`` or newline), so a real negation ("no guaranteed savings", "does not promise
    guaranteed savings", "we cannot guarantee savings") suppresses the hit, but an
    UNRELATED earlier negation in the same sentence ("No setup fee, and guaranteed savings
    ...") does not, and emphatic "not only"/"not just" is not read as a negation ("not only
    guarantees savings" is still a hit)."""
    segment_start = max(text.rfind(mark, 0, start) for mark in ".!?;,\n")
    prefix = text[segment_start + 1 : start]
    window = " ".join(prefix.split()[-2:])
    if _EMPHATIC_RE.search(window):
        return False
    return bool(_NEGATION_RE.search(window))


def _redact_pii(evidence: str) -> str:
    """Strip any email/phone out of matched claim evidence before it is recorded, so a
    claim that happens to span a contact string ("Guaranteed 618-555-9876 savings") does
    not persist raw PII into the git-backed audit metadata via the claim hit."""
    evidence = _EMAIL_RE.sub("<redacted-email>", evidence)
    evidence = _INTL_PHONE_RE.sub("<redacted-phone>", evidence)
    evidence = _PHONE_RE.sub("<redacted-phone>", evidence)
    evidence = _PHONE_SHAPED_RE.sub("<redacted-phone>", evidence)
    evidence = _DIGIT_RUN_RE.sub("<redacted-number>", evidence)
    return evidence


def _claim_hits(text: str) -> list[str]:
    """"code: evidence" for each non-negated forbidden-claim match. Claim evidence is the
    marketing phrase itself (useful for a human reviewer), with any PII redacted out."""
    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, pattern in _RULES[category]:
            for match in re.finditer(pattern, text, re.I):
                if not _is_negated(text, match.start()):
                    hits.append(f"{code}: {_redact_pii(match.group(0))}")
    return hits


# Sentence terminators: runs of .!? collapse to ONE boundary ("Really?!"),
# and a terminator only counts when followed by the start of a new sentence
# (whitespace + capital/quote) or end of text -- so "Version 2.1",
# "bob@example.com about", and "E.g. we draft" do not inflate the locator
# (review rounds 5-6). Erring toward FEWER splits keeps locators <= the
# human count.
# A bare newline is a boundary only when it structurally starts a new
# sentence/block: a blank line, or a line beginning with a capital/quote.
# Markdown soft wraps ("long\nwrapped answer") stay in their sentence
# (review round 8).
# A single newline is NEVER a sentence boundary (soft wraps can precede
# proper nouns too -- capitalization is not the signal); only a blank line
# is a structural block break (review round 9).
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?]+\s+(?=[A-Z\"'(])|[.!?]+\s*\Z|\n\s*\n+"
)


def _boundary_spans(
    text: str, boundary_re: "re.Pattern[str]"
) -> "tuple[list[int], list[tuple[int, int]]]":
    """(starts, spans) between boundary matches -- BOTH computed once per
    document, so per-match lookups are a bisect over the cached starts
    (review rounds 4-5: no per-lookup list rebuild)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in boundary_re.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [s for s, _ in spans], spans


def _span_for(
    bounds: "tuple[list[int], list[tuple[int, int]]]", position: int
) -> "tuple[int, tuple[int, int]]":
    """(index, span) of the precomputed span containing ``position``."""
    starts, spans = bounds
    index = max(0, bisect.bisect_right(starts, position) - 1)
    return index, spans[index]


# Clause = the minimal punctuation-delimited proposition. Dashes, slashes,
# and parentheses are boundaries too (review round 7), so a qualifier can
# only excuse a claim it shares the proposition with; the per-clause COUNT
# stays as the fail-closed backstop for separators not listed here.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;,:\n()/\u2014\u2013]|\s-\s|"
    r"\b(?:and|or|but|however|while|whereas|although|yet)\b",
    re.I,
)


_CLAIM_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|nobody|without|cannot|don|doesn|didn|won|isn|aren)\b|n't\b",
    re.I,
)


_KEYWORD_DISALLOWED_RE = re.compile(r"[^A-Za-z /-]")


def _grammar_safe_keyword(matched: str) -> str:
    """Reduce a matched claim keyword to the schema's bounded alphabet.

    Whitespace (including newlines from wrapped drafts) collapses to single
    spaces and anything outside the grammar's charset is dropped, so the
    producer can never emit a locator its own contract rejects (review
    round 6: a wrapped "Billing\nreally owns" must not invalidate an
    otherwise valid audit)."""
    collapsed = " ".join(matched.split())
    safe = _KEYWORD_DISALLOWED_RE.sub("", collapsed).strip()[:59]
    return safe if safe and safe[0].isalpha() else "claim"


def _unqualified_claims(
    text: str,
    word_re: "re.Pattern[str]",
    qualifier_re: "re.Pattern[str]",
    code: str,
    sentence_bounds: "tuple[list[int], list[tuple[int, int]]]",
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
) -> list[str]:
    """One warning per (code, sentence) for each claim not excused within its
    OWN CLAUSE. FAIL-CLOSED ASSOCIATION, clause-scoped (review rounds 5-6):
    a qualifier only excuses claims in its clause, and excuses at most as
    many claims as there are qualifiers there -- so neither an unknown
    separator (em dash, slash, parens: no clause boundary, but counting
    still catches the extra claim) nor an explicit boundary (comma/but: the
    unrelated qualifier sits in another clause) can hide an unqualified
    claim. Counting is linear; no pairwise scan (round-6 perf).

    Warnings carry NO free text -- only the claim code, the 1-based sentence
    number, and the grammar-safe matched keyword; the schema enforces the
    same grammar at the persistence choke point. The reviewing human locates
    the numbered sentence in the draft artifact beside the audit.
    """
    claims_by_clause: dict[int, list[re.Match]] = {}
    for match in word_re.finditer(text):
        index, (clause_start, _end) = _span_for(clause_bounds, match.start())
        # Polarity (review rounds 8-9): an explicitly negated match is a
        # DENIAL, not an unqualified assertion -- but emphatic "not only"/
        # "not just" is affirmative (same distinction the blocking gate
        # draws), so "We not only draft answers" still registers.
        # Bound to the assertion: the three words immediately before the
        # match plus the match span -- an earlier unrelated negative noun
        # phrase ("With no delay, we draft answers") is not a denial
        # (review round 11).
        prefix_tail = " ".join(text[clause_start : match.start()].split()[-3:])
        polarity_span = prefix_tail + " " + text[match.start() : match.end()]
        if _CLAIM_NEGATION_RE.search(polarity_span) and not _EMPHATIC_RE.search(
            polarity_span
        ):
            continue
        claims_by_clause.setdefault(index, []).append(match)
    if not claims_by_clause:
        return []
    qualifier_counts: dict[int, int] = {}
    qualifier_first_pos: dict[int, int] = {}
    for qualifier in qualifier_re.finditer(text):
        index, _span = _span_for(clause_bounds, qualifier.start())
        qualifier_counts[index] = qualifier_counts.get(index, 0) + 1
        qualifier_first_pos.setdefault(index, qualifier.start())

    warnings: list[str] = []
    seen_sentences: set[int] = set()
    for index in sorted(claims_by_clause):
        claims = claims_by_clause[index]
        # Fronted qualifiers (review rounds 8-9): a qualifier whose own
        # clause holds NO claim ("When evidence exists, ...") carries
        # forward to the IMMEDIATELY following clause, and only WITHIN the
        # same sentence -- a qualifier from a completed sentence can never
        # excuse the next sentence's claim.
        available = qualifier_counts.get(index, 0)
        prev = index - 1
        if (
            prev in qualifier_first_pos
            and prev not in claims_by_clause
            # Same-sentence bound, compared at the QUALIFIER's and CLAIM's
            # actual positions (clause starts sit in boundary whitespace and
            # can map to the wrong sentence).
            and _span_for(sentence_bounds, qualifier_first_pos[prev])[0]
            == _span_for(sentence_bounds, claims[0].start())[0]
        ):
            available += qualifier_counts[prev]
        for claim in claims[available:]:
            sentence_index, _span = _span_for(sentence_bounds, claim.start())
            if sentence_index in seen_sentences:
                continue
            seen_sentences.add(sentence_index)
            warnings.append(
                f"{code}: sentence {sentence_index + 1} "
                f"({_grammar_safe_keyword(claim.group(0))!r})"
            )
    return warnings


def advisory_warnings(text: str) -> list[str]:
    """Deterministic non-blocking reviewer checklist for draft copy (#2136
    item 2, ported from the operator's OWUI verifier tool).

    Produces "needs human review" warnings -- unqualified answer/resolution
    claims, unqualified ownership assertions, report-shape copy that omits
    owner routing, and the standing honest-CTA reminder. Warnings NEVER
    change the ``verify_copy`` verdict and never block promotion; they are
    persisted on the audit artifact for the approving human. The CTA
    reminder is unconditional by design, mirroring the source tool: every
    audit carries at least that one checklist line.
    """
    if not isinstance(text, str):
        raise TypeError("advisory_warnings requires a string; draft body is text")

    # Boundary spans are computed once per draft (review round 4); warnings
    # persist only code + sentence number + matched keyword, so no free-text
    # evidence (and therefore nothing PII-shaped) ever reaches the artifact.
    sentence_bounds = _boundary_spans(text, _SENTENCE_BOUNDARY_RE)
    clause_bounds = _boundary_spans(text, _CLAUSE_BOUNDARY_RE)

    warnings = _unqualified_claims(
        text, _ANSWER_RE, _ANSWER_QUALIFIER_RE, "unqualified-answer-claim",
        sentence_bounds, clause_bounds,
    )
    warnings += _unqualified_claims(
        text, _OWNERSHIP_RE, _OWNERSHIP_QUALIFIER_RE,
        "unqualified-ownership-claim", sentence_bounds, clause_bounds,
    )
    if _REPORT_SHAPE_RE.search(text) and not _has_affirmative_owner_routing(
        text, clause_bounds
    ):
        warnings.append(ADVISORY_OWNER_ROUTING_WARNING)
    warnings.append(_CTA_REMINDER)
    return warnings


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
