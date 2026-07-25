"""Deterministic copy-verification gate for Content Factory drafts (Phase 4.1)
plus the non-blocking advisory checklist (#2136 item 2).

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

EVIDENCE THEOREMS (review rounds 3-12). Two properties hold by construction, not by
pattern completeness:

  1. Gate claim hits never persist a digit: after the readable marker substitutions,
     EVERY remaining digit character is masked ("#"), so no phone/account-shaped value
     in any separator layout can survive into the git-backed audit.
  2. Advisory warnings never persist producer-supplied text AT ALL: each warning is a
     fixed claim code plus a 1-based sentence locator -- names, numbers, and draft
     fragments are unrepresentable, and the v2 schema enforces the same bounded
     grammar at the persistence choke point for every writer.

ADVISORY ENGINE (#2181 hardening). The soft checks run on a small deterministic
linguistic pass instead of positional regex windows:

  - token stream -> sentence spans (terminator + following capital, or blank line)
    -> clause spans (punctuation, coordinators, relativizers);
  - a negation SCOPE model: determiner negation (no/none/nothing before a noun)
    scopes at most two tokens right; verbal negation (not/never/cannot/-n't forms)
    scopes to its clause end; emphatic "not only/just" is affirmative;
  - qualifier GOVERNMENT by adjunct direction: a qualifier governs claims in its own
    clause, a qualifier-only clause governs the claims of the clause it FOLLOWS
    (postmodifier), and a sentence-initial qualifier clause governs the clause that
    follows it (fronted adjunct) -- never both directions at once;
  - routing coverage is bound to the report proposition: evidence counts only in the
    report-shape sentence itself or a later sentence whose subject is anaphoric
    (each/every/it/they/...), so an unrelated ownership statement elsewhere in the
    draft cannot satisfy the checklist.

The remaining precision boundary (deep subordinate-modifier attachment,
Markdown-decorated sentence starts) is declared in the plan and the PR reconciliation
ledger: closing it requires grammatical parsing beyond a deterministic backstop, and
the human approval step remains the authoritative gate.

The claim categories and PII patterns originate from the operator's "Resolution Audit
Draft Verifier" tool (authored by Juan Canfield / Codex), which lived only in the Open
WebUI database and was therefore un-versioned. This repo module is now the canonical gate
(the OWUI copy is superseded and should be re-synced from here).
"""

from __future__ import annotations

import bisect
import re

from atlas_brain.schemas.content_factory import (
    ADVISORY_CTA_REMINDER,
    ADVISORY_OWNER_ROUTING_WARNING,
    CopyVerification,
)

# ---------------------------------------------------------------------------
# Blocking gate: forbidden-claim catalogue (categories unchanged since 4.1)
# ---------------------------------------------------------------------------

_RULES: dict[str, list[tuple[str, str]]] = {
    "outcomes": [
        ("guaranteed-savings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$-]+){0,3}\s+savings\b"),
        ("guaranteed-rankings", r"\bguarante(?:e|es|ed)\b(?:\s+[\w%$-]+){0,3}\s+rankings\b"),
        ("fixed-deflection-percent", r"\b\d{1,3}\s*(?:%|percent)\s+deflection\b"),
        (
            "fixed-ticket-reduction",
            r"\b(?:cut|cuts|reduce|reduces|lower|lowers|drop|drops|shrink|shrinks)\s+(?:your\s+|the\s+)?(?:support\s+)?tickets?(?:\s+volume)?\s+by\s+\d{1,3}\s*(?:%|percent\b)",
        ),
        (
            "fixed-fewer-tickets",
            r"\b\d{1,3}\s*(?:%|percent)\s+(?:fewer|less)\s+(?:support\s+)?tickets?\b",
        ),
    ],
    "automation": [
        ("live-help-center-publishing", r"\blive\s+help[- ]center\s+publishing\b"),
        ("automatic-help-center-updates", r"\bautomatic\s+help[- ]center\s+updates\b"),
        ("automatic-ticket-answering", r"\bautomatic\s+ticket\s+answering\b"),
        ("auto-publish", r"\bauto[- ]publish(?:es|ing|ed)?\b"),
        (
            "automatically-updates-help-center",
            r"\bautomatically\s+(?:updates?|publish(?:es)?)\s+(?:your\s+)?help[- ]center\b",
        ),
        ("answers-tickets-automatically", r"\banswers?\s+tickets?\s+automatically\b"),
    ],
    "replacing_agents": [
        (
            "replace-agents",
            r"\breplac(?:e|es|ing)\s+(?:your\s+|our\s+|the\s+)?(?:support\s+)?agents?\b(?!['’])",
        ),
        (
            "avoid-support-hire",
            r"\bavoid(?:s|ing)?\s+(?:\w+\s+){0,3}(?:hire|hiring)\b",
        ),
    ],
}

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b")
_INTL_PHONE_RE = re.compile(r"\+\d{1,3}(?:[\s().-]?\d){7,12}\b")

# Gate-side negation for forbidden claims: the words immediately before the
# claim within its segment; "not only/just" is emphatic, not negation.
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|cannot)\b|n't\b", re.I)
_EMPHATIC_RE = re.compile(r"\bnot\s+(?:only|just)\b", re.I)

# THEOREM PASS (rounds 10-12): after the readable substitutions, every
# remaining digit character in persisted evidence is masked. There is no
# separator grammar left to enumerate -- a digit cannot survive.
_ANY_DIGIT_RE = re.compile(r"\d")  # Unicode decimal digits, every script


def _is_negated(text: str, start: int) -> bool:
    """Gate polarity: negation within the last two words of the current
    segment (segments end at ``.!?;,`` or newline)."""
    segment_start = max(text.rfind(mark, 0, start) for mark in ".!?;,\n")
    prefix = text[segment_start + 1 : start]
    window = " ".join(prefix.split()[-2:])
    if _EMPHATIC_RE.search(window):
        return False
    return bool(_NEGATION_RE.search(window))


def _redact_pii(evidence: str) -> str:
    """Make claim evidence safe to persist: named markers for the common
    contact shapes, then the digit theorem -- every remaining digit masks."""
    evidence = _EMAIL_RE.sub("<redacted-email>", evidence)
    evidence = _INTL_PHONE_RE.sub("<redacted-phone>", evidence)
    evidence = _PHONE_RE.sub("<redacted-phone>", evidence)
    return _ANY_DIGIT_RE.sub("#", evidence)


def _claim_hits(text: str) -> list[str]:
    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, pattern in _RULES[category]:
            for match in re.finditer(pattern, text, re.I):
                if not _is_negated(text, match.start()):
                    hits.append(f"{code}: {_redact_pii(match.group(0))}")
    return hits


def verify_copy(text: str) -> CopyVerification:
    """Deterministically verify draft copy, producing the ``CopyVerification``
    verdict the #2116 EditorialAudit contract requires for promotion.

    ``fail`` when the copy contains a forbidden marketing claim or raw contact
    PII; ``pass`` otherwise. Persisted hits obey the digit theorem: no digit
    character survives redaction. A ``pass`` means no known-bad pattern
    matched, not provably clean -- the human approval step remains the real
    gate (see module docstring)."""
    if not isinstance(text, str):
        raise TypeError("verify_copy requires a string; draft body is text")

    hits = _claim_hits(text)
    if _EMAIL_RE.search(text):
        hits.append("email: <redacted>")
    if _PHONE_RE.search(text):
        hits.append("phone: <redacted>")

    verdict = "fail" if hits else "pass"
    return CopyVerification(verdict=verdict, hits=hits)


# ---------------------------------------------------------------------------
# Advisory engine: token stream, sentence/clause structure, scope model
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][\w'-]*|\d[\w.-]*")

# Sentence terminators: runs of .!? followed by whitespace + a capital/quote,
# or end of text; a blank line is a structural break. Single newlines (soft
# wraps) never split; digit-internal and abbreviation periods never split.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?]+\s+(?=[A-Z\"'(])|[.!?]+\s*\Z|\n\s*\n+"
)

# Clause boundaries: the minimal proposition. Punctuation (incl. dashes,
# slashes, parens), coordinators, and relativizer/adjunct openers.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;,:\n()/—–]|\s-\s|"
    r"\b(?:and|or|but|however|while|whereas|although|yet|that|which|who|when|if|unless)\b",
    re.I,
)

# Negation vocabulary split by grammatical role (round 12):
#   determiner negation scopes at most two tokens right ("no delay");
#   verbal negation scopes to its clause end ("do not draft ...").
_DET_NEGATION = frozenset({"no", "none", "nothing", "neither"})
_VERBAL_NEGATION = frozenset(
    {"not", "never", "cannot", "nobody", "without", "don", "doesn", "didn",
     "won", "isn", "aren"}
)

_ANSWER_CLAIM_RE = re.compile(
    r"\b(?:drafted\s+answers?|answers?|resolutions?(?!\s+(?:audit|snapshot)))\b",
    re.I,
)
_ANSWER_QUALIFIER_RE = re.compile(
    r"\b(agent resolution|scoped resolution|when (?:that )?evidence exists|if (?:the )?tickets contain|no proven answer)\b",
    re.I,
)
_OWNERSHIP_CLAIM_RE = re.compile(
    r"\b(?:engineering|product|support|cx|policy|ops|operations|billing|success|content|docs|documentation|legal|team|owner)s?\s+(?:\w+\s+){0,2}(?:owns?|is\s+responsible\s+for|are\s+responsible\s+for|should\s+own|must\s+own)\b|\bowned\s+by\b",
    re.I,
)
_OWNERSHIP_QUALIFIER_RE = re.compile(
    r"\b(probable|probably|may|might|could|likely|often|appears|seems|investigate|route|routing|signal)\b",
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

# Report-shape: a report noun and a shape verb in the SAME clause with at most
# three tokens between them ("The report clearly ranks issues"), or a product
# term; polarity applies via the scope model.
_REPORT_NOUNS = frozenset({"report", "reports", "audit", "audits", "snapshot", "snapshots"})
_SHAPE_VERBS = frozenset(
    {"ranks", "rank", "lists", "list", "shows", "show", "includes", "include",
     "names", "name", "delivers", "deliver", "contains", "contain",
     "identifies", "identify", "surfaces", "surface", "highlights", "highlight"}
)
_PRODUCT_TERM_RE = re.compile(
    r"\b(?:resolution\s+audit|resolution\s+snapshot|action\s+queue)\b", re.I
)

# Absence lexemes for routing labels/predicates ("Owner lane: TBD").
_ABSENCE_LEXEMES = frozenset(
    {"unknown", "unassigned", "unresolved", "undecided", "missing", "absent",
     "tbd", "pending", "none", "nobody"}
)
_COPULAR_ABSENCE_RE = re.compile(
    r"\s+(?:is|are|was|were|remains?|stays?)\s+(?:\w+\s+)?"
    r"(?:unknown|unassigned|unresolved|undecided|missing|absent|tbd)\b",
    re.I,
)

# Anaphoric subjects that bind a later sentence back to the report
# proposition ("Each is assigned to ...").
_ANAPHORIC_SUBJECTS = frozenset(
    {"each", "every", "all", "it", "they", "these", "those", "everything", "both"}
)
_REPORT_ITEM_NOUNS = frozenset(
    {"issue", "issues", "ticket", "tickets", "fix", "fixes", "item", "items",
     "question", "questions", "finding", "findings"}
)

_CTA_REMINDER = ADVISORY_CTA_REMINDER


def _boundary_spans(
    text: str, boundary_re: "re.Pattern[str]"
) -> "tuple[list[int], list[tuple[int, int]]]":
    """(starts, spans) between boundary matches, computed once per draft so
    per-position lookups are a bisect over the cached starts."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in boundary_re.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [s for s, _e in spans], spans


def _clause_boundary_kinds(text: str) -> "list[str]":
    """kinds[i] describes the boundary between clause i and clause i+1:
    "word" for adjunct/coordinator openers (when/if/that/and/...), "punct"
    for punctuation (comma, dash, paren, slash, colon...). Postmodifying
    qualifier government may only cross word boundaries -- a dash or comma
    starts a NEW proposition, so a trailing qualifier there does not attach
    to the preceding claim."""
    return [
        "word" if match.group(0).strip().isalpha() else "punct"
        for match in _CLAUSE_BOUNDARY_RE.finditer(text)
    ]


def _span_for(
    bounds: "tuple[list[int], list[tuple[int, int]]]", position: int
) -> "tuple[int, tuple[int, int]]":
    """(index, span) of the precomputed span containing ``position``."""
    starts, spans = bounds
    index = max(0, bisect.bisect_right(starts, position) - 1)
    return index, spans[index]


def _clause_tokens(text: str, span: "tuple[int, int]") -> "list[re.Match]":
    return list(_WORD_RE.finditer(text, span[0], span[1]))


def _negation_scopes(text: str, span: "tuple[int, int]") -> "list[tuple[int, int]]":
    """Negation scopes inside one clause as (start, end) character ranges.

    Determiner negation covers itself plus at most two following tokens;
    verbal negation covers from the trigger to the clause end; emphatic
    "not only/just" produces no scope.
    """
    tokens = _clause_tokens(text, span)
    scopes: list[tuple[int, int]] = []
    for position, token in enumerate(tokens):
        lower = token.group(0).lower()
        if lower in _DET_NEGATION:
            end_token = tokens[min(position + 2, len(tokens) - 1)]
            scopes.append((token.start(), end_token.end()))
        elif lower in _VERBAL_NEGATION or lower.endswith("n't"):
            if (
                lower == "not"
                and position + 1 < len(tokens)
                and tokens[position + 1].group(0).lower() in ("only", "just")
            ):
                continue
            scopes.append((token.start(), span[1]))
    return scopes


def _position_negated(
    text: str,
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    position: int,
    cache: "dict[int, list[tuple[int, int]]]",
) -> bool:
    index, span = _span_for(clause_bounds, position)
    if index not in cache:
        cache[index] = _negation_scopes(text, span)
    return any(start <= position < end for start, end in cache[index])


def _range_negated(
    text: str,
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    start: int,
    end: int,
    cache: "dict[int, list[tuple[int, int]]]",
) -> bool:
    """True when any negation scope INTERSECTS [start, end): denial words can
    sit inside a multiword match ("Billing never owns")."""
    index, span = _span_for(clause_bounds, start)
    if index not in cache:
        cache[index] = _negation_scopes(text, span)
    return any(s < end and start < e for s, e in cache[index])


def _sentence_of(
    sentence_bounds: "tuple[list[int], list[tuple[int, int]]]", position: int
) -> int:
    return _span_for(sentence_bounds, position)[0]


def _unqualified_claims(
    text: str,
    word_re: "re.Pattern[str]",
    qualifier_re: "re.Pattern[str]",
    code: str,
    sentence_bounds: "tuple[list[int], list[tuple[int, int]]]",
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    boundary_kinds: "list[str]",
    negation_cache: "dict[int, list[tuple[int, int]]]",
) -> list[str]:
    """One locator-only warning per (code, sentence) for each claim that is
    neither negated (scope model) nor governed by a qualifier.

    Government by adjunct direction: a qualifier governs claims in its OWN
    clause; a qualifier-bearing, claim-free clause governs the claims of the
    clause it FOLLOWS (postmodifying adjunct); a sentence-initial
    qualifier-bearing, claim-free clause governs the clause AFTER it (fronted
    adjunct). Never both directions at once, so one qualified assertion
    cannot excuse a separate unqualified one elsewhere.

    Warnings carry a fixed code and a sentence number ONLY -- no matched
    text, so producer-supplied names and numbers are unrepresentable
    (round 12).
    """
    claim_clauses: dict[int, list[re.Match]] = {}
    for match in word_re.finditer(text):
        if _range_negated(
            text, clause_bounds, match.start(), match.end(), negation_cache
        ):
            continue
        index, _span = _span_for(clause_bounds, match.start())
        claim_clauses.setdefault(index, []).append(match)
    if not claim_clauses:
        return []

    # A qualifier phrase's clause is taken at its END: adjunct openers like
    # "when" are clause boundaries themselves, so the phrase's content sits
    # in the clause after the opener.
    qualifier_clauses: set[int] = set()
    for qualifier in qualifier_re.finditer(text):
        index, _span = _span_for(clause_bounds, max(qualifier.end() - 1, 0))
        qualifier_clauses.add(index)

    _starts, clause_spans = clause_bounds
    # Adjacency skips token-free clauses (boundary runs like "when that"
    # create empty spans between openers).
    content = [
        bool(_clause_tokens(text, span)) for span in clause_spans
    ]

    def _next_content(index: int) -> "int | None":
        for candidate in range(index + 1, len(clause_spans)):
            if content[candidate]:
                return candidate
        return None

    def _prev_content(index: int) -> "int | None":
        for candidate in range(index - 1, -1, -1):
            if content[candidate]:
                return candidate
        return None

    warnings: list[str] = []
    seen_sentences: set[int] = set()
    for index in sorted(claim_clauses):
        claim_sentence = _sentence_of(
            sentence_bounds, claim_clauses[index][0].start()
        )
        governed = index in qualifier_clauses
        if not governed:
            following = _next_content(index)
            governed = (
                following is not None
                and following in qualifier_clauses
                and following not in claim_clauses
                and _sentence_of(sentence_bounds, clause_spans[following][0])
                == claim_sentence
                # Postmodifier attachment is direct: every boundary crossed
                # must be a word opener (when/if/that/...), never punctuation.
                and all(
                    boundary_kinds[k] == "word"
                    for k in range(index, following)
                    if k < len(boundary_kinds)
                )
            )
        if not governed:
            # Fronted adjunct: the previous content clause is qualifier-
            # bearing, claim-free, in the same sentence, and is that
            # sentence's FIRST content clause.
            previous = _prev_content(index)
            if (
                previous is not None
                and previous in qualifier_clauses
                and previous not in claim_clauses
                and _sentence_of(sentence_bounds, clause_spans[previous][0])
                == claim_sentence
            ):
                before = _prev_content(previous)
                governed = before is None or (
                    _sentence_of(sentence_bounds, clause_spans[before][0])
                    != claim_sentence
                )
        if governed:
            continue
        for claim in claim_clauses[index]:
            sentence_index = _sentence_of(sentence_bounds, claim.start())
            if sentence_index in seen_sentences:
                continue
            seen_sentences.add(sentence_index)
            warnings.append(f"{code}: sentence {sentence_index + 1}")
    return warnings


def _routing_relation_affirmative(
    text: str,
    match: "re.Match",
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    negation_cache: "dict[int, list[tuple[int, int]]]",
) -> bool:
    """A routing relation counts only when nothing negates it: no negation
    scope covers it, no copular absence follows it ("the owner lane is
    unknown"), and no label-style absence trails its clause
    ("Owner lane: TBD" / "Owner lane -- unassigned")."""
    if _range_negated(
        text, clause_bounds, match.start(), match.end(), negation_cache
    ):
        return False
    if _COPULAR_ABSENCE_RE.match(text, match.end()):
        return False
    index, span = _span_for(clause_bounds, match.start())
    if not text[match.end() : span[1]].strip():
        _starts, clause_spans = clause_bounds
        following = index + 1
        if following < len(clause_spans):
            next_tokens = _clause_tokens(text, clause_spans[following])
            if next_tokens and next_tokens[0].group(0).lower() in _ABSENCE_LEXEMES:
                return False
    return True


def _report_shape_sentences(
    text: str,
    sentence_bounds: "tuple[list[int], list[tuple[int, int]]]",
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    negation_cache: "dict[int, list[tuple[int, int]]]",
) -> "set[int]":
    """Sentence indexes carrying an AFFIRMATIVE report-shape assertion: a
    product term, or a report noun with a shape verb at most three tokens
    later in the same clause, outside any negation scope."""
    sentences: set[int] = set()
    for match in _PRODUCT_TERM_RE.finditer(text):
        if not _position_negated(text, clause_bounds, match.start(), negation_cache):
            sentences.add(_sentence_of(sentence_bounds, match.start()))
    _starts, clause_spans = clause_bounds
    for span in clause_spans:
        tokens = _clause_tokens(text, span)
        for position, token in enumerate(tokens):
            if token.group(0).lower() not in _REPORT_NOUNS:
                continue
            window = tokens[position + 1 : position + 5]
            shape_verb = next(
                (w for w in window if w.group(0).lower() in _SHAPE_VERBS), None
            )
            if shape_verb is None:
                continue
            # Polarity across the whole noun-to-verb relation: "The report
            # does not rank issues" is a denial, not a shape assertion.
            if _range_negated(
                text, clause_bounds, token.start(), shape_verb.end(),
                negation_cache,
            ):
                continue
            sentences.add(_sentence_of(sentence_bounds, token.start()))
            break
    return sentences


def _routing_covers_report(
    text: str,
    report_sentences: "set[int]",
    sentence_bounds: "tuple[list[int], list[tuple[int, int]]]",
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    negation_cache: "dict[int, list[tuple[int, int]]]",
) -> bool:
    """Owner routing satisfies the checklist only when bound to the report
    proposition: in a report-shape sentence itself, or in a LATER sentence
    whose clause subject is anaphoric (each/every/it/they/...) and so refers
    back to the report's items. An unrelated ownership statement about a
    different object elsewhere in the draft does not count (round 12)."""
    subject_cache: dict[int, bool] = {}
    for match in _OWNER_ROUTING_RE.finditer(text):
        if not _routing_relation_affirmative(
            text, match, clause_bounds, negation_cache
        ):
            continue
        sentence = _sentence_of(sentence_bounds, match.start())
        if sentence in report_sentences:
            return True
        if any(sentence > report for report in report_sentences):
            index, span = _span_for(clause_bounds, match.start())
            if index not in subject_cache:
                tokens = _clause_tokens(text, span)
                first = tokens[0].group(0).lower() if tokens else ""
                second = tokens[1].group(0).lower() if len(tokens) > 1 else ""
                # Subject position only (round 13): the clause must be ABOUT
                # the report's items -- an anaphoric subject ("Each is
                # assigned...") or a determiner + report-item noun ("These
                # issues are routed..."). An anaphoric token buried in an
                # unrelated object ("owns invoices for each customer") does
                # not bind.
                subject_cache[index] = first in _ANAPHORIC_SUBJECTS or (
                    first in ("the", "these", "those")
                    and second in _REPORT_ITEM_NOUNS
                )
            if subject_cache[index]:
                return True
    return False


def advisory_warnings(text: str) -> list[str]:
    """Deterministic non-blocking reviewer checklist for draft copy (#2136
    item 2), computed by the advisory engine (see module docstring).

    Warnings NEVER change the ``verify_copy`` verdict and never block
    promotion. Each warning is a fixed code plus a sentence locator (or a
    canonical static line); producer-supplied text is unrepresentable, and
    the v2 schema enforces the same bounded grammar at the persistence choke
    point. The CTA reminder is unconditional by design: every audit carries
    at least that one checklist line.
    """
    if not isinstance(text, str):
        raise TypeError("advisory_warnings requires a string; draft body is text")

    sentence_bounds = _boundary_spans(text, _SENTENCE_BOUNDARY_RE)
    clause_bounds = _boundary_spans(text, _CLAUSE_BOUNDARY_RE)
    negation_cache: dict[int, list[tuple[int, int]]] = {}

    boundary_kinds = _clause_boundary_kinds(text)
    warnings = _unqualified_claims(
        text, _ANSWER_CLAIM_RE, _ANSWER_QUALIFIER_RE, "unqualified-answer-claim",
        sentence_bounds, clause_bounds, boundary_kinds, negation_cache,
    )
    warnings += _unqualified_claims(
        text, _OWNERSHIP_CLAIM_RE, _OWNERSHIP_QUALIFIER_RE,
        "unqualified-ownership-claim", sentence_bounds, clause_bounds,
        boundary_kinds, negation_cache,
    )
    report_sentences = _report_shape_sentences(
        text, sentence_bounds, clause_bounds, negation_cache
    )
    if report_sentences and not _routing_covers_report(
        text, report_sentences, sentence_bounds, clause_bounds, negation_cache
    ):
        warnings.append(ADVISORY_OWNER_ROUTING_WARNING)
    warnings.append(_CTA_REMINDER)
    return warnings
