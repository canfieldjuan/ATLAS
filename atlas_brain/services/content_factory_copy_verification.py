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
import unicodedata

from atlas_brain.schemas.content_factory import (
    ABBREVIATIONS,
    ADVISORY_CTA_REMINDER,
    ADVISORY_OWNER_ROUTING_WARNING,
    LAST_WORD_RE,
    SENTENCE_BOUNDARY_RE,
    SENTENCE_STARTERS,
    CopyVerification,
    is_default_ignorable,
    sentence_spans,
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

_LAST_WORD_RE = LAST_WORD_RE
_ABBREVIATIONS = ABBREVIATIONS
_SENTENCE_STARTERS = SENTENCE_STARTERS
_SENTENCE_BOUNDARY_RE = SENTENCE_BOUNDARY_RE

_SCAN_KEEP_CONTROLS = frozenset("\t\n\r\v\f")


def scan_view(text: str) -> str:
    """The view PII detection runs against: same visible content, no
    zero-width characters.

    A zero-width character between digits defeats every contact pattern while
    rendering identically -- `555-123<ZWSP>-4567` and `a@b<ZWSP>.com` both
    passed the gate. That is a formatting-only bypass, not a different kind
    of copy, so it is removed before scanning rather than enumerated as new
    patterns (#2201).

    Admission uses the SHARED `is_default_ignorable` predicate, not a local
    category test. A hand-built set here missed U+034F, which is category Mn
    -- so any rule phrased as "Cf plus some ranges" leaves the bypass open by
    construction. One definition, used by routing keys, this scan, and
    evidence redaction alike.

    This does NOT change which real forms count as PII -- the frozen
    body-copy verdict semantics are untouched -- it only denies the evasion.
    Whitespace controls are kept: they separate tokens, and dropping them
    would join a word to a following number.
    """
    return "".join(
        ch
        for ch in text
        if not (
            is_default_ignorable(ch)
            or unicodedata.category(ch) == "Cf"
            or (unicodedata.category(ch) == "Cc" and ch not in _SCAN_KEEP_CONTROLS)
        )
    )


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
def _mask_digit_chars(evidence: str) -> str:
    """Category-complete digit choke point: any character Python classifies
    as a digit OR numeric (decimal, circled, superscript, vulgar fractions,
    every script) masks. The predicate IS the claim -- no regex class to
    fall behind the word-character class the claim patterns admit."""
    return "".join(
        "#" if (ch.isdigit() or ch.isnumeric()) else ch for ch in evidence
    )


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
    contact shapes, then the digit theorem -- every remaining digit masks.

    Zero-width characters are stripped FIRST (#2201). The digit theorem
    already covered phone digits, but an address carrying a zero-width
    character (`a@b<ZWSP>.com`) evaded the email pattern and its LETTERS are
    not digits, so it would have persisted intact. Dropping invisible
    characters from persisted evidence changes nothing a reader sees.
    """
    evidence = scan_view(evidence)
    evidence = _EMAIL_RE.sub("<redacted-email>", evidence)
    evidence = _INTL_PHONE_RE.sub("<redacted-phone>", evidence)
    evidence = _PHONE_RE.sub("<redacted-phone>", evidence)
    return _mask_digit_chars(evidence)


def _claim_hits(text: str) -> list[str]:
    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, pattern in _RULES[category]:
            for match in re.finditer(pattern, text, re.I):
                if not _is_negated(text, match.start()):
                    hits.append(f"{code}: {_redact_pii(match.group(0))}")
    return hits


def literal_claim_hits(text: str) -> list[str]:
    """Banned-claim matches with NO negation suppression.

    Body copy earns negation handling: "we do not promise guaranteed
    savings" is a denial and reads as one. A text-to-image PROMPT does not
    -- it is an instruction to a renderer, and "poster reading do not
    guarantee savings" still draws the forbidden words onto the poster. The
    grammar around the phrase is invisible once it is pixels, so the phrase
    itself is what matters (#2192 round 3).

    Callers that want prose semantics keep using :func:`verify_copy`.
    """
    hits: list[str] = []
    for category in ("outcomes", "automation", "replacing_agents"):
        for code, pattern in _RULES[category]:
            for match in re.finditer(pattern, text, re.I):
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
    # Contact patterns run against the zero-width-stripped view so a
    # formatting-only insertion cannot defeat the gate (#2201). Claim
    # detection keeps the ORIGINAL text: its locators are sentence indices,
    # and rewriting the input would shift them.
    scanned = scan_view(text)
    if _EMAIL_RE.search(scanned):
        hits.append("email: <redacted>")
    if _PHONE_RE.search(scanned):
        hits.append("phone: <redacted>")

    verdict = "fail" if hits else "pass"
    return CopyVerification(verdict=verdict, hits=hits)


# ---------------------------------------------------------------------------
# Advisory engine: token stream, sentence/clause structure, scope model
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][\w'\u2019-]*|\d[\w.-]*")

# Sentence terminators: runs of .!? followed by whitespace + a capital/quote,
# or end of text; a blank line is a structural break. Single newlines (soft
# wraps) never split; digit-internal and abbreviation periods never split.
# Clause boundaries: the minimal proposition. Punctuation (incl. dashes,
# slashes, parens), coordinators, and relativizer/adjunct openers.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;,:\n()/—–]|\s-\s|"
    r"\b(?:and|or|but|however|while|whereas|although|yet|that|which|who|when|if|unless|because|since|so|therefore|hence|thus)\b",
    re.I,
)

# Negation vocabulary split by grammatical role (round 12):
#   determiner negation scopes at most two tokens right ("no delay");
#   verbal negation scopes to its clause end ("do not draft ...").
_DET_NEGATION = frozenset({"no", "none", "nothing", "neither"})
_VERBAL_NEGATION = frozenset(
    {"not", "never", "cannot", "nobody", "don", "doesn", "didn",
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
    r"\b(?:"
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
# Subjects that are anaphoric ON THEIR OWN -- they take no noun, so their
# reference is necessarily back to the report's items.
_BARE_ANAPHORS = frozenset({"it", "they", "everything"})
# Quantifiers/determiners that MAY carry a noun. `every` alone used to satisfy
# the anaphor test, so "Every invoice is assigned to Billing" covered a report
# about issues (#2189). These bind only when used bare, or when the noun they
# carry is itself a report item.
_QUANTIFIER_SUBJECTS = frozenset(
    {"each", "every", "all", "both", "these", "those"}
)
# What follows a BARE quantifier: a predicate, not a noun. Closed function-word
# class (copulas, auxiliaries, modals).
_PREDICATE_FOLLOWERS = frozenset(
    {"is", "are", "was", "were", "be", "been", "being", "gets", "get", "got",
     "goes", "go", "has", "have", "had", "will", "would", "can", "could",
     "must", "should", "may", "might", "does", "do", "did"}
)
# Pro-forms carry no lexical content: "each ONE is routed" still refers back to
# the report's items, so they continue the anaphor rather than renaming the
# subject the way "each INVOICE" does.
_SUBJECT_PRO_FORMS = frozenset({"one", "ones", "them", "those", "these"})
# A PP after the subject MODIFIES it; the head is what precedes the preposition.
_SUBJECT_MODIFIER_PREPOSITIONS = frozenset(
    {"in", "for", "on", "at", "from", "with", "about", "under", "within", "by"}
)
_ANAPHORIC_SUBJECTS = _BARE_ANAPHORS | _QUANTIFIER_SUBJECTS

_VERB_INITIAL_ROUTING = frozenset(
    {"routes", "routed", "route", "assigned", "owned", "needs", "who"}
)
_REPORT_ITEM_NOUNS = frozenset(
    {"issue", "issues", "ticket", "tickets", "fix", "fixes", "item", "items",
     "question", "questions", "finding", "findings"}
)

_CTA_REMINDER = ADVISORY_CTA_REMINDER


def _subject_binds_to_report(words: "list[str]") -> bool:
    """Whether a clause subject refers back to the report's items.

    ONE definition, used by both the same-sentence and later-sentence paths --
    they previously carried the same test written twice, which is how the
    quantifier hole survived in both (#2189).

    A noun-bearing quantifier is classified by its actual SUBJECT HEAD: the
    last token before the predicate. That handles modifiers and partitives
    uniformly ("each of the open tickets" -> tickets) instead of special-casing
    a fixed token window, which rejected valid modified heads.
    """
    first = words[0] if words else ""
    if first in _BARE_ANAPHORS:
        return True
    if first not in _QUANTIFIER_SUBJECTS:
        return first == "the" and len(words) > 1 and words[1] in _REPORT_ITEM_NOUNS

    # The grammatical head is the noun the quantifier binds, BEFORE any
    # post-modifier. Taking the last pre-predicate token instead read "each
    # ticket in the REPORT" as being about reports and "each invoice for a
    # TICKET" as being about tickets -- one regressed valid routing, the other
    # preserved the original false negative (#2189 round 2).
    rest = words[1:9]
    if rest and rest[0] == "of":
        rest = rest[1:]  # partitive: the head sits inside the of-phrase
    head = ""
    for word in rest:
        if word in _PREDICATE_FOLLOWERS or word in _SUBJECT_MODIFIER_PREPOSITIONS:
            break
        head = word
    if not head:
        # Bare use: the predicate follows the quantifier directly ("Each IS
        # assigned..."), so the reference is necessarily anaphoric.
        return True
    return head in _REPORT_ITEM_NOUNS or head in _SUBJECT_PRO_FORMS


_FOCUS_MODIFIERS = frozenset({"only", "even", "just", "especially", "particularly"})


def _sentence_structure(text: str) -> "tuple[list[int], list[tuple[int, int]]]":
    """(starts, spans) over the SHARED sentence definition in the contracts
    module, so the engine and the locator validator cannot disagree (#2189)."""
    spans = sentence_spans(text)
    return [s for s, _e in spans], spans


def _clause_structure(
    text: str,
) -> "tuple[tuple[list[int], list[tuple[int, int]]], list[str]]":
    """Clause spans plus boundary kinds from ONE filtered pass (they must
    stay aligned). Coordinators split propositions, not phrases: "and"/"or"
    is NOT a boundary when it joins two -ly adverbs ("clearly and
    consistently ranks") or introduces a short trailing noun phrase of at
    most two tokens ("...answers or resolutions."), which keeps a denial's
    scope over its coordinated objects."""
    spans: list[tuple[int, int]] = []
    kinds: list[str] = []
    start = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(text):
        word = match.group(0).strip().lower()
        if word in ("and", "or"):
            before = _LAST_WORD_RE.search(
                text[max(0, match.start() - 40) : match.start()]
            )
            after = re.match(r"\s*([\w'-]+)", text[match.end() :])
            if (
                before is not None
                and after is not None
                and before.group(1).lower().endswith("ly")
                and after.group(1).lower().endswith("ly")
            ):
                continue
            next_boundary = _CLAUSE_BOUNDARY_RE.search(text, match.end())
            segment_end = next_boundary.start() if next_boundary else len(text)
            if len(text[match.end() : segment_end].split()) <= 2:
                continue
        spans.append((start, match.start()))
        kinds.append("word" if word.isalpha() else "punct")
        start = match.end()
    spans.append((start, len(text)))
    return ([s for s, _e in spans], spans), kinds


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
    """Scopes only; see `_clause_negations` for the verbal classification."""
    return _clause_negations(text, span)[0]


def _clause_negations(
    text: str, span: "tuple[int, int]"
) -> "tuple[list[tuple[int, int]], bool]":
    """Negation scopes inside one clause as (start, end) character ranges.

    Determiner negation covers itself plus at most two following tokens;
    verbal negation covers from the trigger to the clause end; emphatic
    "not only/just" produces no scope.
    """
    tokens = _clause_tokens(text, span)
    scopes: list[tuple[int, int]] = []
    has_verbal = False
    for position, token in enumerate(tokens):
        lower = token.group(0).lower()
        if lower == "without":
            # "without" negates its bounded complement ("without evidence we
            # draft answers" keeps the claim); the gerund form denies the
            # action itself ("without drafting answers") to clause end.
            following = (
                tokens[position + 1].group(0).lower()
                if position + 1 < len(tokens)
                else ""
            )
            if following.endswith("ing"):
                scopes.append((token.start(), span[1]))
            else:
                end_token = tokens[min(position + 2, len(tokens) - 1)]
                scopes.append((token.start(), end_token.end()))
            continue
        if lower in _DET_NEGATION:
            if position == 0:
                # Subject-position determiner ("No support agent drafts
                # answers") denies the entire proposition.
                scopes.append((token.start(), span[1]))
            else:
                end_token = tokens[min(position + 2, len(tokens) - 1)]
                scopes.append((token.start(), end_token.end()))
        elif lower in _VERBAL_NEGATION or lower.endswith(("n't", "n\u2019t")):
            if (
                lower == "not"
                and position + 1 < len(tokens)
                and tokens[position + 1].group(0).lower() in ("only", "just")
            ):
                continue
            has_verbal = True
            scopes.append((token.start(), span[1]))
    return scopes, has_verbal


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
        # A qualifier with a NEGATED complement is no qualification at all:
        # "if the tickets contain no proof" must not excuse the claim
        # (the standing honest-gap form "no proven answer" matches as a
        # whole phrase and is unaffected).
        # Case-insensitive, like the qualifier detector itself. Without re.I
        # this check saw "no proof" but not "NO proof", so ordinary generated
        # casing was accepted as a qualification and suppressed the warning
        # (#2189). Polarity is not a property of capitalization.
        if re.match(
            r"\s+(?:[\w'\u2019-]+\s+){0,2}(?:no|not|never|nothing|none|zero)\b",
            text[qualifier.end() :],
            re.IGNORECASE,
        ):
            continue
        index, _span = _span_for(clause_bounds, max(qualifier.end() - 1, 0))
        qualifier_clauses.add(index)

    _starts, clause_spans = clause_bounds
    # Adjacency skips clauses with no substantive tokens: boundary runs
    # ("when that") leave empty spans, and a bare focus modifier ("Only
    # when evidence exists, ...") is part of the fronted qualifier, not a
    # proposition of its own.
    content = [
        any(
            token.group(0).lower() not in _FOCUS_MODIFIERS
            for token in _clause_tokens(text, span)
        )
        for span in clause_spans
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


def _assertion_negated(
    text: str,
    clause_bounds: "tuple[list[int], list[tuple[int, int]]]",
    start: int,
    end: int,
    cache: "dict[int, list[tuple[int, int]]]",
    verbal_cache: "dict[int, bool]",
) -> bool:
    """Whether a negation actually denies the assertion about [start, end).

    Intersecting a range that runs to the CLAUSE END swept in trailing
    adjuncts, so "The Resolution Audit is provided without delay" read as a
    denial (#2189). Polarity is decided by the negation's KIND instead, which
    the scope model already encodes:

      * a scope covering the term itself denies it ("NO Resolution Audit is
        provided");
      * a VERBAL negation governs the predicate wherever the term sits, so it
        denies too ("The Resolution Audit for this month is NOT provided") --
        verbal scopes are exactly those the model extends to the clause end;
      * a bounded scope elsewhere in the clause is an adjunct's own complement
        ("without delay", "with no delay") and denies nothing.

    Also O(1) per term rather than a fresh suffix scan, so a clause with many
    product terms stays linear.
    """
    index, span = _span_for(clause_bounds, start)
    if index not in cache:
        # Computed once per clause -- rescanning per product term is what made
        # a clause with many terms quadratic.
        scopes, has_verbal = _clause_negations(text, span)
        cache[index] = scopes
        verbal_cache[index] = has_verbal
    for scope_start, scope_end in cache[index]:
        if scope_start < end and start < scope_end:
            return True
    # The scope model's OWN verbal classification, not a second matcher: it
    # already treats emphatic "not only/just" as affirmative, and an
    # independent regex over every `not` disagreed with it (#2189 round 2).
    if index not in verbal_cache:
        verbal_cache[index] = _clause_negations(text, span)[1]
    return verbal_cache[index]


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
    verbal_cache: dict[int, bool] = {}
    for match in _PRODUCT_TERM_RE.finditer(text):
        _index, span = _span_for(clause_bounds, match.start())
        # Polarity spans the whole assertion: "The Resolution Audit is not
        # provided." denies the surface, so it is not report-shaped.
        if not _assertion_negated(
            text,
            clause_bounds,
            match.start(),
            match.end(),
            negation_cache,
            verbal_cache,
        ):
            sentences.add(_sentence_of(sentence_bounds, match.start()))
    _starts, clause_spans = clause_bounds
    for span in clause_spans:
        tokens = _clause_tokens(text, span)
        for position, token in enumerate(tokens):
            if token.group(0).lower() not in _REPORT_NOUNS:
                continue
            window = tokens[position + 1 : position + 9]
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
    tokens_cache: dict[int, list] = {}
    shape_clauses: set[int] = set()
    for product in _PRODUCT_TERM_RE.finditer(text):
        shape_clauses.add(_span_for(clause_bounds, product.start())[0])
    for noun_match in re.finditer(
        r"\b(?:report|reports|audit|audits|snapshot|snapshots)\b", text, re.I
    ):
        shape_clauses.add(_span_for(clause_bounds, noun_match.start())[0])
    for match in _OWNER_ROUTING_RE.finditer(text):
        if not _routing_relation_affirmative(
            text, match, clause_bounds, negation_cache
        ):
            continue
        sentence = _sentence_of(sentence_bounds, match.start())
        if sentence in report_sentences:
            index, span = _span_for(clause_bounds, match.start())
            # Same-sentence evidence must still be ABOUT the report: same
            # clause as the report noun, a verb-first clause (coordinated
            # verb phrase sharing the report's subject: "... and routes
            # each issue to ..."), or an anaphoric/report-item subject.
            match_words = match.group(0).split()
            match_first = match_words[0].lower() if match_words else ""
            each_object = re.search(
                r"\beach\s+([\w'\u2019-]+)", match.group(0), re.I
            )
            routed_non_item = (
                each_object is not None
                and each_object.group(1).lower() not in _REPORT_ITEM_NOUNS
            )
            if index in shape_clauses:
                # Sharing the report's clause is not enough when the match
                # explicitly routes a NON-report object ("The report routes
                # each invoice to Billing and ranks issues").
                if not routed_non_item:
                    return True
                continue
            if index not in tokens_cache:
                tokens_cache[index] = _clause_tokens(text, span)
            clause_tokens = tokens_cache[index]
            has_before = bool(clause_tokens) and (
                clause_tokens[0].start() < match.start()
            )
            if not has_before and match_first in _VERB_INITIAL_ROUTING:
                # Coordinated verb phrase sharing the report's subject:
                # "... and routes each issue to the owning team." The
                # routed object must itself be a report item -- "routes
                # each invoice to Billing" concerns invoices, not the
                # ranked issues, and does not cover the report.
                each_object = re.search(
                    r"\beach\s+([\w'\u2019-]+)", match.group(0), re.I
                )
                if not routed_non_item:
                    return True
            if has_before:
                subject_words = [t.group(0).lower() for t in clause_tokens[:9]]
            else:
                subject_words = [w.lower() for w in match_words[:9]]
            if _subject_binds_to_report(subject_words):
                return True
            continue
        if any(sentence > report for report in report_sentences):
            index, span = _span_for(clause_bounds, match.start())
            if index not in subject_cache:
                tokens = _clause_tokens(text, span)
                subject_words = [t.group(0).lower() for t in tokens[:9]]
                # Subject position only (round 13): the clause must be ABOUT
                # the report's items -- an anaphoric subject ("Each is
                # assigned...") or a determiner + report-item noun ("These
                # issues are routed..."). An anaphoric token buried in an
                # unrelated object ("owns invoices for each customer") does
                # not bind.
                subject_cache[index] = _subject_binds_to_report(subject_words)
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

    sentence_bounds = _sentence_structure(text)
    clause_bounds, boundary_kinds = _clause_structure(text)
    negation_cache: dict[int, list[tuple[int, int]]] = {}
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
