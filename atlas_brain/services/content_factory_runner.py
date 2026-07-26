"""Run one Content Factory stage: call its Open WebUI worker, extract the JSON
artifact from the reply, and validate + persist it via the artifact store.

Open WebUI is the external boundary here (it fronts the local model); the store
and the content_factory contracts do the validation, so a worker that returns
malformed output is caught before anything is persisted. This runs in the
atlas_brain env (it uses the store + contracts); OWUI worker wrappers are
addressed by model id.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from atlas_brain.services.content_factory_copy_verification import (
    advisory_warnings,
    literal_claim_hits,
    verify_copy,
)
from atlas_brain.schemas.content_factory import model_for
from atlas_brain.services.content_factory_store import (
    DEFAULT_ROOT,
    ArtifactStoreError,
    job_lock,
    read_committed_artifact_bytes,
    write_artifact,
)

DEFAULT_OWUI_URL = "http://127.0.0.1:8080"

# The editor stage's artifact schemas. The promote decision is gated (via
# #2116's EditorialAudit contract) on copy_verification.verdict == "pass".
# Workers may still emit v1 or v2; the runner normalizes to the current version
# (v3, which carries the advisory checklist AND the draft fingerprint) before
# persisting. Older versions stay frozen and readable -- see the contracts.
_EDITOR_SCHEMA_VERSIONS = {
    "editorial_audit.v1": 1,
    "editorial_audit.v2": 2,
    "editorial_audit.v3": 3,
}
_EDITOR_SCHEMAS = tuple(_EDITOR_SCHEMA_VERSIONS)
_CURRENT_EDITOR_SCHEMA = "editorial_audit.v3"
_CURRENT_EDITOR_VERSION = _EDITOR_SCHEMA_VERSIONS[_CURRENT_EDITOR_SCHEMA]

# A leading ```/```json fence and a trailing fence, so a fenced reply still
# yields its JSON body.
_FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")


class WorkerError(RuntimeError):
    """Raised when the worker call fails or returns no usable JSON artifact."""


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the single JSON object embedded in a worker reply, tolerating code
    fences and surrounding prose. Returns None if no JSON object parses."""
    stripped = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip())).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start : end + 1])
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def call_worker(
    model: str,
    user_content: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_OWUI_URL,
    timeout: float = 420.0,
) -> str:
    """POST one user turn to Open WebUI's chat completions for ``model`` and
    return the assistant's text.

    Raises WorkerError on a transport/HTTP failure, a non-JSON body, or a
    response that carries no assistant message.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read() or b"null")
    except urllib.error.URLError as exc:  # HTTPError is a subclass
        raise WorkerError(f"Open WebUI chat call failed for {model!r}: {exc}") from exc
    except ValueError as exc:
        raise WorkerError(f"Open WebUI returned non-JSON for {model!r}: {exc}") from exc
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise WorkerError(
            f"Open WebUI response for {model!r} has no assistant message"
        ) from exc
    return content or ""


_REPURPOSING_SCHEMA = "repurposing.v1"
_IMAGE_PROMPT_SCHEMA = "image_prompt.v1"


def _deterministic_verdict(text: str, *, empty_reason: str) -> "tuple[dict, list[str]]":
    """(verdict, warnings) computed from ``text``. Blank text fails closed:
    with nothing verified, no artifact may carry a passing verdict."""
    if not text.strip():
        return (
            {"verdict": "fail", "hits": [f"unverified-copy: {empty_reason}"]},
            [],
        )
    return verify_copy(text).model_dump(), advisory_warnings(text)


def _enforce_repurposing(artifact: dict[str, Any]) -> None:
    """OVERWRITE each variant's verdict/checklist from its own body.

    Variants are the copy that ships, so each is verified independently and
    the worker's self-reported values are discarded -- the same discipline
    that makes the editorial audit unable to self-promote. A variant whose
    body is blank fails closed, which in turn makes a worker-asserted
    ``ready_to_publish`` invalid at contract validation.
    """
    if artifact.get("schema") != _REPURPOSING_SCHEMA:
        return
    variants = artifact.get("variants")
    if not isinstance(variants, list):
        return
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        verdict, warnings = _deterministic_verdict(
            str(variant.get("body_markdown") or ""),
            empty_reason="body_markdown is empty; nothing was verified",
        )
        variant["copy_verification"] = verdict
        variant["advisory_warnings"] = warnings


# --- prompt contact-PII classifiers -------------------------------------
#
# EVIDENCE-GATED, not pattern-enumerated. Rounds 3-6 each broke because the
# rule asked "do these digits look like a phone number?" -- a question with
# no closed answer, since dates, RGB triples, dimensions and dialable
# numbers share a digit grammar. The decision asks for POSITIVE EVIDENCE.
#
# Round 8 replaced "dial verb anywhere + any 7-15 digit token anywhere" with
# a SHAPE-FIRST tier split, because the old rule flagged "a person calling
# across a room, RGB palette 255 255 255".
#
# Scoping intent to a window -- the obvious repair -- does not work, and the
# counter-example matters enough to record: in "a call center scene, 1920
# 1080 resolution" the gap from "call" to "1920" is TWO WORDS, exactly the
# gap in "call me, 5551234567". No window separates them. Shape does:
#
#   unambiguous -- fails with NO intent required: E.164, NANP 3-3-4, [3,4]
#                  local, a national trunk prefix, or vanity spelling.
#                  Nothing describes artwork in these shapes.
#   unbroken    -- one 7-15 digit run. Phone-plausible, but serials and
#                  seeds are written identically, so it needs a dial verb
#                  within 3 tokens (which may cross a comma, because
#                  "call me, 5551234567" really is written that way).
#   grouped     -- a shape descriptive numbers also use ([3,3,3] RGB,
#                  [4,4] resolution, [4,2,2] dates). Weakest evidence, so
#                  it needs the tightest government: a dial verb within 2
#                  tokens and no boundary crossed.
#
# Absent both, digits are just description and the prompt passes -- without
# enumerating "RGB", "resolution" or any other descriptive vocabulary,
# which is the string-closure trap AGENTS.md 3k.1 forbids.

_ANY_EMAIL_RE = re.compile(
    r"[^\s@,;:()<>\[\]]+@[^\s@,;:()<>\[\]]+\.[^\s@,;:()<>\[\]]{2,}"
)
# IDNA treats these full-width/ideographic stops as domain-label separators.
# Normalize only for the admission decision; the raw prompt is never copied
# into a finding.
_IDNA_DOT_TRANSLATION = str.maketrans({"\u3002": ".", "\uff0e": ".", "\uff61": "."})

_DIAL_INTENT_RE = re.compile(
    r"\b(?:call|calling|dial|dialling|dialing|phone|telephone|tel|text|txt|"
    r"sms|ring|hotline|helpline|whatsapp|contact|reach)\b",
    re.I,
)
_DIAL_PUNCTUATION = r".\-/"
_DIAL_SEPARATOR_RE = re.compile(rf"[\s{_DIAL_PUNCTUATION}]+")
_NUMERIC_DIAL_SEPARATOR = rf"(?:\s+(?=\d)|[{_DIAL_PUNCTUATION}])"
# E.164 / international: explicit + or 00 prefix then 7-15 digits.
# Slash is deliberately excluded from this unconditional shortcut; slash-
# delimited candidates go through the structural dial decision below so
# renderer dimensions and dates do not become phone evidence.
_E164_RE = re.compile(r"(?:\+|\b00)[\d\s().\-]{7,20}\d")
# North American 3-3-4, the one local shape that is unambiguous.
_NANP_RE = re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")
# A token someone could dial: starts with a digit, 7+ alphanumerics once
# separators are stripped (covers 1-800-GOT-JUNK and 07700 900123 alike).
# Continuation groups are digits (any separator) or LETTERS ONLY when
# hyphen/dot-joined. Vanity spelling is case-insensitive ("1-800-flowers"
# is the same number as "1-800-FLOWERS"), but a space-joined lowercase word
# is the next word rather than part of the number ("07700 900123 today").
# Attachment, not casing, is what distinguishes them.
_DIAL_TOKEN_RE = re.compile(
    rf"(?<!\w)[+]?\d[\dA-Za-z]*(?:{_NUMERIC_DIAL_SEPARATOR}[\dA-Za-z]+){{0,5}}"
)


# Groupings only a dialable number uses. [3,4] is the local form (555-1234),
# [3,3,4] NANP, [1,3,3,4] NANP with the country code written out.
_UNAMBIGUOUS_GROUPINGS = frozenset({(3, 4), (3, 3, 4), (1, 3, 3, 4)})
# Ambiguous number/phoneword shapes need a finite syntactic bridge from the
# dial marker. Arbitrary nearby words are not evidence: the open semantic
# category behind "text for room 212 art deco" cannot be closed by proximity.
# These are function words in direct-address/dial syntax, not content nouns.
_DIAL_BRIDGE_WORDS = frozenset({"at", "me", "on", "us", "via"})
_DIAL_BRIDGE_WORD_RE = re.compile(r"[A-Za-z]+")
_DIAL_BRIDGE_PUNCT_RE = re.compile(r"^[ \t,;:()\-\r\n]*$")


def _is_nanp_digits(digits: str) -> bool:
    """A syntactically valid North American number.

    The NANP forbids 0 and 1 as the leading digit of BOTH the area code and
    the exchange code. That is a spec constraint, not a vocabulary, and it is
    what separates a dialable unbroken 10-digit run from a 10-digit serial or
    seed -- which is the one case shape alone cannot otherwise decide.
    """
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10 or not digits.isdigit():
        return False
    return digits[0] in "23456789" and digits[3] in "23456789"


_KEYPAD = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "22233344455566677778889999" "22233344455566677778889999",
)


def _is_vanity_number(parts: "list[str]", *, international: bool = False) -> bool:
    """Is this mixed digit/letter token a dialable vanity number?

    Letters in a vanity number are DIGIT SUBSTITUTES, so the test is whether
    the keypad-mapped token is a real number -- not merely whether letters and
    digits co-occur. That earlier rule rejected renderer specifications:
    "16-bit-color", "1920-1080-pixel" and "8-bit-style" all have letters
    attached to digits (#2192 round 9).

    Domestic candidates require two conditions, both from the numbering plan
    rather than a word list:

      * the leading digit group is an AREA CODE -- exactly 3 digits, or a "1"
        country prefix followed by 3. "16-bit-color" leads with 2 digits and
        "1920-1080-pixel" with 4, so neither can be a dialable prefix.
      * the keypad-mapped digits form a syntactically valid NANP number.
    """
    if not any(p.isalpha() for p in parts) or not any(p.isdigit() for p in parts):
        return False
    mapped = "".join(parts).translate(_KEYPAD)
    if international:
        # ``00`` is an international access prefix, not part of the E.164
        # number. An explicit +/00 prefix plus dial intent supplies the
        # structural evidence that detached international phonewords need.
        e164_digits = mapped[2:] if mapped.startswith("00") else mapped
        return e164_digits.isdigit() and 7 <= len(e164_digits) <= 15
    digit_groups = [p for p in parts if p.isdigit()]
    lead = digit_groups[0]
    if lead == "1" and len(digit_groups) > 1:
        lead = digit_groups[1]
    if len(lead) != 3:
        return False
    return _is_nanp_digits(mapped)


def _dial_shape(token: str) -> str:
    """Classify a candidate token: 'unambiguous', 'ambiguous', or 'none'.

    The digit-GROUP PROFILE is the discriminator, not the compact length.
    "5551234567" is one unbroken 10-digit run -- nothing describes artwork
    that way -- while "255 255 255" is [3,3,3] and "1920 1080" is [4,4].
    """
    stripped = token.lstrip("+")
    parts = [p for p in _DIAL_SEPARATOR_RE.split(stripped) if p]
    if not parts:
        return "none"
    # A MIXED alphanumeric group is not a number: "1920x1080" is a canvas
    # size. Vanity spelling puts its letters in their own hyphen/dot-joined
    # group ("1-800-FLOWERS"), which is what makes it dialable.
    if any(not (p.isdigit() or p.isalpha()) for p in parts):
        return "none"
    compact = "".join(parts)
    symbol_limit = 17 if stripped.startswith("00") else 15
    if not 7 <= len(compact) <= symbol_limit:
        return "none"
    if any(p.isalpha() for p in parts):
        international = token.startswith("+") or stripped.startswith("00")
        if international and _is_vanity_number(parts, international=True):
            # Explicit international prefix is strong structure, but unlike a
            # fully numeric E.164 token the detached letters remain ambiguous
            # prose. Dial intent supplies the other side of the evidence gate.
            return "unbroken"
        return "unambiguous" if _is_vanity_number(parts) else "none"
    groups = tuple(len(p) for p in parts)
    if compact.startswith("0") and 9 <= len(compact) <= 15 and len(groups) > 1:
        return "unambiguous"  # national trunk prefix: 07700 900123
    if groups in _UNAMBIGUOUS_GROUPINGS:
        return "unambiguous"
    if len(groups) == 1 and _is_nanp_digits(compact):
        # An unbroken run that is a SYNTACTICALLY VALID North American number
        # needs no corroboration: serials do not obey the NANP's constraints
        # by accident often enough to matter, and "5552345678 on a poster" is
        # about to be painted into artwork.
        return "unambiguous"
    if len(groups) == 1:
        # One unbroken 7-15 digit run. Phone-plausible, but serials, seeds and
        # order numbers are written the same way ("serial 12345678 engraved on
        # a plate" must render), so this needs corroboration -- with the wider
        # window, since it is the stronger of the two ambiguous shapes.
        return "unbroken"
    return "grouped"


_TRAILING_ALPHA_RE = re.compile(
    rf"\s+[A-Za-z]+(?:[{_DIAL_PUNCTUATION}][A-Za-z]+)*"
)
_MAX_DIAL_SYMBOLS = 17  # 00 access prefix plus E.164's 15-digit maximum


def _token_candidates(text: str, match: "re.Match[str]") -> "list[str]":
    """The matched token plus every still-dialable alpha-word extension.

    The token regex deliberately stops before space-joined letters so it does
    not swallow ordinary prose after a numeric token. Vanity suffixes may use
    spaces, though, and an arbitrary word-count cap leaves the same grammar
    open. Extend until the international dial bound is exceeded; later words
    can only make the candidate longer, so no dialable candidate is skipped.
    Detached extensions are not evidence by themselves: `_phone_evidence`
    also requires nearby dial intent before admitting one as contact PII.
    """
    candidates = [match.group(0)]
    end = match.end()
    while True:
        extension = _TRAILING_ALPHA_RE.match(text, end)
        if extension is None:
            break
        candidate = text[match.start() : extension.end()]
        compact = _DIAL_SEPARATOR_RE.sub("", candidate.lstrip("+"))
        if len(compact) > _MAX_DIAL_SYMBOLS:
            break
        candidates.append(candidate)
        end = extension.end()
    return candidates


def _is_structural_dial_bridge(gap: str) -> bool:
    """Whether ``gap`` is finite direct-address/dial syntax.

    The open set of renderer/content words defaults to false. Only punctuation
    plus the closed function-word grammar can connect a dial marker to an
    otherwise ambiguous number or detached phoneword.
    """
    words = _DIAL_BRIDGE_WORD_RE.findall(gap)
    if len(words) > 3 or any(
        word.casefold() not in _DIAL_BRIDGE_WORDS for word in words
    ):
        return False
    punctuation = _DIAL_BRIDGE_WORD_RE.sub("", gap)
    if _DIAL_BRIDGE_PUNCT_RE.fullmatch(punctuation) is None:
        return False
    logical_lines = punctuation.replace("\r\n", "\n").replace("\r", "\n")
    return logical_lines.count("\n") <= 1


def _has_structural_dial_intent(
    text: str,
    start: int,
    end: int,
    intents: "list[tuple[int, int]]",
) -> bool:
    """Whether a dial marker structurally governs the candidate span."""
    for intent_start, intent_end in intents:
        if intent_end <= start:
            gap = text[intent_end:start]
        elif end <= intent_start:
            gap = text[end:intent_start]
        else:
            return True
        if _is_structural_dial_bridge(gap):
            return True
    return False


def _phone_evidence(text: str) -> bool:
    """Positive evidence that ``text`` carries a dialable number."""
    # Admission-only normalization: compatibility-equivalent digits, letters,
    # and separators receive one verdict without rewriting the stored prompt.
    text = unicodedata.normalize("NFKC", text)
    if _E164_RE.search(text) or _NANP_RE.search(text):
        return True
    intents = [(m.start(), m.end()) for m in _DIAL_INTENT_RE.finditer(text)]
    for match in _DIAL_TOKEN_RE.finditer(text):
        # Attached vanity spelling is structural evidence. A SPACE-joined
        # suffix is not: "212 art deco" can keypad-map to a valid NANP number
        # while remaining ordinary renderer prose. Detached candidates must
        # therefore also sit under the finite structural dial bridge. This
        # same rule admits
        # explicit international phonewords such as "+44 800 FLOWERS" without
        # pretending they are NANP numbers.
        candidates = _token_candidates(text, match)
        base_shape = _dial_shape(candidates[0])
        if base_shape == "unambiguous":
            return True
        for candidate in candidates[1:]:
            if _dial_shape(candidate) == "none":
                continue
            candidate_end = match.start() + len(candidate)
            if _has_structural_dial_intent(
                text,
                match.start(),
                candidate_end,
                intents,
            ):
                return True
        shape = base_shape
        if shape == "none":
            continue
        if _has_structural_dial_intent(
            text,
            match.start(),
            match.end(),
            intents,
        ):
            return True
    return False


def _prompt_contact_hits(text: str) -> list[str]:
    """Contact-PII findings for renderer instructions.

    Fails on evidence of contact intent; stays silent on description.
    """
    hits: list[str] = []
    if _ANY_EMAIL_RE.search(text.translate(_IDNA_DOT_TRANSLATION)):
        hits.append("email: <redacted>")
    if _phone_evidence(text):
        hits.append("phone: <redacted>")
    return hits


def _enforce_image_prompts(artifact: dict[str, Any]) -> None:
    """Gate the POSITIVE prompt text, one prompt at a time.

    ``negative_prompt`` is deliberately EXCLUDED from the verdict. A negative
    prompt is an exclusion list -- naming a banned phrase there is the
    designer telling the renderer NOT to draw it, which is the correct
    response to this module's own threat model. Folding it into the scan
    made the safest possible prompt set the one that failed (round 1: the
    guard failed on its second side).

    Each prompt is verified INDEPENDENTLY and the results aggregated, so
    joining items can never synthesize a claim that no single rendered
    prompt contains ("...guaranteed" + "savings..." across two prompts).

    PII is stricter here than in body copy: a prompt is an instruction to a
    renderer, and any contact string in it is about to be drawn into an
    image where no text check can reach it. So international phone forms
    fail here even though `verify_copy` leaves the shared body-copy verdict
    semantics unchanged (that scope was frozen deliberately in #2181).
    """
    if artifact.get("schema") != _IMAGE_PROMPT_SCHEMA:
        return
    prompts = artifact.get("prompts")
    if not isinstance(prompts, list):
        return

    texts = [
        str(prompt.get("prompt_text") or "")
        for prompt in prompts
        if isinstance(prompt, dict)
    ]
    if not any(text.strip() for text in texts):
        artifact["copy_verification"] = {
            "verdict": "fail",
            "hits": ["unverified-copy: prompt text is empty; nothing was verified"],
        }
        artifact["advisory_warnings"] = []
        return

    hits: list[str] = []
    for index, text in enumerate(texts, start=1):
        if not text.strip():
            continue
        # Literal matching: prose negation does not un-draw words a renderer
        # is told to put on a poster (round 3).
        for hit in literal_claim_hits(text):
            hits.append(f"prompt {index}: {hit}")
        for contact_hit in _prompt_contact_hits(text):
            hits.append(f"prompt {index}: {contact_hit}")

    artifact["copy_verification"] = {
        "verdict": "fail" if hits else "pass",
        "hits": hits,
    }
    # The advisory layer is tuned for marketing PROSE (answer/ownership
    # claims, report shape). Its sentence locators are not meaningful across
    # a set of independent prompts, so prompt sets carry no advisory
    # warnings rather than ambiguous ones.
    artifact["advisory_warnings"] = []


def _enforce_copy_verification(artifact: dict[str, Any]) -> None:
    """For ANY editorial audit (gated by schema, not stage name -- see below), OVERWRITE
    copy_verification with the deterministic verdict computed from the edited copy,
    discarding any value the worker reported.

    This is what makes the model unable to self-promote: #2116's EditorialAudit contract
    rejects ``recommendation == "promote"`` unless ``copy_verification.verdict == "pass"``,
    and after this the verdict is the deterministic gate's, not the worker's claim. So if
    the edited copy overclaims or leaks PII, the injected "fail" verdict makes a
    worker-asserted "promote" invalid and the store rejects the artifact (fail closed);
    a "revise" recommendation still persists with the recorded hits.

    Gating is by SCHEMA (``editorial_audit.v1``/``v2``), not by the canonical "audit" stage name:
    the store lets a custom stage carry any artifact, and any editorial_audit.v1 can
    promote, so gating by stage name would let a custom-stage audit bypass the gate.

    Empty/blank edited copy fails closed: with nothing to verify, the audit cannot carry a
    passing verdict (otherwise a worker could self-promote by omitting the edited body).
    Verifying the parent draft body in that case is a later refinement (#2136).
    """
    if artifact.get("schema") not in _EDITOR_SCHEMAS:
        return
    # Normalize to the newest audit version: the runner-persisted audit always
    # carries the advisory checklist AND the draft fingerprint. Older versions
    # stay frozen for pre-existing artifacts and direct writers (rollback-safe
    # -- see the contracts module).
    #
    # The anti-laundering rule generalizes across the chain: synthesize
    # schema_version ONLY when the worker's metadata is self-consistent (its
    # schema_version matches the version its own tag declares, or is absent).
    # A reply tagged v2 while claiming schema_version 7 keeps the 7 so the
    # Literal[3] validator rejects it, instead of the runner quietly repairing
    # contradictory worker metadata into a valid artifact.
    declared = _EDITOR_SCHEMA_VERSIONS.get(artifact.get("schema"))
    supplied = artifact.get("schema_version")
    if supplied is None or supplied == declared:
        artifact["schema_version"] = _CURRENT_EDITOR_VERSION
    artifact["schema"] = _CURRENT_EDITOR_SCHEMA
    edited = str(artifact.get("edited_body_markdown") or "")
    if not edited.strip():
        artifact["copy_verification"] = {
            "verdict": "fail",
            "hits": ["unverified-copy: edited_body_markdown is empty; nothing was verified"],
        }
        # No copy, no checklist: worker-supplied warnings are discarded too.
        artifact["advisory_warnings"] = []
        return
    artifact["copy_verification"] = verify_copy(edited).model_dump()
    # Same self-report discipline as the verdict: the advisory checklist is
    # computed deterministically from the edited copy, never taken from the
    # worker (a fabricated empty list would blind the reviewing human).
    artifact["advisory_warnings"] = advisory_warnings(edited)


def _read_job_artifact(
    job_id: str, root: "Path | str", stage: str
) -> "dict[str, Any] | None":
    """A committed artifact from the job's Git tree, or None when unreadable.

    The worktree and index are mutable implementation state. A failed commit
    or abrupt process exit may leave bytes there that no commit records; those
    bytes are never canonical readiness input.
    """
    raw = read_committed_artifact_bytes(job_id, stage, root=root)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _draft_claim_ids(draft: "dict[str, Any]") -> "set[str]":
    """Claim source ids the approved draft established."""
    ids: set[str] = set()
    for claim in draft.get("claims") or []:
        if isinstance(claim, dict):
            source_id = str(claim.get("source_id") or "").strip()
            if source_id:
                ids.add(source_id)
    return ids


def _draft_fingerprint(job_id: str, root: "Path | str") -> "str | None":
    """SHA-256 of the committed draft artifact's canonical bytes.

    The store commits a canonical form, so this is stable for identical
    content and changes when a committed draft body or claims change --
    including a same-revision rerun, which a revision number cannot detect.
    Uncommitted worktree/index residue is deliberately invisible here.
    """
    raw = read_committed_artifact_bytes(job_id, "draft", root=root)
    return hashlib.sha256(raw).hexdigest() if raw is not None else None


def _stamp_draft_fingerprint(
    artifact: dict[str, Any], dispatch_fingerprint: "str | None"
) -> None:
    """Bind an audit to the committed draft present when dispatch began.

    Runner-set, never worker-supplied -- the same discipline as the verdict.
    """
    if artifact.get("schema") not in _EDITOR_SCHEMAS:
        return
    artifact["source_draft_fingerprint"] = dispatch_fingerprint


_SOURCE_BOUND_SCHEMAS = frozenset(
    (*_EDITOR_SCHEMAS, _REPURPOSING_SCHEMA, _IMAGE_PROMPT_SCHEMA)
)
_SOURCE_STAGE_INSTRUCTIONS = {
    "audit": "Review the committed draft and return only an editorial audit JSON artifact.",
    "audit-v2": "Review the committed draft and return only an editorial audit JSON artifact.",
    "repurposing": (
        "Transform the committed draft and return only a repurposing.v1 JSON artifact."
    ),
    "image_prompt": (
        "Derive image prompts from the committed draft and return only an "
        "image_prompt.v1 JSON artifact."
    ),
}
_SOURCE_BOUND_STAGES = frozenset(_SOURCE_STAGE_INSTRUCTIONS)


def _build_source_prompt(
    job_id: str,
    root: "Path | str",
    stage: str,
) -> "tuple[str, str | None]":
    """Build the fixed stage prompt and fingerprint from one draft snapshot."""
    raw = read_committed_artifact_bytes(job_id, "draft", root=root)
    draft: dict[str, Any] | None = None
    if raw is not None:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(
                "source-bound stage requires a valid committed draft artifact"
            ) from exc
        if not isinstance(parsed, dict):
            raise ArtifactStoreError(
                "source-bound stage requires a committed draft JSON object"
            )
        draft = parsed
    draft_json = json.dumps(
        draft,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    user_content = (
        f"{_SOURCE_STAGE_INSTRUCTIONS[stage]}\n\n"
        f"Committed draft JSON:\n{draft_json}"
    )
    fingerprint = hashlib.sha256(raw).hexdigest() if raw is not None else None
    return user_content, fingerprint


def _require_dispatch_source_unchanged(
    artifact: dict[str, Any],
    job_id: str,
    root: "Path | str",
    dispatch_fingerprint: "str | None",
) -> None:
    """Reject a worker result when its source changed while it was running.

    The source snapshot is taken immediately before dispatch. Re-reading it
    under the job lock closes the worker-time race without holding a filesystem
    lock across a potentially long network call. All source-derived artifacts
    use the same check, including unready intermediate Phase 6 results.
    """
    if artifact.get("schema") not in _SOURCE_BOUND_SCHEMAS:
        return
    current = _draft_fingerprint(job_id, root)
    if current != dispatch_fingerprint:
        raise ArtifactStoreError(
            "the committed draft changed while the worker was running; discard "
            "the stale response and rerun the stage against the current draft"
        )


def _enforce_lineage(artifact: dict[str, Any], job_id: str, root: "Path | str") -> None:
    """Readiness flags require a verified tie to the job's approved draft.

    Checked for both Phase 6 artifacts once they claim readiness:
      * every cited claim id exists in that draft (a fabricated id is still
        an orphan), and
      * the declared ``source_draft_revision`` matches the draft actually on
        disk -- otherwise a package can ship copy derived from superseded
        text whenever the claim ids happen to overlap (round 3).

    The artifact is validated FIRST so this branches on the same normalized
    values the contract will accept: a weak worker's ``"false"`` string
    coerces to False here exactly as it does in the model, instead of the
    runner reading raw truthiness and disagreeing with admission (round 3).
    Unready artifacts skip the check -- that is the legitimate intermediate
    state -- and a missing/unreadable draft fails closed.
    """
    schema = artifact.get("schema")
    if schema not in (_REPURPOSING_SCHEMA, _IMAGE_PROMPT_SCHEMA):
        return
    model = model_for(artifact).model_validate(artifact)
    ready = getattr(model, "ready_to_publish", False) or getattr(
        model, "ready_to_generate", False
    )
    if not ready:
        return

    draft = _read_job_artifact(job_id, root, "draft")
    if draft is None:
        raise ArtifactStoreError(
            "readiness requires a readable draft artifact in the job folder "
            "to verify claim lineage and source revision"
        )

    # The plan's premise is that Phase 6 derives from an APPROVED draft.
    # Existence proves the draft ran, not that a human/gate cleared it, so
    # require the job's audit to have promoted it (round 4).
    audit = _read_job_artifact(job_id, root, "audit")
    if audit is None or audit.get("recommendation") != "promote":
        raise ArtifactStoreError(
            "readiness requires an audit artifact recommending 'promote'; "
            "unaudited or revise-state copy cannot ship or render"
        )
    # The approval must be FOR this draft: a revision-1 audit does not
    # authorize revision-2 copy, and another project's audit authorizes
    # nothing here (round 5).
    if audit.get("project_id") != draft.get("project_id"):
        raise ArtifactStoreError(
            f"audit project {audit.get('project_id')!r} does not match the "
            f"draft's {draft.get('project_id')!r}"
        )
    if audit.get("draft_revision", 1) != draft.get("revision", 1):
        raise ArtifactStoreError(
            f"audit approved draft revision {audit.get('draft_revision', 1)} "
            f"but the draft on disk is revision {draft.get('revision', 1)}"
        )
    # Content identity, not just the revision label: a same-revision rerun
    # replaces the body while the number stays 1, and the old approval must
    # not carry over to text nobody reviewed (round 7).
    approved_fingerprint = audit.get("source_draft_fingerprint")
    current_fingerprint = _draft_fingerprint(job_id, root)
    if not approved_fingerprint or approved_fingerprint != current_fingerprint:
        raise ArtifactStoreError(
            "the approving audit does not match the draft currently on disk "
            "(content changed since approval, or the audit predates "
            "content binding); re-run the audit stage before shipping"
        )

    # Cross-project derivation: a matching revision and overlapping ids are
    # not evidence when the draft belongs to a different project (round 4).
    artifact_project = getattr(model, "project_id", None)
    draft_project = draft.get("project_id")
    if artifact_project != draft_project:
        raise ArtifactStoreError(
            f"project mismatch: artifact project_id {artifact_project!r} does "
            f"not match the draft's {draft_project!r}"
        )

    declared = getattr(model, "source_draft_revision", None)
    actual = draft.get("revision", 1)
    if declared is not None and declared != actual:
        raise ArtifactStoreError(
            f"source_draft_revision {declared} does not match the draft on "
            f"disk (revision {actual}); the artifact derives from superseded copy"
        )

    known = _draft_claim_ids(draft)
    cited: set[str] = set()
    for variant in getattr(model, "variants", []) or []:
        for claim_id in variant.derived_from_claims:
            cited.add(claim_id.strip())
    unknown = sorted(cited - known)
    if unknown:
        raise ArtifactStoreError(
            "variant lineage cites claims absent from the draft: "
            + ", ".join(unknown)
        )


def run_stage(
    job_id: str,
    stage: str,
    model: str,
    user_content: str | None,
    *,
    api_key: str,
    base_url: str = DEFAULT_OWUI_URL,
    root: Path | str = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Run one stage end to end: call the worker, extract its JSON artifact, enforce the
    deterministic copy-verification verdict on an editor audit, then validate + persist it
    via the store. Returns the store's record.

    Raises WorkerError if the worker call fails or returns no JSON object, and
    ValueError / pydantic ValidationError (from the store) if the artifact fails
    its contract -- so a malformed or self-promoting stage output is never persisted.
    """
    # Source-bound stages accept no caller prompt payload. The runner owns the
    # instruction and draft serialization, so a callback cannot ignore the
    # snapshotted draft while inheriting its fingerprint.
    prompt_is_source_bound = stage in _SOURCE_BOUND_STAGES
    if prompt_is_source_bound and user_content is not None:
        raise ArtifactStoreError(
            "source-bound stage uses a runner-owned prompt; user_content must be None"
        )
    if prompt_is_source_bound:
        dispatched_content, dispatch_fingerprint = _build_source_prompt(
            job_id, root, stage
        )
    else:
        if not isinstance(user_content, str):
            raise TypeError("non-source stage user_content must be str")
        dispatched_content = user_content
        dispatch_fingerprint = _draft_fingerprint(job_id, root)
    reply = call_worker(
        model, dispatched_content, api_key=api_key, base_url=base_url
    )
    artifact = extract_json(reply)
    if artifact is None:
        raise WorkerError(
            f"stage {stage!r}: worker {model!r} returned no JSON artifact"
        )
    _enforce_copy_verification(artifact)
    _enforce_repurposing(artifact)
    _enforce_image_prompts(artifact)
    if artifact.get("schema") in _SOURCE_BOUND_SCHEMAS and not prompt_is_source_bound:
        raise ArtifactStoreError(
            "source-derived artifact requires a runner-owned source stage prompt"
        )
    # Everything that re-checks the source, reads readiness state, or writes the
    # job sits inside one lock. The pre-dispatch snapshot is compared first;
    # lineage/readiness is then decided against that same committed state and
    # the result commits before the lock is released (#2192 rounds 8-10).
    with job_lock(job_id, root=root):
        _require_dispatch_source_unchanged(
            artifact, job_id, root, dispatch_fingerprint
        )
        _stamp_draft_fingerprint(artifact, dispatch_fingerprint)
        _enforce_lineage(artifact, job_id, root)
        return write_artifact(job_id, stage, artifact, root=root)
