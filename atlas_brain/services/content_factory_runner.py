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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from atlas_brain.services.content_factory_copy_verification import (
    _EMAIL_RE,
    _INTL_PHONE_RE,
    _PHONE_RE,
    advisory_warnings,
    literal_claim_hits,
    verify_copy,
)
from atlas_brain.schemas.content_factory import model_for
from atlas_brain.services.content_factory_store import (
    DEFAULT_ROOT,
    ArtifactStoreError,
    job_dir,
    job_lock,
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

_DIAL_INTENT_RE = re.compile(
    r"\b(?:call|calling|dial|dialling|dialing|phone|telephone|tel|text|txt|"
    r"sms|ring|hotline|helpline|whatsapp|contact|reach)\b",
    re.I,
)
# E.164 / international: explicit + or 00 prefix then 7-15 digits.
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
    r"\b[+]?\d[\dA-Za-z]*(?:(?:[\s](?=\d)|[.\-])[\dA-Za-z]+){0,5}"
)


# Groupings only a dialable number uses. [3,4] is the local form (555-1234),
# [3,3,4] NANP, [1,3,3,4] NANP with the country code written out.
_UNAMBIGUOUS_GROUPINGS = frozenset({(3, 4), (3, 3, 4), (1, 3, 3, 4)})
# Strong punctuation ends a descriptive phrase. A period BETWEEN DIGITS is a
# number separator, not a boundary -- treating it as one shreds "555.1234".
_BOUNDARY_RE = re.compile(r"(?:(?<!\d)\.(?!\d))+|[,;:!?()\[\]{}|/\n\r–—]+")
# How close a dial verb must sit to corroborate an ambiguous token, measured in
# tokens where a punctuation run counts as one. The two windows are NOT taste:
# they track how much evidence the shape itself carries.
#
#   unbroken (3, may cross a boundary) -- "call me, 5551234567" is contact data
#       and calls-to-action really are written across a comma. But the run is
#       also how serials are written, so "a phone on a desk. serial 12345678"
#       (distance 4) stays a scene.
#   grouped  (2, same segment only) -- the weakest shape, so it needs the
#       tightest government. "call me at 12 34 56 78" (2) is contact data;
#       "phone booth photographed at 100 200 300" (3) and "a call center
#       scene, 1920 1080" (crosses a boundary) are scenes. This is what keeps
#       compound nouns -- "call center", "phone booth", "call sheet" -- out
#       without enumerating any of them.
_UNBROKEN_WINDOW = 3
_GROUPED_WINDOW = 2


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


def _dial_shape(token: str) -> str:
    """Classify a candidate token: 'unambiguous', 'ambiguous', or 'none'.

    The digit-GROUP PROFILE is the discriminator, not the compact length.
    "5551234567" is one unbroken 10-digit run -- nothing describes artwork
    that way -- while "255 255 255" is [3,3,3] and "1920 1080" is [4,4].
    """
    parts = [p for p in re.split(r"[\s.\-]+", token.lstrip("+")) if p]
    if not parts:
        return "none"
    # A MIXED alphanumeric group is not a number: "1920x1080" is a canvas
    # size. Vanity spelling puts its letters in their own hyphen/dot-joined
    # group ("1-800-FLOWERS"), which is what makes it dialable.
    if any(not (p.isdigit() or p.isalpha()) for p in parts):
        return "none"
    compact = "".join(parts)
    if not 7 <= len(compact) <= 15:
        return "none"
    if any(p.isalpha() for p in parts):
        return "unambiguous" if any(p.isdigit() for p in parts) else "none"
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


def _gap_profile(gap: str) -> "tuple[int, bool]":
    """(token distance, crossed a strong boundary) for the text between a dial
    verb and a candidate token. A punctuation run counts as one token, so
    distance and boundary-crossing are read off the same measurement."""
    tokens = _BOUNDARY_RE.sub(" \x00 ", gap).split()
    return len(tokens), "\x00" in tokens


def _phone_evidence(text: str) -> bool:
    """Positive evidence that ``text`` carries a dialable number."""
    if _E164_RE.search(text) or _NANP_RE.search(text):
        return True
    intents = [(m.start(), m.end()) for m in _DIAL_INTENT_RE.finditer(text)]
    for match in _DIAL_TOKEN_RE.finditer(text):
        shape = _dial_shape(match.group(0))
        if shape == "unambiguous":
            return True
        if shape == "none":
            continue
        unbroken = shape == "unbroken"
        limit = _UNBROKEN_WINDOW if unbroken else _GROUPED_WINDOW
        for start, end in intents:
            if end <= match.start():
                gap = text[end : match.start()]
            elif match.end() <= start:
                gap = text[match.end() : start]
            else:
                return True
            distance, crossed = _gap_profile(gap)
            if distance <= limit and (unbroken or not crossed):
                return True
    return False


def _prompt_contact_hits(text: str) -> list[str]:
    """Contact-PII findings for renderer instructions.

    Fails on evidence of contact intent; stays silent on description.
    """
    hits: list[str] = []
    if _ANY_EMAIL_RE.search(text):
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
    job_id: str, root: "Path | str", name: str
) -> "dict[str, Any] | None":
    """A persisted artifact from the job folder, or None when unreadable."""
    path = job_dir(job_id, root=root) / name
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
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
    """SHA-256 of the draft artifact's bytes as persisted by the store.

    The store writes a canonical form, so this is stable for identical
    content and changes the moment the draft body or claims change --
    including a same-revision rerun, which a revision number cannot detect.
    """
    path = job_dir(job_id, root=root) / "draft.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _stamp_draft_fingerprint(
    artifact: dict[str, Any], job_id: str, root: "Path | str"
) -> None:
    """Bind an audit to the draft content it actually reviewed. Runner-set,
    never worker-supplied -- the same discipline as the verdict."""
    if artifact.get("schema") not in _EDITOR_SCHEMAS:
        return
    artifact["source_draft_fingerprint"] = _draft_fingerprint(job_id, root)


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

    draft = _read_job_artifact(job_id, root, "draft.json")
    if draft is None:
        raise ArtifactStoreError(
            "readiness requires a readable draft artifact in the job folder "
            "to verify claim lineage and source revision"
        )

    # The plan's premise is that Phase 6 derives from an APPROVED draft.
    # Existence proves the draft ran, not that a human/gate cleared it, so
    # require the job's audit to have promoted it (round 4).
    audit = _read_job_artifact(job_id, root, "audit.json")
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
    user_content: str,
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
    reply = call_worker(model, user_content, api_key=api_key, base_url=base_url)
    artifact = extract_json(reply)
    if artifact is None:
        raise WorkerError(
            f"stage {stage!r}: worker {model!r} returned no JSON artifact"
        )
    _enforce_copy_verification(artifact)
    _enforce_repurposing(artifact)
    _enforce_image_prompts(artifact)
    # Everything that READS the job folder and everything that WRITES it must
    # sit inside one lock. Stamping reads draft.json, the readiness gate reads
    # it again to verify the fingerprint, and the commit lands after both --
    # a concurrent draft rerun anywhere in that window would otherwise ship a
    # "ready" artifact against copy no audit covered (#2192 round 8).
    with job_lock(job_id, root=root):
        _stamp_draft_fingerprint(artifact, job_id, root)
        _enforce_lineage(artifact, job_id, root)
        return write_artifact(job_id, stage, artifact, root=root)
