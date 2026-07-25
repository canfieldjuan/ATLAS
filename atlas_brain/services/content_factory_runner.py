"""Run one Content Factory stage: call its Open WebUI worker, extract the JSON
artifact from the reply, and validate + persist it via the artifact store.

Open WebUI is the external boundary here (it fronts the local model); the store
and the content_factory contracts do the validation, so a worker that returns
malformed output is caught before anything is persisted. This runs in the
atlas_brain env (it uses the store + contracts); OWUI worker wrappers are
addressed by model id.
"""

from __future__ import annotations

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
    write_artifact,
)

DEFAULT_OWUI_URL = "http://127.0.0.1:8080"

# The editor stage's artifact schemas. The promote decision is gated (via
# #2116's EditorialAudit contract) on copy_verification.verdict == "pass".
# Workers may still emit v1; the runner normalizes to v2 (which carries the
# advisory checklist) before persisting.
_EDITOR_SCHEMAS = ("editorial_audit.v1", "editorial_audit.v2")

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


# --- prompt contact-PII classifiers -------------------------------------
#
# These are deliberately NOT the body-copy patterns. A prompt is an
# instruction about to be rendered into pixels, so the decision has to hold
# in BOTH directions: contact data must fail, and ordinary numeric/textual
# description (dates, times, counts, dimensions) must pass. Rounds 3-5 all
# broke on one side or the other of exactly this, so the shape is now:
# candidate -> reject known non-contact shapes -> require contact evidence.

# Address-shaped token, script-independent: closes internationalized email
# (unicode local parts, IDN domains) that an ASCII pattern misses.
_ANY_EMAIL_RE = re.compile(r"[^\s@,;:()<>\[\]]+@[^\s@,;:()<>\[\]]+\.[^\s@,;:()<>\[\]]{2,}")

# Digit sequences with phone-ish separators, then filtered below.
_DIGIT_SEQ_RE = re.compile(r"[+\d][\d\s().\-\u2010-\u2015]{5,}\d")
# Shapes that are NOT contact data however many digits they carry.
_NON_CONTACT_SHAPES = (
    re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$"),      # 2026-07-25
    re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"),     # 07/25/2026
    re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$"),            # 9:45
)
# Vanity numbers carry letters, so digit counting alone never sees them.
_VANITY_PHONE_RE = re.compile(r"\b1[\s.\-]?[89]\d{2}[\s.\-]?[A-Za-z][A-Za-z0-9]{5,}\b")


def _prompt_contact_hits(text: str) -> list[str]:
    """Contact-PII findings for renderer instructions, both directions."""
    hits: list[str] = []
    if _ANY_EMAIL_RE.search(text):
        hits.append("email: <redacted>")
    if _VANITY_PHONE_RE.search(text):
        hits.append("phone: <redacted>")
        return hits
    for match in _DIGIT_SEQ_RE.finditer(text):
        token = match.group(0).strip()
        if any(shape.match(token) for shape in _NON_CONTACT_SHAPES):
            continue
        digits = re.sub(r"\D", "", token)
        # E.164 allows up to 15; below 7 is not a dialable number.
        if 7 <= len(digits) <= 15:
            hits.append("phone: <redacted>")
            break
    return hits

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
    # Normalize to v2: the runner-persisted audit always carries the advisory
    # checklist field; v1 stays frozen for pre-existing artifacts and direct
    # writers (rollback-safe -- see the contracts module). Only an original
    # v1 reply gets its version synthesized -- a v2-tagged reply keeps its
    # own schema_version so the Literal[2] validator rejects contradictory
    # worker metadata instead of the runner laundering it.
    if artifact.get("schema") == "editorial_audit.v1":
        artifact["schema_version"] = 2
    artifact["schema"] = "editorial_audit.v2"
    artifact.setdefault("schema_version", 2)
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
    _enforce_lineage(artifact, job_id, root)
    return write_artifact(job_id, stage, artifact, root=root)
