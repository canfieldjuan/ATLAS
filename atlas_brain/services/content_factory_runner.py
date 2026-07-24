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
    advisory_warnings,
    verify_copy,
)
from atlas_brain.services.content_factory_store import DEFAULT_ROOT, write_artifact

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
    # writers (rollback-safe -- see the contracts module).
    artifact["schema"] = "editorial_audit.v2"
    artifact["schema_version"] = 2
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
    return write_artifact(job_id, stage, artifact, root=root)
