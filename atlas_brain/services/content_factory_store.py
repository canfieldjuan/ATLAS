"""Content Factory artifact store: validate a stage output against its
content_factory contract and persist it to a git-tracked job folder.

The behavior lives here (in atlas_brain, next to the contracts) rather than in an
Open WebUI Function, because OWUI runs in its own Python environment and cannot
import atlas_brain -- vendoring the contracts into an OWUI function would drift
from the source of truth. A thin OWUI Action / API caller (a later slice) invokes
this service.

Layout written:  <root>/jobs/<job_id>/<stage>.json
Each job folder is its own git repo, so every stage write is an auditable commit.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from atlas_brain.schemas.content_factory import model_for

DEFAULT_ROOT = Path.home() / "content-factory"

# A job_id / stage must be a single safe path segment: alphanumeric start, then
# only [A-Za-z0-9._-], and never a ".." sequence. This is the path-traversal
# choke point -- both job_id and stage flow through it before any filesystem use.
# Matched with fullmatch (not $, which in Python matches before a trailing
# newline, so "brief\n" would otherwise slip through into a filename).
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Canonical stage -> version tag. A known stage must carry its matching artifact,
# so a mislabeled write (e.g. a draft under the "brief" stage) is rejected. An
# unknown/custom stage is allowed to carry any artifact.
STAGE_SCHEMAS = {
    "brief": ("content_brief.v1",),
    "evidence": ("evidence_packet.v1",),
    "draft": ("draft.v1",),
    # v1 stays admissible: pre-#2136 artifacts and any direct writer keep
    # working; runner-persisted audits are normalized to v2.
    "audit": ("editorial_audit.v1", "editorial_audit.v2", "editorial_audit.v3"),
    "manifest": ("manifest.v1",),
    # Phase 6: channel variants and image prompts derived from an approved
    # draft. Both are gated by the runner exactly like the audit.
    "repurposing": ("repurposing.v1",),
    "image_prompt": ("image_prompt.v1",),
}

_GIT_NAME = "Content Factory"
_GIT_EMAIL = "content-factory@local"


class ArtifactStoreError(ValueError):
    """Raised for an unsafe path segment or a git failure in the store."""


def _safe_segment(value: str, kind: str) -> str:
    if not isinstance(value, str) or ".." in value or not _SAFE_SEGMENT.fullmatch(value):
        raise ArtifactStoreError(f"unsafe {kind}: {value!r}")
    return value


def job_dir(job_id: str, *, root: Path | str = DEFAULT_ROOT) -> Path:
    """Return the (guarded) folder that holds a job's artifacts."""
    return Path(root) / "jobs" / _safe_segment(job_id, "job_id")


_LOCK_STATE = threading.local()


@contextmanager
def job_lock(job_id: str, *, root: Path | str = DEFAULT_ROOT) -> Iterator[None]:
    """Exclusive per-job lock covering a read-validate-write sequence.

    Readiness enforcement READS draft.json and audit.json, decides, and only
    then persists. Without mutual exclusion another stage run can replace the
    draft inside that window, so a Phase 6 artifact lands as ready beside copy
    its approving audit never covered: the fingerprint was checked against
    content that no longer exists at write time (#2192 round 8). Validating
    before writing is not enough when the thing validated against can move.

    Re-entrant within a thread, so ``run_stage`` holds it across enforcement
    while ``write_artifact`` takes it again without deadlocking; ``flock``
    extends the exclusion across processes. The lock file lives OUTSIDE the
    job folder so it never lands in the job's git history.
    """
    safe = _safe_segment(job_id, "job_id")
    depth = getattr(_LOCK_STATE, "depth", None)
    if depth is None:
        depth = _LOCK_STATE.depth = {}
    if depth.get(safe):
        depth[safe] += 1
        try:
            yield
        finally:
            depth[safe] -= 1
        return

    locks = Path(root) / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    handle = open(locks / f"{safe}.lock", "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        depth[safe] = 1
        yield
    finally:
        depth[safe] = 0
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _git(job: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(job), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # git not installed
        raise ArtifactStoreError("git is not available") from exc
    except subprocess.CalledProcessError as exc:
        raise ArtifactStoreError(
            f"git {' '.join(args)} failed: {(exc.stderr or '').strip()}"
        ) from exc


def _ensure_repo(job: Path) -> None:
    # Init if needed, then isolate the private job repo from inherited global git
    # config on EVERY call (not only at creation, so an externally-initialized job
    # folder is isolated too): a local author, signing pinned off, and hooks
    # neutralized. Otherwise an inherited global commit.gpgsign=true, a global
    # core.hooksPath, or a template hook (all common on dev machines) can make the
    # store's commit fail before the valid artifact is committed. This is local
    # repo config on an isolated store repo, not a --no-verify/--no-gpg-sign
    # bypass of any Atlas gate.
    if not (job / ".git").exists():
        _git(job, "init", "-q")
    _git(job, "config", "user.email", _GIT_EMAIL)
    _git(job, "config", "user.name", _GIT_NAME)
    _git(job, "config", "commit.gpgsign", "false")
    _git(job, "config", "core.hooksPath", os.devnull)


def write_artifact(
    job_id: str,
    stage: str,
    artifact: dict[str, Any],
    *,
    root: Path | str = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Persist a stage artifact under the job's exclusive lock.

    See ``_write_artifact_locked`` for the validation and commit behavior. The
    lock is re-entrant, so a caller that already holds it (``run_stage``, which
    must cover its read-validate-write window) is not blocked by this one.
    """
    with job_lock(job_id, root=root):
        return _write_artifact_locked(job_id, stage, artifact, root=root)


def _write_artifact_locked(
    job_id: str,
    stage: str,
    artifact: dict[str, Any],
    *,
    root: Path | str = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Validate ``artifact`` against its content_factory contract, persist the
    canonical form to ``<root>/jobs/<job_id>/<stage>.json``, and git-commit it.

    Returns a record: ``{job_id, stage, schema, path, sha}``.

    Raises ``ArtifactStoreError`` for an unsafe job_id/stage or a git failure, and
    ``ValueError`` / pydantic ``ValidationError`` for an artifact that fails its
    contract -- validation happens before any filesystem write, so a malformed
    stage output is never persisted.
    """
    stage = _safe_segment(stage, "stage")
    job = job_dir(job_id, root=root)

    # Canonical form carries the version tag only under "schema"; a reserved
    # "artifact_schema" key is non-canonical and rejected here with a specific
    # error before validation (the contracts' extra='forbid' would also reject it).
    if "artifact_schema" in artifact:
        raise ArtifactStoreError(
            "non-canonical artifact: reserved 'artifact_schema' key present; "
            "supply the version tag only under 'schema'"
        )

    # Fail closed: validate before writing. model_for dispatches by the artifact's
    # "schema" tag (ValueError if absent/unknown); model_validate enforces the
    # contract invariants (non-empty citations, required tag, no self-promote, ...).
    model = model_for(artifact).model_validate(artifact)
    canonical = model.model_dump(mode="json")  # serialize_by_alias -> "schema" key
    tag = canonical.get("schema")

    # A known stage must carry its matching schema, so a mislabeled write cannot
    # land as a valid stage artifact.
    expected = STAGE_SCHEMAS.get(stage)
    if expected is not None and tag not in expected:
        raise ArtifactStoreError(
            f"stage/schema mismatch: stage {stage!r} expects one of {expected!r}, "
            f"got {tag!r}"
        )

    # The manifest is the job index; its own job_id must match the folder it lands
    # in, or it would claim to index a different job.
    if tag == "manifest.v1" and canonical.get("job_id") != job_id:
        raise ArtifactStoreError(
            f"manifest job_id {canonical.get('job_id')!r} does not match "
            f"path job_id {job_id!r}"
        )

    job.mkdir(parents=True, exist_ok=True)
    _ensure_repo(job)
    path = job / f"{stage}.json"
    path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False) + "\n")

    # Scope the staged-change check and the commit to THIS file, so an unrelated
    # pre-staged file in the job repo cannot ride into this stage's commit and an
    # identical rewrite still makes no empty commit.
    _git(job, "add", "--", path.name)
    unchanged = (
        subprocess.run(
            ["git", "-C", str(job), "diff", "--cached", "--quiet", "--", path.name]
        ).returncode
        == 0
    )
    if not unchanged:
        _git(job, "commit", "-q", "-m", f"{stage}: {tag}", "--", path.name)
    sha = _git(job, "rev-parse", "HEAD").stdout.strip()

    return {
        "job_id": job_id,
        "stage": stage,
        "schema": canonical.get("schema"),
        "path": str(path),
        "sha": sha,
    }
