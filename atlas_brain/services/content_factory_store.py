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

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from atlas_brain.schemas.content_factory import model_for

DEFAULT_ROOT = Path.home() / "content-factory"

# A job_id / stage must be a single safe path segment: alphanumeric start, then
# only [A-Za-z0-9._-], and never a ".." sequence. This is the path-traversal
# choke point -- both job_id and stage flow through it before any filesystem use.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_GIT_NAME = "Content Factory"
_GIT_EMAIL = "content-factory@local"


class ArtifactStoreError(ValueError):
    """Raised for an unsafe path segment or a git failure in the store."""


def _safe_segment(value: str, kind: str) -> str:
    if not isinstance(value, str) or ".." in value or not _SAFE_SEGMENT.match(value):
        raise ArtifactStoreError(f"unsafe {kind}: {value!r}")
    return value


def job_dir(job_id: str, *, root: Path | str = DEFAULT_ROOT) -> Path:
    """Return the (guarded) folder that holds a job's artifacts."""
    return Path(root) / "jobs" / _safe_segment(job_id, "job_id")


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
    if not (job / ".git").exists():
        _git(job, "init", "-q")
        _git(job, "config", "user.email", _GIT_EMAIL)
        _git(job, "config", "user.name", _GIT_NAME)


def write_artifact(
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

    # Fail closed: validate before writing. model_for dispatches by the artifact's
    # "schema" tag (ValueError if absent/unknown); model_validate enforces the
    # contract invariants (non-empty citations, required tag, no self-promote, ...).
    model = model_for(artifact).model_validate(artifact)
    canonical = model.model_dump(mode="json")  # serialize_by_alias -> "schema" key

    job.mkdir(parents=True, exist_ok=True)
    _ensure_repo(job)
    path = job / f"{stage}.json"
    path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False) + "\n")

    _git(job, "add", "--", path.name)
    # Skip an empty commit when re-writing byte-identical content.
    unchanged = (
        subprocess.run(
            ["git", "-C", str(job), "diff", "--cached", "--quiet"]
        ).returncode
        == 0
    )
    if not unchanged:
        tag = canonical.get("schema", stage)
        _git(job, "commit", "-q", "-m", f"{stage}: {tag}")
    sha = _git(job, "rev-parse", "HEAD").stdout.strip()

    return {
        "job_id": job_id,
        "stage": stage,
        "schema": canonical.get("schema"),
        "path": str(path),
        "sha": sha,
    }
