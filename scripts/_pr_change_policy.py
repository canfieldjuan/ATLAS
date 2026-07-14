"""Shared changed-path policy for PR admission checks.

The helpers in this module inspect a repository as data. They deliberately do
not read PR body text or execute code from the inspected tree, so trusted-base
review scripts can reuse the same classification as local wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess


DEPENDABOT_AUTHORS = frozenset(
    {
        "app/dependabot",
        "dependabot",
        "dependabot[bot]",
    }
)
PLAN_PREFIX = "plans/PR-"
PLAN_SUFFIX = ".md"


class ChangeKind(str, Enum):
    DEPENDABOT = "dependabot"
    DOCS_ONLY = "docs-only"
    NO_CHANGES = "no-changes"
    PLAN_REQUIRED = "plan-required"


class ChangePolicyError(RuntimeError):
    """The changed-path policy could not be evaluated safely."""


@dataclass(frozen=True)
class ChangeClassification:
    kind: ChangeKind
    paths: tuple[str, ...]


def is_dependabot_author(author: str | None) -> bool:
    """Return true for Dependabot identities seen in GitHub PR events."""

    return author is not None and author.strip() in DEPENDABOT_AUTHORS


def _git_stdout(args: list[str], *, repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ChangePolicyError(f"could not run git: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or "git command failed"
        raise ChangePolicyError(detail)
    return proc.stdout


def _merge_base(base_ref: str, *, head_ref: str, repo_root: Path) -> str:
    try:
        _git_stdout(["rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}"], repo_root=repo_root)
    except ChangePolicyError as exc:
        raise ChangePolicyError(f"base ref not found: {base_ref}") from exc
    try:
        _git_stdout(["rev-parse", "--verify", "--quiet", f"{head_ref}^{{commit}}"], repo_root=repo_root)
    except ChangePolicyError as exc:
        raise ChangePolicyError(f"head ref not found: {head_ref}") from exc
    return _git_stdout(["merge-base", head_ref, base_ref], repo_root=repo_root).strip()


def changed_paths(
    base_ref: str,
    *,
    head_ref: str = "HEAD",
    repo_root: Path,
) -> tuple[str, ...]:
    """Return every path changed from the merge base through ``head_ref``."""

    base = _merge_base(base_ref, head_ref=head_ref, repo_root=repo_root)
    payload = _git_stdout(
        ["diff", "--name-only", "-z", "--no-renames", f"{base}...{head_ref}"],
        repo_root=repo_root,
    )
    return tuple(sorted(path for path in payload.split("\0") if path))


def classify_changes(
    *,
    author: str | None,
    base_ref: str,
    head_ref: str = "HEAD",
    repo_root: Path,
) -> ChangeClassification:
    """Classify a PR diff for plan admission without silently guessing."""

    if is_dependabot_author(author):
        return ChangeClassification(kind=ChangeKind.DEPENDABOT, paths=())

    paths = changed_paths(base_ref, head_ref=head_ref, repo_root=repo_root)
    if not paths:
        return ChangeClassification(kind=ChangeKind.NO_CHANGES, paths=paths)
    if all(
        _is_regular_markdown_blob(path, head_ref=head_ref, repo_root=repo_root)
        for path in paths
    ):
        return ChangeClassification(kind=ChangeKind.DOCS_ONLY, paths=paths)
    return ChangeClassification(kind=ChangeKind.PLAN_REQUIRED, paths=paths)


def _is_regular_markdown_blob(path: str, *, head_ref: str, repo_root: Path) -> bool:
    """Return true only for a regular blob with ``.md`` as its sole suffix.

    Documentation-only admission is an exemption, so its proof must be as
    strict as the branch-plan side: a symlink, deleted path, or compound name
    such as ``install.sh.md`` is plan-required rather than silently exempt.
    """

    return Path(path).suffixes == [".md"] and _is_regular_blob(
        path, head_ref=head_ref, repo_root=repo_root
    )


def _is_regular_blob(path: str, *, head_ref: str, repo_root: Path) -> bool:
    """Return true only when ``path`` is a regular blob at ``head_ref``."""

    entry = _git_stdout(["ls-tree", head_ref, "--", path], repo_root=repo_root).strip()
    return entry.startswith("100644 blob") or entry.startswith("100755 blob")


def branch_added_plan_docs(
    base_ref: str,
    *,
    head_ref: str = "HEAD",
    repo_root: Path,
) -> tuple[str, ...]:
    """Return regular plan docs added by this branch, relative to its base."""

    base = _merge_base(base_ref, head_ref=head_ref, repo_root=repo_root)
    payload = _git_stdout(
        [
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=A",
            f"{base}...{head_ref}",
            "--",
            "plans/PR-*.md",
        ],
        repo_root=repo_root,
    )
    candidates = sorted(path for path in payload.split("\0") if path)
    regular: list[str] = []
    for path in candidates:
        if not _is_plan_path(path):
            continue
        if _is_regular_blob(path, head_ref=head_ref, repo_root=repo_root):
            regular.append(path)
    return tuple(regular)


def _is_plan_path(path: str) -> bool:
    return (
        path.startswith(PLAN_PREFIX)
        and path.endswith(PLAN_SUFFIX)
        and "/" not in path[len(PLAN_PREFIX):]
    )
