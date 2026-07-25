"""Private, source-bound execution receipts for EOM operator tools."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA_VERSION = 1
TOOL_MODES = {
    "import_eom_customers_live": {"dry-run", "write"},
    "sync_eom_portal_customers": {"dry-run", "apply"},
}
OUTCOME_KEYS = {
    "created",
    "updated",
    "unchanged",
    "skipped",
    "errors",
    "create-planned",
    "update-planned",
    "import-planned",
}
PORTAL_TOTAL_KEYS = {"demoted", "eligible", "kept"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError("git HEAD did not resolve to a full SHA")
    return value


def _clean_git_sha(repo_root: Path) -> str:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("receipted execution requires a clean worktree")
    return _git_sha(repo_root)


def _script_sha256(script_path: Path) -> str:
    return hashlib.sha256(script_path.read_bytes()).hexdigest()


def _validated_counts(
    values: Mapping[str, int], allowed: set[str], label: str
) -> dict[str, int]:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {label} keys: {sorted(unknown)}")
    normalized: dict[str, int] = {}
    for key, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} values must be non-negative integers")
        normalized[key] = value
    return dict(sorted(normalized.items()))


def _exit_code_for_exception(exc: BaseException) -> int:
    if isinstance(exc, KeyboardInterrupt):
        return 130
    if isinstance(exc, SystemExit):
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return int(exc.code) & 0xFF
        return 1
    return 1


class EomExecutionReceipt:
    """One exclusive in-progress artifact and its atomic final publication."""

    def __init__(
        self,
        *,
        receipt_dir: str | os.PathLike[str],
        tool: str,
        mode: str,
        script_path: str | os.PathLike[str],
        receipt_id: uuid.UUID | None = None,
        started_at_utc: str | None = None,
        git_sha: str | None = None,
    ) -> None:
        if tool not in TOOL_MODES or mode not in TOOL_MODES[tool]:
            raise ValueError("unsupported EOM receipt tool or mode")

        directory = Path(receipt_dir).expanduser()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_stat = directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(
            directory_stat.st_mode
        ):
            raise ValueError("receipt directory must be a real directory")
        if directory_stat.st_uid != os.geteuid():
            raise ValueError("receipt directory must be owned by the current user")
        if directory_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("receipt directory must not be writable by other users")

        script = Path(script_path).resolve()
        run_id = receipt_id or uuid.uuid4()
        started = started_at_utc or _utc_now()
        safe_started = (
            started.replace("-", "").replace(":", "").replace(".", "")
            .replace("+", "").replace("Z", "Z")
        )
        stem = f"{safe_started}_{tool}_{run_id}"

        self.receipt_dir = directory
        self._verify_hard_link_support()
        self.in_progress_path = directory / f"{stem}.in-progress.json"
        self._final_stem = stem
        self._finalized = False
        self._payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": str(run_id),
            "tool": tool,
            "mode": mode,
            "started_at_utc": started,
            "ended_at_utc": None,
            "git_sha": git_sha or _clean_git_sha(script.parent.parent),
            "script_sha256": _script_sha256(script),
            "exit_code": None,
            "outcome_counts": {},
            "portal_totals": None,
            "changed_contact_ids": [],
        }
        self._changed_contact_ids: set[str] = set()
        self._write_exclusive(self.in_progress_path, self._payload)
        self._fsync_directory()

    @staticmethod
    def _serialized(payload: Mapping[str, Any]) -> bytes:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    @classmethod
    def _write_exclusive(cls, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except OSError as exc:
            raise RuntimeError(f"could not create receipt artifact: {path.name}") from exc
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(cls._serialized(payload))
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)

    def _fsync_directory(self) -> None:
        try:
            directory_fd = os.open(self.receipt_dir, os.O_RDONLY)
        except OSError as exc:
            raise RuntimeError("could not open receipt directory for fsync") from exc
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _verify_hard_link_support(self) -> None:
        probe_id = uuid.uuid4()
        source = self.receipt_dir / f".eom-receipt-link-probe-{probe_id}.source"
        target = self.receipt_dir / f".eom-receipt-link-probe-{probe_id}.target"
        try:
            self._write_exclusive(source, {"probe": True})
            try:
                os.link(source, target)
            except OSError as exc:
                raise ValueError(
                    "receipt directory filesystem must support hard links"
                ) from exc
        finally:
            target.unlink(missing_ok=True)
            source.unlink(missing_ok=True)
            self._fsync_directory()

    def _persist_in_progress(self) -> None:
        staged_path = self.in_progress_path.with_name(
            f"{self.in_progress_path.name}.{uuid.uuid4()}.tmp"
        )
        try:
            self._write_exclusive(staged_path, self._payload)
            os.replace(staged_path, self.in_progress_path)
            self._fsync_directory()
        finally:
            staged_path.unlink(missing_ok=True)

    def set_outcome_counts(self, counts: Mapping[str, int]) -> None:
        self._payload["outcome_counts"] = _validated_counts(
            counts, OUTCOME_KEYS, "outcome count"
        )
        self._persist_in_progress()

    def set_portal_totals(self, totals: Mapping[str, int]) -> None:
        self._payload["portal_totals"] = _validated_counts(
            totals, PORTAL_TOTAL_KEYS, "portal total"
        )
        self._persist_in_progress()

    def record_changed_contact_id(self, contact_id: str | uuid.UUID) -> None:
        normalized = str(uuid.UUID(str(contact_id)))
        if normalized in self._changed_contact_ids:
            return
        self._changed_contact_ids.add(normalized)
        self._payload["changed_contact_ids"] = sorted(self._changed_contact_ids)
        self._persist_in_progress()

    def final_path_for(self, exit_code: int) -> Path:
        return self.receipt_dir / f"{self._final_stem}.exit-{exit_code}.json"

    def finalize(self, exit_code: int) -> Path:
        if self._finalized:
            raise RuntimeError("receipt is already finalized")
        if type(exit_code) is not int or exit_code < 0:
            raise ValueError("exit code must be a non-negative integer")

        self._payload["ended_at_utc"] = _utc_now()
        self._payload["exit_code"] = exit_code
        self._payload["changed_contact_ids"] = sorted(self._changed_contact_ids)

        self._persist_in_progress()

        final_path = self.final_path_for(exit_code)
        os.link(self.in_progress_path, final_path)
        self.in_progress_path.unlink()
        self._fsync_directory()
        self._finalized = True
        return final_path


def run_receipted(
    receipt: EomExecutionReceipt | None, operation: Callable[[], int]
) -> int:
    """Run one CLI operation and finalize its receipt for every exit path."""
    try:
        exit_code = operation()
    except BaseException as exc:
        if receipt is not None:
            try:
                receipt.finalize(_exit_code_for_exception(exc))
            except Exception as receipt_error:
                raise receipt_error from exc
        raise
    if receipt is not None:
        receipt.finalize(exit_code)
    return exit_code
