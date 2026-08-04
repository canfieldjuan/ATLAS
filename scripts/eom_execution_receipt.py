"""Private, source-bound execution receipts for EOM operator tools."""

from __future__ import annotations

import contextlib
import asyncio
import hashlib
import json
import os
import stat
import subprocess
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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


class IndeterminateMutation(RuntimeError):
    """Raised when a mutation may have committed but the process was interrupted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root_for(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_sha_for(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root_for(path),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"failed to hash receipt script: {path}") from exc
    return digest.hexdigest()


def _validated_counts(counts: Mapping[str, int]) -> dict[str, int]:
    clean: dict[str, int] = {}
    for key, value in counts.items():
        if key not in OUTCOME_KEYS:
            raise ValueError(f"unsupported outcome count: {key}")
        if type(value) is not int or value < 0:
            raise ValueError(f"outcome count {key} must be a non-negative int")
        clean[key] = value
    return clean


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise RuntimeError(f"failed to open receipt directory for fsync: {path}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_private_directory(path: Path) -> None:
    stat_result = path.stat()
    if stat_result.st_uid != os.getuid():
        raise ValueError(f"receipt directory is not owned by the current user: {path}")
    mode = stat.S_IMODE(stat_result.st_mode)
    if mode != 0o700:
        raise ValueError(f"receipt directory must be private mode 0700: {path}")


class EomExecutionReceipt:
    """Durable non-PII artifact for one EOM operator-tool invocation."""

    def __init__(
        self,
        *,
        receipt_dir: str | Path,
        tool: str,
        mode: str,
        script_path: str | Path,
        receipt_id: str | uuid.UUID | None = None,
    ) -> None:
        if mode not in TOOL_MODES.get(tool, set()):
            raise ValueError(f"unsupported EOM receipt tool/mode: {tool}/{mode}")
        self.receipt_dir = Path(receipt_dir).expanduser().resolve()
        created = False
        try:
            self.receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            created = True
        except FileExistsError:
            pass
        if not self.receipt_dir.is_dir():
            raise ValueError("receipt directory must be a directory")
        if created:
            self.receipt_dir.chmod(0o700)
        _assert_private_directory(self.receipt_dir)
        script = Path(script_path).resolve()
        self.receipt_id = str(uuid.UUID(str(receipt_id or uuid.uuid4())))
        self._changed_contact_ids: set[str] = set()
        self._finalized = False
        self._indeterminate = False
        self._payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "tool": tool,
            "mode": mode,
            "started_at_utc": _utc_now(),
            "git_sha": _git_sha_for(script),
            "script_hash_sha256": _sha256_file(script),
            "outcome_counts": {},
            "changed_contact_ids": [],
            "indeterminate": False,
        }
        self.in_progress_path = (
            self.receipt_dir / f".eom-{tool}-{self.receipt_id}.in-progress.json"
        )
        self._final_stem = f"eom-{tool}-{self.receipt_id}"
        self._persist_in_progress()

    def _assert_private_file(self, path: Path) -> None:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode != 0o600:
            raise RuntimeError(f"receipt file is not private: {path}")

    def _persist_in_progress(self) -> None:
        staged = self.in_progress_path.with_name(
            f"{self.in_progress_path.name}.{uuid.uuid4()}.tmp"
        )
        try:
            _write_json_exclusive(staged, self._payload)
            os.replace(staged, self.in_progress_path)
            self._assert_private_file(self.in_progress_path)
            _fsync_directory(self.receipt_dir)
        finally:
            staged.unlink(missing_ok=True)

    def record_outcome_counts(self, counts: Mapping[str, int]) -> None:
        self._payload["outcome_counts"] = _validated_counts(counts)
        self._persist_in_progress()

    def record_changed_contact_id(self, contact_id: str | uuid.UUID | None) -> None:
        if not contact_id:
            return
        normalized = str(uuid.UUID(str(contact_id)))
        if normalized in self._changed_contact_ids:
            return
        self._changed_contact_ids.add(normalized)
        self._payload["changed_contact_ids"] = sorted(self._changed_contact_ids)
        self._persist_in_progress()

    def record_demotions(self, *, demoted: int, eligible: int, kept: int) -> None:
        for label, value in {
            "demoted": demoted,
            "eligible": eligible,
            "kept": kept,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"demotion total {label} must be a non-negative int")
        self._payload["demotion_totals"] = {
            "demoted": demoted,
            "eligible": eligible,
            "kept": kept,
        }
        self._persist_in_progress()

    @contextlib.asynccontextmanager
    async def mutation_boundary(self) -> AsyncIterator[None]:
        """Mark the receipt indeterminate if cancellation interrupts a mutation."""
        try:
            yield
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            self._indeterminate = True
            self._payload["indeterminate"] = True
            self._persist_in_progress()
            raise

    def final_path_for(self, exit_code: int) -> Path:
        return self.receipt_dir / f"{self._final_stem}.exit-{exit_code}.json"

    def finalize(self, exit_code: int) -> Path:
        if self._finalized:
            raise RuntimeError("receipt is already finalized")
        if self._indeterminate:
            raise IndeterminateMutation(
                "mutation outcome is indeterminate; leaving in-progress receipt"
            )
        if type(exit_code) is not int or exit_code < 0:
            raise ValueError("exit code must be a non-negative int")
        final_payload = {
            **self._payload,
            "ended_at_utc": _utc_now(),
            "exit_code": exit_code,
            "changed_contact_ids": sorted(self._changed_contact_ids),
        }
        final_path = self.final_path_for(exit_code)
        staged = self.in_progress_path.with_name(
            f"{self.in_progress_path.name}.{uuid.uuid4()}.tmp"
        )
        linked = False
        try:
            _write_json_exclusive(staged, final_payload)
            os.link(staged, final_path)
            linked = True
            self._assert_private_file(final_path)
            _fsync_directory(self.receipt_dir)
        except BaseException:
            if linked:
                final_path.unlink(missing_ok=True)
                _fsync_directory(self.receipt_dir)
            raise
        finally:
            staged.unlink(missing_ok=True)
        self._payload = final_payload
        self._finalized = True
        self.in_progress_path.unlink(missing_ok=True)
        _fsync_directory(self.receipt_dir)
        return final_path


def exit_code_for_exception(exc: BaseException) -> int:
    code = getattr(exc, "code", None)
    if type(code) is int and code >= 0:
        return code
    return 130 if isinstance(exc, KeyboardInterrupt) else 1


def run_receipted(receipt: EomExecutionReceipt | None, operation) -> int:
    try:
        exit_code = operation()
    except BaseException as exc:
        if receipt is not None:
            try:
                receipt.finalize(exit_code_for_exception(exc))
            except IndeterminateMutation as indeterminate:
                indeterminate.add_note("left in-progress receipt for operator review")
        raise
    if receipt is not None:
        receipt.finalize(exit_code)
    return exit_code
