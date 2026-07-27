"""Private, source-bound execution receipts for EOM operator tools."""

from __future__ import annotations

import hashlib
import io
import importlib.machinery
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

SCHEMA_VERSION = 1
TOOL_MODES = {
    "import_eom_customers_live": {"dry-run", "write"},
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
_REVIEWED_ENTRYPOINTS = {"scripts/import_eom_customers_live.py"}
_GIT_NO_REPLACE_ENV = "GIT_NO_REPLACE_OBJECTS"
_GIT_SANITIZED_ENV_NAMES = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_CONFIG_COUNT",
}
_GIT_SANITIZED_ENV_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
_GIT_CONFIG_OVERRIDES = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
)
_PYTHON_IMPORT_SUFFIXES = tuple(
    sorted(
        {
            *importlib.machinery.SOURCE_SUFFIXES,
            *importlib.machinery.BYTECODE_SUFFIXES,
            *importlib.machinery.EXTENSION_SUFFIXES,
        },
        key=len,
        reverse=True,
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_env() -> dict[str, str]:
    """Return a Git environment pinned to this cwd and non-executable config."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _GIT_SANITIZED_ENV_NAMES
        and not key.startswith(_GIT_SANITIZED_ENV_PREFIXES)
    }
    env.update(
        {
            _GIT_NO_REPLACE_ENV: "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "0",
        }
    )
    return env


def _git_command(*arguments: str) -> list[str]:
    return ["git", *_GIT_CONFIG_OVERRIDES, *arguments]


def _validate_git_sha(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if (
        len(normalized) != 40
        or any(char not in "0123456789abcdef" for char in normalized)
    ):
        raise RuntimeError(f"{label} did not resolve to a full SHA")
    return normalized


def _git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        _git_command("rev-parse", "--verify", "HEAD^{commit}"),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return _validate_git_sha(result.stdout, "git HEAD")


def _module_path_is_importable(parts: tuple[str, ...]) -> bool:
    return bool(parts) and all(
        part.isidentifier() and not part.startswith(".") for part in parts
    )


def _ignored_path_can_supply_code(
    repo_root: Path, relative_path: str
) -> bool:
    """Return whether an ignored entry can supply code from an import root."""
    if not relative_path:
        return False
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return True
    is_symlink = (repo_root / path).is_symlink()

    artifact_stem = None
    for suffix in _PYTHON_IMPORT_SUFFIXES:
        if path.name.endswith(suffix):
            artifact_stem = path.name[: -len(suffix)]
            break
    if artifact_stem is None:
        return is_symlink and _module_path_is_importable(path.parts)

    module_parts = (*path.parts[:-1], artifact_stem)
    if not module_parts or any(not part for part in module_parts):
        return True
    return _module_path_is_importable(module_parts)


def _tracked_python_entries(
    repo_root: Path, revision: str
) -> list[tuple[str, str, str]]:
    """Return validated mode, blob, and path tuples for tracked Python."""
    tree = subprocess.run(
        _git_command("ls-tree", "-r", "-z", "--full-tree", revision),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    entries: list[tuple[str, str, str]] = []
    for record in tree.stdout.split("\0"):
        if not record:
            continue
        metadata, relative_path = record.split("\t", 1)
        if not relative_path.endswith(".py"):
            continue
        mode, object_type, object_id = metadata.split()
        if (
            object_type != "blob"
            or mode not in {"100644", "100755"}
            or "\n" in relative_path
        ):
            raise RuntimeError(
                "receipted execution requires regular tracked Python source"
            )
        entries.append((mode, object_id, relative_path))
    return entries


def _tracked_python_mismatch_error() -> RuntimeError:
    return RuntimeError(
        "receipted execution requires tracked Python source to match "
        "the reviewed Git revision"
    )


def _verify_tracked_python_matches_revision(
    repo_root: Path, revision: str
) -> None:
    """Compare tracked Python bytes and executable modes with one revision."""
    entries = _tracked_python_entries(repo_root, revision)

    for expected_mode, expected_hash, relative_path in entries:
        source = repo_root / relative_path
        try:
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise _tracked_python_mismatch_error() from exc
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise _tracked_python_mismatch_error()
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                source_bytes = handle.read()
        except OSError as exc:
            raise _tracked_python_mismatch_error() from exc
        finally:
            os.close(descriptor)
        header = f"blob {len(source_bytes)}\0".encode("ascii")
        actual_hash = hashlib.sha1(
            header + source_bytes, usedforsecurity=False
        ).hexdigest()
        actual_mode = "100755" if source_stat.st_mode & 0o111 else "100644"
        if (
            actual_mode != expected_mode
            or actual_hash != expected_hash
        ):
            raise _tracked_python_mismatch_error()


def _materialize_reviewed_python(
    repo_root: Path, git_sha: str, snapshot_root: Path
) -> None:
    """Write the validated revision's Python tree into a private snapshot."""
    entries = _tracked_python_entries(repo_root, git_sha)
    batch = subprocess.run(
        _git_command("cat-file", "--batch"),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        input="".join(
            f"{object_id}\n" for _mode, object_id, _path in entries
        ).encode("ascii"),
        capture_output=True,
        text=False,
    ).stdout
    stream = io.BytesIO(batch)
    files: list[Path] = []
    directories = {snapshot_root}
    for _mode, object_id, relative_path in entries:
        header = stream.readline().decode("ascii").rstrip("\n").split()
        if (
            len(header) != 3
            or header[0] != object_id
            or header[1] != "blob"
            or not header[2].isdigit()
        ):
            raise RuntimeError("could not read reviewed Python source from Git")
        source = stream.read(int(header[2]))
        if len(source) != int(header[2]) or stream.read(1) != b"\n":
            raise RuntimeError("could not read reviewed Python source from Git")
        relative = PurePosixPath(relative_path)
        destination = snapshot_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        directories.update(destination.parents)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source)
        files.append(destination)
    if stream.read():
        raise RuntimeError("unexpected reviewed Python source from Git")

    for path in files:
        path.chmod(0o400)
    for directory in sorted(
        (path for path in directories if path.is_relative_to(snapshot_root)),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)


def _remove_reviewed_python(snapshot_root: Path) -> None:
    """Restore owner permissions only long enough to remove the snapshot."""
    if not snapshot_root.exists():
        return
    for path in snapshot_root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    snapshot_root.chmod(0o700)
    shutil.rmtree(snapshot_root)


def _reject_git_replacement_refs(repo_root: Path) -> None:
    result = subprocess.run(
        _git_command("for-each-ref", "--format=%(refname)", "refs/replace"),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("receipted execution rejects Git replacement refs")


def _cleanup_reviewed_python(snapshot_root: Path) -> None:
    try:
        _remove_reviewed_python(snapshot_root)
    except BaseException as exc:
        print(
            f"warning: could not remove reviewed Python snapshot: {exc}",
            file=sys.stderr,
        )


def establish_source_trust(
    repo_root: Path, *, expected_git_sha: str | None = None
) -> str:
    """Validate checkout inputs before any repository-local import executes."""
    _reject_git_replacement_refs(repo_root)
    git_sha = _git_sha(repo_root)
    if expected_git_sha is not None:
        expected = _validate_git_sha(expected_git_sha, "reviewed Git SHA")
        if git_sha != expected:
            raise RuntimeError(
                "receipted execution requires the launcher and checkout "
                "to resolve the same Git SHA"
            )
    status = subprocess.run(
        _git_command(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("receipted execution requires a clean worktree")
    _verify_tracked_python_matches_revision(repo_root, git_sha)
    ignored = subprocess.run(
        _git_command(
            "ls-files", "-z", "--others", "--ignored", "--exclude-standard"
        ),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    for relative_path in ignored.stdout.split("\0"):
        if _ignored_path_can_supply_code(repo_root, relative_path):
            raise RuntimeError(
                "receipted execution rejects ignored Python import shadows"
            )
    tracked_python = subprocess.run(
        _git_command("ls-files", "-z", "--", "*.py"),
        cwd=repo_root,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    for relative_source in tracked_python.stdout.split("\0"):
        if not relative_source:
            continue
        source = repo_root / relative_source
        legacy_cache = source.with_suffix(f"{source.suffix}c")
        cache_dir = source.parent / "__pycache__"
        if legacy_cache.exists() or any(
            cache_dir.glob(f"{source.stem}.*.pyc")
        ):
            raise RuntimeError(
                "receipted execution rejects cached bytecode for tracked source"
            )
    return git_sha


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
        if not directory.exists():
            raise ValueError("receipt directory must already exist")
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
        self._receipt_dir_device = directory_stat.st_dev
        self._receipt_dir_inode = directory_stat.st_ino
        self._receipt_dir_owner = directory_stat.st_uid
        self._verify_hard_link_support()
        self.in_progress_path = directory / f"{stem}.in-progress.json"
        self._final_stem = stem
        self._finalized = False
        self._persistence_error: Exception | None = None
        self._payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "receipt_id": str(run_id),
            "tool": tool,
            "mode": mode,
            "started_at_utc": started,
            "ended_at_utc": None,
            "git_sha": git_sha or establish_source_trust(script.parent.parent),
            "script_sha256": _script_sha256(script),
            "exit_code": None,
            "outcome_counts": {},
            "changed_contact_ids": [],
        }
        self._changed_contact_ids: set[str] = set()
        self._assert_receipt_directory_current()
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

    def _assert_receipt_directory_current(self) -> None:
        try:
            current = self.receipt_dir.lstat()
        except OSError as exc:
            raise RuntimeError(
                "receipt directory changed after validation"
            ) from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_dev != self._receipt_dir_device
            or current.st_ino != self._receipt_dir_inode
            or current.st_uid != self._receipt_dir_owner
            or current.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise RuntimeError("receipt directory changed after validation")

    def _fsync_directory(self) -> None:
        self._assert_receipt_directory_current()
        try:
            directory_fd = os.open(self.receipt_dir, os.O_RDONLY)
        except OSError as exc:
            raise RuntimeError("could not open receipt directory for fsync") from exc
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _verify_hard_link_support(self) -> None:
        self._assert_receipt_directory_current()
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
        self._assert_receipt_directory_current()
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
        self._persist_recorded_evidence()

    def record_changed_contact_id(self, contact_id: str | uuid.UUID) -> None:
        normalized = str(uuid.UUID(str(contact_id)))
        if normalized in self._changed_contact_ids:
            return
        self._changed_contact_ids.add(normalized)
        self._payload["changed_contact_ids"] = sorted(self._changed_contact_ids)
        self._persist_recorded_evidence()

    def _persist_recorded_evidence(self) -> None:
        if self._persistence_error is not None:
            return
        try:
            self._persist_in_progress()
        except Exception as exc:
            self._persistence_error = exc

    def assert_healthy(self) -> None:
        """Fail before another mutation when evidence persistence has failed."""
        if self._persistence_error is not None:
            raise RuntimeError(
                "could not durably record reconciliation evidence"
            ) from self._persistence_error

    def final_path_for(self, exit_code: int) -> Path:
        return self.receipt_dir / f"{self._final_stem}.exit-{exit_code}.json"

    def finalize(self, exit_code: int) -> Path:
        if self._finalized:
            raise RuntimeError("receipt is already finalized")
        self.assert_healthy()
        self._assert_receipt_directory_current()
        if type(exit_code) is not int or exit_code < 0:
            raise ValueError("exit code must be a non-negative integer")

        final_path = self.final_path_for(exit_code)
        final_payload = {
            **self._payload,
            "ended_at_utc": _utc_now(),
            "exit_code": exit_code,
            "changed_contact_ids": sorted(self._changed_contact_ids),
        }
        staged_path = self.in_progress_path.with_name(
            f"{self.in_progress_path.name}.{uuid.uuid4()}.tmp"
        )
        linked = False
        try:
            self._write_exclusive(staged_path, final_payload)
            os.link(staged_path, final_path)
            linked = True
            self._fsync_directory()
        except BaseException:
            if linked:
                final_path.unlink(missing_ok=True)
            staged_path.unlink(missing_ok=True)
            if linked:
                self._fsync_directory()
            raise

        self._payload = final_payload
        self._finalized = True
        try:
            self.in_progress_path.unlink()
            staged_path.unlink()
            self._fsync_directory()
        except BaseException:
            # The final link has already been published and directory-synced.
            # Post-publication cleanup must not let an interrupt rewrite the
            # observed process status away from the committed receipt outcome.
            return final_path
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


def _launch_reviewed_entrypoint(argv: list[str]) -> None:
    """Validate the checkout, then execute an allowlisted entrypoint revision."""
    if not sys.flags.isolated:
        raise SystemExit(
            "reviewed EOM execution requires isolated Python startup"
        )
    if (
        len(argv) < 4
        or argv[0] != "--launch-reviewed"
        or argv[1] != "--reviewed-git-sha"
    ):
        raise SystemExit(
            "usage: python -I - --launch-reviewed "
            "--reviewed-git-sha SHA "
            "scripts/import_eom_customers_live.py [arguments]"
        )
    reviewed_git_sha = _validate_git_sha(argv[2], "reviewed Git SHA")
    relative_entrypoint = argv[3]
    if relative_entrypoint not in _REVIEWED_ENTRYPOINTS:
        raise SystemExit("unsupported reviewed EOM entrypoint")

    repo_root = Path.cwd().resolve()
    git_sha = establish_source_trust(
        repo_root, expected_git_sha=reviewed_git_sha
    )
    snapshot_root = Path(
        tempfile.mkdtemp(prefix="atlas-eom-reviewed-python-")
    )
    try:
        _materialize_reviewed_python(repo_root, git_sha, snapshot_root)
        source_path = snapshot_root / relative_entrypoint
        source = source_path.read_bytes()

        bootstrap_module = sys.modules[__name__]
        bootstrap_module._BOOTSTRAP_ENTRYPOINT = relative_entrypoint
        bootstrap_module._BOOTSTRAP_GIT_SHA = git_sha
        bootstrap_module._BOOTSTRAP_REPO_ROOT = str(snapshot_root)
        sys.modules["eom_execution_receipt"] = bootstrap_module
        sys.argv = [str(source_path), *argv[4:]]
        namespace = {
            "__name__": "__main__",
            "__file__": str(source_path),
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(source, str(source_path), "exec"), namespace)
    finally:
        _cleanup_reviewed_python(snapshot_root)


if __name__ == "__main__":
    _launch_reviewed_entrypoint(sys.argv[1:])
