#!/usr/bin/env python3
"""Gate Gitleaks baseline and ignore growth from a trusted base checkout."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from dataclasses import dataclass

BASELINE_PATH = "docs/security/gitleaks-baseline.json"
IGNORE_PATH = ".gitleaksignore"
DEFAULT_ROTATION_LABEL = "security-rotation"
ALLOWED_ROTATION_PATHS = {
    BASELINE_PATH,
    IGNORE_PATH,
    "HARDENING.md",
    "docs/SECURITY_GUARDRAILS.md",
}
ALLOWED_ROTATION_PATTERNS = ("plans/PR-*.md",)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    disallowed_paths: tuple[str, ...] = ()
    missing_fingerprints: tuple[str, ...] = ()
    added_ignore_fingerprints: tuple[str, ...] = ()


def parse_labels(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {label.strip() for label in raw.split(",") if label.strip()}


def parse_labels_json(raw: str | None) -> set[str]:
    if not raw:
        return set()
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(label, str) for label in parsed):
        raise ValueError("labels JSON must be an array of strings")
    return set(parsed)


def is_rotation_path_allowed(path: str) -> bool:
    return path in ALLOWED_ROTATION_PATHS or any(
        fnmatch.fnmatchcase(path, pattern) for pattern in ALLOWED_ROTATION_PATTERNS
    )


def evaluate_baseline_rotation(
    changed_paths: set[str],
    *,
    labels: set[str],
    base_has_baseline: bool,
    base_fingerprints: set[str] | None = None,
    candidate_fingerprints: set[str] | None = None,
    base_ignore_fingerprints: set[str] | None = None,
    candidate_ignore_fingerprints: set[str] | None = None,
    rotation_label: str = DEFAULT_ROTATION_LABEL,
) -> Decision:
    baseline_changed = BASELINE_PATH in changed_paths
    base_ignores = base_ignore_fingerprints or set()
    candidate_ignores = candidate_ignore_fingerprints or set()
    added_ignores = (
        candidate_ignores - base_ignores if IGNORE_PATH in changed_paths else set()
    )
    baseline_requires_rotation = baseline_changed and base_has_baseline
    ignore_requires_rotation = bool(added_ignores)

    if not baseline_requires_rotation and not ignore_requires_rotation:
        if baseline_changed and not base_has_baseline:
            return Decision(
                True,
                "No trusted-base Gitleaks baseline exists; initial adoption is allowed.",
            )
        return Decision(True, "Gitleaks baseline and ignore fingerprints unchanged.")

    if rotation_label not in labels:
        if ignore_requires_rotation:
            reason = (
                "Gitleaks ignore growth requires the "
                f"`{rotation_label}` PR label after provider credential "
                "rotation/revocation. Reviewed false-positive evidence is "
                "limited to exact scanner-generated baseline fingerprints and "
                "does not authorize ignore growth; see "
                "docs/SECURITY_GUARDRAILS.md."
            )
        else:
            reason = (
                "Gitleaks baseline changes require the "
                f"`{rotation_label}` PR label after either provider credential "
                "rotation/revocation or the documented reviewed-false-positive "
                "evidence in docs/SECURITY_GUARDRAILS.md."
            )
        return Decision(
            False,
            reason,
            added_ignore_fingerprints=tuple(sorted(added_ignores)),
        )

    disallowed = tuple(
        sorted(path for path in changed_paths if not is_rotation_path_allowed(path))
    )
    if disallowed:
        return Decision(
            False,
            "Gitleaks rotation PRs may only touch the baseline, ignore file, hardening/security docs, and their plan.",
            disallowed,
            added_ignore_fingerprints=tuple(sorted(added_ignores)),
        )

    missing = tuple(
        sorted((base_fingerprints or set()) - (candidate_fingerprints or set()))
        if baseline_changed
        else ()
    )
    if missing:
        return Decision(
            False,
            "Proposed Gitleaks baseline drops trusted-base fingerprints.",
            missing_fingerprints=missing,
            added_ignore_fingerprints=tuple(sorted(added_ignores)),
        )

    return Decision(
        True,
        f"Gitleaks baseline/ignore rotation accepted with `{rotation_label}` label.",
        added_ignore_fingerprints=tuple(sorted(added_ignores)),
    )


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=False, capture_output=True, text=True)


def ref_has_baseline(ref: str) -> bool:
    return ref_has_path(ref, BASELINE_PATH)


def ref_has_path(ref: str, path: str) -> bool:
    proc = run_git(["cat-file", "-e", f"{ref}:{path}"])
    return proc.returncode == 0


def read_baseline(ref: str) -> list[dict[str, object]] | None:
    proc = run_git(["show", f"{ref}:{BASELINE_PATH}"])
    if proc.returncode != 0:
        return None
    parsed = json.loads(proc.stdout)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError(f"{BASELINE_PATH} must be a JSON array of objects")
    return parsed


def baseline_fingerprints(ref: str) -> set[str]:
    entries = read_baseline(ref) or []
    fingerprints = set()
    for entry in entries:
        fingerprint = entry.get("Fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def parse_ignore_fingerprints(raw: str) -> set[str]:
    return {
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def ignore_fingerprints(ref: str) -> set[str]:
    proc = run_git(["show", f"{ref}:{IGNORE_PATH}"])
    if proc.returncode != 0:
        return set()
    return parse_ignore_fingerprints(proc.stdout)


def changed_paths(base_ref: str, head_ref: str = "HEAD") -> set[str]:
    proc = run_git(["diff", "--name-only", f"{base_ref}...{head_ref}", "--"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git diff failed")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="Trusted base ref, for example origin/main.")
    parser.add_argument("--head-ref", default="HEAD", help="Candidate PR head ref. Defaults to HEAD.")
    parser.add_argument(
        "--labels",
        default="",
        help="Deprecated comma-separated pull request labels. Prefer --labels-json.",
    )
    parser.add_argument(
        "--labels-json",
        default="",
        help="JSON array of pull request label names from GitHub Actions toJson().",
    )
    parser.add_argument(
        "--rotation-label",
        default=DEFAULT_ROTATION_LABEL,
        help="PR label required to allow a controlled baseline rotation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        paths = changed_paths(args.base_ref, args.head_ref)
        labels = (
            parse_labels_json(args.labels_json)
            if args.labels_json
            else parse_labels(args.labels)
        )
        base_fps = (
            baseline_fingerprints(args.base_ref)
            if ref_has_baseline(args.base_ref)
            else set()
        )
        candidate_fps = (
            baseline_fingerprints(args.head_ref)
            if BASELINE_PATH in paths and ref_has_baseline(args.head_ref)
            else set()
        )
        base_ignore_fps = ignore_fingerprints(args.base_ref)
        candidate_ignore_fps = (
            ignore_fingerprints(args.head_ref)
            if IGNORE_PATH in paths and ref_has_path(args.head_ref, IGNORE_PATH)
            else set()
        )
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    decision = evaluate_baseline_rotation(
        paths,
        labels=labels,
        base_has_baseline=ref_has_baseline(args.base_ref),
        base_fingerprints=base_fps,
        candidate_fingerprints=candidate_fps,
        base_ignore_fingerprints=base_ignore_fps,
        candidate_ignore_fingerprints=candidate_ignore_fps,
        rotation_label=args.rotation_label,
    )
    print(decision.reason)
    if decision.disallowed_paths:
        print("Disallowed files:")
        for path in decision.disallowed_paths:
            print(f"- {path}")
    if decision.missing_fingerprints:
        print("Missing trusted-base fingerprints:")
        for fingerprint in decision.missing_fingerprints:
            print(f"- {fingerprint}")
    if decision.added_ignore_fingerprints:
        print("Added Gitleaks ignore fingerprints:")
        for fingerprint in decision.added_ignore_fingerprints:
            print(f"- {fingerprint}")
    return 0 if decision.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
