#!/usr/bin/env python3
"""Fail closed when local PR watcher infrastructure can merge PRs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import sys
from typing import Sequence


REPO_DOCS = (
    "AGENTS.md",
    "CLAUDE.md",
    "docs/SESSION_STATE_TEMPLATE.md",
    "docs/ci_cd_autonomous_coding_map.md",
    "docs/long_running_agent_monitoring_spec.md",
    "docs/long_running_session_watcher_handoff.md",
    "docs/autonomous_coding_repo_playbook.md",
)

TRUTHY = {"1", "true", "yes", "on", "enabled"}
TRUTHY_PATTERN = "|".join(sorted(TRUTHY, key=len, reverse=True))
MERGE_SOURCE_PATTERNS = (
    re.compile(r"\bgh\s+pr\s+merge\b", re.IGNORECASE),
    re.compile(r"""["']pr["']\s*,\s*["']merge["']""", re.IGNORECASE),
    re.compile(r"--delete-branch\b", re.IGNORECASE),
)
DOC_AUTHORITY_PATTERNS = (
    re.compile(rf"\bAUTO_MERGE\s*=\s*[\"']?(?:{TRUTHY_PATTERN})[\"']?\b", re.IGNORECASE),
    re.compile(r"Auto-merge:\s*enabled", re.IGNORECASE),
    re.compile(
        r"(?:watcher|timer|notification|bridge|wake\s+bridge)\s+"
        r"(?:can|may|should|will)\s+merge",
        re.IGNORECASE,
    ),
    re.compile(r"Only with explicit standing authorization", re.IGNORECASE),
    re.compile(r"Merge allowed\?\s*\|\s*Yes", re.IGNORECASE),
    re.compile(
        r"^\|[^|\n]*(?:watcher|timer|notification|bridge|wake\s+bridge|green confirmation)[^|\n]*\|"
        r"\s*(?:Yes|Only with explicit standing authorization)\s*\|",
        re.IGNORECASE | re.MULTILINE,
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _find_patterns(path: Path, text: str, patterns: Sequence[re.Pattern[str]], message: str) -> list[Finding]:
    findings: list[Finding] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            findings.append(Finding(str(path), _line_number(text, match.start()), message))
    return findings


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in _read(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = re.sub(r"^export\s+", "", key.strip(), flags=re.IGNORECASE)
        try:
            lexer = shlex.shlex(value.strip(), posix=True)
            lexer.whitespace_split = True
            lexer.commenters = "#"
            parsed = next(iter(lexer), "")
        except ValueError:
            parsed = value.strip().strip("\"'")
        values[key] = parsed.strip()
    return values


def audit_watcher_source(path: Path) -> list[Finding]:
    if not path.exists():
        return []
    return _find_patterns(
        path,
        _read(path),
        MERGE_SOURCE_PATTERNS,
        "watcher executable must not contain PR merge/delete-branch commands",
    )


def audit_config(path: Path) -> list[Finding]:
    values = _parse_env(path)
    configured = values.get("AUTO_MERGE", "0").strip().lower()
    if configured in TRUTHY:
        return [Finding(str(path), 1, "watcher config must use AUTO_MERGE=0")]
    return []


def audit_configs(config_dir: Path) -> list[Finding]:
    if not config_dir.exists():
        return []
    findings: list[Finding] = []
    for path in sorted(config_dir.glob("*.env")):
        findings.extend(audit_config(path))
    return findings


def audit_systemd_units(systemd_dir: Path) -> list[Finding]:
    if not systemd_dir.exists():
        return []
    findings: list[Finding] = []
    for pattern in (
        "atlas-pr-watch*.service",
        "atlas-pr-watch*.timer",
        "atlas-pr-watch*.service.d/*.conf",
        "atlas-pr-watch*.timer.d/*.conf",
    ):
        for path in sorted(systemd_dir.glob(pattern)):
            findings.extend(audit_watcher_source(path))
    return findings


def audit_repo_docs(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel in REPO_DOCS:
        path = repo_root / rel
        if not path.exists():
            continue
        findings.extend(
            _find_patterns(
                path,
                _read(path),
                DOC_AUTHORITY_PATTERNS,
                "repo docs/templates must not grant merge authority to the watcher",
            )
        )
    return findings


def build_findings(
    repo_root: Path,
    *,
    watcher_bin: Path,
    watcher_wrapper: Path,
    config_dir: Path,
    systemd_dir: Path,
    repo_only: bool,
) -> list[Finding]:
    findings = audit_repo_docs(repo_root)
    if not repo_only:
        findings.extend(audit_watcher_source(watcher_bin))
        findings.extend(audit_watcher_source(watcher_wrapper))
        findings.extend(audit_configs(config_dir))
        findings.extend(audit_systemd_units(systemd_dir))
    return findings


def render(findings: Sequence[Finding], *, repo_only: bool) -> None:
    print("PR watcher safety audit")
    print(f"mode: {'repo-only' if repo_only else 'repo + local watcher state'}")
    print("-" * 60)
    if not findings:
        print("OK: watcher docs/config/source grant no merge authority.")
        return
    print("FAIL: watcher merge authority detected")
    for finding in findings:
        print(f"- {finding.path}:{finding.line}: {finding.message}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo-only", action="store_true", help="skip local ~/.local watcher files")
    parser.add_argument(
        "--watcher-bin",
        type=Path,
        default=Path.home() / ".local" / "bin" / "atlas-pr-watch",
    )
    parser.add_argument(
        "--watcher-wrapper",
        type=Path,
        default=Path.home() / ".local" / "bin" / "atlas-pr-watch-and-wake",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.home() / ".config" / "atlas-pr-watchers",
    )
    parser.add_argument(
        "--systemd-dir",
        type=Path,
        default=Path.home() / ".config" / "systemd" / "user",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    findings = build_findings(
        args.repo_root.resolve(),
        watcher_bin=args.watcher_bin.expanduser(),
        watcher_wrapper=args.watcher_wrapper.expanduser(),
        config_dir=args.config_dir.expanduser(),
        systemd_dir=args.systemd_dir.expanduser(),
        repo_only=args.repo_only,
    )
    render(findings, repo_only=args.repo_only)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
