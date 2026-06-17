#!/usr/bin/env python3
"""Audit GitHub Actions workflow supply-chain posture."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_GLOB = "*.yml"
PINNED_REF_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_PULL_REQUEST_TARGET = {
    "security_guardrails.yml": "Gitleaks baseline guard checks out trusted base code and fetches PR head as data.",
}
ALLOWED_ID_TOKEN_WRITE = {
    "claude.yml": "Claude Code action uses OIDC; trigger is owner-gated and workflow token permissions are read-only.",
}


@dataclass(frozen=True)
class Finding:
    level: str
    path: str
    detail: str


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _event_names(on_block: Any) -> set[str]:
    if isinstance(on_block, str):
        return {on_block}
    if isinstance(on_block, list):
        return {event for event in on_block if isinstance(event, str)}
    if isinstance(on_block, dict):
        return {str(event) for event in on_block}
    return set()


def _iter_jobs(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return []
    return [(str(name), job) for name, job in jobs.items() if isinstance(job, dict)]


def _iter_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps")
    return [step for step in _as_list(steps) if isinstance(step, dict)]


def _action_ref(uses: str) -> tuple[str, str] | None:
    if uses.startswith("./") or uses.startswith("docker://"):
        return None
    if "@" not in uses:
        return (uses, "")
    action, ref = uses.rsplit("@", 1)
    return action, ref


def _permissions_write_oidc(permissions: Any) -> bool:
    return isinstance(permissions, dict) and permissions.get("id-token") == "write"


def audit_workflow(path: Path) -> list[Finding]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(workflow, dict):
        return [Finding("ERROR", str(path), "workflow root is not a mapping")]

    findings: list[Finding] = []
    name = path.name
    events = _event_names(workflow.get(True, workflow.get("on")))
    if "pull_request_target" in events and name not in ALLOWED_PULL_REQUEST_TARGET:
        findings.append(Finding("ERROR", str(path), "uses pull_request_target without an explicit allowlist rationale"))

    workflow_oidc = _permissions_write_oidc(workflow.get("permissions"))
    if workflow_oidc and name not in ALLOWED_ID_TOKEN_WRITE:
        findings.append(Finding("ERROR", str(path), "grants workflow-scope id-token: write without an allowlist rationale"))
    elif workflow_oidc:
        findings.append(Finding("WARN", str(path), f"allowed workflow-scope id-token: write: {ALLOWED_ID_TOKEN_WRITE[name]}"))

    for job_name, job in _iter_jobs(workflow):
        if _permissions_write_oidc(job.get("permissions")):
            if name not in ALLOWED_ID_TOKEN_WRITE:
                findings.append(Finding("ERROR", str(path), f"job {job_name} grants id-token: write without an allowlist rationale"))
            else:
                findings.append(Finding("WARN", str(path), f"job {job_name} allowed id-token: write: {ALLOWED_ID_TOKEN_WRITE[name]}"))
        for step in _iter_steps(job):
            uses = step.get("uses")
            if not isinstance(uses, str):
                continue
            ref = _action_ref(uses)
            if ref is None:
                continue
            action, action_ref = ref
            if not PINNED_REF_RE.fullmatch(action_ref):
                findings.append(Finding("WARN", str(path), f"{action}@{action_ref or '<missing ref>'} is not SHA-pinned"))

    return findings


def audit_workflows(workflows_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(workflows_dir.glob(WORKFLOW_GLOB)):
        findings.extend(audit_workflow(path))
    return findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows_dir", nargs="?", default=".github/workflows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings = audit_workflows(Path(args.workflows_dir))
    errors = [finding for finding in findings if finding.level == "ERROR"]
    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.detail}")
    if errors:
        print(f"workflow security posture audit failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("workflow security posture audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
