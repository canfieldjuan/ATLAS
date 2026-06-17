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


WORKFLOW_GLOBS = ("*.yml", "*.yaml")
PINNED_REF_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_PULL_REQUEST_TARGET_JOB = ("security_guardrails.yml", "gitleaks-baseline-guard")
ALLOWED_ID_TOKEN_JOB = ("claude.yml", "claude")
CLAUDE_OWNER_GATE = "github.actor == github.repository_owner"


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
    if permissions == "write-all":
        return True
    return isinstance(permissions, dict) and permissions.get("id-token") == "write"


def _job_runs_on_pull_request_target(job: dict[str, Any]) -> bool:
    condition = job.get("if")
    if not isinstance(condition, str):
        return True
    if "github.event_name != 'pull_request_target'" in condition or 'github.event_name != "pull_request_target"' in condition:
        return False
    if "github.event_name == 'pull_request_target'" in condition or 'github.event_name == "pull_request_target"' in condition:
        return True
    if "github.event_name" in condition and "pull_request_target" not in condition:
        return False
    return "pull_request_target" not in condition


def _is_allowed_pull_request_target_job(path: Path, job_name: str, job: dict[str, Any]) -> bool:
    if (path.name, job_name) != ALLOWED_PULL_REQUEST_TARGET_JOB:
        return False
    if job.get("if") != "github.event_name == 'pull_request_target'":
        return False
    steps = _iter_steps(job)
    if not steps:
        return False
    checkout = steps[0]
    checkout_ref = _action_ref(str(checkout.get("uses", "")))
    if checkout_ref is None or checkout_ref[0] != "actions/checkout" or not PINNED_REF_RE.fullmatch(checkout_ref[1]):
        return False
    with_block = checkout.get("with")
    return isinstance(with_block, dict) and with_block.get("ref") == "${{ github.event.pull_request.base.sha }}"


def _is_allowed_oidc_job(path: Path, job_name: str, job: dict[str, Any]) -> bool:
    if (path.name, job_name) != ALLOWED_ID_TOKEN_JOB:
        return False
    condition = job.get("if")
    return isinstance(condition, str) and CLAUDE_OWNER_GATE in condition


def _iter_container_images(job: dict[str, Any]) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    container = job.get("container")
    if isinstance(container, str):
        images.append(("container", container))
    elif isinstance(container, dict) and isinstance(container.get("image"), str):
        images.append(("container", container["image"]))

    services = job.get("services")
    if isinstance(services, dict):
        for service_name, service in services.items():
            if isinstance(service, str):
                images.append((f"service {service_name}", service))
            elif isinstance(service, dict) and isinstance(service.get("image"), str):
                images.append((f"service {service_name}", service["image"]))
    return images


def audit_workflow(path: Path) -> list[Finding]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(workflow, dict):
        return [Finding("ERROR", str(path), "workflow root is not a mapping")]

    findings: list[Finding] = []
    name = path.name
    events = _event_names(workflow.get(True, workflow.get("on")))
    workflow_oidc = _permissions_write_oidc(workflow.get("permissions"))
    if workflow_oidc:
        findings.append(Finding("ERROR", str(path), "grants workflow-scope id-token: write or write-all without an allowlist rationale"))

    for job_name, job in _iter_jobs(workflow):
        if "pull_request_target" in events and _job_runs_on_pull_request_target(job):
            if _is_allowed_pull_request_target_job(path, job_name, job):
                findings.append(Finding("WARN", str(path), f"job {job_name} allowed pull_request_target: trusted-base checkout guard"))
            else:
                findings.append(Finding("ERROR", str(path), f"job {job_name} can run on pull_request_target without the approved trusted-base guard shape"))

        if _permissions_write_oidc(job.get("permissions")):
            if _is_allowed_oidc_job(path, job_name, job):
                findings.append(Finding("WARN", str(path), f"job {job_name} allowed id-token: write: Claude Code action is owner-gated"))
            else:
                findings.append(Finding("ERROR", str(path), f"job {job_name} grants id-token: write or write-all without the approved owner-gated shape"))

        job_uses = job.get("uses")
        if isinstance(job_uses, str):
            ref = _action_ref(job_uses)
            if ref is not None:
                action, action_ref = ref
                if not PINNED_REF_RE.fullmatch(action_ref):
                    findings.append(Finding("WARN", str(path), f"job {job_name} reusable workflow {action}@{action_ref or '<missing ref>'} is not SHA-pinned"))

        for image_kind, image in _iter_container_images(job):
            if "@sha256:" not in image:
                findings.append(Finding("WARN", str(path), f"job {job_name} {image_kind} image {image} is not digest-pinned"))

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
    paths = sorted({path for glob in WORKFLOW_GLOBS for path in workflows_dir.glob(glob)})
    for path in paths:
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
