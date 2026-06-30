"""Tests for the AGENTS.md section 1b PR-body contract audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_pr_body.py"
SPEC = importlib.util.spec_from_file_location("audit_pr_body", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_pr_body_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_pr_body_module
SPEC.loader.exec_module(audit_pr_body_module)

audit_pr_body = audit_pr_body_module.audit_pr_body
is_dependabot_author = audit_pr_body_module.is_dependabot_author


def _valid_body(plan: str = "plans/PR-Example.md") -> str:
    return "\n".join([
        f"Plan: {plan}",
        "Slice phase: Production hardening",
        "",
        "One-paragraph why.",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])


def _write_plan(tmp_path: Path, plan: str = "plans/PR-Example.md") -> Path:
    plan_path = tmp_path / plan
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# PR-Example\n", encoding="utf-8")
    return tmp_path


def test_valid_body_passes(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)

    assert audit_pr_body(_valid_body(), root=root) == []


def test_empty_body_fails() -> None:
    assert audit_pr_body("  \n\n") == ["PR body is empty"]


def test_missing_plan_lead_line_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Plan: plans/PR-Example.md", "Overview first")

    failures = audit_pr_body(body, root=root)

    assert any("first non-empty line" in failure for failure in failures)


def test_nonexistent_plan_doc_fails(tmp_path: Path) -> None:
    failures = audit_pr_body(_valid_body(), root=tmp_path)

    assert any("does not exist" in failure for failure in failures)


def test_missing_one_paragraph_why_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = "\n".join([
        "Plan: plans/PR-Example.md",
        "Slice phase: Production hardening",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])

    failures = audit_pr_body(body, root=root)

    assert any("one-paragraph why" in failure for failure in failures)


def test_missing_slice_phase_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Slice phase: Production hardening\n", "")

    failures = audit_pr_body(body, root=root)

    assert any("Slice phase" in failure for failure in failures)


def test_missing_section_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("## Parked hardening\nNone.\n", "")

    failures = audit_pr_body(body, root=root)

    assert "missing required section: ## Parked hardening" in failures


def test_out_of_order_sections_fail(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = "\n".join([
        "Plan: plans/PR-Example.md",
        "Slice phase: Production hardening",
        "",
        "One-paragraph why.",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])

    failures = audit_pr_body(body, root=root)

    assert any("out of order" in failure for failure in failures)


def test_extra_sections_between_required_ones_pass(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace(
        "## Verification",
        "## Review notes\nExtra context.\n\n## Verification",
    )

    assert audit_pr_body(body, root=root) == []


def test_slice_phase_after_first_heading_fails(tmp_path: Path) -> None:
    root = _write_plan(tmp_path)
    body = _valid_body().replace("Slice phase: Production hardening\n", "")
    body = body.replace(
        "## Intentional",
        "## Intentional\nSlice phase: Production hardening",
    )

    failures = audit_pr_body(body, root=root)

    assert any("Slice phase" in failure for failure in failures)


def test_dependabot_author_detection() -> None:
    assert is_dependabot_author("app/dependabot")
    assert is_dependabot_author("dependabot[bot]")
    assert is_dependabot_author(" dependabot ")
    assert not is_dependabot_author("canfieldjuan")
    assert not is_dependabot_author(None)


def test_dependabot_cli_exempts_invalid_body() -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as body_file:
        body_file.write("Dependabot generated body without the Atlas plan contract.\n")
        body_path = Path(body_file.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pr-author",
                "app/dependabot",
                str(body_path),
            ],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "Dependabot PR body exempt" in result.stdout
    finally:
        body_path.unlink(missing_ok=True)


def test_normal_author_cli_rejects_same_invalid_body() -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as body_file:
        body_file.write("Dependabot generated body without the Atlas plan contract.\n")
        body_path = Path(body_file.name)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--pr-author",
                "canfieldjuan",
                str(body_path),
            ],
            check=False,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 1
        assert "AGENTS.md section 1b contract" in result.stdout
    finally:
        body_path.unlink(missing_ok=True)
