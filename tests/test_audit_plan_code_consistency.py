"""Fixture tests for scripts/audit_plan_code_consistency.py."""
from __future__ import annotations

import subprocess
import textwrap

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_plan_code_consistency")


def test_parse_claims_uses_exact_section_titles(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Out of scope

        `scripts/imaginary.py`

        ## Scope (this PR)

        `scripts/audit_plan_code_consistency.py`
    """)

    paths, funcs = auditor.parse_claims(plan)

    assert paths == {"scripts/audit_plan_code_consistency.py"}
    assert funcs == set()


def test_parse_claims_accepts_root_and_hyphenated_paths(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `AGENTS.md`
        `plans/PR-Audit-The-Auditors-1.md`
    """)

    paths, _ = auditor.parse_claims(plan)

    assert "AGENTS.md" in paths
    assert "plans/PR-Audit-The-Auditors-1.md" in paths


def test_parse_claims_reads_only_backticked_paths(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        This prose mentions scripts/not_backticked.py but should not enforce it.
        `scripts/audit_plan_code_consistency.py`
    """)

    paths, _ = auditor.parse_claims(plan)

    assert paths == {"scripts/audit_plan_code_consistency.py"}


def test_parse_claims_ignores_backticked_commands_with_paths(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Verification

        `python scripts/audit_plan_doc.py plans/PR-Example.md`
        `uv run scripts/audit_plan_doc.py plans/PR-Example.md`
        `scripts/local_pr_review.sh --current-pr-body-file /tmp/body.md`
        `scripts/local_pr_review.sh plans/Example.md`
    """)

    paths, funcs = auditor.parse_claims(plan)

    assert paths == set()
    assert funcs == set()


def test_parse_claims_preserves_literal_paths_with_spaces(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `docs/path with spaces.md`
        `docs/foo - bar.md`
    """)

    paths, funcs = auditor.parse_claims(plan)

    assert paths == {"docs/foo - bar.md", "docs/path with spaces.md"}
    assert funcs == set()


def test_parse_claims_reads_backticked_function_calls(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Mechanism

        Calls `parse_claims()` but ignores get() because it is too short.
    """)

    _, funcs = auditor.parse_claims(plan)

    assert funcs == {"parse_claims"}


def test_audit_claims_reports_missing_path_and_function(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `scripts/does_not_exist.py`

        ## Mechanism

        Calls `function_that_does_not_exist()`.
    """)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == ["scripts/does_not_exist.py"]
    assert missing_functions == ["function_that_does_not_exist"]


def test_audit_claims_accepts_deleted_branch_path_and_basename(
    auditor, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    scripts = repo / "scripts"
    scripts.mkdir()
    deleted = scripts / "deleted[magic].py"
    non_ascii_deleted = scripts / "résumé.py"
    deleted.write_text("print('bye')\n", encoding="utf-8")
    non_ascii_deleted.write_text("print('bye')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "origin/main"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    deleted.unlink()
    non_ascii_deleted.unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "delete file"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "origin/feature"], cwd=repo, check=True)

    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `scripts/deleted[magic].py`
        `deleted[magic].py`
        `résumé.py`
    """)

    monkeypatch.setattr(auditor, "REPO_ROOT", repo)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == []
    assert missing_functions == []

    missing_paths, missing_functions = auditor.audit_claims(plan, "origin/feature")

    assert set(missing_paths) == {
        "deleted[magic].py",
        "scripts/deleted[magic].py",
        "résumé.py",
    }
    assert missing_functions == []


def test_audit_claims_ignores_gitignored_local_session_state(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `SESSION_STATE.local.md`
    """)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == []
    assert missing_functions == []


def test_audit_claims_accepts_existing_root_path_and_function(auditor):
    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `AGENTS.md`

        ## Mechanism

        Calls `parse_claims()`.
    """)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == []
    assert missing_functions == []


def test_main_success_message_does_not_claim_every_path_exists_on_disk(
    auditor, capsys, monkeypatch, tmp_path
):
    plan = tmp_path / "plan.md"
    plan.write_text(
        textwrap.dedent("""\
            # Example

            ## Scope (this PR)

            `AGENTS.md`
        """),
        encoding="utf-8",
    )

    monkeypatch.setattr(auditor.sys, "argv", ["audit_plan_code_consistency.py", str(plan)])

    assert auditor.main() == 0
    out = capsys.readouterr().out
    assert "path claims resolve" in out
    assert "path claims exist on disk" not in out
