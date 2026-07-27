"""Fixture tests for scripts/audit_plan_code_consistency.py."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
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
        `.venv/bin/python -m pytest tests/example.py`
        `.venv/bin/python tests/example.py`
        `tools/bin/pytest tests/example.py`
        `./tools/run tests/example.py`
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
        `./docs/path with spaces.md`
        `ATLAS Distributed System.txt`
        `node_distributions/ATLAS Distributed System.txt`
    """)

    paths, funcs = auditor.parse_claims(plan)

    assert paths == {
        "./docs/path with spaces.md",
        "ATLAS Distributed System.txt",
        "docs/foo - bar.md",
        "docs/path with spaces.md",
        "node_distributions/ATLAS Distributed System.txt",
    }
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
    renamed = scripts / "renamed-source.py"
    deleted.write_text("print('bye')\n", encoding="utf-8")
    non_ascii_deleted.write_text("print('bye')\n", encoding="utf-8")
    renamed.write_text("print('move me')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "origin/main"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    deleted.unlink()
    non_ascii_deleted.unlink()
    renamed.rename(scripts / "renamed-destination.py")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "delete and rename files"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "origin/feature"], cwd=repo, check=True)

    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `scripts/deleted[magic].py`
        `./scripts/deleted[magic].py`
        `deleted[magic].py`
        `résumé.py`
        `scripts/renamed-source.py`
        `renamed-source.py`
    """)

    monkeypatch.setattr(auditor, "REPO_ROOT", repo)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == []
    assert missing_functions == []

    missing_paths, missing_functions = auditor.audit_claims(plan, "origin/feature")

    assert set(missing_paths) == {
        "./scripts/deleted[magic].py",
        "deleted[magic].py",
        "scripts/deleted[magic].py",
        "résumé.py",
        "scripts/renamed-source.py",
        "renamed-source.py",
    }
    assert missing_functions == []


def test_deleted_basename_claim_stays_within_candidate_roots(
    auditor, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    unrelated = repo / "unrelated"
    unrelated.mkdir()
    retired = unrelated / "retired.py"
    retired.write_text("print('out of scope')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "-M", "main"], cwd=repo, check=True)
    subprocess.run(["git", "branch", "origin/main"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    retired.unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "delete unrelated file"], cwd=repo, check=True, stdout=subprocess.DEVNULL)

    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `retired.py`
        `unrelated/retired.py`
    """)

    monkeypatch.setattr(auditor, "REPO_ROOT", repo)

    missing_paths, missing_functions = auditor.audit_claims(plan)

    assert missing_paths == ["retired.py"]
    assert missing_functions == []


def test_deleted_path_inventory_failure_is_not_suppressed(auditor, monkeypatch):
    def failed_git_diff(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=128,
            stdout=b"",
            stderr=b"fatal: invalid symmetric difference",
        )

    monkeypatch.setattr(auditor.subprocess, "run", failed_git_diff)

    with pytest.raises(auditor.DeletedPathInventoryError) as excinfo:
        auditor._deleted_paths_in_branch_diff("missing-base")

    assert "missing-base" in str(excinfo.value)
    assert "fatal: invalid symmetric difference" in str(excinfo.value)


def test_main_reports_deleted_path_inventory_failure(
    auditor, monkeypatch, tmp_path, capsys
):
    plan = tmp_path / "plan.md"
    plan.write_text(
        textwrap.dedent("""\
            # Example

            ## Scope (this PR)

            `scripts/deleted.py`
        """),
        encoding="utf-8",
    )

    monkeypatch.setattr(auditor.sys, "argv", ["audit", "--base-ref", "missing-base", str(plan)])
    monkeypatch.setattr(
        auditor,
        "_deleted_paths_in_branch_diff",
        lambda _base_ref: (_ for _ in ()).throw(
            auditor.DeletedPathInventoryError("missing-base is not fetchable")
        ),
    )

    assert auditor.main() == 2
    captured = capsys.readouterr()
    assert "deleted-path inventory failed" in captured.err
    assert "missing-base is not fetchable" in captured.err


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


def test_audit_claims_accepts_existing_basename_with_spaces(auditor, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    distribution = repo / "node_distributions"
    distribution.mkdir(parents=True)
    (distribution / "ATLAS Distributed System.txt").write_text("ok\n", encoding="utf-8")

    plan = textwrap.dedent("""\
        # Example

        ## Scope (this PR)

        `ATLAS Distributed System.txt`
    """)

    monkeypatch.setattr(auditor, "REPO_ROOT", repo)

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
