from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_passes_when_base_moved_without_file_overlap(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(repo, "atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx", "branch\n")
    _advance_origin_main(repo, "README.md", "main moved\n")

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "base changed files since branch point: 1" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_fails_when_base_changed_same_file_since_branch_point(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    path = "atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx"
    _commit(repo, path, "branch\n")
    _advance_origin_main(repo, path, "main\n")

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 1
    assert "DRIFT: base branch changed files" in result.stdout
    assert path in result.stdout


def test_cli_warns_when_open_pr_changes_same_file(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    path = "extracted_content_pipeline/services/faq.py"
    _commit(repo, path, "branch\n")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 12,
                "title": "Content Ops overlap",
                "headRefName": "claude/other",
                "url": "https://github.test/pr/12",
            }
        ],
        files={12: [path]},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: open PRs change files" in result.stdout
    assert "#12 Content Ops overlap" in result.stdout
    assert path in result.stdout


def test_cli_ignores_open_pr_for_current_branch(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "atlas-intel-ui/src/api/contentOps.ts"
    _commit(repo, path, "branch\n")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 13,
                "title": "Self PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/13",
            }
        ],
        files={13: [path]},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "checked 1 open PR(s)" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_uses_github_head_ref_for_current_pr_when_detached(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    _git(repo, "checkout", "--detach", "HEAD")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 25,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/25",
            }
        ],
        files={25: [path]},
        bodies={25: "Plan: plans/PR-Current.md\n\nOwnership lane: atlas-workflow\n"},
    )

    result = _run_with_path(
        repo,
        gh_bin,
        ["python", "scripts/audit_pr_session_drift.py"],
        extra_env={"GITHUB_HEAD_REF": "claude/current"},
    )

    assert result.returncode == 1
    assert "current PR body slice phase contract failed" in result.stdout
    assert "current PR body: missing Slice phase" in result.stdout


def test_cli_fails_when_current_pr_body_missing_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 20,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/20",
            }
        ],
        files={20: [path]},
        bodies={20: "Plan: plans/PR-Current.md\n\nOwnership lane: atlas-workflow\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 1
    assert "current PR body slice phase contract failed" in result.stdout
    assert "current PR body: missing Slice phase" in result.stdout


def test_cli_fails_when_current_pr_body_slice_phase_is_invalid(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 21,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/21",
            }
        ],
        files={21: [path]},
        bodies={21: "Plan: plans/PR-Current.md\n\nSlice phase: Discovery\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 1
    assert "invalid Slice phase 'Discovery'" in result.stdout


def test_cli_fails_when_current_pr_body_slice_phase_mismatches_plan(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 22,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/22",
            }
        ],
        files={22: [path]},
        bodies={22: "Plan: plans/PR-Current.md\n\nSlice phase: Robust testing\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 1
    assert "does not match branch plan phase(s) workflow/process" in result.stdout


def test_cli_fails_when_required_pr_body_was_not_checked_before_pr_exists(
    tmp_path: Path,
) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )

    result = _run(
        repo,
        [
            "python",
            "scripts/audit_pr_session_drift.py",
            "--skip-github",
            "--require-current-pr-body",
        ],
    )

    assert result.returncode == 1
    assert "current PR body slice phase contract failed" in result.stdout
    assert "current PR body: not checked" in result.stdout
    assert "--current-pr-body-file" in result.stdout


def test_cli_accepts_current_pr_body_file_before_pr_exists(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    body = repo / "pr-body.md"
    body.write_text(
        "Plan: plans/PR-Current.md\n\nSlice phase: Workflow/process\n",
        encoding="utf-8",
    )

    result = _run(
        repo,
        [
            "python",
            "scripts/audit_pr_session_drift.py",
            "--skip-github",
            "--require-current-pr-body",
            "--current-pr-body-file",
            str(body),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_rejects_current_pr_body_file_missing_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    body = repo / "pr-body.md"
    body.write_text("Plan: plans/PR-Current.md\n", encoding="utf-8")

    result = _run(
        repo,
        [
            "python",
            "scripts/audit_pr_session_drift.py",
            "--skip-github",
            "--require-current-pr-body",
            "--current-pr-body-file",
            str(body),
        ],
    )

    assert result.returncode == 1
    assert "current PR body slice phase contract failed" in result.stdout
    assert "current PR body: missing Slice phase" in result.stdout


def test_cli_accepts_current_pr_body_matching_plan_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 23,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/23",
            }
        ],
        files={23: [path]},
        bodies={23: "Plan: plans/PR-Current.md\n\nSlice phase: Workflow/process\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_accepts_mixed_case_period_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Vertical Slice.\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 24,
                "title": "Current PR",
                "headRefName": "claude/current",
                "url": "https://github.test/pr/24",
            }
        ],
        files={24: [path]},
        bodies={24: "Plan: plans/PR-Current.md\n\nSlice phase: vertical slice.\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "branch slice phases: vertical slice" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_warns_for_unsafe_open_pr_file_path(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(repo, "README.md", "branch\n")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 14,
                "title": "Bad path",
                "headRefName": "claude/bad-path",
                "url": "https://github.test/pr/14",
            }
        ],
        files={14: ["../outside.txt"]},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: GitHub metadata skipped or malformed" in result.stdout
    assert "../outside.txt" in result.stdout


def test_cli_fails_when_changed_plan_doc_missing_ownership_lane(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-No-Lane.md",
        "# PR-No-Lane\n\n## Scope (this PR)\n\nSlice phase: Workflow/process\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 1
    assert "ownership lane contract failed" in result.stdout
    assert "plans/PR-No-Lane.md: missing Ownership lane" in result.stdout


def test_cli_keeps_existing_whole_doc_ownership_lane_parsing(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Top-Lane.md",
        "# PR-Top-Lane\n\nOwnership lane: atlas-workflow\n\n"
        "## Scope (this PR)\n\nSlice phase: Workflow/process\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "branch ownership lanes: atlas-workflow" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_fails_when_changed_plan_doc_missing_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-No-Phase.md",
        "# PR-No-Phase\n\n## Scope (this PR)\n\nOwnership lane: atlas-workflow\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 1
    assert "slice phase contract failed" in result.stdout
    assert "plans/PR-No-Phase.md: missing Slice phase" in result.stdout


def test_cli_fails_when_changed_plan_doc_has_invalid_slice_phase(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Bad-Phase.md",
        "# PR-Bad-Phase\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Exploratory cleanup\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 1
    assert "invalid Slice phase 'Exploratory cleanup'" in result.stdout


def test_cli_reads_plan_metadata_only_from_scope_section(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Scope-Metadata.md",
        "# PR-Scope-Metadata\n\n## Scope (this PR)\n\n"
        "Ownership lane: atlas-workflow\n\nSlice phase: Workflow/process\n\n"
        "## Mechanism\n\n```md\nSlice phase: <phase>\n```\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "invalid Ownership lane '<lane>'" not in result.stdout
    assert "invalid Slice phase '<phase>'" not in result.stdout


def test_cli_allows_modified_legacy_plan_doc_without_ownership_lane(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Legacy.md",
        "# PR-Legacy\n\n## Why this slice exists\n\nOld plan.\n",
    )
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--ff-only", "feature")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "checkout", "-b", "legacy-edit")
    _commit(
        repo,
        "plans/PR-Legacy.md",
        "# PR-Legacy\n\n## Why this slice exists\n\nOld plan edited.\n",
    )

    result = _run(repo, ["python", "scripts/audit_pr_session_drift.py", "--skip-github"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ownership lane contract failed" not in result.stdout


def test_cli_fails_when_open_pr_claims_same_ownership_lane(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\nOwnership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 15,
                "title": "Other FAQ slice",
                "headRefName": "claude/other-faq",
                "url": "https://github.test/pr/15",
            }
        ],
        files={15: ["README.md"]},
        bodies={15: "Plan: plans/PR-Other.md\n\nOwnership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 1
    assert "DRIFT: open PRs claim the same ownership lane" in result.stdout
    assert "#15 Other FAQ slice" in result.stdout
    assert "content-ops/faq-generator" in result.stdout


def test_cli_allows_open_pr_with_different_ownership_lane(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\nOwnership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 16,
                "title": "Landing SEO slice",
                "headRefName": "claude/landing-seo",
                "url": "https://github.test/pr/16",
            }
        ],
        files={16: ["atlas-intel-ui/src/pages/Landing.tsx"]},
        bodies={16: "Ownership lane: content-ops/landing-seo-geo\n\nSlice phase: Workflow/process\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "branch ownership lanes: content-ops/faq-generator" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_warns_when_pr_view_fails_mid_sweep(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(repo, "README.md", "branch\n")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 17,
                "title": "Transient failure",
                "headRefName": "claude/transient",
                "url": "https://github.test/pr/17",
            }
        ],
        files={},
        fail_views={17: "api timed out"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: GitHub metadata skipped or malformed" in result.stdout
    assert "skipped PR #17" in result.stdout
    assert "api timed out" in result.stdout


def test_cli_warns_for_malformed_lane_in_other_pr_body(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\nOwnership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 18,
                "title": "Malformed lane",
                "headRefName": "claude/bad-lane",
                "url": "https://github.test/pr/18",
            }
        ],
        files={18: ["README.md"]},
        bodies={18: "Ownership lane: Content Ops!\n\nSlice phase: Workflow/process\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: GitHub metadata skipped or malformed" in result.stdout
    assert "PR #18 body: invalid Ownership lane" in result.stdout


def test_cli_treats_other_pr_invalid_slice_phase_as_advisory(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _commit(
        repo,
        "plans/PR-Current.md",
        "# PR-Current\n\n## Scope (this PR)\n\n"
        "Ownership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n",
    )
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 77,
                "title": "Legacy bad phase",
                "headRefName": "claude/legacy",
                "url": "https://github.test/pr/77",
            }
        ],
        files={77: ["README.md"]},
        bodies={77: "Ownership lane: content-ops/other-lane\n\nSlice phase: Discovery\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: GitHub metadata skipped or malformed" in result.stdout
    assert "PR #77 body: invalid Slice phase" in result.stdout
    assert "OK: no blocking drift detected" in result.stdout


def test_cli_ignores_current_pr_by_head_oid_when_detached(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, branch="claude/current")
    path = "plans/PR-Current.md"
    _commit(
        repo,
        path,
        "# PR-Current\n\n## Scope (this PR)\n\nOwnership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n",
    )
    head_oid = _git_stdout(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", "HEAD")
    gh_bin = _write_fake_gh(
        tmp_path,
        prs=[
            {
                "number": 19,
                "title": "Self detached",
                "headRefName": "claude/current",
                "headRefOid": head_oid,
                "url": "https://github.test/pr/19",
            }
        ],
        files={19: [path]},
        bodies={19: "Ownership lane: content-ops/faq-generator\n\nSlice phase: Workflow/process\n"},
    )

    result = _run_with_path(repo, gh_bin, ["python", "scripts/audit_pr_session_drift.py"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: no blocking drift detected" in result.stdout


def _write_fixture_repo(tmp_path: Path, *, branch: str = "feature") -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "atlas-intel-ui/src/pages").mkdir(parents=True)
    (repo / "scripts" / "audit_pr_session_drift.py").write_text(
        (REPO_ROOT / "scripts" / "audit_pr_session_drift.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    (repo / "atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx").write_text(
        "base\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "checkout", "-b", branch)
    return repo


def _advance_origin_main(repo: Path, path: str, text: str) -> None:
    current = _git_stdout(repo, "branch", "--show-current")
    _git(repo, "checkout", "main")
    _commit(repo, path, text)
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "checkout", current)


def _commit(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", f"change {path}")


def _write_fake_gh(
    tmp_path: Path,
    *,
    prs: list[dict[str, object]],
    files: dict[int, list[str]],
    bodies: dict[int, str] | None = None,
    fail_views: dict[int, str] | None = None,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    script = f"""#!/usr/bin/env python3
import json
import sys

PRS = {json.dumps(prs)}
FILES = {json.dumps({str(key): value for key, value in files.items()})}
BODIES = {json.dumps({str(key): value for key, value in (bodies or {}).items()})}
FAIL_VIEWS = {json.dumps({str(key): value for key, value in (fail_views or {}).items()})}

args = sys.argv[1:]
if args[:2] == ["pr", "list"]:
    print(json.dumps(PRS))
    raise SystemExit(0)
if args[:2] == ["pr", "view"]:
    number = args[2]
    if number in FAIL_VIEWS:
        print(FAIL_VIEWS[number], file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({{
        "body": BODIES.get(number, ""),
        "files": [{{"path": path}} for path in FILES.get(number, [])],
    }}))
    raise SystemExit(0)
print("unexpected gh args: " + " ".join(args), file=sys.stderr)
raise SystemExit(2)
"""
    gh.write_text(script, encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=repo, check=False, capture_output=True, text=True)


def _run_with_path(
    repo: Path,
    bin_dir: Path,
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        **(extra_env or {}),
    }
    return subprocess.run(
        args,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _git_stdout(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
