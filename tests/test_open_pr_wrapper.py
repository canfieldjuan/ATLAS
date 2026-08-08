from __future__ import annotations

import os
import fcntl
import subprocess
from pathlib import Path
from shutil import copy2, copytree, ignore_patterns

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "open_pr.sh"
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_pr_body.py"
AI_RECONCILIATION_SCRIPT = REPO_ROOT / "scripts" / "audit_ai_reconciliation.py"
CHANGE_POLICY_SCRIPT = REPO_ROOT / "scripts" / "_pr_change_policy.py"
LOCAL_REVIEW_SCRIPT = REPO_ROOT / "scripts" / "local_pr_review.sh"
BRANCH_NAME_SCRIPT = REPO_ROOT / "scripts" / "check_pr_branch_name.py"
OPEN_PR_WRAPPER_MARKER = "<!-- atlas-open-pr-wrapper: v1 -->"


def test_open_pr_create_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper", "--base", "main")

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert (repo / "local-review.log").read_text(encoding="utf-8").strip().startswith(
        "local_pr_review /tmp/atlas-open-pr-body."
    )
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)
    assert OPEN_PR_WRAPPER_MARKER not in body.read_text(encoding="utf-8")


def test_open_pr_edit_passes_body_via_stdin_not_path(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=0)
    guard_log = Path(env["OWNERSHIP_GUARD_LOG"])

    result = _run(repo, env, body)

    assert result.returncode == 0
    assert log.read_text(encoding="utf-8").strip() == "pr edit 17 --repo canfieldjuan/ATLAS --body-file -"
    assert guard_log.read_text(encoding="utf-8").strip() == (
        f"--pr 17 --branch claude/pr-test --head-sha {_git_output(repo, 'rev-parse', 'HEAD')}"
    )
    assert str(body) not in log.read_text(encoding="utf-8")
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_existing_pr_ownership_guard_failure_blocks_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=0)
    env["OWNERSHIP_GUARD_EXIT"] = "23"

    result = _run(repo, env, body)

    assert result.returncode == 23
    assert "fake ownership guard failed" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()
    assert not (repo / "local-review.log").exists()


def test_open_pr_existing_pr_rejects_create_only_args(tmp_path: Path) -> None:
    repo, body, env, _, _ = _ready(tmp_path, view_exit=0)

    result = _run(repo, env, body, "--title", "New title")

    assert result.returncode == 2
    assert "PR already exists" in result.stderr


def test_open_pr_rejects_branch_that_does_not_match_plan_before_fetch(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "switch", "-c", "claude/pr-other")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert "does not match PR plan branch" in result.stderr
    assert "Refreshing origin/main" not in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_dependabot_author_keeps_generated_body_exemption(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _git(repo, "switch", "-c", "dependabot/pip/security-update")
    (repo / "scripts" / "dependabot_example.py").write_text(
        "print('dependency update')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "scripts/dependabot_example.py")
    _git(repo, "commit", "-qm", "dependabot fixture")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo.parent / "body-dependabot.md"
    body.write_text("Generated dependency update body.\n", encoding="utf-8")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)
    env["ATLAS_CURRENT_PR_AUTHOR"] = "dependabot[bot]"

    result = _run(repo, env, body, "--title", "Dependabot fixture")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "pr body audit: PASS (Dependabot PR body exempt)" in result.stdout
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Dependabot fixture --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("args", "extra_env", "expected"),
    [
        (["--body-file", "body.md"], {}, "pass the PR body as BODY_FILE"),
        (["--head", "other"], {}, "refusing target-changing create arg: --head"),
        (["--repo", "other/repo"], {}, "refusing target-changing create arg: --repo"),
        (["--draft"], {}, "refusing draft PR without explicit operator consent"),
        (["--draft=true"], {}, "refusing draft PR without explicit operator consent"),
        (["--draft=TRUE"], {}, "refusing draft PR without explicit operator consent"),
        (["--draft=1"], {}, "refusing draft PR without explicit operator consent"),
        (["--draft=false"], {}, "refusing draft PR without explicit operator consent"),
        (["-d"], {}, "refusing draft PR without explicit operator consent"),
        (["-d=true"], {}, "refusing draft PR without explicit operator consent"),
        (["-d=1"], {}, "refusing draft PR without explicit operator consent"),
        (["-dw"], {}, "refusing draft PR without explicit operator consent"),
        (["-fd"], {}, "refusing draft PR without explicit operator consent"),
        (["-wd"], {}, "refusing draft PR without explicit operator consent"),
        (["-fd=true"], {}, "refusing draft PR without explicit operator consent"),
        (["-fwd"], {}, "refusing draft PR without explicit operator consent"),
        (["-dt", "some title"], {}, "refusing draft PR without explicit operator consent"),
        (["--web"], {}, "refusing browser-based create arg"),
        (["--web=true"], {}, "refusing browser-based create arg"),
        (["-w"], {}, "refusing browser-based create arg"),
        (["-fw"], {}, "refusing browser-based create arg"),
        (["--base", "release"], {}, "refusing non-main base: release"),
        (["-Brelease"], {}, "refusing non-main base: release"),
        ([], {"GH_REPO": "other/repo"}, "refusing GH_REPO target override"),
    ],
)
def test_open_pr_rejects_unsafe_inputs_before_gh(
    tmp_path: Path,
    args: list[str],
    extra_env: dict[str, str],
    expected: str,
) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env.update(extra_env)

    result = _run(repo, env, body, *args)

    assert result.returncode == 2
    assert expected in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_forwards_draft_when_operator_consent_flag_is_set(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["ATLAS_OPEN_PR_DRAFT_CONSENT"] = "1"

    result = _run(repo, env, body, "--draft", "--title", "Draft wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --draft --title Draft wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_forwards_draft_assignment_when_operator_consent_flag_is_set(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["ATLAS_OPEN_PR_DRAFT_CONSENT"] = "1"

    result = _run(repo, env, body, "--draft=true", "--title", "Draft wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --draft=true --title Draft wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_allows_value_shorthand_containing_d_without_consent(tmp_path: Path) -> None:
    # -t takes a value, so the attached text (even containing 'd') is a title,
    # not a shorthand cluster that enables draft mode.
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)

    result = _run(repo, env, body, "-tdraft-note")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create -tdraft-note --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


@pytest.mark.parametrize("args", [["--title", "--draft"], ["-t", "-d"]])
def test_open_pr_allows_draft_shaped_value_of_title_without_consent(
    tmp_path: Path,
    args: list[str],
) -> None:
    # gh consumes the token after --title/-t as the title value, so a
    # "--draft"-shaped value does not enable draft mode and needs no consent.
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)

    result = _run(repo, env, body, *args)

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        f"pr create {' '.join(args)} --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_disables_gh_prompts_so_interactive_draft_is_unreachable(tmp_path: Path) -> None:
    # gh's interactive create survey offers a "Submit as draft" action that
    # never appears in argv; the wrapper must export GH_PROMPT_DISABLED=1 so
    # every gh call is non-interactive and draft mode can only arrive as an
    # argv flag through the consent gate.
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env.pop("GH_PROMPT_DISABLED", None)

    result = _run(repo, env, body, "--title", "Prompt gate")

    assert result.returncode == 0, result.stdout + result.stderr
    prompt_state = Path(env["GH_PROMPT_STATE"])
    assert prompt_state.read_text(encoding="utf-8").strip() == "GH_PROMPT_DISABLED=1"


def _wrapper_expected_outcome(args: list[str]) -> str:
    """Independent admission oracle: walk argv as gh's pflag parser would.

    Returns "draft" when the first offending token carries a draft flag
    (value-blind, matching the wrapper's declared contract), "web" when it
    carries the browser-create flag without a draft flag, and "pass"
    otherwise -- modeling long options with separate or '='-attached values,
    '--' end-of-options, and shorthand clusters where booleans keep scanning
    and a value-taking shorthand consumes the attached remainder or the next
    token. Within one token, draft rejection takes precedence over web.
    """
    val_short = set("aBbFHlmprRtT")
    long_val = {
        "--assignee", "--base", "--body", "--body-file", "--head", "--label",
        "--milestone", "--project", "--recover", "--repo", "--reviewer",
        "--template", "--title",
    }
    i = 0
    while i < len(args):
        tok = args[i]
        i += 1
        draft_tok = web_tok = False
        if tok == "--":
            break
        if tok.startswith("--"):
            name = tok.split("=", 1)[0]
            if name == "--draft":
                draft_tok = True
            elif name == "--web":
                web_tok = True
            elif name in long_val and "=" not in tok:
                i += 1
        elif tok.startswith("-") and len(tok) > 1:
            cluster = tok[1:]
            while cluster:
                ch = cluster[0]
                if ch in val_short:
                    if len(cluster) == 1:
                        i += 1
                    break
                if ch == "d":
                    draft_tok = True
                elif ch == "w":
                    web_tok = True
                cluster = cluster[1:]
                if cluster.startswith("="):
                    break
        if draft_tok:
            return "draft"
        if web_tok:
            return "web"
    return "pass"


def _generate_argv_grammar_cases() -> list[list[str]]:
    # Product of boolean-shorthand positions x value-taking terminators x
    # attached/separate/'=' values, restricted to letters the wrapper does not
    # reject for target/body reasons (booleans f, w, e; value-taking t, l).
    cases: list[tuple[str, ...]] = []
    for pre in ("", "f", "w", "fw", "e"):
        for core in ("", "d"):
            base = pre + core
            if base:
                cases.append((f"-{base}",))
                cases.append((f"-{base}=true",))
                cases.append((f"-{base}=false",))
            for val in ("t", "l"):
                cases.append((f"-{base}{val}attached-value",))
                cases.append((f"-{base}{val}", "separate-value"))
                cases.append((f"-{base}{val}", "-d"))
    cases += [
        ("--draft",), ("--draft=true",), ("--draft=false",),
        ("--title", "--draft"), ("--title", "-d"), ("--title", "t", "--draft"),
        ("--label", "--draft", "--draft"),
        ("--fill", "--draft"), ("--web", "-d"),
        ("--", "--draft"), ("--", "-d"),
        ("-t", "-d"), ("-a", "-d"), ("-t", "-d", "--draft"),
        ("--template", "--draft"), ("--reviewer", "-fd"),
    ]
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for case in cases:
        if case not in seen and all(tok != "-" for tok in case):
            seen.add(case)
            unique.append(list(case))
    return unique


def test_open_pr_draft_admission_matches_gh_argv_grammar(tmp_path: Path) -> None:
    # Grammar-derived closure proof: for every generated argv sequence the
    # wrapper's consent decision must equal the independent pflag oracle.
    # The fixture has no origin remote, so a sequence that passes admission
    # deterministically fails later at the base refresh -- proving admission
    # neither gated it nor needed the network.
    repo = tmp_path / "grammar-repo"
    (repo / "scripts").mkdir(parents=True)
    copy2(SCRIPT, repo / "scripts" / "open_pr.sh")
    copy2(BRANCH_NAME_SCRIPT, repo / "scripts" / "check_pr_branch_name.py")
    copy2(CHANGE_POLICY_SCRIPT, repo / "scripts" / "_pr_change_policy.py")
    subprocess.run(
        ["git", "init", "--initial-branch", "main"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "switch", "-c", "claude/pr-test")
    body = tmp_path / "grammar-body.md"
    body.write_text(_valid_body(), encoding="utf-8")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("GH_REPO", None)
    env.pop("ATLAS_OPEN_PR_DRAFT_CONSENT", None)

    failures = []
    for args in _generate_argv_grammar_cases():
        expected = _wrapper_expected_outcome(args)
        result = subprocess.run(
            ["bash", str(repo / "scripts" / "open_pr.sh"), str(body), *args],
            cwd=repo, env=env, capture_output=True, text=True,
        )
        if "refusing draft PR without explicit operator consent" in result.stderr:
            observed = "draft"
        elif "refusing browser-based create arg" in result.stderr:
            observed = "web"
        else:
            observed = "pass"
        if observed != expected:
            failures.append((args, f"oracle={expected} observed={observed}", result.returncode, result.stderr.strip()))
        elif expected == "pass" and "failed to refresh origin/main" not in result.stderr:
            failures.append((args, "admission pass did not reach base refresh", result.returncode, result.stderr.strip()))
    assert not failures, failures


def test_open_pr_rejects_web_create_even_with_draft_consent(tmp_path: Path) -> None:
    # Draft consent authorizes the flag-based draft path only; the browser
    # flow escapes post-mutation verification and stays rejected.
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["ATLAS_OPEN_PR_DRAFT_CONSENT"] = "1"

    result = _run(repo, env, body, "--draft", "--web")

    assert result.returncode == 2
    assert "refusing browser-based create arg" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_draft_before_any_fetch_side_effect(tmp_path: Path) -> None:
    # Argument admission must run before refresh_base_ref: with origin
    # destroyed, an unauthorized --draft still fails on consent, not on fetch.
    import shutil

    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    shutil.rmtree(tmp_path / "origin.git")

    result = _run(repo, env, body, "--draft")

    assert result.returncode == 2
    assert "refusing draft PR without explicit operator consent" in result.stderr
    assert "failed to refresh origin/main" not in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_invalid_body_before_gh(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write_body(repo)
    body = repo.parent / "body-invalid.md"
    body.write_text(_valid_body().replace("## Cold diff reconstruction\n- Changed: scripts/example.sh:1 updates the wrapper.\n- Contract match: traces to the body contract.\n- Gaps: none.\n\n", ""), encoding="utf-8")
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 1
    assert "missing required section: ## Cold diff reconstruction" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_accepts_docs_only_body(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    body = _write_docs_only_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Docs only")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit Markdown-only body exemption" in result.stdout
    assert log.read_text(encoding="utf-8").strip() == "pr create --title Docs only --repo canfieldjuan/ATLAS --base main --body-file -"
    assert stdin_capture.read_text(encoding="utf-8") == body.read_text(encoding="utf-8")


def test_open_pr_accepts_normal_ssh_origin_url(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, origin_url="ssh://git@github.com/canfieldjuan/ATLAS.git")
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_accepts_case_insensitive_repo_identity_after_create(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path, origin_url="git@github.com:canfieldjuan/atlas.git")
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=1)

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/atlas --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_rejects_unpublished_current_head_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "commit", "--allow-empty", "-qm", "unpushed")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert "current HEAD is" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_missing_remote_branch_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "push", "-q", "origin", "--delete", "claude/pr-test")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 1
    assert "failed to refresh origin/claude/pr-test" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_parallel_local_writer_before_review(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    lock_path = repo / ".git" / "open_pr_wrapper.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert "another open_pr.sh mutation is already running" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_reads_fetched_head_without_tracking_ref(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    _git(repo, "config", "remote.origin.fetch", "+refs/heads/main:refs/remotes/origin/main")
    _git(repo, "update-ref", "-d", "refs/remotes/origin/claude/pr-test")

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").strip() == (
        "pr create --title Workflow wrapper --repo canfieldjuan/ATLAS --base main --body-file -"
    )
    assert stdin_capture.read_text(encoding="utf-8") == _stamped_body(body)


def test_open_pr_rejects_local_review_failure_before_gh(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["LOCAL_REVIEW_EXIT"] = "42"

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 42
    assert "Running final local PR review before GitHub mutation" in result.stdout
    assert not log.exists()
    assert not stdin_capture.exists()


@pytest.mark.parametrize(
    ("env_flag", "expected"),
    [
        ("LOCAL_REVIEW_ADVANCE_REMOTE", "origin/claude/pr-test changed after review"),
        ("LOCAL_REVIEW_ADVANCE_BOTH", "current HEAD changed after review"),
        ("LOCAL_REVIEW_MUTATE_BODY", "PR body changed after review"),
    ],
)
def test_open_pr_rejects_snapshot_changes_after_local_review(
    tmp_path: Path,
    env_flag: str,
    expected: str,
) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env[env_flag] = "1"

    result = _run(repo, env, body, "--title", "Workflow wrapper")

    assert result.returncode == 2
    assert expected in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


@pytest.mark.parametrize(("view_exit", "args"), [(1, ["--title", "Workflow wrapper"]), (0, [])])
def test_open_pr_real_local_review_failure_blocks_mutation_branches(
    tmp_path: Path,
    view_exit: int,
    args: list[str],
) -> None:
    repo = _write_fixture_repo(tmp_path, real_local_review=True)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=view_exit)

    result = _run(repo, env, body, *args)

    assert result.returncode != 0
    assert "Running final local PR review before GitHub mutation" in result.stdout
    assert "worktree has uncommitted changes" not in result.stderr
    assert "==> Pre-push audit wrapper" in result.stdout
    assert "==> Plan shape: plans/PR-Test.md" in result.stdout
    assert "plans/PR-Test.md: missing Ownership lane" in result.stdout
    assert "plans/PR-Test.md: missing Slice phase" in result.stdout
    assert "No such file or directory" not in result.stderr
    gh_log = log.read_text(encoding="utf-8") if log.exists() else ""
    assert "pr create" not in gh_log
    assert "pr edit" not in gh_log
    captured_stdin = stdin_capture.read_text(encoding="utf-8") if stdin_capture.exists() else ""
    assert captured_stdin == ""


def test_open_pr_rejects_mismatched_existing_pr_identity_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=1)
    env["GH_PR_LIST_JSON"] = (
        '[{"number":17,"headRefName":"claude/pr-test","baseRefName":"release",'
        '"headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},"isCrossRepository":false}]'
    )

    result = _run(repo, env, body)

    assert result.returncode == 1
    assert "outside canfieldjuan/ATLAS->main" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def test_open_pr_rejects_stale_existing_pr_head_before_edit(tmp_path: Path) -> None:
    repo, body, env, log, stdin_capture = _ready(tmp_path, view_exit=0)
    stale_head = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=repo, text=True).strip()
    env["GH_PR_LIST_JSON"] = (
        f'[{{"number":17,"headRefName":"claude/pr-test","headRefOid":"{stale_head}",'
        '"baseRefName":"main","headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},'
        '"isCrossRepository":false}]'
    )

    result = _run(repo, env, body)

    assert result.returncode == 2
    assert "existing PR head does not match reviewed head" in result.stderr
    assert not log.exists()
    assert not stdin_capture.exists()


def _ready(tmp_path: Path, *, view_exit: int) -> tuple[Path, Path, dict[str, str], Path, Path]:
    repo = _write_fixture_repo(tmp_path)
    body = _write_body(repo)
    env, log, stdin_capture = _fake_gh_env(tmp_path, view_exit=view_exit)
    return repo, body, env, log, stdin_capture


def _run(repo: Path, env: dict[str, str], body: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/open_pr.sh", str(body), *args],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _stamped_body(body: Path) -> str:
    return body.read_text(encoding="utf-8") + f"\n{OPEN_PR_WRAPPER_MARKER}\n"


def _write_fixture_repo(
    tmp_path: Path,
    *,
    real_local_review: bool = False,
    origin_url: str = "git@github.com:canfieldjuan/ATLAS.git",
) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    if real_local_review:
        copytree(REPO_ROOT / "scripts", repo / "scripts", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "ci", repo / "ci", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "docs", repo / "docs", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / ".github", repo / ".github", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        (repo / "tests").mkdir(exist_ok=True)
        copy2(
            REPO_ROOT / "tests" / "test_security_guardrails_workflow.py",
            repo / "tests" / "test_security_guardrails_workflow.py",
        )
        copytree(REPO_ROOT / "extracted", repo / "extracted", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        for package_root in REPO_ROOT.glob("extracted_*"):
            if package_root.is_dir():
                copytree(package_root, repo / package_root.name, dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copytree(REPO_ROOT / "atlas_brain", repo / "atlas_brain", dirs_exist_ok=True, ignore=ignore_patterns("__pycache__"))
        copy2(REPO_ROOT / "AGENTS.md", repo / "AGENTS.md")
        copy2(REPO_ROOT / "CLAUDE.md", repo / "CLAUDE.md")
    else:
        copy2(SCRIPT, repo / "scripts" / "open_pr.sh")
        copy2(AUDIT_SCRIPT, repo / "scripts" / "audit_pr_body.py")
        copy2(AI_RECONCILIATION_SCRIPT, repo / "scripts" / "audit_ai_reconciliation.py")
        copy2(CHANGE_POLICY_SCRIPT, repo / "scripts" / "_pr_change_policy.py")
        copy2(BRANCH_NAME_SCRIPT, repo / "scripts" / "check_pr_branch_name.py")
        (repo / "scripts" / "check_session_pr_ownership.py").write_text(
            """#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

log = os.environ.get("OWNERSHIP_GUARD_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv[1:]) + "\\n")
exit_code = int(os.environ.get("OWNERSHIP_GUARD_EXIT", "0"))
if exit_code:
    print("fake ownership guard failed", file=sys.stderr)
raise SystemExit(exit_code)
""",
            encoding="utf-8",
        )
        (repo / "scripts" / "check_session_pr_ownership.py").chmod(0o755)
        (repo / "scripts" / "local_pr_review.sh").write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf 'local_pr_review %s\\n' "${ATLAS_CURRENT_PR_BODY_FILE:-}" >> local-review.log
if [ "${LOCAL_REVIEW_ADVANCE_REMOTE:-}" = "1" ]; then
    git --git-dir="$(git config --get atlas.testRemoteGitDir)" update-ref refs/heads/claude/pr-test "$(git rev-parse HEAD^)"
fi
if [ "${LOCAL_REVIEW_ADVANCE_BOTH:-}" = "1" ]; then
    printf 'post-review\\n' >> scripts/example.py
    git add scripts/example.py
    git -c user.email=t@example.com -c user.name=t commit -qm post-review
    git push -q origin HEAD:claude/pr-test
fi
if [ "${LOCAL_REVIEW_MUTATE_BODY:-}" = "1" ]; then
    printf '\\npost-review body mutation\\n' >> "${ATLAS_CURRENT_PR_BODY_FILE}"
fi
exit "${LOCAL_REVIEW_EXIT:-0}"
""",
            encoding="utf-8",
        )
    (repo / "scripts" / "local_pr_review.sh").chmod(0o755)
    subprocess.run(["git", "init", "--initial-branch", "main"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    _git(repo, "config", f"url.{remote}.insteadOf", origin_url)
    _git(repo, "config", "atlas.testRemoteGitDir", str(remote))
    _git(repo, "remote", "add", "origin", origin_url)
    _git(repo, "push", "-q", "-u", "origin", "main")
    _git(repo, "switch", "-c", "claude/pr-test")
    return repo


def _write_body(repo: Path) -> Path:
    (repo / "plans").mkdir(exist_ok=True)
    (repo / "plans" / "PR-Test.md").write_text("# Test plan\n", encoding="utf-8")
    (repo / "scripts" / "example.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "plans/PR-Test.md", "scripts/example.py")
    _git(repo, "commit", "-qm", "planned change")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo.parent / "body.md"
    body.write_text(_valid_body(), encoding="utf-8")
    return body


def _write_docs_only_body(repo: Path) -> Path:
    doc = repo / "docs" / "example.md"
    doc.parent.mkdir()
    doc.write_text("# docs only\n", encoding="utf-8")
    _git(repo, "add", "docs/example.md")
    _git(repo, "commit", "-qm", "docs only")
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    body = repo.parent / "body-docs-only.md"
    body.write_text("Docs-only: true\n\nCorrect a documentation typo.\n", encoding="utf-8")
    return body


def _valid_body() -> str:
    return "\n".join([
        "Plan: plans/PR-Test.md",
        "Slice phase: Workflow/process",
        "Ownership lane: dev-workflow/process-gate-enrollment",
        "",
        "One-paragraph why.",
        "",
        "## Intentional",
        "- a trade-off",
        "",
        "## AI reconciliation",
        "- no-findings",
        "",
        "## Deferred",
        "- a follow-up",
        "",
        "## Parked hardening",
        "None.",
        "",
        "## Cold diff reconstruction",
        "- Changed: scripts/example.sh:1 updates the wrapper.",
        "- Contract match: traces to the body contract.",
        "- Gaps: none.",
        "",
        "## Verification",
        "- pytest passed",
        "",
        "## Mechanical verification",
        "- Command: pytest tests/test_open_pr_wrapper.py - Result: passed - Environment: local",
        "",
        "## Diff size",
        "2 files, +10 / -2",
    ])


def _fake_gh_env(tmp_path: Path, *, view_exit: int) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "gh-argv.txt"
    stdin_capture = tmp_path / "gh-stdin.txt"
    created_flag = tmp_path / "gh-created-pr"
    ownership_guard_log = tmp_path / "ownership-guard-argv.txt"
    state_file = tmp_path / "SESSION_STATE.codex-test.local.md"
    state_file.write_text(
        """# Atlas Builder Session State

## Owned Active PR

PR: none
Branch: claude/pr-test
Expected head SHA: none

## PRs This Session May Touch

- #17 Workflow wrapper -- fixture-owned PR.

## PRs This Session Must Not Touch

""",
        encoding="utf-8",
    )
    gh = bin_dir / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "repo" ] && [ "$2" = "view" ]; then
    printf 'canfieldjuan/ATLAS\\n'
    exit 0
fi
if [ "$1" = "pr" ] && [ "$2" = "list" ]; then
    if [ -n "${GH_PR_LIST_JSON:-}" ]; then
        printf '%s\\n' "${GH_PR_LIST_JSON}"
        exit 0
    fi
    current_branch="$(git branch --show-current)"
    if [ "${GH_VIEW_EXIT}" = "0" ] || [ -f "${GH_CREATED_PR_FLAG}" ]; then
        printf '[{"number":17,"headRefName":"%s","headRefOid":"%s","baseRefName":"main","headRepository":{"nameWithOwner":"canfieldjuan/ATLAS"},"isCrossRepository":false}]\\n' "$current_branch" "$(git rev-parse HEAD)"
    else
        printf '[]\\n'
    fi
    exit 0
fi
printf '%s\\n' "$*" > "${GH_ARGV_LOG}"
printf 'GH_PROMPT_DISABLED=%s\\n' "${GH_PROMPT_DISABLED:-}" > "${GH_PROMPT_STATE}"
cat > "${GH_STDIN_CAPTURE}"
if [ "$1" = "pr" ] && [ "$2" = "create" ]; then
    : > "${GH_CREATED_PR_FLAG}"
fi
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_VIEW_EXIT": str(view_exit),
        "GH_ARGV_LOG": str(log),
        "GH_STDIN_CAPTURE": str(stdin_capture),
        "GH_CREATED_PR_FLAG": str(created_flag),
        "GH_PROMPT_STATE": str(tmp_path / "gh-prompt-state.txt"),
        "OWNERSHIP_GUARD_LOG": str(ownership_guard_log),
        "ATLAS_SESSION_STATE_FILE": str(state_file),
        "PYTHONDONTWRITEBYTECODE": "1",
    }, log, stdin_capture


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
