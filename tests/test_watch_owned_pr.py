from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "watch_owned_pr.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_watcher(tmp_path: Path, *, scenario: str, sha: str = "head-a") -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        f"""\
        #!/usr/bin/env sh
        if [ "$4" = "origin/main:ci/gates.yml" ]; then
          printf '%s\n' 'gates:'
          printf '%s\n' '  - id: required-a'
          printf '%s\n' '    name: Required A'
          printf '%s\n' '    context: required-a'
          printf '%s\n' '    enforcement: branch_required # supported inline comment'
          printf '%s\n' '    trusted_base: true'
          printf '%s\n' '    workflow: .github/workflows/required_a.yml'
          printf '%s\n' '    local_command: null'
          printf '%s\n' '  - id: advisory-a'
          printf '%s\n' '    name: Advisory A'
          printf '%s\n' '    context: advisory-a'
          printf '%s\n' '    enforcement: advisory'
          printf '%s\n' '    trusted_base: false'
          printf '%s\n' '    workflow: .github/workflows/advisory_a.yml'
          printf '%s\n' '    local_command: null'
          exit 0
        fi
        if [ "$4" = "origin/main:scripts/check_required_status_checks.py" ]; then
          cat '{ROOT / "scripts" / "check_required_status_checks.py"}'
          exit 0
        fi
        exit 2
        """,
    )
    _write_executable(
        fake_bin / "gh",
        """\
        #!/usr/bin/env python3
        import json
        import os
        import sys

        args = sys.argv[1:]
        scenario = os.environ.get("WATCHER_SCENARIO", "ready")
        expected_sha = os.environ.get("SHA", "head-a")
        state_file = os.environ.get("WATCHER_STATE_FILE", "")
        if args[:2] == ["api", "repos/owner/repo/pulls/7"]:
            if scenario == "head_moves_before_ready":
                count = 0
                if state_file and os.path.exists(state_file):
                    with open(state_file, "r", encoding="utf-8") as handle:
                        count = int(handle.read() or "0")
                if state_file:
                    with open(state_file, "w", encoding="utf-8") as handle:
                        handle.write(str(count + 1))
                print("head-b" if count else "head-a")
                raise SystemExit(0)
            print(expected_sha)
        elif args[:2] == ["api", "--paginate"] and args[2] == f"repos/owner/repo/commits/{expected_sha}/check-runs?per_page=100":
            check_state_file = state_file + ".checks" if state_file else ""
            check_count = 0
            if check_state_file and os.path.exists(check_state_file):
                with open(check_state_file, "r", encoding="utf-8") as handle:
                    check_count = int(handle.read() or "0")
            if check_state_file:
                with open(check_state_file, "w", encoding="utf-8") as handle:
                    handle.write(str(check_count + 1))
            status = "in_progress" if scenario == "final_required_check_reruns" and check_count > 0 else "completed"
            conclusion = None if status == "in_progress" else "success"
            print(json.dumps({"check_runs": [
                {
                    "name": "required-a",
                    "status": status,
                    "conclusion": conclusion,
                    "started_at": "2026-07-27T00:00:00Z",
                    "app": {"id": 15368},
                }
            ]}))
        elif args[:2] == ["api", "graphql"]:
            joined = " ".join(args)
            if "reviewThreads" in joined:
                review_decision = ""
                if scenario == "clean_comment_changes_requested":
                    review_decision = "CHANGES_REQUESTED"
                if scenario == "final_decision_changes":
                    thread_state_file = state_file + ".threads" if state_file else ""
                    count = 0
                    if thread_state_file and os.path.exists(thread_state_file):
                        with open(thread_state_file, "r", encoding="utf-8") as handle:
                            count = int(handle.read() or "0")
                    if thread_state_file:
                        with open(thread_state_file, "w", encoding="utf-8") as handle:
                            handle.write(str(count + 1))
                    if count > 0:
                        review_decision = "CHANGES_REQUESTED"
                if scenario == "thread_graphql_errors":
                    print(json.dumps({"errors": [{"message": "partial failure"}], "data": {"repository": {"pullRequest": {"reviewThreads": None}}}}))
                    raise SystemExit(0)
                if scenario == "thread_malformed_page_info":
                    print(json.dumps({"data": {"repository": {"pullRequest": {
                        "state": "OPEN",
                        "merged": False,
                        "mergeable": "MERGEABLE",
                        "mergeStateStatus": "CLEAN",
                        "reviewDecision": "",
                        "reviewThreads": {
                            "pageInfo": {},
                            "nodes": [],
                        },
                    }}}}))
                    raise SystemExit(0)
                if scenario == "copilot_thread":
                    nodes = [{
                        "isResolved": False,
                        "isOutdated": False,
                        "comments": {"nodes": [{"author": {"login": "copilot-pull-request-reviewer[bot]"}}]},
                    }]
                elif scenario == "outdated_codex_thread":
                    nodes = [{
                        "isResolved": False,
                        "isOutdated": True,
                        "comments": {"nodes": [{"author": {"login": "chatgpt-codex-connector"}}]},
                    }]
                elif scenario == "codex_thread":
                    nodes = [{
                        "isResolved": False,
                        "isOutdated": False,
                        "comments": {"nodes": [{"author": {"login": "chatgpt-codex-connector"}}]},
                    }]
                else:
                    nodes = []
                print(json.dumps({"data": {"repository": {"pullRequest": {
                    "state": "OPEN",
                    "merged": False,
                    "mergeable": "MERGEABLE",
                    "mergeStateStatus": "CLEAN",
                    "reviewDecision": review_decision,
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": False},
                        "nodes": nodes,
                    },
                }}}}))
            elif "reviews(first:100" in joined:
                review_state_file = state_file + ".reviews" if state_file else ""
                review_count = 0
                if review_state_file and os.path.exists(review_state_file):
                    with open(review_state_file, "r", encoding="utf-8") as handle:
                        review_count = int(handle.read() or "0")
                if review_state_file:
                    with open(review_state_file, "w", encoding="utf-8") as handle:
                        handle.write(str(review_count + 1))
                if scenario in {"no_review", "clean_comment", "paginated_clean_comment", "wrong_author_clean_comment", "stale_clean_comment"}:
                    nodes = []
                    has_next = False
                    cursor = None
                elif scenario == "final_review_disappears" and review_count > 0:
                    nodes = []
                    has_next = False
                    cursor = None
                elif scenario == "review_graphql_errors":
                    print(json.dumps({"errors": [{"message": "partial failure"}], "data": {"repository": {"pullRequest": {"reviews": None}}}}))
                    raise SystemExit(0)
                elif scenario == "review_malformed_page_info":
                    print(json.dumps({"data": {"repository": {"pullRequest": {"reviews": {
                        "pageInfo": {},
                        "nodes": [],
                    }}}}}))
                    raise SystemExit(0)
                elif scenario == "paginated_review" and "cursor=c2" not in joined:
                    nodes = [{"author": {"login": "human"}, "commit": {"oid": expected_sha}, "state": "APPROVED"}]
                    has_next = True
                    cursor = "c2"
                elif scenario == "helper_review":
                    nodes = [{
                        "author": {"login": "codex-helper"},
                        "commit": {"oid": expected_sha},
                        "state": "COMMENTED",
                    }]
                    has_next = False
                    cursor = None
                elif scenario == "changes_requested_review":
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "commit": {"oid": expected_sha},
                        "state": "CHANGES_REQUESTED",
                    }]
                    has_next = False
                    cursor = None
                else:
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "commit": {"oid": expected_sha},
                        "state": "COMMENTED",
                    }]
                    has_next = False
                    cursor = None
                print(json.dumps({"data": {"repository": {"pullRequest": {"reviews": {
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    "nodes": nodes,
                }}}}}))
            elif "comments(first:100" in joined:
                comment_state_file = state_file + ".comments" if state_file else ""
                comment_count = 0
                if comment_state_file and os.path.exists(comment_state_file):
                    with open(comment_state_file, "r", encoding="utf-8") as handle:
                        comment_count = int(handle.read() or "0")
                if comment_state_file:
                    with open(comment_state_file, "w", encoding="utf-8") as handle:
                        handle.write(str(comment_count + 1))
                if scenario in {"clean_comment", "clean_comment_changes_requested"}:
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "body": "Codex Review: Didn't find any major issues\\n\\n**Reviewed commit:** `" + expected_sha[:10] + "`",
                        "bodyText": "",
                    }]
                    has_next = False
                    cursor = None
                elif scenario == "wrong_author_clean_comment":
                    nodes = [{
                        "author": {"login": "codex-helper"},
                        "body": "Codex Review: Didn't find any major issues\\n\\n**Reviewed commit:** `" + expected_sha[:10] + "`",
                        "bodyText": "",
                    }]
                    has_next = False
                    cursor = None
                elif scenario == "stale_clean_comment":
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "body": "Codex Review: Didn't find any major issues\\n\\n**Reviewed commit:** `bbbbbbbbbb`",
                        "bodyText": "",
                    }]
                    has_next = False
                    cursor = None
                elif scenario == "paginated_clean_comment" and "cursor=c2" not in joined:
                    nodes = []
                    has_next = True
                    cursor = "c2"
                elif scenario == "paginated_clean_comment":
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "body": "Codex Review: Didn't find any major issues\\n\\n**Reviewed commit:** `" + expected_sha[:10] + "`",
                        "bodyText": "",
                    }]
                    has_next = False
                    cursor = None
                elif scenario == "final_clean_comment_disappears" and comment_count > 0:
                    nodes = []
                    has_next = False
                    cursor = None
                elif scenario == "comment_graphql_errors":
                    print(json.dumps({"errors": [{"message": "partial failure"}], "data": {"repository": {"pullRequest": {"comments": None}}}}))
                    raise SystemExit(0)
                elif scenario == "comment_malformed_page_info":
                    print(json.dumps({"data": {"repository": {"pullRequest": {"comments": {
                        "pageInfo": {},
                        "nodes": [],
                    }}}}}))
                    raise SystemExit(0)
                else:
                    nodes = []
                    has_next = False
                    cursor = None
                print(json.dumps({"data": {"repository": {"pullRequest": {"comments": {
                    "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    "nodes": nodes,
                }}}}}))
            else:
                print("unexpected graphql query: " + joined, file=sys.stderr)
                raise SystemExit(2)
        else:
            print("unexpected gh args: " + repr(args), file=sys.stderr)
            raise SystemExit(2)
        """,
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "GH_TOKEN": "fake-token",
        "REPO": "owner/repo",
        "PR": "7",
        "SHA": sha,
        "CYCLES": "0",
        "WATCHER_SCENARIO": scenario,
        "WATCHER_STATE_FILE": str(tmp_path / "watcher-state.txt"),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_watcher_reports_ready_with_current_head_codex_review(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="ready")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "codex-head-attestations=1" in result.stdout


def test_watcher_ignores_unresolved_non_codex_thread(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="copilot_thread")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "threads=0" in result.stdout


def test_watcher_blocks_without_current_head_codex_review(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="no_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTIONABLE" not in result.stdout
    assert "watch window elapsed" in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_requires_exact_codex_connector_review_identity(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="helper_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_blocks_on_unresolved_codex_thread(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="codex_thread")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTIONABLE" in result.stdout
    assert "threads=1" in result.stdout


def test_watcher_blocks_on_outdated_unresolved_codex_thread(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="outdated_codex_thread")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTIONABLE" in result.stdout
    assert "threads=1" in result.stdout


def test_watcher_does_not_count_changes_requested_as_head_review(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="changes_requested_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_accepts_current_head_codex_clean_comment(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="clean_comment", sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "codex-head-attestations=1" in result.stdout


def test_watcher_finds_clean_comment_after_pagination(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="paginated_clean_comment", sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "attestation-pages=3" in result.stdout


def test_watcher_rejects_wrong_author_clean_comment(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="wrong_author_clean_comment", sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_rejects_stale_clean_comment(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="stale_clean_comment", sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_changes_requested_overrides_clean_comment(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="clean_comment_changes_requested", sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE" in result.stdout
    assert "decision=CHANGES_REQUESTED" in result.stdout


def test_watcher_retries_malformed_thread_snapshot(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="thread_graphql_errors")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "GraphQL reviewThreads snapshot incomplete/malformed" in result.stdout
    assert "watch window elapsed" in result.stdout


def test_watcher_retries_malformed_thread_page_info(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="thread_malformed_page_info")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "GraphQL reviewThreads snapshot incomplete/malformed" in result.stdout
    assert "watch window elapsed" in result.stdout


def test_watcher_finds_current_head_codex_review_after_pagination(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="paginated_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "attestation-pages=3" in result.stdout


def test_watcher_blocks_malformed_review_pagination(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="review_graphql_errors")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE" in result.stdout
    assert "attestations-complete=false" in result.stdout


def test_watcher_blocks_malformed_review_page_info(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="review_malformed_page_info")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE" in result.stdout
    assert "attestations-complete=false" in result.stdout


def test_watcher_blocks_malformed_comment_pagination(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="comment_graphql_errors")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE" in result.stdout
    assert "attestations-complete=false" in result.stdout


def test_watcher_blocks_malformed_comment_page_info(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="comment_malformed_page_info")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE" in result.stdout
    assert "attestations-complete=false" in result.stdout


def test_watcher_revalidates_head_before_reporting_ready(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="head_moves_before_ready")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "HEAD-MOVED: head-a -> head-b" in result.stdout


def test_watcher_revalidates_decision_before_reporting_ready(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="final_decision_changes")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE: final-read" in result.stdout
    assert "decision=CHANGES_REQUESTED" in result.stdout


def test_watcher_revalidates_reviews_before_reporting_ready(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="final_review_disappears")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE: final-read" in result.stdout
    assert "codex-head-attestations=0" in result.stdout


def test_watcher_revalidates_required_checks_before_reporting_ready(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="final_required_check_reruns")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "ACTIONABLE: final-read" in result.stdout
    assert "req-unsettled=1" in result.stdout
