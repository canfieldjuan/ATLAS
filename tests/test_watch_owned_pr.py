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


def _run_watcher(tmp_path: Path, *, scenario: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        """\
        #!/usr/bin/env sh
        if [ "$4" = "origin/main:scripts/check_required_status_checks.py" ]; then
          printf '%s\n' 'GITHUB_ACTIONS_APP_ID = 15368'
          printf '%s\n' 'DEFAULT_REQUIRED_CONTEXTS = ('
          printf '%s\n' '    "required-a",'
          printf '%s\n' ')'
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
        if args[:2] == ["api", "repos/owner/repo/pulls/7"]:
            print("head-a")
        elif args[:3] == ["api", "--paginate", "repos/owner/repo/commits/head-a/check-runs?per_page=100"]:
            print(json.dumps({"check_runs": [
                {
                    "name": "required-a",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-07-27T00:00:00Z",
                    "app": {"id": 15368},
                }
            ]}))
        elif args[:2] == ["api", "graphql"]:
            joined = " ".join(args)
            if "reviewThreads" in joined:
                if scenario == "copilot_thread":
                    nodes = [{
                        "isResolved": False,
                        "isOutdated": False,
                        "comments": {"nodes": [{"author": {"login": "copilot-pull-request-reviewer[bot]"}}]},
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
                    "reviewDecision": "",
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": False},
                        "nodes": nodes,
                    },
                }}}}))
            elif "reviews(first:100" in joined:
                if scenario == "no_review":
                    nodes = []
                    has_next = False
                    cursor = None
                elif scenario == "paginated_review" and "cursor=c2" not in joined:
                    nodes = [{"author": {"login": "human"}, "commit": {"oid": "head-a"}, "state": "APPROVED"}]
                    has_next = True
                    cursor = "c2"
                elif scenario == "helper_review":
                    nodes = [{
                        "author": {"login": "codex-helper"},
                        "commit": {"oid": "head-a"},
                        "state": "COMMENTED",
                    }]
                    has_next = False
                    cursor = None
                else:
                    nodes = [{
                        "author": {"login": "chatgpt-codex-connector"},
                        "commit": {"oid": "head-a"},
                        "state": "COMMENTED",
                    }]
                    has_next = False
                    cursor = None
                print(json.dumps({"data": {"repository": {"pullRequest": {"reviews": {
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
        "SHA": "head-a",
        "CYCLES": "0",
        "WATCHER_SCENARIO": scenario,
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
    assert "codex-head-reviews=1" in result.stdout


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
    assert "codex-head-reviews=0" in result.stdout


def test_watcher_requires_exact_codex_connector_review_identity(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="helper_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" not in result.stdout
    assert "codex-head-reviews=0" in result.stdout


def test_watcher_blocks_on_unresolved_codex_thread(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="codex_thread")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ACTIONABLE" in result.stdout
    assert "threads=1" in result.stdout


def test_watcher_finds_current_head_codex_review_after_pagination(tmp_path: Path) -> None:
    result = _run_watcher(tmp_path, scenario="paginated_review")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MERGE-READY" in result.stdout
    assert "review-pages=2" in result.stdout
