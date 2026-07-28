from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pr_watcher.py"
INSTALLER = ROOT / "scripts" / "install_codex_wake_bridge.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


watcher = _load("pr_watcher", SCRIPT)
wake_bridge = _load("codex_wake_bridge_for_watcher_tests", ROOT / "scripts" / "codex_wake_bridge.py")


def _pr(*, head: str = "head-a", state: str = "OPEN", draft: bool = False, decision: str = "", merge: str = "CLEAN") -> dict[str, Any]:
    return {
        "number": 7,
        "title": "Watcher producer",
        "url": "https://github.test/owner/repo/pull/7",
        "baseRefName": "main",
        "headRefName": "claude/pr-watcher",
        "headRefOid": head,
        "mergeStateStatus": merge,
        "reviewDecision": decision,
        "isDraft": draft,
        "state": state,
    }


def _check(name: str, bucket: str = "pass") -> dict[str, Any]:
    return {"name": name, "bucket": bucket, "state": "SUCCESS"}


def _thread_page(nodes: list[dict[str, Any]] | None = None, *, has_next: bool = False, cursor: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": nodes or [],
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    }
                }
            }
        }
    }


def _review_page(nodes: list[dict[str, Any]] | None = None, *, has_next: bool = False, cursor: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviews": {
                        "nodes": nodes
                        if nodes is not None
                        else [
                            {
                                "author": {"login": "chatgpt-codex-connector"},
                                "commit": {"oid": "head-a"},
                                "state": "COMMENTED",
                            }
                        ],
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    }
                }
            }
        }
    }


def _comment_page(nodes: list[dict[str, Any]] | None = None, *, has_next: bool = False, cursor: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "comments": {
                        "nodes": nodes or [],
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                    }
                }
            }
        }
    }


def _thread(
    *,
    thread_id: str = "thread-1",
    resolved: bool = False,
    outdated: bool = False,
    author: str = "chatgpt-codex-connector",
    path: str = "scripts/pr_watcher.py",
    line: int = 12,
) -> dict[str, Any]:
    return {
        "id": thread_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "line": line,
        "comments": {"nodes": [{"author": {"login": author}}]},
    }


def _response(payload: Any, code: int = 0, stderr: str = "") -> tuple[int, str, str]:
    return code, json.dumps(payload), stderr


class FakeRun:
    def __init__(
        self,
        *,
        pr_responses: list[tuple[int, str, str]] | None = None,
        all_checks: tuple[int, str, str] | None = None,
        required_checks: tuple[int, str, str] | None = None,
        required_policy: tuple[int, str, str] | None = None,
        reviews: tuple[int, str, str] | None = None,
        thread_pages: list[tuple[int, str, str]] | None = None,
        review_pages: list[tuple[int, str, str]] | None = None,
        comment_pages: list[tuple[int, str, str]] | None = None,
        reconciliation: tuple[int, str, str] = (0, "clean", ""),
        git_status: tuple[int, str, str] = (0, "", ""),
    ) -> None:
        self.pr_responses = list(pr_responses or [_response(_pr()), _response(_pr()), _response(_pr())])
        self.last_pr_response = self.pr_responses[-1] if self.pr_responses else _response(_pr())
        self.all_checks = all_checks or _response([_check("required-a"), _check("optional-a")])
        self.required_checks = required_checks or _response([_check("required-a")])
        self.required_policy = required_policy or _response(
            {
                "contexts": ["required-a"],
                "checks": [{"context": "required-a", "app_id": 15368}],
            }
        )
        self.reviews = reviews or _response({"comments": [], "reviews": []})
        self.thread_pages = list(thread_pages or [_response(_thread_page())])
        self.review_pages = list(review_pages or [_response(_review_page())])
        self.comment_pages = list(comment_pages or [_response(_comment_page())])
        self.reconciliation = reconciliation
        self.git_status = git_status
        self.commands: list[list[str]] = []

    def __call__(self, command, *, cwd: Path):
        args = list(command)
        self.commands.append(args)
        if args[:3] == ["gh", "pr", "view"] and "--comments" not in args:
            if self.pr_responses:
                self.last_pr_response = self.pr_responses.pop(0)
            return self.last_pr_response
        if args[:3] == ["gh", "pr", "checks"]:
            return self.required_checks if "--required" in args else self.all_checks
        if args[:2] == ["gh", "api"] and args[2] != "graphql":
            return self.required_policy
        if args[:3] == ["gh", "pr", "view"] and "--comments" in args:
            return self.reviews
        if args[:3] == ["gh", "api", "graphql"]:
            query = " ".join(args)
            if "reviews(first:100" in query:
                return self.review_pages.pop(0)
            if "comments(first:100" in query:
                return self.comment_pages.pop(0)
            return self.thread_pages.pop(0)
        if (
            len(args) > 1
            and args[0] == sys.executable
            and Path(args[1]).name == watcher.RECONCILIATION_CHECKER_NAME
        ):
            return self.reconciliation
        if args[:3] == ["git", "status", "--porcelain"]:
            return self.git_status
        if args[:3] == ["systemctl", "--user", "disable"]:
            return 0, "", ""
        raise AssertionError(f"unexpected command: {args}")


def _config(tmp_path: Path, *, head: str = "head-a", extra: str = "") -> tuple[Path, Path, Path]:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    state_dir = tmp_path / "state"
    (config_dir / "session.env").write_text(
        textwrap.dedent(
            f"""\
            LABEL="Watcher producer"
            REPO_DIR="{repo_dir}"
            PR="7"
            REPO="owner/repo"
            HEAD_SHA="{head}"
            POLL_MINUTES="30"
            AUTO_MERGE="0"
            NOTIFY="0"
            {extra}
            """
        ),
        encoding="utf-8",
    )
    return config_dir, state_dir, repo_dir


def _produce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeRun, *, head: str = "head-a", extra: str = "") -> dict[str, Any]:
    config_dir, state_dir, _repo_dir = _config(tmp_path, head=head, extra=extra)
    monkeypatch.setattr(watcher, "_run", fake)
    code, status, _block = watcher.produce("session", config_dir=config_dir, state_dir=state_dir)
    assert code == 0
    assert json.loads((state_dir / "session.json").read_text(encoding="utf-8")) == status
    assert not list(state_dir.glob(".*.tmp"))
    return status


def test_valid_snapshot_is_ready_and_accepted_by_consumer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun()
    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "ready_for_human_merge"
    assert status["readiness"] == {
        "version": 1,
        "evaluated_head_sha": "head-a",
        "required_check_count": 1,
        "required_checks_complete": True,
        "required_check_failures": [],
        "required_check_pending": [],
        "review_threads_complete": True,
        "review_thread_pages_fetched": 1,
        "unresolved_review_threads": [],
        "codex_reviews_complete": True,
        "codex_review_pages_fetched": 2,
        "codex_head_review_count": 1,
        "review_decision": "",
        "merge_state_status": "CLEAN",
    }
    assert wake_bridge.readiness_blockers(status) == []
    reconciliation_commands = [
        command
        for command in fake.commands
        if len(command) > 1 and Path(command[1]).name == watcher.RECONCILIATION_CHECKER_NAME
    ]
    assert reconciliation_commands == [
        [
            sys.executable,
            str(watcher.TRUSTED_RECONCILIATION_CHECKER),
            "--pr",
            "7",
            "--repo",
            "owner/repo",
        ]
    ]
    assert watcher.TRUSTED_RECONCILIATION_CHECKER.parent.name == watcher.RECONCILIATION_LIB_DIR


def test_post_review_metadata_controls_readiness_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(
        pr_responses=[
            _response(_pr()),
            _response(_pr()),
            _response(_pr(decision="CHANGES_REQUESTED")),
        ]
    )

    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "attention"
    assert status["pr"]["reviewDecision"] == "CHANGES_REQUESTED"
    assert status["readiness"]["review_decision"] == "CHANGES_REQUESTED"
    assert "review decision has changes requested" in wake_bridge.readiness_blockers(status)


def test_thread_snapshot_is_collected_after_codex_review_pagination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(thread_pages=[_response(_thread_page([_thread()]))])

    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "attention"
    assert status["readiness"]["codex_head_review_count"] == 1
    assert status["readiness"]["unresolved_review_threads"] == [
        {
            "id": "thread-1",
            "is_outdated": False,
            "path": "scripts/pr_watcher.py",
            "line": 12,
        }
    ]
    graphql_kinds = [
        (
            "reviews"
            if "reviews(first:100" in " ".join(command)
            else "comments"
            if "comments(first:100" in " ".join(command)
            else "threads"
        )
        for command in fake.commands
        if command[:3] == ["gh", "api", "graphql"]
    ]
    assert graphql_kinds == ["reviews", "comments", "threads"]
    reconciliation_index = next(
        i
        for i, command in enumerate(fake.commands)
        if len(command) > 1 and Path(command[1]).name == watcher.RECONCILIATION_CHECKER_NAME
    )
    thread_index = next(
        i
        for i, command in enumerate(fake.commands)
        if command[:3] == ["gh", "api", "graphql"] and "reviews(first:100" not in " ".join(command)
    )
    assert thread_index < reconciliation_index


def test_invalid_repo_config_raises_before_transport(tmp_path: Path) -> None:
    config_dir, state_dir, _repo_dir = _config(tmp_path)
    path = config_dir / "session.env"
    path.write_text(path.read_text(encoding="utf-8").replace("owner/repo", "not-a-slug"), encoding="utf-8")

    with pytest.raises(ValueError, match="owner/name"):
        watcher.produce("session", config_dir=config_dir, state_dir=state_dir)


@pytest.mark.parametrize(
    ("required", "expected_state", "expected_fragment"),
    [
        ([], "pending", "required-a (not reported)"),
        ([_check("required-a", "pending")], "pending", "required-a"),
        ([_check("required-a", "fail")], "attention", "required-a (fail)"),
        ([_check("required-a", "cancel")], "attention", "required-a (cancel)"),
        ([_check("required-a", "skipping")], "attention", "required-a (skipping)"),
        ([_check("required-a", "mystery")], "attention", "required-a (mystery)"),
    ],
)
def test_required_check_boundaries_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required: list[dict[str, Any]],
    expected_state: str,
    expected_fragment: str | None,
) -> None:
    all_checks = required or [_check("optional-a")]
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(all_checks=_response(all_checks), required_checks=_response(required)),
    )

    assert status["state"] == expected_state
    assert status["readiness"]["required_checks_complete"] is False
    if expected_fragment:
        combined = status["readiness"]["required_check_failures"] + status["readiness"]["required_check_pending"]
        assert expected_fragment in combined
    assert wake_bridge.readiness_blockers(status)


def test_empty_required_policy_cannot_be_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(required_checks=_response([]), required_policy=_response({"contexts": [], "checks": []})),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["required_check_count"] == 0
    assert "required check count must be at least 1" in wake_bridge.readiness_blockers(status)


def test_reported_required_row_cannot_mask_empty_policy_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            required_checks=_response([_check("reported-only")]),
            required_policy=_response({"contexts": [], "checks": []}),
        ),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["required_check_count"] == 1
    assert status["readiness"]["required_checks_complete"] is False
    assert "required-status policy has no contexts/checks" in status["checks_error"]
    assert wake_bridge.readiness_blockers(status)


def test_review_change_is_actionable_while_checks_are_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            all_checks=_response([_check("required-a", "pending")]),
            required_checks=_response([_check("required-a", "pending")]),
            reviews=_response({"comments": [{"id": "new-review"}], "reviews": []}),
        ),
    )

    assert status["review_changed"] is True
    assert status["check_pending"] == ["required-a"]
    assert status["state"] == "review_changed"


@pytest.mark.parametrize("malformed", [["not-an-object"], [{"name": "x"}], [{"bucket": "pass"}]])
def test_malformed_required_check_rows_are_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: list[Any],
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(required_checks=_response(malformed)),
    )

    assert status["state"] == "attention"
    assert status["checks_error"]
    assert status["readiness"]["required_checks_complete"] is False


def test_optional_non_skipped_failure_still_blocks_green_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            all_checks=_response([_check("required-a"), _check("optional-a", "fail")]),
            required_checks=_response([_check("required-a")]),
        ),
    )

    assert status["state"] == "attention"
    assert status["check_failures"] == ["optional-a (fail)"]
    assert status["readiness"]["required_checks_complete"] is True


def test_optional_skipped_check_does_not_block_green_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            all_checks=_response([_check("required-a"), _check("optional-a", "skipping")]),
            required_checks=_response([_check("required-a")]),
        ),
    )

    assert status["state"] == "ready_for_human_merge"
    assert status["check_failures"] == []


def test_paginates_threads_and_keeps_outdated_unresolved_codex_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        _response(
            _thread_page(
                [
                    _thread(thread_id="outdated-codex", outdated=True),
                    _thread(thread_id="copilot-open", author="copilot-pull-request-reviewer[bot]"),
                ],
                has_next=True,
                cursor="cursor-1",
            )
        ),
        _response(_thread_page([_thread(thread_id="open-codex")])),
    ]

    status = _produce(tmp_path, monkeypatch, FakeRun(thread_pages=pages))

    assert status["state"] == "attention"
    assert status["readiness"]["review_threads_complete"] is True
    assert status["readiness"]["review_thread_pages_fetched"] == 2
    assert status["readiness"]["unresolved_review_threads"] == [
        {"id": "outdated-codex", "is_outdated": True, "path": "scripts/pr_watcher.py", "line": 12},
        {"id": "open-codex", "is_outdated": False, "path": "scripts/pr_watcher.py", "line": 12},
    ]


def test_changes_requested_codex_review_does_not_satisfy_head_review_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            review_pages=[
                _response(
                    _review_page(
                        [
                            {
                                "author": {"login": "chatgpt-codex-connector"},
                                "commit": {"oid": "head-a"},
                                "state": "CHANGES_REQUESTED",
                            }
                        ]
                    )
                )
            ],
        ),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["codex_head_review_count"] == 0


def test_unresolved_non_codex_thread_does_not_block_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            thread_pages=[
                _response(
                    _thread_page(
                        [
                            _thread(
                                thread_id="copilot-open",
                                author="copilot-pull-request-reviewer[bot]",
                            )
                        ]
                    )
                )
            ],
        ),
    )

    assert status["state"] == "ready_for_human_merge"
    assert status["readiness"]["unresolved_review_threads"] == []


def test_current_head_codex_review_is_required_for_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            review_pages=[
                _response(
                    _review_page(
                        [
                            {
                                "author": {"login": "chatgpt-codex-connector"},
                                "commit": {"oid": "old-head"},
                                "state": "COMMENTED",
                            }
                        ]
                    )
                )
            ],
        ),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["codex_reviews_complete"] is True
    assert status["readiness"]["codex_head_review_count"] == 0
    assert "current-head Codex review attestation is missing" in wake_bridge.readiness_blockers(status)


def test_current_head_review_requires_exact_codex_connector_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            review_pages=[
                _response(
                    _review_page(
                        [
                            {
                                "author": {"login": "codex-helper"},
                                "commit": {"oid": "head-a"},
                                "state": "COMMENTED",
                            }
                        ]
                    )
                )
            ],
        ),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["codex_head_review_count"] == 0


def test_codex_review_pagination_reaches_later_current_head_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            review_pages=[
                _response(
                    _review_page(
                        [
                            {
                                "author": {"login": "human-reviewer"},
                                "commit": {"oid": "head-a"},
                                "state": "APPROVED",
                            }
                        ],
                        has_next=True,
                        cursor="review-cursor",
                    )
                ),
                _response(_review_page()),
            ],
        ),
    )

    assert status["state"] == "ready_for_human_merge"
    assert status["readiness"]["codex_review_pages_fetched"] == 3
    assert status["readiness"]["codex_head_review_count"] == 1


def test_codex_clean_comment_attests_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            pr_responses=[_response(_pr(head=head)), _response(_pr(head=head)), _response(_pr(head=head))],
            review_pages=[_response(_review_page(nodes=[]))],
            comment_pages=[
                _response(
                    _comment_page(
                        [
                            {
                                "author": {"login": "chatgpt-codex-connector"},
                                "body": "Codex Review: Didn't find any major issues\n\n**Reviewed commit:** `aaaaaaaaaa`",
                                "bodyText": "",
                            }
                        ]
                    )
                )
            ],
        ),
        head=head,
    )

    assert status["state"] == "ready_for_human_merge"
    assert status["readiness"]["codex_head_review_count"] == 1


def test_authorless_comment_is_ignored_for_codex_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            pr_responses=[_response(_pr(head=head)), _response(_pr(head=head)), _response(_pr(head=head))],
            review_pages=[_response(_review_page(nodes=[]))],
            comment_pages=[
                _response(
                    _comment_page(
                        [
                            {
                                "author": None,
                                "body": "Codex Review: Didn't find any major issues\n\n**Reviewed commit:** `aaaaaaaaaa`",
                                "bodyText": "",
                            }
                        ]
                    )
                )
            ],
        ),
        head=head,
    )

    assert status["state"] == "attention"
    assert status["readiness"]["codex_reviews_complete"] is True
    assert status["readiness"]["codex_head_review_count"] == 0


def test_codex_comment_pagination_reaches_later_current_head_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(
            pr_responses=[_response(_pr(head=head)), _response(_pr(head=head)), _response(_pr(head=head))],
            review_pages=[_response(_review_page(nodes=[]))],
            comment_pages=[
                _response(_comment_page(has_next=True, cursor="comment-cursor")),
                _response(
                    _comment_page(
                        [
                            {
                                "author": {"login": "chatgpt-codex-connector"},
                                "body": "Codex Review: Didn't find any major issues\n\n**Reviewed commit:** `aaaaaaaaaa`",
                                "bodyText": "",
                            }
                        ]
                    )
                ),
            ],
        ),
        head=head,
    )

    assert status["state"] == "ready_for_human_merge"
    assert status["readiness"]["codex_review_pages_fetched"] == 3
    assert status["readiness"]["codex_head_review_count"] == 1


def test_codex_review_pagination_failure_blocks_ready_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(watcher, "MAX_REVIEW_PAGES", 1)
    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(review_pages=[_response(_review_page(has_next=True, cursor="again"))]),
    )

    assert status["state"] == "attention"
    assert status["readiness"]["codex_reviews_complete"] is False
    assert "review pagination exceeded 1 pages" in status["codex_reviews_error"]


def test_head_change_during_collection_is_attention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun(pr_responses=[_response(_pr(head="head-a")), _response(_pr(head="head-b"))])

    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "attention"
    assert status["head_mismatch"] is True
    assert status["pr"]["headRefOid"] == "head-b"
    assert status["readiness"]["evaluated_head_sha"] == "head-a"
    assert "evaluated head SHA does not match PR head" in wake_bridge.readiness_blockers(status)


def test_head_change_after_codex_review_pagination_is_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRun(
        pr_responses=[
            _response(_pr(head="head-a")),
            _response(_pr(head="head-a")),
            _response(_pr(head="head-b")),
        ]
    )

    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "attention"
    assert status["head_mismatch"] is True
    assert status["pr"]["headRefOid"] == "head-a"
    assert status["readiness"]["evaluated_head_sha"] == "head-a"
    assert "head_mismatch" in wake_bridge.readiness_blockers(status)


def test_base_change_during_collection_invalidates_required_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _pr()
    final = _pr()
    final["baseRefName"] = "release"

    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(pr_responses=[_response(initial), _response(final)]),
    )

    assert status["state"] == "attention"
    assert "base branch changed during watcher observation" in status["view_error"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("number", 0),
        ("headRefOid", ""),
        ("isDraft", "false"),
        ("reviewDecision", 7),
        ("mergeStateStatus", None),
    ],
)
def test_malformed_pr_metadata_is_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    malformed = _pr()
    malformed[field] = replacement

    status = _produce(
        tmp_path,
        monkeypatch,
        FakeRun(pr_responses=[_response(malformed), _response(malformed)]),
    )

    assert status["state"] == "attention"
    assert status["view_error"]


@pytest.mark.parametrize(
    ("fake", "field"),
    [
        (FakeRun(all_checks=(2, "", "checks unavailable")), "checks_error"),
        (FakeRun(required_checks=(2, "", "required unavailable")), "checks_error"),
        (FakeRun(required_policy=(2, "", "policy unavailable")), "checks_error"),
        (FakeRun(required_policy=_response({"contexts": "required-a", "checks": []})), "checks_error"),
        (FakeRun(reviews=(2, "", "reviews unavailable")), "reviews_error"),
        (FakeRun(thread_pages=[_response({"errors": [{"message": "denied"}]})]), "review_threads_error"),
        (
            FakeRun(pr_responses=[_response(_pr()), (2, "", "refresh unavailable")]),
            "view_error",
        ),
    ],
)
def test_github_read_failures_are_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeRun,
    field: str,
) -> None:
    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == "attention"
    assert status[field]


def test_required_policy_transport_failure_reaches_attention_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, state_dir, _repo_dir = _config(tmp_path)

    def transport(command, **_kwargs):
        args = list(command)
        if args[:3] == ["gh", "pr", "view"] and "--comments" not in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(_pr()), "")
        if args[:3] == ["gh", "pr", "checks"]:
            return subprocess.CompletedProcess(args, 0, json.dumps([_check("required-a")]), "")
        if args[:2] == ["gh", "api"] and args[2] != "graphql":
            return subprocess.CompletedProcess(args, 1, "", "policy denied")
        if args[:3] == ["gh", "pr", "view"] and "--comments" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps({"comments": [], "reviews": []}), "")
        if args[:3] == ["gh", "api", "graphql"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(_thread_page()), "")
        if len(args) > 1 and args[0] == sys.executable:
            return subprocess.CompletedProcess(args, 0, "clean", "")
        if args[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected transport command: {args}")

    monkeypatch.setattr(watcher.subprocess, "run", transport)

    code, status, _block = watcher.produce("session", config_dir=config_dir, state_dir=state_dir)

    assert code == 0
    assert status["state"] == "attention"
    assert "policy denied" in status["checks_error"]


@pytest.mark.parametrize(
    ("exception", "expected_code", "expected_text"),
    [
        (subprocess.TimeoutExpired(["gh"], watcher.COMMAND_TIMEOUT_SECONDS), 124, "timed out"),
        (OSError("missing executable"), 127, "could not run"),
    ],
)
def test_command_transport_errors_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception: BaseException,
    expected_code: int,
    expected_text: str,
) -> None:
    def fail(*_args, **_kwargs):
        raise exception

    monkeypatch.setattr(watcher.subprocess, "run", fail)

    code, stdout, stderr = watcher._run(["gh", "pr", "view"], cwd=tmp_path)

    assert code == expected_code
    assert stdout == ""
    assert expected_text in stderr


def test_missing_cursor_and_page_cap_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _config(tmp_path)
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(watcher, "_run_json", lambda *args, **kwargs: (_thread_page(has_next=True, cursor=None), None))
    unresolved, pages, complete, error = watcher._fetch_threads(7, "owner/repo", cwd=repo_dir)
    assert unresolved == []
    assert pages == 1
    assert complete is False
    assert error == "GraphQL pagination cursor is missing"

    monkeypatch.setattr(watcher, "MAX_THREAD_PAGES", 2)
    monkeypatch.setattr(watcher, "_run_json", lambda *args, **kwargs: (_thread_page(has_next=True, cursor="again"), None))
    _unresolved, pages, complete, error = watcher._fetch_threads(7, "owner/repo", cwd=repo_dir)
    assert pages == 2
    assert complete is False
    assert error == "review-thread pagination exceeded 2 pages"


@pytest.mark.parametrize(
    "node",
    [
        {"id": "", "isResolved": False, "isOutdated": False, "path": "x.py", "line": 1},
        {"id": "t", "isResolved": "no", "isOutdated": False, "path": "x.py", "line": 1},
        {"id": "t", "isResolved": False, "isOutdated": False, "path": 7, "line": 1},
        {"id": "t", "isResolved": False, "isOutdated": False, "path": "x.py", "line": True},
    ],
)
def test_malformed_review_thread_nodes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node: dict[str, Any],
) -> None:
    _config(tmp_path)
    repo_dir = tmp_path / "repo"
    monkeypatch.setattr(
        watcher,
        "_run_json",
        lambda *args, **kwargs: (_thread_page([node]), None),
    )

    unresolved, pages, complete, error = watcher._fetch_threads(7, "owner/repo", cwd=repo_dir)

    assert unresolved == []
    assert pages == 1
    assert complete is False
    assert error


@pytest.mark.parametrize(
    ("pr_value", "all_checks", "reconciliation", "git_status", "expected"),
    [
        (_pr(draft=True), None, (0, "clean", ""), (0, "", ""), "attention"),
        (_pr(decision="CHANGES_REQUESTED"), None, (0, "clean", ""), (0, "", ""), "attention"),
        (_pr(merge="DIRTY"), None, (0, "clean", ""), (0, "", ""), "attention"),
        (_pr(merge="UNSTABLE"), [_check("required-a", "pending")], (0, "clean", ""), (0, "", ""), "pending"),
        (_pr(state="MERGED"), None, (0, "clean", ""), (0, "", ""), "closed"),
        (_pr(), None, (1, "", "reconciliation failed"), (0, "", ""), "attention"),
        (_pr(), None, (0, "clean", ""), (0, " M file.py", ""), "attention"),
    ],
)
def test_control_flow_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pr_value: dict[str, Any],
    all_checks: list[dict[str, Any]] | None,
    reconciliation: tuple[int, str, str],
    git_status: tuple[int, str, str],
    expected: str,
) -> None:
    checks = all_checks or [_check("required-a")]
    fake = FakeRun(
        pr_responses=[_response(pr_value), _response(pr_value)],
        all_checks=_response(checks),
        required_checks=_response(checks[:1]),
        reconciliation=reconciliation,
        git_status=git_status,
    )

    status = _produce(tmp_path, monkeypatch, fake)

    assert status["state"] == expected


def test_missing_expected_head_and_truthy_auto_merge_are_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _produce(tmp_path, monkeypatch, FakeRun(), head="", extra='AUTO_MERGE="1"')

    assert status["state"] == "attention"
    assert status["head_mismatch"] is True
    assert "unsafe auto-merge config ignored" in status["merge_error"]


def test_session_receipt_sanitizes_multiline_marker_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = tmp_path / "SESSION_STATE.local.md"
    session_state.write_text("# State\n", encoding="utf-8")
    poisoned = _pr()
    poisoned["title"] = f"title\n{watcher.MARKER_END}\nmalicious"
    fake = FakeRun(pr_responses=[_response(poisoned), _response(poisoned)])

    _produce(
        tmp_path,
        monkeypatch,
        fake,
        extra=f'SESSION_STATE="{session_state}"',
    )

    text = session_state.read_text(encoding="utf-8")
    assert text.count(watcher.MARKER_START) == 1
    assert text.count(watcher.MARKER_END) == 1
    assert "PR: #7 title [watcher-marker] malicious" in text


def test_malformed_previous_counts_become_attention_instead_of_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir, state_dir, _repo_dir = _config(tmp_path)
    state_dir.mkdir()
    (state_dir / "session.json").write_text(
        json.dumps({"comment_count": "many", "review_count": False}),
        encoding="utf-8",
    )
    monkeypatch.setattr(watcher, "_run", FakeRun())

    code, status, _block = watcher.produce("session", config_dir=config_dir, state_dir=state_dir)

    assert code == 0
    assert status["state"] == "attention"
    assert "invalid comment_count" in status["previous_state_error"]
    assert "invalid review_count" in status["previous_state_error"]


def test_secondary_session_receipt_failure_does_not_block_primary_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = tmp_path / "SESSION_STATE.local.md"
    session_state.write_text("# State\n", encoding="utf-8")
    config_dir, state_dir, _repo_dir = _config(
        tmp_path,
        extra=f'SESSION_STATE="{session_state}"',
    )
    monkeypatch.setattr(watcher, "_run", FakeRun())

    def fail_receipt(_path: Path, _block: str) -> None:
        raise OSError("read-only session state")

    monkeypatch.setattr(watcher, "_write_session_state", fail_receipt)

    code, status, block = watcher.produce("session", config_dir=config_dir, state_dir=state_dir)

    assert code == 0
    assert status["state"] == "ready_for_human_merge"
    assert (state_dir / "session.json").exists()
    assert "Receipt errors: session-state receipt failed" in block


def test_installed_entrypoint_writes_consumer_accepted_snapshot(tmp_path: Path) -> None:
    bin_dir = tmp_path / "installed"
    systemd_dir = tmp_path / "systemd"
    install = subprocess.run(
        [sys.executable, str(INSTALLER), "--bin-dir", str(bin_dir), "--systemd-dir", str(systemd_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            args = sys.argv[1:]
            metadata = {
                "number": 7,
                "title": "Installed producer",
                "url": "https://github.test/owner/repo/pull/7",
                "baseRefName": "main",
                "baseRefOid": "base-a",
                "changedFiles": 0,
                "headRefName": "claude/pr-watcher",
                "headRefOid": "head-a",
                "mergeStateStatus": "CLEAN",
                "reviewDecision": "",
                "isDraft": False,
                "state": "OPEN",
            }
            if args[:2] == ["pr", "checks"]:
                print(json.dumps([{"name": "required-a", "bucket": "pass", "state": "SUCCESS"}]))
            elif args[:2] == ["pr", "view"] and "-q" in args:
                print("")
            elif args[:2] == ["pr", "view"] and "--comments" in args:
                print(json.dumps({"comments": [], "reviews": []}))
            elif args[:2] == ["pr", "view"]:
                print(json.dumps(metadata))
            elif args[:2] == ["api", "repos/owner/repo/branches/main/protection/required_status_checks"]:
                print(json.dumps({"contexts": ["required-a"], "checks": [{"context": "required-a", "app_id": 15368}]}))
            elif args[:2] == ["api", "repos/owner/repo/git/trees/base-a?recursive=1"]:
                print(json.dumps({"truncated": False, "tree": []}))
            elif args[:2] == ["api", "repos/owner/repo/git/trees/head-a?recursive=1"]:
                print(json.dumps({"truncated": False, "tree": []}))
            elif args and args[0] == "api" and any("repos/owner/repo/pulls/7/files" in arg for arg in args):
                print("")
            elif args[:2] == ["api", "graphql"]:
                query = " ".join(args)
                if "reviews(first:100" in query:
                    payload = {"data": {"repository": {"pullRequest": {"headRefOid": "head-a", "reviews": {"nodes": [{"author": {"login": "chatgpt-codex-connector"}, "commit": {"oid": "head-a"}, "state": "COMMENTED"}], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
                elif "comments(first:100" in query:
                    payload = {"data": {"repository": {"pullRequest": {"headRefOid": "head-a", "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
                elif "comments(first:1)" in query:
                    payload = {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
                else:
                    payload = {"data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}}
                print(json.dumps(payload))
            else:
                print("unexpected gh args: " + repr(args), file=sys.stderr)
                raise SystemExit(2)
            """
        ),
        encoding="utf-8",
    )
    fake_gh.chmod(fake_gh.stat().st_mode | stat.S_IXUSR)
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    state_dir = tmp_path / "state"
    (config_dir / "installed.env").write_text(
        textwrap.dedent(
            f"""\
            LABEL="Installed producer"
            REPO_DIR="{ROOT}"
            PR="7"
            REPO="owner/repo"
            HEAD_SHA="head-a"
            POLL_MINUTES="30"
            AUTO_MERGE="0"
            NOTIFY="0"
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        [str(bin_dir / "atlas-pr-watch"), "installed", "--config-dir", str(config_dir), "--state-dir", str(state_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((state_dir / "installed.json").read_text(encoding="utf-8"))
    assert status["state"] == "ready_for_human_merge"
    assert wake_bridge.readiness_blockers(status) == []
