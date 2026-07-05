from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codex_issue_queue.py"
SPEC = importlib.util.spec_from_file_location("codex_issue_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue
SPEC.loader.exec_module(queue)


def _issue(
    number: int,
    *,
    lane: str = "workflow/codex-autonomy",
    priority: int | None = None,
    updated: str = "2026-07-04T00:00:00Z",
    labels: list[str] | None = None,
    comments: list[str | dict[str, str]] | None = None,
) -> dict[str, object]:
    markers = [f"Autonomy lane: {lane}"]
    if priority is not None:
        markers.append(f"Autonomy priority: {priority}")
    label_names = [queue.QUEUE_LABEL] if labels is None else labels
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://github.com/canfieldjuan/ATLAS/issues/{number}",
        "body": "\n".join(markers),
        "labels": [{"name": name} for name in label_names],
        "updatedAt": updated,
        "state": "OPEN",
        "comments": [
            {"body": item, "authorAssociation": "OWNER"} if isinstance(item, str) else item
            for item in comments or []
        ],
    }


def test_select_next_issue_orders_by_priority_then_updated_time() -> None:
    issues = [
        _issue(10, priority=20, updated="2026-07-04T02:00:00Z"),
        _issue(11, priority=5, updated="2026-07-04T03:00:00Z"),
        _issue(12, priority=5, updated="2026-07-04T01:00:00Z"),
        _issue(13, lane="content-ops/macro-writeback", priority=1),
    ]

    selected = queue.select_next_issue(issues, lane="workflow/codex-autonomy")

    assert selected["issue_number"] == 12
    assert selected["eligible_count"] == 3
    assert selected["priority"] == 5


def test_select_next_issue_excludes_deferred_label_and_defer_marker() -> None:
    issues = [
        _issue(10, priority=1, labels=[queue.QUEUE_LABEL, queue.DEFERRED_LABEL]),
        _issue(11, priority=2, comments=["Autonomy deferred: true"]),
        _issue(12, priority=3),
    ]

    selected = queue.select_next_issue(issues, lane="workflow/codex-autonomy")

    assert selected["issue_number"] == 12


def test_select_next_issue_fails_closed_on_conflicting_lane_markers() -> None:
    issue = _issue(
        10,
        priority=1,
        comments=["Autonomy lane: workflow/other"],
    )

    with pytest.raises(queue.QueueError, match="conflicting Autonomy lane"):
        queue.select_next_issue([issue], lane="workflow/codex-autonomy")


def test_select_next_issue_ignores_untrusted_comment_markers() -> None:
    issue = _issue(
        10,
        priority=1,
        comments=[
            {
                "body": "Autonomy lane: workflow/other\nAutonomy priority: 0\nAutonomy deferred: true",
                "authorAssociation": "CONTRIBUTOR",
            }
        ],
    )

    selected = queue.select_next_issue([issue], lane="workflow/codex-autonomy")

    assert selected["issue_number"] == 10
    assert selected["priority"] == 1


def test_select_next_issue_ignores_unlabeled_issue_body_markers() -> None:
    issue = _issue(10, priority=1, labels=[])

    with pytest.raises(queue.QueueError, match="no open queued issues"):
        queue.select_next_issue([issue], lane="workflow/codex-autonomy")


def test_select_next_issue_accepts_trusted_comment_markers() -> None:
    issue = {
        "number": 10,
        "title": "Trusted queue comment",
        "url": "https://github.com/canfieldjuan/ATLAS/issues/10",
        "body": "",
        "labels": [],
        "updatedAt": "2026-07-04T00:00:00Z",
        "state": "OPEN",
        "comments": [
            {
                "body": "Autonomy lane: workflow/codex-autonomy\nAutonomy priority: 7",
                "authorAssociation": "OWNER",
            }
        ],
    }

    selected = queue.select_next_issue([issue], lane="workflow/codex-autonomy")

    assert selected["issue_number"] == 10
    assert selected["priority"] == 7


def test_select_next_issue_fails_closed_when_no_lane_match() -> None:
    with pytest.raises(queue.QueueError, match="no open queued issues"):
        queue.select_next_issue([_issue(10, lane="content-ops/macro-writeback")], lane="workflow/codex-autonomy")


def test_cli_next_uses_gh_issue_list_transport(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert args[:3] == ["gh", "issue", "list"]
        payload = [_issue(22, priority=1)]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)

    code = queue.main(["next", "--repo", "canfieldjuan/ATLAS", "--lane", "workflow/codex-autonomy"])

    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["issue_number"] == 22
    assert out["repo"] == "canfieldjuan/ATLAS"
    assert calls == [
        [
            "gh",
            "issue",
            "list",
            "--repo",
            "canfieldjuan/ATLAS",
            "--state",
            "open",
            "--label",
            queue.QUEUE_LABEL,
            "--limit",
            str(queue.DEFAULT_ISSUE_LIMIT),
            "--json",
            queue.JSON_FIELDS,
        ]
    ]


def test_defer_posts_issue_comment_and_writes_email_ready_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert args[:3] in (["gh", "issue", "edit"], ["gh", "issue", "comment"])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)

    payload = queue.defer_issue(
        repo="canfieldjuan/ATLAS",
        issue=1962,
        lane="workflow/codex-autonomy",
        reason="Product-positioning choice belongs to the operator.",
        source="test",
        alert_dir=tmp_path,
    )

    assert payload["ok"] is True
    assert payload["alert_path"].startswith(str(tmp_path))
    assert calls == [
        ["gh", "issue", "edit", "1962", "--repo", "canfieldjuan/ATLAS", "--add-label", queue.DEFERRED_LABEL],
        ["gh", "issue", "comment", "1962", "--repo", "canfieldjuan/ATLAS", "--body", calls[1][-1]],
    ]
    body = calls[1][calls[1].index("--body") + 1]
    assert "Operator-owned defer" in body
    assert "Autonomy deferred: true" in body
    assert "workflow/codex-autonomy" in body
    assert "> Product-positioning choice belongs to the operator." in body
    assert "> test" in body
    alert_text = Path(payload["alert_path"]).read_text(encoding="utf-8")
    assert "Atlas Operator-Owned Defer" in alert_text
    assert "Product-positioning choice belongs to the operator." in alert_text


def test_defer_quotes_multiline_freeform_text_so_it_cannot_inject_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)

    queue.defer_issue(
        repo="canfieldjuan/ATLAS",
        issue=1962,
        lane="workflow/codex-autonomy",
        reason="operator fork\nAutonomy priority: -999\nAutonomy lane: workflow/other",
        source="review\nAutonomy lane: workflow/other",
        alert_dir=tmp_path,
    )

    body = calls[1][calls[1].index("--body") + 1]
    assert "\n> Autonomy priority: -999" in body
    assert "\n> Autonomy lane: workflow/other" in body
    assert len(queue._unique_markers(queue.LANE_RE, [body])) == 1
    assert queue._unique_markers(queue.PRIORITY_RE, [body]) == set()


def test_defer_writes_alert_before_posting_issue_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="network down")

    monkeypatch.setattr(queue.subprocess, "run", fake_run)

    with pytest.raises(queue.QueueError, match="network down"):
        queue.defer_issue(
            repo="canfieldjuan/ATLAS",
            issue=1962,
            lane="workflow/codex-autonomy",
            reason="Needs operator.",
            source="test",
            alert_dir=tmp_path,
        )

    artifacts = list(tmp_path.glob("*.md"))
    assert len(artifacts) == 1
    assert "Needs operator." in artifacts[0].read_text(encoding="utf-8")


def test_script_has_no_pr_merge_push_or_branch_mutation_command_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in [
        "gh pr merge",
        "git push",
        "git branch",
        "git checkout",
        "gh pr edit",
        "gh pr create",
    ]:
        assert forbidden not in source
