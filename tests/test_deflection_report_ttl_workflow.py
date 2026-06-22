from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "content_ops_deflection_report_ttl_purge.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ttl_purge_workflow_runs_only_on_schedule_or_manual_dispatch() -> None:
    text = _workflow_text()

    assert "  schedule:" in text
    assert '    - cron: "23 8 * * *"' in text
    assert "  workflow_dispatch:" in text
    assert "  pull_request:" not in text
    assert "  push:" not in text


def test_ttl_purge_workflow_defaults_to_30_days_and_bounded_batches() -> None:
    text = _workflow_text()

    assert "      retention_days:" in text
    assert '        default: "30"' in text
    assert "      limit:" in text
    assert '        default: "1000"' in text
    assert 'retention_days="${RETENTION_DAYS:-30}"' in text
    assert 'delete_limit="${DELETE_LIMIT:-1000}"' in text


def test_ttl_purge_workflow_keeps_database_url_out_of_argv() -> None:
    text = _workflow_text()

    assert "EXTRACTED_DATABASE_URL: ${{ secrets.EXTRACTED_DATABASE_URL }}" in text
    assert "--database-url-env EXTRACTED_DATABASE_URL" in text
    assert re.search(r"--database-url(?!-env)\b", text) is None


def test_ttl_purge_workflow_deletes_by_default_but_manual_dispatch_can_dry_run() -> None:
    text = _workflow_text()

    assert "      dry_run:" in text
    assert "        type: boolean" in text
    assert "        default: false" in text
    assert 'dry_run="${DRY_RUN:-false}"' in text
    assert 'if [ "$dry_run" != "true" ]; then' in text
    assert "args+=(--confirm-delete)" in text


def test_ttl_purge_workflow_has_minimal_permissions_and_pinned_actions() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read" in text
    assert "pull-requests: write" not in text
    assert re.search(r"uses: actions/checkout@[0-9a-f]{40}", text)
    assert re.search(r"uses: actions/setup-python@[0-9a-f]{40}", text)
    assert "uses: actions/checkout@v" not in text
    assert "uses: actions/setup-python@v" not in text
