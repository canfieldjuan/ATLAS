"""Tests for the campaign reasoning context cleanup CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "cleanup_extracted_campaign_reasoning_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cleanup_extracted_campaign_reasoning_contexts",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
cleanup_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup_cli)


class _Repository:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def delete_stale_contexts(self, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return 4


def test_parse_args_requires_explicit_age_threshold() -> None:
    """Operators must choose the cleanup age instead of inheriting a hidden default."""

    with pytest.raises(SystemExit):
        cleanup_cli._parse_args(["--database-url", "postgres://example"])


def test_parse_args_rejects_non_positive_age_threshold() -> None:
    """Invalid age thresholds should fail through argparse, not a traceback."""

    with pytest.raises(SystemExit):
        cleanup_cli._parse_args([
            "--database-url",
            "postgres://example",
            "--older-than-days",
            "0",
        ])


@pytest.mark.asyncio
async def test_cleanup_dry_run_uses_repository_count_path() -> None:
    """Default cleanup mode is dry-run and preserves account/mode filters."""

    repository = _Repository()

    result = await cleanup_cli._cleanup(
        repository,  # type: ignore[arg-type]
        account_id="acct-1",
        target_mode="vendor_retention",
        older_than_days=30,
        apply=False,
    )

    assert result == {
        "status": "dry_run",
        "affected": 4,
        "older_than_days": 30,
        "account_id": "acct-1",
        "target_mode": "vendor_retention",
    }
    assert repository.calls[0]["dry_run"] is True
    assert repository.calls[0]["scope"].account_id == "acct-1"
    assert repository.calls[0]["target_mode"] == "vendor_retention"


@pytest.mark.asyncio
async def test_cleanup_apply_uses_repository_delete_path() -> None:
    """The CLI only deletes when --apply is passed."""

    repository = _Repository()

    result = await cleanup_cli._cleanup(
        repository,  # type: ignore[arg-type]
        account_id=None,
        target_mode=None,
        older_than_days=45,
        apply=True,
    )

    assert result["status"] == "deleted"
    assert result["affected"] == 4
    assert repository.calls[0]["dry_run"] is False
    assert repository.calls[0]["scope"] is None
    assert repository.calls[0]["target_mode"] is None
