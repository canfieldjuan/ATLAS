from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest

from atlas_brain import _content_ops_calibration_library as calib
from atlas_brain._content_ops_calibration_library import (
    ContentOpsCalibrationLibraryReadError,
    ContentOpsCalibrationLibraryRepository,
)
from atlas_brain._content_ops_review_workflow import (
    ContentOpsReviewRequest,
    run_content_ops_review,
)
from extracted_content_pipeline.adversarial_pass import (
    AdversarialFinding,
    AdversarialFindingCategory,
    AdversarialPass,
)
from extracted_content_pipeline.calibration_library import CalibrationLabel
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.content_pr import RulePacketVersions


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "335_content_ops_calibration_library.sql"
)


class _Pool:
    def __init__(self, *, fetch_rows=None) -> None:
        self.fetch_rows = list(fetch_rows or [])
        self.fetch_calls: list[dict] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        return self.fetch_rows


class _FailingFetchPool(_Pool):
    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        raise RuntimeError("database unavailable")


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "account_id": uuid.uuid4(),
        "example_id": "overclaim-001",
        "label": "overclaim",
        "excerpt": "guaranteed 99.99% uptime",
        "reasoning": "No SLA backs this number.",
        "source": "curated",
        "metadata": "{}",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "archived_at": None,
    }
    row.update(overrides)
    return row


# -- migration ---------------------------------------------------------------


def test_migration_is_tenant_scoped_and_teachable_only() -> None:
    sql = MIGRATION.read_text()
    assert "CREATE TABLE IF NOT EXISTS content_ops_calibration_library" in sql
    assert "account_id        UUID NOT NULL REFERENCES saas_accounts(id)" in sql
    assert "excerpt           TEXT NOT NULL" in sql
    assert "reasoning         TEXT NOT NULL" in sql
    assert "chk_content_ops_calibration_library_excerpt" in sql
    assert "chk_content_ops_calibration_library_reasoning" in sql
    assert "uq_content_ops_calibration_library_account_example_id_active" in sql
    assert "ON content_ops_calibration_library (account_id, lower(btrim(example_id)))" in sql
    assert "WHERE archived_at IS NULL" in sql


def test_migration_label_check_matches_the_enum() -> None:
    # Drift guard: the DB label CHECK must list exactly the CalibrationLabel
    # values, so adding a label without updating the migration fails here.
    sql = MIGRATION.read_text()
    for label in CalibrationLabel:
        assert f"'{label.value}'" in sql
    # No stray label values in the CHECK beyond the enum.
    import re

    check = re.search(r"label IN \(([^)]*)\)", sql, re.DOTALL)
    assert check is not None
    listed = {token.strip().strip("'") for token in check.group(1).split(",") if token.strip()}
    assert listed == {label.value for label in CalibrationLabel}


# -- repository reader -------------------------------------------------------


@pytest.mark.asyncio
async def test_reader_returns_calibration_examples_for_tenant() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[
        _row(account_id=account_id, example_id="overclaim-001", label="overclaim"),
        _row(account_id=account_id, example_id="voice-001", label="voice_drift",
             excerpt="synergize", reasoning="off brand"),
    ])
    repo = ContentOpsCalibrationLibraryRepository(pool)

    examples = await repo.list_calibration_examples(scope=TenantScope(account_id=str(account_id)))

    assert tuple(e.example_id for e in examples) == ("overclaim-001", "voice-001")
    assert examples[0].label == CalibrationLabel.OVERCLAIM
    assert examples[1].label == CalibrationLabel.VOICE_DRIFT
    call = pool.fetch_calls[0]
    assert "WHERE account_id = $1" in call["query"]
    assert "archived_at IS NULL" in call["query"]
    assert call["args"] == (account_id,)


@pytest.mark.asyncio
async def test_reader_dedupes_by_example_id_and_skips_unknown_label() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[
        _row(account_id=account_id, example_id="dup", label="overclaim", excerpt="first"),
        _row(account_id=account_id, example_id="dup", label="overclaim", excerpt="second"),
        _row(account_id=account_id, example_id="bad", label="not_a_label"),
    ])
    repo = ContentOpsCalibrationLibraryRepository(pool)

    examples = await repo.list_calibration_examples(scope=TenantScope(account_id=str(account_id)))

    assert tuple(e.example_id for e in examples) == ("dup",)
    assert examples[0].excerpt == "first"  # first wins on dedup


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [TenantScope(), TenantScope(account_id="not-a-uuid")])
async def test_reader_returns_empty_on_invalid_scope_without_reading(scope: TenantScope) -> None:
    # Evidence, not a gate: an unusable scope yields no anchors, not an error.
    pool = _Pool(fetch_rows=[_row()])
    repo = ContentOpsCalibrationLibraryRepository(pool)
    assert await repo.list_calibration_examples(scope=scope) == ()
    assert pool.fetch_calls == []


@pytest.mark.asyncio
async def test_reader_wraps_database_read_failure() -> None:
    repo = ContentOpsCalibrationLibraryRepository(_FailingFetchPool())
    with pytest.raises(ContentOpsCalibrationLibraryReadError, match="calibration library read failed"):
        await repo.list_calibration_examples(scope=TenantScope(account_id=str(uuid.uuid4())))


# -- review integration: server anchors surface, failure degrades ------------


_OVERCLAIM_PASS = (
    AdversarialPass(pass_id="p1", findings=(
        AdversarialFinding(category=AdversarialFindingCategory.OVERCLAIM, message="40% unbacked", evidence="cuts 40%"),
    )),
)
_PINNED = RulePacketVersions(
    brief="b", brand_voice="v", claim_registry="c", compliance="x", channel_schema="s",
)


class _OkRegistry:
    async def list_registry_claims(self, *, scope):
        return {}


@pytest.mark.asyncio
async def test_review_surfaces_repository_anchors() -> None:
    account_id = uuid.uuid4()
    repo = ContentOpsCalibrationLibraryRepository(_Pool(fetch_rows=[
        _row(account_id=account_id, example_id="overclaim-001", label="overclaim"),
    ]))

    result = await run_content_ops_review(
        ContentOpsReviewRequest(rule_packet=_PINNED, coverage=(), adversarial_passes=_OVERCLAIM_PASS),
        scope=TenantScope(account_id=str(account_id)),
        registry_reader=_OkRegistry(),
        calibration_reader=repo,
    )
    assert tuple(a.example_id for a in result.calibration_anchors) == ("overclaim-001",)


@pytest.mark.asyncio
async def test_review_degrades_when_repository_read_fails() -> None:
    from extracted_content_pipeline.calibration_library import CalibrationExample

    repo = ContentOpsCalibrationLibraryRepository(_FailingFetchPool())
    request_anchor = (CalibrationExample(example_id="oc-req", excerpt="req", label=CalibrationLabel.OVERCLAIM, reasoning="r"),)

    result = await run_content_ops_review(
        ContentOpsReviewRequest(
            rule_packet=_PINNED, coverage=(),
            adversarial_passes=_OVERCLAIM_PASS, calibration_examples=request_anchor,
        ),
        scope=TenantScope(account_id=str(uuid.uuid4())),
        registry_reader=_OkRegistry(),
        calibration_reader=repo,
    )
    # The verdict is unaffected and the request-supplied anchor still surfaces.
    assert tuple(a.example_id for a in result.calibration_anchors) == ("oc-req",)
