from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import json
import uuid

import pytest

from atlas_brain import _content_ops_claim_registry as registry
from atlas_brain._content_ops_review_workflow import (
    ContentOpsReviewRequest,
    TenantClaimRegistryReadError,
    run_content_ops_review,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.claims_map import ExtractedClaim
from extracted_content_pipeline.content_pr import (
    CoverageRow,
    CoverageStatus,
    RulePacketVersions,
)
from extracted_content_pipeline.review_contract import ReviewDecision, RiskTier


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "storage"
    / "migrations"
    / "334_content_ops_claim_registry.sql"
)


class _Pool:
    def __init__(self, *, fetchrow_result=None, fetch_rows=None) -> None:
        self.fetchrow_result = fetchrow_result
        self.fetch_rows = list(fetch_rows or [])
        self.fetchrow_calls: list[dict] = []
        self.fetch_calls: list[dict] = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append({"query": str(query), "args": args})
        return self.fetchrow_result

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
        "registry_id": "feature.sso",
        "approved_wording": "SSO is included on every plan",
        "risk_tier": "medium",
        "expires_on": None,
        "metadata": json.dumps({"source": "operator"}),
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        "archived_at": None,
    }
    row.update(overrides)
    return row


def _payload(**overrides):
    values = {
        "registry_id": " feature.sso ",
        "approved_wording": " SSO is included on every plan ",
        "risk_tier": "medium",
        "metadata": {"source": "operator"},
    }
    values.update(overrides)
    return values


def test_content_ops_claim_registry_migration_is_tenant_scoped() -> None:
    sql = MIGRATION.read_text()

    assert "CREATE TABLE IF NOT EXISTS content_ops_claim_registry" in sql
    assert "account_id        UUID NOT NULL REFERENCES saas_accounts(id)" in sql
    assert "approved_wording  TEXT NOT NULL" in sql
    assert "expires_on        DATE" in sql
    assert "archived_at       TIMESTAMPTZ" in sql
    assert "chk_content_ops_claim_registry_risk_tier" in sql
    assert "uq_content_ops_claim_registry_account_registry_id_active" in sql
    assert "ON content_ops_claim_registry (account_id, lower(btrim(registry_id)))" in sql
    assert "WHERE archived_at IS NULL" in sql


@pytest.mark.asyncio
async def test_create_registry_claim_normalizes_and_inserts_claim() -> None:
    account_id = uuid.uuid4()
    expiration = date(2026, 12, 31)
    pool = _Pool(
        fetchrow_result=_row(
            account_id=account_id,
            risk_tier="high",
            expires_on=expiration,
        )
    )

    record = await registry.create_registry_claim(
        pool,
        account_id=account_id,
        payload=_payload(
            registry_id=" Feature.SSO ",
            risk_tier="HIGH",
            expires_on="2026-12-31",
        ),
    )

    call = pool.fetchrow_calls[0]
    assert "INSERT INTO content_ops_claim_registry" in call["query"]
    args = call["args"]
    assert args[0] == account_id
    assert args[1] == "feature.sso"
    assert args[2] == "SSO is included on every plan"
    assert args[3] == "high"
    assert args[4] == expiration
    assert json.loads(args[5]) == {"source": "operator"}
    assert record.risk_tier == RiskTier.HIGH
    assert record.as_registry_claim().expiration == expiration


@pytest.mark.asyncio
async def test_create_registry_claim_treats_non_text_optional_fields_as_missing() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(account_id=account_id, risk_tier=None))

    record = await registry.create_registry_claim(
        pool,
        account_id=account_id,
        payload=_payload(risk_tier={"label": "medium"}, expires_on=123, metadata=[]),
    )

    args = pool.fetchrow_calls[0]["args"]
    assert args[3] is None
    assert args[4] is None
    assert json.loads(args[5]) == {}
    assert record.risk_tier is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(registry_id=None), "registry id is required"),
        (_payload(registry_id=42), "registry id is required"),
        (_payload(approved_wording=" "), "Approved wording is required"),
        (_payload(risk_tier="severe"), "Invalid risk tier"),
        (_payload(expires_on="06/07/2026"), "Invalid expiration date"),
    ],
)
async def test_create_registry_claim_rejects_invalid_payload(
    payload: dict,
    message: str,
) -> None:
    pool = _Pool(fetchrow_result=_row())

    with pytest.raises(ValueError, match=message):
        await registry.create_registry_claim(
            pool,
            account_id=uuid.uuid4(),
            payload=payload,
        )

    assert pool.fetchrow_calls == []


@pytest.mark.asyncio
async def test_update_registry_claim_is_tenant_scoped() -> None:
    account_id = uuid.uuid4()
    claim_id = uuid.uuid4()
    pool = _Pool(fetchrow_result=_row(id=claim_id, account_id=account_id))

    record = await registry.update_registry_claim(
        pool,
        account_id=account_id,
        claim_id=claim_id,
        payload=_payload(approved_wording="SSO is available on Pro plans"),
    )

    assert record is not None
    call = pool.fetchrow_calls[0]
    compact_sql = " ".join(call["query"].split())
    assert "UPDATE content_ops_claim_registry" in compact_sql
    assert "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL" in compact_sql
    assert call["args"][:4] == (
        claim_id,
        account_id,
        "feature.sso",
        "SSO is available on Pro plans",
    )


@pytest.mark.asyncio
async def test_update_registry_claim_returns_none_for_missing_row() -> None:
    pool = _Pool(fetchrow_result=None)

    record = await registry.update_registry_claim(
        pool,
        account_id=uuid.uuid4(),
        claim_id=uuid.uuid4(),
        payload=_payload(),
    )

    assert record is None


@pytest.mark.asyncio
async def test_list_registry_claim_records_returns_display_records() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[_row(account_id=account_id, registry_id=" Feature.SSO ")])

    records = await registry.list_registry_claim_records(pool, account_id=account_id)

    assert records[0].account_id == account_id
    assert records[0].registry_id == "feature.sso"
    assert records[0].metadata == {"source": "operator"}
    assert records[0].as_registry_claim().approved_wording == (
        "SSO is included on every plan"
    )
    call = pool.fetch_calls[0]
    assert "WHERE account_id = $1" in call["query"]
    assert "archived_at IS NULL" in call["query"]
    assert call["args"] == (account_id,)


@pytest.mark.asyncio
async def test_expire_registry_claim_is_tenant_scoped_and_remains_readable() -> None:
    account_id = uuid.uuid4()
    claim_id = uuid.uuid4()
    expiration = date(2026, 6, 7)
    pool = _Pool(
        fetchrow_result=_row(
            id=claim_id,
            account_id=account_id,
            expires_on=expiration,
        )
    )

    record = await registry.expire_registry_claim(
        pool,
        account_id=account_id,
        claim_id=claim_id,
        expires_on=expiration,
    )

    assert record is not None
    assert record.expires_on == expiration
    call = pool.fetchrow_calls[0]
    compact_sql = " ".join(call["query"].split())
    assert "SET expires_on = $3" in compact_sql
    assert "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL" in compact_sql
    assert call["args"] == (claim_id, account_id, expiration)


@pytest.mark.asyncio
async def test_archive_registry_claim_is_tenant_scoped() -> None:
    account_id = uuid.uuid4()
    claim_id = uuid.uuid4()
    pool = _Pool(fetchrow_result={"id": claim_id})

    archived = await registry.archive_registry_claim(
        pool,
        account_id=account_id,
        claim_id=claim_id,
    )

    assert archived is True
    call = pool.fetchrow_calls[0]
    compact_sql = " ".join(call["query"].split())
    assert "WHERE id = $1 AND account_id = $2 AND archived_at IS NULL" in compact_sql
    assert call["args"] == (claim_id, account_id)


@pytest.mark.asyncio
async def test_archive_registry_claim_returns_false_for_missing_row() -> None:
    archived = await registry.archive_registry_claim(
        _Pool(fetchrow_result=None),
        account_id=uuid.uuid4(),
        claim_id=uuid.uuid4(),
    )

    assert archived is False


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [TenantScope(), TenantScope(account_id="not-a-uuid")])
async def test_repository_reader_fails_closed_on_invalid_tenant_scope(
    scope: TenantScope,
) -> None:
    pool = _Pool(fetch_rows=[_row()])
    repository = registry.ContentOpsClaimRegistryRepository(pool)

    with pytest.raises(TenantClaimRegistryReadError, match="valid tenant scope required"):
        await repository.list_registry_claims(scope=scope)

    assert pool.fetch_calls == []


@pytest.mark.asyncio
async def test_repository_reader_wraps_database_read_failure() -> None:
    repository = registry.ContentOpsClaimRegistryRepository(_FailingFetchPool())

    with pytest.raises(TenantClaimRegistryReadError, match="claim registry read failed"):
        await repository.list_registry_claims(
            scope=TenantScope(account_id=str(uuid.uuid4()))
        )


@pytest.mark.asyncio
async def test_repository_reader_returns_registry_claim_mapping() -> None:
    account_id = uuid.uuid4()
    expiration = date(2026, 12, 31)
    pool = _Pool(
        fetch_rows=[
            _row(
                account_id=account_id,
                registry_id="Pricing.Discount",
                approved_wording="Save up to 30% on eligible annual plans",
                risk_tier="high",
                expires_on=expiration,
            )
        ]
    )
    repository = registry.ContentOpsClaimRegistryRepository(pool)

    claims = await repository.list_registry_claims(
        scope=TenantScope(account_id=str(account_id))
    )

    assert tuple(claims) == ("pricing.discount",)
    claim = claims["pricing.discount"]
    assert claim.approved_wording == "Save up to 30% on eligible annual plans"
    assert claim.risk_tier == RiskTier.HIGH
    assert claim.expiration == expiration
    assert pool.fetch_calls[0]["args"] == (account_id,)


@pytest.mark.asyncio
async def test_review_service_consumes_repository_reader() -> None:
    account_id = uuid.uuid4()
    pool = _Pool(fetch_rows=[_row(account_id=account_id, registry_id="Feature.SSO")])
    repository = registry.ContentOpsClaimRegistryRepository(pool)

    result = await run_content_ops_review(
        ContentOpsReviewRequest(
            asset_id="asset-1",
            rule_packet=RulePacketVersions(
                brief="brief-v1",
                brand_voice="voice-v1",
                claim_registry="claims-db-v1",
                compliance="compliance-v1",
                channel_schema="channel-v1",
            ),
            coverage=(
                CoverageRow(
                    rule_id="CLAIM-01",
                    requirement="Claim is registered",
                    status=CoverageStatus.PASS,
                    evidence="registry row",
                ),
            ),
            extracted_claims=(
                ExtractedClaim(
                    text="SSO is included on every plan",
                    location="hero",
                    registry_id="feature.sso",
                ),
            ),
            as_of=date(2026, 6, 7),
        ),
        scope=TenantScope(account_id=str(account_id)),
        registry_reader=repository,
    )

    assert result.decision == ReviewDecision.APPROVED
    assert result.mapped_claims[0].approved_wording == "SSO is included on every plan"
    assert pool.fetch_calls[0]["args"] == (account_id,)


@pytest.mark.asyncio
async def test_review_service_blocks_invalid_repository_scope() -> None:
    pool = _Pool(fetch_rows=[_row()])
    repository = registry.ContentOpsClaimRegistryRepository(pool)

    result = await run_content_ops_review(
        ContentOpsReviewRequest(
            asset_id="asset-1",
            rule_packet=RulePacketVersions(
                brief="brief-v1",
                brand_voice="voice-v1",
                claim_registry="claims-db-v1",
                compliance="compliance-v1",
                channel_schema="channel-v1",
            ),
            coverage=(
                CoverageRow(
                    rule_id="CLAIM-01",
                    requirement="Claim is registered",
                    status=CoverageStatus.PASS,
                    evidence="registry row",
                ),
            ),
            extracted_claims=(
                ExtractedClaim(
                    text="SSO is included on every plan",
                    location="hero",
                    registry_id="feature.sso",
                ),
            ),
            as_of=date(2026, 6, 7),
        ),
        scope=TenantScope(account_id="not-a-uuid"),
        registry_reader=repository,
    )

    assert result.decision == ReviewDecision.BLOCKED
    assert result.reasons == ("valid tenant scope required",)
    assert result.mapped_claims == ()
    assert pool.fetch_calls == []
