from __future__ import annotations

from datetime import date

import pytest

from extracted_competitive_intelligence.services.b2b.challenger_dashboard_claims import (
    ChallengerDashboardClaimsPortNotConfigured,
    DirectDisplacementClaimRow,
    aggregate_direct_displacement_claim,
    aggregate_direct_displacement_claims_for_challenger,
    aggregate_direct_displacement_claims_for_incumbent,
    configure_direct_displacement_claim_reader,
    configure_direct_displacement_claims_for_challenger_reader,
    configure_direct_displacement_claims_for_incumbent_reader,
)


SAMPLE_AS_OF_DATE = date(2026, 5, 4)
SHORT_ANALYSIS_WINDOW_DAYS = 90
MEDIUM_ANALYSIS_WINDOW_DAYS = 180
DEFAULT_ANALYSIS_WINDOW_DAYS = 365
CHALLENGER_ROW_LIMIT = 7
INCUMBENT_ROW_LIMIT = 9


@pytest.fixture(autouse=True)
def _reset_challenger_claim_readers():
    configure_direct_displacement_claim_reader(None)
    configure_direct_displacement_claims_for_challenger_reader(None)
    configure_direct_displacement_claims_for_incumbent_reader(None)
    yield
    configure_direct_displacement_claim_reader(None)
    configure_direct_displacement_claims_for_challenger_reader(None)
    configure_direct_displacement_claims_for_incumbent_reader(None)


@pytest.mark.asyncio
async def test_single_pair_claim_fails_closed_without_host_reader():
    with pytest.raises(ChallengerDashboardClaimsPortNotConfigured):
        await aggregate_direct_displacement_claim(
            object(),
            challenger="NewCo",
            incumbent="OldCo",
            as_of_date=SAMPLE_AS_OF_DATE,
            analysis_window_days=DEFAULT_ANALYSIS_WINDOW_DAYS,
        )


@pytest.mark.asyncio
async def test_incumbent_claim_rows_fail_closed_without_host_reader():
    with pytest.raises(ChallengerDashboardClaimsPortNotConfigured):
        await aggregate_direct_displacement_claims_for_incumbent(
            object(),
            incumbent="OldCo",
            as_of_date=SAMPLE_AS_OF_DATE,
            analysis_window_days=DEFAULT_ANALYSIS_WINDOW_DAYS,
        )


@pytest.mark.asyncio
async def test_claim_readers_delegate_to_configured_host_adapters():
    pool = object()
    as_of_date = SAMPLE_AS_OF_DATE
    seen: dict[str, object] = {}
    single_claim = {"claim_id": "claim-1"}
    challenger_rows = [
        DirectDisplacementClaimRow(
            challenger="NewCo",
            incumbent="OldCo",
            claim={"claim_id": "claim-2"},
        )
    ]
    incumbent_rows = [
        DirectDisplacementClaimRow(
            challenger="NextCo",
            incumbent="OldCo",
            claim={"claim_id": "claim-3"},
        )
    ]

    async def single_reader(
        pool_arg,
        *,
        challenger,
        incumbent,
        as_of_date,
        analysis_window_days,
    ):
        seen["single"] = (pool_arg, challenger, incumbent, as_of_date, analysis_window_days)
        return single_claim

    async def challenger_reader(
        pool_arg,
        *,
        challenger,
        as_of_date,
        analysis_window_days,
        limit,
    ):
        seen["challenger"] = (pool_arg, challenger, as_of_date, analysis_window_days, limit)
        return challenger_rows

    async def incumbent_reader(
        pool_arg,
        *,
        incumbent,
        as_of_date,
        analysis_window_days,
        limit,
    ):
        seen["incumbent"] = (pool_arg, incumbent, as_of_date, analysis_window_days, limit)
        return incumbent_rows

    configure_direct_displacement_claim_reader(single_reader)
    configure_direct_displacement_claims_for_challenger_reader(challenger_reader)
    configure_direct_displacement_claims_for_incumbent_reader(incumbent_reader)

    assert await aggregate_direct_displacement_claim(
        pool,
        challenger="NewCo",
        incumbent="OldCo",
        as_of_date=as_of_date,
        analysis_window_days=MEDIUM_ANALYSIS_WINDOW_DAYS,
    ) is single_claim
    assert await aggregate_direct_displacement_claims_for_challenger(
        pool,
        challenger="NewCo",
        as_of_date=as_of_date,
        analysis_window_days=SHORT_ANALYSIS_WINDOW_DAYS,
        limit=CHALLENGER_ROW_LIMIT,
    ) == challenger_rows
    assert await aggregate_direct_displacement_claims_for_incumbent(
        pool,
        incumbent="OldCo",
        as_of_date=as_of_date,
        analysis_window_days=DEFAULT_ANALYSIS_WINDOW_DAYS,
        limit=INCUMBENT_ROW_LIMIT,
    ) == incumbent_rows

    assert seen["single"] == (
        pool,
        "NewCo",
        "OldCo",
        as_of_date,
        MEDIUM_ANALYSIS_WINDOW_DAYS,
    )
    assert seen["challenger"] == (
        pool,
        "NewCo",
        as_of_date,
        SHORT_ANALYSIS_WINDOW_DAYS,
        CHALLENGER_ROW_LIMIT,
    )
    assert seen["incumbent"] == (
        pool,
        "OldCo",
        as_of_date,
        DEFAULT_ANALYSIS_WINDOW_DAYS,
        INCUMBENT_ROW_LIMIT,
    )
