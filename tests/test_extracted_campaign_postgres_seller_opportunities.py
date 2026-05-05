from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_postgres_seller_opportunities import (
    prepare_seller_campaign_opportunities,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_extracted_seller_campaign_opportunities",
        ROOT / "scripts/prepare_extracted_seller_campaign_opportunities.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self, rows=None, execute_results=None):
        self.rows = list(rows or [])
        self.execute_results = list(execute_results or [])
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.rows

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        if self.execute_results:
            return self.execute_results.pop(0)
        return "INSERT 0 1"

    async def close(self):
        self.closed = True


def _seller_intel_row(**overrides):
    row = {
        "seller_target_id": "00000000-0000-0000-0000-000000000001",
        "seller_name": "Acme Seller",
        "company_name": "Acme Products",
        "email": "owner@example.com",
        "seller_type": "private_label",
        "categories": ["supplements"],
        "storefront_url": "https://example.com/store",
        "notes": "Top seller",
        "category": "supplements",
        "snapshot_date": "2026-05-04",
        "total_reviews": 250,
        "total_brands": 12,
        "total_products": 44,
        "top_pain_points": [
            {"complaint": "capsules leak", "count": 8},
            {"complaint": "bad aftertaste", "count": 5},
        ],
        "feature_gaps": [{"request": "third-party testing", "count": 4}],
        "competitive_flows": [{"from_brand": "Brand A", "to_brand": "Brand B"}],
        "brand_health": [{"brand": "Brand A", "health_score": 62}],
        "safety_signals": [{"category": "labeling", "flagged_count": 2}],
        "manufacturing_insights": [{"suggestion": "seal bottle lids", "count": 3}],
        "top_root_causes": [{"cause": "packaging", "count": 6}],
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_reads_targets_and_snapshots() -> None:
    pool = _Pool(rows=[_seller_intel_row()])

    result = await prepare_seller_campaign_opportunities(
        pool,
        account_id="acct-1",
        category="supplements",
        limit=7,
    )

    query, args = pool.fetch_calls[0]
    assert "FROM \"seller_targets\" st" in query
    assert "FROM \"category_intelligence_snapshots\" cis" in query
    assert "cis.subcategory IS NULL" in query
    assert "st.status = $1" in query
    assert "$2 = ANY(st.categories)" in query
    assert args == ("active", "supplements", 7)
    assert result.as_dict()["prepared"] == 1
    assert result.target_ids == (
        "seller:00000000-0000-0000-0000-000000000001:supplements",
    )


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_inserts_normalized_campaign_row() -> None:
    pool = _Pool(rows=[_seller_intel_row()])

    await prepare_seller_campaign_opportunities(pool, account_id="acct-1")

    query, args = pool.execute_calls[0]
    assert "INSERT INTO \"campaign_opportunities\"" in query
    assert args[:10] == (
        "acct-1",
        "seller:00000000-0000-0000-0000-000000000001:supplements",
        "amazon_seller",
        "Acme Seller",
        None,
        "Acme Seller",
        "owner@example.com",
        "private_label",
        250,
        19,
    )
    assert json.loads(args[10]) == [
        "capsules leak",
        "bad aftertaste",
        "packaging",
    ]
    assert json.loads(args[11]) == ["Brand A", "Brand B"]
    raw_payload = json.loads(args[13])
    assert raw_payload["seller_target"]["seller_type"] == "private_label"
    assert raw_payload["category_intelligence"]["category_stats"]["total_reviews"] == 250


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_replace_existing_scopes_delete() -> None:
    pool = _Pool(rows=[_seller_intel_row()], execute_results=["DELETE 2", "INSERT 0 1"])

    result = await prepare_seller_campaign_opportunities(
        pool,
        account_id="acct-1",
        replace_existing=True,
    )

    delete_query, delete_args = pool.execute_calls[0]
    assert "DELETE FROM \"campaign_opportunities\"" in delete_query
    assert "target_mode = $1" in delete_query
    assert "target_id = ANY($2::text[])" in delete_query
    assert "account_id = $3" in delete_query
    assert delete_args == (
        "amazon_seller",
        ["seller:00000000-0000-0000-0000-000000000001:supplements"],
        "acct-1",
    )
    assert result.replaced == 2


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_skips_rows_without_target_key() -> None:
    pool = _Pool(rows=[_seller_intel_row(seller_target_id="")])

    result = await prepare_seller_campaign_opportunities(pool)

    assert result.prepared == 0
    assert result.skipped == 1
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_rejects_bad_table_identifier() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await prepare_seller_campaign_opportunities(
            _Pool(),
            seller_targets_table="seller_targets;drop",
        )


def test_prepare_seller_opportunities_cli_parses_host_options() -> None:
    cli = _load_cli_module()

    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--account-id",
        "acct-1",
        "--category",
        "supplements",
        "--replace-existing",
    ])

    assert args.database_url == "postgres://example"
    assert args.account_id == "acct-1"
    assert args.category == "supplements"
    assert args.replace_existing is True


@pytest.mark.asyncio
async def test_prepare_seller_opportunities_cli_wires_pool_and_result(
    monkeypatch,
    capsys,
) -> None:
    cli = _load_cli_module()
    pool = _Pool(rows=[_seller_intel_row()])
    created_urls: list[str] = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "run",
            "--database-url",
            "postgres://example",
            "--account-id",
            "acct-1",
            "--limit",
            "1",
        ],
    )

    exit_code = await cli._main()

    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    output = json.loads(capsys.readouterr().out)
    assert output["prepared"] == 1
    assert output["target_mode"] == "amazon_seller"
