from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_postgres_seller_category_intelligence import (
    CategoryIntelligenceLimits,
    aggregate_seller_category_intelligence,
    refresh_seller_category_intelligence,
    save_seller_category_intelligence_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_extracted_seller_category_intelligence",
        ROOT / "scripts/refresh_extracted_seller_category_intelligence.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self):
        self.fetchrow_results = []
        self.fetch_results = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((str(query), args))
        return self.fetchrow_results.pop(0) if self.fetchrow_results else None

    async def fetch(self, query, *args):
        self.fetch_calls.append((str(query), args))
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def execute(self, query, *args):
        self.execute_calls.append((str(query), args))
        return "INSERT 0 1"

    async def close(self):
        self.closed = True


def _known_brand_rows() -> list[dict[str, str]]:
    return [{"brand": "Brand A"}, {"brand": "Brand B"}, {"brand": "Brand C"}]


def _seed_aggregate_results(
    pool: _Pool,
    *,
    total_reviews: int = 120,
    include_known_brands: bool = True,
) -> None:
    pool.fetchrow_results.append({
        "total_reviews": total_reviews,
        "total_products": 30,
        "total_brands": 8,
    })
    results = []
    if include_known_brands:
        results.append(_known_brand_rows())
    results.extend([
        [
            {
                "brand": "Brand A",
                "total_reviews": 20,
                "repurchase_yes": 8,
                "repurchase_no": 2,
                "safety_count": 0,
                "deep_count": 8,
            }
        ],
        [{"complaint": "leaky bottle", "count": 7, "severity": None, "affected_brands": 3}],
        [{"request": "third-party testing", "count": 5, "brand_count": 2, "avg_rating": 2.8}],
        [
            {
                "reviewed_brand": "Brand A",
                "compared_product": "Brand B Pro Bottle",
                "direction": "switched_to",
                "count": 1,
            },
            {
                "reviewed_brand": "Brand A",
                "compared_product": "Brand B Travel Kit",
                "direction": "switched_to",
                "count": 2,
            },
            {
                "reviewed_brand": "Brand A",
                "compared_product": "Brand C Charger",
                "direction": "used_with",
                "count": 10,
            },
            {
                "reviewed_brand": "Brand A",
                "compared_product": "unmodeled competitor",
                "direction": "switched_to",
                "count": 2,
            },
        ],
        [{"brand": "Brand A", "category": "labeling", "description": "missing", "flagged_count": 2}],
        [{"suggestion": "better seal", "count": 3, "affected_asins": ["a1", "a2"]}],
        [{"cause": "packaging", "count": 9}],
    ])
    pool.fetch_results.extend(results)


@pytest.mark.asyncio
async def test_aggregate_seller_category_intelligence_builds_snapshot() -> None:
    pool = _Pool()
    _seed_aggregate_results(pool)

    snapshot = await aggregate_seller_category_intelligence(
        pool,
        "supplements",
        min_reviews=50,
    )

    assert snapshot is not None
    assert snapshot["category"] == "supplements"
    assert snapshot["category_stats"] == {
        "total_reviews": 120,
        "total_brands": 8,
        "total_products": 30,
        "date_range": "all available data",
    }
    assert snapshot["top_pain_points"][0]["severity"] == "medium"
    assert snapshot["feature_gaps"][0]["avg_rating"] == 2.8
    assert snapshot["competitive_flows"] == [
        {
            "from_brand": "Brand A",
            "to_brand": "Brand B",
            "direction": "switched_to",
            "count": 3,
        },
        {
            "from_brand": "Brand A",
            "to_brand": "Unmodeled Competitor",
            "direction": "switched_to",
            "count": 2,
        },
    ]
    assert snapshot["brand_health"][0]["trend"] == "rising"

    brand_query, brand_args = pool.fetch_calls[0]
    assert "SELECT brand" in brand_query
    assert brand_args == ()
    flow_query, flow_args = pool.fetch_calls[4]
    assert "comp ->> 'product_name'" in flow_query
    assert "comp ->> 'direction'" in flow_query
    assert "JOIN \"product_metadata\" pm ON pm.asin = pr.asin" in flow_query
    assert "from_product" not in flow_query
    assert "to_product" not in flow_query
    assert flow_args == ("supplements",)


@pytest.mark.asyncio
async def test_aggregate_seller_category_intelligence_skips_low_volume() -> None:
    pool = _Pool()
    _seed_aggregate_results(pool, total_reviews=10)

    snapshot = await aggregate_seller_category_intelligence(
        pool,
        "supplements",
        min_reviews=50,
    )

    assert snapshot is None
    assert pool.fetch_calls == []


@pytest.mark.asyncio
async def test_aggregate_seller_category_intelligence_accepts_custom_limits() -> None:
    pool = _Pool()
    _seed_aggregate_results(pool)

    await aggregate_seller_category_intelligence(
        pool,
        "supplements",
        min_reviews=50,
        intelligence_limits=CategoryIntelligenceLimits(
            min_brand_reviews=3,
            brand_row_limit=7,
            default_severity="unknown",
        ),
    )

    brand_query, brand_args = pool.fetch_calls[1]
    assert "HAVING COUNT(*) >= $2" in brand_query
    assert brand_args == ("supplements", 3, 7)


@pytest.mark.asyncio
async def test_save_seller_category_intelligence_snapshot_upserts_json() -> None:
    pool = _Pool()
    snapshot = {
        "category": "supplements",
        "category_stats": {
            "total_reviews": 120,
            "total_brands": 8,
            "total_products": 30,
        },
        "top_pain_points": [{"complaint": "leaky bottle"}],
        "feature_gaps": [{"request": "third-party testing"}],
        "competitive_flows": [{"from_brand": "A", "to_brand": "B"}],
        "brand_health": [{"brand": "A", "health_score": 80}],
        "safety_signals": [{"brand": "A", "flagged_count": 2}],
        "manufacturing_insights": [{"suggestion": "better seal"}],
        "top_root_causes": [{"cause": "packaging"}],
    }

    await save_seller_category_intelligence_snapshot(pool, snapshot)

    query, args = pool.execute_calls[0]
    assert "INSERT INTO \"category_intelligence_snapshots\"" in query
    assert "ON CONFLICT (category, COALESCE(subcategory, ''), snapshot_date)" in query
    assert args[:4] == ("supplements", 120, 8, 30)
    assert json.loads(args[4]) == [{"complaint": "leaky bottle"}]
    assert json.loads(args[10]) == [{"cause": "packaging"}]


@pytest.mark.asyncio
async def test_refresh_discovers_categories_and_saves_snapshots() -> None:
    pool = _Pool()
    pool.fetch_results = [
        [{"source_category": "supplements", "review_count": 120}],
        _known_brand_rows(),
    ]
    _seed_aggregate_results(pool, include_known_brands=False)

    result = await refresh_seller_category_intelligence(pool, min_reviews=50, limit=5)

    discover_query, discover_args = pool.fetch_calls[0]
    assert "FROM \"product_reviews\"" in discover_query
    assert discover_args == (50, 5)
    assert result.as_dict() == {
        "refreshed": 1,
        "skipped": 0,
        "categories": ["supplements"],
    }
    assert pool.execute_calls


@pytest.mark.asyncio
async def test_refresh_uses_explicit_categories_without_discovery() -> None:
    pool = _Pool()
    pool.fetch_results = [_known_brand_rows()]
    _seed_aggregate_results(pool, include_known_brands=False)

    result = await refresh_seller_category_intelligence(
        pool,
        categories=("supplements",),
        min_reviews=50,
    )

    assert pool.fetchrow_calls[0][1] == ("supplements",)
    assert "SELECT brand" in pool.fetch_calls[0][0]
    assert "GROUP BY source_category" not in pool.fetch_calls[0][0]
    assert result.refreshed == 1


@pytest.mark.asyncio
async def test_refresh_does_not_limit_explicit_categories() -> None:
    pool = _Pool()
    pool.fetch_results = [_known_brand_rows()]
    _seed_aggregate_results(pool, include_known_brands=False)
    _seed_aggregate_results(pool, include_known_brands=False)
    _seed_aggregate_results(pool, include_known_brands=False)

    result = await refresh_seller_category_intelligence(
        pool,
        categories=("supplements", "skincare", "coffee"),
        min_reviews=50,
        limit=1,
    )

    assert [args for _query, args in pool.fetchrow_calls] == [
        ("supplements",),
        ("skincare",),
        ("coffee",),
    ]
    assert result.as_dict() == {
        "refreshed": 3,
        "skipped": 0,
        "categories": ["supplements", "skincare", "coffee"],
    }
    known_brand_queries = [
        query
        for query, _args in pool.fetch_calls
        if "SELECT brand" in query and "GROUP BY brand" in query
    ]
    assert len(known_brand_queries) == 1


@pytest.mark.asyncio
async def test_refresh_deduplicates_explicit_categories() -> None:
    pool = _Pool()
    pool.fetch_results = [_known_brand_rows()]
    _seed_aggregate_results(pool, include_known_brands=False)
    _seed_aggregate_results(pool, include_known_brands=False)

    result = await refresh_seller_category_intelligence(
        pool,
        categories=("supplements", "supplements", "skincare", "skincare"),
        min_reviews=50,
    )

    assert [args for _query, args in pool.fetchrow_calls] == [
        ("supplements",),
        ("skincare",),
    ]
    assert result.as_dict() == {
        "refreshed": 2,
        "skipped": 0,
        "categories": ["supplements", "skincare"],
    }


@pytest.mark.asyncio
async def test_refresh_rejects_bad_table_identifier() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        await refresh_seller_category_intelligence(
            _Pool(),
            categories=("supplements",),
            reviews_table="product_reviews;drop",
        )


def test_refresh_cli_parses_categories_and_tables() -> None:
    cli = _load_cli_module()

    args = cli._parse_args([
        "--database-url",
        "postgres://example",
        "--category",
        "supplements",
        "--category",
        "skincare",
        "--min-reviews",
        "25",
        "--reviews-table",
        "reviews",
    ])

    assert args.database_url == "postgres://example"
    assert args.category == ["supplements", "skincare"]
    assert args.min_reviews == 25
    assert args.reviews_table == "reviews"


@pytest.mark.asyncio
async def test_refresh_cli_wires_pool_and_result(monkeypatch, capsys) -> None:
    cli = _load_cli_module()
    pool = _Pool()
    pool.fetch_results = [_known_brand_rows()]
    _seed_aggregate_results(pool, include_known_brands=False)
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
            "--category",
            "supplements",
        ],
    )

    exit_code = await cli._main()

    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    output = json.loads(capsys.readouterr().out)
    assert output["refreshed"] == 1
    assert output["categories"] == ["supplements"]
