"""
Live integration tests for atlas_brain.pipelines.comparisons.

Tests the full normalization pipeline against real DB data.
Run: pytest tests/test_comparisons_live.py -v -s
"""

import asyncio
import json
import logging

import pytest
import pytest_asyncio

from atlas_brain.pipelines.comparisons import (
    ADJACENCY_DIRECTIONS,
    COMPETITOR_NOISE,
    fetch_competitive_flows,
    is_trusted_known_brand,
    load_known_brands,
    normalize_brand,
    normalize_canonical_brand,
    normalize_competitive_flows,
    sanitize_metadata_brand,
)
from atlas_brain.config import settings

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def db_pool():
    import atlas_brain.storage.database as db_module
    from atlas_brain.storage.database import get_db_pool

    db_module._db_pool = None
    pool = get_db_pool()
    await pool.initialize()
    yield pool
    await pool.close()
    db_module._db_pool = None


# ---------------------------------------------------------------------------
# Unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestNormalizeBrand:
    @pytest.fixture(autouse=True)
    def _reset_comparison_settings(self):
        cfg = settings.comparison_normalization
        original_invalid = cfg.invalid_known_brands
        original_terms = cfg.suspicious_singleton_terms
        original_chars = cfg.suspicious_singleton_chars
        original_max_products = cfg.suspicious_singleton_max_products
        yield
        cfg.invalid_known_brands = original_invalid
        cfg.suspicious_singleton_terms = original_terms
        cfg.suspicious_singleton_chars = original_chars
        cfg.suspicious_singleton_max_products = original_max_products

    def test_noise_filtered(self):
        assert normalize_brand("amazon", {}) is None
        assert normalize_brand("charger", {}) is None
        assert normalize_brand("other", {}) is None

    def test_short_unknown_filtered(self):
        assert normalize_brand("ab", {}) is None

    def test_short_known_kept(self):
        assert normalize_brand("LG", {"lg": "LG"}) == "LG"

    def test_brand_collapse(self):
        known = {"cuisinart": "Cuisinart"}
        assert normalize_brand("Cuisinart DCC-3200WP1", known) == "Cuisinart"
        assert normalize_brand("Cuisinart DGB-700BC", known) == "Cuisinart"

    def test_unknown_title_cased(self):
        assert normalize_brand("some random brand", {}) == "Some Random Brand"

    def test_empty_returns_none(self):
        assert normalize_brand("", {}) is None
        assert normalize_brand("  ", {}) is None

    def test_digit_filtered(self):
        assert normalize_brand("123", {}) is None

    def test_invalid_known_brand_filtered(self):
        settings.comparison_normalization.invalid_known_brands = "ipad,magsafe"
        assert normalize_brand("iPad", {}) is None
        assert normalize_brand("MagSafe", {}) is None

    def test_sanitize_metadata_brand_filters_invalid_exact_values(self):
        settings.comparison_normalization.invalid_known_brands = "ipad,magsafe,mx master 2s,blade grinder"
        assert sanitize_metadata_brand("IPAD") == ""
        assert sanitize_metadata_brand("MagSafe") == ""
        assert sanitize_metadata_brand("MX Master 2S") == ""
        assert sanitize_metadata_brand("Blade Grinder") == ""

    def test_sanitize_metadata_brand_keeps_legit_brand(self):
        assert sanitize_metadata_brand("Hamilton Beach") == "Hamilton Beach"

    def test_normalize_canonical_brand_requires_trusted_known_brand(self):
        known = {"hamilton beach": "Hamilton Beach"}
        assert normalize_canonical_brand("Hamilton Beach", known) == "Hamilton Beach"
        assert normalize_canonical_brand("IPAD", known) is None

    def test_trusted_known_brand_rejects_suspicious_singleton(self):
        settings.comparison_normalization.invalid_known_brands = "ipad"
        settings.comparison_normalization.suspicious_singleton_terms = "cloth"
        assert is_trusted_known_brand("IPAD", 1) is False
        assert is_trusted_known_brand("iPadCleaningCloth", 1) is False
        assert is_trusted_known_brand("Case Logic", 20) is True


class TestNormalizeCompetitiveFlows:
    def test_direction_switched_from(self):
        """switched_from: reviewer came FROM other TO reviewed product."""
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "BrandA",
                    "direction": "switched_from",
                    "reviewed_brand": "BrandB",
                    "count": 5,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 1
        assert flows[0]["from_brand"] == "Branda"
        assert flows[0]["to_brand"] == "Brandb"
        assert flows[0]["is_directional"] is True

    def test_direction_switched_to(self):
        """switched_to: reviewer LEFT reviewed product FOR other."""
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "BrandA",
                    "direction": "switched_to",
                    "reviewed_brand": "BrandB",
                    "count": 3,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 1
        assert flows[0]["from_brand"] == "Brandb"
        assert flows[0]["to_brand"] == "Branda"
        assert flows[0]["is_directional"] is True

    def test_direction_compared(self):
        """compared: non-directional, reviewed -> other."""
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "BrandA",
                    "direction": "compared",
                    "reviewed_brand": "BrandB",
                    "count": 2,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 1
        assert flows[0]["from_brand"] == "Brandb"
        assert flows[0]["to_brand"] == "Branda"
        assert flows[0]["is_directional"] is False

    def test_adjacency_filtered(self):
        """used_with etc. should be excluded."""
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "Cable",
                    "direction": "used_with",
                    "reviewed_brand": "BrandB",
                    "count": 10,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 0

    def test_self_reference_filtered(self):
        """Same brand on both sides should be dropped."""
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "Keurig K-Elite",
                    "direction": "compared",
                    "reviewed_brand": "Keurig",
                    "count": 5,
                }
            ],
            known_brands={"keurig": "Keurig"},
        )
        assert len(flows) == 0

    def test_noise_filtered(self):
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "amazon",
                    "direction": "compared",
                    "reviewed_brand": "BrandB",
                    "count": 20,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 0

    def test_brand_collapse_and_merge(self):
        """Two model names from same brand should collapse and sum counts."""
        known = {"cuisinart": "Cuisinart", "keurig": "Keurig"}
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "Cuisinart DCC-3200",
                    "direction": "switched_from",
                    "reviewed_brand": "Keurig",
                    "count": 5,
                },
                {
                    "product_name": "Cuisinart DGB-700",
                    "direction": "switched_from",
                    "reviewed_brand": "Keurig",
                    "count": 3,
                },
            ],
            known_brands=known,
        )
        assert len(flows) == 1
        assert flows[0]["from_brand"] == "Cuisinart"
        assert flows[0]["to_brand"] == "Keurig"
        assert flows[0]["count"] == 8

    def test_sorted_by_count_desc(self):
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "BrandA",
                    "direction": "compared",
                    "reviewed_brand": "BrandC",
                    "count": 1,
                },
                {
                    "product_name": "BrandB",
                    "direction": "compared",
                    "reviewed_brand": "BrandC",
                    "count": 10,
                },
            ],
            known_brands={},
        )
        assert flows[0]["count"] > flows[1]["count"]

    def test_all_required_keys_present(self):
        flows = normalize_competitive_flows(
            [
                {
                    "product_name": "BrandA",
                    "direction": "switched_to",
                    "reviewed_brand": "BrandB",
                    "count": 1,
                }
            ],
            known_brands={},
        )
        assert len(flows) == 1
        f = flows[0]
        assert "from_brand" in f
        assert "to_brand" in f
        assert "direction" in f
        assert "is_directional" in f
        assert "count" in f


# ---------------------------------------------------------------------------
# Live DB integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_known_brands(db_pool):
    """Verify load_known_brands returns a non-empty dict from real DB."""
    brands = await load_known_brands(db_pool)
    assert isinstance(brands, dict)
    assert len(brands) > 0
    # Keys should be lowercase, values original casing
    for k, v in list(brands.items())[:5]:
        assert k == k.lower(), f"Key should be lowercase: {k}"
        assert v.strip() != "", f"Value should be non-empty: {v}"
        logger.info("  Known brand: %s -> %s", k, v)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_load_known_brands_excludes_live_dirty_brand_rows(db_pool):
    """Dirty product_metadata brands should not become trusted canonical brands."""
    raw_rows = await db_pool.fetch(
        "SELECT DISTINCT lower(brand) AS brand_key FROM product_metadata "
        "WHERE brand IS NOT NULL AND brand != '' "
        "AND lower(brand) IN ('ipad', 'ipadcleaningcloth')"
    )
    if not raw_rows:
        pytest.skip("No known dirty canonical brand rows in current DB")

    known = await load_known_brands(db_pool)
    dirty_keys = {row["brand_key"] for row in raw_rows}
    for brand_key in dirty_keys:
        assert brand_key not in known, f"Dirty canonical brand leaked into known_brands: {brand_key}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_competitive_flows_global(db_pool):
    """Fetch competitive flows globally (no category filter) and validate output shape."""
    flows = await fetch_competitive_flows(
        db_pool, min_mentions=2, limit=20
    )
    assert isinstance(flows, list)
    logger.info("Global flows returned: %d", len(flows))

    for f in flows:
        # Required keys
        assert "from_brand" in f, f"Missing from_brand: {f}"
        assert "to_brand" in f, f"Missing to_brand: {f}"
        assert "direction" in f, f"Missing direction: {f}"
        assert "is_directional" in f, f"Missing is_directional: {f}"
        assert "count" in f, f"Missing count: {f}"

        # No empty brands
        assert f["from_brand"].strip(), f"Empty from_brand: {f}"
        assert f["to_brand"].strip(), f"Empty to_brand: {f}"

        # No self-references
        assert f["from_brand"].lower() != f["to_brand"].lower(), (
            f"Self-reference: {f}"
        )

        # No adjacency directions
        assert f["direction"] not in ADJACENCY_DIRECTIONS, (
            f"Adjacency leaked through: {f}"
        )

        # No noise terms
        assert f["from_brand"].lower() not in COMPETITOR_NOISE, (
            f"Noise in from_brand: {f}"
        )
        assert f["to_brand"].lower() not in COMPETITOR_NOISE, (
            f"Noise in to_brand: {f}"
        )

        # Count >= min_mentions
        assert f["count"] >= 2, f"Count below threshold: {f}"

        logger.info(
            "  %s -> %s (%s, %d mentions)",
            f["from_brand"], f["to_brand"], f["direction"], f["count"],
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_competitive_flows_coffee_category(db_pool):
    """Test the Coffee, Tea & Espresso category that originally surfaced the bugs."""
    cat_filter = json.dumps(["Coffee, Tea & Espresso"])
    flows = await fetch_competitive_flows(
        db_pool,
        where_clause="pm.categories @> $1::jsonb",
        params=[cat_filter],
        min_mentions=1,
        limit=30,
    )
    logger.info("Coffee category flows: %d", len(flows))

    if not flows:
        pytest.skip("No Coffee, Tea & Espresso data in DB")

    brands_seen = set()
    for f in flows:
        # Validate structure
        assert f["from_brand"] and f["to_brand"], f"Empty brand: {f}"
        assert f["from_brand"].lower() != f["to_brand"].lower(), (
            f"Self-ref: {f}"
        )
        assert f["direction"] not in ADJACENCY_DIRECTIONS

        # Check for model number leaks (should be brand names, not "Cuisinart DCC-3200WP1")
        for brand_field in ("from_brand", "to_brand"):
            val = f[brand_field]
            # Model numbers typically have digits + dashes
            has_model_pattern = any(
                c.isdigit() for c in val
            ) and "-" in val
            if has_model_pattern:
                logger.warning(
                    "  POSSIBLE MODEL NUMBER LEAK: %s = %s", brand_field, val
                )

        brands_seen.add(f["from_brand"])
        brands_seen.add(f["to_brand"])
        logger.info(
            "  %s -> %s [%s] (%d)",
            f["from_brand"], f["to_brand"], f["direction"], f["count"],
        )

    logger.info("Distinct brands in Coffee flows: %s", brands_seen)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fetch_competitive_flows_source_category(db_pool):
    """Test with source_category filter (amazon_seller pattern)."""
    # Find a category with data
    cat_row = await db_pool.fetchrow(
        "SELECT source_category, COUNT(*) AS c "
        "FROM product_reviews "
        "WHERE deep_enrichment_status = 'enriched' "
        "AND deep_extraction->'product_comparisons' IS NOT NULL "
        "GROUP BY source_category ORDER BY c DESC LIMIT 1"
    )
    if not cat_row:
        pytest.skip("No enriched reviews in DB")

    category = cat_row["source_category"]
    logger.info("Testing with source_category: %s (%d reviews)", category, cat_row["c"])

    flows = await fetch_competitive_flows(
        db_pool,
        where_clause="pr.source_category = $1",
        params=[category],
        min_mentions=1,
        limit=15,
    )

    assert isinstance(flows, list)
    logger.info("Flows for %s: %d", category, len(flows))
    for f in flows:
        assert f["from_brand"] and f["to_brand"]
        assert f["direction"] not in ADJACENCY_DIRECTIONS


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_empty_brands_in_flows(db_pool):
    """The original bug: to_brand was always empty. Verify it never is."""
    flows = await fetch_competitive_flows(
        db_pool, min_mentions=1, limit=100
    )
    for f in flows:
        assert f["from_brand"].strip() != "", f"Empty from_brand! {f}"
        assert f["to_brand"].strip() != "", f"Empty to_brand! {f}"

    logger.info(
        "Verified %d flows, zero empty brands.", len(flows)
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_brand_collapse_in_live_data(db_pool):
    """Verify brand collapse works with real known brands from product_metadata."""
    known = await load_known_brands(db_pool)

    # Get raw comparisons from DB
    rows = await db_pool.fetch(
        """
        SELECT comp->>'product_name' AS pn, pm.brand
        FROM product_reviews pr
        JOIN product_metadata pm ON pm.asin = pr.asin,
             jsonb_array_elements(
                 CASE jsonb_typeof(pr.deep_extraction->'product_comparisons')
                      WHEN 'array' THEN pr.deep_extraction->'product_comparisons'
                      ELSE '[]'::jsonb
                 END
             ) AS comp
        WHERE pr.deep_enrichment_status = 'enriched'
          AND pm.brand IS NOT NULL AND pm.brand != ''
        LIMIT 200
        """
    )

    collapsed_count = 0
    for r in rows:
        pn = r["pn"] or ""
        if not pn.strip():
            continue
        words = pn.split()
        if len(words) > 1:
            first_lower = words[0].lower()
            if first_lower in known:
                normalized = normalize_brand(pn, known)
                if normalized and normalized != pn:
                    collapsed_count += 1

    logger.info(
        "Brand collapse applied to %d / %d multi-word product names",
        collapsed_count, len(rows),
    )
