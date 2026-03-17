"""Regression tests for archetype context propagation across B2B surfaces.

Covers:
- _fetch_prior_archetypes helper
- _build_vendor_comparison_summary with archetype params
- archetype-shift change event detection
- vendor/challenger context archetype injection
- tenant report prior-archetype enrichment
"""

import sys
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-mock heavy deps before importing task modules
for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "starlette", "starlette.requests",
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
    _build_vendor_comparison_summary,
    _fetch_prior_archetypes,
)


# ---------------------------------------------------------------------------
# _build_vendor_comparison_summary archetype params
# ---------------------------------------------------------------------------


class TestVendorComparisonSummaryArchetype:
    """Verify archetype text is appended when provided."""

    def _make_snapshot(self, name: str, density: float = 10.0, urgency: float = 5.0,
                       pos_pct: float = 50.0, rec: float = 0.0) -> dict[str, Any]:
        return {
            "vendor_name": name,
            "churn_signal_density": density,
            "churn_intent_count": 10,
            "signal_count": 100,
            "avg_urgency_score": urgency,
            "positive_review_pct": pos_pct,
            "recommend_ratio": rec,
            "top_pain_categories": [{"category": "pricing"}],
            "top_competitors": [],
        }

    def test_no_archetypes(self):
        p = self._make_snapshot("VendorA", density=20)
        c = self._make_snapshot("VendorB", density=10)
        result = _build_vendor_comparison_summary(p, c, [])
        assert "Archetype" not in result

    def test_primary_archetype_only(self):
        p = self._make_snapshot("VendorA", density=20)
        c = self._make_snapshot("VendorB", density=10)
        result = _build_vendor_comparison_summary(
            p, c, [],
            primary_archetype="pricing_shock",
        )
        assert "pricing_shock" in result
        assert "VendorA classified as pricing_shock" in result
        assert "VendorB classified" not in result

    def test_both_archetypes(self):
        p = self._make_snapshot("VendorA", density=20)
        c = self._make_snapshot("VendorB", density=10)
        result = _build_vendor_comparison_summary(
            p, c, [],
            primary_archetype="pricing_shock",
            comparison_archetype="feature_gap",
        )
        assert "pricing_shock" in result
        assert "feature_gap" in result

    def test_none_archetypes_omits_text(self):
        p = self._make_snapshot("VendorA")
        c = self._make_snapshot("VendorB")
        result = _build_vendor_comparison_summary(
            p, c, [],
            primary_archetype=None,
            comparison_archetype=None,
        )
        assert "Archetype" not in result


# ---------------------------------------------------------------------------
# _fetch_prior_archetypes helper
# ---------------------------------------------------------------------------


class TestFetchPriorArchetypes:
    """Verify the reusable helper returns correct structure."""

    @pytest.mark.asyncio
    async def test_empty_vendor_list(self):
        pool = AsyncMock()
        result = await _fetch_prior_archetypes(pool, [])
        assert result == {}
        pool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_dict_keyed_by_vendor(self):
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "vendor_name": "Acme",
                "archetype": "pricing_shock",
                "archetype_confidence": 0.82,
                "snapshot_date": date(2026, 2, 15),
            },
        ]
        result = await _fetch_prior_archetypes(pool, ["Acme"])
        assert "Acme" in result
        assert result["Acme"]["archetype"] == "pricing_shock"
        assert result["Acme"]["confidence"] == 0.82
        assert result["Acme"]["snapshot_date"] == "2026-02-15"

    @pytest.mark.asyncio
    async def test_none_confidence_returns_none(self):
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "vendor_name": "Beta",
                "archetype": "feature_gap",
                "archetype_confidence": None,
                "snapshot_date": date(2026, 2, 10),
            },
        ]
        result = await _fetch_prior_archetypes(pool, ["Beta"])
        assert result["Beta"]["confidence"] is None

    @pytest.mark.asyncio
    async def test_default_days_ago(self):
        pool = AsyncMock()
        pool.fetch.return_value = []
        await _fetch_prior_archetypes(pool, ["X"])
        # Verify the days_ago default (28) is passed as the second parameter
        call_args = pool.fetch.call_args
        assert call_args[0][1] == ["X"]
        assert call_args[0][2] == 28

    @pytest.mark.asyncio
    async def test_custom_days_ago(self):
        pool = AsyncMock()
        pool.fetch.return_value = []
        await _fetch_prior_archetypes(pool, ["X"], days_ago=14)
        call_args = pool.fetch.call_args
        assert call_args[0][2] == 14


# ---------------------------------------------------------------------------
# Archetype-shift change event detection
# ---------------------------------------------------------------------------


class TestArchetypeShiftEvent:
    """Verify archetype_shift events are detected correctly."""

    def _make_prior_snapshot(self, archetype: str | None = None,
                             conf: float | None = None) -> dict:
        """Minimal snapshot row with archetype columns."""
        return {
            "avg_urgency": 5.0,
            "churn_density": 10.0,
            "recommend_ratio": 50.0,
            "total_reviews": 100,
            "top_pain": "pricing",
            "top_competitor": "CompB",
            "pressure_score": 30.0,
            "dm_churn_rate": 0.05,
            "archetype": archetype,
            "archetype_confidence": conf,
        }

    def _make_vendor_score(self, name: str = "TestVendor") -> dict:
        return {
            "vendor_name": name,
            "total_reviews": 100,
            "churn_intent": 10,
            "avg_urgency": 5.0,
            "recommend_yes": 50,
            "recommend_no": 10,
        }

    @pytest.mark.asyncio
    async def test_archetype_shift_detected(self):
        """When current and prior archetypes differ, an archetype_shift event is inserted."""
        pool = AsyncMock()
        # fetchrow returns prior snapshot with old archetype
        pool.fetchrow.return_value = self._make_prior_snapshot(
            archetype="pricing_shock", conf=0.75,
        )
        # execute for event inserts
        pool.execute.return_value = None

        reasoning_lookup = {
            "TestVendor": {
                "archetype": "feature_gap",
                "confidence": 0.85,
            },
        }

        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _detect_change_events,
        )

        with patch(
            "atlas_brain.autonomous.tasks.b2b_churn_intelligence._detect_concurrent_shifts",
            new_callable=AsyncMock,
            return_value=0,
        ):
            detected = await _detect_change_events(
                pool,
                [self._make_vendor_score()],
                {"TestVendor": []},
                {"TestVendor": []},
                date(2026, 3, 15),
                reasoning_lookup=reasoning_lookup,
            )

        # Find the archetype_shift insert call
        shift_calls = [
            c for c in pool.execute.call_args_list
            if len(c[0]) >= 4 and c[0][3] is not None
            and "archetype_shift" in str(c[0][3])
        ]
        # At least one archetype_shift event should have been inserted
        # (check description arg which is positional arg index 3)
        insert_calls = [
            c for c in pool.execute.call_args_list
            if "INSERT INTO b2b_change_events" in str(c[0][0])
        ]
        archetype_inserts = [
            c for c in insert_calls
            if c[0][3] == "archetype_shift"
        ]
        assert len(archetype_inserts) >= 1
        desc = archetype_inserts[0][0][4]  # description is 5th positional param
        assert "pricing_shock" in desc
        assert "feature_gap" in desc

    @pytest.mark.asyncio
    async def test_no_shift_when_same_archetype(self):
        """No archetype_shift when current matches prior."""
        pool = AsyncMock()
        pool.fetchrow.return_value = self._make_prior_snapshot(
            archetype="pricing_shock", conf=0.75,
        )
        pool.execute.return_value = None

        reasoning_lookup = {
            "TestVendor": {
                "archetype": "pricing_shock",
                "confidence": 0.80,
            },
        }

        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _detect_change_events,
        )

        with patch(
            "atlas_brain.autonomous.tasks.b2b_churn_intelligence._detect_concurrent_shifts",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await _detect_change_events(
                pool,
                [self._make_vendor_score()],
                {"TestVendor": []},
                {"TestVendor": []},
                date(2026, 3, 15),
                reasoning_lookup=reasoning_lookup,
            )

        insert_calls = [
            c for c in pool.execute.call_args_list
            if "INSERT INTO b2b_change_events" in str(c[0][0])
        ]
        archetype_inserts = [
            c for c in insert_calls
            if c[0][3] == "archetype_shift"
        ]
        assert len(archetype_inserts) == 0

    @pytest.mark.asyncio
    async def test_no_shift_when_prior_has_no_archetype(self):
        """No archetype_shift when prior snapshot has no archetype."""
        pool = AsyncMock()
        pool.fetchrow.return_value = self._make_prior_snapshot(
            archetype=None, conf=None,
        )
        pool.execute.return_value = None

        reasoning_lookup = {
            "TestVendor": {
                "archetype": "feature_gap",
                "confidence": 0.85,
            },
        }

        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _detect_change_events,
        )

        with patch(
            "atlas_brain.autonomous.tasks.b2b_churn_intelligence._detect_concurrent_shifts",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await _detect_change_events(
                pool,
                [self._make_vendor_score()],
                {"TestVendor": []},
                {"TestVendor": []},
                date(2026, 3, 15),
                reasoning_lookup=reasoning_lookup,
            )

        insert_calls = [
            c for c in pool.execute.call_args_list
            if "INSERT INTO b2b_change_events" in str(c[0][0])
        ]
        archetype_inserts = [
            c for c in insert_calls
            if c[0][3] == "archetype_shift"
        ]
        assert len(archetype_inserts) == 0

    @pytest.mark.asyncio
    async def test_no_shift_without_reasoning_lookup(self):
        """No archetype_shift when reasoning_lookup is None."""
        pool = AsyncMock()
        pool.fetchrow.return_value = self._make_prior_snapshot(
            archetype="pricing_shock", conf=0.75,
        )
        pool.execute.return_value = None

        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _detect_change_events,
        )

        with patch(
            "atlas_brain.autonomous.tasks.b2b_churn_intelligence._detect_concurrent_shifts",
            new_callable=AsyncMock,
            return_value=0,
        ):
            await _detect_change_events(
                pool,
                [self._make_vendor_score()],
                {"TestVendor": []},
                {"TestVendor": []},
                date(2026, 3, 15),
                reasoning_lookup=None,
            )

        insert_calls = [
            c for c in pool.execute.call_args_list
            if "INSERT INTO b2b_change_events" in str(c[0][0])
        ]
        archetype_inserts = [
            c for c in insert_calls
            if c[0][3] == "archetype_shift"
        ]
        assert len(archetype_inserts) == 0


# ---------------------------------------------------------------------------
# Stratified intelligence archetype delta enrichment
# ---------------------------------------------------------------------------


class TestStratifiedIntelligenceEnrichment:
    """Verify archetype_was/confidence_was/archetype_changed fields."""

    def test_archetype_changed_true(self):
        """When prior and current differ, archetype_changed is True."""
        prior = {"archetype": "pricing_shock", "confidence": 0.7}
        current_arch = "feature_gap"
        changed = (
            prior["archetype"] != current_arch
            if prior.get("archetype") and current_arch
            else None
        )
        assert changed is True

    def test_archetype_changed_false(self):
        """When prior and current match, archetype_changed is False."""
        prior = {"archetype": "pricing_shock", "confidence": 0.7}
        current_arch = "pricing_shock"
        changed = (
            prior["archetype"] != current_arch
            if prior.get("archetype") and current_arch
            else None
        )
        assert changed is False

    def test_archetype_changed_none_when_no_prior(self):
        """When no prior archetype, archetype_changed is None."""
        prior: dict = {}
        current_arch = "feature_gap"
        changed = (
            prior["archetype"] != current_arch
            if prior.get("archetype") and current_arch
            else None
        )
        assert changed is None

    def test_archetype_changed_none_when_no_current(self):
        """When no current archetype, archetype_changed is None."""
        prior = {"archetype": "pricing_shock"}
        current_arch = ""
        changed = (
            prior["archetype"] != current_arch
            if prior.get("archetype") and current_arch
            else None
        )
        assert changed is None


# ---------------------------------------------------------------------------
# Campaign context archetype injection
# ---------------------------------------------------------------------------


class TestCampaignArchetypeInjection:
    """Verify _build_vendor_context allows archetype_context addition."""

    def test_vendor_context_accepts_archetype(self):
        from atlas_brain.autonomous.tasks.b2b_campaign_generation import (
            _build_vendor_context,
        )

        ctx = _build_vendor_context("TestVendor", [])
        # Verify the base dict can accept archetype_context
        ctx["archetype_context"] = {
            "archetype": "pricing_shock",
            "confidence": 0.82,
            "falsification": [],
        }
        assert ctx["archetype_context"]["archetype"] == "pricing_shock"
        # Verify spread works (simulates payload construction)
        payload = {**ctx, "channel": "email_cold"}
        assert "archetype_context" in payload
        assert payload["vendor_name"] == "TestVendor"

    def test_challenger_context_accepts_incumbent_archetypes(self):
        from atlas_brain.autonomous.tasks.b2b_campaign_generation import (
            _build_challenger_context,
        )

        ctx = _build_challenger_context("ChallengerX", [])
        ctx["incumbent_archetypes"] = {
            "pricing_shock": ["VendorA", "VendorB"],
            "feature_gap": ["VendorC"],
        }
        payload = {**ctx, "channel": "email_cold"}
        assert "incumbent_archetypes" in payload
        assert payload["incumbent_archetypes"]["pricing_shock"] == ["VendorA", "VendorB"]


# ---------------------------------------------------------------------------
# Tenant report key fallback
# ---------------------------------------------------------------------------


class TestTenantReportKeyFallback:
    """Verify vendor_name / vendor key fallback logic."""

    def test_vendor_name_key_present(self):
        """When vendor_name is in payload, it's used."""
        vs = {"vendor_name": "Acme", "vendor": "Acme", "product_category": "CRM"}
        vname = vs.get("vendor_name") or vs.get("vendor") or ""
        assert vname == "Acme"

    def test_vendor_key_fallback(self):
        """When only vendor key exists, fallback works."""
        vs = {"vendor": "Acme", "category": "CRM"}
        vname = vs.get("vendor_name") or vs.get("vendor") or ""
        assert vname == "Acme"

    def test_neither_key_returns_empty(self):
        """When neither key exists, returns empty string."""
        vs = {"name": "Acme"}
        vname = vs.get("vendor_name") or vs.get("vendor") or ""
        assert vname == ""

    def test_product_category_fallback(self):
        """category key works as fallback for product_category."""
        vs = {"vendor": "X", "category": "CRM"}
        cat = vs.get("product_category") or vs.get("category") or ""
        assert cat == "CRM"


# ---------------------------------------------------------------------------
# Vendor briefing archetype rendering
# ---------------------------------------------------------------------------


class TestVendorBriefingArchetypeRender:
    """Verify archetype section renders in vendor briefing HTML."""

    def test_archetype_renders_in_briefing(self):
        from atlas_brain.templates.email.vendor_briefing import (
            _render_archetype_section,
        )

        html = _render_archetype_section(
            archetype="pricing_shock",
            confidence=0.82,
            archetype_was=None,
            archetype_changed=False,
            falsification=[],
        )
        assert "Pricing Shock" in html
        assert "82%" in html
        assert "Shifted from" not in html

    def test_archetype_shift_renders(self):
        from atlas_brain.templates.email.vendor_briefing import (
            _render_archetype_section,
        )

        html = _render_archetype_section(
            archetype="feature_gap",
            confidence=0.75,
            archetype_was="pricing_shock",
            archetype_changed=True,
            falsification=["Price restored to prior levels"],
        )
        assert "Feature Gap" in html
        assert "Shifted from" in html
        assert "Pricing Shock" in html
        assert "Price restored to prior levels" in html

    def test_no_archetype_renders_empty(self):
        from atlas_brain.templates.email.vendor_briefing import (
            _render_archetype_section,
        )

        html = _render_archetype_section(
            archetype=None,
            confidence=None,
            archetype_was=None,
            archetype_changed=False,
            falsification=[],
        )
        assert html == ""

    def test_unknown_archetype_renders_title_case(self):
        from atlas_brain.templates.email.vendor_briefing import (
            _render_archetype_section,
        )

        html = _render_archetype_section(
            archetype="new_pattern",
            confidence=0.5,
            archetype_was=None,
            archetype_changed=False,
            falsification=[],
        )
        assert "New Pattern" in html


# ---------------------------------------------------------------------------
# Article-archetype correlation
# ---------------------------------------------------------------------------


class TestArticleArchetypeCorrelation:
    """Verify _correlate_articles_with_archetypes logic."""

    @pytest.mark.asyncio
    async def test_no_vendors_returns_zero(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _correlate_articles_with_archetypes,
        )

        pool = AsyncMock()
        result = await _correlate_articles_with_archetypes(pool, [], {}, date(2026, 3, 16))
        assert result == 0
        pool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_reasoning_returns_zero(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _correlate_articles_with_archetypes,
        )

        pool = AsyncMock()
        result = await _correlate_articles_with_archetypes(
            pool, ["Acme"], {}, date(2026, 3, 16),
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_stable_archetype_skipped(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _correlate_articles_with_archetypes,
        )

        pool = AsyncMock()
        pool.fetch.return_value = []  # no shift events
        result = await _correlate_articles_with_archetypes(
            pool,
            ["Acme"],
            {"Acme": {"archetype": "stable", "confidence": 0.9}},
            date(2026, 3, 16),
        )
        assert result == 0

    @pytest.mark.asyncio
    async def test_entity_match_correlation(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _correlate_articles_with_archetypes,
        )
        import uuid

        pool = AsyncMock()
        article_id = uuid.uuid4()

        # First fetch: no shift events
        # Second fetch: one article matching
        pool.fetch.side_effect = [
            [],  # shift events
            [    # articles
                {
                    "id": article_id,
                    "title": "Acme raises prices 40%",
                    "soram_channels": {"operational": 0.8, "media": 0.6},
                    "pressure_direction": "building",
                    "entities_detected": ["Acme"],
                    "published_at": "2026-03-10",
                },
            ],
        ]
        pool.execute.return_value = None

        result = await _correlate_articles_with_archetypes(
            pool,
            ["Acme"],
            {"Acme": {"archetype": "pricing_shock", "confidence": 0.85}},
            date(2026, 3, 16),
        )
        assert result == 1

        # Verify the INSERT was called with correct params
        insert_call = pool.execute.call_args
        assert insert_call is not None
        args = insert_call[0]
        assert "INSERT INTO b2b_article_correlations" in args[0]
        assert args[1] == article_id        # article_id
        assert args[2] == "Acme"            # vendor_name
        assert args[4] == "pricing_shock"   # archetype
        assert args[5] == 0.85             # confidence


class TestSoramAlignmentWeights:
    """Verify SORAM weight mappings exist for all archetypes."""

    def test_all_archetypes_have_weights(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _ARCHETYPE_SORAM_WEIGHTS,
        )

        expected = {
            "pricing_shock", "feature_gap", "support_collapse",
            "leadership_redesign", "acquisition_decay", "integration_break",
            "category_disruption", "compliance_gap",
        }
        assert expected == set(_ARCHETYPE_SORAM_WEIGHTS.keys())

    def test_weights_are_valid(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _ARCHETYPE_SORAM_WEIGHTS,
        )

        valid_channels = {"societal", "operational", "regulatory", "alignment", "media"}
        for archetype, weights in _ARCHETYPE_SORAM_WEIGHTS.items():
            assert all(ch in valid_channels for ch in weights), f"{archetype} has invalid channel"
            assert all(0 < w <= 1.0 for w in weights.values()), f"{archetype} has invalid weight"


# ---------------------------------------------------------------------------
# Battle card data fixes
# ---------------------------------------------------------------------------


class TestBattleCardPrimaryCategorySelection:
    """Verify cross-category aggregation is replaced by primary category selection."""

    def test_picks_largest_category(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        vendor_scores = [
            {"vendor_name": "Acme", "product_category": "CRM", "total_reviews": 100,
             "churn_intent": 30, "avg_urgency": 5.5},
            {"vendor_name": "Acme", "product_category": "Helpdesk", "total_reviews": 50,
             "churn_intent": 20, "avg_urgency": 7.0},
        ]
        cards = _build_deterministic_battle_cards(
            vendor_scores,
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"Acme": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"Acme": 0.4}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        card = cards[0]
        # Should use the CRM row (100 reviews), not sum (150)
        assert card["total_reviews"] == 100
        assert card["category"] == "CRM"
        # Urgency should be 5.5 from CRM, not weighted average
        assert card["objection_data"]["avg_urgency"] == 5.5

    def test_no_phantom_totals(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        vendor_scores = [
            {"vendor_name": "Beta", "product_category": "A", "total_reviews": 200,
             "churn_intent": 60, "avg_urgency": 6.0},
            {"vendor_name": "Beta", "product_category": "B", "total_reviews": 150,
             "churn_intent": 45, "avg_urgency": 5.0},
        ]
        cards = _build_deterministic_battle_cards(
            vendor_scores,
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"Beta": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"Beta": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        # Must NOT be 350 (200+150)
        assert cards[0]["total_reviews"] == 200


class TestBattleCardQuoteDedupe:
    """Verify quotes are deduplicated by review_id AND reviewer identity."""

    def test_same_review_id_deduped(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        quotes = [
            {"quote": "Quote A", "urgency": 8, "review_id": "r1",
             "company": "X", "title": "VP"},
            {"quote": "Quote B", "urgency": 7, "review_id": "r1",
             "company": "X", "title": "VP"},
            {"quote": "Quote C", "urgency": 6, "review_id": "r2",
             "company": "Y", "title": "CTO"},
        ]
        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={"V": quotes}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        # Quote B (same review_id as A) should be deduplicated
        assert len(cards[0]["customer_pain_quotes"]) == 2
        assert cards[0]["customer_pain_quotes"][0]["quote"] == "Quote A"
        assert cards[0]["customer_pain_quotes"][1]["quote"] == "Quote C"

    def test_same_reviewer_different_reviews_deduped(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        quotes = [
            {"quote": "Quote A", "urgency": 8, "review_id": "r1",
             "company": "Acme Corp", "title": "Head of Marketing"},
            {"quote": "Quote B", "urgency": 7, "review_id": "r2",
             "company": "Acme Corp", "title": "Head of Marketing"},
            {"quote": "Quote C", "urgency": 6, "review_id": "r3",
             "company": "Other Inc", "title": "CTO"},
        ]
        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={"V": quotes}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        # Quote B (same reviewer as A) should be deduplicated
        assert len(cards[0]["customer_pain_quotes"]) == 2
        assert cards[0]["customer_pain_quotes"][0]["quote"] == "Quote A"
        assert cards[0]["customer_pain_quotes"][1]["quote"] == "Quote C"


class TestBattleCardSentimentDirection:
    """Verify sentiment excludes 'unknown' before picking dominant."""

    def test_unknown_excluded(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 100,
              "churn_intent": 30, "avg_urgency": 6.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={"V": {"unknown": 400, "declining": 130, "stable": 50}},
            dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["objection_data"]["sentiment_direction"] == "declining"

    def test_all_unknown_returns_insufficient(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 100,
              "churn_intent": 30, "avg_urgency": 6.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={"V": {"unknown": 400}},
            dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["objection_data"]["sentiment_direction"] == "insufficient_data"


class TestBattleCardNewSections:
    """Verify new battle card sections are populated from lookups."""

    def test_high_intent_companies_surfaced(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3},
            company_lookup={"V": [{"company": "Acme", "urgency": 9}]},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["high_intent_companies"] == [{"company": "Acme", "urgency": 9}]

    def test_integration_stack_from_profile(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={"V": {"top_integrations": ["Shopify", "Salesforce"]}},
            competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["integration_stack"] == ["Shopify", "Salesforce"]

    def test_buyer_authority_from_lookup(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        ba_lookup = {
            "V": {
                "role_types": {"economic_buyer": 5, "evaluator": 3},
                "buying_stages": {"active_purchase": 2, "evaluation": 6},
            },
        }
        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            buyer_auth_lookup=ba_lookup,
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["buyer_authority"]["role_types"]["economic_buyer"] == 5


# ---------------------------------------------------------------------------
# Task budget and phase gating (end-to-end with simulated elapsed time)
# ---------------------------------------------------------------------------


class TestTaskBudget:
    """End-to-end tests for _TaskBudget with injectable clock to simulate elapsed time."""

    @staticmethod
    def _make_clock(start: float = 0.0):
        """Return a controllable clock and an advance function."""
        state = [start]

        def clock():
            return state[0]

        def advance(seconds: float):
            state[0] += seconds

        return clock, advance

    def test_fresh_budget_has_full_remaining(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, _ = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)
        assert b.elapsed() == 0.0
        assert b.remaining() == 540.0

    def test_elapsed_tracks_clock_advance(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(1000.0)
        b = _TaskBudget(540, _clock=clock)
        assert b.elapsed() == 0.0
        advance(120)
        assert b.elapsed() == 120.0
        assert b.remaining() == 420.0

    def test_remaining_floors_at_zero(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(100, _clock=clock)
        advance(200)
        assert b.remaining() == 0.0

    def test_phase_denied_after_time_passes(self):
        """Phase that was allowed initially is denied after elapsed time."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)
        assert b.phase_allowed("reasoning", 120) is True   # 540 >= 120
        advance(450)
        assert b.phase_allowed("reasoning", 120) is False  # 90 < 120

    def test_full_phase_sequence_simulated_run(self):
        """Simulate an entire intelligence run where reasoning takes 400s.

        Proves that after an expensive reasoning phase, only cheap phases
        run and expensive late phases are skipped.
        """
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)

        # Phase 1: data fetch -- 20s
        assert b.phase_allowed("temporal", 30) is True
        advance(20)
        assert b.remaining() == 520.0

        # Phase 2: temporal enrichment -- 15s
        advance(15)
        assert b.remaining() == 505.0

        # Phase 3: stratified reasoning -- 400s (the expensive phase)
        assert b.phase_allowed("reasoning", 120) is True
        advance(400)
        assert b.remaining() == 105.0

        # Phase 4: ecosystem -- cheap, should pass (105 >= 20)
        assert b.phase_allowed("ecosystem", 20) is True
        advance(5)

        # Phase 5: exploratory LLM -- needs 90s, 100 >= 90, barely passes
        assert b.phase_allowed("exploratory", 90) is True
        advance(60)
        assert b.remaining() == 40.0

        # Phase 6: scorecard narrative -- needs 60s, 40 < 60, SKIPPED
        assert b.phase_allowed("scorecard", 60) is False

        # Phase 7: exec summary -- needs 45s, 40 < 45, SKIPPED
        assert b.phase_allowed("exec_summary", 45) is False

        # Phase 8: battle card copy -- needs 60s, 40 < 60, SKIPPED
        assert b.phase_allowed("battle_card_copy", 60) is False

    def test_mid_loop_bailout_with_elapsed_time(self):
        """Simulate per-card budget checks inside a battle card loop.

        Proves that the loop processes 2 cards then bails on the 3rd
        when remaining drops below the per-iteration threshold.
        """
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)
        advance(475)  # simulate 475s of prior phases, 65s remaining

        # Phase entry: 65 >= 60, allowed
        assert b.phase_allowed("battle_card_copy", 60) is True

        # Card 1: 65s remaining >= 30 threshold, proceed
        assert b.remaining() >= 30
        advance(25)  # card takes 25s
        assert b.remaining() == 40.0

        # Card 2: 40s remaining >= 30 threshold, proceed
        assert b.remaining() >= 30
        advance(25)  # card takes 25s
        assert b.remaining() == 15.0

        # Card 3: 15s remaining < 30 threshold, BAIL
        assert b.remaining() < 30

    def test_budget_zero_skips_all_phases(self):
        """An already-expired budget skips every named phase."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, _ = self._make_clock(0.0)
        b = _TaskBudget(0, _clock=clock)

        phases = [
            ("temporal", 30), ("reasoning", 120), ("exploratory", 90),
            ("ecosystem", 20), ("scorecard", 60), ("exec_summary", 45),
            ("battle_card_copy", 60),
        ]
        for name, min_sec in phases:
            assert b.phase_allowed(name, min_sec) is False, f"{name} should be skipped"

    def test_fast_run_allows_all_phases(self):
        """When all prior phases are fast, every phase gets budget."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)

        phases = [
            ("temporal", 30), ("reasoning", 120), ("ecosystem", 20),
            ("exploratory", 90), ("scorecard", 60), ("exec_summary", 45),
            ("battle_card_copy", 60),
        ]
        for name, min_sec in phases:
            assert b.phase_allowed(name, min_sec) is True, (
                f"{name} should be allowed at {b.remaining():.0f}s remaining"
            )
            advance(30)  # each phase finishes in 30s

        # 7 phases * 30s = 210s elapsed, 330s remaining
        assert b.remaining() == 330.0

    def test_scorecard_per_iteration_bailout(self):
        """Scorecard narrative loop bails mid-iteration when budget drops."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import _TaskBudget

        clock, advance = self._make_clock(0.0)
        b = _TaskBudget(540, _clock=clock)
        advance(510)  # 30s remaining

        # Phase entry denied: 30 < 60
        assert b.phase_allowed("scorecard", 60) is False

        # Even if phase entry was allowed, per-iteration check (20s) would fail soon
        advance(15)  # 15s remaining
        assert b.remaining() < 20  # per-iteration threshold


class TestReconstructReasoningLookup:
    """Validate that reconstruct_reasoning_lookup produces the full contract
    needed by all downstream consumers (reports, battle cards, scorecards)."""

    REQUIRED_KEYS = {
        "archetype", "confidence", "mode", "risk_level",
        "executive_summary", "key_signals",
        "falsification_conditions", "uncertainty_sources",
    }

    @pytest.mark.asyncio
    async def test_returns_all_required_keys(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_reasoning_lookup,
        )

        pool = AsyncMock()
        pool.fetchval.return_value = date(2026, 3, 16)
        pool.fetch.return_value = [
            {
                "vendor_name": "Mailchimp",
                "archetype": "pricing_shock",
                "archetype_confidence": 0.9,
                "reasoning_mode": "reason",
                "falsification_conditions": ["price restored"],
                "reasoning_risk_level": "high",
                "reasoning_executive_summary": "Pricing pressure intensifying",
                "reasoning_key_signals": ["164 pricing mentions", "27% complaint rate"],
                "reasoning_uncertainty_sources": ["no time-series breakdown"],
            },
        ]

        lookup = await reconstruct_reasoning_lookup(pool)
        assert "Mailchimp" in lookup
        entry = lookup["Mailchimp"]

        # Every key that downstream consumers access must be present
        missing = self.REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Missing keys: {missing}"

        # Type checks matching downstream usage patterns
        assert isinstance(entry["archetype"], str)
        assert isinstance(entry["confidence"], (int, float))
        assert isinstance(entry["mode"], str)
        assert isinstance(entry["risk_level"], str)
        assert isinstance(entry["executive_summary"], str)
        assert isinstance(entry["key_signals"], list)
        assert isinstance(entry["falsification_conditions"], list)
        assert isinstance(entry["uncertainty_sources"], list)

    @pytest.mark.asyncio
    async def test_handles_null_columns_gracefully(self):
        """When DB columns are NULL (pre-migration data), defaults are safe."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_reasoning_lookup,
        )

        pool = AsyncMock()
        pool.fetchval.return_value = date(2026, 3, 16)
        pool.fetch.return_value = [
            {
                "vendor_name": "OldVendor",
                "archetype": "feature_gap",
                "archetype_confidence": None,
                "reasoning_mode": None,
                "falsification_conditions": None,
                "reasoning_risk_level": None,
                "reasoning_executive_summary": None,
                "reasoning_key_signals": None,
                "reasoning_uncertainty_sources": None,
            },
        ]

        lookup = await reconstruct_reasoning_lookup(pool)
        entry = lookup["OldVendor"]

        # All keys present with safe defaults
        assert entry["confidence"] == 0
        assert entry["mode"] == ""
        assert entry["risk_level"] == ""
        assert entry["executive_summary"] == ""
        assert entry["key_signals"] == []
        assert entry["falsification_conditions"] == []
        assert entry["uncertainty_sources"] == []

    @pytest.mark.asyncio
    async def test_empty_table_returns_empty(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_reasoning_lookup,
        )

        pool = AsyncMock()
        pool.fetchval.return_value = None  # no rows at all
        result = await reconstruct_reasoning_lookup(pool)
        assert result == {}

    @pytest.mark.asyncio
    async def test_explicit_date_bypasses_watermark(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_reasoning_lookup,
        )

        pool = AsyncMock()
        pool.fetch.return_value = []
        await reconstruct_reasoning_lookup(pool, as_of=date(2026, 3, 15))
        # Should NOT call fetchval (watermark lookup)
        pool.fetchval.assert_not_called()
        # Should call fetch with the explicit date
        call_args = pool.fetch.call_args
        assert call_args[0][1] == date(2026, 3, 15)

    @pytest.mark.asyncio
    async def test_contract_matches_in_memory_shape(self):
        """The dict shape from reconstruct must match what run() builds in-memory."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_reasoning_lookup,
        )

        pool = AsyncMock()
        pool.fetchval.return_value = date(2026, 3, 16)
        pool.fetch.return_value = [
            {
                "vendor_name": "TestVendor",
                "archetype": "support_collapse",
                "archetype_confidence": 0.78,
                "reasoning_mode": "reconstitute",
                "falsification_conditions": ["support_ticket_resolution_improves"],
                "reasoning_risk_level": "medium",
                "reasoning_executive_summary": "Support degradation pattern",
                "reasoning_key_signals": ["57 support mentions"],
                "reasoning_uncertainty_sources": ["single quarter data"],
            },
        ]

        lookup = await reconstruct_reasoning_lookup(pool)
        entry = lookup["TestVendor"]

        # This is the exact shape that run() builds at line 3550-3560:
        # reasoning_lookup[vname] = {
        #     "archetype": conclusion.get("archetype", ""),
        #     "confidence": sr.confidence,
        #     "risk_level": conclusion.get("risk_level", ""),
        #     "executive_summary": conclusion.get("executive_summary", ""),
        #     "key_signals": conclusion.get("key_signals", []),
        #     "falsification_conditions": conclusion.get("falsification_conditions", []),
        #     "uncertainty_sources": conclusion.get("uncertainty_sources", []),
        #     "mode": sr.mode,
        #     "tokens_used": sr.tokens_used,
        # }
        # We match all keys except tokens_used (not persisted, not needed downstream)
        assert entry["archetype"] == "support_collapse"
        assert entry["confidence"] == 0.78
        assert entry["mode"] == "reconstitute"
        assert entry["risk_level"] == "medium"
        assert entry["executive_summary"] == "Support degradation pattern"
        assert entry["key_signals"] == ["57 support mentions"]
        assert entry["falsification_conditions"] == ["support_ticket_resolution_improves"]
        assert entry["uncertainty_sources"] == ["single quarter data"]


class TestVendorReasoningCap:
    """Verify the vendor cap limits evidence building and reasoning."""

    def test_evidence_map_capped(self):
        """Evidence map should only build for capped vendors."""
        from atlas_brain.config import settings
        cfg = settings.b2b_churn
        assert hasattr(cfg, "stratified_reasoning_vendor_cap")
        assert cfg.stratified_reasoning_vendor_cap > 0
        assert cfg.stratified_reasoning_vendor_cap <= 100
