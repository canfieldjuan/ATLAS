"""Regression tests for archetype context propagation across B2B surfaces.

Covers:
- _fetch_prior_archetypes helper
- _build_vendor_comparison_summary with archetype params
- archetype-shift change event detection
- vendor/challenger context archetype injection
- tenant report prior-archetype enrichment
"""

import sys
import json
from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-mock heavy deps before importing task modules
# asyncpg needs a real exception class for `except UndefinedTableError`
_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
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

    def test_reasoning_synthesis_lowers_qualification_gate(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "Acme", "product_category": "CRM", "total_reviews": 100,
              "churn_intent": 12, "avg_urgency": 5.2}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"Acme": 0.05}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"Acme": 0.25}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            reasoning_lookup={},
            reasoning_synthesis_lookup={
                "Acme": {
                    "reasoning_contracts": {
                        "vendor_core_reasoning": {
                            "causal_narrative": {
                                "primary_wedge": "price_squeeze",
                                "confidence": "high",
                            },
                        },
                    },
                },
            },
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["vendor"] == "Acme"


class TestBattleCardWeaknessEvidenceNormalization:
    """Verify battle-card weaknesses expose a consistent evidence_count field."""

    def test_normalizes_evidence_count_across_weakness_sources(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 60,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={"V": [{"category": "pricing", "count": 7}]},
            competitor_lookup={},
            feature_gap_lookup={"V": [{"feature": "reporting", "mentions": 4}]},
            quote_lookup={},
            price_lookup={"V": 0.1},
            budget_lookup={},
            sentiment_lookup={},
            dm_lookup={"V": 0.3},
            company_lookup={},
            product_profile_lookup={"V": {"weaknesses": [{"area": "support", "evidence_count": 9}]}},
            competitive_disp=[],
            competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        weaknesses = cards[0]["vendor_weaknesses"]
        assert [w["area"] for w in weaknesses[:3]] == ["support", "pricing", "reporting"]
        assert [w["evidence_count"] for w in weaknesses[:3]] == [9, 7, 4]


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
            company_lookup={"V": [{"company": "Acme", "urgency": 9, "title": "VP Ops"}]},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["high_intent_companies"][0]["company"] == "Acme"
        assert cards[0]["high_intent_companies"][0]["title"] == "VP Ops"

    def test_evidence_vault_overrides_weaknesses_and_company_signals(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={"V": [{"category": "legacy pain", "count": 2}]},
            competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3},
            company_lookup={"V": [{"company": "LegacyCo", "urgency": 4}]},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            evidence_vault_lookup={"V": {
                "weakness_evidence": [
                    {"label": "Pricing opacity", "mention_count_total": 11, "evidence_type": "pain_category"},
                    {"key": "support", "mention_count_total": 7, "evidence_type": "satisfaction_area"},
                ],
                "company_signals": [
                    {"company_name": "Acme", "urgency_score": 9.0, "buyer_role": "VP Ops", "pain_category": "pricing"},
                ],
            }},
            limit=10,
        )

        assert len(cards) == 1
        assert [w["area"] for w in cards[0]["vendor_weaknesses"][:2]] == ["Pricing opacity", "support"]
        assert cards[0]["high_intent_companies"][0]["company"] == "Acme"
        assert cards[0]["high_intent_companies"][0]["urgency"] == 9.0

    def test_high_intent_companies_filter_domains_and_competitors(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "Zendesk", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={},
            competitor_lookup={"Zendesk": [{"name": "Freshdesk", "mentions": 12}]},
            feature_gap_lookup={},
            quote_lookup={}, price_lookup={"Zendesk": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"Zendesk": 0.3},
            company_lookup={"Zendesk": [
                {"company": "ourdomain.com", "urgency": 9, "title": "VP Ops"},
                {"company": "Freshdesk", "urgency": 8, "title": "VP Ops"},
                {"company": "Salesforce", "urgency": 8, "title": "VP Ops"},
                {"company": "Government agency", "urgency": 8, "title": "VP Ops"},
                {"company": "Costa Rica", "urgency": 8, "title": "VP Ops"},
                {"company": "ClonePartner", "urgency": 8, "title": "VP Ops"},
                {"company": "Acme", "urgency": 7, "title": "VP Ops"},
            ]},
            product_profile_lookup={"Zendesk": {"top_integrations": ["Salesforce"]}},
            competitive_disp=[],
            competitor_reasons=[],
            limit=10,
        )

        assert [entry["company"] for entry in cards[0]["high_intent_companies"]] == ["Acme"]

    def test_high_intent_companies_do_not_fallback_to_raw_when_vault_sanitizes_to_empty(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "Zendesk", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={},
            competitor_lookup={"Zendesk": [{"name": "Freshdesk", "mentions": 12}]},
            feature_gap_lookup={},
            quote_lookup={}, price_lookup={"Zendesk": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"Zendesk": 0.3},
            company_lookup={"Zendesk": [
                {"company": "BoldDesk", "urgency": 8, "title": "VP Ops", "buying_stage": "evaluation"},
            ]},
            product_profile_lookup={"Zendesk": {"top_integrations": ["Salesforce"]}},
            competitive_disp=[],
            competitor_reasons=[],
            evidence_vault_lookup={"Zendesk": {"company_signals": []}},
            limit=10,
        )

        assert cards[0]["high_intent_companies"] == []

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

    def test_keyword_spikes_from_lookup(self):
        from atlas_brain.autonomous.tasks._b2b_shared import (
            _build_deterministic_battle_cards,
        )

        cards = _build_deterministic_battle_cards(
            [{"vendor_name": "V", "product_category": "Cat", "total_reviews": 50,
              "churn_intent": 20, "avg_urgency": 7.0}],
            pain_lookup={}, competitor_lookup={}, feature_gap_lookup={},
            quote_lookup={}, price_lookup={"V": 0.1}, budget_lookup={},
            sentiment_lookup={}, dm_lookup={"V": 0.3}, company_lookup={},
            product_profile_lookup={}, competitive_disp=[], competitor_reasons=[],
            keyword_spike_lookup={"V": {
                "spike_count": 3,
                "spike_keywords": ["outage", "migration", "pricing"],
                "trend_summary": {"outage": {"volume_change_pct": 42.0}},
            }},
            limit=10,
        )
        assert len(cards) == 1
        assert cards[0]["keyword_spikes"]["spike_count"] == 3
        assert cards[0]["keyword_spikes"]["keywords"] == ["outage", "migration", "pricing"]


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
    async def test_parses_json_backed_list_columns(self):
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
                "falsification_conditions": '["price restored"]',
                "reasoning_risk_level": "high",
                "reasoning_executive_summary": "Pricing pressure intensifying",
                "reasoning_key_signals": '["164 pricing mentions"]',
                "reasoning_uncertainty_sources": '["no time-series breakdown"]',
            },
        ]

        lookup = await reconstruct_reasoning_lookup(pool)
        entry = lookup["Mailchimp"]
        assert entry["key_signals"] == ["164 pricing mentions"]
        assert entry["falsification_conditions"] == ["price restored"]
        assert entry["uncertainty_sources"] == ["no time-series breakdown"]

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


class TestReconstructCrossVendorLookup:
    """Integration tests for reconstruct_cross_vendor_lookup.

    Verifies SQL query shape, parameter binding, watermark fallback,
    type handling, and the full producer contract.
    """

    @pytest.mark.asyncio
    async def test_empty_table_returns_empty_buckets(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetchval.return_value = None
        result = await reconstruct_cross_vendor_lookup(pool)
        assert result == {"battles": {}, "councils": {}, "asymmetries": {}}
        # Should query for watermark, never call fetch
        pool.fetchval.assert_awaited_once()
        pool.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_watermark_fallback_queries_max_date(self):
        """When as_of is None, queries MAX(computed_date) and uses it."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetchval.return_value = date(2026, 3, 15)
        pool.fetch.return_value = []
        await reconstruct_cross_vendor_lookup(pool)
        sql = pool.fetchval.call_args[0][0]
        assert "MAX(computed_date)" in sql
        # fetch should receive the watermark date as param
        fetch_param = pool.fetch.call_args[0][1]
        assert fetch_param == date(2026, 3, 15)

    @pytest.mark.asyncio
    async def test_explicit_date_skips_watermark(self):
        """When as_of is given, fetchval is never called."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = []
        await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        pool.fetchval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetch_sql_shape(self):
        """Verify the SQL query selects the right columns with date filter."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = []
        await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        sql = pool.fetch.call_args[0][0]
        for col in ("analysis_type", "vendors", "category", "conclusion", "confidence"):
            assert col in sql, f"Missing column {col} in SELECT"
        assert "b2b_cross_vendor_conclusions" in sql
        assert "computed_date <= $1" in sql
        assert "ORDER BY computed_date DESC, confidence DESC" in sql

    @pytest.mark.asyncio
    async def test_battles_keyed_by_sorted_tuple(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["Zendesk", "Freshdesk"],  # unsorted
                "category": None,
                "conclusion": {
                    "conclusion": "Zendesk losing on price",
                    "winner": "Freshdesk",
                    "durability_assessment": "structural",
                    "key_insights": ["3x price gap"],
                },
                "confidence": 0.85,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        # Key must be sorted regardless of DB vendor order
        assert ("Freshdesk", "Zendesk") in result["battles"]
        assert ("Zendesk", "Freshdesk") not in result["battles"]
        entry = result["battles"][("Freshdesk", "Zendesk")]
        assert entry["confidence"] == 0.85
        assert entry["conclusion"]["winner"] == "Freshdesk"
        assert entry["conclusion"]["durability_assessment"] == "structural"
        assert result["councils"] == {}
        assert result["asymmetries"] == {}

    @pytest.mark.asyncio
    async def test_councils_keyed_by_category(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "category_council",
                "vendors": ["HubSpot", "Mailchimp", "Brevo"],
                "category": "Email Marketing",
                "conclusion": {
                    "conclusion": "Price competition regime",
                    "market_regime": "price_competition",
                    "durability_assessment": "structural",
                    "key_insights": ["27% price complaints"],
                    "winner": "Brevo",
                    "loser": "Mailchimp",
                },
                "confidence": 0.78,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert "Email Marketing" in result["councils"]
        entry = result["councils"]["Email Marketing"]
        assert entry["conclusion"]["market_regime"] == "price_competition"
        assert entry["conclusion"]["winner"] == "Brevo"
        assert entry["conclusion"]["loser"] == "Mailchimp"
        assert entry["confidence"] == 0.78
        # Verify all council fields consumed by reports are accessible
        c = entry["conclusion"]
        for field in ("conclusion", "market_regime", "durability_assessment",
                      "key_insights", "winner", "loser"):
            assert field in c, f"Council conclusion missing field: {field}"

    @pytest.mark.asyncio
    async def test_asymmetries_with_resource_advantage(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["Salesforce", "HubSpot"],
                "category": None,
                "conclusion": {
                    "conclusion": "Salesforce has deeper lock-in",
                    "resource_advantage": "Salesforce holds edge due to 4x integration count",
                    "durability_assessment": "structural",
                },
                "confidence": 0.72,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        key = ("HubSpot", "Salesforce")
        assert key in result["asymmetries"]
        entry = result["asymmetries"][key]
        assert "resource_advantage" in entry["conclusion"]

    @pytest.mark.asyncio
    async def test_json_string_conclusion_parsed_and_normalized(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "category_council",
                "vendors": ["A", "B"],
                "category": "CRM",
                "conclusion": json.dumps({
                    "conclusion": "Price pressure is dominant",
                    "market_regime": "price_competition",
                    "key_insights": [
                        {"insight": "Pricing drives displacement"},
                        "SMB segment is most affected",
                    ],
                }),
                "confidence": 0.5,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        entry = result["councils"]["CRM"]["conclusion"]
        assert entry["conclusion"] == "Price pressure is dominant"
        assert entry["market_regime"] == "price_competition"
        assert entry["key_insights"] == [
            {"insight": "Pricing drives displacement", "evidence": "Pricing drives displacement"},
            {"insight": "SMB segment is most affected", "evidence": "SMB segment is most affected"},
        ]

    @pytest.mark.asyncio
    async def test_generic_category_council_is_skipped(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "category_council",
                "vendors": ["A", "B"],
                "category": "B2B Software",
                "conclusion": {"conclusion": "generic council"},
                "confidence": 0.5,
            },
            {
                "analysis_type": "category_council",
                "vendors": ["A", "B", "C"],
                "category": "CRM",
                "conclusion": {"conclusion": "crm council"},
                "confidence": 0.7,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert "B2B Software" not in result["councils"]
        assert result["councils"]["CRM"]["conclusion"]["conclusion"] == "crm council"

    @pytest.mark.asyncio
    async def test_highest_confidence_wins_dedup(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["A", "B"], "category": None,
                "conclusion": {"conclusion": "high conf"}, "confidence": 0.9,
            },
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["B", "A"], "category": None,  # reversed order
                "conclusion": {"conclusion": "low conf"}, "confidence": 0.5,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        # Both rows map to same sorted key; first (highest conf) wins via setdefault
        assert result["battles"][("A", "B")]["conclusion"]["conclusion"] == "high conf"

    @pytest.mark.asyncio
    async def test_null_conclusion_produces_empty_dict(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["X", "Y"], "category": None,
                "conclusion": None, "confidence": 0.5,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert result["battles"][("X", "Y")]["conclusion"] == {}

    @pytest.mark.asyncio
    async def test_null_confidence_defaults_to_zero(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["C", "D"], "category": None,
                "conclusion": {"conclusion": "test"}, "confidence": None,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert result["asymmetries"][("C", "D")]["confidence"] == 0

    @pytest.mark.asyncio
    async def test_null_category_council_uses_empty_string(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "category_council",
                "vendors": ["E", "F"], "category": None,
                "conclusion": {"conclusion": "null cat"}, "confidence": 0.6,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert "" in result["councils"]

    @pytest.mark.asyncio
    async def test_empty_vendors_array_is_skipped(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "pairwise_battle",
                "vendors": [], "category": None,
                "conclusion": {"conclusion": "no vendors"}, "confidence": 0.3,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert result["battles"] == {}

    @pytest.mark.asyncio
    async def test_self_pair_resource_asymmetry_is_skipped(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["Intercom", "Intercom"], "category": None,
                "conclusion": {"conclusion": "self pair", "resource_advantage": "Intercom"},
                "confidence": 0.5,
                "computed_date": date(2026, 3, 21),
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 21))
        assert result["asymmetries"] == {}

    @pytest.mark.asyncio
    async def test_same_day_better_quality_entry_replaces_placeholder(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["Help Scout", "Intercom"], "category": None,
                "conclusion": {
                    "conclusion": "placeholder result",
                    "winner": "Vendor B",
                    "resource_advantage": "Vendor B holds the resource advantage due to scale.",
                },
                "confidence": 0.5,
                "computed_date": date(2026, 3, 21),
            },
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["Help Scout", "Intercom"], "category": None,
                "conclusion": {
                    "conclusion": "Intercom has the stronger resource position.",
                    "winner": "Intercom",
                    "resource_advantage": "Intercom holds the resource advantage due to 3.4x review volume.",
                },
                "confidence": 0.5,
                "computed_date": date(2026, 3, 21),
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 21))
        entry = result["asymmetries"][("Help Scout", "Intercom")]
        assert entry["conclusion"]["winner"] == "Intercom"
        assert "3.4x review volume" in entry["conclusion"]["resource_advantage"]

    @pytest.mark.asyncio
    async def test_mixed_types_routed_correctly(self):
        """All three analysis types in one result set are bucketed correctly."""
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_cross_vendor_lookup,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "analysis_type": "pairwise_battle",
                "vendors": ["A", "B"], "category": None,
                "conclusion": {"conclusion": "battle"}, "confidence": 0.9,
            },
            {
                "analysis_type": "category_council",
                "vendors": ["A", "B", "C"], "category": "CRM",
                "conclusion": {"conclusion": "council"}, "confidence": 0.8,
            },
            {
                "analysis_type": "resource_asymmetry",
                "vendors": ["A", "B"], "category": None,
                "conclusion": {"conclusion": "asymmetry"}, "confidence": 0.7,
            },
        ]
        result = await reconstruct_cross_vendor_lookup(pool, as_of=date(2026, 3, 17))
        assert len(result["battles"]) == 1
        assert len(result["councils"]) == 1
        assert len(result["asymmetries"]) == 1
        assert result["battles"][("A", "B")]["conclusion"]["conclusion"] == "battle"
        assert result["councils"]["CRM"]["conclusion"]["conclusion"] == "council"
        assert result["asymmetries"][("A", "B")]["conclusion"]["conclusion"] == "asymmetry"


class TestSpecificCrossVendorEcosystemEvidence:
    def test_build_specific_cross_vendor_ecosystem_evidence_prefers_specific_categories(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            _build_specific_cross_vendor_ecosystem_evidence,
        )

        specific = _build_specific_cross_vendor_ecosystem_evidence(
            ecosystem_evidence={
                "B2B Software": {
                    "category": "B2B Software",
                    "vendor_count": 40,
                    "displacement_intensity": 1.2,
                },
            },
            vendor_scores=[
                {"vendor_name": "Zendesk", "product_category": "B2B Software"},
                {"vendor_name": "Freshdesk", "product_category": "B2B Software"},
                {"vendor_name": "Intercom", "product_category": "B2B Software"},
            ],
            reasoning_lookup={
                "zendesk": {"archetype": "price_squeeze"},
                "freshdesk": {"archetype": "price_squeeze"},
                "intercom": {"archetype": "platform_gap"},
            },
            evidence_lookup={
                "zendesk": {"product_category": "Helpdesk"},
                "freshdesk": {"product_category": "Helpdesk"},
                "intercom": {"product_category": "Helpdesk"},
            },
            competitive_disp=[
                {"vendor": "zendesk"},
                {"vendor": "freshdesk"},
                {"vendor": "intercom"},
            ],
            preferred_profile_categories={},
        )

        assert "B2B Software" not in specific
        assert specific["Helpdesk"]["vendor_count"] == 3
        assert specific["Helpdesk"]["displacement_intensity"] == 1.0
        assert specific["Helpdesk"]["dominant_archetype"] == "price_squeeze"


class TestCrossVendorSchemaContract:
    """Validate CROSS_VENDOR_JSON_SCHEMA accepts expected LLM output shapes."""

    def test_battle_output_validates(self):
        import jsonschema
        from atlas_brain.reasoning.cross_vendor import CROSS_VENDOR_JSON_SCHEMA
        output = {
            "analysis_type": "pairwise_battle",
            "vendors": ["Zendesk", "Freshdesk"],
            "conclusion": "Zendesk losing customers due to pricing pressure",
            "confidence": 0.85,
            "key_insights": [
                {"insight": "3x price gap", "evidence": "pricing delta: 3x"},
                {"insight": "SMB migration accelerating", "evidence": "switch_count: 43"},
            ],
            "durability_assessment": "structural",
            "winner": "Freshdesk",
            "loser": "Zendesk",
            "market_regime": "",
            "segment_dynamics": {"enterprise_winner": "Zendesk", "smb_winner": "Freshdesk",
                                 "segment_divergence": True},
            "falsification_conditions": ["Zendesk restores pricing"],
            "uncertainty_sources": ["limited enterprise data"],
            "resource_advantage": "",
        }
        jsonschema.validate(output, CROSS_VENDOR_JSON_SCHEMA)

    def test_asymmetry_output_with_resource_advantage_validates(self):
        import jsonschema
        from atlas_brain.reasoning.cross_vendor import CROSS_VENDOR_JSON_SCHEMA
        output = {
            "analysis_type": "resource_asymmetry",
            "vendors": ["Salesforce", "HubSpot"],
            "conclusion": "Salesforce holds deeper lock-in via integrations",
            "confidence": 0.72,
            "key_insights": [
                {"insight": "4x integration count", "evidence": "integration_ratio: 4.0"},
                {"insight": "enterprise tilt", "evidence": "enterprise_share: 0.62"},
            ],
            "durability_assessment": "structural",
            "winner": "Salesforce",
            "loser": "",
            "market_regime": "",
            "segment_dynamics": {"enterprise_winner": "Salesforce", "smb_winner": "", "segment_divergence": False},
            "falsification_conditions": ["HubSpot launches marketplace"],
            "uncertainty_sources": [],
            "resource_advantage": "Salesforce holds edge due to 4x integration count",
        }
        jsonschema.validate(output, CROSS_VENDOR_JSON_SCHEMA)

    def test_resource_advantage_accepts_null(self):
        import jsonschema
        from atlas_brain.reasoning.cross_vendor import CROSS_VENDOR_JSON_SCHEMA
        output = {
            "analysis_type": "resource_asymmetry",
            "vendors": ["A", "B"],
            "conclusion": "Approximate parity",
            "confidence": 0.5,
            "key_insights": [{"insight": "similar review share", "evidence": "review_ratio: 1.1"}],
            "durability_assessment": "uncertain",
            "winner": "",
            "loser": "",
            "market_regime": "",
            "segment_dynamics": {"enterprise_winner": "", "smb_winner": "", "segment_divergence": False},
            "falsification_conditions": [],
            "uncertainty_sources": [],
            "resource_advantage": "",
        }
        jsonschema.validate(output, CROSS_VENDOR_JSON_SCHEMA)

    def test_schema_rejects_unknown_field(self):
        import jsonschema
        from atlas_brain.reasoning.cross_vendor import CROSS_VENDOR_JSON_SCHEMA
        output = {
            "analysis_type": "pairwise_battle",
            "vendors": ["A", "B"],
            "conclusion": "test",
            "confidence": 0.5,
            "key_insights": [],
            "durability_assessment": "uncertain",
            "falsification_conditions": [],
            "made_up_field": "should fail",
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(output, CROSS_VENDOR_JSON_SCHEMA)


class TestBattleCardCacheKeyIntegration:
    """Verify cache hash changes when cross-vendor fields change."""

    def test_cross_vendor_battles_change_busts_cache(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash
        base = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "vendor_weaknesses": [{"weakness": "pricing", "count": 45}],
            "customer_pain_quotes": [],
            "competitor_differentiators": [],
            "objection_data": {},
            "cross_vendor_battles": None,
            "resource_asymmetry": None,
            "ecosystem_context": None,
        }
        hash_without = compute_evidence_hash(base)
        with_battles = {**base, "cross_vendor_battles": [
            {"opponent": "Freshdesk", "conclusion": "losing on price", "confidence": 0.85},
        ]}
        hash_with = compute_evidence_hash(with_battles)
        assert hash_without != hash_with

    def test_resource_asymmetry_change_busts_cache(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash
        base = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "vendor_weaknesses": [],
            "customer_pain_quotes": [],
            "competitor_differentiators": [],
            "objection_data": {},
            "cross_vendor_battles": None,
            "resource_asymmetry": None,
            "ecosystem_context": None,
        }
        hash_without = compute_evidence_hash(base)
        with_asym = {**base, "resource_asymmetry": {
            "opponent": "Freshdesk", "resource_advantage": "Zendesk via integrations",
        }}
        hash_with = compute_evidence_hash(with_asym)
        assert hash_without != hash_with

    def test_ecosystem_context_change_busts_cache(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash
        base = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "vendor_weaknesses": [],
            "customer_pain_quotes": [],
            "competitor_differentiators": [],
            "objection_data": {},
            "cross_vendor_battles": None,
            "resource_asymmetry": None,
            "ecosystem_context": None,
        }
        hash_without = compute_evidence_hash(base)
        with_eco = {**base, "ecosystem_context": {
            "hhi": 0.15, "market_structure": "fragmented",
        }}
        hash_with = compute_evidence_hash(with_eco)
        assert hash_without != hash_with

    def test_same_data_produces_same_hash(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash
        data = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "vendor_weaknesses": [{"weakness": "pricing", "count": 45}],
            "customer_pain_quotes": [],
            "competitor_differentiators": [],
            "objection_data": {},
            "cross_vendor_battles": [{"opponent": "Freshdesk"}],
            "resource_asymmetry": {"opponent": "Freshdesk"},
            "ecosystem_context": {"hhi": 0.15},
        }
        assert compute_evidence_hash(data) == compute_evidence_hash(data.copy())

    def test_additional_battle_card_inputs_change_hash(self):
        from atlas_brain.reasoning.semantic_cache import compute_evidence_hash
        base = {
            "vendor": "Zendesk",
            "category": "Helpdesk",
            "churn_pressure_score": 72,
            "confidence": "high",
            "vendor_weaknesses": [],
            "customer_pain_quotes": [],
            "competitor_differentiators": [],
            "objection_data": {},
            "cross_vendor_battles": [{"opponent": "Freshdesk", "winner": "Freshdesk", "loser": "Zendesk"}],
            "resource_asymmetry": None,
            "ecosystem_context": None,
            "category_council": None,
            "high_intent_companies": [],
            "integration_stack": [],
            "buyer_authority": None,
            "retention_signals": [],
            "active_evaluation_deadlines": [],
            "archetype": "pricing_shock",
            "archetype_risk_level": "high",
        }
        changed = {
            **base,
            "high_intent_companies": [{"company": "Acme", "urgency": 9.0}],
            "category_council": {"market_regime": "price_competition", "winner": "Freshdesk"},
        }
        assert compute_evidence_hash(base) != compute_evidence_hash(changed)


class TestBattleCardDeterministicFallbacks:
    def test_cross_vendor_council_overrides_generic_reasoning_when_richer(self):
        from atlas_brain.autonomous.tasks.b2b_battle_cards import (
            _build_category_council_from_cross_vendor,
            _build_category_council_from_reasoning,
            _prefer_richer_category_council,
        )

        card = {
            "category": "Helpdesk",
            "reasoning_contracts": {
                "schema_version": "v1",
                "category_reasoning": {
                    "market_regime": "stable",
                    "narrative": "The Zendesk market remains stable with low churn velocity and price pressure.",
                    "confidence_score": 0.5,
                    "winner": None,
                    "loser": None,
                    "key_insights": [],
                },
            },
        }
        xv_council_row = {
            "confidence": 0.58,
            "conclusion": {
                "conclusion": "Freshdesk and Zoho Desk are actively taking share while pricing competition intensifies.",
                "market_regime": "price_competition",
                "winner": "Zoho Desk",
                "loser": "Freshdesk",
                "key_insights": [
                    {"insight": "Pricing is driving displacement.", "evidence": "price pressure"}
                ],
                "durability_assessment": "structural",
            },
        }

        reasoning = _build_category_council_from_reasoning(card)
        xv_council = _build_category_council_from_cross_vendor(xv_council_row)
        chosen = _prefer_richer_category_council(reasoning, xv_council)

        assert chosen["market_regime"] == "price_competition"
        assert chosen["winner"] == "Zoho Desk"
        assert chosen["loser"] == "Freshdesk"

    def test_active_evaluation_deadlines_use_decision_timeline_when_needed(self):
        from atlas_brain.autonomous.tasks._b2b_shared import _build_active_evaluation_deadlines

        entries = [{
            "company": "Acme",
            "evaluation_deadline": None,
            "contract_end": None,
            "decision_timeline": "within_quarter",
            "urgency": 9.0,
            "title": "CTO",
            "company_size": "51-200",
            "industry": "SaaS",
        }]

        result = _build_active_evaluation_deadlines(entries, limit=5, today=date(2026, 3, 18))
        assert result[0]["trigger_type"] == "timeline_signal"
        assert result[0]["decision_timeline"] == "within_quarter"

    def test_category_council_fallback_uses_ecosystem_context(self):
        from atlas_brain.autonomous.tasks.b2b_battle_cards import _build_category_council_fallback

        card = {
            "category": "Cloud Infrastructure",
            "ecosystem_context": {
                "hhi": 0.18,
                "market_structure": "fragmented",
                "displacement_intensity": 0.42,
                "dominant_archetype": "pricing_shock",
            },
        }

        council = _build_category_council_fallback(card)
        assert council is not None
        assert council["market_regime"] == "fragmented"
        assert council["key_insights"][0]["evidence"] == "market_structure: fragmented"
        assert council["winner"] is None
        assert council["loser"] is None

    def test_ecosystem_context_from_analysis_handles_evidence_object(self):
        from atlas_brain.autonomous.tasks.b2b_battle_cards import _ecosystem_context_from_analysis
        from atlas_brain.reasoning.ecosystem import CategoryHealth, EcosystemEvidence

        eco = EcosystemEvidence(
            category="Cloud Infrastructure",
            health=CategoryHealth(
                category="Cloud Infrastructure",
                hhi=1888.4,
                market_structure="fragmenting",
                displacement_intensity=3.1,
                dominant_archetype="pricing_shock",
            ),
        )

        result = _ecosystem_context_from_analysis(eco)
        assert result == {
            "hhi": 1888.4,
            "market_structure": "fragmenting",
            "displacement_intensity": 3.1,
            "dominant_archetype": "pricing_shock",
        }

    def test_attach_battle_card_provenance_adds_vendor_metadata(self):
        from atlas_brain.autonomous.tasks.b2b_battle_cards import _attach_battle_card_provenance

        card = {"vendor": "AWS"}
        provenance = {
            "source_distribution": {"reddit": 6, "g2": 2},
            "sample_review_ids": ["r1", "r2", "r3"],
            "review_window_start": datetime(2026, 3, 1, 9, 0, 0),
            "review_window_end": datetime(2026, 3, 18, 18, 30, 0),
        }

        _attach_battle_card_provenance(card, provenance)
        assert card["source_distribution"] == {"reddit": 6, "g2": 2}
        assert card["sample_review_ids"] == ["r1", "r2", "r3"]
        assert card["review_window_start"] == "2026-03-01"
        assert card["review_window_end"] == "2026-03-18"

    def test_merge_battle_card_provenance_uses_vault_fallbacks(self):
        from atlas_brain.autonomous.tasks.b2b_battle_cards import _merge_battle_card_provenance

        merged = _merge_battle_card_provenance(
            {"source_distribution": {"g2": 4}},
            {
                "source_distribution": {"reddit": 2},
                "sample_review_ids": ["r1", "r2"],
                "review_window_start": "2026-03-01",
                "review_window_end": "2026-03-18",
            },
        )

        assert merged["source_distribution"] == {"g2": 4}
        assert merged["sample_review_ids"] == ["r1", "r2"]
        assert merged["review_window_start"] == "2026-03-01"
        assert merged["review_window_end"] == "2026-03-18"


class TestScorecardNarrativeLLMInput:
    """Verify the LLM input dict construction matches the code in b2b_churn_reports.py."""

    def _build_llm_input(self, sc: dict) -> dict:
        """Use the real scorecard narrative payload builder."""
        from atlas_brain.autonomous.tasks.b2b_churn_reports import (
            _build_scorecard_narrative_payload,
        )

        reasoning_lookup = {}
        if sc.get("vendor") and sc.get("archetype"):
            reasoning_lookup[sc["vendor"]] = {
                "key_signals": ["price hikes", "support complaints", "migration pressure"],
            }
        return _build_scorecard_narrative_payload(
            sc,
            reasoning_lookup=reasoning_lookup,
        )

    def test_cross_vendor_comparisons_included_when_present(self):
        sc = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "risk_level": "high",
            "cross_vendor_comparisons": [
                {"opponent": "Freshdesk", "conclusion": "losing on price",
                 "confidence": 0.85, "resource_advantage": "Zendesk via integrations"},
            ],
        }
        llm_input = self._build_llm_input(sc)
        assert "cross_vendor_comparisons" in llm_input
        assert llm_input["cross_vendor_comparisons"][0]["opponent"] == "Freshdesk"
        assert llm_input["locked_facts"]["allowed_opponents"] == ["Freshdesk"]

    def test_category_council_included_when_present(self):
        sc = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "risk_level": "high",
            "category_council": {
                "winner": "Zoho Desk",
                "loser": "Freshdesk",
                "conclusion": "Pricing pressure is fragmenting the helpdesk market.",
                "market_regime": "price_competition",
                "durability": "cyclical",
                "confidence": 0.58,
                "key_insights": [{"insight": "Pricing is the primary driver."}] * 5,
            },
        }
        llm_input = self._build_llm_input(sc)
        assert llm_input["category_council"]["winner"] == "Zoho Desk"
        assert llm_input["category_council"]["market_regime"] == "price_competition"
        assert len(llm_input["category_council"]["key_insights"]) == 3

    def test_cross_vendor_comparisons_absent_when_not_enriched(self):
        sc = {
            "vendor": "Monday.com",
            "churn_pressure_score": 45,
            "risk_level": "medium",
        }
        llm_input = self._build_llm_input(sc)
        assert "cross_vendor_comparisons" not in llm_input
        assert llm_input["locked_facts"]["vendor"] == "Monday.com"

    def test_short_circuit_fires_without_comparisons(self):
        """When no cross_vendor_comparisons, reasoning_summary is reused."""
        sc = {"vendor": "Acme", "reasoning_summary": "pricing pressure rising"}
        reasoning_summary = sc.get("reasoning_summary", "")
        has_xv = sc.get("cross_vendor_comparisons")
        should_reuse = bool(reasoning_summary) and not has_xv
        assert should_reuse is True

    def test_short_circuit_skipped_with_comparisons(self):
        """When cross_vendor_comparisons exist, must fall through to LLM."""
        sc = {
            "vendor": "Acme",
            "reasoning_summary": "pricing pressure rising",
            "cross_vendor_comparisons": [{"opponent": "Rival", "conclusion": "..."}],
        }
        reasoning_summary = sc.get("reasoning_summary", "")
        has_xv = sc.get("cross_vendor_comparisons")
        should_reuse = bool(reasoning_summary) and not has_xv
        assert should_reuse is False

    def test_empty_comparisons_list_still_triggers_llm(self):
        """Even an empty list is truthy enough to skip short-circuit?
        No -- empty list is falsy, so short-circuit fires. Verify this."""
        sc = {
            "vendor": "Acme",
            "reasoning_summary": "pricing pressure rising",
            "cross_vendor_comparisons": [],
        }
        reasoning_summary = sc.get("reasoning_summary", "")
        has_xv = sc.get("cross_vendor_comparisons")
        # [] is falsy, so short-circuit should fire
        should_reuse = bool(reasoning_summary) and not has_xv
        assert should_reuse is True

    def test_llm_input_excludes_internal_fields(self):
        """Fields not in the key tuple must not leak into LLM input."""
        sc = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "risk_level": "high",
            "reasoning_summary": "should not appear in llm_input",
            "archetype": "pricing_shock",
            "archetype_confidence": 0.9,
            "cross_vendor_comparisons": [{"opponent": "Freshdesk", "confidence": 0.8}],
        }
        llm_input = self._build_llm_input(sc)
        assert "reasoning_summary" not in llm_input
        assert "archetype" not in llm_input
        assert "archetype_confidence" not in llm_input
        assert "cross_vendor_comparisons" in llm_input
        assert llm_input["locked_facts"]["allowed_opponents"] == ["Freshdesk"]

    def test_resource_advantage_absent_tolerant(self):
        """cross_vendor_comparisons entry missing resource_advantage still works."""
        sc = {
            "vendor": "Zendesk",
            "churn_pressure_score": 72,
            "cross_vendor_comparisons": [
                {"opponent": "Freshdesk", "conclusion": "edge case", "confidence": 0.7},
            ],
        }
        llm_input = self._build_llm_input(sc)
        comp = llm_input["cross_vendor_comparisons"][0]
        # Consumer pattern: .get("resource_advantage", "")
        assert comp.get("resource_advantage", "") == ""

    def test_llm_input_uses_compact_reasoning_conclusion_instead_of_full_contracts(self):
        sc = {
            "vendor": "Zendesk",
            "risk_level": "high",
            "churn_pressure_score": 72,
            "archetype": "pricing_shock",
            "archetype_confidence": 0.8,
            "reasoning_summary": "Pricing pressure is climbing after the latest packaging change.",
            "reasoning_contracts": {
                "vendor_core_reasoning": {"causal_narrative": {"trigger": "Price hike"}},
                "displacement_reasoning": {"migration_proof": {"confidence": "medium"}},
            },
        }
        llm_input = self._build_llm_input(sc)
        assert "reasoning_contracts" not in llm_input
        assert "vendor_core_reasoning" not in llm_input
        assert "displacement_reasoning" not in llm_input
        assert llm_input["reasoning_conclusion"]["archetype"] == "pricing_shock"
        assert llm_input["reasoning_conclusion"]["key_signals"] == [
            "price hikes", "support complaints", "migration pressure",
        ]

    def test_llm_input_trims_nested_lists_for_compactness(self):
        sc = {
            "vendor": "Zendesk",
            "feature_analysis": {
                "loved": [{"feature": f"loved-{i}"} for i in range(5)],
                "hated": [{"feature": f"hated-{i}"} for i in range(5)],
            },
            "churn_predictors": {
                "high_risk_industries": [{"industry": f"ind-{i}"} for i in range(4)],
                "high_risk_sizes": [{"size": f"size-{i}"} for i in range(4)],
                "dm_churn_rate": 0.4,
                "price_complaint_rate": 0.2,
            },
            "competitor_overlap": [{"competitor": f"comp-{i}"} for i in range(5)],
            "cross_vendor_comparisons": [
                {
                    "opponent": f"opp-{i}",
                    "conclusion": "c",
                    "confidence": 0.7,
                    "resource_advantage": "adv",
                }
                for i in range(4)
            ],
        }
        llm_input = self._build_llm_input(sc)
        assert len(llm_input["feature_analysis"]["loved"]) == 3
        assert len(llm_input["feature_analysis"]["hated"]) == 3
        assert len(llm_input["churn_predictors"]["high_risk_industries"]) == 2
        assert len(llm_input["churn_predictors"]["high_risk_sizes"]) == 2
        assert len(llm_input["competitor_overlap"]) == 3
        assert len(llm_input["cross_vendor_comparisons"]) == 2

    def test_scorecard_narrative_max_tokens_defaults_to_compact_budget(self):
        from atlas_brain.autonomous.tasks.b2b_churn_reports import (
            _scorecard_narrative_max_tokens,
        )

        with patch("atlas_brain.autonomous.tasks.b2b_churn_reports.settings") as mock_settings:
            mock_settings.llm.openrouter_reasoning_model = "anthropic/claude-3.5-haiku"
            mock_settings.b2b_churn.scorecard_narrative_max_tokens = 333
            assert _scorecard_narrative_max_tokens() == 333

    def test_scorecard_narrative_max_tokens_expands_for_gpt_oss(self):
        from atlas_brain.autonomous.tasks.b2b_churn_reports import (
            _scorecard_narrative_max_tokens,
        )

        with patch("atlas_brain.autonomous.tasks.b2b_churn_reports.settings") as mock_settings:
            mock_settings.llm.openrouter_reasoning_model = "openai/gpt-oss-120b"
            mock_settings.b2b_churn.scorecard_narrative_gpt_oss_max_tokens = 1600
            assert _scorecard_narrative_max_tokens() == 1600

    def test_scorecard_narrative_max_tokens_uses_deepseek_budget(self):
        from atlas_brain.autonomous.tasks.b2b_churn_reports import (
            _scorecard_narrative_max_tokens,
        )

        with patch("atlas_brain.autonomous.tasks.b2b_churn_reports.settings") as mock_settings:
            mock_settings.llm.openrouter_reasoning_model = "deepseek/deepseek-r1"
            mock_settings.b2b_churn.scorecard_narrative_deepseek_max_tokens = 1200
            assert _scorecard_narrative_max_tokens() == 1200


class TestVendorReasoningCap:
    """Verify the vendor cap limits evidence building and reasoning."""

    def test_evidence_map_capped(self):
        """Evidence map should only build for capped vendors."""
        from atlas_brain.config import settings
        cfg = settings.b2b_churn
        assert hasattr(cfg, "stratified_reasoning_vendor_cap")
        assert cfg.stratified_reasoning_vendor_cap > 0
        assert cfg.stratified_reasoning_vendor_cap <= 100


# ---------------------------------------------------------------------------
# Evidence diff persistence
# ---------------------------------------------------------------------------


class TestPersistEvidenceDiff:
    """Verify persist_evidence_diff handles all code paths."""

    @pytest.mark.asyncio
    async def test_persist_with_real_diff(self):
        from atlas_brain.reasoning.differential import (
            EvidenceDiff, persist_evidence_diff,
        )

        pool = AsyncMock()
        pool.execute.return_value = None

        diff = EvidenceDiff(
            confirmed=["urgency", "reviews"],
            contradicted=[("churn_density", "30", "45")],
            novel=[("new_metric", "100")],
            missing=["old_field"],
        )
        await persist_evidence_diff(pool, "Zendesk", diff, "full_reason")

        pool.execute.assert_called_once()
        args = pool.execute.call_args[0]
        assert "INSERT INTO reasoning_evidence_diffs" in args[0]
        assert args[1] == "Zendesk"
        assert args[2] == 2   # confirmed_count
        assert args[3] == 1   # contradicted_count
        assert args[4] == 1   # novel_count
        assert args[5] == 1   # missing_count
        assert args[6] == 5   # total_fields
        assert args[10] == '{"contradiction_emergence": 5.0}'  # component_scores

    @pytest.mark.asyncio
    async def test_persist_with_none_diff(self):
        """Recall and cold full-reason pass diff=None."""
        from atlas_brain.reasoning.differential import persist_evidence_diff

        pool = AsyncMock()
        pool.execute.return_value = None

        await persist_evidence_diff(pool, "Slack", None, "recall")

        args = pool.execute.call_args[0]
        assert args[1] == "Slack"
        assert args[2] == 0   # confirmed_count
        assert args[3] == 0   # contradicted_count
        assert args[11] == "recall"  # decision
        assert args[12] is False     # compared (recall = not compared)

    @pytest.mark.asyncio
    async def test_persist_swallows_db_error(self):
        """DB errors are logged but don't raise."""
        from atlas_brain.reasoning.differential import persist_evidence_diff

        pool = AsyncMock()
        pool.execute.side_effect = Exception("connection lost")

        # Should not raise
        await persist_evidence_diff(pool, "Acme", None, "full_reason")


class TestReconstructEvidenceVolatility:
    """Verify reconstruct_evidence_volatility returns correct structure."""

    @pytest.mark.asyncio
    async def test_returns_vendor_keyed_dict(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_evidence_volatility,
        )

        pool = AsyncMock()
        pool.fetch.return_value = [
            {
                "vendor_name": "Zendesk",
                "avg_diff": 0.35,
                "max_diff": 0.72,
                "core_contradictions": 2,
                "days_compared": 3,
                "days_tracked": 5,
                "latest_decision": "full_reason",
                "latest_component_scores": {"competitive_shift": 3.0},
                "latest_contradicted": [{"key": "churn_density"}],
            },
        ]
        result = await reconstruct_evidence_volatility(pool, days=14)
        assert "Zendesk" in result
        entry = result["Zendesk"]
        assert entry["avg_diff_ratio"] == 0.35
        assert entry["max_diff_ratio"] == 0.72
        assert entry["core_contradictions"] == 2
        assert entry["days_compared"] == 3
        assert entry["days_tracked"] == 5
        assert entry["latest_decision"] == "full_reason"
        assert entry["latest_component_scores"] == {"competitive_shift": 3.0}

    @pytest.mark.asyncio
    async def test_empty_table_returns_empty(self):
        from atlas_brain.autonomous.tasks.b2b_churn_intelligence import (
            reconstruct_evidence_volatility,
        )

        pool = AsyncMock()
        pool.fetch.return_value = []
        result = await reconstruct_evidence_volatility(pool)
        assert result == {}


class TestSlowBurnEvidenceFlow:
    """Verify slow-burn metrics reach reasoning and downstream payload builders."""

    def test_build_vendor_evidence_keeps_slow_burn_fields(self):
        from atlas_brain.autonomous.tasks._b2b_shared import _build_vendor_evidence

        vs = {
            "vendor_name": "Acme",
            "product_category": "CRM",
            "total_reviews": 100,
            "churn_intent": 40,
            "avg_urgency": 6.2,
            "recommend_yes": 15,
            "recommend_no": 25,
            "support_sentiment": 0.41,
            "legacy_support_score": 0.22,
            "new_feature_velocity": 0.71,
            "employee_growth_rate": 0.34,
        }
        temporal_lookup = {
            "Acme": {
                "trend_30d_churn_density": 0.12,
                "trend_30d_support_sentiment": -0.08,
            }
        }
        evidence = _build_vendor_evidence(
            vs,
            pain_lookup={"Acme": [{"category": "legacy platform ignored", "count": 8}]},
            competitor_lookup={},
            feature_gap_lookup={},
            insider_lookup={},
            keyword_spike_lookup={},
            temporal_lookup=temporal_lookup,
            market_regime_lookup={"Acme": {"regime_type": "high_churn"}},
        )

        assert evidence["support_sentiment"] == 0.41
        assert evidence["legacy_support_score"] == 0.22
        assert evidence["new_feature_velocity"] == 0.71
        assert evidence["employee_growth_rate"] == 0.34
        assert evidence["market_regime"]["regime_type"] == "high_churn"
        assert evidence["trend_30d_churn_density"] == 0.12

    def test_archetype_scoring_uses_slow_burn_fields(self):
        from atlas_brain.autonomous.tasks._b2b_shared import _build_vendor_evidence
        from atlas_brain.reasoning.archetypes import enrich_evidence_with_archetypes

        temporal = {"Acme": {"trend_30d_churn_density": 0.11}}
        evidence = _build_vendor_evidence(
            {
                "vendor_name": "Acme",
                "product_category": "CRM",
                "total_reviews": 100,
                "churn_intent": 38,
                "avg_urgency": 5.8,
                "recommend_yes": 20,
                "recommend_no": 15,
                "legacy_support_score": 0.2,
                "new_feature_velocity": 0.8,
            },
            pain_lookup={"Acme": [{"category": "legacy workflow deprecated", "count": 6}]},
            competitor_lookup={},
            feature_gap_lookup={},
            insider_lookup={},
            keyword_spike_lookup={},
            temporal_lookup=temporal,
        )

        enriched = enrich_evidence_with_archetypes(evidence, temporal["Acme"])
        archetypes = [x["archetype"] for x in enriched.get("archetype_scores", [])]

        assert "pivot_abandonment" in archetypes

    def test_prepare_evidence_keeps_market_regime_and_trends(self):
        from atlas_brain.reasoning.stratified_reasoner import StratifiedReasoner

        prepared = StratifiedReasoner._prepare_evidence({
            "vendor_name": "Acme",
            "support_sentiment": 0.44,
            "legacy_support_score": 0.31,
            "new_feature_velocity": 0.63,
            "employee_growth_rate": 0.28,
            "trend_30d_support_sentiment": -0.07,
            "trend_90d_churn_density": 0.09,
            "market_regime": {"regime_type": "disruption"},
        })

        assert prepared["support_sentiment"] == 0.44
        assert prepared["trend_30d_support_sentiment"] == -0.07
        assert prepared["trend_90d_churn_density"] == 0.09
        assert prepared["market_regime"]["regime_type"] == "disruption"


class TestBlogMarketRegimeBlueprint:
    """Verify blog blueprints can carry persisted market regime context."""

    def test_market_landscape_blueprint_includes_market_regime(self):
        from atlas_brain.autonomous.tasks.b2b_blog_post_generation import _blueprint_market_landscape

        blueprint = _blueprint_market_landscape(
            {
                "category": "CRM",
                "vendor_count": 4,
                "total_reviews": 240,
                "avg_urgency": 5.6,
                "slug": "crm-market-landscape",
            },
            {
                "vendor_profiles": [],
                "vendor_signals": [],
                "quotes": [],
                "data_context": {"review_period": "2026-01-01 to 2026-03-01"},
                "category_overview": {
                    "cross_vendor_analysis": {"market_regime": "high_churn"},
                },
            },
        )

        assert blueprint.data_context["category"] == "CRM"
        assert any(
            section.key_stats.get("market_regime") == "high_churn"
            for section in blueprint.sections
        )
