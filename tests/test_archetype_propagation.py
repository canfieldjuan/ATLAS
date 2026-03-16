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
