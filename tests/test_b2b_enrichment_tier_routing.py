"""Per-tier routing decision tests.

Tier 1 = verbatim extraction + phrase_metadata. Cheap, large-pool work that
benefits from running on local vLLM.

Tier 2 = pain_categories, competitor classification, buyer_authority,
sentiment_trajectory. Nuance classification that benefits from a frontier
model. The original `enrichment_local_only` flag forced both tiers to vLLM,
which sent Tier 2 nuance work to a model not strong enough to do it well.

These tests lock the per-tier routing contract: Tier 1 honors local_only,
Tier 2 can escape it via enrichment_tier2_force_openrouter.
"""

from __future__ import annotations

from types import SimpleNamespace

from atlas_brain.autonomous.tasks.b2b_enrichment import _resolve_tier_routing


def _cfg(**overrides) -> SimpleNamespace:
    base = {
        "enrichment_local_only": False,
        "enrichment_openrouter_model": "anthropic/claude-haiku-4-5",
        "openrouter_api_key": "test-key",
        "enrichment_tier2_force_openrouter": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_both_tiers_use_openrouter_when_keys_present():
    cfg = _cfg()
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is True
    assert t2 is True


def test_local_only_routes_both_tiers_to_vllm_when_force_off():
    cfg = _cfg(enrichment_local_only=True, enrichment_tier2_force_openrouter=False)
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is False
    assert t2 is False


def test_local_only_with_tier2_force_keeps_tier1_local():
    """The whole point of the force flag: Tier 1 stays local for cost,
    Tier 2 escapes for nuance."""
    cfg = _cfg(enrichment_local_only=True, enrichment_tier2_force_openrouter=True)
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is False
    assert t2 is True


def test_tier2_force_requires_openrouter_creds():
    """Force flag is meaningless without OpenRouter API key + model."""
    cfg = _cfg(
        enrichment_local_only=True,
        enrichment_tier2_force_openrouter=True,
        openrouter_api_key="",
    )
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is False
    assert t2 is False  # no creds -> stays local even with force flag


def test_tier2_force_requires_openrouter_model():
    cfg = _cfg(
        enrichment_local_only=True,
        enrichment_tier2_force_openrouter=True,
        enrichment_openrouter_model="",
    )
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is False
    assert t2 is False


def test_local_only_override_arg_takes_precedence():
    """The single-row code path passes local_only_override; the helper
    must trust the override over cfg.enrichment_local_only."""
    cfg = _cfg(enrichment_local_only=False, enrichment_tier2_force_openrouter=False)
    t1, t2 = _resolve_tier_routing(cfg, local_only_override=True)
    assert t1 is False  # override forced local
    assert t2 is False


def test_tier2_force_holds_under_local_only_override():
    """Even when the override flips local_only on, force_openrouter still
    routes Tier 2 to OpenRouter."""
    cfg = _cfg(enrichment_local_only=False, enrichment_tier2_force_openrouter=True)
    t1, t2 = _resolve_tier_routing(cfg, local_only_override=True)
    assert t1 is False
    assert t2 is True


def test_no_openrouter_creds_routes_both_tiers_local():
    cfg = _cfg(openrouter_api_key="", enrichment_openrouter_model="")
    t1, t2 = _resolve_tier_routing(cfg)
    assert t1 is False
    assert t2 is False
