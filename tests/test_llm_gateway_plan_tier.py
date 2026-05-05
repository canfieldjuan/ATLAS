"""Tests for the LLM Gateway plan tier (PR-D2).

Pins the contract that ``llm_gateway`` is a first-class product with
its own plan ordering, rate limits, and feature limits -- without
touching the existing consumer / B2B / vendor plan tables.
"""

from __future__ import annotations

import pytest


# ---- Plan ordering -------------------------------------------------------


def test_llm_gateway_plan_order_present():
    from atlas_brain.auth.dependencies import LLM_GATEWAY_PLAN_ORDER

    assert LLM_GATEWAY_PLAN_ORDER == [
        "llm_trial",
        "llm_starter",
        "llm_growth",
        "llm_pro",
    ]


def test_llm_gateway_plan_order_distinct_from_consumer():
    """A consumer-tier user must not satisfy a require_llm_plan check
    -- the namespaces must not overlap."""
    from atlas_brain.auth.dependencies import (
        LLM_GATEWAY_PLAN_ORDER,
        PLAN_ORDER,
        B2B_PLAN_ORDER,
    )

    assert set(LLM_GATEWAY_PLAN_ORDER).isdisjoint(set(PLAN_ORDER))
    assert set(LLM_GATEWAY_PLAN_ORDER).isdisjoint(set(B2B_PLAN_ORDER))


# ---- require_llm_plan -----------------------------------------------------


def test_require_llm_plan_rejects_unknown_tier():
    from atlas_brain.auth.dependencies import require_llm_plan

    with pytest.raises(ValueError, match="Invalid LLM Gateway plan tier 'starter'"):
        require_llm_plan("starter")


def test_require_llm_plan_rejects_b2b_tier():
    from atlas_brain.auth.dependencies import require_llm_plan

    with pytest.raises(ValueError, match="Invalid LLM Gateway plan tier 'b2b_pro'"):
        require_llm_plan("b2b_pro")


def test_require_llm_plan_accepts_known_tiers():
    from atlas_brain.auth.dependencies import (
        LLM_GATEWAY_PLAN_ORDER,
        require_llm_plan,
    )

    for plan in LLM_GATEWAY_PLAN_ORDER:
        dep = require_llm_plan(plan)
        assert callable(dep)


# ---- VALID_PRODUCTS -------------------------------------------------------


def test_valid_products_includes_llm_gateway():
    from atlas_brain.api.auth import VALID_PRODUCTS

    assert "llm_gateway" in VALID_PRODUCTS


def test_valid_products_does_not_drop_existing_tiers():
    """Adding llm_gateway must not regress the existing product set."""
    from atlas_brain.api.auth import VALID_PRODUCTS

    for existing in ("consumer", "b2b_retention", "b2b_challenger"):
        assert existing in VALID_PRODUCTS


# ---- PLAN_RATE_LIMITS -----------------------------------------------------


def test_plan_rate_limits_covers_llm_tiers():
    from atlas_brain.auth.rate_limit import PLAN_RATE_LIMITS

    assert PLAN_RATE_LIMITS["llm_trial"] == "100/hour"
    assert PLAN_RATE_LIMITS["llm_starter"] == "1000/hour"
    assert PLAN_RATE_LIMITS["llm_growth"] == "10000/hour"
    assert PLAN_RATE_LIMITS["llm_pro"] == "100000/hour"


def test_plan_rate_limits_lookup_via_dynamic_limit():
    """``_dynamic_limit`` is the slowapi callable that maps the
    composite rate-limit key back to a rate string. It must resolve
    LLM-Gateway plans to the right rate."""
    from atlas_brain.auth.rate_limit import _dynamic_limit

    assert _dynamic_limit("llm_starter|account-uuid") == "1000/hour"
    assert _dynamic_limit("llm_pro|account-uuid") == "100000/hour"


def test_plan_rate_limits_unknown_plan_falls_back():
    """Unknown plans (e.g. legacy or future tiers) get the default
    instead of raising -- prevents a config drift from producing 500s."""
    from atlas_brain.auth.rate_limit import _DEFAULT_LIMIT, _dynamic_limit

    assert _dynamic_limit("future_plan|x") == _DEFAULT_LIMIT


# ---- LLM_PLAN_LIMITS ------------------------------------------------------


def test_llm_plan_limits_shape():
    from atlas_brain.api.billing import LLM_PLAN_LIMITS

    expected_keys = {"monthly_token_limit", "cache_enabled", "batch_enabled", "byok_keys_max"}
    for plan in ("llm_trial", "llm_starter", "llm_growth", "llm_pro"):
        assert plan in LLM_PLAN_LIMITS
        assert set(LLM_PLAN_LIMITS[plan].keys()) == expected_keys


def test_llm_plan_limits_pro_is_unlimited():
    """``-1`` is the unlimited sentinel used elsewhere in PLAN_LIMITS
    (B2B Pro vendors=-1). Reusing the same convention here keeps
    enforcement code simple."""
    from atlas_brain.api.billing import LLM_PLAN_LIMITS

    assert LLM_PLAN_LIMITS["llm_pro"]["monthly_token_limit"] == -1
    assert LLM_PLAN_LIMITS["llm_pro"]["byok_keys_max"] == -1


def test_llm_plan_limits_trial_disables_batch():
    """Anthropic batch is the 50% cost-saver feature -- reserved for
    paying tiers so the trial cannot abuse it for free volume."""
    from atlas_brain.api.billing import LLM_PLAN_LIMITS

    assert LLM_PLAN_LIMITS["llm_trial"]["batch_enabled"] is False
    assert LLM_PLAN_LIMITS["llm_starter"]["batch_enabled"] is True
    assert LLM_PLAN_LIMITS["llm_growth"]["batch_enabled"] is True
    assert LLM_PLAN_LIMITS["llm_pro"]["batch_enabled"] is True


def test_llm_plan_limits_token_caps_increase_monotonically():
    """Higher-tier plans always have at-least-as-large quotas (with
    -1 = unlimited treated as max). Catches a config-edit regression."""
    from atlas_brain.api.billing import LLM_PLAN_LIMITS

    quotas = [
        LLM_PLAN_LIMITS[t]["monthly_token_limit"]
        for t in ("llm_trial", "llm_starter", "llm_growth", "llm_pro")
    ]
    # Replace the unlimited sentinel for the comparison.
    normalized = [10**18 if q == -1 else q for q in quotas]
    assert normalized == sorted(normalized)


# ---- PRICE_TO_PLAN init -------------------------------------------------


def test_init_price_map_picks_up_llm_price_ids(monkeypatch):
    """When the Stripe price IDs land in the env, ``_init_price_map``
    must surface them in ``PRICE_TO_PLAN`` so checkout webhooks
    resolve to the right plan tier."""
    monkeypatch.setenv("ATLAS_SAAS_ENABLED", "false")
    monkeypatch.setenv("ATLAS_SAAS_STRIPE_LLM_STARTER_PRICE_ID", "price_starter_test")
    monkeypatch.setenv("ATLAS_SAAS_STRIPE_LLM_GROWTH_PRICE_ID", "price_growth_test")
    monkeypatch.setenv("ATLAS_SAAS_STRIPE_LLM_PRO_PRICE_ID", "price_pro_test")

    import importlib
    import atlas_brain.config as config_mod

    importlib.reload(config_mod)

    import atlas_brain.api.billing as billing_mod

    importlib.reload(billing_mod)
    billing_mod.PRICE_TO_PLAN.clear()
    billing_mod._init_price_map()

    assert billing_mod.PRICE_TO_PLAN["price_starter_test"] == "llm_starter"
    assert billing_mod.PRICE_TO_PLAN["price_growth_test"] == "llm_growth"
    assert billing_mod.PRICE_TO_PLAN["price_pro_test"] == "llm_pro"


# ---- Codex fixes (post-review on PR-D2) ----------------------------------


def test_plan_name_to_config_key_covers_llm_tiers():
    """Codex P1 #2 fix: ``create_checkout`` resolves ``plan='llm_*'``
    via this dict. Without the mapping, customers cannot upgrade from
    llm_trial to a paid LLM tier through the normal checkout path."""
    from atlas_brain.api.billing import PLAN_NAME_TO_CONFIG_KEY

    assert PLAN_NAME_TO_CONFIG_KEY["llm_starter"] == "stripe_llm_starter_price_id"
    assert PLAN_NAME_TO_CONFIG_KEY["llm_growth"] == "stripe_llm_growth_price_id"
    assert PLAN_NAME_TO_CONFIG_KEY["llm_pro"] == "stripe_llm_pro_price_id"


def test_register_assigns_llm_trial_for_llm_gateway_product():
    """Codex P1 #1 fix: a new account with ``product=llm_gateway``
    must land on the ``llm_trial`` plan -- otherwise
    ``require_llm_plan('llm_trial')`` rejects the brand-new self-serve
    user because plan='trial' (consumer) is not in
    LLM_GATEWAY_PLAN_ORDER. We verify the assignment logic at the
    source-text level since the actual registration flow is DB-bound;
    integration tests against a live Postgres are a separate fixture."""
    import inspect

    from atlas_brain.api.auth import register

    src = inspect.getsource(register)
    assert 'is_llm_gateway = product == "llm_gateway"' in src
    assert 'plan = "llm_trial"' in src


def test_trial_expiration_check_includes_llm_trial():
    """Codex P2 fix: trial-expiration checks in both ``require_auth``
    and ``require_api_key`` must include ``llm_trial`` so an expired
    LLM trial cannot keep authenticating after ``trial_ends_at``."""
    import inspect

    from atlas_brain.auth.dependencies import require_auth, require_api_key

    auth_src = inspect.getsource(require_auth)
    api_src = inspect.getsource(require_api_key)

    assert '"llm_trial"' in auth_src, (
        "require_auth's trial-expiration check must include llm_trial"
    )
    assert '"llm_trial"' in api_src, (
        "require_api_key's trial-expiration check must include llm_trial"
    )
