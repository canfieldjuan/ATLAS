from types import SimpleNamespace

from extracted_content_pipeline.content_ops_cache_policy import (
    ContentOpsExactCachePolicy,
    normalize_content_ops_cache_policy,
)


def test_normalize_content_ops_cache_policy_returns_canonical_values():
    assert normalize_content_ops_cache_policy(None) is None
    assert normalize_content_ops_cache_policy(" exact-cache ") == "exact"
    assert normalize_content_ops_cache_policy("exact_cache") == "exact"
    assert normalize_content_ops_cache_policy("no-store") == "no_store"
    assert normalize_content_ops_cache_policy("off") == "no_store"


def test_normalize_content_ops_cache_policy_rejects_unsupported_value():
    try:
        normalize_content_ops_cache_policy("semantic")
    except ValueError as exc:
        assert "unsupported content_ops_cache_policy" in str(exc)
    else:
        raise AssertionError("expected unsupported cache policy to raise")


def test_policy_defaults_to_no_store_when_exact_cache_disabled():
    decision = ContentOpsExactCachePolicy().decide({
        "account_id": "acct-1",
        "asset_type": "blog_post",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "exact_cache_disabled"
    assert decision.trace_metadata() == {
        "cache_mode": "no_store",
        "cache_reason": "exact_cache_disabled",
    }


def test_policy_honors_explicit_no_store_request():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "blog_post",
        "cache_policy": "no-store",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "policy_no_store"


def test_policy_requires_explicit_exact_request_when_enabled():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "blog_post",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "policy_no_store"


def test_policy_rejects_unknown_cache_policy_value():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "blog_post",
        "cache_policy": "semantic",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "unsupported_cache_policy"


def test_policy_requires_account_scope_before_exact_cache():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "asset_type": "blog_post",
        "cache_policy": "exact",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "missing_account_scope"


def test_policy_rejects_unsupported_asset_type():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "faq_markdown",
        "cache_policy": "exact",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "unsupported_asset_type"


def test_policy_blocks_support_ticket_customer_data_by_default():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "landing_page",
        "cache_policy": "exact",
        "input_provider": "atlas_support_ticket_request",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "customer_data_no_store"


def test_policy_allows_exact_cache_for_account_scoped_non_customer_asset():
    decision = ContentOpsExactCachePolicy(exact_cache_enabled=True).decide({
        "account_id": "acct-1",
        "asset_type": "landing_page",
        "cache_policy": "exact",
    })

    assert decision.mode == "exact"
    assert decision.cacheable is True
    assert decision.reason == "eligible"
    assert decision.namespace == "content_ops.landing_page"
    assert decision.account_id == "acct-1"
    assert decision.trace_metadata() == {
        "cache_mode": "exact",
        "cache_reason": "eligible",
        "cache_namespace": "content_ops.landing_page",
        "cache_account_id": "acct-1",
    }


def test_policy_can_be_configured_from_settings_namespace():
    policy = ContentOpsExactCachePolicy.from_settings(SimpleNamespace(
        exact_cache_enabled=True,
        customer_data_exact_cache_enabled=True,
        exact_cache_namespace_prefix="tenant_content_ops",
    ))

    decision = policy.decide({
        "account_id": "acct-1",
        "asset_type": "blog_post",
        "source_type": "support_ticket",
        "cache_policy": "exact",
    })

    assert decision.mode == "exact"
    assert decision.namespace == "tenant_content_ops.blog_post"


def test_policy_from_settings_parses_false_string_flags_as_disabled():
    policy = ContentOpsExactCachePolicy.from_settings(SimpleNamespace(
        exact_cache_enabled="false",
        customer_data_exact_cache_enabled="false",
        exact_cache_namespace_prefix="tenant_content_ops",
    ))

    decision = policy.decide({
        "account_id": "acct-1",
        "asset_type": "landing_page",
        "input_provider": "atlas_support_ticket_request",
        "cache_policy": "exact",
    })

    assert decision.mode == "no_store"
    assert decision.reason == "exact_cache_disabled"
