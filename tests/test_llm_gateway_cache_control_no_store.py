"""Regression tests for PR-D6f: ``Cache-Control: no-store`` bypass.

Pins two contracts:

1. ``_cache_control_disables_cache`` correctly parses the
   ``Cache-Control`` header per RFC 7234 -- ``no-store`` matches
   regardless of case, accompanying directives, or value suffix;
   non-matching tokens (``max-age``, substrings) do not.
2. The chat() function gates BOTH the lookup branch AND the store
   branch on ``not cache_disabled`` so a no-store request bypasses
   both legs of the cache.

Tests for the helper use file-text inspection to extract and
re-execute its source; tests for the gate use file-text assertions
on the chat() body. Bypasses the gateway module's full settings
stack which depends on
``ATLAS_SAAS_API_KEY_PEPPER`` / ``ATLAS_SAAS_BYOK_ENCRYPTION_KEK``.

See plans/PR-D6f-cache-control-no-store.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PATH = ROOT / "atlas_brain" / "api" / "llm_gateway.py"


@pytest.fixture(scope="module")
def gateway_source() -> str:
    return GATEWAY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def helper():
    """Extract _cache_control_disables_cache from the source and exec
    it in a sealed namespace. Avoids importing the full gateway
    module (which requires SAAS env config)."""
    src = GATEWAY_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"^def _cache_control_disables_cache\(.*?(?=\n(?:\ndef |\nclass |\n@router\.))",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "Could not locate _cache_control_disables_cache in source"
    namespace: dict = {}
    exec(match.group(0), namespace)
    return namespace["_cache_control_disables_cache"]


# ---- Helper parsing -------------------------------------------------------


def test_helper_returns_false_for_none(helper):
    assert helper(None) is False


def test_helper_returns_false_for_empty_string(helper):
    assert helper("") is False
    assert helper("   ") is False


def test_helper_returns_true_for_no_store_alone(helper):
    assert helper("no-store") is True


def test_helper_is_case_insensitive(helper):
    assert helper("NO-STORE") is True
    assert helper("No-Store") is True


def test_helper_returns_true_when_combined_with_other_directives(helper):
    assert helper("no-cache, no-store") is True
    assert helper("no-store, max-age=0") is True
    assert helper("public, no-store, must-revalidate") is True


def test_helper_returns_false_for_unrelated_directives(helper):
    assert helper("max-age=0") is False
    assert helper("no-cache") is False
    assert helper("public") is False
    assert helper("private, max-age=600") is False


def test_helper_does_not_match_substrings(helper):
    """``maybe-no-store`` and ``no-stored`` are different tokens; the
    parser tokenizes on commas + equals, not substring."""
    assert helper("maybe-no-store") is False
    assert helper("no-stored") is False


def test_helper_handles_value_suffix_on_other_directives(helper):
    """``max-age=N`` has a value suffix; the parser strips before
    comparing the directive name. ``no-store`` itself doesn't take
    a value, but be permissive in case some client sends one."""
    assert helper("max-age=600") is False
    assert helper("no-store=anything") is True


def test_helper_tolerates_leading_trailing_whitespace(helper):
    assert helper("  no-store  ") is True
    assert helper("no-cache,  no-store  ,max-age=0") is True


# ---- Source-text contract assertions -------------------------------------


def test_chat_lookup_branch_is_gated_on_cache_disabled(gateway_source):
    """The exact-cache lookup must skip when cache_disabled is True."""
    # The lookup gate combines the feature flag and the header bypass.
    pattern = "is_llm_gateway_exact_cache_enabled() and not cache_disabled"
    assert pattern in gateway_source, (
        "Cache lookup must be gated on (feature_flag AND not cache_disabled)"
    )


def test_chat_store_branch_is_gated_on_cache_disabled(gateway_source):
    """The post-call store must skip when cache_disabled is True."""
    # The store gate is the same shape as the lookup gate.
    pattern = (
        "if text and is_llm_gateway_exact_cache_enabled() and not cache_disabled:"
    )
    assert pattern in gateway_source, (
        "Cache store must be gated on (text AND feature_flag AND not cache_disabled)"
    )


def test_chat_signature_accepts_cache_control_header(gateway_source):
    """chat() takes a Cache-Control header param via FastAPI Header."""
    # The header param uses alias="Cache-Control" (HTTP header name).
    assert 'cache_control: str | None = Header(default=None, alias="Cache-Control")' in gateway_source, (
        "chat() must accept Cache-Control header via FastAPI Header param"
    )


def test_cache_disabled_is_computed_once_before_lookup(gateway_source):
    """``cache_disabled`` is computed from the header at the top of the
    handler so both lookup and store branches see the same value."""
    # Source order: cache_disabled assignment must precede both gates.
    assignment_idx = gateway_source.find(
        "cache_disabled = _cache_control_disables_cache(cache_control)"
    )
    lookup_idx = gateway_source.find("await lookup_cached_text(")
    store_idx = gateway_source.find("await store_cached_text(")

    assert assignment_idx > 0
    assert lookup_idx > 0 and assignment_idx < lookup_idx, (
        "cache_disabled must be computed before the lookup branch"
    )
    assert store_idx > 0 and assignment_idx < store_idx, (
        "cache_disabled must be computed before the store branch"
    )
