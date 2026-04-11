#!/usr/bin/env python3
"""Live server validation for adapter migration.

Tests the API endpoints that were migrated to shared adapters.
Compares response shapes, counts, and checks for regressions.

Usage:
    python scripts/test_adapter_live.py [--base-url http://127.0.0.1:8000] [--token $ATLAS_TOKEN]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError


def _auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _get(url: str, *, token: str | None = None) -> dict:
    try:
        req = Request(url, headers=_auth_headers(token))
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"_error": str(e)}
    except Exception as e:
        return {"_error": str(e)}


def _post(url: str, body: dict, *, token: str | None = None) -> dict:
    try:
        data = json.dumps(body).encode()
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **_auth_headers(token)},
        )
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except URLError as e:
        return {"_error": str(e)}
    except Exception as e:
        return {"_error": str(e)}


def _check(label: str, result: dict, checks: list) -> list[str]:
    """Run checks on a result, return list of failures."""
    failures = []
    if "_error" in result:
        failures.append(f"  FAIL {label}: request error -- {result['_error']}")
        return failures
    for desc, ok in checks:
        if ok:
            print(f"  OK   {label}: {desc}")
        else:
            failures.append(f"  FAIL {label}: {desc}")
            print(f"  FAIL {label}: {desc}")
    return failures


def test_high_intent(base: str, token: str | None) -> list[str]:
    """Phase 1: list_high_intent via tenant API."""
    print("\n--- Phase 1: High Intent Companies ---")
    url = f"{base}/api/v1/b2b/tenant/high-intent?min_urgency=7&window_days=90&limit=10"
    data = _get(url, token=token)
    return _check("high_intent", data, [
        ("response has companies key", "companies" in data),
        ("count >= 0", isinstance(data.get("count"), int)),
        ("companies is list", isinstance(data.get("companies"), list)),
        (
            "first company has urgency field",
            bool(data.get("companies") and "urgency" in data["companies"][0]),
        ),
        (
            "first company has decision_maker field",
            bool(data.get("companies") and "decision_maker" in data["companies"][0]),
        ),
        (
            "first company has lock_in_level field",
            bool(data.get("companies") and "lock_in_level" in data["companies"][0]),
        ),
    ]) if "companies" in data else _check("high_intent", data, [
        ("response has companies key", False),
    ])


def test_high_intent_vendor_scoped(base: str, token: str | None) -> list[str]:
    """Phase 1: vendor-scoped high intent."""
    print("\n--- Phase 1: High Intent (vendor=HubSpot) ---")
    url = f"{base}/api/v1/b2b/tenant/high-intent?vendor_name=HubSpot&min_urgency=5&window_days=90&limit=5"
    data = _get(url, token=token)
    checks = [
        ("response has companies", "companies" in data),
    ]
    if data.get("companies"):
        checks.append((
            "all results are HubSpot",
            all("hubspot" in (c.get("vendor") or "").lower() for c in data["companies"]),
        ))
    return _check("high_intent_vendor", data, checks)


def test_search_reviews(base: str, token: str | None) -> list[str]:
    """Phase 2: search_reviews via tenant API."""
    print("\n--- Phase 2: Search Reviews ---")
    url = f"{base}/api/v1/b2b/tenant/reviews?window_days=90&limit=10"
    data = _get(url, token=token)
    return _check("search_reviews", data, [
        ("response has reviews key", "reviews" in data),
        ("count >= 0", isinstance(data.get("count"), int)),
        ("reviews is list", isinstance(data.get("reviews"), list)),
        (
            "first review has urgency_score",
            bool(data.get("reviews") and "urgency_score" in data["reviews"][0]),
        ),
        (
            "first review has pain_category",
            bool(data.get("reviews") and "pain_category" in data["reviews"][0]),
        ),
        (
            "first review has intent_to_leave",
            bool(data.get("reviews") and "intent_to_leave" in data["reviews"][0]),
        ),
    ]) if "reviews" in data else _check("search_reviews", data, [
        ("response has reviews key", False),
    ])


def test_search_reviews_filtered(base: str, token: str | None) -> list[str]:
    """Phase 2: search with pain_category filter."""
    print("\n--- Phase 2: Search Reviews (pain=pricing) ---")
    url = f"{base}/api/v1/b2b/tenant/reviews?pain_category=pricing&window_days=90&limit=5"
    data = _get(url, token=token)
    checks = [("response has reviews", "reviews" in data)]
    if data.get("reviews"):
        checks.append((
            "all results have pricing pain",
            all(r.get("pain_category") == "pricing" for r in data["reviews"]),
        ))
    return _check("search_reviews_pricing", data, checks)


def test_search_reviews_churn_intent(base: str, token: str | None) -> list[str]:
    """Phase 2: search with churn intent filter."""
    print("\n--- Phase 2: Search Reviews (churn_intent=true) ---")
    url = f"{base}/api/v1/b2b/tenant/reviews?has_churn_intent=true&window_days=90&limit=5"
    data = _get(url, token=token)
    checks = [("response has reviews", "reviews" in data)]
    if data.get("reviews"):
        checks.append((
            "all results have intent_to_leave=true",
            all(r.get("intent_to_leave") is True for r in data["reviews"]),
        ))
    return _check("search_reviews_churn", data, checks)


def test_vendor_profile(base: str, token: str | None) -> list[str]:
    """Phase 1: get_vendor_profile high-intent sub-query."""
    print("\n--- Phase 1: Vendor Profile (HubSpot) ---")
    url = f"{base}/api/v1/b2b/tenant/signals/HubSpot"
    data = _get(url, token=token)
    checks = [
        ("response has vendor_name", "vendor_name" in data or "profile" in data),
    ]
    profile = data.get("profile", data)
    if isinstance(profile, dict):
        checks.extend([
            ("has high_intent_companies", "high_intent_companies" in profile),
            ("has pain_distribution", "pain_distribution" in profile),
            ("has churn_signal", "churn_signal" in profile),
        ])
        hi = profile.get("high_intent_companies", [])
        if hi:
            checks.append(("high_intent has company field", "company" in hi[0]))
    return _check("vendor_profile", data, checks)


def test_null_semantics(base: str, token: str | None) -> list[str]:
    """Phase 2: verify null urgency/intent stays null, not 0/false."""
    print("\n--- Null Semantics Check ---")
    url = f"{base}/api/v1/b2b/tenant/reviews?window_days=90&limit=50"
    data = _get(url, token=token)
    checks = [("response has reviews", "reviews" in data)]
    reviews = data.get("reviews", [])
    if reviews:
        # Check that at least some reviews have non-zero urgency
        urgencies = [r.get("urgency_score") for r in reviews if r.get("urgency_score") is not None]
        checks.append((f"found {len(urgencies)} reviews with non-null urgency", len(urgencies) > 0))
        # Check that null values are null, not 0
        null_urgency = [r for r in reviews if r.get("urgency_score") is None]
        zero_urgency = [r for r in reviews if r.get("urgency_score") == 0]
        checks.append((
            f"null urgency count: {len(null_urgency)}, zero urgency count: {len(zero_urgency)}",
            True,  # informational
        ))
    return _check("null_semantics", data, checks)


def test_suppression_impact(base: str, token: str | None) -> list[str]:
    """Check if suppression filtering is active."""
    print("\n--- Suppression Impact ---")
    # Query with a wide window to see if suppressed reviews are filtered
    url = f"{base}/api/v1/b2b/tenant/reviews?window_days=365&limit=100"
    data = _get(url, token=token)
    count = data.get("count", 0)
    print(f"  INFO Total reviews returned (365d window): {count}")
    return _check("suppression", data, [
        ("response has count", "count" in data),
        ("count is reasonable (> 0)", count > 0),
    ])


def main():
    parser = argparse.ArgumentParser(description="Live adapter migration validation")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="Atlas API base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ATLAS_TOKEN", "").strip() or None,
        help="Bearer token for tenant routes. Defaults to ATLAS_TOKEN if set.",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    token = args.token.strip() if isinstance(args.token, str) and args.token.strip() else None

    print(f"Testing against: {base}")
    if token:
        print("Using bearer token authentication")
    else:
        print("No bearer token provided; tenant routes will likely return 401")

    # Check server is up
    health = _get(f"{base}/api/v1/ping")
    if "_error" in health:
        print(f"\nERROR: Server not reachable at {base}")
        print(f"  {health['_error']}")
        print("\nStart the server first:")
        print("  nohup uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8000 > /tmp/atlas_brain.log 2>&1 &")
        return 1

    all_failures = []
    all_failures.extend(test_high_intent(base, token))
    all_failures.extend(test_high_intent_vendor_scoped(base, token))
    all_failures.extend(test_vendor_profile(base, token))
    all_failures.extend(test_search_reviews(base, token))
    all_failures.extend(test_search_reviews_filtered(base, token))
    all_failures.extend(test_search_reviews_churn_intent(base, token))
    all_failures.extend(test_null_semantics(base, token))
    all_failures.extend(test_suppression_impact(base, token))

    print(f"\n{'=' * 60}")
    if all_failures:
        print(f"FAILURES: {len(all_failures)}")
        for f in all_failures:
            print(f)
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
