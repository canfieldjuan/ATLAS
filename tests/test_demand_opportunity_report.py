"""Unit tests for the demand-opportunity report's pure aggregation + filter.

No database required -- covers the relevance filter, the
overall_dissatisfaction baseline exclusion, the opportunity ranking, and the
feature-gap / competitor rollups, which is the logic that determines the
report's correctness.
"""

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "demand_opportunity_report",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "demand_opportunity_report.py",
)
dor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dor)


def _review(vendor, urgency, pains, **kw):
    return {
        "vendor": vendor,
        "urgency": urgency,
        "pain_categories": [{"category": c, "severity": "primary"} for c in pains],
        "feature_gaps": kw.get("feature_gaps", []),
        "specific_complaints": kw.get("specific_complaints", []),
        "competitors": kw.get("competitors", []),
        "pricing_phrases": kw.get("pricing_phrases", []),
        "positive_aspects": kw.get("positive_aspects", []),
        "quotable_phrases": kw.get("quotable_phrases", []),
        "price_increase": kw.get("price_increase", False),
        "source": kw.get("source", "G2"),
    }


def test_offtopic_detection_high_confidence_only():
    # Common-word-vendor contamination -> off-topic.
    assert dor._review_is_offtopic(_review("Copper", 5, ["pricing"],
        pricing_phrases=["craft 30 stone axes vs the copper axe"]))
    assert dor._review_is_offtopic(_review("Copper", 5, ["pricing"],
        quotable_phrases=["1500 durability compared to 150"]))
    assert dor._review_is_offtopic(_review("Close", 5, ["features"],
        feature_gaps=["silver plated copper cable is flexible"]))
    # Legitimate SaaS reviews must NOT be dropped (no false positives on plan
    # names or ordinary words).
    assert not dor._review_is_offtopic(_review("Zoho CRM", 5, ["pricing"],
        pricing_phrases=["the Silver plan is $12/user/month"]))
    assert not dor._review_is_offtopic(_review("Salesforce", 5, ["ux"],
        feature_gaps=["needs a more durable workflow audit trail"]))


def test_aggregate_excludes_offtopic_and_counts_them():
    reviews = [
        _review("Copper", 8, ["pricing"], pricing_phrases=["110 silver per day for wood"]),
        _review("Zoho CRM", 6, ["pricing"], pricing_phrases=["$15/user/mo too high"]),
        _review("Salesforce", 4, ["pricing"]),
    ]
    data = dor.aggregate(reviews)
    assert data["filtered_offtopic"] == 1
    assert data["filtered_vendors"] == {"Copper": 1}
    assert data["total_reviews"] == 2  # off-topic Copper review excluded
    pricing = next(t for t in data["themes"] if t["theme"] == "pricing")
    assert pricing["reviews"] == 2  # only the two clean reviews counted


def test_overall_dissatisfaction_is_baseline_not_a_theme():
    reviews = [
        _review("Close", 3, ["overall_dissatisfaction"]),
        _review("Close", 3, ["overall_dissatisfaction"]),
        _review("Pipedrive", 5, ["support"]),
    ]
    data = dor.aggregate(reviews)
    assert data["baseline_overall_dissatisfaction"] == 2
    themes = {t["theme"] for t in data["themes"]}
    assert "overall_dissatisfaction" not in themes
    assert "support" in themes


def test_opportunity_ranking_rewards_breadth_and_urgency():
    # Theme A: 2 vendors, high urgency. Theme B: 1 vendor, low urgency.
    reviews = [
        _review("Salesforce", 8, ["pricing"]),
        _review("Zoho CRM", 8, ["pricing"]),
        _review("Salesforce", 2, ["api_limitations"]),
    ]
    data = dor.aggregate(reviews)
    ranked = [t["theme"] for t in data["themes"]]
    assert ranked[0] == "pricing"  # broader + more urgent -> higher score
    pricing = next(t for t in data["themes"] if t["theme"] == "pricing")
    assert pricing["vendor_breadth"] == "2/2"


def test_feature_gaps_and_competitors_rollup():
    reviews = [
        _review("Close", 5, ["features"], feature_gaps=["deeper reporting", "Deeper Reporting"],
                competitors=[{"name": "HubSpot", "context": "compared"}]),
        _review("Pipedrive", 5, ["features"], feature_gaps=["deeper reporting"],
                competitors=[{"name": "HubSpot", "context": "considering"}]),
    ]
    data = dor.aggregate(reviews)
    top_gap = data["feature_gaps"][0]
    assert top_gap["count"] == 3  # case-normalized dedup across reviews
    assert "reporting" in top_gap["gap"].lower()
    hubspot = next(c for c in data["competitors"] if c["name"] == "HubSpot")
    assert hubspot["count"] == 2
    assert hubspot["contexts"] == {"compared": 1, "considering": 1}
