from atlas_brain.autonomous.tasks.b2b_blog_post_generation import PostBlueprint
from atlas_brain.services import blog_quality as mod


def _blueprint(data_context: dict | None = None) -> PostBlueprint:
    return PostBlueprint(
        topic_type="migration_guide",
        slug="switch-to-shopify-2026-03",
        suggested_title="Migration Guide: Why Teams Are Switching to Shopify",
        tags=["shopify", "migration"],
        data_context=data_context or {},
        sections=[],
        charts=[],
        quotable_phrases=[],
    )


def test_blog_quality_revalidation_merges_latest_audit():
    blueprint = _blueprint(
        {
            "reasoning_anchor_examples": {
                "outlier_or_named_account": [
                    {
                        "witness_id": "w1",
                        "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                        "time_anchor": "Q2 renewal",
                        "numeric_literals": {"currency_mentions": ["$200k/year"]},
                        "competitor": "BigCommerce",
                        "pain_category": "pricing",
                    }
                ]
            },
            "reasoning_witness_highlights": [
                {
                    "witness_id": "w1",
                    "excerpt_text": "a customer hit a $200k/year renewal issue in Q2",
                    "time_anchor": "Q2 renewal",
                    "numeric_literals": {"currency_mentions": ["$200k/year"]},
                    "competitor": "BigCommerce",
                    "pain_category": "pricing",
                }
            ],
            "reasoning_reference_ids": {"witness_ids": ["w1"]},
        }
    )

    result = mod.blog_quality_revalidation(
        blueprint=blueprint,
        content={
            "title": blueprint.suggested_title,
            "description": "desc",
            "content": "<p>The Q2 renewal now carries a $200k/year pricing issue versus BigCommerce.</p>",
        },
        boundary="generation",
        report={
            "status": "pass",
            "score": 88,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": [],
        },
    )

    audit = result["audit"]
    assert audit["status"] == "pass"
    assert audit["boundary"] == "generation"
    assert set(audit["matched_groups"]) == {
        "timing_terms",
        "numeric_terms",
        "competitor_terms",
        "pain_terms",
    }
    assert result["data_context"]["latest_quality_audit"]["failure_explanation"]["anchor_count"] == 1
    assert result["data_context"]["generation_quality"]["score"] == 88


def test_blog_quality_revalidation_classifies_unsupported_claims():
    blueprint = _blueprint({"vendor": "Shopify"})

    result = mod.blog_quality_revalidation(
        blueprint=blueprint,
        content={
            "title": blueprint.suggested_title,
            "description": "desc",
            "content": "<p>Generic draft body.</p>",
        },
        boundary="publish",
        report={
            "status": "pass",
            "score": 82,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": ["unsupported_data_claim:Magento"],
        },
    )

    explanation = result["audit"]["failure_explanation"]
    assert result["audit"]["status"] == "fail"
    assert explanation["primary_blocker"] == "critical_warning_unresolved:unsupported_data_claim:Magento"
    assert explanation["cause_type"] == "unsupported_claim"


def test_blog_quality_revalidation_flags_missing_upstream_context():
    blueprint = _blueprint({"vendor": "Shopify"})

    result = mod.blog_quality_revalidation(
        blueprint=blueprint,
        content={
            "title": blueprint.suggested_title,
            "description": "desc",
            "content": "<p>Generic draft body.</p>",
        },
        boundary="backfill",
        report={
            "status": "fail",
            "score": 60,
            "threshold": 70,
            "blocking_issues": [
                "witness_specificity:content does not reference any witness-backed anchor despite anchors being available"
            ],
            "warnings": [],
        },
    )

    explanation = result["audit"]["failure_explanation"]
    assert explanation["cause_type"] == "upstream_data_missing"
    assert "reasoning_anchor_examples" in explanation["missing_inputs"]


def test_merge_blog_first_pass_quality_data_context():
    merged = mod.merge_blog_first_pass_quality_data_context(
        data_context={"vendor": "Shopify"},
        audit={
            "status": "fail",
            "score": 82,
            "threshold": 70,
            "blocking_issues": [],
            "warnings": ["unsupported_data_claim:Magento"],
            "failure_explanation": {"primary_blocker": "unsupported_data_claim:Magento"},
        },
    )

    assert merged["latest_first_pass_quality_audit"]["boundary"] == "generation_first_pass"
    assert merged["latest_first_pass_quality_audit"]["warnings"] == ["unsupported_data_claim:Magento"]
    assert mod.latest_blog_first_pass_quality_audit(merged)["score"] == 82
    assert mod.blog_first_pass_failure_explanation(merged)["primary_blocker"] == "unsupported_data_claim:Magento"
