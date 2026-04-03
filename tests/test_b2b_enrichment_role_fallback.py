def _base_result():
    return {
        "churn_signals": {
            "intent_to_leave": True,
            "actively_evaluating": False,
            "migration_in_progress": False,
            "support_escalation": False,
            "contract_renewal_mentioned": False,
        },
        "urgency_score": 7,
        "reviewer_context": {
            "role_level": "unknown",
            "decision_maker": False,
        },
        "buyer_authority": {
            "role_type": "unknown",
            "has_budget_authority": False,
            "executive_sponsor_mentioned": False,
            "buying_stage": "unknown",
        },
    }


def test_validate_enrichment_normalizes_legacy_decision_maker_role():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    result["buyer_authority"]["role_type"] = "decision_maker"

    assert _validate_enrichment(result)
    assert result["buyer_authority"]["role_type"] == "economic_buyer"


def test_validate_enrichment_infers_economic_buyer_from_director_role_level():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    result["reviewer_context"]["role_level"] = "director"

    assert _validate_enrichment(result, {"reviewer_title": "Director of IT"})
    assert result["buyer_authority"]["role_type"] == "economic_buyer"


def test_validate_enrichment_infers_champion_from_manager_role_level():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    result["reviewer_context"]["role_level"] = "manager"

    assert _validate_enrichment(result, {"reviewer_title": "Customer Success Manager"})
    assert result["buyer_authority"]["role_type"] == "champion"


def test_validate_enrichment_infers_evaluator_from_technical_title_during_evaluation():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    result["reviewer_context"]["role_level"] = "ic"
    result["buyer_authority"]["buying_stage"] = "evaluation"

    assert _validate_enrichment(result, {"reviewer_title": "Solutions Architect"})
    assert result["buyer_authority"]["role_type"] == "evaluator"


def test_validate_enrichment_ignores_noisy_title_and_uses_role_level_fallback():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    result["reviewer_context"]["role_level"] = "ic"

    assert _validate_enrichment(result, {"reviewer_title": "Repeat Churn Signal (Score: 10.0)"})
    assert result["buyer_authority"]["role_type"] == "end_user"


def test_sanitize_reviewer_title_nulls_repeat_churn_signal():
    from atlas_brain.services.b2b.reviewer_identity import sanitize_reviewer_title

    assert sanitize_reviewer_title("Repeat Churn Signal (Score: 10.0)") is None
    assert sanitize_reviewer_title("Product Manager") == "Product Manager"


def test_validate_enrichment_infers_economic_buyer_from_purchase_text():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    source_row = {
        "reviewer_title": "",
        "content_type": "review",
        "summary": "Renewal decision",
        "review_text": "We decided to switch at renewal after I signed off on the purchase.",
    }

    assert _validate_enrichment(result, source_row)
    assert result["buyer_authority"]["role_type"] == "economic_buyer"
    assert result["reviewer_context"]["decision_maker"] is True


def test_validate_enrichment_infers_evaluator_from_text_without_title():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    source_row = {
        "reviewer_title": "",
        "content_type": "community_discussion",
        "summary": "Shortlisting alternatives",
        "review_text": "Our team is evaluating alternatives and running a proof of concept this month.",
    }

    assert _validate_enrichment(result, source_row)
    assert result["buyer_authority"]["role_type"] == "evaluator"


def test_validate_enrichment_infers_end_user_from_usage_text_without_title():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    source_row = {
        "reviewer_title": "",
        "content_type": "review",
        "summary": "Daily workflow",
        "review_text": "We use HubSpot daily for pipeline updates and I use it for reporting.",
    }

    assert _validate_enrichment(result, source_row)
    assert result["buyer_authority"]["role_type"] == "end_user"


def test_validate_enrichment_does_not_treat_build_language_as_purchase_authority():
    from atlas_brain.autonomous.tasks.b2b_enrichment import _validate_enrichment

    result = _base_result()
    source_row = {
        "reviewer_title": "",
        "content_type": "community_discussion",
        "summary": "Built a workaround",
        "review_text": "I decided to build my own plugin because the missing feature was annoying.",
    }

    assert _validate_enrichment(result, source_row)
    assert result["buyer_authority"]["role_type"] == "unknown"


def test_queue_version_upgrades_returns_zero_when_auto_requeue_disabled(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_enrichment as mod

    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "enrichment_auto_requeue_parser_upgrades",
        False,
    )

    class _Pool:
        async def fetchval(self, *_args, **_kwargs):
            raise AssertionError("fetchval should not be called when requeue is disabled")

    result = __import__("asyncio").run(mod._queue_version_upgrades(_Pool()))
    assert result == 0


def test_queue_version_upgrades_counts_updated_rows(monkeypatch):
    from atlas_brain.autonomous.tasks import b2b_enrichment as mod

    monkeypatch.setattr(
        mod.settings.b2b_churn,
        "enrichment_auto_requeue_parser_upgrades",
        True,
    )

    class _Parser:
        version = "g2:1"

    monkeypatch.setattr(
        "atlas_brain.services.scraping.parsers.get_all_parsers",
        lambda: {"g2": _Parser()},
    )

    class _Pool:
        def __init__(self):
            self.calls = []

        async def fetchval(self, sql, source_name, current_version):
            self.calls.append((sql, source_name, current_version))
            return 7

    pool = _Pool()
    result = __import__("asyncio").run(mod._queue_version_upgrades(pool))

    assert result == 7
    assert len(pool.calls) == 1
    sql, source_name, current_version = pool.calls[0]
    assert "WITH updated AS" in sql
    assert source_name == "g2"
    assert current_version == "g2:1"
