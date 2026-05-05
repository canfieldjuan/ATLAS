"""Host integration port for battle-card shared support helpers."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Iterable, Protocol


STANDALONE_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"


class BattleCardSupportPortNotConfigured(RuntimeError):
    """Raised when a host has not registered battle-card support."""


class BattleCardSupportPort(Protocol):
    """Host operations used by battle-card assembly."""

    def _battle_card_best_supported_quote(self, *args: Any, **kwargs: Any) -> str: ...
    def _battle_card_fallback_recommended_plays(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]: ...
    def _battle_card_quote_terms(self, *args: Any, **kwargs: Any) -> list[str]: ...
    def _battle_card_safe_play_text(self, *args: Any, **kwargs: Any) -> str: ...
    def _battle_card_safe_summary(self, *args: Any, **kwargs: Any) -> str: ...
    def _battle_card_structured_proof_text(self, *args: Any, **kwargs: Any) -> str: ...
    def _battle_card_winning_position(self, *args: Any, **kwargs: Any) -> str: ...
    def _sanitize_battle_card_sales_copy(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def _battle_card_allowed_quotes(self, *args: Any, **kwargs: Any) -> list[str]: ...
    def _battle_card_has_duplicate_recommended_play_segments(self, *args: Any, **kwargs: Any) -> bool: ...
    def _validate_battle_card_sales_copy(self, *args: Any, **kwargs: Any) -> list[str]: ...
    def _build_battle_card_locked_facts(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def _build_metric_ledger(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    async def has_complete_core_run_marker(self, pool: Any, report_date: date) -> bool: ...
    async def latest_complete_core_report_date(self, pool: Any) -> date | None: ...
    async def describe_core_run_gap(self, pool: Any, report_date: date) -> str: ...
    async def update_execution_progress(self, task: Any, *, stage: str, progress_current: int | None = None, progress_total: int | None = None, progress_message: str | None = None, **counters: Any) -> None: ...
    def normalize_test_vendors(self, raw: Any) -> list[str]: ...
    def apply_vendor_scope_to_churn_inputs(self, data: dict[str, Any], vendor_names: list[str] | str | None) -> tuple[dict[str, Any], list[str]]: ...
    async def load_best_reasoning_views(self, pool: Any, vendor_names: list[str], *, as_of: date | None = None, analysis_window_days: int = 30) -> dict[str, Any]: ...
    def build_reasoning_lookup_from_views(self, synthesis_views: dict[str, Any]) -> dict[str, dict[str, Any]]: ...
    def _aggregate_competitive_disp(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def _build_deterministic_battle_cards(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def _build_pain_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_competitor_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_feature_gap_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_use_case_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_sentiment_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, int]]: ...
    def _build_buyer_auth_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict]: ...
    def _build_keyword_spike_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict]: ...
    def _battle_card_provenance_from_evidence_vault(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def _build_deterministic_battle_card_competitive_landscape(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def _build_deterministic_battle_card_weakness_analysis(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: ...
    def _build_positive_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_department_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_usage_duration_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _build_timeline_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]: ...
    def _canonicalize_vendor(self, raw: str) -> str: ...
    def _align_vendor_intelligence_records_to_scorecards(self, *args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]: ...
    async def read_vendor_scorecards(self, pool: Any, *, window_days: int, min_reviews: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]: ...
    async def read_vendor_intelligence_records(self, pool: Any, *, as_of: date, analysis_window_days: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]: ...
    async def _fetch_latest_account_intelligence(self, pool: Any, *, as_of: date, analysis_window_days: int) -> dict[str, dict[str, Any]]: ...
    async def _fetch_competitive_displacement_source_of_truth(self, pool: Any, *, as_of: date, analysis_window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_pain_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_feature_gaps(self, pool: Any, window_days: int, *, min_mentions: int = 2) -> list[dict[str, Any]]: ...
    async def _fetch_price_complaint_rates(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_dm_churn_rates(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_churning_companies(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_quotable_evidence(self, pool: Any, window_days: int, *, min_urgency: float = 4.5) -> list[dict[str, Any]]: ...
    async def _fetch_budget_signals(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_use_case_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_sentiment_trajectory(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_buyer_authority_summary(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_timeline_signals(self, pool: Any, window_days: int, *, limit: int = 50) -> list[dict[str, Any]]: ...
    async def _fetch_keyword_spikes(self, pool: Any) -> list[dict[str, Any]]: ...
    async def _fetch_product_profiles(self, pool: Any) -> list[dict[str, Any]]: ...
    async def _fetch_competitor_reasons(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_data_context(self, pool: Any, window_days: int) -> dict[str, Any]: ...
    async def _fetch_vendor_provenance(self, pool: Any, window_days: int) -> dict[str, dict]: ...
    async def _fetch_review_text_aggregates(self, pool: Any, window_days: int) -> tuple[list[dict], list[dict]]: ...
    async def _fetch_department_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]: ...
    async def _fetch_contract_context_distribution(self, pool: Any, window_days: int) -> tuple[list[dict], list[dict]]: ...


class _BridgeBattleCardSupportPort:
    def _shared(self) -> Any:
        from ...autonomous.tasks import _b2b_shared as shared

        return shared

    def _battle_card_best_supported_quote(self, *args: Any, **kwargs: Any) -> str:
        return self._shared()._battle_card_best_supported_quote(*args, **kwargs)

    def _battle_card_fallback_recommended_plays(self, *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        return self._shared()._battle_card_fallback_recommended_plays(*args, **kwargs)

    def _battle_card_quote_terms(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._shared()._battle_card_quote_terms(*args, **kwargs)

    def _battle_card_safe_play_text(self, *args: Any, **kwargs: Any) -> str:
        return self._shared()._battle_card_safe_play_text(*args, **kwargs)

    def _battle_card_safe_summary(self, *args: Any, **kwargs: Any) -> str:
        return self._shared()._battle_card_safe_summary(*args, **kwargs)

    def _battle_card_structured_proof_text(self, *args: Any, **kwargs: Any) -> str:
        return self._shared()._battle_card_structured_proof_text(*args, **kwargs)

    def _battle_card_winning_position(self, *args: Any, **kwargs: Any) -> str:
        return self._shared()._battle_card_winning_position(*args, **kwargs)

    def _sanitize_battle_card_sales_copy(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._shared()._sanitize_battle_card_sales_copy(*args, **kwargs)

    def _battle_card_allowed_quotes(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._shared()._battle_card_allowed_quotes(*args, **kwargs)

    def _battle_card_has_duplicate_recommended_play_segments(self, *args: Any, **kwargs: Any) -> bool:
        return self._shared()._battle_card_has_duplicate_recommended_play_segments(*args, **kwargs)

    def _validate_battle_card_sales_copy(self, *args: Any, **kwargs: Any) -> list[str]:
        return self._shared()._validate_battle_card_sales_copy(*args, **kwargs)

    def _build_battle_card_locked_facts(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._shared()._build_battle_card_locked_facts(*args, **kwargs)

    def _build_metric_ledger(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._shared()._build_metric_ledger(*args, **kwargs)

    async def has_complete_core_run_marker(self, pool: Any, report_date: date) -> bool:
        return await self._shared().has_complete_core_run_marker(pool, report_date)

    async def latest_complete_core_report_date(self, pool: Any) -> date | None:
        return await self._shared().latest_complete_core_report_date(pool)

    async def describe_core_run_gap(self, pool: Any, report_date: date) -> str:
        return await self._shared().describe_core_run_gap(pool, report_date)

    async def update_execution_progress(
        self,
        task: Any,
        *,
        stage: str,
        progress_current: int | None = None,
        progress_total: int | None = None,
        progress_message: str | None = None,
        **counters: Any,
    ) -> None:
        from ...autonomous.tasks._execution_progress import _update_execution_progress

        await _update_execution_progress(
            task,
            stage=stage,
            progress_current=progress_current,
            progress_total=progress_total,
            progress_message=progress_message,
            **counters,
        )

    def normalize_test_vendors(self, raw: Any) -> list[str]:
        from ...autonomous.tasks.b2b_churn_intelligence import _normalize_test_vendors

        return _normalize_test_vendors(raw)

    def apply_vendor_scope_to_churn_inputs(
        self,
        data: dict[str, Any],
        vendor_names: list[str] | str | None,
    ) -> tuple[dict[str, Any], list[str]]:
        from ...autonomous.tasks.b2b_churn_intelligence import (
            _apply_vendor_scope_to_churn_inputs,
        )

        return _apply_vendor_scope_to_churn_inputs(data, vendor_names)

    async def load_best_reasoning_views(
        self,
        pool: Any,
        vendor_names: list[str],
        *,
        as_of: date | None = None,
        analysis_window_days: int = 30,
    ) -> dict[str, Any]:
        from ...autonomous.tasks._b2b_synthesis_reader import load_best_reasoning_views

        return await load_best_reasoning_views(
            pool,
            vendor_names,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    def build_reasoning_lookup_from_views(
        self,
        synthesis_views: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        from ...autonomous.tasks._b2b_synthesis_reader import (
            build_reasoning_lookup_from_views,
        )

        return build_reasoning_lookup_from_views(synthesis_views)

    def _aggregate_competitive_disp(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._shared()._aggregate_competitive_disp(*args, **kwargs)

    def _build_deterministic_battle_cards(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._shared()._build_deterministic_battle_cards(*args, **kwargs)

    def _build_pain_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_pain_lookup(*args, **kwargs)

    def _build_competitor_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_competitor_lookup(*args, **kwargs)

    def _build_feature_gap_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_feature_gap_lookup(*args, **kwargs)

    def _build_use_case_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_use_case_lookup(*args, **kwargs)

    def _build_sentiment_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, int]]:
        return self._shared()._build_sentiment_lookup(*args, **kwargs)

    def _build_buyer_auth_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict]:
        return self._shared()._build_buyer_auth_lookup(*args, **kwargs)

    def _build_keyword_spike_lookup(self, *args: Any, **kwargs: Any) -> dict[str, dict]:
        return self._shared()._build_keyword_spike_lookup(*args, **kwargs)

    def _battle_card_provenance_from_evidence_vault(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._shared()._battle_card_provenance_from_evidence_vault(*args, **kwargs)

    def _build_deterministic_battle_card_competitive_landscape(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._shared()._build_deterministic_battle_card_competitive_landscape(*args, **kwargs)

    def _build_deterministic_battle_card_weakness_analysis(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._shared()._build_deterministic_battle_card_weakness_analysis(*args, **kwargs)

    def _build_positive_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_positive_lookup(*args, **kwargs)

    def _build_department_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_department_lookup(*args, **kwargs)

    def _build_usage_duration_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_usage_duration_lookup(*args, **kwargs)

    def _build_timeline_lookup(self, *args: Any, **kwargs: Any) -> dict[str, list[dict]]:
        return self._shared()._build_timeline_lookup(*args, **kwargs)

    def _canonicalize_vendor(self, raw: str) -> str:
        return self._shared()._canonicalize_vendor(raw)

    def _align_vendor_intelligence_records_to_scorecards(self, *args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        return self._shared()._align_vendor_intelligence_records_to_scorecards(*args, **kwargs)

    async def read_vendor_scorecards(self, pool: Any, *, window_days: int, min_reviews: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        return await self._shared().read_vendor_scorecards(
            pool,
            window_days=window_days,
            min_reviews=min_reviews,
            vendor_names=vendor_names,
        )

    async def read_vendor_intelligence_records(self, pool: Any, *, as_of: date, analysis_window_days: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        return await self._shared().read_vendor_intelligence_records(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
            vendor_names=vendor_names,
        )

    async def _fetch_latest_account_intelligence(self, pool: Any, *, as_of: date, analysis_window_days: int) -> dict[str, dict[str, Any]]:
        return await self._shared()._fetch_latest_account_intelligence(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    async def _fetch_competitive_displacement_source_of_truth(self, pool: Any, *, as_of: date, analysis_window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_competitive_displacement_source_of_truth(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    async def _fetch_pain_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_pain_distribution(pool, window_days)

    async def _fetch_feature_gaps(self, pool: Any, window_days: int, *, min_mentions: int = 2) -> list[dict[str, Any]]:
        return await self._shared()._fetch_feature_gaps(
            pool,
            window_days,
            min_mentions=min_mentions,
        )

    async def _fetch_price_complaint_rates(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_price_complaint_rates(pool, window_days)

    async def _fetch_dm_churn_rates(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_dm_churn_rates(pool, window_days)

    async def _fetch_churning_companies(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_churning_companies(pool, window_days)

    async def _fetch_quotable_evidence(self, pool: Any, window_days: int, *, min_urgency: float = 4.5) -> list[dict[str, Any]]:
        return await self._shared()._fetch_quotable_evidence(
            pool,
            window_days,
            min_urgency=min_urgency,
        )

    async def _fetch_budget_signals(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_budget_signals(pool, window_days)

    async def _fetch_use_case_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_use_case_distribution(pool, window_days)

    async def _fetch_sentiment_trajectory(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_sentiment_trajectory(pool, window_days)

    async def _fetch_buyer_authority_summary(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_buyer_authority_summary(pool, window_days)

    async def _fetch_timeline_signals(self, pool: Any, window_days: int, *, limit: int = 50) -> list[dict[str, Any]]:
        return await self._shared()._fetch_timeline_signals(pool, window_days, limit=limit)

    async def _fetch_keyword_spikes(self, pool: Any) -> list[dict[str, Any]]:
        return await self._shared()._fetch_keyword_spikes(pool)

    async def _fetch_product_profiles(self, pool: Any) -> list[dict[str, Any]]:
        return await self._shared()._fetch_product_profiles(pool)

    async def _fetch_competitor_reasons(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_competitor_reasons(pool, window_days)

    async def _fetch_data_context(self, pool: Any, window_days: int) -> dict[str, Any]:
        return await self._shared()._fetch_data_context(pool, window_days)

    async def _fetch_vendor_provenance(self, pool: Any, window_days: int) -> dict[str, dict]:
        return await self._shared()._fetch_vendor_provenance(pool, window_days)

    async def _fetch_review_text_aggregates(self, pool: Any, window_days: int) -> tuple[list[dict], list[dict]]:
        return await self._shared()._fetch_review_text_aggregates(pool, window_days)

    async def _fetch_department_distribution(self, pool: Any, window_days: int) -> list[dict[str, Any]]:
        return await self._shared()._fetch_department_distribution(pool, window_days)

    async def _fetch_contract_context_distribution(self, pool: Any, window_days: int) -> tuple[list[dict], list[dict]]:
        return await self._shared()._fetch_contract_context_distribution(pool, window_days)


_configured_support_port: BattleCardSupportPort | None = None
_bridge_support_port = _BridgeBattleCardSupportPort()


def configure_battle_card_support_port(port: BattleCardSupportPort | None) -> None:
    """Register the host adapter for battle-card support."""
    global _configured_support_port
    _configured_support_port = port


def get_battle_card_support_port() -> BattleCardSupportPort:
    """Return configured host support or the non-standalone bridge adapter."""
    if _configured_support_port is not None:
        return _configured_support_port
    if os.environ.get(STANDALONE_ENV_VAR) == "1":
        raise BattleCardSupportPortNotConfigured(
            "No battle-card support port has been configured"
        )
    return _bridge_support_port


def _battle_card_best_supported_quote(*args: Any, **kwargs: Any) -> str:
    return get_battle_card_support_port()._battle_card_best_supported_quote(*args, **kwargs)


def _battle_card_fallback_recommended_plays(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
    return get_battle_card_support_port()._battle_card_fallback_recommended_plays(*args, **kwargs)


def _battle_card_quote_terms(*args: Any, **kwargs: Any) -> list[str]:
    return get_battle_card_support_port()._battle_card_quote_terms(*args, **kwargs)


def _battle_card_safe_play_text(*args: Any, **kwargs: Any) -> str:
    return get_battle_card_support_port()._battle_card_safe_play_text(*args, **kwargs)


def _battle_card_safe_summary(*args: Any, **kwargs: Any) -> str:
    return get_battle_card_support_port()._battle_card_safe_summary(*args, **kwargs)


def _battle_card_structured_proof_text(*args: Any, **kwargs: Any) -> str:
    return get_battle_card_support_port()._battle_card_structured_proof_text(*args, **kwargs)


def _battle_card_winning_position(*args: Any, **kwargs: Any) -> str:
    return get_battle_card_support_port()._battle_card_winning_position(*args, **kwargs)


def _sanitize_battle_card_sales_copy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return get_battle_card_support_port()._sanitize_battle_card_sales_copy(*args, **kwargs)


def _battle_card_allowed_quotes(*args: Any, **kwargs: Any) -> list[str]:
    return get_battle_card_support_port()._battle_card_allowed_quotes(*args, **kwargs)


def _battle_card_has_duplicate_recommended_play_segments(*args: Any, **kwargs: Any) -> bool:
    return get_battle_card_support_port()._battle_card_has_duplicate_recommended_play_segments(*args, **kwargs)


def _validate_battle_card_sales_copy(*args: Any, **kwargs: Any) -> list[str]:
    return get_battle_card_support_port()._validate_battle_card_sales_copy(*args, **kwargs)


def _build_battle_card_locked_facts(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return get_battle_card_support_port()._build_battle_card_locked_facts(*args, **kwargs)


def _build_metric_ledger(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return get_battle_card_support_port()._build_metric_ledger(*args, **kwargs)


async def has_complete_core_run_marker(pool: Any, report_date: date) -> bool:
    return await get_battle_card_support_port().has_complete_core_run_marker(pool, report_date)


async def latest_complete_core_report_date(pool: Any) -> date | None:
    return await get_battle_card_support_port().latest_complete_core_report_date(pool)


async def describe_core_run_gap(pool: Any, report_date: date) -> str:
    return await get_battle_card_support_port().describe_core_run_gap(pool, report_date)


async def update_execution_progress(
    task: Any,
    *,
    stage: str,
    progress_current: int | None = None,
    progress_total: int | None = None,
    progress_message: str | None = None,
    **counters: Any,
) -> None:
    return await get_battle_card_support_port().update_execution_progress(
        task,
        stage=stage,
        progress_current=progress_current,
        progress_total=progress_total,
        progress_message=progress_message,
        **counters,
    )


def normalize_test_vendors(raw: Any) -> list[str]:
    return get_battle_card_support_port().normalize_test_vendors(raw)


def apply_vendor_scope_to_churn_inputs(
    data: dict[str, Any],
    vendor_names: list[str] | str | None,
) -> tuple[dict[str, Any], list[str]]:
    return get_battle_card_support_port().apply_vendor_scope_to_churn_inputs(
        data,
        vendor_names,
    )


async def load_best_reasoning_views(
    pool: Any,
    vendor_names: list[str],
    *,
    as_of: date | None = None,
    analysis_window_days: int = 30,
) -> dict[str, Any]:
    return await get_battle_card_support_port().load_best_reasoning_views(
        pool,
        vendor_names,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


def build_reasoning_lookup_from_views(
    synthesis_views: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return get_battle_card_support_port().build_reasoning_lookup_from_views(
        synthesis_views
    )


def _aggregate_competitive_disp(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return get_battle_card_support_port()._aggregate_competitive_disp(*args, **kwargs)


def _build_deterministic_battle_cards(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return get_battle_card_support_port()._build_deterministic_battle_cards(*args, **kwargs)


def _build_pain_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_pain_lookup(*args, **kwargs)


def _build_competitor_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_competitor_lookup(*args, **kwargs)


def _build_feature_gap_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_feature_gap_lookup(*args, **kwargs)


def _build_use_case_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_use_case_lookup(*args, **kwargs)


def _build_sentiment_lookup(*args: Any, **kwargs: Any) -> dict[str, dict[str, int]]:
    return get_battle_card_support_port()._build_sentiment_lookup(*args, **kwargs)


def _build_buyer_auth_lookup(*args: Any, **kwargs: Any) -> dict[str, dict]:
    return get_battle_card_support_port()._build_buyer_auth_lookup(*args, **kwargs)


def _build_keyword_spike_lookup(*args: Any, **kwargs: Any) -> dict[str, dict]:
    return get_battle_card_support_port()._build_keyword_spike_lookup(*args, **kwargs)


def _battle_card_provenance_from_evidence_vault(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return get_battle_card_support_port()._battle_card_provenance_from_evidence_vault(*args, **kwargs)


def _build_deterministic_battle_card_competitive_landscape(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return get_battle_card_support_port()._build_deterministic_battle_card_competitive_landscape(*args, **kwargs)


def _build_deterministic_battle_card_weakness_analysis(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return get_battle_card_support_port()._build_deterministic_battle_card_weakness_analysis(*args, **kwargs)


def _build_positive_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_positive_lookup(*args, **kwargs)


def _build_department_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_department_lookup(*args, **kwargs)


def _build_usage_duration_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_usage_duration_lookup(*args, **kwargs)


def _build_timeline_lookup(*args: Any, **kwargs: Any) -> dict[str, list[dict]]:
    return get_battle_card_support_port()._build_timeline_lookup(*args, **kwargs)


def _canonicalize_vendor(raw: str) -> str:
    return get_battle_card_support_port()._canonicalize_vendor(raw)


def _align_vendor_intelligence_records_to_scorecards(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return get_battle_card_support_port()._align_vendor_intelligence_records_to_scorecards(*args, **kwargs)


async def read_vendor_scorecards(pool: Any, *, window_days: int, min_reviews: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    return await get_battle_card_support_port().read_vendor_scorecards(
        pool,
        window_days=window_days,
        min_reviews=min_reviews,
        vendor_names=vendor_names,
    )


async def read_vendor_intelligence_records(pool: Any, *, as_of: date, analysis_window_days: int, vendor_names: Iterable[Any] | None = None) -> list[dict[str, Any]]:
    return await get_battle_card_support_port().read_vendor_intelligence_records(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
        vendor_names=vendor_names,
    )


async def _fetch_latest_account_intelligence(pool: Any, *, as_of: date, analysis_window_days: int) -> dict[str, dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_latest_account_intelligence(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def _fetch_competitive_displacement_source_of_truth(pool: Any, *, as_of: date, analysis_window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_competitive_displacement_source_of_truth(
        pool,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def _fetch_pain_distribution(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_pain_distribution(pool, window_days)


async def _fetch_feature_gaps(pool: Any, window_days: int, *, min_mentions: int = 2) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_feature_gaps(
        pool,
        window_days,
        min_mentions=min_mentions,
    )


async def _fetch_price_complaint_rates(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_price_complaint_rates(pool, window_days)


async def _fetch_dm_churn_rates(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_dm_churn_rates(pool, window_days)


async def _fetch_churning_companies(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_churning_companies(pool, window_days)


async def _fetch_quotable_evidence(pool: Any, window_days: int, *, min_urgency: float = 4.5) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_quotable_evidence(
        pool,
        window_days,
        min_urgency=min_urgency,
    )


async def _fetch_budget_signals(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_budget_signals(pool, window_days)


async def _fetch_use_case_distribution(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_use_case_distribution(pool, window_days)


async def _fetch_sentiment_trajectory(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_sentiment_trajectory(pool, window_days)


async def _fetch_buyer_authority_summary(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_buyer_authority_summary(pool, window_days)


async def _fetch_timeline_signals(pool: Any, window_days: int, *, limit: int = 50) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_timeline_signals(
        pool,
        window_days,
        limit=limit,
    )


async def _fetch_keyword_spikes(pool: Any) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_keyword_spikes(pool)


async def _fetch_product_profiles(pool: Any) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_product_profiles(pool)


async def _fetch_competitor_reasons(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_competitor_reasons(pool, window_days)


async def _fetch_data_context(pool: Any, window_days: int) -> dict[str, Any]:
    return await get_battle_card_support_port()._fetch_data_context(pool, window_days)


async def _fetch_vendor_provenance(pool: Any, window_days: int) -> dict[str, dict]:
    return await get_battle_card_support_port()._fetch_vendor_provenance(pool, window_days)


async def _fetch_review_text_aggregates(pool: Any, window_days: int) -> tuple[list[dict], list[dict]]:
    return await get_battle_card_support_port()._fetch_review_text_aggregates(pool, window_days)


async def _fetch_department_distribution(pool: Any, window_days: int) -> list[dict[str, Any]]:
    return await get_battle_card_support_port()._fetch_department_distribution(pool, window_days)


async def _fetch_contract_context_distribution(pool: Any, window_days: int) -> tuple[list[dict], list[dict]]:
    return await get_battle_card_support_port()._fetch_contract_context_distribution(
        pool,
        window_days,
    )


__all__ = [
    "BattleCardSupportPort",
    "BattleCardSupportPortNotConfigured",
    "STANDALONE_ENV_VAR",
    "apply_vendor_scope_to_churn_inputs",
    "build_reasoning_lookup_from_views",
    "configure_battle_card_support_port",
    "get_battle_card_support_port",
    "load_best_reasoning_views",
    "normalize_test_vendors",
    "update_execution_progress",
]
