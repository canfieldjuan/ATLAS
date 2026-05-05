"""Host integration port for vendor briefing intelligence readers."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Protocol


STANDALONE_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"


class VendorBriefingIntelligencePortNotConfigured(RuntimeError):
    """Raised when a host has not registered vendor briefing intelligence support."""


class VendorBriefingIntelligencePort(Protocol):
    """Host operations used by vendor briefing assembly."""

    def reasoning_int(self, value: Any) -> int | None:
        """Coerce reasoning contract values into integers."""

    def timing_summary_payload(
        self,
        timing_intelligence: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Return timing summary text, metrics, and trigger labels."""

    def align_vendor_intelligence_record_to_scorecard(
        self,
        scorecard: dict[str, Any] | None,
        record: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Align one vendor evidence-vault record against one scorecard row."""

    def inject_synthesis_freshness(
        self,
        entry: dict[str, Any],
        view: Any,
        *,
        requested_as_of: date | None = None,
    ) -> None:
        """Apply synthesis freshness metadata to an output entry."""

    def load_synthesis_view(
        self,
        raw: dict[str, Any],
        vendor_name: str,
        schema_version: str = "",
        as_of_date: date | str | None = None,
    ) -> Any:
        """Construct a host-owned synthesis view."""

    async def load_best_reasoning_view(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date | None = None,
        analysis_window_days: int = 30,
    ) -> Any | None:
        """Load the best reasoning view for one vendor."""

    async def load_prior_reasoning_snapshots(
        self,
        pool: Any,
        vendor_names: list[str],
        *,
        before_date: date | None = None,
        analysis_window_days: int = 30,
    ) -> dict[str, dict[str, Any]]:
        """Load prior reasoning snapshots for vendors."""

    async def read_vendor_company_signal_review_queue(
        self,
        pool: Any,
        *,
        vendor_name: str,
        window_days: int | None = None,
        preview_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return pending company-signal review context for one vendor."""

    async def read_vendor_intelligence_record(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date,
        analysis_window_days: int,
    ) -> dict[str, Any] | None:
        """Return the canonical vendor-intelligence row with run metadata."""

    async def read_vendor_intelligence(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date,
        analysis_window_days: int,
    ) -> dict[str, Any] | None:
        """Return the canonical vendor evidence-vault payload."""

    async def read_vendor_scorecard_detail(
        self,
        pool: Any,
        vendor_name: str,
    ) -> dict[str, Any] | None:
        """Return the detailed derived vendor scorecard row."""

    async def read_vendor_quote_evidence(
        self,
        pool: Any,
        *,
        vendor_name: str,
        window_days: int = 90,
        min_urgency: float = 5.0,
        limit: int = 10,
        sources: list[str] | None = None,
        pain_filter: str | None = None,
        require_quotes: bool = False,
        recency_column: str = "enriched_at",
    ) -> list[dict[str, Any]]:
        """Return row-level quote evidence for one vendor."""


class VendorBriefingRuntimePort(Protocol):
    """Host operations used by vendor briefing LLM/cache runtime."""

    def normalize_openrouter_model(self, model: Any | None, *, context: str = "") -> str:
        """Return the host-normalized OpenRouter model name."""

    def clean_llm_output(self, text: str) -> str:
        """Clean provider output before JSON parsing."""

    def get_campaign_llm(self) -> Any:
        """Return the host campaign LLM used by account-card enrichment."""

    def build_llm_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        """Build host chat message objects for LLM/cache calls."""

    def prepare_b2b_exact_stage_request(self, stage_id: str, **kwargs: Any) -> Any:
        """Build a host exact-cache request for a declared B2B stage."""

    async def lookup_b2b_exact_stage_text(self, request: Any) -> Any | None:
        """Look up exact-cache text for a declared B2B stage."""

    async def store_b2b_exact_stage_text(self, request: Any, **kwargs: Any) -> bool:
        """Store exact-cache text for a declared B2B stage."""

    def trace_llm_call(self, span_name: str, **kwargs: Any) -> None:
        """Record host LLM usage telemetry."""


class _BridgeVendorBriefingIntelligencePort:
    def reasoning_int(self, value: Any) -> int | None:
        from ...autonomous.tasks import _b2b_shared as shared

        return shared._reasoning_int(value)

    def timing_summary_payload(
        self,
        timing_intelligence: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], list[str]]:
        from ...autonomous.tasks import _b2b_shared as shared

        return shared._timing_summary_payload(timing_intelligence)

    def align_vendor_intelligence_record_to_scorecard(
        self,
        scorecard: dict[str, Any] | None,
        record: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        from ...autonomous.tasks import _b2b_shared as shared

        return shared._align_vendor_intelligence_record_to_scorecard(scorecard, record)

    def inject_synthesis_freshness(
        self,
        entry: dict[str, Any],
        view: Any,
        *,
        requested_as_of: date | None = None,
    ) -> None:
        from ...autonomous.tasks._b2b_synthesis_reader import inject_synthesis_freshness

        inject_synthesis_freshness(
            entry,
            view,
            requested_as_of=requested_as_of,
        )

    def load_synthesis_view(
        self,
        raw: dict[str, Any],
        vendor_name: str,
        schema_version: str = "",
        as_of_date: date | str | None = None,
    ) -> Any:
        from ...autonomous.tasks._b2b_synthesis_reader import load_synthesis_view

        return load_synthesis_view(
            raw,
            vendor_name,
            schema_version=schema_version,
            as_of_date=as_of_date,
        )

    async def load_best_reasoning_view(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date | None = None,
        analysis_window_days: int = 30,
    ) -> Any | None:
        from ...autonomous.tasks._b2b_synthesis_reader import load_best_reasoning_view

        return await load_best_reasoning_view(
            pool,
            vendor_name,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    async def load_prior_reasoning_snapshots(
        self,
        pool: Any,
        vendor_names: list[str],
        *,
        before_date: date | None = None,
        analysis_window_days: int = 30,
    ) -> dict[str, dict[str, Any]]:
        from ...autonomous.tasks._b2b_synthesis_reader import (
            load_prior_reasoning_snapshots,
        )

        return await load_prior_reasoning_snapshots(
            pool,
            vendor_names,
            before_date=before_date,
            analysis_window_days=analysis_window_days,
        )

    async def read_vendor_company_signal_review_queue(
        self,
        pool: Any,
        *,
        vendor_name: str,
        window_days: int | None = None,
        preview_limit: int | None = None,
    ) -> dict[str, Any]:
        from ...autonomous.tasks import _b2b_shared as shared

        return await shared.read_vendor_company_signal_review_queue(
            pool,
            vendor_name=vendor_name,
            window_days=window_days,
            preview_limit=preview_limit,
        )

    async def read_vendor_intelligence_record(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date,
        analysis_window_days: int,
    ) -> dict[str, Any] | None:
        from ...autonomous.tasks import _b2b_shared as shared

        return await shared.read_vendor_intelligence_record(
            pool,
            vendor_name,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    async def read_vendor_intelligence(
        self,
        pool: Any,
        vendor_name: str,
        *,
        as_of: date,
        analysis_window_days: int,
    ) -> dict[str, Any] | None:
        from ...autonomous.tasks import _b2b_shared as shared

        return await shared.read_vendor_intelligence(
            pool,
            vendor_name,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
        )

    async def read_vendor_scorecard_detail(
        self,
        pool: Any,
        vendor_name: str,
    ) -> dict[str, Any] | None:
        from ...autonomous.tasks import _b2b_shared as shared

        return await shared.read_vendor_scorecard_detail(pool, vendor_name)

    async def read_vendor_quote_evidence(
        self,
        pool: Any,
        *,
        vendor_name: str,
        window_days: int = 90,
        min_urgency: float = 5.0,
        limit: int = 10,
        sources: list[str] | None = None,
        pain_filter: str | None = None,
        require_quotes: bool = False,
        recency_column: str = "enriched_at",
    ) -> list[dict[str, Any]]:
        from ...autonomous.tasks import _b2b_shared as shared

        return await shared.read_vendor_quote_evidence(
            pool,
            vendor_name=vendor_name,
            window_days=window_days,
            min_urgency=min_urgency,
            limit=limit,
            sources=sources,
            pain_filter=pain_filter,
            require_quotes=require_quotes,
            recency_column=recency_column,
        )

    def normalize_openrouter_model(self, model: Any | None, *, context: str = "") -> str:
        from ...pipelines.llm import normalize_openrouter_model

        return normalize_openrouter_model(model, context=context)

    def clean_llm_output(self, text: str) -> str:
        from ...pipelines.llm import clean_llm_output

        return clean_llm_output(text)

    def get_campaign_llm(self) -> Any:
        from ...services.llm_router import get_llm

        return get_llm("campaign")

    def build_llm_messages(self, system_prompt: str, user_prompt: str) -> list[Any]:
        from ...services.protocols import Message

        return [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

    def prepare_b2b_exact_stage_request(self, stage_id: str, **kwargs: Any) -> Any:
        from ...services.b2b.cache_runner import prepare_b2b_exact_stage_request

        return prepare_b2b_exact_stage_request(stage_id, **kwargs)

    async def lookup_b2b_exact_stage_text(self, request: Any) -> Any | None:
        from ...services.b2b.cache_runner import lookup_b2b_exact_stage_text

        return await lookup_b2b_exact_stage_text(request)

    async def store_b2b_exact_stage_text(self, request: Any, **kwargs: Any) -> bool:
        from ...services.b2b.cache_runner import store_b2b_exact_stage_text

        return await store_b2b_exact_stage_text(request, **kwargs)

    def trace_llm_call(self, span_name: str, **kwargs: Any) -> None:
        from ...pipelines.llm import trace_llm_call

        trace_llm_call(span_name, **kwargs)


_configured_intelligence_port: VendorBriefingIntelligencePort | None = None
_configured_runtime_port: VendorBriefingRuntimePort | None = None
_bridge_intelligence_port = _BridgeVendorBriefingIntelligencePort()
_RUNTIME_METHODS = frozenset({
    "normalize_openrouter_model",
    "clean_llm_output",
    "get_campaign_llm",
    "build_llm_messages",
    "prepare_b2b_exact_stage_request",
    "lookup_b2b_exact_stage_text",
    "store_b2b_exact_stage_text",
    "trace_llm_call",
})


def configure_vendor_briefing_intelligence_port(
    port: VendorBriefingIntelligencePort | None,
) -> None:
    """Register the host adapter for vendor briefing intelligence support."""
    global _configured_intelligence_port
    _configured_intelligence_port = port


def configure_vendor_briefing_runtime_port(
    port: VendorBriefingRuntimePort | None,
) -> None:
    """Register the host adapter for vendor briefing LLM/cache support."""
    global _configured_runtime_port
    _configured_runtime_port = port


def get_vendor_briefing_intelligence_port() -> VendorBriefingIntelligencePort:
    """Return configured host support or the non-standalone bridge adapter."""
    if _configured_intelligence_port is not None:
        return _configured_intelligence_port
    if os.environ.get(STANDALONE_ENV_VAR) == "1":
        raise VendorBriefingIntelligencePortNotConfigured(
            "No vendor briefing intelligence port has been configured"
        )
    return _bridge_intelligence_port


def _has_vendor_briefing_runtime_methods(port: Any) -> bool:
    return all(
        callable(getattr(port, method_name, None))
        for method_name in _RUNTIME_METHODS
    )


def _get_vendor_briefing_runtime_port() -> VendorBriefingRuntimePort:
    if _configured_runtime_port is not None:
        return _configured_runtime_port
    if (
        _configured_intelligence_port is not None
        and _has_vendor_briefing_runtime_methods(_configured_intelligence_port)
    ):
        runtime_port: Any = _configured_intelligence_port
        return runtime_port
    return _bridge_intelligence_port


def reasoning_int(value: Any) -> int | None:
    return get_vendor_briefing_intelligence_port().reasoning_int(value)


def timing_summary_payload(
    timing_intelligence: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], list[str]]:
    return get_vendor_briefing_intelligence_port().timing_summary_payload(
        timing_intelligence
    )


def align_vendor_intelligence_record_to_scorecard(
    scorecard: dict[str, Any] | None,
    record: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return (
        get_vendor_briefing_intelligence_port()
        .align_vendor_intelligence_record_to_scorecard(scorecard, record)
    )


def inject_synthesis_freshness(
    entry: dict[str, Any],
    view: Any,
    *,
    requested_as_of: date | None = None,
) -> None:
    return get_vendor_briefing_intelligence_port().inject_synthesis_freshness(
        entry,
        view,
        requested_as_of=requested_as_of,
    )


def load_synthesis_view(
    raw: dict[str, Any],
    vendor_name: str,
    schema_version: str = "",
    as_of_date: date | str | None = None,
) -> Any:
    return get_vendor_briefing_intelligence_port().load_synthesis_view(
        raw,
        vendor_name,
        schema_version=schema_version,
        as_of_date=as_of_date,
    )


async def load_best_reasoning_view(
    pool: Any,
    vendor_name: str,
    *,
    as_of: date | None = None,
    analysis_window_days: int = 30,
) -> Any | None:
    return await get_vendor_briefing_intelligence_port().load_best_reasoning_view(
        pool,
        vendor_name,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def load_prior_reasoning_snapshots(
    pool: Any,
    vendor_names: list[str],
    *,
    before_date: date | None = None,
    analysis_window_days: int = 30,
) -> dict[str, dict[str, Any]]:
    return await get_vendor_briefing_intelligence_port().load_prior_reasoning_snapshots(
        pool,
        vendor_names,
        before_date=before_date,
        analysis_window_days=analysis_window_days,
    )


async def read_vendor_company_signal_review_queue(
    pool: Any,
    *,
    vendor_name: str,
    window_days: int | None = None,
    preview_limit: int | None = None,
) -> dict[str, Any]:
    return await get_vendor_briefing_intelligence_port().read_vendor_company_signal_review_queue(
        pool,
        vendor_name=vendor_name,
        window_days=window_days,
        preview_limit=preview_limit,
    )


async def read_vendor_intelligence_record(
    pool: Any,
    vendor_name: str,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    return await get_vendor_briefing_intelligence_port().read_vendor_intelligence_record(
        pool,
        vendor_name,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def read_vendor_intelligence(
    pool: Any,
    vendor_name: str,
    *,
    as_of: date,
    analysis_window_days: int,
) -> dict[str, Any] | None:
    return await get_vendor_briefing_intelligence_port().read_vendor_intelligence(
        pool,
        vendor_name,
        as_of=as_of,
        analysis_window_days=analysis_window_days,
    )


async def read_vendor_scorecard_detail(
    pool: Any,
    vendor_name: str,
) -> dict[str, Any] | None:
    return await get_vendor_briefing_intelligence_port().read_vendor_scorecard_detail(
        pool,
        vendor_name,
    )


async def read_vendor_quote_evidence(
    pool: Any,
    *,
    vendor_name: str,
    window_days: int = 90,
    min_urgency: float = 5.0,
    limit: int = 10,
    sources: list[str] | None = None,
    pain_filter: str | None = None,
    require_quotes: bool = False,
    recency_column: str = "enriched_at",
) -> list[dict[str, Any]]:
    return await get_vendor_briefing_intelligence_port().read_vendor_quote_evidence(
        pool,
        vendor_name=vendor_name,
        window_days=window_days,
        min_urgency=min_urgency,
        limit=limit,
        sources=sources,
        pain_filter=pain_filter,
        require_quotes=require_quotes,
        recency_column=recency_column,
    )


def normalize_openrouter_model(model: Any | None, *, context: str = "") -> str:
    return _get_vendor_briefing_runtime_port().normalize_openrouter_model(
        model,
        context=context,
    )


def clean_llm_output(text: str) -> str:
    return _get_vendor_briefing_runtime_port().clean_llm_output(text)


def get_campaign_llm() -> Any:
    return _get_vendor_briefing_runtime_port().get_campaign_llm()


def build_llm_messages(system_prompt: str, user_prompt: str) -> list[Any]:
    return _get_vendor_briefing_runtime_port().build_llm_messages(
        system_prompt,
        user_prompt,
    )


def prepare_b2b_exact_stage_request(stage_id: str, **kwargs: Any) -> Any:
    return _get_vendor_briefing_runtime_port().prepare_b2b_exact_stage_request(
        stage_id,
        **kwargs,
    )


async def lookup_b2b_exact_stage_text(request: Any) -> Any | None:
    return await _get_vendor_briefing_runtime_port().lookup_b2b_exact_stage_text(
        request,
    )


async def store_b2b_exact_stage_text(request: Any, **kwargs: Any) -> bool:
    return await _get_vendor_briefing_runtime_port().store_b2b_exact_stage_text(
        request,
        **kwargs,
    )


def trace_llm_call(span_name: str, **kwargs: Any) -> None:
    return _get_vendor_briefing_runtime_port().trace_llm_call(
        span_name,
        **kwargs,
    )


__all__ = [
    "STANDALONE_ENV_VAR",
    "VendorBriefingIntelligencePort",
    "VendorBriefingIntelligencePortNotConfigured",
    "VendorBriefingRuntimePort",
    "align_vendor_intelligence_record_to_scorecard",
    "build_llm_messages",
    "clean_llm_output",
    "configure_vendor_briefing_intelligence_port",
    "configure_vendor_briefing_runtime_port",
    "get_campaign_llm",
    "get_vendor_briefing_intelligence_port",
    "inject_synthesis_freshness",
    "lookup_b2b_exact_stage_text",
    "load_best_reasoning_view",
    "load_prior_reasoning_snapshots",
    "load_synthesis_view",
    "normalize_openrouter_model",
    "prepare_b2b_exact_stage_request",
    "read_vendor_company_signal_review_queue",
    "read_vendor_intelligence",
    "read_vendor_intelligence_record",
    "read_vendor_quote_evidence",
    "read_vendor_scorecard_detail",
    "reasoning_int",
    "store_b2b_exact_stage_text",
    "timing_summary_payload",
    "trace_llm_call",
]
