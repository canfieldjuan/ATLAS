"""Host integration port for vendor briefing intelligence readers."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Protocol


STANDALONE_ENV_VAR = "EXTRACTED_PIPELINE_STANDALONE"


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


_configured_intelligence_port: VendorBriefingIntelligencePort | None = None
_bridge_intelligence_port = _BridgeVendorBriefingIntelligencePort()


def configure_vendor_briefing_intelligence_port(
    port: VendorBriefingIntelligencePort | None,
) -> None:
    """Register the host adapter for vendor briefing intelligence support."""
    global _configured_intelligence_port
    _configured_intelligence_port = port


def get_vendor_briefing_intelligence_port() -> VendorBriefingIntelligencePort:
    """Return configured host support or the non-standalone bridge adapter."""
    if _configured_intelligence_port is not None:
        return _configured_intelligence_port
    if os.environ.get(STANDALONE_ENV_VAR) == "1":
        raise VendorBriefingIntelligencePortNotConfigured(
            "No vendor briefing intelligence port has been configured"
        )
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


# Pre-existing main drift: PR #225 ("Route vendor briefing synthesis
# reader through intelligence port") added a synthesis-reader port
# surface to atlas + the competitive_intelligence mirror but missed
# the content_pipeline mirror. b2b_vendor_briefing.py imports the
# four symbols below from this module; without these re-exports the
# package fails import-smoke. The underlying implementations already
# live in ``_b2b_synthesis_reader`` -- this is purely a missing
# re-export layer. A future content_pipeline-side port refactor can
# replace these with proper port methods.


def inject_synthesis_freshness(
    entry: dict[str, Any],
    view: Any,
    *,
    requested_as_of: date | None = None,
) -> None:
    """Apply synthesis freshness metadata to an output entry."""
    from ...autonomous.tasks._b2b_synthesis_reader import (
        inject_synthesis_freshness as _impl,
    )

    _impl(entry, view, requested_as_of=requested_as_of)


def load_synthesis_view(
    raw: dict[str, Any],
    vendor_name: str,
    schema_version: str = "",
    as_of_date: date | str | None = None,
) -> Any:
    """Build the synthesis view object for a vendor's evidence vault."""
    from ...autonomous.tasks._b2b_synthesis_reader import (
        load_synthesis_view as _impl,
    )

    return _impl(
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
    """Return the best available reasoning view for a vendor."""
    from ...autonomous.tasks._b2b_synthesis_reader import (
        load_best_reasoning_view as _impl,
    )

    return await _impl(
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
    """Return prior reasoning snapshots keyed by vendor name."""
    from ...autonomous.tasks._b2b_synthesis_reader import (
        load_prior_reasoning_snapshots as _impl,
    )

    return await _impl(
        pool,
        vendor_names,
        before_date=before_date,
        analysis_window_days=analysis_window_days,
    )


__all__ = [
    "STANDALONE_ENV_VAR",
    "VendorBriefingIntelligencePort",
    "VendorBriefingIntelligencePortNotConfigured",
    "align_vendor_intelligence_record_to_scorecard",
    "configure_vendor_briefing_intelligence_port",
    "get_vendor_briefing_intelligence_port",
    "inject_synthesis_freshness",
    "load_best_reasoning_view",
    "load_prior_reasoning_snapshots",
    "load_synthesis_view",
    "read_vendor_company_signal_review_queue",
    "read_vendor_intelligence",
    "read_vendor_intelligence_record",
    "read_vendor_quote_evidence",
    "read_vendor_scorecard_detail",
    "reasoning_int",
    "timing_summary_payload",
]
