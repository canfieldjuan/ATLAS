"""Host integration port for competitive-set planning support."""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Protocol


STANDALONE_ENV_VAR = "EXTRACTED_COMP_INTEL_STANDALONE"


class CompetitiveSetReasoningPortNotConfigured(RuntimeError):
    """Raised when a host has not registered competitive-set reasoning support."""


class CompetitiveSetReasoningPort(Protocol):
    """Host operations used by competitive-set preview and planning helpers."""

    @property
    def schema_version(self) -> str:
        """Return the current vendor reasoning schema version."""

    async def fetch_all_pool_layers(
        self,
        pool: Any,
        *,
        as_of: date,
        analysis_window_days: int,
        vendor_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return evidence pool layers for the requested vendors."""

    async def read_vendor_scorecard_details(
        self,
        pool: Any,
        *,
        vendor_names: list[str],
    ) -> list[dict[str, Any]]:
        """Return scorecard detail rows for category fallback."""

    def compute_pool_hash(self, layers: dict[str, Any]) -> str:
        """Return the normalized evidence-pool hash."""

    def compute_pool_hash_legacy(self, layers: dict[str, Any]) -> str:
        """Return the legacy evidence-pool hash for transition compatibility."""

    def coerce_as_of_date(self, value: Any) -> date | None:
        """Coerce persisted as_of_date values to date objects."""

    def classify_vendor_reasoning_decision(
        self,
        *,
        vendor_name: str,
        today: date,
        evidence_hash: str,
        latest_row: dict[str, Any] | None,
        force: bool,
        max_stale_days: int,
        rerun_if_missing_packet_artifacts: bool,
        rerun_if_missing_reference_ids: bool,
        hash_matches_prior: bool,
    ) -> dict[str, Any]:
        """Classify whether one vendor should rerun reasoning."""


class _BridgeCompetitiveSetReasoningPort:
    @property
    def schema_version(self) -> str:
        from ...autonomous.tasks import b2b_reasoning_synthesis as synthesis_mod

        return str(synthesis_mod._SCHEMA_VERSION)

    async def fetch_all_pool_layers(
        self,
        pool: Any,
        *,
        as_of: date,
        analysis_window_days: int,
        vendor_names: list[str],
    ) -> dict[str, dict[str, Any]]:
        from ...autonomous.tasks._b2b_shared import fetch_all_pool_layers

        return await fetch_all_pool_layers(
            pool,
            as_of=as_of,
            analysis_window_days=analysis_window_days,
            vendor_names=vendor_names,
        )

    async def read_vendor_scorecard_details(
        self,
        pool: Any,
        *,
        vendor_names: list[str],
    ) -> list[dict[str, Any]]:
        from ...autonomous.tasks._b2b_shared import read_vendor_scorecard_details

        return await read_vendor_scorecard_details(pool, vendor_names=vendor_names)

    def compute_pool_hash(self, layers: dict[str, Any]) -> str:
        from ...autonomous.tasks import b2b_reasoning_synthesis as synthesis_mod

        return str(synthesis_mod._compute_pool_hash(layers))

    def compute_pool_hash_legacy(self, layers: dict[str, Any]) -> str:
        from ...autonomous.tasks import b2b_reasoning_synthesis as synthesis_mod

        return str(synthesis_mod._compute_pool_hash_legacy(layers))

    def coerce_as_of_date(self, value: Any) -> date | None:
        from ...autonomous.tasks import b2b_reasoning_synthesis as synthesis_mod

        return synthesis_mod._coerce_as_of_date(value)

    def classify_vendor_reasoning_decision(
        self,
        *,
        vendor_name: str,
        today: date,
        evidence_hash: str,
        latest_row: dict[str, Any] | None,
        force: bool,
        max_stale_days: int,
        rerun_if_missing_packet_artifacts: bool,
        rerun_if_missing_reference_ids: bool,
        hash_matches_prior: bool,
    ) -> dict[str, Any]:
        from ...autonomous.tasks import b2b_reasoning_synthesis as synthesis_mod

        return synthesis_mod._classify_vendor_reasoning_decision(
            vendor_name=vendor_name,
            today=today,
            evidence_hash=evidence_hash,
            latest_row=latest_row,
            force=force,
            max_stale_days=max_stale_days,
            rerun_if_missing_packet_artifacts=rerun_if_missing_packet_artifacts,
            rerun_if_missing_reference_ids=rerun_if_missing_reference_ids,
            hash_matches_prior=hash_matches_prior,
        )


_configured_reasoning_port: CompetitiveSetReasoningPort | None = None
_bridge_reasoning_port = _BridgeCompetitiveSetReasoningPort()


def configure_competitive_set_reasoning_port(
    port: CompetitiveSetReasoningPort | None,
) -> None:
    """Register the host adapter for competitive-set reasoning support."""
    global _configured_reasoning_port
    _configured_reasoning_port = port


def get_competitive_set_reasoning_port() -> CompetitiveSetReasoningPort:
    """Return configured host support or the non-standalone bridge adapter."""
    if _configured_reasoning_port is not None:
        return _configured_reasoning_port
    if os.environ.get(STANDALONE_ENV_VAR) == "1":
        raise CompetitiveSetReasoningPortNotConfigured(
            "No competitive-set reasoning port has been configured"
        )
    return _bridge_reasoning_port


__all__ = [
    "CompetitiveSetReasoningPort",
    "CompetitiveSetReasoningPortNotConfigured",
    "STANDALONE_ENV_VAR",
    "configure_competitive_set_reasoning_port",
    "get_competitive_set_reasoning_port",
]
