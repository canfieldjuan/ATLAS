"""Host factories for the Content Ops reasoning context provider.

PR #402 shipped the route-level ``reasoning_context_provider``
seam in ``extracted_content_pipeline/api/control_surfaces.py``
plus the bundle's per-request ``with_reasoning_context()``
derivation. This module is the host adapter the route mount
calls to obtain a configured provider (or ``None``).

Two providers are offered side-by-side:

* ``build_content_ops_reasoning_context_provider`` (PR #462) --
  file-backed reference adapter. Operators opt in by setting
  ``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
  readable by ``FileCampaignReasoningContextProvider``.
* ``build_postgres_content_ops_reasoning_context_provider`` --
  DB-backed adapter (this slice). Operators opt in by setting
  ``ATLAS_CONTENT_OPS_REASONING_DB_ENABLED=true``; the factory
  binds to the host's existing asyncpg pool.

``select_content_ops_reasoning_context_provider`` is the chooser
the route mount passes as the ``reasoning_context_provider``
kwarg: it tries DB first, falls back to file, returns ``None``
when neither is configured. Both factories preserve the
WARN-and-fall-back contract -- a misconfigured provider must
not crash the route mount or block the entire Content Ops
surface for all tenants.

``describe_content_ops_reasoning_context_provider`` mirrors that
same DB > file > none selection for the catalog route so operator
readiness reflects the provider that can actually be built, not
just whether the route was passed a chooser callback.

Lives at ``atlas_brain/`` root with underscore prefix to dodge
the heavy ``atlas_brain.services`` import chain (same pattern as
``_content_ops_infrastructure.py`` / ``_content_ops_scope.py``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


logger = logging.getLogger(__name__)


_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH"
_DB_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_DB_ENABLED"
_INTERVENTION_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_INTERVENTION_ENABLED"
_INTERVENTION_REPORT_TABLE = "intelligence_reports"
_INTERVENTION_REPORT_TYPE = "intervention"
_INTERVENTION_SUMMARY_LIMIT = 1200


def build_content_ops_reasoning_context_provider(
    *,
    path_factory: Callable[[], str | None] | None = None,
    loader_factory: Callable[[str], Any] | None = None,
) -> Any | None:
    """Return the configured reasoning context provider, or
    ``None`` when the host hasn't opted in.

    Hosts opt in by setting
    ``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
    readable by ``FileCampaignReasoningContextProvider``. Failures
    (missing file, parse errors) resolve to ``None`` with a
    warning logged -- the reasoning provider is enrichment, a
    bad file must not block the entire Content Ops surface for
    all tenants.

    DI kwargs let tests stub the env-var read and the loader
    without touching the filesystem or the heavy
    ``extracted_content_pipeline.campaign_reasoning_data`` module.
    """

    path = (path_factory or _read_env_path)()
    if not path:
        return None
    if not Path(path).is_file():
        logger.warning(
            "Content Ops reasoning context path %s does not exist; "
            "provider stays unwired.",
            path,
        )
        return None

    loader = loader_factory or _default_loader
    try:
        return loader(path)
    except Exception as exc:
        logger.warning(
            "Failed to load Content Ops reasoning context from %s: %s",
            path,
            exc,
        )
        return None


def _read_env_path() -> str | None:
    """Read the env var; return ``None`` for unset / empty."""

    value = os.environ.get(_ENV_VAR)
    return value or None


def _default_loader(path: str) -> Any:
    """Lazy import the package's file-backed loader.

    Keeps this module light enough to import in dependency-light
    dev envs; the heavier
    ``extracted_content_pipeline.campaign_reasoning_data`` only
    loads when the env var is actually set.
    """

    from extracted_content_pipeline.campaign_reasoning_data import (
        load_campaign_reasoning_context_provider,
    )

    return load_campaign_reasoning_context_provider(path)


def build_postgres_content_ops_reasoning_context_provider(
    *,
    enabled_factory: Callable[[], bool] | None = None,
    pool_factory: Callable[[], Any] | None = None,
    repository_factory: Callable[[Any], Any] | None = None,
) -> Any | None:
    """Return a Postgres-backed reasoning context provider, or
    ``None`` when the host hasn't opted in or the pool is not
    available yet.

    Hosts opt in by setting ``ATLAS_CONTENT_OPS_REASONING_DB_ENABLED``
    to a truthy value (``true`` / ``1`` / ``yes``). The factory
    pulls the host's existing asyncpg pool via
    ``atlas_brain.storage.database.get_db_pool`` and binds a
    ``PostgresCampaignReasoningContextRepository`` against it.

    A missing pool (DB not initialized yet, or the host runs
    without Postgres) resolves to ``None`` with a WARN log -- the
    route mount stays unwired rather than crashing.

    DI kwargs let tests stub the env-var read, the pool acquire,
    and the repository construction without touching the
    filesystem or the heavy ``extracted_content_pipeline``
    storage modules.
    """

    if not (enabled_factory or _read_db_enabled)():
        return None

    try:
        pool = (pool_factory or _default_pool_factory)()
    except Exception as exc:
        logger.warning(
            "Content Ops reasoning DB pool acquire failed: %s; "
            "DB-backed provider stays unwired.",
            exc,
        )
        return None
    if pool is None:
        logger.warning(
            "Content Ops reasoning DB enabled but pool is not "
            "available; DB-backed provider stays unwired.",
        )
        return None
    if not getattr(pool, "is_initialized", True):
        logger.warning(
            "Content Ops reasoning DB enabled but pool is not "
            "initialized yet (DB persistence disabled, or startup "
            "ordering); DB-backed provider stays unwired so the "
            "chooser can fall back to the file provider.",
        )
        return None

    factory = repository_factory or _default_repository_factory
    try:
        return factory(pool)
    except Exception as exc:
        logger.warning(
            "Failed to construct Content Ops reasoning DB "
            "repository: %s; DB-backed provider stays unwired.",
            exc,
        )
        return None


def build_intervention_content_ops_reasoning_context_provider(
    *,
    enabled_factory: Callable[[], bool] | None = None,
    pool_factory: Callable[[], Any] | None = None,
    provider_factory: Callable[[Any], Any] | None = None,
) -> Any | None:
    """Return an Atlas intervention-report reasoning provider.

    Hosts opt in by setting
    ``ATLAS_CONTENT_OPS_REASONING_INTERVENTION_ENABLED`` to a truthy
    value. The provider reads already-persisted
    ``intelligence_reports`` rows with ``report_type='intervention'``
    and normalizes them into the same campaign reasoning context port
    used by file and DB providers.

    A missing or uninitialized pool resolves to ``None`` with WARN so
    the chooser can fall back to the file provider. This factory does
    not run the intervention pipeline; it only consumes completed rows.
    """

    if not (enabled_factory or _read_intervention_enabled)():
        return None

    try:
        pool = (pool_factory or _default_pool_factory)()
    except Exception as exc:
        logger.warning(
            "Content Ops intervention reasoning pool acquire failed: %s; "
            "intervention provider stays unwired.",
            exc,
        )
        return None
    if pool is None:
        logger.warning(
            "Content Ops intervention reasoning enabled but pool is not "
            "available; intervention provider stays unwired.",
        )
        return None
    if not getattr(pool, "is_initialized", True):
        logger.warning(
            "Content Ops intervention reasoning enabled but pool is not "
            "initialized yet; intervention provider stays unwired.",
        )
        return None

    factory = provider_factory or InterventionReportCampaignReasoningProvider
    try:
        return factory(pool)
    except Exception as exc:
        logger.warning(
            "Failed to construct Content Ops intervention reasoning "
            "provider: %s; intervention provider stays unwired.",
            exc,
        )
        return None


class InterventionReportCampaignReasoningProvider:
    """Campaign reasoning provider backed by Atlas intervention reports."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def read_campaign_reasoning_context(
        self,
        *,
        scope: Any,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> Any | None:
        """Return normalized reasoning from the latest matching report."""

        del scope
        del target_mode
        selectors = _intervention_candidate_selectors(
            target_id=target_id,
            opportunity=opportunity,
        )
        if not selectors:
            return None

        row = await self.pool.fetchrow(
            f"""
            SELECT id, entity_name, entity_type, report_type,
                   time_window_days, report_text, structured_data,
                   pressure_snapshot, requested_by, created_at
            FROM {_INTERVENTION_REPORT_TABLE}
            WHERE report_type = $1
              AND lower(entity_name) = ANY($2::text[])
            ORDER BY created_at DESC
            LIMIT 1
            """,
            _INTERVENTION_REPORT_TYPE,
            list(selectors),
        )
        if row is None:
            return None
        context = _intervention_report_to_campaign_context(_row_to_dict(row))
        return context if context.has_content() else None


def select_content_ops_reasoning_context_provider(
    *,
    db_factory: Callable[[], Any | None] | None = None,
    intervention_factory: Callable[[], Any | None] | None = None,
    file_factory: Callable[[], Any | None] | None = None,
) -> Any | None:
    """Pick the configured reasoning provider (DB > intervention > file > None).

    The route mount passes this chooser as the
    ``reasoning_context_provider`` kwarg. DB takes precedence because
    explicit campaign context rows are tenant-scoped and product-owned.
    Intervention output is the Atlas-native automatic fallback. The
    file-backed adapter stays available as a single-tenant / staging
    fallback. ``None`` means no provider is configured -- the bundle's
    existing ``with_reasoning_context()`` derivation already handles
    that.

    Both factories own their own WARN-and-fall-back behavior so
    the chooser stays trivial -- failing factories return
    ``None`` and the chooser advances to the next.
    """

    db_pick = (db_factory or build_postgres_content_ops_reasoning_context_provider)()
    if db_pick is not None:
        return db_pick
    intervention_pick = (
        intervention_factory or build_intervention_content_ops_reasoning_context_provider
    )()
    if intervention_pick is not None:
        return intervention_pick
    return (file_factory or build_content_ops_reasoning_context_provider)()


def describe_content_ops_reasoning_context_provider(
    *,
    db_factory: Callable[[], Any | None] | None = None,
    intervention_factory: Callable[[], Any | None] | None = None,
    file_factory: Callable[[], Any | None] | None = None,
) -> dict[str, Any]:
    """Return the catalog-safe reasoning provider readiness.

    The execute path still calls ``select_content_ops_reasoning_context_provider``
    per request. This companion keeps GET /control-surfaces honest:
    a mounted chooser that resolves to ``None`` reports
    ``configured=False`` instead of looking ready.
    """

    db_pick = (db_factory or build_postgres_content_ops_reasoning_context_provider)()
    if db_pick is not None:
        return {"configured": True, "source": "db"}

    intervention_pick = (
        intervention_factory or build_intervention_content_ops_reasoning_context_provider
    )()
    if intervention_pick is not None:
        return {"configured": True, "source": "intervention"}

    file_pick = (file_factory or build_content_ops_reasoning_context_provider)()
    if file_pick is not None:
        return {"configured": True, "source": "file"}

    return {"configured": False, "source": "none"}


def _read_db_enabled() -> bool:
    """Parse the DB opt-in env var; default ``False`` when unset
    or unrecognized so a typo'd value resolves to "off"."""

    value = os.environ.get(_DB_ENV_VAR, "").strip().lower()
    return value in {"true", "1", "yes", "on"}


def _read_intervention_enabled() -> bool:
    """Parse the intervention opt-in env var."""

    value = os.environ.get(_INTERVENTION_ENV_VAR, "").strip().lower()
    return value in {"true", "1", "yes", "on"}


def _intervention_candidate_selectors(
    *,
    target_id: str,
    opportunity: Mapping[str, Any],
) -> tuple[str, ...]:
    values = [
        target_id,
        opportunity.get("target_id"),
        opportunity.get("id"),
        opportunity.get("company_name"),
        opportunity.get("company"),
        opportunity.get("account_name"),
        opportunity.get("account"),
        opportunity.get("vendor_name"),
        opportunity.get("vendor"),
    ]
    seen: set[str] = set()
    selectors: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        selectors.append(text)
    return tuple(selectors)


def _intervention_report_to_campaign_context(row: Mapping[str, Any]) -> Any:
    from extracted_content_pipeline.services.campaign_reasoning_context import (
        normalize_campaign_reasoning_context,
    )

    structured = _json_mapping(row.get("structured_data"))
    pressure = _json_mapping(row.get("pressure_snapshot"))
    report_text = str(row.get("report_text") or "").strip()
    entity_name = str(row.get("entity_name") or "").strip()
    pipeline_id = str(structured.get("pipeline_id") or "").strip()
    stages = _json_mapping(structured.get("stage_statuses"))
    safety_warnings = _json_list(structured.get("safety_warnings"))

    payload: dict[str, Any] = {
        "reasoning_context": {
            "wedge": "intervention",
            "summary": _truncate(report_text, _INTERVENTION_SUMMARY_LIMIT),
            "why_now": _intervention_why_now(row, safety_warnings),
            "recommended_action": "Use the latest intervention analysis to shape content.",
            "top_theses": _intervention_top_theses(entity_name, report_text),
            "proof_points": _intervention_proof_points(stages, pressure),
            "coverage_limits": safety_warnings,
            "reference_ids": _intervention_reference_ids(row, pipeline_id),
            "scope_summary": {
                "selection_strategy": "atlas_intervention_report",
                "entity_name": entity_name,
                "entity_type": row.get("entity_type"),
                "report_type": row.get("report_type"),
                "time_window_days": row.get("time_window_days"),
                "created_at": str(row.get("created_at") or ""),
                "requested_by": row.get("requested_by"),
                "stages_completed": structured.get("stages_completed"),
                "stages_total": structured.get("stages_total"),
            },
        }
    }
    return normalize_campaign_reasoning_context(payload)


def _intervention_top_theses(
    entity_name: str,
    report_text: str,
) -> list[dict[str, Any]]:
    if not report_text:
        return []
    return [{
        "claim": f"Latest intervention analysis for {entity_name or 'target'}",
        "summary": _truncate(report_text, _INTERVENTION_SUMMARY_LIMIT),
        "confidence": "medium",
    }]


def _intervention_proof_points(
    stages: Mapping[str, Any],
    pressure: Mapping[str, Any],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for stage, status in stages.items():
        if status in (None, "", [], {}):
            continue
        points.append({
            "label": f"{stage} status",
            "value": status,
            "interpretation": "Intervention pipeline stage state.",
        })
    score = pressure.get("pressure_score")
    if score not in (None, "", [], {}):
        points.append({
            "label": "pressure_score",
            "value": score,
            "interpretation": "Pressure snapshot at intervention generation time.",
        })
    return points


def _intervention_reference_ids(
    row: Mapping[str, Any],
    pipeline_id: str,
) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    report_id = str(row.get("id") or "").strip()
    if report_id:
        refs["intelligence_reports"] = [report_id]
    if pipeline_id:
        refs["intervention_pipeline"] = [pipeline_id]
    return refs


def _intervention_why_now(
    row: Mapping[str, Any],
    safety_warnings: Sequence[Any],
) -> str:
    if safety_warnings:
        return str(safety_warnings[0])
    created_at = str(row.get("created_at") or "").strip()
    if created_at:
        return f"Latest intervention report created at {created_at}."
    return "Latest intervention report matched this Content Ops target."


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if item not in (None, "", [], {})]
    return []


def _truncate(value: str, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _default_pool_factory() -> Any:
    """Lazy import the host's asyncpg pool getter.

    Keeps this module light enough to import in dependency-light
    dev envs; ``atlas_brain.storage.database`` only loads when
    the DB env var is actually set.
    """

    from atlas_brain.storage.database import get_db_pool

    return get_db_pool()


def _default_repository_factory(pool: Any) -> Any:
    """Lazy import the package's Postgres repository.

    Mirrors ``_default_loader``: the heavier
    ``extracted_content_pipeline.campaign_reasoning_postgres``
    module loads only when the DB env var is set + a pool is
    available.
    """

    from extracted_content_pipeline.campaign_reasoning_postgres import (
        PostgresCampaignReasoningContextRepository,
    )

    return PostgresCampaignReasoningContextRepository(pool=pool)


__all__ = [
    "InterventionReportCampaignReasoningProvider",
    "build_content_ops_reasoning_context_provider",
    "build_intervention_content_ops_reasoning_context_provider",
    "build_postgres_content_ops_reasoning_context_provider",
    "describe_content_ops_reasoning_context_provider",
    "select_content_ops_reasoning_context_provider",
]
