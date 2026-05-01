"""Shared helpers for B2B Anthropic batch enablement."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks._b2b_batch_utils")

_ANTHROPIC_BATCH_SUCCESS_STATUSES = {
    "cache_hit",
    "batch_succeeded",
    "fallback_succeeded",
}


def exact_stage_request_fingerprint(request: Any) -> str:
    payload = {
        "namespace": str(getattr(request, "namespace", "") or ""),
        "provider": str(getattr(request, "provider", "") or ""),
        "model": str(getattr(request, "model", "") or ""),
        "request_envelope": getattr(request, "request_envelope", None) or {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stored_request_fingerprint(row: Any) -> str | None:
    metadata = row.get("request_metadata") if hasattr(row, "get") else None
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("request_fingerprint")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def task_metadata(task: Any) -> dict[str, Any]:
    metadata = getattr(task, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def metadata_bool(metadata: dict[str, Any], keys: tuple[str, ...], default: bool) -> bool:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        logger.warning("Ignoring invalid %s=%r on task metadata", key, value)
    return bool(default)


def metadata_int(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
    default: int,
    *,
    min_value: int = 1,
) -> int:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        try:
            if value is not None:
                return max(min_value, int(value))
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s=%r on task metadata", key, value)
    return max(min_value, int(default))


def anthropic_batch_requested(
    task: Any,
    *,
    global_default: bool,
    task_default: bool,
    global_keys: tuple[str, ...] = ("anthropic_batch_enabled",),
    task_keys: tuple[str, ...] = (),
) -> bool:
    metadata = task_metadata(task)
    global_enabled = metadata_bool(metadata, global_keys, global_default)
    task_enabled = metadata_bool(metadata, task_keys, task_default)
    return bool(global_enabled and task_enabled)


def anthropic_batch_min_items(
    task: Any,
    *,
    default: int,
    keys: tuple[str, ...],
    min_value: int = 1,
) -> int:
    return metadata_int(task_metadata(task), keys, default, min_value=min_value)


def anthropic_model_name(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("anthropic/"):
        text = text.split("/", 1)[1].strip()
    if text.startswith("claude"):
        return text
    return None


def is_anthropic_llm(value: Any) -> bool:
    provider = str(getattr(value, "name", "") or "").strip().lower()
    if provider == "anthropic":
        return True
    if provider:
        return False
    model = getattr(value, "model", "") or getattr(value, "model_id", "")
    if anthropic_model_name(model) is not None:
        return True
    class_name = type(value).__name__.strip().lower()
    module_name = str(getattr(type(value), "__module__", "") or "").strip().lower()
    return "anthropic" in class_name or module_name.endswith(".anthropic") or ".anthropic." in module_name


def anthropic_model_candidates(*values: Any, current_llm: Any | None = None) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        model = anthropic_model_name(value)
        if model and model not in seen:
            seen.add(model)
            candidates.append(model)

    if current_llm is not None:
        provider = str(getattr(current_llm, "name", "") or "").strip().lower()
        model = str(getattr(current_llm, "model", "") or "").strip()
        if provider in {"anthropic", "openrouter"}:
            _add(model)
    for value in values:
        _add(value)
    return candidates


def resolve_anthropic_batch_llm(
    *,
    current_llm: Any | None = None,
    target_model_candidates: tuple[Any, ...] = (),
):
    from ...config import settings
    from ...pipelines.llm import get_pipeline_llm
    from ...services import llm_registry

    candidates = anthropic_model_candidates(*target_model_candidates, current_llm=current_llm)

    if is_anthropic_llm(current_llm):
        current_model = anthropic_model_name(getattr(current_llm, "model", ""))
        if not candidates or current_model in candidates:
            return current_llm

    batch_llm = get_pipeline_llm(workload="anthropic")
    if is_anthropic_llm(batch_llm):
        current_model = anthropic_model_name(getattr(batch_llm, "model", ""))
        if not candidates or current_model in candidates:
            return batch_llm

    target_model = candidates[0] if candidates else None
    if not target_model:
        return batch_llm if is_anthropic_llm(batch_llm) else None

    api_key = (
        settings.llm.anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ATLAS_LLM_ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        logger.warning(
            "Anthropic batch requested for %s but no Anthropic API key is configured",
            target_model,
        )
        return batch_llm if is_anthropic_llm(batch_llm) else None

    slot_name = f"b2b_batch_anthropic::{target_model}"
    existing_slot = llm_registry.get_slot(slot_name)
    if is_anthropic_llm(existing_slot):
        existing_model = anthropic_model_name(getattr(existing_slot, "model", ""))
        if existing_model == target_model:
            return existing_slot

    try:
        return llm_registry.activate_slot(
            slot_name,
            "anthropic",
            model=target_model,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(
            "Failed to activate Anthropic batch slot %s for model %s: %s",
            slot_name,
            target_model,
            exc,
        )
    return batch_llm if is_anthropic_llm(batch_llm) else None


async def reconcile_existing_batch_artifacts(
    *,
    pool: Any,
    llm: Any | None,
    task_name: str,
    artifact_type: str,
    artifact_ids: list[str],
    expected_request_fingerprints: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reconcile matching Anthropic batch rows and return the best-known item state.

    This lets a rerun reuse already-completed provider work and avoid duplicating
    requests that are still in flight after a process restart.
    """
    if (
        not artifact_ids
        or pool is None
        or not getattr(pool, "is_initialized", False)
        or not hasattr(pool, "fetch")
    ):
        return {}

    artifact_ids = [str(value) for value in artifact_ids if str(value or "").strip()]
    if not artifact_ids:
        return {}

    fingerprint_map = None
    if expected_request_fingerprints is not None:
        fingerprint_map = {
            str(key): str(value).strip()
            for key, value in expected_request_fingerprints.items()
            if str(value or "").strip()
        }

    rows = await pool.fetch(
        """
        SELECT
            i.*,
            b.status AS batch_status,
            b.provider_batch_id,
            b.created_at AS batch_created_at
        FROM anthropic_message_batch_items i
        JOIN anthropic_message_batches b ON b.id = i.batch_id
        WHERE b.task_name = $1
          AND i.artifact_type = $2
          AND i.artifact_id = ANY($3::text[])
        ORDER BY i.artifact_id ASC, b.created_at DESC, i.created_at DESC
        """,
        task_name,
        artifact_type,
        artifact_ids,
    )
    if not rows:
        return {}

    batch_ids_to_reconcile = {
        str(row["batch_id"])
        for row in rows
        if str(row.get("batch_status") or "") == "in_progress"
        and str(row.get("provider_batch_id") or "").strip()
    }
    if batch_ids_to_reconcile and llm is not None:
        from ...services.b2b.anthropic_batch import reconcile_anthropic_message_batch

        for batch_id in sorted(batch_ids_to_reconcile):
            try:
                await reconcile_anthropic_message_batch(
                    llm=llm,
                    local_batch_id=batch_id,
                    pool=pool,
                )
            except Exception:
                logger.exception(
                    "Failed reconciling existing Anthropic batch %s for %s/%s",
                    batch_id,
                    task_name,
                    artifact_type,
                )
        rows = await pool.fetch(
            """
            SELECT
                i.*,
                b.status AS batch_status,
                b.provider_batch_id,
                b.created_at AS batch_created_at
            FROM anthropic_message_batch_items i
            JOIN anthropic_message_batches b ON b.id = i.batch_id
            WHERE b.task_name = $1
              AND i.artifact_type = $2
              AND i.artifact_id = ANY($3::text[])
            ORDER BY i.artifact_id ASC, b.created_at DESC, i.created_at DESC
            """,
            task_name,
            artifact_type,
            artifact_ids,
        )

    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row["artifact_id"]), []).append(row)

    selected: dict[str, dict[str, Any]] = {}
    for artifact_id, artifact_rows in grouped.items():
        if fingerprint_map is not None:
            expected_fingerprint = fingerprint_map.get(artifact_id)
            if not expected_fingerprint:
                continue
            artifact_rows = [
                row for row in artifact_rows
                if _stored_request_fingerprint(row) == expected_fingerprint
            ]
            if not artifact_rows:
                continue

        success_row = next(
            (
                row
                for row in artifact_rows
                if str(row.get("status") or "") in _ANTHROPIC_BATCH_SUCCESS_STATUSES
                and str(row.get("response_text") or "").strip()
            ),
            None,
        )
        if success_row is not None:
            status = str(success_row.get("status") or "")
            selected[artifact_id] = {
                "state": "succeeded",
                "status": status,
                "cached": status == "cache_hit",
                "response_text": str(success_row.get("response_text") or ""),
                "error_text": str(success_row.get("error_text") or "") or None,
                "batch_id": str(success_row["batch_id"]),
                "custom_id": str(success_row["custom_id"]),
            }
            continue

        pending_row = next(
            (
                row
                for row in artifact_rows
                if str(row.get("status") or "") == "pending"
                and str(row.get("provider_batch_id") or "").strip()
            ),
            None,
        )
        if pending_row is not None:
            selected[artifact_id] = {
                "state": "pending",
                "status": str(pending_row.get("status") or ""),
                "cached": False,
                "response_text": None,
                "error_text": str(pending_row.get("error_text") or "") or None,
                "batch_id": str(pending_row["batch_id"]),
                "custom_id": str(pending_row["custom_id"]),
            }

    return selected
