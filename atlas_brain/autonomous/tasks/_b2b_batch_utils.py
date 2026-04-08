"""Shared helpers for B2B Anthropic batch enablement."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("atlas.autonomous.tasks._b2b_batch_utils")


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
    return anthropic_model_name(getattr(value, "model", "") or getattr(value, "model_id", "")) is not None


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
