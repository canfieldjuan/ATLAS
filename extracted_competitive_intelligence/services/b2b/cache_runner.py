"""Exact-cache bridge for extracted competitive intelligence."""

from __future__ import annotations

import importlib as _importlib
import os as _os
from dataclasses import dataclass
from typing import Any


def _bridge(module_name: str) -> None:
    src = _importlib.import_module(module_name)
    globals_dict = globals()
    for name in dir(src):
        if not name.startswith("__"):
            globals_dict[name] = getattr(src, name)


if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    _os.environ.setdefault("EXTRACTED_LLM_INFRA_STANDALONE", "1")

    from extracted_llm_infrastructure.services.b2b.cache_strategy import (
        require_b2b_cache_strategy,
    )
    from . import llm_exact_cache

    @dataclass(frozen=True)
    class ExactStageRequest:
        stage_id: str
        namespace: str
        provider: str
        model: str
        request_envelope: dict[str, Any]

    def _resolve_provider_model(
        *,
        llm: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[str, str]:
        resolved_provider = str(provider or "")
        resolved_model = str(model or "")
        if llm is not None and (not resolved_provider or not resolved_model):
            llm_provider, llm_model = llm_exact_cache.llm_identity(llm)
            resolved_provider = resolved_provider or llm_provider
            resolved_model = resolved_model or llm_model
        return resolved_provider, resolved_model

    def prepare_b2b_exact_stage_request(
        stage_id: str,
        *,
        llm: Any | None = None,
        provider: str | None = None,
        model: str | None = None,
        messages: Any,
        max_tokens: int | None,
        temperature: float | None,
        response_format: dict[str, Any] | None = None,
        guided_json: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ExactStageRequest:
        strategy = require_b2b_cache_strategy(stage_id, expected_mode="exact")
        if not strategy.namespace:
            raise ValueError(f"Stage '{stage_id}' is missing an exact-cache namespace")
        resolved_provider, resolved_model = _resolve_provider_model(
            llm=llm,
            provider=provider,
            model=model,
        )
        request_envelope = llm_exact_cache.build_request_envelope(
            provider=resolved_provider,
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            guided_json=guided_json,
            extra=extra,
        )
        return ExactStageRequest(
            stage_id=stage_id,
            namespace=str(strategy.namespace),
            provider=resolved_provider,
            model=resolved_model,
            request_envelope=request_envelope,
        )

    def bind_b2b_exact_stage_request(
        stage_id: str,
        *,
        provider: str,
        model: str,
        request_envelope: dict[str, Any],
    ) -> ExactStageRequest:
        strategy = require_b2b_cache_strategy(stage_id, expected_mode="exact")
        if not strategy.namespace:
            raise ValueError(f"Stage '{stage_id}' is missing an exact-cache namespace")
        return ExactStageRequest(
            stage_id=stage_id,
            namespace=str(strategy.namespace),
            provider=str(provider or ""),
            model=str(model or ""),
            request_envelope=request_envelope,
        )

    async def lookup_b2b_exact_stage_text(
        request: ExactStageRequest,
        *,
        pool: Any | None = None,
    ) -> llm_exact_cache.B2BLLMExactCacheHit | None:
        return await llm_exact_cache.lookup_cached_text(
            request.namespace,
            request.request_envelope,
            pool=pool,
        )

    async def store_b2b_exact_stage_text(
        request: ExactStageRequest,
        *,
        response_text: str,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pool: Any | None = None,
    ) -> bool:
        return await llm_exact_cache.store_cached_text(
            request.namespace,
            request.request_envelope,
            provider=request.provider,
            model=request.model,
            response_text=response_text,
            usage=usage,
            metadata=metadata,
            pool=pool,
        )

else:
    _bridge("atlas_brain.services.b2b.cache_runner")


del _bridge, _importlib, _os
