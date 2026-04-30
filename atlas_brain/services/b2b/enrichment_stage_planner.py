from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedStagePlan:
    stage_id: str
    provider: str
    model: str
    backend: str
    request: Any
    request_fingerprint: str
    work_fingerprint: str
    payload_json: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    workload: str


def stage_backend_name(*, batch_enabled: bool, provider: str) -> str:
    if batch_enabled:
        return "anthropic_batch"
    return "direct_openrouter" if provider == "openrouter" else "direct_vllm"


def build_tier1_stage_plan(
    *,
    row: dict[str, Any],
    payload_json: str,
    system_prompt: str,
    model: str,
    provider: str,
    batch_enabled: bool,
    run_id: str | None,
    prepare_stage_request: Any,
    max_tokens: int,
    guided_json: dict[str, Any] | None,
) -> PreparedStagePlan:
    request, request_fingerprint, work_fingerprint = prepare_stage_request(
        "b2b_enrichment.tier1",
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_content=payload_json,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
        guided_json=guided_json,
    )
    return PreparedStagePlan(
        stage_id="b2b_enrichment.tier1",
        provider=provider,
        model=str(model or ""),
        backend=stage_backend_name(batch_enabled=batch_enabled, provider=provider),
        request=request,
        request_fingerprint=str(request_fingerprint),
        work_fingerprint=str(work_fingerprint),
        payload_json=payload_json,
        messages=[
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": payload_json},
        ],
        metadata={"tier": 1, "workload": "anthropic_batch" if batch_enabled else "direct"},
        workload="anthropic_batch" if batch_enabled else "direct",
    )


def build_tier2_stage_plan(
    *,
    row: dict[str, Any],
    base_payload: dict[str, Any],
    tier1_result: dict[str, Any],
    system_prompt: str,
    model: str,
    provider: str,
    batch_enabled: bool,
    run_id: str | None,
    prepare_stage_request: Any,
    prompt_for_content_type: Any,
    max_tokens: int,
    workload: str,
) -> PreparedStagePlan:
    payload = dict(base_payload)
    payload["tier1_specific_complaints"] = tier1_result.get("specific_complaints", [])
    payload["tier1_quotable_phrases"] = tier1_result.get("quotable_phrases", [])
    payload_json = json.dumps(payload)
    resolved_prompt = prompt_for_content_type(system_prompt, payload.get("content_type"))
    request, request_fingerprint, work_fingerprint = prepare_stage_request(
        "b2b_enrichment.tier2",
        provider=provider,
        model=model,
        system_prompt=str(resolved_prompt or ""),
        user_content=payload_json,
        max_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return PreparedStagePlan(
        stage_id="b2b_enrichment.tier2",
        provider=provider,
        model=str(model or ""),
        backend=stage_backend_name(batch_enabled=batch_enabled, provider=provider),
        request=request,
        request_fingerprint=str(request_fingerprint),
        work_fingerprint=str(work_fingerprint),
        payload_json=payload_json,
        messages=[
            {"role": "system", "content": str(resolved_prompt or "")},
            {"role": "user", "content": payload_json},
        ],
        metadata={"tier": 2, "workload": workload},
        workload=str(workload or ""),
    )
