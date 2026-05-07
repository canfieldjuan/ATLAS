"""Execute runnable AI Content Ops generation plans through host services."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .campaign_ports import TenantScope
from .control_surfaces import ContentOpsRequest, request_from_mapping
from .generation_plan import GenerationPlan, GenerationPlanStep, build_generation_plan
from .landing_page_ports import MarketingCampaign


@dataclass(frozen=True)
class ContentOpsExecutionServices:
    """Host-provided services for runnable content-ops outputs."""

    campaign: Any | None = None
    blog_post: Any | None = None
    report: Any | None = None
    landing_page: Any | None = None
    sales_brief: Any | None = None

    def for_output(self, output: str) -> Any | None:
        if output == "email_campaign":
            return self.campaign
        if output == "blog_post":
            return self.blog_post
        if output == "report":
            return self.report
        if output == "landing_page":
            return self.landing_page
        if output == "sales_brief":
            return self.sales_brief
        return None

    def configured_outputs(self) -> tuple[str, ...]:
        outputs: list[str] = []
        for output in (
            "email_campaign",
            "blog_post",
            "report",
            "landing_page",
            "sales_brief",
        ):
            if self.for_output(output) is not None:
                outputs.append(output)
        return tuple(outputs)


@dataclass(frozen=True)
class ContentOpsStepExecution:
    """Result for one attempted generation step."""

    output: str
    runner: str
    status: str
    result: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "runner": self.runner,
            "status": self.status,
            "result": dict(self.result),
            "error": self.error,
        }


@dataclass(frozen=True)
class ContentOpsExecutionResult:
    """Execution result for a content-ops request."""

    status: str
    plan: GenerationPlan
    steps: tuple[ContentOpsStepExecution, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "plan": self.plan.as_dict(),
            "steps": [step.as_dict() for step in self.steps],
            "errors": [dict(error) for error in self.errors],
        }


async def execute_content_ops_request(
    request: ContentOpsRequest,
    *,
    services: ContentOpsExecutionServices,
    scope: TenantScope | None = None,
) -> ContentOpsExecutionResult:
    """Execute a runnable content-ops plan using host-provided services."""

    plan = build_generation_plan(request)
    if not plan.can_execute:
        return ContentOpsExecutionResult(
            status="blocked",
            plan=plan,
            errors=({"reason": "plan_not_executable"},),
        )

    resolved_scope = scope or TenantScope()
    filters = _filters_from_inputs(request.inputs)
    step_results = await asyncio.gather(*(
        _execute_step(
            step,
            request=request,
            service=services.for_output(step.output),
            scope=resolved_scope,
            filters=filters,
        )
        for step in plan.steps
    ))
    executed = [step_result for step_result, _error in step_results]
    errors = [error for _step_result, error in step_results if error is not None]

    status = "completed" if not errors else "partial"
    return ContentOpsExecutionResult(
        status=status,
        plan=plan,
        steps=tuple(executed),
        errors=tuple(errors),
    )


async def _execute_step(
    step: GenerationPlanStep,
    *,
    request: ContentOpsRequest,
    service: Any | None,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> tuple[ContentOpsStepExecution, Mapping[str, Any] | None]:
    if service is None:
        error = {"output": step.output, "reason": "service_not_configured"}
        return _failed_step(step, "service_not_configured"), error
    try:
        result = await _run_step(
            step,
            request=request,
            service=service,
            scope=scope,
            filters=filters,
        )
    except Exception as exc:
        error = {"output": step.output, "reason": str(exc)}
        return _failed_step(step, str(exc)), error
    return (
        ContentOpsStepExecution(
            output=step.output,
            runner=step.runner,
            status="completed",
            result=_result_dict(result),
        ),
        None,
    )


async def execute_content_ops_from_mapping(
    payload: Mapping[str, Any],
    *,
    services: ContentOpsExecutionServices,
    scope: TenantScope | None = None,
) -> dict[str, Any]:
    """Dict-friendly execution wrapper for host API routes."""

    result = await execute_content_ops_request(
        request_from_mapping(payload),
        services=services,
        scope=scope,
    )
    return result.as_dict()


async def _run_step(
    step: GenerationPlanStep,
    *,
    request: ContentOpsRequest,
    service: Any,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    if step.output == "landing_page":
        return await service.generate(
            scope=scope,
            campaign=_marketing_campaign_from_inputs(request.inputs),
        )
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        limit=request.limit,
        filters=filters,
    )


def _filters_from_inputs(inputs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    filters = inputs.get("filters")
    return filters if isinstance(filters, Mapping) else None


def _marketing_campaign_from_inputs(inputs: Mapping[str, Any]) -> MarketingCampaign:
    offer = _clean(inputs.get("offer"))
    audience = _clean(inputs.get("audience"))
    name = _clean(inputs.get("campaign_name")) or offer or _clean(inputs.get("target_account"))
    return MarketingCampaign(
        name=name,
        persona=audience,
        value_prop=offer,
        vendors=_string_tuple(inputs.get("vendors")),
        categories=_string_tuple(inputs.get("categories")),
        tags=_string_tuple(inputs.get("tags")),
        context={
            str(key): value
            for key, value in inputs.items()
            if key not in {"campaign_name", "offer", "audience", "vendors", "categories", "tags"}
        },
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _result_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_dict"):
        data = result.as_dict()
        return dict(data) if isinstance(data, Mapping) else {"value": data}
    if isinstance(result, Mapping):
        return dict(result)
    return {"value": result}


def _failed_step(step: GenerationPlanStep, error: str) -> ContentOpsStepExecution:
    return ContentOpsStepExecution(
        output=step.output,
        runner=step.runner,
        status="failed",
        error=error,
    )


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "ContentOpsExecutionResult",
    "ContentOpsExecutionServices",
    "ContentOpsStepExecution",
    "execute_content_ops_from_mapping",
    "execute_content_ops_request",
]
