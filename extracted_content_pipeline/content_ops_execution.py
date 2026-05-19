"""Execute runnable AI Content Ops generation plans through host services."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .campaign_ports import TenantScope
from .control_surfaces import OUTPUT_CATALOG, ContentOpsRequest, request_from_mapping
from .generation_plan import GenerationPlan, GenerationPlanStep, build_generation_plan
from .landing_page_ports import MarketingCampaign
from .reasoning_signals import REASONING_VALIDATION_BLOCKED

logger = logging.getLogger(__name__)

_REASONING_OUTPUT_TO_ATTR: Mapping[str, str] = MappingProxyType({
    "email_campaign": "campaign",
    "blog_post": "blog_post",
    "report": "report",
    "landing_page": "landing_page",
    "sales_brief": "sales_brief",
})
_REASONING_AWARE_OUTPUTS = tuple(_REASONING_OUTPUT_TO_ATTR)


@dataclass(frozen=True)
class ContentOpsExecutionServices:
    """Host-provided services for runnable content-ops outputs.

    Services bundled here run **concurrently** when multiple outputs
    are requested in a single plan -- the executor uses
    ``asyncio.gather`` over plan steps. Each service must therefore be
    safe under concurrent ``generate()`` invocation. Pure functions
    and pool-backed services are fine; services with shared in-memory
    state need locking or per-call scoping. See
    ``execute_content_ops_request`` for the full concurrency contract.
    """

    campaign: Any | None = None
    blog_post: Any | None = None
    report: Any | None = None
    landing_page: Any | None = None
    sales_brief: Any | None = None
    signal_extraction: Any | None = None
    faq_markdown: Any | None = None
    reasoning_provider_configured: bool = False
    reasoning_provider_outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reasoning_provider_outputs and not self.reasoning_provider_configured:
            raise ValueError(
                "reasoning_provider_outputs requires reasoning_provider_configured=True"
            )

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
        if output == "signal_extraction":
            return self.signal_extraction
        if output == "faq_markdown":
            return self.faq_markdown
        return None

    def configured_outputs(self) -> tuple[str, ...]:
        outputs: list[str] = []
        for output in (
            "email_campaign",
            "blog_post",
            "report",
            "landing_page",
            "sales_brief",
            "signal_extraction",
            "faq_markdown",
        ):
            if _has_generate_method(self.for_output(output)):
                outputs.append(output)
        return tuple(outputs)

    def with_reasoning_context(
        self,
        provider: Any | None,
        outputs: Sequence[str] | None = None,
    ) -> "ContentOpsExecutionServices":
        """Return a derived bundle with each reasoning-aware service rebound.

        PR-ControlSurfaces-Reasoning-Provider: the /execute route resolves
        a host-supplied ``reasoning_context_provider`` per request and
        derives a fresh services bundle so the cached service instances
        in ``execution_services_provider`` are not mutated. Services that
        opt in expose ``with_reasoning_context(provider) -> Self``;
        services that don't (e.g. signal_extraction, which doesn't
        consume reasoning context) are passed through unchanged. When
        ``outputs`` is ``None``, all reasoning-aware services are rebound.
        Passing a sequence scopes rebinding to those output ids; an empty
        sequence rebinds none. Multiple targeted calls accumulate active
        output metadata. Passing ``provider=None`` for selected outputs clears
        those outputs from the active metadata while preserving other bindings.
        """

        if isinstance(outputs, str):
            raise TypeError("outputs must be a sequence of output ids, not a string")
        selected = frozenset(_REASONING_AWARE_OUTPUTS if outputs is None else outputs)
        unknown = selected - frozenset(_REASONING_AWARE_OUTPUTS)
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"unknown reasoning-aware outputs: {joined}")
        existing_outputs = (
            frozenset(self.reasoning_provider_outputs)
            if self.reasoning_provider_outputs
            else frozenset(_REASONING_AWARE_OUTPUTS)
            if self.reasoning_provider_configured
            else frozenset()
        )
        active_outputs = (
            existing_outputs | selected
            if provider is not None
            else existing_outputs - selected
        )
        active_tuple = (
            ()
            if active_outputs == frozenset(_REASONING_AWARE_OUTPUTS)
            else tuple(sorted(active_outputs))
        )
        services = {
            "campaign": self.campaign,
            "blog_post": self.blog_post,
            "report": self.report,
            "landing_page": self.landing_page,
            "sales_brief": self.sales_brief,
        }
        for output, attr in _REASONING_OUTPUT_TO_ATTR.items():
            if output in selected:
                services[attr] = _rebind_reasoning(services[attr], provider)
        return ContentOpsExecutionServices(
            campaign=services["campaign"],
            blog_post=services["blog_post"],
            report=services["report"],
            landing_page=services["landing_page"],
            sales_brief=services["sales_brief"],
            # signal_extraction stays as-is; it does not consume reasoning.
            signal_extraction=self.signal_extraction,
            # faq_markdown stays as-is; it does not consume reasoning.
            faq_markdown=self.faq_markdown,
            reasoning_provider_configured=bool(active_outputs),
            reasoning_provider_outputs=active_tuple,
        )

    def reasoning_provider_active_for(self, output: str) -> bool:
        if not self.reasoning_provider_configured:
            return False
        if not self.reasoning_provider_outputs:
            return True
        return output in self.reasoning_provider_outputs


def _rebind_reasoning(service: Any | None, provider: Any | None) -> Any | None:
    if service is None:
        return None
    helper = getattr(service, "with_reasoning_context", None)
    if helper is None:
        return service
    return helper(provider)


@dataclass(frozen=True)
class ContentOpsStepExecution:
    """Result for one attempted generation step."""

    output: str
    runner: str
    status: str
    result: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    reasoning: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "output": self.output,
            "runner": self.runner,
            "status": self.status,
            "result": dict(self.result),
            "error": self.error,
        }
        if self.reasoning:
            data["reasoning"] = dict(self.reasoning)
        return data


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
    """Execute a runnable content-ops plan using host-provided services.

    **Concurrency contract (PR-Audit-MINOR-Batch-3):** plan steps run
    concurrently via ``asyncio.gather``. Host-injected services may
    therefore see overlapping ``generate()`` calls. Hosts must inject
    services that are **safe under concurrent invocation** -- pure
    functions, services backed by a connection pool, or services with
    appropriate locks. A service that maintains in-memory state
    (counters, queues, caches without per-call scoping) will race
    silently. Hosts migrating from a single-step execution path
    should audit their service for shared state before adopting the
    control surface.
    """

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
            reasoning_provider_configured=services.reasoning_provider_active_for(
                step.output
            ),
        )
        for step in plan.steps
    ))
    executed = [step_result for step_result, _error in step_results]
    errors = [error for _step_result, error in step_results if error is not None]

    # Distinguish "every step failed" from "some failed." Pre-fix, all-failed
    # was reported as "partial," which misled operator dashboards into
    # assuming some steps succeeded.
    if not errors:
        status = "completed"
    elif plan.steps and len(errors) >= len(plan.steps):
        status = "failed"
    else:
        status = "partial"
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
    reasoning_provider_configured: bool,
) -> tuple[ContentOpsStepExecution, Mapping[str, Any] | None]:
    if not _has_generate_method(service):
        error = _step_error_dict(step, "service_not_configured")
        return (
            _failed_step(
                step,
                "service_not_configured",
                service=service,
                reasoning_provider_configured=reasoning_provider_configured,
            ),
            error,
        )
    try:
        result = await _run_step(
            step,
            request=request,
            service=service,
            scope=scope,
            filters=filters,
        )
    except Exception as exc:
        error = _step_error_dict(step, str(exc))
        return (
            _failed_step(
                step,
                str(exc),
                service=service,
                reasoning_provider_configured=reasoning_provider_configured,
            ),
            error,
        )
    result_dict = _result_dict(result)
    return (
        ContentOpsStepExecution(
            output=step.output,
            runner=step.runner,
            status="completed",
            result=result_dict,
            reasoning=_step_reasoning_audit(
                step,
                service,
                result=result_dict,
                reasoning_provider_configured=reasoning_provider_configured,
            ),
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
    """Dispatch a step to its per-output handler.

    PR-OptionA-1 refactor: replaces the previous "landing_page special-case
    + everything-else generic" branch with a per-output handler table so
    each output's `step.config` can be threaded into its service signature
    without growing more `if step.output == ...` branches. New outputs
    register a one-line handler entry instead of editing this function.
    """
    handler = _DISPATCH.get(step.output, _dispatch_default)
    return await handler(
        step=step,
        service=service,
        request=request,
        scope=scope,
        filters=filters,
    )


async def _dispatch_email_campaign(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "scope": scope,
        "target_mode": request.target_mode,
        "limit": request.limit,
        "filters": filters,
        "channels": _step_config_sequence(step.config, "channels"),
        "temperature": _step_config_float(step.config, "temperature"),
        "max_tokens": _step_config_int(step.config, "max_tokens"),
        "parse_retry_attempts": _step_config_int(step.config, "parse_retry_attempts"),
        "quality_revalidation_enabled": _step_config_bool(
            step.config, "quality_revalidation_enabled"
        ),
        "quality_prompt_proof_term_limit": _step_config_int(
            step.config, "quality_prompt_proof_term_limit"
        ),
        "parse_retry_response_excerpt_chars": _step_config_int(
            step.config, "parse_retry_response_excerpt_chars"
        ),
    }
    opportunity_defaults = _opportunity_defaults_from_inputs(request.inputs)
    if opportunity_defaults is not None:
        kwargs["opportunity_defaults"] = opportunity_defaults
    return await service.generate(**kwargs)


async def _dispatch_report(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        limit=request.limit,
        filters=filters,
        default_report_type=_step_config_text(step.config, "default_report_type"),
        temperature=_step_config_float(step.config, "temperature"),
        max_tokens=_step_config_int(step.config, "max_tokens"),
        parse_retry_attempts=_step_config_int(step.config, "parse_retry_attempts"),
        parse_retry_response_excerpt_chars=_step_config_int(
            step.config, "parse_retry_response_excerpt_chars"
        ),
        quality_gates_enabled=_step_config_bool(step.config, "quality_gates_enabled"),
    )


async def _dispatch_sales_brief(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        limit=request.limit,
        filters=filters,
        default_brief_type=_step_config_text(step.config, "default_brief_type"),
        temperature=_step_config_float(step.config, "temperature"),
        max_tokens=_step_config_int(step.config, "max_tokens"),
        parse_retry_attempts=_step_config_int(step.config, "parse_retry_attempts"),
        parse_retry_response_excerpt_chars=_step_config_int(
            step.config, "parse_retry_response_excerpt_chars"
        ),
        quality_gates_enabled=_step_config_bool(step.config, "quality_gates_enabled"),
    )


async def _dispatch_landing_page(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    del filters  # unused: landing pages take a campaign, not a target_mode
    return await service.generate(
        scope=scope,
        campaign=_marketing_campaign_from_inputs(request.inputs),
        temperature=_step_config_float(step.config, "temperature"),
        max_tokens=_step_config_int(step.config, "max_tokens"),
        parse_retry_attempts=_step_config_int(step.config, "parse_retry_attempts"),
        parse_retry_response_excerpt_chars=_step_config_int(
            step.config, "parse_retry_response_excerpt_chars"
        ),
        quality_gates_enabled=_step_config_bool(step.config, "quality_gates_enabled"),
    )


async def _dispatch_blog_post(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    # PR-OptionA-2: blog_post graduates from _dispatch_default into its own
    # handler so the LLM-tuning kwargs (temperature/max_tokens/
    # parse_retry_attempts) reach BlogPostGenerationService.generate. The
    # generic dispatcher remains for genuinely no-config future outputs.
    # PR-Blog-Topic-Per-Call: ``topic`` is now load-bearing too; the plan
    # emits ``step.config["topic"]`` from ``request.inputs.get("topic")``
    # and the service's prompt has a ``{topic}`` placeholder.
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        limit=request.limit,
        filters=filters,
        temperature=_step_config_float(step.config, "temperature"),
        max_tokens=_step_config_int(step.config, "max_tokens"),
        parse_retry_attempts=_step_config_int(step.config, "parse_retry_attempts"),
        parse_retry_response_excerpt_chars=_step_config_int(
            step.config, "parse_retry_response_excerpt_chars"
        ),
        quality_gates_enabled=_step_config_bool(step.config, "quality_gates_enabled"),
        topic=_step_config_text(step.config, "topic"),
    )


async def _dispatch_signal_extraction(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    del filters
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        source_material=request.inputs.get("source_material"),
        limit=request.limit,
        max_text_chars=_step_config_int(step.config, "max_text_chars"),
    )


async def _dispatch_faq_markdown(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    del filters
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        source_material=request.inputs.get("source_material"),
        title=_step_config_text(step.config, "title"),
        max_items=_step_config_int(step.config, "max_items"),
        max_evidence_per_item=_step_config_int(step.config, "max_evidence_per_item"),
        source_types=_step_config_sequence(step.config, "source_types"),
        max_text_chars=_step_config_int(step.config, "max_text_chars"),
    )


async def _dispatch_default(
    *,
    step: GenerationPlanStep,
    service: Any,
    request: ContentOpsRequest,
    scope: TenantScope,
    filters: Mapping[str, Any] | None,
) -> Any:
    """Fallback for outputs that don't yet thread step.config kwargs.

    Currently used by `blog_post`. Future outputs that need per-call
    config kwargs should register their own handler in `_DISPATCH`.
    """
    del step  # config is informational for outputs without per-call overrides
    return await service.generate(
        scope=scope,
        target_mode=request.target_mode,
        limit=request.limit,
        filters=filters,
    )


_DISPATCH: Mapping[str, Any] = {
    "email_campaign": _dispatch_email_campaign,
    "report": _dispatch_report,
    "sales_brief": _dispatch_sales_brief,
    "landing_page": _dispatch_landing_page,
    "blog_post": _dispatch_blog_post,
    "signal_extraction": _dispatch_signal_extraction,
    "faq_markdown": _dispatch_faq_markdown,
}

_MAX_REASONING_VALIDATION_FAILURES = 50


def _step_config_text(config: Mapping[str, Any], key: str) -> str | None:
    """Pull a non-empty string from step.config or return None."""
    raw = config.get(key) if isinstance(config, Mapping) else None
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _step_config_int(config: Mapping[str, Any], key: str) -> int | None:
    """Pull an int from step.config or return None.

    Used for ``max_tokens`` / ``parse_retry_attempts``. Returns None on
    missing key, non-numeric values, or booleans (Python's ``True/False``
    are technically ``int`` subclasses; we don't want a mis-typed bool to
    silently coerce to 0/1 here).
    """
    if not isinstance(config, Mapping):
        return None
    raw = config.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _step_config_float(config: Mapping[str, Any], key: str) -> float | None:
    """Pull a float from step.config or return None.

    Used for ``temperature``. Returns None on missing key, non-numeric
    values, or booleans (same rationale as ``_step_config_int``).
    """
    if not isinstance(config, Mapping):
        return None
    raw = config.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _step_config_bool(config: Mapping[str, Any], key: str) -> bool | None:
    """Pull a bool from step.config or return None.

    Used for ``quality_revalidation_enabled``. Returns None on missing
    key or non-bool values; we deliberately avoid coercing strings like
    ``"yes"`` / ``"true"`` because a mis-typed step.config entry should
    fall through to the service's config default rather than silently
    landing the wrong value.
    """
    if not isinstance(config, Mapping):
        return None
    raw = config.get(key)
    if isinstance(raw, bool):
        return raw
    return None


def _step_config_sequence(
    config: Mapping[str, Any],
    key: str,
) -> tuple[str, ...] | None:
    """Pull a non-empty string sequence from step.config or return None.

    Accepts either a list/tuple or a comma-separated string. Empty values
    return None so the service falls through to its construction-time
    default rather than coercing to an empty channel set.
    """
    if not isinstance(config, Mapping):
        return None
    raw = config.get(key)
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        return None
    return tuple(items) if items else None


def _filters_from_inputs(inputs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    filters = inputs.get("filters")
    return filters if isinstance(filters, Mapping) else None


def _opportunity_defaults_from_inputs(inputs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    defaults: dict[str, Any] = {}
    selling = inputs.get("selling")
    selling_defaults = dict(selling) if isinstance(selling, Mapping) else {}
    booking_url = (
        _clean(inputs.get("booking_url"))
        or _clean(inputs.get("selling_booking_url"))
    )
    if booking_url and not _clean(selling_defaults.get("booking_url")):
        selling_defaults["booking_url"] = booking_url
    selling_defaults = {
        str(key): value
        for key, value in selling_defaults.items()
        if str(key).strip() and value not in (None, "", [], {})
    }
    if selling_defaults:
        defaults["selling"] = selling_defaults
    return defaults or None


# PR-OptionA-4: explicit allowlist of MarketingCampaign.context fields.
# Prior shape was a "negative-list inversion" -- everything not on a small
# excluded set leaked into context. That meant standard control-surface
# inputs (target_account, opportunity_id, channels, report_type, brief_type,
# filters, ...) all flowed into the landing-page service's campaign payload
# as if they were intentional context. Audit MAJOR.
#
# The allowlist is "fields the LLM is allowed to see in campaign.context,"
# not "fields the prompt consumes by name." The packaged landing-page
# prompt uses ``{campaign_json}`` as a generic JSON dump, so the LLM sees
# whatever is in context whether or not the prompt names it. The list
# starts conservative (domain-context fields hosts have asked for) and
# grows by explicit additions; unrelated request inputs no longer leak.
#
# Backwards-compat note: hosts whose custom landing-page prompts reference
# context fields outside this allowlist will see empty values post-fix.
# Add the field here to restore visibility -- see
# ``plans/PR-OptionA-4.md`` Migration section.
_MARKETING_CAMPAIGN_CONTEXT_FIELDS: frozenset[str] = frozenset({
    "industry",
    "pain_points",
    "differentiators",
    "customer_segments",
    "key_metrics",
    "proof_points",
    "competitive_alternatives",
})


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
            if key in _MARKETING_CAMPAIGN_CONTEXT_FIELDS
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


def _step_error_dict(step: GenerationPlanStep, error: str) -> dict[str, Any]:
    """Build the result-level error dict for a failed step.

    Aligns the shape with ``ContentOpsStepExecution`` (output, runner,
    error). ``reason`` is preserved as a backwards-compat alias for
    hosts that key on the older field name; future cleanup can drop it
    once host migration completes.
    """
    return {
        "output": step.output,
        "runner": step.runner,
        "error": error,
        "reason": error,
    }


def _failed_step(
    step: GenerationPlanStep,
    error: str,
    *,
    service: Any | None,
    reasoning_provider_configured: bool,
) -> ContentOpsStepExecution:
    return ContentOpsStepExecution(
        output=step.output,
        runner=step.runner,
        status="failed",
        error=error,
        reasoning=_step_reasoning_audit(
            step,
            service,
            reasoning_provider_configured=reasoning_provider_configured,
        ),
    )


def _has_generate_method(service: Any | None) -> bool:
    return callable(getattr(service, "generate", None))


def _step_reasoning_audit(
    step: GenerationPlanStep,
    service: Any | None,
    *,
    result: Mapping[str, Any] | None = None,
    reasoning_provider_configured: bool,
) -> dict[str, Any]:
    definition = OUTPUT_CATALOG.get(step.output)
    requirement = (
        definition.reasoning_requirement
        if definition is not None
        else "absent"
    )
    service_supports = callable(getattr(service, "with_reasoning_context", None))
    if (
        requirement == "absent"
        and not service_supports
        and not reasoning_provider_configured
    ):
        return {}
    audit = {
        "requirement": requirement,
        "service_supports_reasoning": service_supports,
        "provider_configured": reasoning_provider_configured,
    }
    contexts_used = _reasoning_contexts_used(result)
    if contexts_used is not None:
        audit["contexts_used"] = contexts_used
    consumed_contexts = _consumed_reasoning_contexts(result)
    if consumed_contexts:
        audit["consumed_contexts"] = consumed_contexts
    validation_failures, validation_failures_truncated = (
        _reasoning_validation_failures(result)
    )
    if validation_failures:
        audit["validation_blocked"] = True
        audit["validation_failures"] = validation_failures
        if validation_failures_truncated:
            audit["validation_failures_truncated"] = True
        logger.warning(
            "content_ops_strict_validation_blocked",
            extra={
                "output": step.output,
                "failure_count": len(validation_failures),
                "truncated": validation_failures_truncated,
            },
        )
    return audit


def _reasoning_contexts_used(result: Mapping[str, Any] | None) -> int | None:
    if result is None:
        return None
    raw = result.get("reasoning_contexts_used")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int) and raw >= 0:
        return raw
    return None


def _consumed_reasoning_contexts(
    result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if result is None:
        return []
    raw = result.get("consumed_reasoning_contexts")
    if isinstance(raw, Mapping):
        return [dict(raw)] if raw else []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    contexts: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping) and item:
            contexts.append(dict(item))
    return contexts


def _reasoning_validation_failures(
    result: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], bool]:
    if result is None:
        return [], False
    raw_errors = result.get("errors")
    if isinstance(raw_errors, Mapping):
        candidates = (raw_errors,)
    elif isinstance(raw_errors, Sequence) and not isinstance(
        raw_errors, (str, bytes, bytearray)
    ):
        candidates = tuple(item for item in raw_errors if isinstance(item, Mapping))
    else:
        return [], False
    failures: list[dict[str, Any]] = []
    truncated = False
    for item in candidates:
        reason = _clean(item.get("reason") or item.get("error"))
        if reason != REASONING_VALIDATION_BLOCKED and not reason.startswith(
            f"{REASONING_VALIDATION_BLOCKED}:"
        ):
            continue
        if len(failures) >= _MAX_REASONING_VALIDATION_FAILURES:
            truncated = True
            break
        failure: dict[str, Any] = {"reason": REASONING_VALIDATION_BLOCKED}
        target_id = _clean(item.get("target_id"))
        if target_id:
            failure["target_id"] = target_id
        blockers = _reasoning_validation_blockers(reason)
        if blockers:
            failure["blockers"] = blockers
        failures.append(failure)
    return failures, truncated


def _reasoning_validation_blockers(reason: str) -> list[str]:
    suffix = reason.removeprefix(REASONING_VALIDATION_BLOCKED)
    if not suffix.startswith(":"):
        return []
    return [item.strip() for item in suffix[1:].split(",") if item.strip()]


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "ContentOpsExecutionResult",
    "ContentOpsExecutionServices",
    "ContentOpsStepExecution",
    "execute_content_ops_from_mapping",
    "execute_content_ops_request",
]
