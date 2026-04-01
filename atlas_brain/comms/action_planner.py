"""
Post-call action planner.

Takes a call transcript + full CustomerContext, feeds them to an LLM
with the action_planning skill, and returns a structured action plan.

The plan is stored on the call_transcripts row (proposed_actions) and
sent via ntfy for user approval.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ..config import settings
from ..services.customer_context import CustomerContext
from ..skills import get_skill_registry

logger = logging.getLogger("atlas.comms.action_planner")


def _format_customer_context(ctx: CustomerContext) -> str:
    """Format CustomerContext into a text block for the LLM prompt."""
    if ctx.is_empty:
        return "No prior customer history available."

    parts = []

    # Contact info
    c = ctx.contact
    parts.append(f"Name: {c.get('full_name', 'Unknown')}")
    if c.get("phone"):
        parts.append(f"Phone: {c['phone']}")
    if c.get("email"):
        parts.append(f"Email: {c['email']}")
    if c.get("address"):
        parts.append(f"Address: {c['address']}")
    if c.get("contact_type"):
        parts.append(f"Type: {c['contact_type']}")
    if c.get("tags"):
        parts.append(f"Tags: {', '.join(c['tags'])}")
    if c.get("notes"):
        parts.append(f"Notes: {c['notes']}")

    # Past interactions
    if ctx.interactions:
        parts.append(f"\nInteraction history ({len(ctx.interactions)} recent):")
        for ix in ctx.interactions[:5]:
            date = ix.get("occurred_at", "")
            if isinstance(date, datetime):
                date = date.strftime("%Y-%m-%d")
            parts.append(f"  - [{ix.get('interaction_type', '?')}] {date}: {ix.get('summary', '')}")

    # Past appointments
    if ctx.appointments:
        parts.append(f"\nAppointments ({len(ctx.appointments)}):")
        for appt in ctx.appointments[:5]:
            start = appt.get("start_time", "")
            if isinstance(start, datetime):
                start = start.strftime("%Y-%m-%d %H:%M")
            status = appt.get("status", "")
            service = appt.get("service_type", "")
            parts.append(f"  - {start} | {service} | {status}")

    # Past calls
    if ctx.call_transcripts:
        parts.append(f"\nPrevious calls ({len(ctx.call_transcripts)}):")
        for call in ctx.call_transcripts[:3]:
            date = call.get("created_at", "")
            if isinstance(date, datetime):
                date = date.strftime("%Y-%m-%d")
            summary = call.get("summary", "")
            parts.append(f"  - {date}: {summary[:120]}")

    # Sent emails
    if ctx.sent_emails:
        parts.append(f"\nSent emails ({len(ctx.sent_emails)}):")
        for em in ctx.sent_emails[:3]:
            subj = em.get("subject", "")
            sent = em.get("sent_at", "")
            if isinstance(sent, datetime):
                sent = sent.strftime("%Y-%m-%d")
            parts.append(f"  - {sent}: {subj}")

    # SMS history
    if ctx.sms_messages:
        parts.append(f"\nSMS history ({len(ctx.sms_messages)}):")
        for sms in ctx.sms_messages[:5]:
            date = sms.get("created_at", "")
            if isinstance(date, datetime):
                date = date.strftime("%Y-%m-%d %H:%M")
            direction = sms.get("direction", "?")
            body_preview = (sms.get("body") or "")[:80]
            parts.append(f"  - [{direction}] {date}: {body_preview}")

    # Invoices
    if ctx.invoices:
        outstanding = [i for i in ctx.invoices if i.get("status") in ("sent", "partial", "overdue")]
        outstanding_total = sum(float(i.get("amount_due", 0)) for i in outstanding)
        overdue = [i for i in outstanding if i.get("status") == "overdue"]
        parts.append(f"\nInvoices ({len(ctx.invoices)} total, {len(outstanding)} outstanding = ${outstanding_total:.2f}, {len(overdue)} overdue):")
        for inv in ctx.invoices[:5]:
            inv_date = inv.get("issue_date", "")
            parts.append(
                f"  - {inv.get('invoice_number', '?')} | ${inv.get('total_amount', 0):.2f} | {inv.get('status', '?')} | {inv_date}"
            )

    # B2B competitive intelligence
    if ctx.b2b_churn_signals:
        parts.append(f"\nB2B Competitive Intelligence ({len(ctx.b2b_churn_signals)} vendor{'s' if len(ctx.b2b_churn_signals) != 1 else ''}):")
        for sig in ctx.b2b_churn_signals:
            vendor = sig.get("vendor_name", "Unknown")
            category = sig.get("product_category", "")
            urgency = sig.get("avg_urgency_score", 0)
            pains = sig.get("top_pain_categories", [])
            top_pain = pains[0].get("category", "") if pains else ""
            competitors = sig.get("top_competitors", [])
            top_threat = competitors[0].get("name", "") if competitors else ""
            line = f"  - {vendor}"
            if category:
                line += f" ({category})"
            line += f": urgency {urgency:.1f}/10"
            if top_pain:
                line += f", pain: {top_pain}"
            if top_threat:
                line += f", top threat: {top_threat}"
            parts.append(line)

    return "\n".join(parts)


async def generate_action_plan(
    transcript_id: UUID,
    call_summary: str,
    extracted_data: dict,
    customer_context: CustomerContext,
    business_context=None,
) -> list[dict]:
    """Generate a structured action plan using LLM + CustomerContext.

    Returns a list of action dicts: [{action, priority, params, rationale}]
    """
    from ..services.protocols import Message
    from ..services.llm_router import get_triage_llm
    from ..services import llm_registry

    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No LLM available for action planning")
        return _fallback_plan(extracted_data)

    # Build business context string
    ctx_parts = []
    if business_context:
        if hasattr(business_context, "name"):
            ctx_parts.append(f"Business: {business_context.name}")
        if hasattr(business_context, "business_type") and business_context.business_type:
            ctx_parts.append(f"Type: {business_context.business_type}")
        if hasattr(business_context, "services") and business_context.services:
            ctx_parts.append(f"Services: {', '.join(business_context.services)}")
    business_context_str = "\n".join(ctx_parts) if ctx_parts else "General business"

    customer_context_str = _format_customer_context(customer_context)
    extracted_str = json.dumps(extracted_data, separators=(",", ":"), default=str)

    # Load skill prompt
    skill = get_skill_registry().get("call/action_planning")
    if skill:
        system_prompt = (
            skill.content
            .replace("{business_context}", business_context_str)
            .replace("{customer_context}", customer_context_str)
            .replace("{call_summary}", call_summary or "No summary available")
            .replace("{extracted_data}", extracted_str)
        )
    else:
        system_prompt = (
            f"You are an AI assistant planning follow-up actions after a customer call.\n"
            f"Business: {business_context_str}\n"
            f"Customer: {customer_context_str}\n"
            f"Call summary: {call_summary}\n"
            f"Extracted data: {extracted_str}\n"
            f"Return a JSON array of actions: [{{action, priority, params, rationale}}]"
        )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content="Generate the action plan for this call."),
    ]

    loop = asyncio.get_running_loop()
    result = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: llm.chat(
                messages=messages,
                max_tokens=settings.call_intelligence.llm_max_tokens,
                temperature=settings.call_intelligence.llm_temperature,
            ),
        ),
        timeout=settings.call_intelligence.llm_timeout,
    )

    _usage = result.get("usage", {})
    if _usage.get("input_tokens"):
        logger.info("action_planner LLM tokens: in=%d out=%d",
                     _usage["input_tokens"], _usage.get("output_tokens", 0))
        from ..pipelines.llm import trace_llm_call
        _trace = result.get("_trace_meta", {})
        _customer_name = extracted_data.get("customer_name") or (
            customer_context.contact.get("full_name") if not customer_context.is_empty else None
        )
        trace_llm_call(
            "comms.action_planner",
            input_tokens=_usage["input_tokens"],
            output_tokens=_usage.get("output_tokens", 0),
            cached_tokens=_trace.get("cached_tokens") or _trace.get("cache_read_tokens"),
            cache_write_tokens=_trace.get("cache_write_tokens") or _trace.get("cache_creation_tokens"),
            billable_input_tokens=_trace.get("billable_input_tokens"),
            model=getattr(llm, "model", ""),
            provider=getattr(llm, "name", ""),
            input_data={
                "messages": [
                    {"role": m.role, "content": m.content[:500]} for m in messages
                ],
            },
            output_data={
                "response": result.get("response", "")[:2000],
            },
            api_endpoint=_trace.get("api_endpoint"),
            provider_request_id=_trace.get("provider_request_id"),
            ttft_ms=_trace.get("ttft_ms"),
            inference_time_ms=_trace.get("inference_time_ms"),
            queue_time_ms=_trace.get("queue_time_ms"),
            metadata={
                "transcript_id": str(transcript_id),
                "customer_name": _customer_name,
                "intent": extracted_data.get("intent"),
                "business_name": getattr(business_context, "name", None) if business_context else None,
                "pipeline": "action_planner",
            },
        )
    text = result.get("response", "").strip()
    if not text:
        return _fallback_plan(extracted_data)

    # Strip <think> tags (Qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return _parse_plan(text, extracted_data)


def _parse_plan(text: str, extracted_data: dict) -> list[dict]:
    """Parse the LLM output into a list of action dicts."""
    # Find JSON array
    arr_start = text.find("[")
    arr_end = text.rfind("]")

    if arr_start >= 0 and arr_end > arr_start:
        try:
            actions = json.loads(text[arr_start : arr_end + 1])
            if isinstance(actions, list):
                # Validate each action has required fields
                valid = []
                for a in actions:
                    if isinstance(a, dict) and a.get("action"):
                        valid.append({
                            "action": a["action"],
                            "priority": a.get("priority", 99),
                            "params": a.get("params", {}),
                            "rationale": a.get("rationale", ""),
                        })
                if valid:
                    valid.sort(key=lambda x: x["priority"])
                    return valid
        except json.JSONDecodeError:
            logger.warning("Failed to parse action plan JSON")

    return _fallback_plan(extracted_data)


def _fallback_plan(extracted_data: dict) -> list[dict]:
    """Build a basic plan from extracted_data when LLM is unavailable or fails."""
    actions = []
    intent = extracted_data.get("intent", "")

    if intent in ("estimate_request", "booking", "create_appointment"):
        actions.append({
            "action": "book_appointment",
            "priority": 1,
            "params": {
                "customer_name": extracted_data.get("customer_name"),
                "date": extracted_data.get("preferred_date"),
                "time": extracted_data.get("preferred_time"),
                "service": ", ".join(extracted_data.get("services_mentioned") or []),
                "address": extracted_data.get("address"),
            },
            "rationale": f"Customer intent: {intent}",
        })

    if extracted_data.get("customer_email"):
        actions.append({
            "action": "send_email",
            "priority": len(actions) + 1,
            "params": {
                "to": extracted_data["customer_email"],
                "type": "confirmation",
            },
            "rationale": "Email address available; send confirmation",
        })

    phone = extracted_data.get("customer_phone")
    if phone:
        actions.append({
            "action": "send_sms",
            "priority": len(actions) + 1,
            "params": {
                "to": phone,
                "type": "confirmation",
            },
            "rationale": "Phone number available; send SMS confirmation",
        })

    if not actions:
        actions.append({
            "action": "none",
            "priority": 1,
            "params": {},
            "rationale": "No actionable follow-up identified",
        })

    return actions
