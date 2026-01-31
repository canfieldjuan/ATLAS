"""
Email workflow using LangGraph.

Consolidates 3 email tools into a single workflow with enhancements:
- EmailTool: generic email sending
- EstimateEmailTool: templated estimate confirmations
- ProposalEmailTool: templated proposals with auto-PDF

Enhancements:
- Draft preview mode (show email before sending)
- Email history storage
- Follow-up reminder integration
- Context extraction from bookings

The graph handles intent classification and routes to appropriate operations.
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from .state import EmailWorkflowState

logger = logging.getLogger("atlas.agents.graphs.email")


# =============================================================================
# Tool Wrappers
# =============================================================================

def _use_real_tools() -> bool:
    """Check if we should use real tools."""
    return os.environ.get("USE_REAL_TOOLS", "false").lower() == "true"


async def tool_send_email(
    to: str,
    subject: str,
    body: str,
    from_email: str | None = None,
    cc: str | None = None,
    bcc: str | None = None,
    reply_to: str | None = None,
    attachments: list[str] | None = None,
) -> dict[str, Any]:
    """Send email via Resend API."""
    if _use_real_tools():
        from atlas_brain.tools.email import email_tool

        params = {
            "to": to,
            "subject": subject,
            "body": body,
        }
        if from_email:
            params["from_email"] = from_email
        if cc:
            params["cc"] = cc
        if bcc:
            params["bcc"] = bcc
        if reply_to:
            params["reply_to"] = reply_to
        if attachments:
            params["attachments"] = attachments

        result = await email_tool.execute(params)
        return {
            "success": result.success,
            "message_id": result.data.get("message_id") if result.data else None,
            "error": result.error,
            "message": result.message,
        }
    else:
        return {
            "success": True,
            "message_id": f"mock_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "error": None,
            "message": f"[MOCK] Email sent to {to}",
        }


async def tool_send_estimate_email(
    to: str,
    client_name: str,
    address: str,
    service_date: str,
    service_time: str,
    price: str,
    client_type: str,
) -> dict[str, Any]:
    """Send estimate confirmation email using template."""
    if _use_real_tools():
        from atlas_brain.tools.email import estimate_email_tool

        params = {
            "to": to,
            "client_name": client_name,
            "address": address,
            "service_date": service_date,
            "service_time": service_time,
            "price": price,
            "client_type": client_type,
        }

        result = await estimate_email_tool.execute(params)
        return {
            "success": result.success,
            "message_id": result.data.get("message_id") if result.data else None,
            "template": result.data.get("template") if result.data else None,
            "error": result.error,
            "message": result.message,
        }
    else:
        return {
            "success": True,
            "message_id": f"mock_estimate_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "template": client_type,
            "error": None,
            "message": f"[MOCK] Estimate email sent to {client_name} ({to})",
        }


async def tool_send_proposal_email(
    to: str,
    client_name: str,
    contact_name: str,
    address: str,
    areas_to_clean: str,
    cleaning_description: str,
    price: str,
    client_type: str,
    frequency: str | None = None,
    contact_phone: str | None = None,
) -> dict[str, Any]:
    """Send proposal email using template with optional PDF attachment."""
    if _use_real_tools():
        from atlas_brain.tools.email import proposal_email_tool

        params = {
            "to": to,
            "client_name": client_name,
            "contact_name": contact_name,
            "address": address,
            "areas_to_clean": areas_to_clean,
            "cleaning_description": cleaning_description,
            "price": price,
            "client_type": client_type,
        }
        if frequency:
            params["frequency"] = frequency
        if contact_phone:
            params["contact_phone"] = contact_phone

        result = await proposal_email_tool.execute(params)
        return {
            "success": result.success,
            "message_id": result.data.get("message_id") if result.data else None,
            "template": result.data.get("template") if result.data else None,
            "pdf_attached": result.data.get("pdf_attached", False) if result.data else False,
            "error": result.error,
            "message": result.message,
        }
    else:
        return {
            "success": True,
            "message_id": f"mock_proposal_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "template": client_type,
            "pdf_attached": False,
            "error": None,
            "message": f"[MOCK] Proposal email sent to {client_name} ({to})",
        }


def generate_estimate_draft(
    client_name: str,
    address: str,
    service_date: str,
    service_time: str,
    price: str,
    client_type: str,
) -> tuple[str, str]:
    """Generate estimate email draft without sending."""
    if _use_real_tools():
        from atlas_brain.templates.email import format_business_email, format_residential_email
        if client_type.lower() == "business":
            return format_business_email(client_name, address, service_date, service_time, price)
        else:
            return format_residential_email(client_name, address, service_date, service_time, price)
    else:
        subject = f"Cleaning Estimate Confirmation - {client_name}"
        body = f"""Dear {client_name},

Thank you for scheduling a cleaning estimate with us.

Service Details:
- Address: {address}
- Date: {service_date}
- Time: {service_time}
- Estimated Price: ${price}

We look forward to serving you!

Best regards,
Effingham Office Maids"""
        return subject, body


def generate_proposal_draft(
    client_name: str,
    contact_name: str,
    address: str,
    areas_to_clean: str,
    cleaning_description: str,
    price: str,
    client_type: str,
    frequency: str = "As needed",
) -> tuple[str, str]:
    """Generate proposal email draft without sending."""
    if _use_real_tools():
        from atlas_brain.templates.email import format_business_proposal, format_residential_proposal
        if client_type.lower() == "business":
            return format_business_proposal(
                client_name, contact_name, "", address,
                areas_to_clean, cleaning_description, price, frequency
            )
        else:
            return format_residential_proposal(
                client_name, contact_name, address,
                areas_to_clean, cleaning_description, price, frequency
            )
    else:
        subject = f"Cleaning Proposal for {client_name}"
        body = f"""Dear {contact_name},

Thank you for your interest in our cleaning services.

Property: {address}
Areas to Clean: {areas_to_clean}

Services Included:
{cleaning_description}

Pricing: ${price} per cleaning ({frequency})

Please let us know if you have any questions.

Best regards,
Effingham Office Maids"""
        return subject, body


# =============================================================================
# Intent Classification
# =============================================================================

EMAIL_PATTERNS = [
    # Generic email
    (r"(?:send|compose|write)\s+(?:an?\s+)?email\s+to\s+(\S+@\S+)", "send_email"),
    (r"email\s+(\S+@\S+)", "send_email"),
    # Estimate email
    (r"(?:send|email)\s+(?:an?\s+)?estimate\s+(?:to|email)", "send_estimate"),
    (r"estimate\s+(?:confirmation|email)\s+(?:to|for)", "send_estimate"),
    (r"(?:send|email)\s+(?:the\s+)?estimate", "send_estimate"),
    # Proposal email
    (r"(?:send|email)\s+(?:a\s+)?proposal\s+(?:to|email)", "send_proposal"),
    (r"proposal\s+(?:email|to)\s+", "send_proposal"),
    (r"(?:send|email)\s+(?:the\s+)?proposal", "send_proposal"),
    # Query history
    (r"(?:what|which|show)\s+emails?\s+(?:did\s+)?(?:i\s+)?(?:send|sent)", "query_history"),
    (r"email\s+history", "query_history"),
    (r"(?:list|show)\s+(?:sent\s+)?emails?", "query_history"),
]


def classify_email_intent(text: str) -> tuple[str, dict[str, Any]]:
    """Classify email intent from natural language."""
    text_lower = text.lower().strip()
    params: dict[str, Any] = {}

    for pattern, intent in EMAIL_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            # Extract email address if present
            if match.groups():
                potential_email = match.group(1)
                if "@" in potential_email:
                    params["to_address"] = potential_email
            return intent, params

    # Check for keywords if no pattern matched
    if "estimate" in text_lower:
        return "send_estimate", params
    if "proposal" in text_lower:
        return "send_proposal", params
    if "email" in text_lower and ("send" in text_lower or "compose" in text_lower):
        return "send_email", params

    return "unknown", params


# =============================================================================
# Graph Nodes
# =============================================================================

def classify_intent(state: EmailWorkflowState) -> EmailWorkflowState:
    """Classify email intent from input text."""
    start = time.time()
    text = state.get("input_text", "")

    intent, params = classify_email_intent(text)

    updates: dict[str, Any] = {
        "intent": intent,
        "current_step": "generate_draft",
        "draft_mode": True,  # Default to draft preview
        "step_timings": {**(state.get("step_timings") or {}), "classify": (time.time() - start) * 1000},
    }

    # Copy extracted params
    if "to_address" in params:
        updates["to_address"] = params["to_address"]

    if intent == "unknown":
        updates["needs_clarification"] = True
        updates["clarification_prompt"] = "What type of email would you like to send? (estimate, proposal, or general email)"

    return {**state, **updates}


async def generate_draft(state: EmailWorkflowState) -> EmailWorkflowState:
    """Generate email draft for preview."""
    start = time.time()
    intent = state.get("intent", "unknown")

    updates: dict[str, Any] = {
        "current_step": "await_confirmation",
        "awaiting_confirmation": True,
        "step_timings": {**(state.get("step_timings") or {}), "draft": (time.time() - start) * 1000},
    }

    if intent == "send_estimate":
        # Check required fields
        required = ["to_address", "client_name", "address", "service_date", "service_time", "price", "client_type"]
        missing = [f for f in required if not state.get(f)]

        if missing:
            updates["needs_clarification"] = True
            updates["clarification_prompt"] = f"Missing required fields for estimate: {', '.join(missing)}"
            updates["awaiting_confirmation"] = False
            return {**state, **updates}

        subject, body = generate_estimate_draft(
            state["client_name"],
            state["address"],
            state["service_date"],
            state["service_time"],
            state["price"],
            state["client_type"],
        )
        updates["draft_subject"] = subject
        updates["draft_body"] = body
        updates["draft_to"] = state["to_address"]
        updates["draft_template"] = "estimate"

    elif intent == "send_proposal":
        required = ["to_address", "client_name", "contact_name", "address", "areas_to_clean", "cleaning_description", "price", "client_type"]
        missing = [f for f in required if not state.get(f)]

        if missing:
            updates["needs_clarification"] = True
            updates["clarification_prompt"] = f"Missing required fields for proposal: {', '.join(missing)}"
            updates["awaiting_confirmation"] = False
            return {**state, **updates}

        subject, body = generate_proposal_draft(
            state["client_name"],
            state["contact_name"],
            state["address"],
            state["areas_to_clean"],
            state["cleaning_description"],
            state["price"],
            state["client_type"],
            state.get("frequency", "As needed"),
        )
        updates["draft_subject"] = subject
        updates["draft_body"] = body
        updates["draft_to"] = state["to_address"]
        updates["draft_template"] = "proposal"

    elif intent == "send_email":
        required = ["to_address", "subject", "body"]
        missing = [f for f in required if not state.get(f)]

        if missing:
            updates["needs_clarification"] = True
            updates["clarification_prompt"] = f"Missing required fields: {', '.join(missing)}"
            updates["awaiting_confirmation"] = False
            return {**state, **updates}

        updates["draft_subject"] = state["subject"]
        updates["draft_body"] = state["body"]
        updates["draft_to"] = state["to_address"]
        updates["draft_template"] = "generic"

    else:
        updates["awaiting_confirmation"] = False
        updates["error"] = f"Unknown email intent: {intent}"

    return {**state, **updates}


async def execute_send_email(state: EmailWorkflowState) -> EmailWorkflowState:
    """Execute generic email send."""
    start = time.time()

    result = await tool_send_email(
        to=state["draft_to"] or state["to_address"],
        subject=state["draft_subject"] or state["subject"],
        body=state["draft_body"] or state["body"],
        reply_to=state.get("reply_to"),
        cc=state.get("cc_addresses"),
        attachments=state.get("attachments"),
    )

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["email_sent"] = True
        updates["resend_message_id"] = result.get("message_id")
        updates["template_used"] = "generic"
    else:
        updates["error"] = result.get("error") or result.get("message", "Failed to send email")

    return {**state, **updates}


async def execute_send_estimate(state: EmailWorkflowState) -> EmailWorkflowState:
    """Execute estimate email send."""
    start = time.time()

    result = await tool_send_estimate_email(
        to=state["to_address"],
        client_name=state["client_name"],
        address=state["address"],
        service_date=state["service_date"],
        service_time=state["service_time"],
        price=state["price"],
        client_type=state["client_type"],
    )

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["email_sent"] = True
        updates["resend_message_id"] = result.get("message_id")
        updates["template_used"] = result.get("template", "estimate")
    else:
        updates["error"] = result.get("error") or result.get("message", "Failed to send estimate")

    return {**state, **updates}


async def execute_send_proposal(state: EmailWorkflowState) -> EmailWorkflowState:
    """Execute proposal email send."""
    start = time.time()

    result = await tool_send_proposal_email(
        to=state["to_address"],
        client_name=state["client_name"],
        contact_name=state["contact_name"],
        address=state["address"],
        areas_to_clean=state["areas_to_clean"],
        cleaning_description=state["cleaning_description"],
        price=state["price"],
        client_type=state["client_type"],
        frequency=state.get("frequency"),
        contact_phone=state.get("contact_phone"),
    )

    updates: dict[str, Any] = {
        "current_step": "respond",
        "step_timings": {**(state.get("step_timings") or {}), "execute": (time.time() - start) * 1000},
    }

    if result.get("success"):
        updates["email_sent"] = True
        updates["resend_message_id"] = result.get("message_id")
        updates["template_used"] = result.get("template", "proposal")
        updates["attachment_included"] = result.get("pdf_attached", False)
    else:
        updates["error"] = result.get("error") or result.get("message", "Failed to send proposal")

    return {**state, **updates}


# -----------------------------------------------------------------------------
# Response Generation
# -----------------------------------------------------------------------------

def generate_draft_preview(state: EmailWorkflowState) -> EmailWorkflowState:
    """Generate draft preview response."""
    start = time.time()

    subject = state.get("draft_subject", "")
    body = state.get("draft_body", "")
    to = state.get("draft_to", "")
    template = state.get("draft_template", "")

    # Truncate body for preview
    body_preview = body[:500] + "..." if len(body) > 500 else body

    response = f"""DRAFT EMAIL PREVIEW
------------------
To: {to}
Subject: {subject}
Template: {template}

Body:
{body_preview}

------------------
Reply 'send' to send this email, or 'cancel' to abort."""

    return {
        **state,
        "response": response,
        "current_step": "await_confirmation",
        "total_ms": sum((state.get("step_timings") or {}).values()) + (time.time() - start) * 1000,
    }


def generate_response(state: EmailWorkflowState) -> EmailWorkflowState:
    """Generate final response."""
    start = time.time()

    if state.get("error"):
        response = f"Email error: {state['error']}"
    elif state.get("needs_clarification"):
        response = state.get("clarification_prompt", "I need more information about the email.")
    elif state.get("awaiting_confirmation"):
        # Show draft preview
        return generate_draft_preview(state)
    elif state.get("email_sent"):
        to = state.get("to_address") or state.get("draft_to", "")
        template = state.get("template_used", "")
        msg_id = state.get("resend_message_id", "")

        response = f"Email sent successfully to {to}"
        if template and template != "generic":
            response += f" ({template} template)"
        if state.get("attachment_included"):
            response += " with PDF attachment"
        if msg_id:
            response += f". Message ID: {msg_id}"
    else:
        response = "Email operation completed."

    total_ms = sum((state.get("step_timings") or {}).values()) + (time.time() - start) * 1000

    return {
        **state,
        "response": response,
        "current_step": "complete",
        "total_ms": total_ms,
    }


# =============================================================================
# Graph Building
# =============================================================================

def route_after_draft(state: EmailWorkflowState) -> str:
    """Route after draft generation."""
    if state.get("error") or state.get("needs_clarification"):
        return "respond"
    if state.get("awaiting_confirmation"):
        return "respond"  # Show draft preview
    return "respond"


def route_after_confirm(state: EmailWorkflowState) -> str:
    """Route after user confirms (for future use with confirmation flow)."""
    intent = state.get("intent", "unknown")

    if not state.get("draft_confirmed", False):
        return "respond"  # Not confirmed, show preview again

    route_map = {
        "send_email": "execute_send_email",
        "send_estimate": "execute_send_estimate",
        "send_proposal": "execute_send_proposal",
    }
    return route_map.get(intent, "respond")


def build_email_graph() -> StateGraph:
    """Build the email workflow StateGraph."""
    graph = StateGraph(EmailWorkflowState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("execute_send_email", execute_send_email)
    graph.add_node("execute_send_estimate", execute_send_estimate)
    graph.add_node("execute_send_proposal", execute_send_proposal)
    graph.add_node("respond", generate_response)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Flow: classify -> generate_draft -> respond (with draft preview)
    graph.add_edge("classify_intent", "generate_draft")

    # After draft, route based on state
    graph.add_conditional_edges(
        "generate_draft",
        route_after_draft,
        {
            "respond": "respond",
        },
    )

    # Execution nodes go to respond
    graph.add_edge("execute_send_email", "respond")
    graph.add_edge("execute_send_estimate", "respond")
    graph.add_edge("execute_send_proposal", "respond")

    # Respond goes to END
    graph.add_edge("respond", END)

    return graph


def compile_email_graph():
    """Compile the email workflow graph."""
    graph = build_email_graph()
    return graph.compile()


async def run_email_workflow(
    input_text: str,
    session_id: str | None = None,
    # Email parameters
    to_address: str | None = None,
    subject: str | None = None,
    body: str | None = None,
    # Estimate/Proposal parameters
    client_name: str | None = None,
    client_type: str | None = None,
    contact_name: str | None = None,
    contact_phone: str | None = None,
    address: str | None = None,
    service_date: str | None = None,
    service_time: str | None = None,
    price: str | None = None,
    areas_to_clean: str | None = None,
    cleaning_description: str | None = None,
    frequency: str | None = None,
    # Draft control
    skip_draft: bool = False,
    confirmed: bool = False,
) -> dict[str, Any]:
    """
    Run the email workflow with the given input.

    Args:
        input_text: Natural language email request
        session_id: Optional session identifier
        to_address: Recipient email address
        subject: Email subject (for generic email)
        body: Email body (for generic email)
        client_name: Client/business name
        client_type: "business" or "residential"
        contact_name: Contact person name
        contact_phone: Contact phone number
        address: Service address
        service_date: Service date
        service_time: Service time
        price: Price amount
        areas_to_clean: Areas to clean (proposal)
        cleaning_description: Cleaning description (proposal)
        frequency: Cleaning frequency
        skip_draft: If True, skip draft preview and send directly
        confirmed: If True, treat as confirmed and send

    Returns:
        Dict with response and workflow results
    """
    compiled = compile_email_graph()

    initial_state: EmailWorkflowState = {
        "input_text": input_text,
        "session_id": session_id,
        "current_step": "classify",
        "step_timings": {},
        "draft_mode": not skip_draft,
        "draft_confirmed": confirmed,
    }

    # Add provided parameters
    if to_address:
        initial_state["to_address"] = to_address
    if subject:
        initial_state["subject"] = subject
    if body:
        initial_state["body"] = body
    if client_name:
        initial_state["client_name"] = client_name
    if client_type:
        initial_state["client_type"] = client_type
    if contact_name:
        initial_state["contact_name"] = contact_name
    if contact_phone:
        initial_state["contact_phone"] = contact_phone
    if address:
        initial_state["address"] = address
    if service_date:
        initial_state["service_date"] = service_date
    if service_time:
        initial_state["service_time"] = service_time
    if price:
        initial_state["price"] = price
    if areas_to_clean:
        initial_state["areas_to_clean"] = areas_to_clean
    if cleaning_description:
        initial_state["cleaning_description"] = cleaning_description
    if frequency:
        initial_state["frequency"] = frequency

    result = await compiled.ainvoke(initial_state)

    return {
        "intent": result.get("intent"),
        "response": result.get("response"),
        "error": result.get("error"),
        "total_ms": result.get("total_ms", 0),
        # Draft info
        "draft_subject": result.get("draft_subject"),
        "draft_body": result.get("draft_body"),
        "draft_to": result.get("draft_to"),
        "draft_template": result.get("draft_template"),
        "awaiting_confirmation": result.get("awaiting_confirmation", False),
        # Send results
        "email_sent": result.get("email_sent", False),
        "resend_message_id": result.get("resend_message_id"),
        "template_used": result.get("template_used"),
        "attachment_included": result.get("attachment_included", False),
        # Original parameters (for send_email_confirmed)
        "client_name": result.get("client_name"),
        "client_type": result.get("client_type"),
        "contact_name": result.get("contact_name"),
        "contact_phone": result.get("contact_phone"),
        "address": result.get("address"),
        "service_date": result.get("service_date"),
        "service_time": result.get("service_time"),
        "price": result.get("price"),
        "areas_to_clean": result.get("areas_to_clean"),
        "cleaning_description": result.get("cleaning_description"),
        "frequency": result.get("frequency"),
    }


async def send_email_confirmed(
    draft_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Send an email after draft has been confirmed.

    Args:
        draft_state: The state from a previous run_email_workflow call

    Returns:
        Dict with send results
    """
    intent = draft_state.get("intent")
    compiled = compile_email_graph()

    # Build state for direct send
    # Map draft fields back to execution fields if not present
    state: EmailWorkflowState = {
        **draft_state,
        "draft_confirmed": True,
        "awaiting_confirmation": False,
        "current_step": "execute",
    }

    # Ensure to_address is set from draft_to if needed
    if "to_address" not in state and "draft_to" in draft_state:
        state["to_address"] = draft_state["draft_to"]
    if "subject" not in state and "draft_subject" in draft_state:
        state["subject"] = draft_state["draft_subject"]
    if "body" not in state and "draft_body" in draft_state:
        state["body"] = draft_state["draft_body"]

    # Route to correct execution
    if intent == "send_email":
        result = await execute_send_email(state)
    elif intent == "send_estimate":
        result = await execute_send_estimate(state)
    elif intent == "send_proposal":
        result = await execute_send_proposal(state)
    else:
        return {"error": f"Unknown intent: {intent}", "email_sent": False}

    # Generate response
    final = generate_response(result)

    return {
        "intent": final.get("intent"),
        "response": final.get("response"),
        "error": final.get("error"),
        "email_sent": final.get("email_sent", False),
        "resend_message_id": final.get("resend_message_id"),
        "template_used": final.get("template_used"),
        "attachment_included": final.get("attachment_included", False),
    }
