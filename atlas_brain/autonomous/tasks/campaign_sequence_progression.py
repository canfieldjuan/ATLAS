"""
Campaign sequence progression task.

Runs hourly (configurable).  For each active sequence whose
next_step_after has passed, gathers engagement data and previous
email history, generates the next step via LLM, queues it for
auto-send, and notifies via ntfy.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

import httpx

from ...config import settings
from ...storage.database import get_db_pool
from ...storage.models import ScheduledTask
from .campaign_audit import log_campaign_event

logger = logging.getLogger("atlas.autonomous.tasks.campaign_sequence_progression")


def _build_engagement_summary(seq: dict) -> str:
    """Build a human-readable engagement summary for the LLM prompt."""
    parts: list[str] = []

    opens = seq.get("open_count", 0)
    clicks = seq.get("click_count", 0)

    if opens > 0:
        parts.append(f"Opened {opens} time(s)")
        if seq.get("last_opened_at"):
            parts.append(f"Last opened: {seq['last_opened_at'].isoformat()}")
    else:
        parts.append("No opens recorded")

    if clicks > 0:
        parts.append(f"Clicked {clicks} time(s)")
        if seq.get("last_clicked_at"):
            parts.append(f"Last clicked: {seq['last_clicked_at'].isoformat()}")
    else:
        parts.append("No clicks recorded")

    if seq.get("reply_received_at"):
        intent = seq.get("reply_intent", "unknown")
        summary = seq.get("reply_summary", "")
        parts.append(f"Reply received ({intent}): {summary[:200]}")

    return "\n".join(parts)


def _build_previous_emails(campaigns: list[dict]) -> str:
    """Format previous campaign emails for the LLM prompt."""
    if not campaigns:
        return "No previous emails sent."

    parts: list[str] = []
    for c in campaigns:
        step = c.get("step_number", "?")
        subject = c.get("subject", "(no subject)")
        body = (c.get("body") or "")[:500]
        status = c.get("status", "")
        sent_at = c.get("sent_at", "")
        parts.append(
            f"--- Step {step} (status: {status}, sent: {sent_at}) ---\n"
            f"Subject: {subject}\n"
            f"{body}\n"
        )
    return "\n".join(parts)


async def _generate_next_step(
    seq: dict,
    previous_campaigns: list[dict],
) -> dict | None:
    """Generate the next email in the sequence via LLM.

    Returns {subject, body, cta, angle_reasoning} or None on failure.
    """
    from ...skills import get_skill_registry
    from ...services.llm_router import get_triage_llm
    from ...services import llm_registry
    from ...services.protocols import Message

    llm = get_triage_llm() or llm_registry.get_active()
    if not llm:
        logger.warning("No LLM available for sequence progression")
        return None

    # Parse context from JSONB (needed early for skill selection)
    company_context = seq.get("company_context") or {}
    selling_context = seq.get("selling_context") or {}
    if isinstance(company_context, str):
        company_context = json.loads(company_context)
    if isinstance(selling_context, str):
        selling_context = json.loads(selling_context)

    # Select skill based on sequence type
    is_seller_seq = company_context.get("recipient_type") == "amazon_seller"
    skill_name = (
        "digest/amazon_seller_campaign_sequence"
        if is_seller_seq
        else "digest/b2b_campaign_sequence"
    )
    skill = get_skill_registry().get(skill_name)
    if not skill:
        logger.error("Skill %s not found", skill_name)
        return None

    # Days since last email
    days_since = "N/A"
    if seq.get("last_sent_at"):
        delta = datetime.now(timezone.utc) - seq["last_sent_at"]
        days_since = str(delta.days)

    engagement = _build_engagement_summary(seq)
    prev_emails = _build_previous_emails(previous_campaigns)

    # Build template replacements (shared + skill-specific)
    replacements = {
        "{company_name}": seq.get("company_name", ""),
        "{company_context}": json.dumps(company_context, indent=2, default=str),
        "{selling_context}": json.dumps(selling_context, indent=2, default=str),
        "{current_step}": str(seq.get("current_step", 1) + 1),
        "{max_steps}": str(seq.get("max_steps", 4)),
        "{days_since_last}": days_since,
        "{engagement_summary}": engagement,
        "{previous_emails}": prev_emails,
    }

    # Extra placeholders for Amazon seller sequence skill
    if is_seller_seq:
        cat_intel = company_context.get("category_intelligence", {})
        replacements["{recipient_name}"] = company_context.get("seller_name", "")
        replacements["{recipient_company}"] = company_context.get("seller_name", "")
        replacements["{recipient_type}"] = "amazon_seller"
        replacements["{category}"] = company_context.get("category", "")
        replacements["{category_intelligence}"] = json.dumps(cat_intel, indent=2, default=str)

    system_prompt = skill.content
    for placeholder, value in replacements.items():
        system_prompt = system_prompt.replace(placeholder, value)

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content="Generate the next email in this sequence."),
    ]

    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda msgs=messages: llm.chat(
                    messages=msgs,
                    max_tokens=1024,
                    temperature=0.7,
                ),
            ),
            timeout=60,
        )

        text = result.get("response", "").strip()
        if not text:
            return None

        # Strip <think> tags (Qwen3)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

        # Try parsing the whole text as JSON first (most reliable)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "subject" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: find outermost { ... } block (handles nested braces)
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict) and "subject" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return None

    except Exception as exc:
        logger.warning("LLM generation failed for sequence %s: %s", seq.get("id"), exc)
        return None


async def _send_ntfy_notification(
    company_name: str,
    step_number: int,
    subject: str,
    sequence_id: str,
    campaign_id: str,
) -> None:
    """Send ntfy notification about queued campaign step."""
    if not settings.alerts.ntfy_enabled:
        return

    ntfy_url = f"{settings.alerts.ntfy_url.rstrip('/')}/{settings.alerts.ntfy_topic}"
    api_url = settings.email_draft.atlas_api_url.rstrip("/")
    delay_min = settings.campaign_sequence.auto_send_delay_seconds // 60

    message = (
        f"Step {step_number} queued for {company_name}\n"
        f"Subject: {subject}\n"
        f"Auto-sending in {delay_min} min unless cancelled."
    )

    cancel_url = f"{api_url}/api/v1/b2b/campaigns/{campaign_id}/cancel"
    actions = f"http, Cancel Auto-Send, {cancel_url}, method=POST, clear=true"

    headers = {
        "Title": f"Campaign Step {step_number}: {company_name}",
        "Priority": "default",
        "Tags": "outbox,campaign",
        "Actions": actions,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(ntfy_url, content=message, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("ntfy notification failed for sequence %s: %s", sequence_id, exc)


async def run(task: ScheduledTask) -> dict:
    """Check for active sequences due for progression and generate next steps."""
    cfg = settings.campaign_sequence
    if not cfg.enabled:
        return {"_skip_synthesis": "Campaign sequences disabled"}

    pool = get_db_pool()
    if not pool.is_initialized:
        return {"_skip_synthesis": "Database unavailable"}

    now = datetime.now(timezone.utc)

    # Find active sequences due for next step
    sequences = await pool.fetch(
        """
        SELECT cs.*
        FROM campaign_sequences cs
        WHERE cs.status = 'active'
          AND cs.next_step_after IS NOT NULL
          AND cs.next_step_after <= $1
          AND cs.recipient_email IS NOT NULL
          AND cs.current_step < cs.max_steps
          AND NOT EXISTS (
              SELECT 1 FROM b2b_campaigns bc
              WHERE bc.sequence_id = cs.id AND bc.status = 'queued'
          )
        ORDER BY cs.next_step_after ASC
        LIMIT 10
        """,
        now,
    )

    if not sequences:
        return {"_skip_synthesis": True, "due_sequences": 0}

    progressed = 0

    for seq in sequences:
        seq_dict = dict(seq)
        seq_id = seq_dict["id"]
        next_step = seq_dict["current_step"] + 1

        # Load all previous campaigns for this sequence
        previous = await pool.fetch(
            """
            SELECT id, step_number, subject, body, status, sent_at,
                   opened_at, clicked_at, esp_message_id
            FROM b2b_campaigns
            WHERE sequence_id = $1
            ORDER BY step_number ASC
            """,
            seq_id,
        )
        previous_dicts = [dict(r) for r in previous]

        # Generate next email via LLM
        content = await _generate_next_step(seq_dict, previous_dicts)
        if not content:
            logger.warning("Failed to generate step %d for sequence %s", next_step, seq_id)
            continue

        # Insert new campaign row as queued
        campaign_id = await pool.fetchval(
            """
            INSERT INTO b2b_campaigns
                (company_name, batch_id, partner_id, channel,
                 subject, body, status, approved_at,
                 sequence_id, step_number, recipient_email, from_email,
                 metadata)
            VALUES ($1, $2, $3, 'email_followup',
                    $4, $5, 'queued', $6,
                    $7, $8, $9, $10, $11)
            RETURNING id
            """,
            seq_dict["company_name"],
            seq_dict["batch_id"],
            seq_dict.get("partner_id"),
            content.get("subject", ""),
            content.get("body", ""),
            now,  # approved_at = NOW() for cancel window
            seq_id,
            next_step,
            seq_dict["recipient_email"],
            cfg.resend_from_email,
            json.dumps({
                "cta": content.get("cta", ""),
                "angle_reasoning": content.get("angle_reasoning", ""),
            }),
        )

        # Update sequence step count (but NOT next_step_after -- set after actual send)
        await pool.execute(
            "UPDATE campaign_sequences SET current_step = $1, updated_at = $2 WHERE id = $3",
            next_step, now, seq_id,
        )

        # Audit log
        await log_campaign_event(
            pool, event_type="generated", source="system",
            campaign_id=campaign_id, sequence_id=seq_id,
            step_number=next_step,
            subject=content.get("subject"),
            body=content.get("body"),
            recipient_email=seq_dict["recipient_email"],
            metadata={"angle_reasoning": content.get("angle_reasoning", "")},
        )
        await log_campaign_event(
            pool, event_type="queued", source="system",
            campaign_id=campaign_id, sequence_id=seq_id,
            step_number=next_step,
            recipient_email=seq_dict["recipient_email"],
        )

        # ntfy notification
        await _send_ntfy_notification(
            company_name=seq_dict["company_name"],
            step_number=next_step,
            subject=content.get("subject", ""),
            sequence_id=str(seq_id),
            campaign_id=str(campaign_id),
        )

        progressed += 1
        logger.info(
            "Queued step %d for sequence %s (%s): %s",
            next_step, seq_id, seq_dict["company_name"], content.get("subject", ""),
        )

    return {
        "_skip_synthesis": True,
        "due_sequences": len(sequences),
        "progressed": progressed,
    }
