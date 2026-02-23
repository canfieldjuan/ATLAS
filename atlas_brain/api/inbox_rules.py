"""
REST API for user-defined inbox automation rules.

Provides CRUD operations, a dry-run test endpoint, and bulk reorder
for email inbox rules evaluated by the intake pipeline.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..storage.database import get_db_pool

logger = logging.getLogger("atlas.api.inbox_rules")

router = APIRouter(prefix="/email/inbox-rules", tags=["email-inbox-rules"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InboxRuleCreate(BaseModel):
    name: str
    enabled: bool = True
    position: int = 0
    # Conditions
    sender_domain: str | None = None
    sender_contains: str | None = None
    subject_contains: str | None = None
    category: str | None = None
    has_unsubscribe: bool | None = None
    priority: str | None = None
    replyable: bool | None = None
    is_known_contact: bool | None = None
    # Actions
    set_priority: str | None = None
    set_category: str | None = None
    set_replyable: bool | None = None
    label: str | None = None
    skip_llm: bool = False
    skip_notify: bool = False
    archive: bool = False


class InboxRuleUpdate(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    position: int | None = None
    sender_domain: str | None = None
    sender_contains: str | None = None
    subject_contains: str | None = None
    category: str | None = None
    has_unsubscribe: bool | None = None
    priority: str | None = None
    replyable: bool | None = None
    is_known_contact: bool | None = None
    set_priority: str | None = None
    set_category: str | None = None
    set_replyable: bool | None = None
    label: str | None = None
    skip_llm: bool | None = None
    skip_notify: bool | None = None
    archive: bool | None = None


class InboxRuleTest(BaseModel):
    sender: str
    subject: str
    category: str | None = None
    priority: str | None = None
    has_unsubscribe: bool = False
    replyable: bool | None = None
    is_known_contact: bool = False


class ReorderItem(BaseModel):
    id: str
    position: int


class ReorderBody(BaseModel):
    rules: list[ReorderItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row) -> dict:
    """Convert a DB row to a JSON-safe dict."""
    d = {}
    for key in row.keys():
        val = row[key]
        if isinstance(val, UUID):
            d[key] = str(val)
        elif isinstance(val, datetime):
            d[key] = val.isoformat()
        else:
            d[key] = val
    return d


def _rule_matches(rule: dict, email: dict[str, Any]) -> bool:
    """Check whether all non-NULL conditions in *rule* match *email*.

    All specified conditions must match (AND logic).  A NULL condition
    means "don't check this field".
    """
    sender = (email.get("from") or email.get("sender") or "").lower()
    subject = (email.get("subject") or "").lower()

    if rule["sender_domain"] is not None:
        domain = rule["sender_domain"].lower()
        # Extract domain from sender address
        at_idx = sender.rfind("@")
        if at_idx == -1 or not sender[at_idx + 1:].rstrip(">").endswith(domain):
            return False

    if rule["sender_contains"] is not None:
        if rule["sender_contains"].lower() not in sender:
            return False

    if rule["subject_contains"] is not None:
        if rule["subject_contains"].lower() not in subject:
            return False

    if rule["category"] is not None:
        if email.get("category") != rule["category"]:
            return False

    if rule["has_unsubscribe"] is not None:
        if email.get("has_unsubscribe") != rule["has_unsubscribe"]:
            return False

    if rule["priority"] is not None:
        if email.get("priority") != rule["priority"]:
            return False

    if rule["replyable"] is not None:
        if email.get("replyable") != rule["replyable"]:
            return False

    if rule["is_known_contact"] is not None:
        has_contact = email.get("_contact_id") is not None
        if rule["is_known_contact"] != has_contact:
            return False

    return True


def _apply_rule_actions(rule: dict, email: dict[str, Any]) -> dict[str, Any]:
    """Apply rule actions to email dict.  Returns a summary of applied actions."""
    applied: dict[str, Any] = {}

    if rule["set_priority"]:
        email["priority"] = rule["set_priority"]
        applied["set_priority"] = rule["set_priority"]

    if rule["set_category"]:
        email["category"] = rule["set_category"]
        applied["set_category"] = rule["set_category"]

    if rule["set_replyable"] is not None:
        email["replyable"] = rule["set_replyable"]
        applied["set_replyable"] = rule["set_replyable"]

    if rule["archive"]:
        email["replyable"] = False
        email["_auto_executed"] = True
        applied["archive"] = True

    if rule["skip_llm"]:
        email["replyable"] = False
        applied["skip_llm"] = True

    if rule["skip_notify"]:
        email["_auto_executed"] = True
        applied["skip_notify"] = True

    return applied


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def list_rules(
    enabled_only: bool = Query(default=False, description="Only return enabled rules"),
):
    """List all inbox rules ordered by position."""
    pool = get_db_pool()
    if not pool.is_initialized:
        return {"count": 0, "rules": []}

    if enabled_only:
        rows = await pool.fetch(
            "SELECT * FROM email_inbox_rules WHERE enabled = true ORDER BY position ASC",
        )
    else:
        rows = await pool.fetch(
            "SELECT * FROM email_inbox_rules ORDER BY position ASC",
        )
    rules = [_row_to_dict(r) for r in rows]
    return {"count": len(rules), "rules": rules}


@router.get("/{rule_id}")
async def get_rule(rule_id: UUID):
    """Get a single inbox rule by ID."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        "SELECT * FROM email_inbox_rules WHERE id = $1", rule_id,
    )
    if not row:
        raise HTTPException(404, "Rule not found")
    return _row_to_dict(row)


@router.post("/", status_code=201)
async def create_rule(body: InboxRuleCreate):
    """Create a new inbox rule."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    row = await pool.fetchrow(
        """
        INSERT INTO email_inbox_rules
            (name, enabled, position,
             sender_domain, sender_contains, subject_contains,
             category, has_unsubscribe, priority, replyable, is_known_contact,
             set_priority, set_category, set_replyable, label,
             skip_llm, skip_notify, archive)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18)
        RETURNING *
        """,
        body.name, body.enabled, body.position,
        body.sender_domain, body.sender_contains, body.subject_contains,
        body.category, body.has_unsubscribe, body.priority,
        body.replyable, body.is_known_contact,
        body.set_priority, body.set_category, body.set_replyable, body.label,
        body.skip_llm, body.skip_notify, body.archive,
    )
    logger.info("Created inbox rule %s: %s", row["id"], body.name)
    return _row_to_dict(row)


@router.put("/{rule_id}")
async def update_rule(rule_id: UUID, body: InboxRuleUpdate):
    """Update an existing inbox rule (partial update)."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    # exclude_unset keeps explicitly-sent null values (to clear conditions)
    # but omits fields the client didn't include at all.
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    # Always bump updated_at
    updates["updated_at"] = datetime.now(timezone.utc)

    set_parts = []
    values = []
    for i, (col, val) in enumerate(updates.items(), start=1):
        set_parts.append(f"{col} = ${i}")
        values.append(val)

    values.append(rule_id)
    query = f"""
        UPDATE email_inbox_rules
        SET {', '.join(set_parts)}
        WHERE id = ${len(values)}
        RETURNING *
    """
    row = await pool.fetchrow(query, *values)
    if not row:
        raise HTTPException(404, "Rule not found")
    logger.info("Updated inbox rule %s", rule_id)
    return _row_to_dict(row)


@router.delete("/{rule_id}")
async def delete_rule(rule_id: UUID):
    """Delete an inbox rule."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    result = await pool.execute(
        "DELETE FROM email_inbox_rules WHERE id = $1", rule_id,
    )
    # asyncpg returns "DELETE N"
    try:
        deleted = int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        deleted = 0

    if deleted == 0:
        raise HTTPException(404, "Rule not found")
    logger.info("Deleted inbox rule %s", rule_id)
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Bulk reorder
# ---------------------------------------------------------------------------

@router.post("/reorder")
async def reorder_rules(body: ReorderBody):
    """Bulk-update positions for multiple rules."""
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    async with pool.transaction() as conn:
        for item in body.rules:
            await conn.execute(
                """
                UPDATE email_inbox_rules
                SET position = $1, updated_at = NOW()
                WHERE id = $2
                """,
                item.position, UUID(item.id),
            )
    logger.info("Reordered %d inbox rules", len(body.rules))
    return {"reordered": len(body.rules)}


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------

@router.post("/test")
async def test_rules(body: InboxRuleTest):
    """Test enabled rules against a sample email (dry-run).

    Returns the first matching rule and the actions that would be applied.
    """
    pool = get_db_pool()
    if not pool.is_initialized:
        raise HTTPException(503, "Database not available")

    rules = await pool.fetch(
        "SELECT * FROM email_inbox_rules WHERE enabled = true ORDER BY position ASC",
    )
    if not rules:
        return {"matched": False, "rule": None, "actions": {}}

    # Build a fake email dict from the test body
    fake_email: dict[str, Any] = {
        "from": body.sender,
        "subject": body.subject,
        "category": body.category,
        "priority": body.priority,
        "has_unsubscribe": body.has_unsubscribe,
        "replyable": body.replyable,
        "_contact_id": "fake" if body.is_known_contact else None,
    }

    for rule in rules:
        if _rule_matches(rule, fake_email):
            actions = _apply_rule_actions(rule, fake_email)
            return {
                "matched": True,
                "rule": _row_to_dict(rule),
                "actions": actions,
            }

    return {"matched": False, "rule": None, "actions": {}}
