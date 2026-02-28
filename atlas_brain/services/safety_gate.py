"""
Safety & compliance orchestration for the intervention pipeline.

Provides four enforcement layers:
1. Content filtering  -- flag/block prohibited patterns in LLM output
2. Approval workflow   -- create approval request, approve/reject, check status
3. Audit logging       -- immutable event log via atlas_events table
4. Human review gate   -- block execution until a human approves

Usage:
    from atlas_brain.services.safety_gate import SafetyGate

    gate = SafetyGate()
    check = gate.check_content(llm_output)
    if check["blocked"]:
        ...  # content filtering caught something

    approval_id = await gate.request_approval(pipeline_id, "narrative_architect", ...)
    status = await gate.check_approval(approval_id)
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("atlas.services.safety_gate")

# Patterns that must never appear in intervention output.
# These catch deceptive, coercive, or identity-misrepresenting content.
_PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    (r"\bimpersonat(?:e|ing|ion)\b", "impersonation"),
    (r"\bfabricat(?:e|ed|ing)\s+(?:facts?|evidence|data)", "fabricated_facts"),
    (r"\bblackmail\b", "blackmail"),
    (r"\bextort(?:ion|ing)?\b", "extortion"),
    (r"\bthreaten(?:s|ed|ing)?\s+(?:to\s+)?(?:harm|violence|physical)", "threat_of_harm"),
    (r"\bmanipulat(?:e|ing)\s+(?:evidence|records|data)", "evidence_manipulation"),
    (r"\bdoxx?(?:ing|ed)?\b", "doxxing"),
    (r"\bphishing\b", "phishing"),
    (r"\bsocial\s+engineer(?:ing)?\b", "social_engineering"),
]

# Compiled once at module load
_COMPILED_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), label) for pat, label in _PROHIBITED_PATTERNS
]

# Risk level thresholds for auto-approval
_AUTO_APPROVE_MAX_RISK = "MEDIUM"  # LOW and MEDIUM can auto-approve
_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Default approval expiry
_APPROVAL_EXPIRY_HOURS = 72


class SafetyGate:
    """Central safety enforcement for the intervention pipeline."""

    # ---- Content Filtering ----

    def check_content(self, text: str) -> dict[str, Any]:
        """Scan text for prohibited patterns.

        Returns:
            {
                "passed": bool,
                "blocked": bool,
                "flags": [{"pattern": str, "match": str, "position": int}],
            }
        """
        if not text:
            return {"passed": True, "blocked": False, "flags": []}

        flags: list[dict[str, Any]] = []
        for compiled, label in _COMPILED_PATTERNS:
            for match in compiled.finditer(text):
                flags.append({
                    "pattern": label,
                    "match": match.group(),
                    "position": match.start(),
                })

        blocked = len(flags) > 0
        return {
            "passed": not blocked,
            "blocked": blocked,
            "flags": flags,
        }

    # ---- Approval Workflow ----

    async def request_approval(
        self,
        pipeline_id: str,
        stage: str,
        entity_name: str,
        requested_by: str,
        safety_checks: Optional[dict] = None,
        expiry_hours: int = _APPROVAL_EXPIRY_HOURS,
    ) -> Optional[str]:
        """Create an approval request for a pipeline stage.

        Returns the approval_id, or None if DB is unavailable.
        """
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            logger.warning("Cannot create approval request: DB not initialized")
            return None

        approval_id = str(uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        try:
            await pool.execute(
                """
                INSERT INTO intervention_approvals (
                    id, pipeline_id, stage, entity_name,
                    status, requested_by, safety_checks, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                """,
                approval_id,
                pipeline_id,
                stage,
                entity_name,
                "pending",
                requested_by,
                json.dumps(safety_checks or {}),
                expires_at,
            )

            # Log to audit trail
            await self.log_event(
                event_type="intervention.approval_requested",
                source=requested_by,
                entity_name=entity_name,
                payload={
                    "approval_id": approval_id,
                    "pipeline_id": pipeline_id,
                    "stage": stage,
                    "expires_at": expires_at.isoformat(),
                },
            )

            logger.info(
                "Approval requested: %s for %s stage %s (expires %s)",
                approval_id, entity_name, stage, expires_at.isoformat(),
            )
            return approval_id

        except Exception:
            logger.exception("Failed to create approval request")
            return None

    async def check_approval(self, approval_id: str) -> dict[str, Any]:
        """Check the status of an approval request.

        Returns:
            {"status": "pending"|"approved"|"rejected"|"expired"|"not_found", ...}
        """
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return {"status": "error", "detail": "Database not initialized"}

        row = await pool.fetchrow(
            """
            SELECT id, pipeline_id, stage, entity_name, status,
                   requested_by, reviewed_by, review_notes,
                   safety_checks, created_at, reviewed_at, expires_at
            FROM intervention_approvals
            WHERE id = $1
            """,
            approval_id,
        )

        if not row:
            return {"status": "not_found"}

        result = dict(row)

        # Check expiry
        if (
            result["status"] == "pending"
            and result.get("expires_at")
            and datetime.now(timezone.utc) > result["expires_at"]
        ):
            await pool.execute(
                "UPDATE intervention_approvals SET status = 'expired' WHERE id = $1",
                approval_id,
            )
            result["status"] = "expired"

        # Serialize timestamps
        for key in ("created_at", "reviewed_at", "expires_at"):
            if result.get(key):
                result[key] = result[key].isoformat()
        result["id"] = str(result["id"])
        result["pipeline_id"] = str(result["pipeline_id"])

        return result

    async def approve(
        self,
        approval_id: str,
        reviewed_by: str,
        notes: str = "",
    ) -> bool:
        """Approve a pending request. Returns True if successful."""
        return await self._review(approval_id, "approved", reviewed_by, notes)

    async def reject(
        self,
        approval_id: str,
        reviewed_by: str,
        notes: str = "",
    ) -> bool:
        """Reject a pending request. Returns True if successful."""
        return await self._review(approval_id, "rejected", reviewed_by, notes)

    async def _review(
        self,
        approval_id: str,
        status: str,
        reviewed_by: str,
        notes: str,
    ) -> bool:
        """Update approval status."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return False

        try:
            result = await pool.execute(
                """
                UPDATE intervention_approvals
                SET status = $2, reviewed_by = $3, review_notes = $4,
                    reviewed_at = NOW()
                WHERE id = $1 AND status = 'pending'
                """,
                approval_id,
                status,
                reviewed_by,
                notes,
            )

            updated = result.endswith("1") if isinstance(result, str) else False

            if updated:
                # Fetch entity_name for audit log
                row = await pool.fetchrow(
                    "SELECT entity_name, pipeline_id, stage FROM intervention_approvals WHERE id = $1",
                    approval_id,
                )
                entity_name = row["entity_name"] if row else "unknown"
                await self.log_event(
                    event_type=f"intervention.{status}",
                    source=reviewed_by,
                    entity_name=entity_name,
                    payload={
                        "approval_id": approval_id,
                        "pipeline_id": str(row["pipeline_id"]) if row else "",
                        "stage": row["stage"] if row else "",
                        "notes": notes,
                    },
                )
                logger.info("Approval %s %s by %s", approval_id, status, reviewed_by)

            return updated

        except Exception:
            logger.exception("Failed to update approval %s", approval_id)
            return False

    async def list_pending(
        self,
        entity_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List pending approval requests."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            return []

        if entity_name:
            rows = await pool.fetch(
                """
                SELECT id, pipeline_id, stage, entity_name, requested_by,
                       safety_checks, created_at, expires_at
                FROM intervention_approvals
                WHERE status = 'pending' AND entity_name ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                f"%{entity_name}%",
                limit,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT id, pipeline_id, stage, entity_name, requested_by,
                       safety_checks, created_at, expires_at
                FROM intervention_approvals
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )

        results = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d["id"])
            d["pipeline_id"] = str(d["pipeline_id"])
            for key in ("created_at", "expires_at"):
                if d.get(key):
                    d[key] = d[key].isoformat()
            results.append(d)
        return results

    # ---- Risk Assessment ----

    def assess_risk(
        self,
        sensor_summary: dict,
        pressure: dict,
        content_check: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Assess the overall risk level for a pipeline run.

        Returns:
            {
                "risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
                "auto_approve_eligible": bool,
                "factors": [str],
            }
        """
        factors: list[str] = []
        risk_score = 0

        # Sensor-based risk
        sensor_level = sensor_summary.get("dominant_risk_level", "LOW")
        risk_score += _RISK_ORDER.get(sensor_level, 0)
        if sensor_level in ("HIGH", "CRITICAL"):
            factors.append(f"Sensor composite: {sensor_level}")

        # Pressure-based risk
        pressure_score = pressure.get("pressure_score", 0)
        if isinstance(pressure_score, (int, float)):
            if pressure_score >= 8:
                risk_score += 2
                factors.append(f"Critical pressure: {pressure_score}/10")
            elif pressure_score >= 6:
                risk_score += 1
                factors.append(f"Elevated pressure: {pressure_score}/10")

        # Content filtering risk
        if content_check and content_check.get("blocked"):
            risk_score += 3
            flag_labels = [f["pattern"] for f in content_check.get("flags", [])]
            factors.append(f"Content flags: {', '.join(flag_labels)}")

        # Map score to level
        if risk_score >= 4:
            level = "CRITICAL"
        elif risk_score >= 3:
            level = "HIGH"
        elif risk_score >= 1:
            level = "MEDIUM"
        else:
            level = "LOW"

        auto_eligible = _RISK_ORDER.get(level, 0) <= _RISK_ORDER.get(_AUTO_APPROVE_MAX_RISK, 1)

        return {
            "risk_level": level,
            "risk_score": risk_score,
            "auto_approve_eligible": auto_eligible,
            "factors": factors,
        }

    # ---- Audit Logging ----

    async def log_event(
        self,
        event_type: str,
        source: str,
        entity_name: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Log an immutable audit event to the atlas_events table."""
        from ..storage.database import get_db_pool

        pool = get_db_pool()
        if not pool.is_initialized:
            logger.debug("Audit log skipped (DB not initialized): %s", event_type)
            return

        try:
            await pool.execute(
                """
                INSERT INTO atlas_events (event_type, source, entity_type, entity_id, payload)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                event_type,
                source,
                "intervention",
                entity_name,
                json.dumps(payload or {}, default=str),
            )
        except Exception:
            logger.exception("Failed to write audit event: %s", event_type)

    # ---- Composite Gate Check ----

    async def gate_check(
        self,
        pipeline_id: str,
        stage: str,
        entity_name: str,
        stage_output: str,
        sensor_summary: dict,
        pressure: dict,
        requested_by: str,
    ) -> dict[str, Any]:
        """Run all safety checks and return a gate decision.

        Returns:
            {
                "allowed": bool,
                "content_check": {...},
                "risk_assessment": {...},
                "approval_id": str or None,
                "reason": str,
            }
        """
        # 1. Content filtering
        content_check = self.check_content(stage_output)

        # 2. Risk assessment
        risk = self.assess_risk(sensor_summary, pressure, content_check)

        # 3. Log the gate check
        await self.log_event(
            event_type="intervention.gate_check",
            source=requested_by,
            entity_name=entity_name,
            payload={
                "pipeline_id": pipeline_id,
                "stage": stage,
                "content_passed": content_check["passed"],
                "risk_level": risk["risk_level"],
                "risk_score": risk["risk_score"],
            },
        )

        # 4. Decision
        if content_check["blocked"]:
            await self.log_event(
                event_type="intervention.blocked_by_content_filter",
                source="safety_gate",
                entity_name=entity_name,
                payload={
                    "pipeline_id": pipeline_id,
                    "stage": stage,
                    "flags": content_check["flags"],
                },
            )
            return {
                "allowed": False,
                "content_check": content_check,
                "risk_assessment": risk,
                "approval_id": None,
                "reason": f"Content filter blocked: {', '.join(f['pattern'] for f in content_check['flags'])}",
            }

        if risk["auto_approve_eligible"]:
            # Low/medium risk -- auto-approve with audit trail
            await self.log_event(
                event_type="intervention.auto_approved",
                source="safety_gate",
                entity_name=entity_name,
                payload={
                    "pipeline_id": pipeline_id,
                    "stage": stage,
                    "risk_level": risk["risk_level"],
                },
            )
            return {
                "allowed": True,
                "content_check": content_check,
                "risk_assessment": risk,
                "approval_id": None,
                "reason": f"Auto-approved (risk: {risk['risk_level']})",
            }

        # High/critical risk -- require human approval
        approval_id = await self.request_approval(
            pipeline_id=pipeline_id,
            stage=stage,
            entity_name=entity_name,
            requested_by=requested_by,
            safety_checks={
                "content_check": content_check,
                "risk_assessment": risk,
            },
        )

        return {
            "allowed": False,
            "content_check": content_check,
            "risk_assessment": risk,
            "approval_id": approval_id,
            "reason": f"Human approval required (risk: {risk['risk_level']}). Approval ID: {approval_id}",
        }


# Module-level singleton
_gate: Optional[SafetyGate] = None


def get_safety_gate() -> SafetyGate:
    """Get the module-level SafetyGate singleton."""
    global _gate
    if _gate is None:
        _gate = SafetyGate()
    return _gate
