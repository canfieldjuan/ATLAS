"""B2B Churn MCP -- write-back tools for client-driven intelligence."""

import json
from datetime import date
from typing import Optional

from ._shared import VALID_REPORT_TYPES, get_pool, logger
from .server import mcp
from .write_ports import (
    WriteIntelligencePortNotConfigured,
    get_accounts_in_motion_builder,
    get_blog_matcher,
    get_challenger_brief_builder,
    get_reasoning_view_loader,
)

_VALID_ANALYSIS_TYPES = ("pairwise_battle", "category_council", "resource_asymmetry")
_VALID_CHANNELS = ("email_cold", "email_followup", "linkedin", "phone", "custom")


# -------------------------------------------------------------------
# 1. persist_conclusion
# -------------------------------------------------------------------


@mcp.tool()
async def persist_conclusion(
    analysis_type: str,
    vendors: str,
    conclusion: str,
    confidence: float,
    evidence_hash: str,
    llm_model: str,
    category: Optional[str] = None,
) -> str:
    """Write a cross-vendor conclusion to b2b_cross_vendor_conclusions.

    analysis_type: pairwise_battle, category_council, or resource_asymmetry
    vendors: JSON array of vendor names (2+ non-empty strings)
    conclusion: JSON object with the conclusion payload
    confidence: Confidence score between 0.0 and 1.0
    evidence_hash: Non-empty hash identifying the evidence set
    llm_model: Model that produced this conclusion (e.g. 'mcp-client:claude-sonnet-4-20250514')
    category: Optional category name (used for category_council)
    """
    if analysis_type not in _VALID_ANALYSIS_TYPES:
        return json.dumps({"success": False, "error": f"analysis_type must be one of {_VALID_ANALYSIS_TYPES}"})

    try:
        vendor_list = json.loads(vendors)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"success": False, "error": "vendors must be a valid JSON array"})
    if not isinstance(vendor_list, list) or len(vendor_list) < 2:
        return json.dumps({"success": False, "error": "vendors must contain at least 2 entries"})
    if not all(isinstance(v, str) and v.strip() for v in vendor_list):
        return json.dumps({"success": False, "error": "All vendor entries must be non-empty strings"})

    try:
        conclusion_obj = json.loads(conclusion)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"success": False, "error": "conclusion must be valid JSON"})
    if not isinstance(conclusion_obj, dict):
        return json.dumps({"success": False, "error": "conclusion must be a JSON object"})

    confidence = max(0.0, min(1.0, confidence))

    if not evidence_hash or not evidence_hash.strip():
        return json.dumps({"success": False, "error": "evidence_hash must be non-empty"})
    if not llm_model or not llm_model.strip():
        return json.dumps({"success": False, "error": "llm_model must be non-empty"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        sorted_vendors = sorted(v.strip() for v in vendor_list)

        row = await pool.fetchrow(
            """
            INSERT INTO b2b_cross_vendor_conclusions
                (analysis_type, vendors, category, conclusion, confidence,
                 evidence_hash, tokens_used, cached, llm_model)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, 0, false, $7)
            RETURNING id, created_at
            """,
            analysis_type,
            sorted_vendors,
            category,
            json.dumps(conclusion_obj),
            confidence,
            evidence_hash.strip(),
            llm_model.strip(),
        )

        return json.dumps({
            "success": True,
            "id": str(row["id"]),
            "created_at": str(row["created_at"]),
        })
    except Exception:
        logger.exception("persist_conclusion error")
        return json.dumps({"success": False, "error": "Internal error"})


# -------------------------------------------------------------------
# 2. persist_report
# -------------------------------------------------------------------


@mcp.tool()
async def persist_report(
    report_type: str,
    intelligence_data: str,
    executive_summary: str,
    llm_model: str,
    vendor_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    report_date: Optional[str] = None,
    status: str = "draft",
) -> str:
    """Write or upsert a report to b2b_intelligence.

    report_type: Must be a valid report type (weekly_churn_feed, vendor_scorecard, etc.)
    intelligence_data: JSON object with the report payload
    executive_summary: Non-empty summary, max 5000 chars
    llm_model: Model that produced this report (e.g. 'mcp-client:claude-sonnet-4-20250514')
    vendor_filter: Optional vendor name scope
    category_filter: Optional category scope
    report_date: ISO date string (defaults to today)
    status: 'draft' (default) or 'published'
    """
    if report_type not in VALID_REPORT_TYPES:
        return json.dumps({"success": False, "error": f"report_type must be one of {VALID_REPORT_TYPES}"})

    try:
        data_obj = json.loads(intelligence_data)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"success": False, "error": "intelligence_data must be valid JSON"})
    if not isinstance(data_obj, dict):
        return json.dumps({"success": False, "error": "intelligence_data must be a JSON object"})

    if not executive_summary or not executive_summary.strip():
        return json.dumps({"success": False, "error": "executive_summary must be non-empty"})
    if len(executive_summary) > 5000:
        return json.dumps({"success": False, "error": "executive_summary exceeds 5000 chars"})

    if not llm_model or not llm_model.strip():
        return json.dumps({"success": False, "error": "llm_model must be non-empty"})

    if status not in ("draft", "published"):
        return json.dumps({"success": False, "error": "status must be 'draft' or 'published'"})

    rdate = date.today()
    if report_date:
        try:
            rdate = date.fromisoformat(report_date)
        except ValueError:
            return json.dumps({"success": False, "error": "report_date must be ISO format (YYYY-MM-DD)"})

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        row = await pool.fetchrow(
            """
            INSERT INTO b2b_intelligence
                (report_date, report_type, vendor_filter, category_filter,
                 intelligence_data, executive_summary, status, llm_model)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
            ON CONFLICT (report_date, report_type,
                         LOWER(COALESCE(vendor_filter, '')),
                         LOWER(COALESCE(category_filter, '')),
                         COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid))
            DO UPDATE SET
                intelligence_data = EXCLUDED.intelligence_data,
                executive_summary = EXCLUDED.executive_summary,
                status = EXCLUDED.status,
                llm_model = EXCLUDED.llm_model,
                created_at = now()
            RETURNING id, created_at
            """,
            rdate,
            report_type,
            vendor_filter,
            category_filter,
            json.dumps(data_obj),
            executive_summary.strip(),
            status,
            llm_model.strip(),
        )

        return json.dumps({
            "success": True,
            "id": str(row["id"]),
            "report_date": str(rdate),
            "upserted": True,
            "created_at": str(row["created_at"]),
        })
    except Exception:
        logger.exception("persist_report error")
        return json.dumps({"success": False, "error": "Internal error"})


# -------------------------------------------------------------------
# 3. build_challenger_brief
# -------------------------------------------------------------------


@mcp.tool()
async def build_challenger_brief(
    incumbent: str,
    challenger: str,
    persist: bool = True,
    max_target_accounts: int = 50,
) -> str:
    """Trigger deterministic challenger brief assembly for a vendor pair.

    Reuses the same data-fetching and assembly logic as the nightly pipeline.
    No LLM calls -- purely deterministic from pre-computed artifacts.

    incumbent: Incumbent vendor name
    challenger: Challenger vendor name (must differ from incumbent)
    persist: Whether to save to b2b_intelligence (default True)
    max_target_accounts: Max target accounts in brief (default 50, range 5-200)
    """
    if not incumbent or not incumbent.strip():
        return json.dumps({"success": False, "error": "incumbent must be non-empty"})
    if not challenger or not challenger.strip():
        return json.dumps({"success": False, "error": "challenger must be non-empty"})
    if incumbent.strip().lower() == challenger.strip().lower():
        return json.dumps({"success": False, "error": "challenger must differ from incumbent"})

    max_target_accounts = max(5, min(200, max_target_accounts))

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        builder = get_challenger_brief_builder()
        result = await builder(
            pool,
            incumbent=incumbent.strip(),
            challenger=challenger.strip(),
            persist=persist,
            max_target_accounts=max_target_accounts,
        )
        return json.dumps(result, default=str)
    except WriteIntelligencePortNotConfigured as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("build_challenger_brief error")
        return json.dumps({"success": False, "error": "Internal error"})


# -------------------------------------------------------------------
# 4. build_accounts_in_motion
# -------------------------------------------------------------------


@mcp.tool()
async def build_accounts_in_motion(
    vendor_name: str,
    persist: bool = True,
    min_urgency: float = 5.0,
    max_accounts: int = 100,
) -> str:
    """Trigger deterministic accounts-in-motion assembly for a vendor.

    Reuses the same scoring and aggregation logic as the nightly pipeline.
    No LLM calls -- purely deterministic from pre-computed artifacts.

    vendor_name: Vendor to build AIM report for
    persist: Whether to save to b2b_intelligence (default True)
    min_urgency: Minimum urgency score to include (default 5.0, range 3.0-10.0)
    max_accounts: Maximum accounts in report (default 100, range 5-500)
    """
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name must be non-empty"})

    min_urgency = max(3.0, min(10.0, min_urgency))
    max_accounts = max(5, min(500, max_accounts))

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        builder = get_accounts_in_motion_builder()
        result = await builder(
            pool,
            vendor_name=vendor_name.strip(),
            persist=persist,
            min_urgency=min_urgency,
            max_accounts=max_accounts,
        )
        return json.dumps(result, default=str)
    except WriteIntelligencePortNotConfigured as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception:
        logger.exception("build_accounts_in_motion error")
        return json.dumps({"success": False, "error": "Internal error"})


# -------------------------------------------------------------------
# 5. draft_campaign
# -------------------------------------------------------------------


@mcp.tool()
async def draft_campaign(
    company_name: str,
    vendor_name: str,
    channel: str,
    body: str,
    llm_model: str,
    subject: Optional[str] = None,
    cta: Optional[str] = None,
    opportunity_score: Optional[int] = None,
    urgency_score: Optional[float] = None,
) -> str:
    """Create a campaign draft row in b2b_campaigns with status='draft'.

    company_name: Target company name
    vendor_name: Vendor being displaced
    channel: email_cold, email_followup, linkedin, phone, or custom
    body: Campaign message body (max 10000 chars)
    llm_model: Model that generated this content (e.g. 'mcp-client:claude-sonnet-4-20250514')
    subject: Required for email channels (email_cold, email_followup)
    cta: Optional call-to-action text
    opportunity_score: Optional score 0-100
    urgency_score: Optional score 0.0-10.0
    """
    if not company_name or not company_name.strip():
        return json.dumps({"success": False, "error": "company_name must be non-empty"})
    if not vendor_name or not vendor_name.strip():
        return json.dumps({"success": False, "error": "vendor_name must be non-empty"})

    if channel not in _VALID_CHANNELS:
        return json.dumps({"success": False, "error": f"channel must be one of {_VALID_CHANNELS}"})

    if not body or not body.strip():
        return json.dumps({"success": False, "error": "body must be non-empty"})
    if len(body) > 10000:
        return json.dumps({"success": False, "error": "body exceeds 10000 chars"})

    if not llm_model or not llm_model.strip():
        return json.dumps({"success": False, "error": "llm_model must be non-empty"})

    if channel in ("email_cold", "email_followup") and (not subject or not subject.strip()):
        return json.dumps({"success": False, "error": "subject is required for email channels"})

    if opportunity_score is not None:
        opportunity_score = max(0, min(100, opportunity_score))
    if urgency_score is not None:
        urgency_score = max(0.0, min(10.0, urgency_score))

    try:
        pool = get_pool()
        if not pool.is_initialized:
            return json.dumps({"success": False, "error": "Database not ready"})

        # Fetch relevant blog posts before inserting so they can be stored in metadata
        blog_posts: list[dict] = []
        try:
            fetch_blogs = get_blog_matcher()
            if fetch_blogs is not None:
                blog_posts = await fetch_blogs(
                    pool,
                    pipeline="b2b",
                    vendor_name=vendor_name.strip(),
                    include_drafts=False,
                    limit=3,
                )
        except Exception:
            logger.debug("Blog lookup failed for draft_campaign vendor=%s", vendor_name)

        metadata: dict = {}
        if blog_posts:
            metadata["blog_posts"] = blog_posts

        # Store reasoning context for audit trail
        try:
            load_reasoning_view = get_reasoning_view_loader()
            if load_reasoning_view is not None:
                reasoning_view = await load_reasoning_view(
                    pool,
                    vendor_name.strip(),
                )
            else:
                reasoning_view = None
            if reasoning_view is not None:
                wedge = reasoning_view.primary_wedge
                reasoning_meta: dict = {
                    "wedge": wedge.value if wedge else "",
                    "confidence": reasoning_view.confidence("causal_narrative"),
                    "schema_version": reasoning_view.schema_version,
                }
                wts = reasoning_view.why_they_stay
                if wts and wts.get("summary"):
                    reasoning_meta["why_they_stay"] = wts["summary"]
                if reasoning_view.confidence_limits:
                    reasoning_meta["confidence_limits"] = reasoning_view.confidence_limits
                metadata["reasoning"] = reasoning_meta
        except Exception:
            logger.debug("Reasoning lookup failed for draft_campaign vendor=%s", vendor_name)

        row = await pool.fetchrow(
            """
            INSERT INTO b2b_campaigns
                (company_name, vendor_name, channel, subject, body, cta,
                 opportunity_score, urgency_score, llm_model, status, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'draft', $10::jsonb)
            RETURNING id, created_at
            """,
            company_name.strip(),
            vendor_name.strip(),
            channel,
            subject.strip() if subject else None,
            body.strip(),
            cta.strip() if cta else None,
            opportunity_score,
            urgency_score,
            llm_model.strip(),
            json.dumps(metadata),
        )

        result = {
            "success": True,
            "campaign_id": str(row["id"]),
            "status": "draft",
            "created_at": str(row["created_at"]),
        }
        if blog_posts:
            result["blog_posts"] = blog_posts

        return json.dumps(result)
    except Exception:
        logger.exception("draft_campaign error")
        return json.dumps({"success": False, "error": "Internal error"})
