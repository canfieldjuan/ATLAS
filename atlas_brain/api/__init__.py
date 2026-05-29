"""
API routers for Atlas Brain.
"""

import logging

from fastapi import APIRouter, HTTPException

from .alerts import router as alerts_router
from .comms import router as comms_router
from .devices import router as devices_router
from .health import router as health_router
from .llm import router as llm_router
from .query import router as query_router
from .session import router as session_router
from .vision import router as vision_router
from .recognition import router as recognition_router
from .speaker import router as speaker_router
from .identity import router as identity_router
from .edge import router as edge_router
from .orchestrated import router as orchestrated_router
from .autonomous import router as autonomous_router
from .presence import router as presence_router
from .proactive_actions import router as proactive_actions_router
from .email_drafts import router as email_drafts_router
from .email_actions import router as email_actions_router
from .inbox_rules import router as inbox_rules_router
from .invoicing import router as invoicing_router
from .contacts import router as contacts_router
from .reasoning import router as reasoning_router
from .security import router as security_router
from .settings import router as settings_router
from .system import router as system_router
from .b2b_reviews import router as b2b_reviews_router
from .b2b_scrape import router as b2b_scrape_router
from .intelligence import router as intelligence_router
from .b2b_affiliates import tenant_router as b2b_tenant_affiliates_router
from .b2b_campaigns import router as b2b_campaigns_router
from .b2b_tenant_dashboard import (
    router as b2b_tenant_router,
)
from .vendor_targets import router as vendor_targets_router
from .consumer_dashboard import router as consumer_dashboard_router
from .seller_campaigns import router as seller_campaigns_router
from .api_keys import router as api_keys_router
from .auth import router as auth_router
from .billing import router as billing_router
from .blog_admin import router as blog_admin_router
from .byok_keys import router as byok_keys_router
from .llm_gateway import router as llm_gateway_router
from .blog_public import router as blog_public_router
from .prospects import router as prospects_router
from .b2b_vendor_briefing import router as b2b_vendor_briefing_router
from .b2b_crm_events import router as b2b_crm_events_router
from .admin_costs import router as admin_costs_router
from .universal_scrape import router as universal_scrape_router
from .pipeline_visibility import router as pipeline_visibility_router
from .b2b_win_loss import router as b2b_win_loss_router
from .b2b_evidence import router as b2b_evidence_router
from .b2b_vendor_claims import router as b2b_vendor_claims_router
from .b2b_challenger_claims import router as b2b_challenger_claims_router
from fastapi import Depends
from ..auth.dependencies import require_b2b_plan

logger = logging.getLogger("atlas.api")

try:
    from .models import router as models_router
except Exception as exc:
    models_router = None
    logger.warning("Models router disabled during api package import: %s", exc)

try:
    from .video import router as video_router
except Exception as exc:
    video_router = None
    logger.warning("Video router disabled during api package import: %s", exc)

# Main router that aggregates all sub-routers
router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)
if models_router is not None:
    router.include_router(models_router)
router.include_router(devices_router)
router.include_router(alerts_router)
router.include_router(comms_router)
router.include_router(llm_router)
router.include_router(session_router)
router.include_router(vision_router)
if video_router is not None:
    router.include_router(video_router)
router.include_router(recognition_router)
router.include_router(speaker_router)
router.include_router(identity_router)
router.include_router(edge_router)
router.include_router(orchestrated_router)
router.include_router(autonomous_router)
router.include_router(presence_router)
router.include_router(proactive_actions_router)
router.include_router(email_drafts_router)
router.include_router(email_actions_router)
router.include_router(inbox_rules_router)
router.include_router(invoicing_router)
router.include_router(contacts_router)
router.include_router(reasoning_router)
router.include_router(security_router)
router.include_router(system_router)
router.include_router(settings_router)
router.include_router(b2b_reviews_router)
router.include_router(b2b_scrape_router)
router.include_router(intelligence_router)
router.include_router(b2b_tenant_affiliates_router)
router.include_router(b2b_campaigns_router)
router.include_router(b2b_tenant_router)
router.include_router(vendor_targets_router)
router.include_router(consumer_dashboard_router)
router.include_router(seller_campaigns_router)
router.include_router(api_keys_router)
router.include_router(auth_router)
router.include_router(billing_router)
router.include_router(byok_keys_router)
router.include_router(llm_gateway_router)
router.include_router(blog_admin_router)
router.include_router(blog_public_router)
router.include_router(prospects_router)
router.include_router(b2b_vendor_briefing_router)
router.include_router(b2b_crm_events_router)
router.include_router(admin_costs_router)
router.include_router(universal_scrape_router)
router.include_router(pipeline_visibility_router)
router.include_router(b2b_win_loss_router)
router.include_router(b2b_evidence_router)
router.include_router(b2b_vendor_claims_router)
router.include_router(b2b_challenger_claims_router)

# AI Content Ops control-surface routes (preview / plan / execute /
# control-surfaces). Mounted with no execution_services_provider yet
# -- preview / plan / GET control-surfaces work; execute correctly
# returns 503 ("Content Ops execution services are not configured.")
# until the host wires execution services in a follow-up slice.
#
# Auth: gated behind require_b2b_plan("b2b_growth") -- same dependency
# the existing /api/v1/b2b/campaigns router uses, since Content Ops is
# the same B2B audience. Without this every /api/v1/content-ops/*
# endpoint would be reachable without a token (the frontend's
# ProtectedRoute is UI-only gating).
#
# The `extracted_content_pipeline` import is inside the try so the
# host's prod Docker image (which copies only ./atlas_brain) doesn't
# crash at startup; the image installs the extracted package via the
# requirements path when present, otherwise the content-ops surface
# is logged as disabled and existing routes keep serving.
try:
    from extracted_content_pipeline.api.control_surfaces import (
        ContentOpsControlSurfaceApiConfig,
        create_content_ops_control_surface_router,
    )
    from .._content_ops_import_admission import (
        build_content_ops_import_admission_gate,
    )
    from extracted_content_pipeline.api.generated_assets import (
        create_generated_asset_router,
        create_public_landing_page_router,
    )
    from extracted_content_pipeline.api.faq_search import (
        create_faq_deflection_search_router,
    )
    from .._content_ops_input_provider import build_content_ops_input_provider
    from .._content_ops_infrastructure import (
        build_content_ops_llm_client,
        build_content_ops_skill_store,
    )
    from .._content_ops_macro_writeback import (
        build_content_ops_macro_publish_provider,
    )
    from .._content_ops_services import build_content_ops_execution_services
    from .._content_ops_scope import (
        build_content_ops_scope,
        set_current_auth_user,
    )
    from .._content_ops_reasoning import (
        describe_content_ops_reasoning_context_provider,
        select_content_ops_reasoning_context_provider,
    )
    from ..auth.dependencies import AuthUser
    from ..config import settings
    from ..storage.database import get_db_pool

    async def _capture_content_ops_auth_user(
        user: AuthUser = Depends(require_b2b_plan("b2b_growth")),
    ) -> AuthUser:
        """Bridge the per-request AuthUser to the
        `_content_ops_scope` ContextVar so the route's
        `scope_provider` can read it. Composing on top of
        `require_b2b_plan` keeps the existing auth gate (paying
        tier check, B2B product check, past-due guard) intact.
        Closes the Codex P1 cross-tenant safety issue from
        PR #454 (E2): drafts now persist under the
        authenticated tenant's account_id rather than empty
        string.
        """

        set_current_auth_user(user)
        return user

    async def _require_content_ops_usage_operator(
        user: AuthUser = Depends(_capture_content_ops_auth_user),
    ) -> AuthUser:
        if bool(getattr(user, "is_platform_admin", False)):
            return user
        raise HTTPException(status_code=403, detail="Platform admin access required")

    def _content_ops_cache_policy_default(_scope: object) -> str | None:
        policy = str(
            getattr(
                settings.b2b_campaign,
                "content_ops_cache_policy_default",
                "",
            )
            or ""
        ).strip()
        return policy or None

    content_ops_config = ContentOpsControlSurfaceApiConfig()
    content_ops_router = create_content_ops_control_surface_router(
        config=content_ops_config,
        dependencies=[Depends(_capture_content_ops_auth_user)],
        execution_services_provider=lambda: (
            build_content_ops_execution_services(enable_db_services=True)
        ),
        scope_provider=build_content_ops_scope,
        reasoning_context_provider=select_content_ops_reasoning_context_provider,
        reasoning_status_provider=describe_content_ops_reasoning_context_provider,
        input_provider=build_content_ops_input_provider(pool_provider=get_db_pool),
        cache_policy_default_provider=_content_ops_cache_policy_default,
        opportunity_import_pool_provider=get_db_pool,
        usage_pool_provider=get_db_pool,
        usage_dependencies=[Depends(_require_content_ops_usage_operator)],
        ingestion_import_admission_provider=lambda: (
            build_content_ops_import_admission_gate(
                max_concurrency=content_ops_config.ingestion_import_max_concurrency
            )
        ),
    )
    router.include_router(content_ops_router)
    content_assets_router = create_generated_asset_router(
        pool_provider=get_db_pool,
        scope_provider=build_content_ops_scope,
        llm_provider=build_content_ops_llm_client,
        skills_provider=build_content_ops_skill_store,
        macro_publish_provider=lambda: build_content_ops_macro_publish_provider(
            pool_provider=get_db_pool,
        ),
        dependencies=[Depends(_capture_content_ops_auth_user)],
    )
    router.include_router(content_assets_router)
    faq_search_router = create_faq_deflection_search_router(
        pool_provider=get_db_pool,
        scope_provider=build_content_ops_scope,
        dependencies=[Depends(_capture_content_ops_auth_user)],
    )
    router.include_router(faq_search_router)
    public_landing_page_router = create_public_landing_page_router(
        pool_provider=get_db_pool,
    )
    router.include_router(public_landing_page_router)
except Exception as exc:  # pragma: no cover - defensive at import time
    logger.warning(
        "Content Ops router disabled during api package import: %s", exc
    )
