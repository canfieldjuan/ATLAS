"""
API routers for Atlas Brain.
"""

from fastapi import APIRouter

from .alerts import router as alerts_router
from .comms import router as comms_router
from .devices import router as devices_router
from .health import router as health_router
from .llm import router as llm_router
from .models import router as models_router
from .query import router as query_router
from .session import router as session_router
from .vision import router as vision_router
from .video import router as video_router
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
from .b2b_affiliates import router as b2b_affiliates_router
from .b2b_campaigns import router as b2b_campaigns_router
from .b2b_dashboard import router as b2b_dashboard_router
from .vendor_targets import router as vendor_targets_router
from .consumer_dashboard import router as consumer_dashboard_router
from .seller_campaigns import router as seller_campaigns_router

# Main router that aggregates all sub-routers
router = APIRouter()

router.include_router(health_router)
router.include_router(query_router)
router.include_router(models_router)
router.include_router(devices_router)
router.include_router(alerts_router)
router.include_router(comms_router)
router.include_router(llm_router)
router.include_router(session_router)
router.include_router(vision_router)
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
router.include_router(b2b_affiliates_router)
router.include_router(b2b_campaigns_router)
router.include_router(b2b_dashboard_router)
router.include_router(vendor_targets_router)
router.include_router(consumer_dashboard_router)
router.include_router(seller_campaigns_router)
