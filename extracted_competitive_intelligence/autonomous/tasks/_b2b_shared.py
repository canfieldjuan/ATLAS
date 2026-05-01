"""Phase 1 bridge: re-exports atlas_brain.autonomous.tasks._b2b_shared.

``from X import *`` does not pull underscore-prefixed names per Python
semantics, but downstream scaffolded modules (b2b_battle_cards.py,
b2b_vendor_briefing.py) import private helpers like
``_align_vendor_intelligence_record_to_scorecard`` directly. We enumerate
the underscore names explicitly so ``from ._b2b_shared import _foo``
inside the scaffold resolves at runtime. Phase 2 replaces this bridge
with a standalone implementation gated on EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from atlas_brain.autonomous.tasks._b2b_shared import *  # noqa: F401,F403
from atlas_brain.autonomous.tasks._b2b_shared import (  # noqa: F401
    _align_vendor_intelligence_record_to_scorecard,
    _timing_summary_payload,
    _reasoning_int,
    read_vendor_company_signal_review_queue,
    read_vendor_intelligence_record,
    read_vendor_intelligence,
    read_vendor_scorecard_detail,
    read_vendor_quote_evidence,
)
