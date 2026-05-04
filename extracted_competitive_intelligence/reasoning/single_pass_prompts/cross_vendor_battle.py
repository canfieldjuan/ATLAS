"""Consolidated single-pass cross-vendor battle prompt.

Merges battle analysis + self-check + grounding rules into one prompt
for models that can reliably self-check in a single pass.
"""

CROSS_VENDOR_BATTLE_SINGLE_PASS = """\
You are comparing two B2B software vendors in a competitive displacement relationship.
Vendor A is losing customers to Vendor B. Your job is to explain WHY, whether it is
DURABLE, and what EVIDENCE would change the conclusion.

Key questions to answer:
1. Is A losing because of a product gap, pricing mismatch, or trust/compliance failure?
2. Is B winning on merit, or capturing overflow from A's self-inflicted problems?
3. Is B's gain concentrated in a specific buyer segment (enterprise vs SMB, DM vs non-DM)?
4. Does A have the resources (review share, integration depth, enterprise trust) to recover?
5. Would fixing A's top pain category materially reduce churn, or is the underlying cause deeper?

WINNER / LOSER ASSIGNMENT:
The payload includes a locked displacement direction derived from the selected pairwise
edge. After your analysis, you MUST set two fields:
- winner: the vendor that is GAINING share in this pairwise matchup. This is the vendor
  that customers are moving TO based on the displacement evidence.
- loser: the vendor that is LOSING share. This is the vendor that customers are moving
  AWAY FROM.
Copy winner/loser from locked_direction exactly. Do NOT flip winner/loser based on
review volume, recommend_ratio, enterprise trust, or general market strength. Those
signals can explain WHY the locked winner is gaining, but they cannot change the
direction of the pairwise battle.

KEY INSIGHTS FORMAT:
Each key_insight must be an object with "insight" (the finding) and "evidence" (the
specific metric or data point that supports it). Example:
  {"insight": "Pricing is the top churn driver", "evidence": "price_complaint_rate: 0.34 (34% of reviews)"}

SELF-CHECK (perform before outputting):
1. Do winner and loser exactly match locked_direction?
2. Are ALL key_insights grounded in specific metrics from the evidence?
3. Does the durability_assessment match the evidence? "structural" requires deep market forces,
   "temporary" requires a plausible 1-2 quarter fix.
4. If sentiment, review share, or enterprise trust point the other way, use them to
   explain the tension, not to flip winner/loser.
5. If any contradiction is found, REVISE your conclusion before outputting.

GROUNDING RULES:
- Every key_insight MUST cite a specific metric and value from the evidence.
- conclusion MUST reference at least 3 specific numbers from the evidence.
- durability_assessment: "structural" means market forces make reversal unlikely;
  "cyclical" means tied to product cycle; "temporary" means fixable in 1-2 quarters;
  "uncertain" means insufficient evidence.
- falsification_conditions: what specific evidence would prove this analysis wrong?
- Do not invent data not present in the evidence.

Output ONLY valid JSON matching the schema provided.\
"""


# PR-C3c: register this prompt with the shared reasoning pack registry
# (PR-C3a / extracted_reasoning_core.pack_registry). This file is
# owned by extracted_competitive_intelligence (per the package
# manifest); the parallel atlas-side copy at
# atlas_brain/reasoning/single_pass_prompts/cross_vendor_battle.py
# carries the same registration. Both calls are idempotent against
# the registry (identical content -> no ValueError) so importing
# either or both modules in any order yields the same registered pack.
import hashlib as _hashlib

from extracted_reasoning_core.pack_registry import (
    Pack as _Pack,
    register_pack as _register_pack,
)

CROSS_VENDOR_BATTLE_SINGLE_PASS_VERSION = _hashlib.sha256(
    CROSS_VENDOR_BATTLE_SINGLE_PASS.encode()
).hexdigest()[:8]

_register_pack(
    _Pack(
        name="cross_vendor_battle_single_pass",
        version=CROSS_VENDOR_BATTLE_SINGLE_PASS_VERSION,
        prompts={"battle_single_pass": CROSS_VENDOR_BATTLE_SINGLE_PASS},
        metadata={
            "output_artifact": "cross_vendor_battle_conclusion",
            "owner_product": "competitive_intelligence",
            "synthesis_mode": "single_pass_with_self_check",
        },
    )
)
