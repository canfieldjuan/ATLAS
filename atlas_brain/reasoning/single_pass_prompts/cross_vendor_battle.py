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
After your analysis, you MUST set two fields:
- winner: the vendor that is GAINING share in this pairwise matchup. This is the vendor
  that customers are moving TO based on the displacement evidence.
- loser: the vendor that is LOSING share. This is the vendor that customers are moving
  AWAY FROM.
Determine winner/loser from displacement direction, not from who has more reviews or
higher ratings. If reviews show customers switching from A to B, then B is the winner
and A is the loser. If the direction is ambiguous or bidirectional, set winner to the
vendor with the stronger net inflow based on switch_count and mention volume, and note
the ambiguity in your conclusion.

KEY INSIGHTS FORMAT:
Each key_insight must be an object with "insight" (the finding) and "evidence" (the
specific metric or data point that supports it). Example:
  {"insight": "Pricing is the top churn driver", "evidence": "price_complaint_rate: 0.34 (34% of reviews)"}

SELF-CHECK (perform before outputting):
1. Does the declared winner have better displacement metrics than the loser?
   If the loser has improving velocity_churn_density while the winner is worsening, reconsider.
2. If the loser has a recommend_ratio 20+ points higher than the winner, justify or revise.
3. Are ALL key_insights grounded in specific metrics from the evidence?
4. Does the durability_assessment match the evidence? "structural" requires deep market forces,
   "temporary" requires a plausible 1-2 quarter fix.
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
