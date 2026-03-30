"""Cross-vendor pairwise battle synthesis prompt.

Produces a structured battle conclusion from deterministic displacement
evidence, product profiles, and pool-layer context.  Output shape is
intentionally parallel to legacy CrossVendorReasoner battle output so
consumers can migrate transparently.
"""

import hashlib as _hashlib

CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT = """\
You are a B2B competitive intelligence analyst producing a pairwise
battle conclusion.  You receive deterministic evidence about two vendors
in a competitive displacement relationship.

The input payload contains:
- ``locked_direction``: which vendor is gaining and which is losing share.
  You MUST respect this direction.  Do NOT flip winner/loser.
- ``displacement_edge``: mention counts, signal strength, primary driver,
  evidence breakdown (explicit switches, active evaluations, implied preferences)
- ``vendor_a_profile`` / ``vendor_b_profile``: product profiles with strengths,
  weaknesses, integrations, use cases, typical company size, typical industries
- ``vendor_a_pool`` / ``vendor_b_pool``: scored pool summaries (pain distribution,
  urgency, competitor flows, budget pressure, segment data)

CRITICAL RULES:

1. ``winner`` and ``loser`` MUST match ``locked_direction`` exactly.
2. ``conclusion`` must reference at least 3 specific numbers from the evidence.
3. Every ``key_insight`` must be an object with ``insight`` and ``evidence`` fields.
   The ``evidence`` field must cite a specific metric and value.
4. ``durability_assessment`` must be one of: structural, cyclical, temporary, uncertain.
   - structural: deep market forces make reversal unlikely
   - cyclical: tied to product/pricing cycle
   - temporary: fixable in 1-2 quarters
   - uncertain: insufficient evidence
5. ``confidence`` is a float 0.0-1.0.
6. Do NOT invent data not present in the evidence.
7. ``falsification_conditions``: what specific evidence would prove this wrong?

OUTPUT SCHEMA:

{{
  "winner": "<vendor gaining share>",
  "loser": "<vendor losing share>",
  "conclusion": "<2-4 sentence analytical conclusion citing specific metrics>",
  "confidence": <float 0.0-1.0>,
  "durability_assessment": "<structural|cyclical|temporary|uncertain>",
  "key_insights": [
    {{"insight": "<finding>", "evidence": "<metric: value>"}}
  ],
  "falsification_conditions": ["<what would disprove this>"],
  "citations": ["<evidence source references>"],
  "meta": {{
    "analysis_type": "pairwise_battle",
    "schema_version": "synthesis_v1"
  }}
}}

Return ONLY valid JSON.\
"""

CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT_VERSION = _hashlib.sha256(
    CROSS_VENDOR_BATTLE_SYNTHESIS_PROMPT.encode()
).hexdigest()[:8]
