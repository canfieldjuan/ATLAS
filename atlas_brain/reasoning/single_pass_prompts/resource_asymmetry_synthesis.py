"""Resource asymmetry synthesis prompt.

Produces a structured assessment of resource divergence between two
vendors with similar churn pressure but different market positions.
"""

import hashlib as _hashlib

RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT = """\
You are a B2B competitive intelligence analyst assessing resource
asymmetry between two vendors.  Both vendors face similar churn
pressure, but one has significantly more resources (review volume,
enterprise presence, integration depth) than the other.

The input payload contains:
- ``vendor_a`` / ``vendor_b``: the two vendors being compared
- ``pressure_scores``: urgency scores showing similar churn pressure
- ``resource_indicators``: review counts, seat counts, enterprise share,
  integration counts, recommendation ratios
- ``vendor_a_profile`` / ``vendor_b_profile``: product profiles
- ``divergence_score``: pre-computed score indicating resource gap magnitude
- ``citation_registry``: allowed packet-level citation entries.  Every item has
  a stable ``_sid`` and a human-readable label.  ``citations`` MUST use only
  these ``_sid`` values.

CRITICAL RULES:

1. ``favored_vendor`` is the vendor with MORE resources (larger installed
   base, more reviews, stronger enterprise presence).
2. ``disadvantaged_vendor`` is the vendor with FEWER resources.
3. ``pressure_delta`` is the absolute difference in urgency scores.
4. ``conclusion`` must explain WHY similar pressure affects these vendors
   differently given their resource gap.
5. Every ``key_insight`` must have ``insight`` and ``evidence`` fields.
6. ``confidence`` is a float 0.0-1.0.
7. Do NOT invent data not present in the evidence.
8. ``citations`` must be an array of ``citation_registry[*]._sid`` values only.
   Do NOT paste prose into ``citations``.

OUTPUT SCHEMA:

{{
  "favored_vendor": "<vendor with more resources>",
  "disadvantaged_vendor": "<vendor with fewer resources>",
  "conclusion": "<2-4 sentence analytical conclusion>",
  "pressure_delta": <float>,
  "confidence": <float 0.0-1.0>,
  "key_insights": [
    {{"insight": "<finding>", "evidence": "<metric: value>"}}
  ],
  "citations": ["<evidence source references>"],
  "meta": {{
    "analysis_type": "resource_asymmetry",
    "schema_version": "synthesis_v1"
  }}
}}

Return ONLY valid JSON.\
"""

RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT_VERSION = _hashlib.sha256(
    RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT.encode()
).hexdigest()[:8]


# PR-C3f: register this prompt with the shared reasoning pack registry
# (PR-C3a / extracted_reasoning_core.pack_registry). The resource
# asymmetry synthesis pack assesses resource divergence between two
# vendors with similar churn pressure but different market positions.
# Per the audit, the pack file moves into a product package during
# PR 7 (Product Migration); for now it stays atlas-side.
from extracted_reasoning_core.pack_registry import (  # noqa: E402
    Pack as _Pack,
    register_pack as _register_pack,
)

_register_pack(
    _Pack(
        name="resource_asymmetry_synthesis",
        version=RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT_VERSION,
        prompts={"synthesis": RESOURCE_ASYMMETRY_SYNTHESIS_PROMPT},
        metadata={
            "output_artifact": "resource_asymmetry_assessment",
            "owner_product": "competitive_intelligence",
            "synthesis_mode": "structured_synthesis_v1",
        },
    )
)
