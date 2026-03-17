"""Cross-vendor reasoning layer for B2B churn intelligence.

Runs after per-vendor stratified reasoning completes. Three analysis modes:
  A. Pairwise battle reasoning (why is A losing to B?)
  B. Category council reasoning (market regime, structural dynamics)
  C. Resource-asymmetry reasoning (weakness = choice or inability?)

Consumes per-vendor evidence + reasoning results + ecosystem data, produces
cross-vendor conclusions for scorecards, battle cards, and executive summaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from .llm_utils import resolve_stratified_llm
from .semantic_cache import CacheEntry, SemanticCache

logger = logging.getLogger("atlas.reasoning.cross_vendor")

# ---------------------------------------------------------------------------
# JSON schema for structured cross-vendor output
# ---------------------------------------------------------------------------

CROSS_VENDOR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "analysis_type",
        "vendors",
        "conclusion",
        "confidence",
        "key_insights",
        "durability_assessment",
        "falsification_conditions",
    ],
    "properties": {
        "analysis_type": {
            "type": "string",
            "enum": ["pairwise_battle", "category_council", "resource_asymmetry"],
        },
        "vendors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "conclusion": {
            "type": "string",
            "description": "3-5 sentence synthesis of the cross-vendor analysis",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "durability_assessment": {
            "type": "string",
            "enum": ["structural", "cyclical", "temporary", "uncertain"],
        },
        "winner": {
            "type": ["string", "null"],
        },
        "loser": {
            "type": ["string", "null"],
        },
        "segment_dynamics": {
            "type": ["object", "null"],
            "properties": {
                "enterprise_winner": {"type": ["string", "null"]},
                "smb_winner": {"type": ["string", "null"]},
                "segment_divergence": {"type": "boolean"},
            },
        },
        "market_regime": {
            "type": ["string", "null"],
        },
        "falsification_conditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertainty_sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_BATTLE_SYSTEM_PROMPT = """\
You are comparing two B2B software vendors in a competitive displacement relationship.
Vendor A is losing customers to Vendor B. Your job is to explain WHY, whether it is
DURABLE, and what EVIDENCE would change the conclusion.

Key questions to answer:
1. Is A losing because of a product gap, pricing mismatch, or trust/compliance failure?
2. Is B winning on merit, or capturing overflow from A's self-inflicted problems?
3. Is B's gain concentrated in a specific buyer segment (enterprise vs SMB, DM vs non-DM)?
4. Does A have the resources (review share, integration depth, enterprise trust) to recover?
5. Would fixing A's top pain category materially reduce churn, or is the underlying cause deeper?

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

_CATEGORY_COUNCIL_SYSTEM_PROMPT = """\
You are analyzing the competitive landscape of a B2B software category.
You have metrics for each vendor plus the displacement graph between them.

Key questions:
1. What is the market REGIME? (feature_competition, price_competition, trust_compliance,
   platform_consolidation, ai_disruption, fragmentation)
2. Are displacement flows structural (market share transfer) or cyclical (experimentation)?
3. Which vendors are gaining vs losing in each buyer SEGMENT (enterprise, mid-market, SMB)?
4. Is any vendor positioned as a category default, or is the market fragmenting?
5. What common cause (if any) explains correlated churn across multiple vendors?

GROUNDING RULES:
- Every key_insight MUST cite a specific metric and value from the evidence.
- conclusion MUST reference at least 3 specific numbers from the evidence.
- market_regime must be one concrete term, not a vague description.
- winner/loser: name the single strongest gainer and loser, or null if no clear one.
- falsification_conditions: what specific evidence would prove this analysis wrong?
- Do not invent data not present in the evidence.

Output ONLY valid JSON matching the schema provided.\
"""

_ASYMMETRY_SYSTEM_PROMPT = """\
Two vendors show similar churn pressure scores but differ in resource signals.
Your job is to assess whether each vendor's weakness is a CHOICE (prioritization,
margin defense, deliberate segment shedding) or an INABILITY (execution failure,
capital constraint, talent loss).

Key resource indicators to compare:
- Review share (proxy for installed base)
- typical_company_size distribution (enterprise tilt = pricing power + switching cost)
- Integration count (lock-in depth)
- buyer_authority mix (DM churn rate vs non-DM)
- Displacement direction (net gainer vs net loser)
- Insider signals (talent drain, leadership quality -- when available)

GROUNDING RULES:
- Every key_insight MUST cite a specific metric and value from the evidence.
- conclusion MUST reference at least 3 specific numbers from the evidence.
- durability_assessment: "structural" if resource gap is deep; "temporary" if fixable.
- winner: which vendor is better positioned long-term, or null.
- falsification_conditions: what specific evidence would prove this analysis wrong?
- Do not invent data not present in the evidence.

Output ONLY valid JSON matching the schema provided.\
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrossVendorResult:
    """Result of a cross-vendor reasoning analysis."""

    analysis_type: str  # "pairwise_battle", "category_council", "resource_asymmetry"
    vendors: list[str]
    conclusion: dict[str, Any]  # full JSON output
    confidence: float
    evidence_hash: str
    tokens_used: int
    cached: bool
    pattern_sig: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _combined_evidence_hash(*dicts: dict[str, Any]) -> str:
    """Compute a deterministic hash from multiple evidence dicts."""
    combined = json.dumps(dicts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Trim evidence dict for LLM payload (cap lists, round floats)."""
    out: dict[str, Any] = {}
    for k, v in evidence.items():
        if isinstance(v, list) and len(v) > 5:
            out[k] = v[:5]
        elif isinstance(v, float):
            out[k] = round(v, 2)
        else:
            out[k] = v
    return out


def _build_battle_payload(
    vendor_a: str,
    vendor_b: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    displacement_edge: dict[str, Any],
    profile_a: dict[str, Any] | None = None,
    profile_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the user payload for a pairwise battle LLM call."""
    payload: dict[str, Any] = {
        "vendor_a": {"name": vendor_a, **_compact_evidence(evidence_a)},
        "vendor_b": {"name": vendor_b, **_compact_evidence(evidence_b)},
        "displacement": _compact_evidence(displacement_edge),
    }
    if profile_a:
        payload["vendor_a"]["product_profile"] = _compact_evidence(profile_a)
    if profile_b:
        payload["vendor_b"]["product_profile"] = _compact_evidence(profile_b)
    return payload


def _build_category_payload(
    category: str,
    vendor_evidence: dict[str, dict[str, Any]],
    ecosystem: dict[str, Any],
    displacement_flows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the user payload for a category council LLM call."""
    vendors = []
    for vname, ev in vendor_evidence.items():
        vendors.append({"name": vname, **_compact_evidence(ev)})
    flows = []
    for f in displacement_flows[:15]:
        flows.append(_compact_evidence(f))
    return {
        "category": category,
        "ecosystem": _compact_evidence(ecosystem),
        "vendors": vendors,
        "displacement_flows": flows,
    }


def _build_asymmetry_payload(
    vendor_a: str,
    vendor_b: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    profile_a: dict[str, Any] | None = None,
    profile_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the user payload for a resource-asymmetry LLM call."""
    payload: dict[str, Any] = {
        "vendor_a": {"name": vendor_a, **_compact_evidence(evidence_a)},
        "vendor_b": {"name": vendor_b, **_compact_evidence(evidence_b)},
    }
    if profile_a:
        payload["vendor_a"]["product_profile"] = _compact_evidence(profile_a)
    if profile_b:
        payload["vendor_b"]["product_profile"] = _compact_evidence(profile_b)
    return payload


# ---------------------------------------------------------------------------
# Core reasoner
# ---------------------------------------------------------------------------

class CrossVendorReasoner:
    """Cross-vendor LLM reasoning across battles, categories, and asymmetry pairs."""

    def __init__(self, cache: SemanticCache) -> None:
        self._cache = cache

    async def _call_llm(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        pattern_sig: str,
        evidence_hash: str,
        analysis_type: str,
        vendors: list[str],
    ) -> CrossVendorResult:
        """Shared LLM call with cache lookup/store."""
        # --- Try cache recall ---
        cached_entry = await self._cache.lookup(pattern_sig)
        if cached_entry is not None and cached_entry.evidence_hash == evidence_hash:
            logger.info("Cross-vendor cache hit: %s", pattern_sig)
            conclusion = cached_entry.conclusion
            return CrossVendorResult(
                analysis_type=analysis_type,
                vendors=vendors,
                conclusion=conclusion,
                confidence=cached_entry.effective_confidence or cached_entry.confidence,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=True,
                pattern_sig=pattern_sig,
            )

        # --- LLM call ---
        from ..pipelines.llm import parse_json_response, trace_llm_call
        from ..services.protocols import Message
        from ..services.tracing import build_business_trace_context
        from .config import ReasoningConfig

        _rcfg = ReasoningConfig()
        llm = resolve_stratified_llm(_rcfg)
        if llm is None:
            logger.error("No LLM available for cross-vendor reasoning")
            return CrossVendorResult(
                analysis_type=analysis_type,
                vendors=vendors,
                conclusion={"error": "no_llm_available"},
                confidence=0.0,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
                pattern_sig=pattern_sig,
            )

        messages = [
            Message(role="system", content=system_prompt),
            Message(
                role="user",
                content=json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
            ),
        ]

        trace_metadata = {
            "workflow": "cross_vendor_reasoning",
            "phase": f"cross_vendor_{analysis_type}",
            "vendors": vendors,
        }
        business = build_business_trace_context(
            workflow="cross_vendor_reasoning",
            vendor_name=vendors[0] if vendors else None,
        )
        if business:
            trace_metadata["business"] = business

        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(
                llm.chat,
                messages=messages,
                max_tokens=_rcfg.max_tokens,
                temperature=_rcfg.temperature,
                guided_json=CROSS_VENDOR_JSON_SCHEMA,
                response_format={"type": "json_object"},
            )
            text = result.get("response", "").strip()
            usage = result.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            trace_meta = result.get("_trace_meta", {})
            duration_ms = (time.monotonic() - t0) * 1000

            trace_llm_call(
                span_name=f"reasoning.cross_vendor.{analysis_type}",
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                model=getattr(llm, "model", getattr(llm, "model_id", "")),
                provider=getattr(llm, "name", ""),
                duration_ms=duration_ms,
                metadata=trace_metadata,
                input_data={"messages": [{"role": m.role, "content": m.content[:500]} for m in messages]},
                output_data={"response": text[:2000]} if text else None,
                api_endpoint=trace_meta.get("api_endpoint"),
                provider_request_id=trace_meta.get("provider_request_id"),
                ttft_ms=trace_meta.get("ttft_ms"),
                inference_time_ms=trace_meta.get("inference_time_ms"),
                queue_time_ms=trace_meta.get("queue_time_ms"),
            )

            # Clean think tags
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            conclusion = parse_json_response(text, recover_truncated=True)

        except Exception:
            logger.exception("Cross-vendor LLM call failed for %s", pattern_sig)
            return CrossVendorResult(
                analysis_type=analysis_type,
                vendors=vendors,
                conclusion={"error": "llm_failed"},
                confidence=0.0,
                evidence_hash=evidence_hash,
                tokens_used=0,
                cached=False,
                pattern_sig=pattern_sig,
            )

        confidence = conclusion.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        # --- Store in cache ---
        try:
            entry = CacheEntry(
                pattern_sig=pattern_sig,
                pattern_class=f"cross_vendor_{analysis_type}",
                conclusion=conclusion,
                confidence=confidence,
                falsification_conditions=conclusion.get("falsification_conditions", []),
                uncertainty_sources=conclusion.get("uncertainty_sources", []),
                vendor_name=vendors[0] if vendors else None,
                evidence_hash=evidence_hash,
            )
            await self._cache.store(entry)
        except Exception:
            logger.debug("Failed to cache cross-vendor result %s", pattern_sig, exc_info=True)

        return CrossVendorResult(
            analysis_type=analysis_type,
            vendors=vendors,
            conclusion=conclusion,
            confidence=confidence,
            evidence_hash=evidence_hash,
            tokens_used=tokens,
            cached=False,
            pattern_sig=pattern_sig,
        )

    # ----- Mode A: Pairwise Battle -----

    async def analyze_battle(
        self,
        vendor_a: str,
        vendor_b: str,
        *,
        evidence_a: dict[str, Any],
        evidence_b: dict[str, Any],
        displacement_edge: dict[str, Any],
        product_profile_a: dict[str, Any] | None = None,
        product_profile_b: dict[str, Any] | None = None,
    ) -> CrossVendorResult:
        """Pairwise battle reasoning: why is vendor_a losing to vendor_b?"""
        sorted_pair = sorted([vendor_a, vendor_b])
        eh = _combined_evidence_hash(evidence_a, evidence_b, displacement_edge)
        pattern_sig = f"xv:battle:{sorted_pair[0]}:{sorted_pair[1]}:{eh}"

        payload = _build_battle_payload(
            vendor_a, vendor_b, evidence_a, evidence_b,
            displacement_edge, product_profile_a, product_profile_b,
        )
        return await self._call_llm(
            _BATTLE_SYSTEM_PROMPT, payload, pattern_sig, eh,
            "pairwise_battle", [vendor_a, vendor_b],
        )

    # ----- Mode B: Category Council -----

    async def analyze_category(
        self,
        category: str,
        *,
        vendor_evidence: dict[str, dict[str, Any]],
        ecosystem: dict[str, Any],
        displacement_flows: list[dict[str, Any]],
    ) -> CrossVendorResult:
        """Category council reasoning: market regime and structural dynamics."""
        eh = _combined_evidence_hash(
            vendor_evidence, ecosystem, {"flows": displacement_flows},
        )
        cat_safe = category.lower().replace(" ", "_")
        pattern_sig = f"xv:category:{cat_safe}:{eh}"

        payload = _build_category_payload(
            category, vendor_evidence, ecosystem, displacement_flows,
        )
        vendors = sorted(vendor_evidence.keys())
        return await self._call_llm(
            _CATEGORY_COUNCIL_SYSTEM_PROMPT, payload, pattern_sig, eh,
            "category_council", vendors,
        )

    # ----- Mode C: Resource Asymmetry -----

    async def analyze_asymmetry(
        self,
        vendor_a: str,
        vendor_b: str,
        *,
        evidence_a: dict[str, Any],
        evidence_b: dict[str, Any],
        profile_a: dict[str, Any] | None = None,
        profile_b: dict[str, Any] | None = None,
    ) -> CrossVendorResult:
        """Resource-asymmetry reasoning: choice vs inability assessment."""
        sorted_pair = sorted([vendor_a, vendor_b])
        eh = _combined_evidence_hash(evidence_a, evidence_b)
        pattern_sig = f"xv:asymmetry:{sorted_pair[0]}:{sorted_pair[1]}:{eh}"

        payload = _build_asymmetry_payload(
            vendor_a, vendor_b, evidence_a, evidence_b, profile_a, profile_b,
        )
        return await self._call_llm(
            _ASYMMETRY_SYSTEM_PROMPT, payload, pattern_sig, eh,
            "resource_asymmetry", [vendor_a, vendor_b],
        )
