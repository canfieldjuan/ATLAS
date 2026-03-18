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
from .multi_pass import multi_pass_reason
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
        "resource_advantage": {
            "type": ["string", "null"],
            "description": "Which vendor holds the resource advantage and why (1 sentence). Null if parity.",
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
- resource_advantage: one sentence naming which vendor holds the resource edge and
  the primary reason (e.g., "Vendor A holds the resource advantage due to 3x review
  share and deeper enterprise penetration"). Null if approximate parity.
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


def _vendor_name_tokens(raw: Any) -> tuple[str, ...]:
    """Tokenize vendor names for tolerant but bounded matching."""
    return tuple(re.findall(r"[a-z0-9]+", str(raw or "").lower()))


def _vendor_name_key_tokens(raw: Any) -> set[str]:
    """Return the meaningful subset of vendor tokens for matching."""
    tokens = _vendor_name_tokens(raw)
    long_tokens = {tok for tok in tokens if len(tok) > 3}
    return long_tokens or set(tokens)


def _vendor_name_matches(target_name: Any, candidate_name: Any) -> bool:
    """Match vendor labels while tolerating common suffix variations."""
    target_tokens = _vendor_name_key_tokens(target_name)
    candidate_tokens = _vendor_name_key_tokens(candidate_name)
    if not target_tokens or not candidate_tokens:
        return False
    return target_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(target_tokens)


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
    market_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the user payload for a category council LLM call."""
    vendors = []
    for vname, ev in vendor_evidence.items():
        vendors.append({"name": vname, **_compact_evidence(ev)})
    flows = []
    for f in displacement_flows[:15]:
        flows.append(_compact_evidence(f))
    payload = {
        "category": category,
        "ecosystem": _compact_evidence(ecosystem),
        "vendors": vendors,
        "displacement_flows": flows,
    }
    if market_regime:
        payload["market_pulse"] = market_regime
    return payload


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


def _find_battle_contradictions(
    conclusion: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    """Check for contradictions in pairwise battle reasoning."""
    contradictions = []
    winner = conclusion.get("winner")
    loser = conclusion.get("loser")
    
    if winner and loser:
        v_a = payload.get("vendor_a", {})
        v_b = payload.get("vendor_b", {})

        def _get_data(target_name: Any) -> dict[str, Any]:
            name_a = v_a.get("name", "")
            name_b = v_b.get("name", "")
            if _vendor_name_matches(target_name, name_a):
                return v_a
            if _vendor_name_matches(target_name, name_b):
                return v_b
            return {}

        loser_data = _get_data(loser)
        winner_data = _get_data(winner)
        
        # 1. Check velocity contradiction
        l_vel = loser_data.get("velocity_churn_density", 0) or 0
        w_vel = winner_data.get("velocity_churn_density", 0) or 0
        
        if l_vel < -0.1 and w_vel > 0.1:
            contradictions.append(
                f"Winner is declared as {winner}, but {loser} has improving churn velocity ({l_vel}) "
                f"while {winner} is worsening ({w_vel})"
            )
            
        # 2. Check sentiment contradiction (recommend_ratio)
        # Higher is better
        l_rr = loser_data.get("recommend_ratio", 0) or 0
        w_rr = winner_data.get("recommend_ratio", 0) or 0
        
        # If loser has much better sentiment (>20 points higher)
        if l_rr > (w_rr + 20):
             contradictions.append(
                f"Winner is declared as {winner}, but {loser} has significantly better net sentiment "
                f"(NPS proxy: {l_rr} vs {w_rr})"
            )

    return contradictions


def _find_category_contradictions(
    conclusion: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    """Check for contradictions in category council reasoning."""
    contradictions = []
    regime = conclusion.get("market_regime")
    
    eco = payload.get("ecosystem", {})
    hhi = eco.get("hhi")
    vendor_count = eco.get("vendor_count", 0)
    
    # Check HHI/Count vs Regime
    if regime == "platform_consolidation":
        if hhi and hhi < 1000:
            contradictions.append(
                f"Market regime is 'platform_consolidation' but HHI is low ({hhi}), suggesting fragmentation"
            )
        if vendor_count > 20:
             contradictions.append(
                f"Market regime is 'platform_consolidation' but vendor count is high ({vendor_count})"
            )
            
    elif regime == "fragmentation":
        if hhi and hhi > 2500:
            contradictions.append(
                f"Market regime is 'fragmentation' but HHI is high ({hhi}), suggesting concentration"
            )
        if vendor_count < 5:
             contradictions.append(
                f"Market regime is 'fragmentation' but vendor count is low ({vendor_count})"
            )
        
    return contradictions


def _find_asymmetry_contradictions(
    conclusion: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    """Check for contradictions in resource asymmetry reasoning."""
    contradictions = []
    advantage = conclusion.get("resource_advantage") # text description
    if not advantage:
        return []
        
    v_a = payload.get("vendor_a", {})
    v_b = payload.get("vendor_b", {})
    name_a = v_a.get("name", "")
    name_b = v_b.get("name", "")
    
    # Helper to check who the LLM attributed advantage to
    def _attributed_to(name: Any) -> bool:
        return _vendor_name_matches(name, advantage)
        
    attributed_a = _attributed_to(name_a) or "Vendor A" in advantage
    attributed_b = _attributed_to(name_b) or "Vendor B" in advantage
    
    # 1. Review Count Contradiction
    reviews_a = v_a.get("total_reviews", 0) or 0
    reviews_b = v_b.get("total_reviews", 0) or 0
    
    if attributed_a and reviews_b > (reviews_a * 5):
        contradictions.append(
            f"Resource advantage attributed to {name_a}, but {name_b} has 5x more reviews "
            f"({reviews_b} vs {reviews_a})"
        )
    elif attributed_b and reviews_a > (reviews_b * 5):
        contradictions.append(
            f"Resource advantage attributed to {name_b}, but {name_a} has 5x more reviews "
            f"({reviews_a} vs {reviews_b})"
        )
        
    # 2. Funding/Insider Contradiction (if available)
    # Check for 'insider_signal_count' or 'budget_context' if present in compact payload
    # Note: _compact_evidence might strip some of this, but let's check what we have.
    # Assuming insider_signal_count is preserved if high signal.
    
    insider_a = v_a.get("insider_signal_count", 0) or 0
    insider_b = v_b.get("insider_signal_count", 0) or 0
    
    if attributed_a and insider_b > (insider_a + 5):
         contradictions.append(
            f"Resource advantage attributed to {name_a}, but {name_b} has significantly more insider signals "
            f"({insider_b} vs {insider_a})"
        )
    elif attributed_b and insider_a > (insider_b + 5):
         contradictions.append(
            f"Resource advantage attributed to {name_b}, but {name_a} has significantly more insider signals "
            f"({insider_a} vs {insider_b})"
        )

    return contradictions


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
        from .llm_utils import resolve_stratified_llm_light

        _rcfg = ReasoningConfig()
        # Pairwise battles are Tier 1 (heavy); councils + asymmetry are Tier 2 (light)
        if analysis_type == "pairwise_battle":
            llm = resolve_stratified_llm(_rcfg)
            llm_light = resolve_stratified_llm_light(_rcfg)
        else:
            llm = resolve_stratified_llm_light(_rcfg)
            llm_light = None  # same model for all passes
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

        # Select contradiction finder
        finder = None
        if _rcfg.multi_pass_enabled:
            if analysis_type == "pairwise_battle":
                finder = _find_battle_contradictions
            elif analysis_type == "category_council":
                finder = _find_category_contradictions
            elif analysis_type == "resource_asymmetry":
                finder = _find_asymmetry_contradictions

        try:
            if _rcfg.multi_pass_enabled and finder:
                mp_result = await multi_pass_reason(
                    llm=llm,
                    llm_light=llm_light,
                    system_prompt=system_prompt,
                    evidence_payload=payload,
                    json_schema=CROSS_VENDOR_JSON_SCHEMA,
                    max_tokens=_rcfg.max_tokens,
                    temperature=_rcfg.temperature,
                    enabled=True,
                    verify_enabled=_rcfg.multi_pass_verify_enabled,
                    verify_min_reviews=_rcfg.multi_pass_verify_min_reviews,
                    verify_min_snapshot_days=_rcfg.multi_pass_verify_min_snapshot_days,
                    verify_min_grounded_signals=_rcfg.multi_pass_verify_min_grounded_signals,
                    verify_confidence_cap=_rcfg.multi_pass_verify_confidence_cap,
                    light_pass_max_tokens=_rcfg.multi_pass_light_max_tokens,
                    challenge_confidence_floor=_rcfg.multi_pass_challenge_confidence_floor,
                    challenge_min_reviews=_rcfg.multi_pass_challenge_min_reviews,
                    challenge_mixed_polarity_min_share=_rcfg.multi_pass_challenge_mixed_polarity_min_share,
                    challenge_high_impact_churn_density=_rcfg.multi_pass_challenge_high_impact_churn_density,
                    challenge_high_impact_avg_urgency=_rcfg.multi_pass_challenge_high_impact_avg_urgency,
                    challenge_high_impact_displacement_mentions=_rcfg.multi_pass_challenge_high_impact_displacement_mentions,
                    ground_change_threshold=_rcfg.multi_pass_ground_change_threshold,
                    ground_always=_rcfg.multi_pass_ground_always,
                    span_prefix=f"reasoning.cross_vendor.{analysis_type}",
                    trace_metadata=trace_metadata,
                    contradiction_finder=finder,
                )
                conclusion = mp_result.final_conclusion
                tokens = mp_result.total_tokens
            else:
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
        market_regime: dict[str, Any] | None = None,
    ) -> CrossVendorResult:
        """Category council reasoning: market regime and structural dynamics."""
        eh = _combined_evidence_hash(
            vendor_evidence, ecosystem, {"flows": displacement_flows},
            market_regime or {},
        )
        cat_safe = category.lower().replace(" ", "_")
        pattern_sig = f"xv:category:{cat_safe}:{eh}"

        payload = _build_category_payload(
            category, vendor_evidence, ecosystem, displacement_flows, market_regime,
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
