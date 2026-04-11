"""
Win/Loss Predictor API.

Aggregates displacement data, churn signals, pain points, buyer profiles,
and product profiles into a win probability prediction with evidence.

Data gates enforce minimum thresholds per factor -- vendors with insufficient
data return is_gated=True with clear messaging instead of fake probabilities.
"""

import csv
import io
import asyncio
import json
import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..auth.dependencies import AuthUser, require_b2b_plan
from ..auth.rate_limit import limiter, _dynamic_limit
from ..pipelines.llm import call_llm_with_skill, clean_llm_output, parse_json_response
from ..config import settings
from ..services.b2b.cache_runner import (
    lookup_b2b_exact_stage_text,
    prepare_b2b_exact_skill_stage_request,
    store_b2b_exact_stage_text,
)
from ..services.b2b.llm_exact_cache import CacheUnavailable
from ..services.vendor_registry import resolve_vendor_name
from ..storage.database import get_db_pool

class _CacheHit(Exception):
    pass


logger = logging.getLogger("atlas.api.b2b_win_loss")

router = APIRouter(prefix="/b2b/predict", tags=["b2b-win-loss"])


# -- Scoring configuration ---------------------------------------------------

WEIGHTS = {
    "displacement_momentum": 0.25,
    "churn_severity": 0.20,
    "pain_concentration": 0.15,
    "dm_engagement": 0.15,
    "segment_match": 0.10,
    "historical_outcomes": 0.15,
}

DISPLACEMENT_BASELINE = 100
URGENCY_MAX = 10.0
CONFIDENCE_HIGH_THRESHOLD = 4
CONFIDENCE_MEDIUM_THRESHOLD = 2
SEGMENT_MATCH_HIT = 0.8
SEGMENT_MATCH_MISS = 0.3
SEGMENT_MATCH_NEUTRAL = 0.5

# Verdict probability thresholds
VERDICT_STRONG = 0.7
VERDICT_MODERATE = 0.5
VERDICT_CHALLENGING = 0.3

# -- Data gate thresholds ----------------------------------------------------
# Minimum data required for each factor to produce a reliable score.
# Factors below threshold get score=0 and are excluded from the weighted avg.

MIN_REVIEWS = 5
MIN_EDGES = 2
MIN_PAIN_POINTS = 2
MIN_BUYER_PROFILES = 2
MIN_PRODUCT_PROFILE = 1
MIN_OUTCOME_SEQUENCES = 3
MIN_FACTORS_FOR_PREDICTION = 2


# -- Request / response models -----------------------------------------------

class WinLossRequest(BaseModel):
    vendor_name: str = Field(..., min_length=1, description="Target vendor to sell against")
    company_size: Optional[str] = Field(
        None, description="startup, smb, mid_market, enterprise"
    )
    industry: Optional[str] = Field(None, description="Target industry/vertical")


class DataGate(BaseModel):
    factor: str
    required: int
    actual: int
    sufficient: bool


class Factor(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=1)
    weight: float
    evidence: str
    data_points: int = 0
    gated: bool = Field(False, description="True if insufficient data for this factor")


class SwitchingTrigger(BaseModel):
    trigger: str
    frequency: int = 0
    urgency: float = 0
    source: str = ""


class ProofQuote(BaseModel):
    quote: str
    source: str = ""
    role_type: str = ""
    urgency: float = 0


class Objection(BaseModel):
    objection: str
    frequency: int = 0
    counter: str = ""


class RecentPrediction(BaseModel):
    prediction_id: str
    vendor_name: str
    win_probability: float
    confidence: str
    is_gated: bool = False
    created_at: str


class WinLossResponse(BaseModel):
    vendor_name: str
    win_probability: float = Field(..., ge=0, le=1)
    confidence: str
    verdict: str
    is_gated: bool = Field(False, description="True if insufficient data for prediction")
    data_gates: list[DataGate] = []
    factors: list[Factor] = []
    switching_triggers: list[SwitchingTrigger] = []
    proof_quotes: list[ProofQuote] = []
    objections: list[Objection] = []
    displacement_targets: list[dict] = Field(default_factory=list)
    segment_match: Optional[dict] = None
    data_coverage: dict = Field(default_factory=dict)
    weights_source: str = "static"
    calibration_version: Optional[int] = None
    recommended_approach: Optional[str] = None
    lead_with: list[str] = []
    talking_points: list[str] = []
    timing_advice: Optional[str] = None
    risk_factors: list[str] = []
    prediction_id: Optional[str] = None


# -- Category label map -------------------------------------------------------

_AREA_LABELS = {
    "features": "Strong Feature Set",
    "ux": "Good User Experience",
    "integration": "Solid Integration Ecosystem",
    "overall_dissatisfaction": "General Customer Satisfaction",
    "support": "Responsive Support",
    "pricing": "Competitive Pricing",
    "reliability": "Platform Reliability",
    "performance": "Good Performance",
    "onboarding": "Smooth Onboarding",
    "security": "Strong Security",
    "contract_lock_in": "Contract Lock-In",
    "technical_debt": "Technical Debt",
    "other": "Other",
}


def _label(area: str) -> str:
    """Convert an internal area slug to a human-readable label."""
    return _AREA_LABELS.get(area, area.replace("_", " ").title())


# -- Calibration dimension -> predictor factor mapping ------------------------
# Each predictor factor maps to one or more calibration dimensions.
# When calibration data exists, the average lift across mapped dimensions
# adjusts the static weight with a conservative blending factor.

_FACTOR_DIMENSION_MAP = {
    "displacement_momentum": ["context_keyword"],
    "churn_severity": ["urgency_bucket"],
    "pain_concentration": [],           # no direct calibration dimension
    "dm_engagement": ["role_type", "buying_stage"],
    "segment_match": ["seat_bucket"],
    "historical_outcomes": [],          # self-calibrating from actual outcomes
}

# How aggressively to blend calibrated lift into static weights (0-1).
# 0.3 = conservative: 30% calibration influence, 70% static.
CALIBRATION_BLEND_ALPHA = 0.3

# LLM strategy synthesis params
STRATEGY_MAX_TOKENS = 1024
STRATEGY_TEMPERATURE = 0.4

# Output caps (max items returned per section)
MAX_DISPLACEMENT_TARGETS = 5
MAX_SWITCHING_TRIGGERS = 8
MAX_PROOF_QUOTES = 6
MAX_OBJECTIONS = 5
MAX_PAIN_TRIGGERS = 8
MAX_WEAKNESS_TRIGGERS = 5
MAX_STRENGTH_OBJECTIONS = 5

# Maps frontend company_size slugs to the DB size-range keys in
# b2b_product_profiles.typical_company_size (e.g. {"1-50": 0.04, ...}).
_SIZE_SLUG_TO_RANGES = {
    "startup": ["1-50"],
    "smb": ["51-200"],
    "mid_market": ["201-1000"],
    "enterprise": ["1000+"],
}


async def _load_calibrated_weights(pool) -> tuple[dict[str, float], str, Optional[int]]:
    """Load calibrated weights from score_calibration_weights table.

    Returns (weights_dict, source_label, model_version).
    Falls back to static WEIGHTS if no calibration data.
    """
    latest_version = await pool.fetchval(
        "SELECT MAX(model_version) FROM score_calibration_weights"
    )
    if latest_version is None:
        return dict(WEIGHTS), "static", None

    rows = await pool.fetch(
        """
        SELECT dimension, dimension_value, lift, total_sequences
        FROM score_calibration_weights
        WHERE model_version = $1
        """,
        latest_version,
    )
    if not rows:
        return dict(WEIGHTS), "static", None

    # Group lifts by dimension
    dim_lifts: dict[str, list[float]] = {}
    total_sequences = 0
    for r in rows:
        dim = r["dimension"]
        lift = float(r["lift"] or 1.0)
        seqs = int(r["total_sequences"] or 0)
        dim_lifts.setdefault(dim, []).append(lift)
        total_sequences += seqs

    # Compute adjusted weights per factor
    adjusted = dict(WEIGHTS)
    for factor_key, dimensions in _FACTOR_DIMENSION_MAP.items():
        if not dimensions:
            continue
        # Average lift across mapped dimensions
        lifts = []
        for dim in dimensions:
            if dim in dim_lifts:
                lifts.extend(dim_lifts[dim])
        if not lifts:
            continue
        avg_lift = sum(lifts) / len(lifts)
        # Blend: static * (1 + (avg_lift - 1.0) * alpha)
        # lift=1.0 means neutral, >1 means outperforms, <1 underperforms
        delta = (avg_lift - 1.0) * CALIBRATION_BLEND_ALPHA
        adjusted[factor_key] = max(0.0, WEIGHTS[factor_key] * (1.0 + delta))

    # Re-normalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted, "calibrated", int(latest_version)


# -- Helpers ------------------------------------------------------------------

def _safe_json(value) -> list | dict | None:
    """Parse a JSONB field that may be a dict, list, or JSON string."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


async def _persist_prediction(pool, account_id: str, req: WinLossRequest, resp: WinLossResponse) -> Optional[str]:
    """Save prediction to b2b_win_loss_predictions. Returns prediction UUID or None on failure."""
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO b2b_win_loss_predictions (
                account_id, vendor_name, company_size, industry,
                win_probability, confidence, verdict, is_gated,
                recommended_approach, lead_with, talking_points, timing_advice, risk_factors,
                factors, data_gates, switching_triggers, proof_quotes, objections,
                displacement_targets, segment_match, data_coverage,
                weights_source, calibration_version
            ) VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18,
                $19, $20, $21,
                $22, $23
            )
            RETURNING id
            """,
            account_id,
            resp.vendor_name,
            req.company_size,
            req.industry,
            float(resp.win_probability),
            resp.confidence,
            resp.verdict,
            resp.is_gated,
            resp.recommended_approach,
            json.dumps(resp.lead_with),
            json.dumps(resp.talking_points),
            resp.timing_advice,
            json.dumps(resp.risk_factors),
            json.dumps([f.model_dump() for f in resp.factors]),
            json.dumps([g.model_dump() for g in resp.data_gates]),
            json.dumps([t.model_dump() for t in resp.switching_triggers]),
            json.dumps([q.model_dump() for q in resp.proof_quotes]),
            json.dumps([o.model_dump() for o in resp.objections]),
            json.dumps(resp.displacement_targets),
            json.dumps(resp.segment_match) if resp.segment_match else None,
            json.dumps(resp.data_coverage),
            resp.weights_source,
            resp.calibration_version,
        )
        return str(row["id"]) if row else None
    except Exception as e:
        logger.warning("Failed to persist prediction: %s", e)
        return None


# -- Core prediction logic (reusable by single + compare endpoints) -----------

async def _compute_prediction(
    pool,
    vendor: str,
    company_size: Optional[str] = None,
    industry: Optional[str] = None,
) -> WinLossResponse:

    """Run the full prediction pipeline for a single vendor. Returns WinLossResponse."""
    from ..autonomous.tasks._b2b_shared import read_vendor_signal_detail_exact

    signal = await read_vendor_signal_detail_exact(
        pool,
        vendor_name=vendor,
    )
    review_count = int((signal or {}).get("total_reviews") or 0)

    # -- Pre-check: count data availability per factor ------------------------
    raw_counts = await pool.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM b2b_displacement_edges
             WHERE from_vendor = $1) AS edges,
            (SELECT COUNT(*) FROM b2b_vendor_pain_points
             WHERE vendor_name = $1) AS pain_points,
            (SELECT COUNT(*) FROM b2b_vendor_buyer_profiles
             WHERE vendor_name = $1) AS buyer_profiles,
            (SELECT COUNT(*) FROM b2b_product_profiles
             WHERE vendor_name = $1) AS product_profiles,
            COALESCE((SELECT COUNT(*) FROM campaign_sequences cs
             JOIN b2b_campaigns c ON cs.last_campaign_id = c.id
             WHERE c.vendor_name = $1 AND cs.outcome IS NOT NULL
               AND cs.outcome != 'pending'), 0) AS outcome_sequences
        """,
        vendor,
    )
    counts = dict(raw_counts or {})
    counts["reviews"] = review_count

    gates = [
        DataGate(factor="Churn Reviews", required=MIN_REVIEWS,
                 actual=int(counts["reviews"]),
                 sufficient=int(counts["reviews"]) >= MIN_REVIEWS),
        DataGate(factor="Displacement Edges", required=MIN_EDGES,
                 actual=int(counts["edges"]),
                 sufficient=int(counts["edges"]) >= MIN_EDGES),
        DataGate(factor="Pain Points", required=MIN_PAIN_POINTS,
                 actual=int(counts["pain_points"]),
                 sufficient=int(counts["pain_points"]) >= MIN_PAIN_POINTS),
        DataGate(factor="Buyer Profiles", required=MIN_BUYER_PROFILES,
                 actual=int(counts["buyer_profiles"]),
                 sufficient=int(counts["buyer_profiles"]) >= MIN_BUYER_PROFILES),
        DataGate(factor="Product Profile", required=MIN_PRODUCT_PROFILE,
                 actual=int(counts["product_profiles"]),
                 sufficient=int(counts["product_profiles"]) >= MIN_PRODUCT_PROFILE),
        DataGate(factor="Campaign Outcomes", required=MIN_OUTCOME_SEQUENCES,
                 actual=int(counts["outcome_sequences"]),
                 sufficient=int(counts["outcome_sequences"]) >= MIN_OUTCOME_SEQUENCES),
    ]

    passed_gates = sum(1 for g in gates if g.sufficient)
    is_gated = passed_gates < MIN_FACTORS_FOR_PREDICTION

    if is_gated:
        gated_resp = WinLossResponse(
            vendor_name=vendor,
            win_probability=0,
            confidence="insufficient",
            verdict=(
                f"Insufficient data to predict win probability against {vendor}. "
                f"Only {passed_gates} of {MIN_FACTORS_FOR_PREDICTION} required "
                f"data sources have enough coverage."
            ),
            is_gated=True,
            data_gates=gates,
            data_coverage={
                "reviews": int(counts["reviews"]),
                "edges": int(counts["edges"]),
                "pain_points": int(counts["pain_points"]),
                "buyer_profiles": int(counts["buyer_profiles"]),
                "outcomes": int(counts["outcome_sequences"]),
                "product_profiles": int(counts["product_profiles"]),
            },
        )
        return gated_resp

    # -- Full prediction -------------------------------------------------------

    factors: list[Factor] = []
    switching_triggers: list[SwitchingTrigger] = []
    proof_quotes: list[ProofQuote] = []
    objections: list[Objection] = []
    displacement_targets: list[dict] = []
    segment_data: Optional[dict] = None
    data_coverage: dict = {
        "reviews": int(counts["reviews"]),
        "edges": int(counts["edges"]),
        "pain_points": int(counts["pain_points"]),
        "buyer_profiles": int(counts["buyer_profiles"]),
        "outcomes": int(counts["outcome_sequences"]),
        "product_profiles": int(counts["product_profiles"]),
    }

    # -- Load weights (calibrated if available, static fallback) ----------------
    weights, weights_source, cal_version = await _load_calibrated_weights(pool)

    # -- 1. Displacement momentum ---------------------------------------------
    edge_gate = gates[1]
    if edge_gate.sufficient:
        edges = await pool.fetch(
            """
            SELECT to_vendor, mention_count, primary_driver, signal_strength,
                   confidence_score, key_quote
            FROM b2b_displacement_edges
            WHERE from_vendor = $1
            ORDER BY mention_count DESC
            LIMIT 20
            """,
            vendor,
        )
        total_outflow = sum(r["mention_count"] for r in edges)
        displacement_score = min(1.0, total_outflow / DISPLACEMENT_BASELINE)
        displacement_targets.extend([
            {
                "vendor": r["to_vendor"],
                "mentions": r["mention_count"],
                "driver": r["primary_driver"],
                "strength": r["signal_strength"],
            }
            for r in edges[:MAX_DISPLACEMENT_TARGETS]
        ])
        drivers = [r["primary_driver"] for r in edges if r["primary_driver"]]
        evidence = (
            f"{total_outflow} displacement mentions across {len(edges)} alternatives. "
            f"Top driver: {drivers[0] if drivers else 'mixed'}."
        )
        factors.append(Factor(
            name="Displacement Momentum", score=displacement_score,
            weight=weights["displacement_momentum"], evidence=evidence,
            data_points=total_outflow,
        ))
    else:
        factors.append(Factor(
            name="Displacement Momentum", score=0, weight=0,
            evidence=f"Insufficient data ({edge_gate.actual}/{edge_gate.required} edges).",
            data_points=0, gated=True,
        ))

    # -- 2. Churn signal severity ----------------------------------------------
    review_gate = gates[0]
    if review_gate.sufficient:
        if signal:
            urgency = float(signal["avg_urgency_score"] or 0) / URGENCY_MAX
            total_revs = max(int(signal["total_reviews"] or 1), 1)
            churn_rate = float(signal["churn_intent_count"] or 0) / total_revs
            churn_score = min(1.0, (urgency * 0.6) + (min(churn_rate * 5, 1.0) * 0.4))
            avg_urg = float(signal["avg_urgency_score"] or 0)
            evidence = (
                f"{signal['churn_intent_count'] or 0}/{signal['total_reviews'] or 0} reviews show churn intent. "
                f"Avg urgency: {avg_urg:.1f}/10. "
                f"Archetype: {signal['archetype'] or 'unclassified'}."
            )
            factors.append(Factor(
                name="Churn Severity", score=churn_score,
                weight=weights["churn_severity"], evidence=evidence,
                data_points=int(signal["total_reviews"] or 0),
            ))
        else:
            factors.append(Factor(
                name="Churn Severity", score=0, weight=0,
                evidence="No churn signal computed for this vendor.",
                data_points=0, gated=True,
            ))
    else:
        factors.append(Factor(
            name="Churn Severity", score=0, weight=0,
            evidence=f"Insufficient data ({review_gate.actual}/{review_gate.required} reviews).",
            data_points=0, gated=True,
        ))

    # -- Proof quotes from b2b_vendor_witnesses --------------------------------
    witnesses = await pool.fetch(
        """
        SELECT excerpt_text, source, reviewer_title, pain_category,
               salience_score, witness_type
        FROM b2b_vendor_witnesses
        WHERE vendor_name = $1
          AND excerpt_text IS NOT NULL
          AND excerpt_text != ''
        ORDER BY salience_score DESC NULLS LAST
        LIMIT 6
        """,
        vendor,
    )
    for w in witnesses:
        proof_quotes.append(ProofQuote(
            quote=w["excerpt_text"],
            source=w["source"] or "",
            role_type=w["reviewer_title"] or w["witness_type"] or "",
            urgency=float(w["salience_score"] or 0),
        ))

    # -- 3. Pain concentration -------------------------------------------------
    pain_gate = gates[2]
    if pain_gate.sufficient:
        pains_rows = await pool.fetch(
            """
            SELECT pain_category, mention_count, primary_count, avg_urgency
            FROM b2b_vendor_pain_points
            WHERE vendor_name = $1
            ORDER BY mention_count DESC
            """,
            vendor,
        )
        total_mentions = sum(r["mention_count"] for r in pains_rows)
        top_pain = pains_rows[0]
        top_pct = (top_pain["mention_count"] / total_mentions) if total_mentions > 0 else 0
        pain_score = min(1.0, top_pct + (float(top_pain["avg_urgency"] or 0) / URGENCY_MAX) * 0.3)
        categories = [r["pain_category"] for r in pains_rows[:3]]
        evidence = (
            f"Top pain: {top_pain['pain_category']} ({top_pain['mention_count']} mentions, "
            f"{top_pct:.0%} of all complaints). "
            f"Categories: {', '.join(categories)}."
        )
        for pt in pains_rows[:MAX_PAIN_TRIGGERS]:
            switching_triggers.append(SwitchingTrigger(
                trigger=pt["pain_category"],
                frequency=int(pt["mention_count"] or 0),
                urgency=float(pt["avg_urgency"] or 0),
                source="pain_points",
            ))
        factors.append(Factor(
            name="Pain Concentration", score=pain_score,
            weight=weights["pain_concentration"], evidence=evidence,
            data_points=len(pains_rows),
        ))
    else:
        factors.append(Factor(
            name="Pain Concentration", score=0, weight=0,
            evidence=f"Insufficient data ({pain_gate.actual}/{pain_gate.required} pain categories).",
            data_points=0, gated=True,
        ))

    # -- 4. Decision-maker engagement ------------------------------------------
    dm_gate = gates[3]
    if dm_gate.sufficient:
        dm_rows = await pool.fetch(
            """
            SELECT role_type, buying_stage, review_count, dm_count, avg_urgency
            FROM b2b_vendor_buyer_profiles
            WHERE vendor_name = $1
            ORDER BY dm_count DESC
            """,
            vendor,
        )
        total_reviews = sum(r["review_count"] for r in dm_rows)
        total_dms = sum(r["dm_count"] for r in dm_rows)
        dm_ratio = total_dms / max(total_reviews, 1)
        active_buyers = [r for r in dm_rows if r["buying_stage"] in ("active_purchase", "evaluation")]
        active_urgency = (
            sum(float(r["avg_urgency"] or 0) for r in active_buyers) / max(len(active_buyers), 1)
        )
        dm_score = min(1.0, (dm_ratio * 0.5) + (active_urgency / URGENCY_MAX) * 0.5)
        evidence = (
            f"{total_dms} decision-makers across {total_reviews} reviews. "
            f"{len(active_buyers)} in active purchase/evaluation. "
            f"Avg urgency: {active_urgency:.1f}/10."
        )
        factors.append(Factor(
            name="Decision-Maker Engagement", score=dm_score,
            weight=weights["dm_engagement"], evidence=evidence,
            data_points=total_dms,
        ))
    else:
        factors.append(Factor(
            name="Decision-Maker Engagement", score=0, weight=0,
            evidence=f"Insufficient data ({dm_gate.actual}/{dm_gate.required} buyer profiles).",
            data_points=0, gated=True,
        ))

    # -- 5. Segment match ------------------------------------------------------
    profile_gate = gates[4]
    if profile_gate.sufficient:
        profile = await pool.fetchrow(
            """
            SELECT strengths, weaknesses, typical_company_size, typical_industries,
                   commonly_switched_from, commonly_compared_to, recommend_rate,
                   profile_summary, total_reviews_analyzed
            FROM b2b_product_profiles
            WHERE vendor_name = $1
            ORDER BY last_computed_at DESC NULLS LAST
            LIMIT 1
            """,
            vendor,
        )
        if profile:
            sizes = _safe_json(profile["typical_company_size"]) or []
            industries = _safe_json(profile["typical_industries"]) or []

            size_match = SEGMENT_MATCH_NEUTRAL
            industry_match = SEGMENT_MATCH_NEUTRAL

            if company_size and sizes:
                cs_lower = company_size.lower()
                matched = False
                if isinstance(sizes, dict):
                    # DB format: {"1-50": 0.04, "51-200": 0.54, ...}
                    # Map the slug to range keys and check if proportion > 0
                    target_ranges = _SIZE_SLUG_TO_RANGES.get(cs_lower, [])
                    for rng in target_ranges:
                        if float(sizes.get(rng, 0) or 0) > 0:
                            matched = True
                            break
                    if not matched and not target_ranges:
                        # Unknown slug -- try direct key match
                        matched = cs_lower in {k.lower() for k in sizes}
                elif isinstance(sizes, list):
                    size_lower = [str(s).lower() for s in sizes]
                    matched = any(cs_lower == s for s in size_lower)
                if matched:
                    size_match = SEGMENT_MATCH_HIT
                elif sizes:
                    size_match = SEGMENT_MATCH_MISS

            if industry and industries:
                ind_query = industry.lower()
                matched_ind = False
                if isinstance(industries, dict):
                    ind_keys = [k.lower() for k in industries]
                    matched_ind = any(ind_query == k or ind_query in k for k in ind_keys)
                elif isinstance(industries, list):
                    ind_lower = [str(s).lower() for s in industries]
                    matched_ind = any(ind_query == s or ind_query in s for s in ind_lower)
                if matched_ind:
                    industry_match = SEGMENT_MATCH_HIT
                elif industries:
                    industry_match = SEGMENT_MATCH_MISS

            segment_score = (size_match + industry_match) / 2.0
            segment_data = {
                "typical_sizes": sizes[:5] if isinstance(sizes, list) else sizes,
                "typical_industries": industries[:5] if isinstance(industries, list) else industries,
                "size_match": size_match,
                "industry_match": industry_match,
            }
            evidence = (
                f"Typical sizes: {sizes[:3] if isinstance(sizes, list) else sizes}. "
                f"Industries: {industries[:3] if isinstance(industries, list) else industries}."
            )

            # Objections from strengths -- areas where the vendor is strong
            # become objections a sales rep will face from loyal customers.
            # Enrich with avg rating score and recommend_rate for context.
            recommend_rate = float(profile["recommend_rate"] or 0)
            strengths = _safe_json(profile["strengths"]) or []
            if isinstance(strengths, list):
                for s in strengths[:MAX_STRENGTH_OBJECTIONS]:
                    if isinstance(s, dict):
                        area = s.get("area", "")
                        score = float(s.get("score", 0) or 0)
                        evidence_count = int(s.get("evidence_count", 0) or 0)
                        # Build a counter explaining why this is hard to beat
                        counter_parts = []
                        if score >= 4.0:
                            counter_parts.append(f"Avg rating {score:.1f}/5 across {evidence_count} reviews")
                        elif score >= 3.0:
                            counter_parts.append(f"Mixed reviews ({score:.1f}/5) but {evidence_count} cite this")
                        if recommend_rate >= 0.7:
                            counter_parts.append(f"{recommend_rate:.0%} recommend rate")
                        objections.append(Objection(
                            objection=_label(area),
                            frequency=evidence_count,
                            counter=". ".join(counter_parts) if counter_parts else "",
                        ))
                    elif isinstance(s, str):
                        objections.append(Objection(objection=_label(s)))

            # Weaknesses as switching triggers
            weaknesses = _safe_json(profile["weaknesses"]) or []
            if isinstance(weaknesses, list):
                existing = {t.trigger.lower() for t in switching_triggers}
                for w in weaknesses[:MAX_WEAKNESS_TRIGGERS]:
                    if isinstance(w, dict):
                        trigger_name = _label(w.get("area", str(w)))
                        if trigger_name.lower() not in existing:
                            switching_triggers.append(SwitchingTrigger(
                                trigger=trigger_name,
                                frequency=int(w.get("evidence_count", 0) or 0),
                                source="product_profile",
                            ))

            factors.append(Factor(
                name="Segment Match", score=segment_score,
                weight=weights["segment_match"], evidence=evidence,
                data_points=int(profile["total_reviews_analyzed"] or 0),
            ))
        else:
            factors.append(Factor(
                name="Segment Match", score=0, weight=0,
                evidence="Product profile not computed yet.",
                data_points=0, gated=True,
            ))
    else:
        factors.append(Factor(
            name="Segment Match", score=0, weight=0,
            evidence="No product profile available.",
            data_points=0, gated=True,
        ))

    # -- 6. Historical outcomes ------------------------------------------------
    outcome_gate = gates[5]
    if outcome_gate.sufficient:
        outcomes = await pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE cs.outcome IN ('meeting_booked','deal_opened','deal_won')) AS positive,
                COUNT(*) FILTER (WHERE cs.outcome = 'deal_won') AS won,
                COUNT(*) FILTER (WHERE cs.outcome = 'deal_lost') AS lost
            FROM campaign_sequences cs
            JOIN b2b_campaigns c ON cs.last_campaign_id = c.id
            WHERE c.vendor_name = $1
              AND cs.outcome IS NOT NULL
              AND cs.outcome != 'pending'
            """,
            vendor,
        )
        if outcomes and outcomes["total"] > 0:
            positive_rate = outcomes["positive"] / outcomes["total"]
            outcome_score = min(1.0, positive_rate * 1.2)
            evidence = (
                f"{outcomes['positive']}/{outcomes['total']} campaigns had positive outcomes. "
                f"Wins: {outcomes['won']}, losses: {outcomes['lost']}."
            )
            factors.append(Factor(
                name="Historical Win Rate", score=outcome_score,
                weight=weights["historical_outcomes"], evidence=evidence,
                data_points=int(outcomes["total"]),
            ))
        else:
            factors.append(Factor(
                name="Historical Win Rate", score=0, weight=0,
                evidence="No outcome data recorded.",
                data_points=0, gated=True,
            ))
    else:
        factors.append(Factor(
            name="Historical Win Rate", score=0, weight=0,
            evidence=f"Insufficient data ({outcome_gate.actual}/{outcome_gate.required} sequences).",
            data_points=0, gated=True,
        ))

    # -- Compute final probability (gated factors excluded) --------------------
    active_factors = [f for f in factors if not f.gated]

    # Safety: if inner queries returned None despite gate counts passing,
    # all factors may be gated even though is_gated was False above.
    if len(active_factors) < MIN_FACTORS_FOR_PREDICTION:
        return WinLossResponse(
            vendor_name=vendor,
            win_probability=0,
            confidence="insufficient",
            verdict=(
                f"Insufficient data to predict win probability against {vendor}. "
                f"Factor queries returned no usable data."
            ),
            is_gated=True,
            data_gates=gates,
            factors=factors,
            data_coverage=data_coverage,
        )

    total_weight = sum(f.weight for f in active_factors)
    if total_weight > 0:
        weighted_sum = sum(f.score * (f.weight / total_weight) for f in active_factors)
        win_probability = round(weighted_sum, 3)
    else:
        win_probability = 0.0

    # Confidence
    if len(active_factors) >= CONFIDENCE_HIGH_THRESHOLD:
        confidence = "high"
    elif len(active_factors) >= CONFIDENCE_MEDIUM_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    # Verdict
    gated_count = sum(1 for f in factors if f.gated)
    caveat = f" ({gated_count} factor(s) excluded due to insufficient data.)" if gated_count > 0 else ""

    if win_probability >= VERDICT_STRONG:
        verdict = f"Strong opportunity. {vendor} users are actively looking for alternatives.{caveat}"
    elif win_probability >= VERDICT_MODERATE:
        verdict = f"Moderate opportunity. {vendor} has exploitable weaknesses but loyal segments.{caveat}"
    elif win_probability >= VERDICT_CHALLENGING:
        verdict = f"Challenging deal. {vendor} has strong retention in most segments.{caveat}"
    else:
        verdict = f"Uphill battle. {vendor} has strong product-market fit and low churn signals.{caveat}"

    switching_triggers.sort(key=lambda t: (t.urgency, t.frequency), reverse=True)
    proof_quotes.sort(key=lambda q: q.urgency, reverse=True)

    # -- LLM strategy synthesis ------------------------------------------------
    recommended_approach: Optional[str] = None
    lead_with: list[str] = []
    talking_points: list[str] = []
    timing_advice: Optional[str] = None
    risk_factors: list[str] = []

    try:
        strategy_payload = {
            "vendor_name": vendor,
            "win_probability": win_probability,
            "confidence": confidence,
            "factors": [f.model_dump() for f in factors],
            "switching_triggers": [t.model_dump() for t in switching_triggers[:MAX_SWITCHING_TRIGGERS]],
            "proof_quotes": [q.model_dump() for q in proof_quotes[:MAX_PROOF_QUOTES]],
            "objections": [o.model_dump() for o in objections[:MAX_OBJECTIONS]],
            "displacement_targets": displacement_targets,
            "segment_match": segment_data,
        }

        win_loss_model = str(settings.b2b_churn.win_loss_model).strip()
        payload_text = json.dumps(strategy_payload, sort_keys=True, ensure_ascii=True)

        # Check exact cache first
        cache_request = None
        try:
            cache_request, _ = prepare_b2b_exact_skill_stage_request(
                "win_loss.strategy",
                skill_name="digest/win_loss_strategy",
                payload=payload_text,
                provider="openrouter",
                model=win_loss_model,
                max_tokens=STRATEGY_MAX_TOKENS,
                temperature=STRATEGY_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            cached = await lookup_b2b_exact_stage_text(cache_request)
            if cached is not None:
                parsed = parse_json_response(cached["response_text"], recover_truncated=True)
                if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                    recommended_approach = parsed.get("recommended_approach")
                    lead_with = parsed.get("lead_with") or []
                    talking_points = parsed.get("talking_points") or []
                    timing_advice = parsed.get("timing_advice")
                    risk_factors = parsed.get("risk_factors") or []
                    raise _CacheHit()
        except CacheUnavailable:
            cache_request = None
        except _CacheHit:
            pass
        else:
            raw = await asyncio.to_thread(
                call_llm_with_skill,
                "digest/win_loss_strategy",
                strategy_payload,
                workload="openrouter",
                try_openrouter=True,
                openrouter_model=win_loss_model,
                max_tokens=STRATEGY_MAX_TOKENS,
                temperature=STRATEGY_TEMPERATURE,
                response_format={"type": "json_object"},
            )

            if raw:
                parsed = parse_json_response(raw, recover_truncated=True)
                if isinstance(parsed, dict) and not parsed.get("_parse_fallback"):
                    recommended_approach = parsed.get("recommended_approach")
                    lead_with = parsed.get("lead_with") or []
                    talking_points = parsed.get("talking_points") or []
                    timing_advice = parsed.get("timing_advice")
                    risk_factors = parsed.get("risk_factors") or []

                    if cache_request is not None:
                        await store_b2b_exact_stage_text(
                            cache_request,
                            response_text=clean_llm_output(raw),
                            metadata={"model": win_loss_model},
                        )
    except _CacheHit:
        pass
    except Exception as e:
        logger.warning("LLM strategy synthesis failed: %s", e)

    response = WinLossResponse(
        vendor_name=vendor,
        win_probability=win_probability,
        confidence=confidence,
        verdict=verdict,
        is_gated=False,
        data_gates=gates,
        factors=factors,
        switching_triggers=switching_triggers[:MAX_SWITCHING_TRIGGERS],
        proof_quotes=proof_quotes[:MAX_PROOF_QUOTES],
        objections=objections[:MAX_OBJECTIONS],
        displacement_targets=displacement_targets,
        segment_match=segment_data,
        data_coverage=data_coverage,
        weights_source=weights_source,
        calibration_version=cal_version,
        recommended_approach=recommended_approach,
        lead_with=lead_with,
        talking_points=talking_points,
        timing_advice=timing_advice,
        risk_factors=risk_factors,
    )
    return response


# -- Main endpoint ------------------------------------------------------------

@router.post("/win-loss", response_model=WinLossResponse)
@limiter.limit(_dynamic_limit)
async def predict_win_loss(
    request: Request,
    req: WinLossRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Predict win probability when selling against a given vendor."""
    pool = get_db_pool()
    vendor = req.vendor_name.strip()

    resolved = await resolve_vendor_name(vendor)
    if resolved:
        vendor = resolved

    response = await _compute_prediction(pool, vendor, req.company_size, req.industry)
    pid = await _persist_prediction(pool, user.account_id, req, response)
    response.prediction_id = pid
    return response


# -- Comparison endpoint ------------------------------------------------------

class WinLossCompareRequest(BaseModel):
    vendor_a: str = Field(..., min_length=1, description="First vendor to sell against")
    vendor_b: str = Field(..., min_length=1, description="Second vendor to sell against")
    company_size: Optional[str] = None
    industry: Optional[str] = None


class FactorComparison(BaseModel):
    name: str
    vendor_a_score: float
    vendor_b_score: float
    advantage: str  # "a", "b", or "tie"


class WinLossCompareResponse(BaseModel):
    vendor_a: WinLossResponse
    vendor_b: WinLossResponse
    easier_target: str
    probability_delta: float
    factor_comparison: list[FactorComparison] = []


@router.post("/win-loss/compare", response_model=WinLossCompareResponse)
@limiter.limit(_dynamic_limit)
async def compare_win_loss(
    request: Request,
    req: WinLossCompareRequest,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Compare win probability between two vendors side by side."""
    pool = get_db_pool()

    # Resolve vendor names
    vendor_a = req.vendor_a.strip()
    vendor_b = req.vendor_b.strip()
    resolved_a = await resolve_vendor_name(vendor_a)
    resolved_b = await resolve_vendor_name(vendor_b)
    if resolved_a:
        vendor_a = resolved_a
    if resolved_b:
        vendor_b = resolved_b

    if vendor_a.lower() == vendor_b.lower():
        raise HTTPException(status_code=400, detail="Cannot compare a vendor against itself")

    # Run both predictions
    resp_a, resp_b = await asyncio.gather(
        _compute_prediction(pool, vendor_a, req.company_size, req.industry),
        _compute_prediction(pool, vendor_b, req.company_size, req.industry),
    )

    # Persist both
    dummy_req_a = WinLossRequest(vendor_name=vendor_a, company_size=req.company_size, industry=req.industry)
    dummy_req_b = WinLossRequest(vendor_name=vendor_b, company_size=req.company_size, industry=req.industry)
    pid_a = await _persist_prediction(pool, user.account_id, dummy_req_a, resp_a)
    pid_b = await _persist_prediction(pool, user.account_id, dummy_req_b, resp_b)
    resp_a.prediction_id = pid_a
    resp_b.prediction_id = pid_b

    # Build factor comparison
    factor_map_b = {f.name: f for f in resp_b.factors}
    factor_comparison = []
    for fa in resp_a.factors:
        fb = factor_map_b.get(fa.name)
        fb_score = fb.score if fb and not fb.gated else 0.0
        fa_score = fa.score if not fa.gated else 0.0
        if fa_score > fb_score:
            adv = "a"
        elif fb_score > fa_score:
            adv = "b"
        else:
            adv = "tie"
        factor_comparison.append(FactorComparison(
            name=fa.name,
            vendor_a_score=fa_score,
            vendor_b_score=fb_score,
            advantage=adv,
        ))

    delta = round(resp_a.win_probability - resp_b.win_probability, 3)
    if resp_a.win_probability > resp_b.win_probability:
        easier = vendor_a
    elif resp_b.win_probability > resp_a.win_probability:
        easier = vendor_b
    else:
        easier = "tie"

    return WinLossCompareResponse(
        vendor_a=resp_a,
        vendor_b=resp_b,
        easier_target=easier,
        probability_delta=abs(delta),
        factor_comparison=factor_comparison,
    )


# -- Recent predictions -------------------------------------------------------

MAX_RECENT_PREDICTIONS = 50


@router.get("/win-loss/recent")
async def list_recent_predictions(
    limit: int = Query(10, ge=1, le=MAX_RECENT_PREDICTIONS),
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """List recent win/loss predictions for the current tenant."""
    pool = get_db_pool()
    rows = await pool.fetch(
        """
        SELECT id, vendor_name, win_probability, confidence, is_gated, created_at
        FROM b2b_win_loss_predictions
        WHERE account_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        user.account_id, limit,
    )
    return {
        "predictions": [
            RecentPrediction(
                prediction_id=str(r["id"]),
                vendor_name=r["vendor_name"],
                win_probability=float(r["win_probability"]),
                confidence=r["confidence"],
                is_gated=r["is_gated"],
                created_at=r["created_at"].isoformat(),
            )
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/win-loss/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Fetch a stored win/loss prediction by ID."""
    pool = get_db_pool()
    try:
        pid = _uuid.UUID(prediction_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    row = await pool.fetchrow(
        """
        SELECT * FROM b2b_win_loss_predictions
        WHERE id = $1 AND account_id = $2
        """,
        pid, user.account_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        data_gates = [DataGate(**g) for g in (_safe_json(row["data_gates"]) or [])]
    except Exception:
        data_gates = []
    try:
        factors = [Factor(**f) for f in (_safe_json(row["factors"]) or [])]
    except Exception:
        factors = []
    try:
        switching_triggers = [SwitchingTrigger(**t) for t in (_safe_json(row["switching_triggers"]) or [])]
    except Exception:
        switching_triggers = []
    try:
        proof_quotes = [ProofQuote(**q) for q in (_safe_json(row["proof_quotes"]) or [])]
    except Exception:
        proof_quotes = []
    try:
        objections = [Objection(**o) for o in (_safe_json(row["objections"]) or [])]
    except Exception:
        objections = []

    return WinLossResponse(
        vendor_name=row["vendor_name"],
        win_probability=float(row["win_probability"]),
        confidence=row["confidence"],
        verdict=row["verdict"] or "",
        is_gated=row["is_gated"],
        data_gates=data_gates,
        factors=factors,
        switching_triggers=switching_triggers,
        proof_quotes=proof_quotes,
        objections=objections,
        displacement_targets=_safe_json(row["displacement_targets"]) or [],
        segment_match=_safe_json(row["segment_match"]),
        data_coverage=_safe_json(row["data_coverage"]) or {},
        weights_source=row["weights_source"] or "static",
        calibration_version=row["calibration_version"],
        recommended_approach=row["recommended_approach"],
        lead_with=_safe_json(row["lead_with"]) or [],
        talking_points=_safe_json(row["talking_points"]) or [],
        timing_advice=row["timing_advice"],
        risk_factors=_safe_json(row["risk_factors"]) or [],
        prediction_id=str(row["id"]),
    )


# -- CSV export ---------------------------------------------------------------


@router.get("/win-loss/{prediction_id}/csv")
async def export_prediction_csv(
    prediction_id: str,
    user: AuthUser = Depends(require_b2b_plan("b2b_trial")),
):
    """Export a stored prediction as CSV."""
    pool = get_db_pool()
    try:
        pid = _uuid.UUID(prediction_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    row = await pool.fetchrow(
        "SELECT * FROM b2b_win_loss_predictions WHERE id = $1 AND account_id = $2",
        pid, user.account_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Summary section
    writer.writerow(["Win/Loss Prediction Report"])
    writer.writerow(["Vendor", row["vendor_name"]])
    writer.writerow(["Win Probability", f"{float(row['win_probability']) * 100:.1f}%"])
    writer.writerow(["Confidence", row["confidence"]])
    writer.writerow(["Gated", "Yes" if row["is_gated"] else "No"])
    writer.writerow(["Weights", row["weights_source"] or "static"])
    if row.get("calibration_version"):
        writer.writerow(["Calibration Version", row["calibration_version"]])
    if row.get("company_size"):
        writer.writerow(["Company Size", row["company_size"]])
    if row.get("industry"):
        writer.writerow(["Industry", row["industry"]])
    writer.writerow(["Created", row["created_at"].isoformat()])
    writer.writerow(["Verdict", row["verdict"] or ""])
    writer.writerow([])

    # Factors
    factors = _safe_json(row["factors"]) or []
    if factors:
        writer.writerow(["Scoring Factors"])
        writer.writerow(["Factor", "Score", "Weight", "Data Points", "Evidence"])
        for f in factors:
            writer.writerow([
                f.get("name", ""),
                f"{float(f.get('score') or 0) * 100:.1f}%",
                f"{float(f.get('weight') or 0) * 100:.1f}%",
                f.get("data_points", 0),
                f.get("evidence", ""),
            ])
        writer.writerow([])

    # Switching triggers
    triggers = _safe_json(row["switching_triggers"]) or []
    if triggers:
        writer.writerow(["Switching Triggers"])
        writer.writerow(["Trigger", "Source", "Frequency", "Urgency"])
        for t in triggers:
            writer.writerow([
                t.get("trigger", ""),
                t.get("source", ""),
                t.get("frequency", 0),
                t.get("urgency", 0),
            ])
        writer.writerow([])

    # Displacement targets
    targets = _safe_json(row["displacement_targets"]) or []
    if targets:
        writer.writerow(["Displacement Targets"])
        writer.writerow(["Vendor", "Driver", "Mentions", "Strength"])
        for d in targets:
            writer.writerow([
                d.get("vendor", ""),
                d.get("driver", ""),
                d.get("mentions", 0),
                d.get("strength", ""),
            ])
        writer.writerow([])

    # Proof quotes
    quotes = _safe_json(row["proof_quotes"]) or []
    if quotes:
        writer.writerow(["Proof Quotes"])
        writer.writerow(["Quote", "Source", "Role"])
        for q in quotes:
            writer.writerow([
                q.get("quote", ""),
                q.get("source", ""),
                q.get("role_type", ""),
            ])
        writer.writerow([])

    # Objections
    objections = _safe_json(row["objections"]) or []
    if objections:
        writer.writerow(["Objections"])
        writer.writerow(["Objection", "Frequency", "Counter"])
        for o in objections:
            writer.writerow([
                o.get("objection", ""),
                o.get("frequency", 0),
                o.get("counter", ""),
            ])
        writer.writerow([])

    # Strategy
    if row["recommended_approach"]:
        writer.writerow(["Recommended Approach"])
        writer.writerow([row["recommended_approach"]])
        lead_with = _safe_json(row["lead_with"]) or []
        if lead_with:
            writer.writerow(["Lead With", ", ".join(lead_with)])
        points = _safe_json(row["talking_points"]) or []
        if points:
            writer.writerow(["Talking Points"])
            for i, tp in enumerate(points, 1):
                writer.writerow([f"  {i}. {tp}"])
        if row["timing_advice"]:
            writer.writerow(["Timing", row["timing_advice"]])
        risks = _safe_json(row["risk_factors"]) or []
        if risks:
            writer.writerow(["Risk Factors"])
            for rf in risks:
                writer.writerow([f"  - {rf}"])

    buf.seek(0)
    # Sanitize vendor name for safe filename (strip non-alphanumeric chars)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["vendor_name"].lower())
    filename = f"win_loss_{safe_name}_{row['created_at'].strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
