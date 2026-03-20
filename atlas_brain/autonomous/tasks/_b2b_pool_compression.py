"""Scored pool compression with source traceability for reasoning synthesis v2.

Replaces the inline ``_compress_layers()`` in the synthesis task with:
- Per-item relevance scoring (recency 0.3, signal_strength 0.5, uniqueness 0.2)
- Deterministic source IDs (``pool:kind:key``) on every item
- Pre-computed aggregates with ``{value, _sid}`` wrappers
- A ``CompressedPacket`` that serializes to an LLM-ready payload

Review UUIDs are carried through in ``SourceRef.review_ids`` where the pool
provides them (vault weakness/strength, company signals, provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Scoring coefficients (weights must sum to 1.0)
# ---------------------------------------------------------------------------
W_RECENCY = 0.3
W_SIGNAL = 0.5
W_UNIQUENESS = 0.2

# Recency defaults when no timestamp is available
RECENCY_HIGH = 1.0       # recent / time-sensitive items (spikes, deadlines)
RECENCY_DEFAULT = 0.8    # items with no explicit recency signal
RECENCY_LOW = 0.7        # segment/category items (less time-sensitive)

# Uniqueness penalty thresholds (Jaccard word overlap)
OVERLAP_HIGH = 0.7       # above this -> heavy penalty (0.3)
OVERLAP_MEDIUM = 0.5     # above this -> moderate penalty (0.6)
PENALTY_HIGH = 0.3
PENALTY_MEDIUM = 0.6

# Temporal spike magnitude normalization divisor
SPIKE_MAGNITUDE_MAX = 5.0

# Decision-maker intent boost
DM_INTENT_BOOST = 0.2

# Evidence window depth threshold (days)
MIN_EVIDENCE_WINDOW_DAYS = 14

# Do not surface tiny segments to sales-facing synthesis.
MIN_SEGMENT_SAMPLE_SIZE = 5


# ---------------------------------------------------------------------------
# Source reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SourceRef:
    """Deterministic source identifier for a pool item."""

    pool: str            # e.g. "vault", "segment", "temporal"
    kind: str            # e.g. "weakness", "strength", "flow"
    key: str             # e.g. "pricing", "enterprise_ops_team"
    review_ids: tuple[str, ...] = ()

    @property
    def source_id(self) -> str:
        return f"{self.pool}:{self.kind}:{self.key}"


# ---------------------------------------------------------------------------
# Scored item + tracked aggregate
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ScoredItem:
    """A pool item with its relevance score and source reference."""

    data: dict[str, Any]
    score: float
    source_ref: SourceRef


@dataclass(frozen=True, slots=True)
class TrackedAggregate:
    """A pre-computed number the LLM must reference, with source provenance."""

    label: str
    value: Any
    source_id: str


# ---------------------------------------------------------------------------
# Compressed packet
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CompressedPacket:
    """Full compressed input for one vendor's reasoning synthesis."""

    vendor_name: str
    pools: dict[str, list[ScoredItem]]
    aggregates: list[TrackedAggregate]
    source_registry: dict[str, SourceRef] = field(default_factory=dict)

    def source_ids(self) -> frozenset[str]:
        """All valid source IDs in this packet."""
        ids: set[str] = set()
        for items in self.pools.values():
            for item in items:
                ids.add(item.source_ref.source_id)
        for agg in self.aggregates:
            ids.add(agg.source_id)
        return frozenset(ids)

    def to_llm_payload(self) -> dict[str, Any]:
        """Serialize to JSON-ready dict with ``_sid`` on every item."""
        payload: dict[str, Any] = {}
        for pool_name, items in self.pools.items():
            pool_out: list[dict[str, Any]] = []
            for si in items:
                entry = dict(si.data)
                entry["_sid"] = si.source_ref.source_id
                pool_out.append(entry)
            payload[pool_name] = pool_out

        agg_out: dict[str, dict[str, Any]] = {}
        for agg in self.aggregates:
            agg_out[agg.label] = {"value": agg.value, "_sid": agg.source_id}
        payload["precomputed_aggregates"] = agg_out
        return payload


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Create a URL-safe slug from text for use in source IDs."""
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )[:60]


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _uniqueness_penalty(
    item: dict[str, Any],
    higher_ranked: list[dict[str, Any]],
) -> float:
    """Penalize items that are textually similar to higher-ranked ones.

    Simple overlap check on the string representation.  Returns 1.0 for
    unique, lower for duplicates.
    """
    if not higher_ranked:
        return 1.0
    item_str = str(item).lower()
    for prev in higher_ranked:
        prev_str = str(prev).lower()
        # Jaccard-ish overlap on word sets
        iw = set(item_str.split())
        pw = set(prev_str.split())
        if not iw or not pw:
            continue
        overlap = len(iw & pw) / max(len(iw | pw), 1)
        if overlap > OVERLAP_HIGH:
            return PENALTY_HIGH
        if overlap > OVERLAP_MEDIUM:
            return PENALTY_MEDIUM
    return 1.0


# ---------------------------------------------------------------------------
# Per-pool scorers
# ---------------------------------------------------------------------------

def _score_evidence_vault(
    ev: dict[str, Any], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score evidence vault weakness and strength items."""
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    ms = ev.get("metric_snapshot") or {}
    prov = ev.get("provenance") or {}

    # Aggregates from metric snapshot
    for metric_key in (
        "total_reviews", "churn_density", "displacement_mention_count",
        "avg_rating", "negative_review_pct",
    ):
        val = ms.get(metric_key)
        if val is not None:
            aggregates.append(TrackedAggregate(
                label=metric_key,
                value=val,
                source_id=f"vault:metric:{metric_key}",
            ))

    # Evidence window from provenance
    for prov_key in ("enrichment_window_start", "enrichment_window_end"):
        val = prov.get(prov_key)
        if val is not None:
            aggregates.append(TrackedAggregate(
                label=prov_key,
                value=str(val),
                source_id=f"vault:provenance:{prov_key}",
            ))

    # Score weakness evidence
    weaknesses = ev.get("weakness_evidence") or []
    w_max_mc = (
        max(
            (
                _safe_float(
                    x.get("mention_count_total", x.get("mention_count", 0)),
                )
                for x in weaknesses
            ),
            default=1.0,
        )
        if weaknesses else 1.0
    )
    higher_ranked: list[dict[str, Any]] = []
    for w in weaknesses:
        cat = (
            w.get("key")
            or w.get("label")
            or w.get("category")
            or w.get("theme")
            or "unknown"
        )
        mc = _safe_float(w.get("mention_count_total", w.get("mention_count", 0)))
        signal = mc / max(w_max_mc, 1.0) if w_max_mc > 0 else 0.0
        uniq = _uniqueness_penalty(w, higher_ranked)
        score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * signal + W_UNIQUENESS * uniq

        rids = tuple(w.get("supporting_review_ids") or w.get("review_ids") or [])
        items.append(ScoredItem(
            data=w,
            score=score,
            source_ref=SourceRef(
                pool="vault", kind="weakness", key=_slug(cat),
                review_ids=rids,
            ),
        ))
        higher_ranked.append(w)

    # Score strength evidence
    strengths = ev.get("strength_evidence") or []
    s_max_mc = (
        max(
            (
                _safe_float(
                    x.get("mention_count_total", x.get("mention_count", 0)),
                )
                for x in strengths
            ),
            default=1.0,
        )
        if strengths else 1.0
    )
    higher_ranked = []
    for s in strengths:
        cat = (
            s.get("key")
            or s.get("label")
            or s.get("category")
            or s.get("theme")
            or "unknown"
        )
        mc = _safe_float(s.get("mention_count_total", s.get("mention_count", 0)))
        signal = mc / max(s_max_mc, 1.0) if s_max_mc > 0 else 0.0
        uniq = _uniqueness_penalty(s, higher_ranked)
        score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * signal + W_UNIQUENESS * uniq

        rids = tuple(s.get("supporting_review_ids") or s.get("review_ids") or [])
        items.append(ScoredItem(
            data=s,
            score=score,
            source_ref=SourceRef(
                pool="vault", kind="strength", key=_slug(cat),
                review_ids=rids,
            ),
        ))
        higher_ranked.append(s)

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items * 2], aggregates  # 2x: weaknesses + strengths


def _score_segment(
    seg: dict[str, Any], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score segment intelligence items.

    Real structure: affected_roles, affected_departments,
    contract_segments, affected_company_sizes, usage_duration_segments,
    budget_pressure, buying_stage_distribution, top_use_cases_under_pressure.
    """
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    # Affected departments (list of dicts with department, churn_rate, review_count)
    departments = seg.get("affected_departments") or []
    for d in departments:
        if not isinstance(d, dict):
            continue
        sample_size = _safe_int(d.get("review_count", 0))
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            continue
        name = d.get("department") or "unknown"
        churn = _safe_float(d.get("churn_rate", 0))
        score = W_RECENCY * RECENCY_LOW + W_SIGNAL * churn + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=d,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="department", key=_slug(name),
            ),
        ))

    # Affected roles (list of dicts with role_type, review_count, churn_rate)
    roles = seg.get("affected_roles") or []
    for r in roles:
        if not isinstance(r, dict):
            continue
        sample_size = _safe_int(r.get("review_count", 0))
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            continue
        name = r.get("role_type") or "unknown"
        churn = _safe_float(r.get("churn_rate", 0))
        score = W_RECENCY * RECENCY_LOW + W_SIGNAL * churn + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=r,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="role", key=_slug(name),
            ),
        ))

    # Contract segments (list of dicts with segment, count, churn_rate)
    contracts = seg.get("contract_segments") or []
    for c in contracts:
        if not isinstance(c, dict):
            continue
        sample_size = _safe_int(c.get("count", 0))
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            continue
        name = c.get("segment") or "unknown"
        churn = _safe_float(c.get("churn_rate", 0))
        score = W_RECENCY * RECENCY_LOW + W_SIGNAL * churn + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=c,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="contract", key=_slug(name),
            ),
        ))

    # Budget pressure (dict with dm_churn_rate, price_increase_rate, etc.)
    bp = seg.get("budget_pressure")
    if isinstance(bp, dict):
        dm_churn = bp.get("dm_churn_rate")
        if dm_churn is not None:
            aggregates.append(TrackedAggregate(
                label="dm_churn_rate",
                value=dm_churn,
                source_id="segment:budget:dm_churn_rate",
            ))
        pi_rate = bp.get("price_increase_rate")
        if pi_rate is not None:
            aggregates.append(TrackedAggregate(
                label="price_increase_rate",
                value=pi_rate,
                source_id="segment:budget:price_increase_rate",
            ))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items], aggregates


def _score_temporal(
    temp: dict[str, Any], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score temporal intelligence items."""
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    ts = temp.get("timeline_signal_summary") or {}
    for sig_key in (
        "evaluation_deadline_signals", "contract_end_signals",
        "renewal_signals", "budget_cycle_signals",
    ):
        val = ts.get(sig_key)
        if val is not None:
            aggregates.append(TrackedAggregate(
                label=sig_key,
                value=val,
                source_id=f"temporal:signal:{sig_key}",
            ))

    # Keyword spikes - structure is {spike_count, spike_keywords, keyword_details}
    ks_raw = temp.get("keyword_spikes") or {}
    if isinstance(ks_raw, dict):
        spike_count = ks_raw.get("spike_count", 0)
        if spike_count:
            aggregates.append(TrackedAggregate(
                label="spike_count",
                value=spike_count,
                source_id="temporal:spike:spike_count",
            ))
        # keyword_details has the structured items
        spikes = ks_raw.get("keyword_details") or []
    elif isinstance(ks_raw, list):
        spikes = ks_raw
    else:
        spikes = []

    for sp in spikes:
        if isinstance(sp, str):
            # Plain keyword string, no magnitude info
            items.append(ScoredItem(
                data={"keyword": sp},
                score=W_RECENCY * RECENCY_HIGH + W_SIGNAL * 0.5 + W_UNIQUENESS * 1.0,
                source_ref=SourceRef(
                    pool="temporal", kind="spike", key=_slug(sp),
                ),
            ))
            continue
        if not isinstance(sp, dict):
            continue
        kw = sp.get("keyword") or sp.get("term") or "unknown"
        magnitude = _safe_float(sp.get("magnitude") or sp.get("spike_ratio", 0))
        score = W_RECENCY * RECENCY_HIGH + W_SIGNAL * min(magnitude / SPIKE_MAGNITUDE_MAX, 1.0) + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=sp,
            score=score,
            source_ref=SourceRef(
                pool="temporal", kind="spike", key=_slug(kw),
            ),
        ))

    # Evaluation deadlines
    deadlines = temp.get("evaluation_deadlines") or []
    for dl in deadlines:
        label = dl.get("label") or dl.get("deadline") or "unknown"
        score = W_RECENCY * RECENCY_HIGH + W_SIGNAL * RECENCY_DEFAULT + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=dl,
            score=score,
            source_ref=SourceRef(
                pool="temporal", kind="deadline", key=_slug(str(label)),
            ),
        ))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items], aggregates


def _score_displacement(
    disp: list[dict[str, Any]], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score displacement flow items."""
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    total_switches = 0
    total_evals = 0
    d_max_mc = (
        max(
            (_safe_float((x.get("flow_summary") or {}).get("mention_count", 0)) for x in disp),
            default=1.0,
        )
        if disp else 1.0
    )

    for d in disp:
        to_vendor = d.get("to_vendor") or d.get("competitor") or "unknown"
        fs = d.get("flow_summary") or {}
        mc = _safe_float(fs.get("mention_count", 0))
        switches = int(_safe_float(fs.get("explicit_switch_count", 0)))
        evals = int(_safe_float(fs.get("active_evaluation_count", 0)))
        total_switches += switches
        total_evals += evals

        signal = mc / max(d_max_mc, 1.0) if d_max_mc > 0 else 0.0
        score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * signal + W_UNIQUENESS * 1.0

        items.append(ScoredItem(
            data=d,
            score=score,
            source_ref=SourceRef(
                pool="displacement", kind="flow",
                key=_slug(to_vendor),
            ),
        ))

    aggregates.append(TrackedAggregate(
        label="total_explicit_switches",
        value=total_switches,
        source_id="displacement:aggregate:total_explicit_switches",
    ))
    aggregates.append(TrackedAggregate(
        label="total_active_evaluations",
        value=total_evals,
        source_id="displacement:aggregate:total_active_evaluations",
    ))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items], aggregates


def _score_category(
    cat: dict[str, Any], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score category dynamics items.

    Real structure: category, vendor_count, market_regime,
    displacement_flow_count, council_summary, cross_category_comparison.
    """
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    # Category-level aggregates
    for agg_key in ("vendor_count", "displacement_flow_count"):
        val = cat.get(agg_key)
        if val is not None:
            aggregates.append(TrackedAggregate(
                label=agg_key,
                value=val,
                source_id=f"category:aggregate:{agg_key}",
            ))

    # Market regime as a scored item
    regime = cat.get("market_regime")
    if isinstance(regime, dict):
        regime_type = regime.get("regime_type") or "unknown"
        confidence = _safe_float(regime.get("confidence", 0))
        score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * confidence + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=regime,
            score=score,
            source_ref=SourceRef(
                pool="category", kind="regime", key=_slug(regime_type),
            ),
        ))

    # Council summary if present
    council = cat.get("council_summary")
    if isinstance(council, dict):
        items.append(ScoredItem(
            data=council,
            score=W_RECENCY * RECENCY_LOW + W_SIGNAL * 0.5 + W_UNIQUENESS * 1.0,
            source_ref=SourceRef(
                pool="category", kind="council", key="summary",
            ),
        ))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items], aggregates


def _score_accounts(
    accts: dict[str, Any], max_items: int,
) -> tuple[list[ScoredItem], list[TrackedAggregate]]:
    """Score account intelligence items."""
    items: list[ScoredItem] = []
    aggregates: list[TrackedAggregate] = []

    summary = accts.get("summary") or {}
    for agg_key in (
        "total_accounts",
        "decision_maker_count",
        "high_intent_count",
        "active_eval_signal_count",
    ):
        val = summary.get(agg_key)
        if val is not None:
            aggregates.append(TrackedAggregate(
                label=agg_key,
                value=val,
                source_id=f"accounts:summary:{agg_key}",
            ))

    account_list = accts.get("accounts") or []
    for a in account_list:
        if not isinstance(a, dict):
            continue
        name = a.get("company_name") or a.get("company") or a.get("name") or "unknown"
        intent = _safe_float(
            a.get("urgency_score") or a.get("intent_score") or a.get("confidence_score", 0),
        )
        is_dm = bool(a.get("decision_maker") or a.get("is_decision_maker"))
        signal = intent
        if is_dm:
            signal = min(signal + DM_INTENT_BOOST, 1.0)
        score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * signal + W_UNIQUENESS * 1.0

        rids = tuple(a.get("review_ids") or [])
        items.append(ScoredItem(
            data=a,
            score=score,
            source_ref=SourceRef(
                pool="accounts", kind="company", key=_slug(name),
                review_ids=rids,
            ),
        ))

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:max_items], aggregates


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compress_vendor_pools(
    vendor_name: str,
    layers: dict[str, Any],
    *,
    max_items_per_pool: int = 8,
) -> CompressedPacket:
    """Compress and score all pool layers for one vendor.

    Returns a ``CompressedPacket`` with scored items and pre-computed
    aggregates ready for LLM serialization.
    """
    all_pools: dict[str, list[ScoredItem]] = {}
    all_aggregates: list[TrackedAggregate] = []
    source_reg: dict[str, SourceRef] = {}

    # Evidence vault
    ev = layers.get("evidence_vault") or {}
    if ev:
        items, aggs = _score_evidence_vault(ev, max_items_per_pool)
        all_pools["evidence_vault"] = items
        all_aggregates.extend(aggs)

    # Segment
    seg = layers.get("segment") or {}
    if seg:
        items, aggs = _score_segment(seg, max_items_per_pool)
        all_pools["segment"] = items
        all_aggregates.extend(aggs)

    # Temporal
    temp = layers.get("temporal") or {}
    if temp:
        items, aggs = _score_temporal(temp, max_items_per_pool)
        all_pools["temporal"] = items
        all_aggregates.extend(aggs)

    # Displacement
    disp = layers.get("displacement") or []
    if disp:
        items, aggs = _score_displacement(disp, max_items_per_pool)
        all_pools["displacement"] = items
        all_aggregates.extend(aggs)

    # Category
    cat = layers.get("category") or {}
    if cat:
        items, aggs = _score_category(cat, max_items_per_pool)
        all_pools["category"] = items
        all_aggregates.extend(aggs)

    # Accounts
    accts = layers.get("accounts") or {}
    if accts:
        items, aggs = _score_accounts(accts, max_items_per_pool)
        all_pools["accounts"] = items
        all_aggregates.extend(aggs)

    # Build source registry
    for pool_items in all_pools.values():
        for si in pool_items:
            source_reg[si.source_ref.source_id] = si.source_ref
    for agg in all_aggregates:
        # Create a synthetic SourceRef for aggregates
        parts = agg.source_id.split(":", 2)
        if len(parts) == 3:
            source_reg[agg.source_id] = SourceRef(
                pool=parts[0], kind=parts[1], key=parts[2],
            )

    return CompressedPacket(
        vendor_name=vendor_name,
        pools=all_pools,
        aggregates=all_aggregates,
        source_registry=source_reg,
    )
