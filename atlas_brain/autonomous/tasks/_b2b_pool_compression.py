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

from ._b2b_shared import _canonicalize_competitor, _segment_role_multiplier
from ._b2b_witnesses import build_vendor_witness_artifacts

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
ACTIVE_EVAL_ACCOUNT_BOOST = 0.1

# Evidence window depth threshold (days)
MIN_EVIDENCE_WINDOW_DAYS = 14

# Do not surface tiny segments to sales-facing synthesis.
MIN_SEGMENT_SAMPLE_SIZE = 5

_REASONING_CORE_AGGREGATE_LABELS = frozenset({
    "total_reviews",
    "reviews_in_analysis_window",
    "reviews_in_recent_window",
    "churn_density",
    "avg_urgency",
    "recommend_yes",
    "recommend_no",
    "recommend_ratio",
    "avg_rating",
    "negative_review_pct",
    "positive_review_pct",
    "price_complaint_rate",
    "dm_churn_rate",
    "displacement_mention_count",
    "total_explicit_switches",
    "total_active_evaluations",
    "total_flow_mentions",
    "vendor_count",
    "displacement_flow_count",
    "regime_confidence",
    "regime_avg_churn_velocity",
    "regime_avg_price_pressure",
    "keyword_spike_count",
    "spike_count",
    "evaluation_deadline_signals",
    "contract_end_signals",
    "renewal_signals",
    "budget_cycle_signals",
    "sentiment_declining",
    "sentiment_stable",
    "sentiment_improving",
    "sentiment_total",
    "declining_pct",
    "improving_pct",
    "price_increase_rate",
    "price_increase_count",
    "annual_spend_signal_count",
    "price_per_seat_signal_count",
    "segment_active_eval_signal_count",
    "company_signal_count",
    "company_signal_high_urgency_count",
    "company_signal_evaluation_count",
    "company_signal_active_purchase_count",
    "company_signal_decision_maker_count",
    "total_accounts",
    "decision_maker_count",
    "high_intent_count",
    "active_eval_signal_count",
    "enrichment_window_start",
    "enrichment_window_end",
})


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


def _retain_kind_coverage(
    items: list[ScoredItem],
    *,
    max_items: int,
    required_kinds: tuple[str, ...],
) -> list[ScoredItem]:
    """Keep at least one item for each required kind when a pool is crowded."""
    ranked = sorted(items, key=lambda x: x.score, reverse=True)
    if len(ranked) <= max_items:
        return ranked[:max_items]

    selected: list[ScoredItem] = []
    used_ids: set[int] = set()
    for kind in required_kinds:
        for idx, item in enumerate(ranked):
            if idx in used_ids:
                continue
            if item.source_ref.kind != kind:
                continue
            selected.append(item)
            used_ids.add(idx)
            break

    for idx, item in enumerate(ranked):
        if idx in used_ids:
            continue
        selected.append(item)
        used_ids.add(idx)
        if len(selected) >= max_items:
            break

    selected.sort(key=lambda x: x.score, reverse=True)
    return selected[:max_items]


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
    # Governance / tension signals (Phase 2)
    metric_ledger: list[dict[str, Any]] = field(default_factory=list)
    contradiction_rows: list[dict[str, Any]] = field(default_factory=list)
    minority_signals: list[dict[str, Any]] = field(default_factory=list)
    coverage_gaps: list[dict[str, Any]] = field(default_factory=list)
    retention_proof: list[dict[str, Any]] = field(default_factory=list)
    witness_pack: list[dict[str, Any]] = field(default_factory=list)
    section_packets: dict[str, Any] = field(default_factory=dict)

    def source_ids(self) -> frozenset[str]:
        """All valid source IDs in this packet."""
        ids: set[str] = set()
        for items in self.pools.values():
            for item in items:
                ids.add(item.source_ref.source_id)
        for agg in self.aggregates:
            ids.add(agg.source_id)
        # Include governance source IDs
        for entry in self.metric_ledger:
            sid = entry.get("_sid")
            if sid:
                ids.add(sid)
        for entry in self.contradiction_rows:
            sid = entry.get("_sid")
            if sid:
                ids.add(sid)
        for entry in self.minority_signals:
            sid = entry.get("_sid")
            if sid:
                ids.add(sid)
        for entry in self.coverage_gaps:
            sid = entry.get("_sid")
            if sid:
                ids.add(sid)
        for entry in self.retention_proof:
            sid = entry.get("_sid")
            if sid:
                ids.add(sid)
        for witness in self.witness_pack:
            sid = witness.get("_sid") or witness.get("witness_id")
            if sid:
                ids.add(str(sid))
        return frozenset(ids)

    def _reasoning_aggregate_source_ids(self) -> set[str]:
        """Aggregate IDs worth sending to the reasoning model."""
        retained_prefixes = {
            item.source_ref.source_id
            for items in self.pools.values()
            for item in items
        }
        allowed: set[str] = set()
        for agg in self.aggregates:
            sid = agg.source_id
            if agg.label in _REASONING_CORE_AGGREGATE_LABELS:
                allowed.add(sid)
                continue
            for prefix in retained_prefixes:
                if sid == prefix or sid.startswith(f"{prefix}:"):
                    allowed.add(sid)
                    break
        return allowed

    def to_llm_payload(
        self,
        *,
        compact_metric_ledger: bool = False,
        compact_aggregates: bool = False,
    ) -> dict[str, Any]:
        """Serialize to JSON-ready dict with ``_sid`` on every item."""
        payload: dict[str, Any] = {}
        for pool_name, items in self.pools.items():
            pool_out: list[dict[str, Any]] = []
            for si in items:
                entry = dict(si.data)
                entry["_sid"] = si.source_ref.source_id
                pool_out.append(entry)
            payload[pool_name] = pool_out

        allowed_aggregate_ids = (
            self._reasoning_aggregate_source_ids()
            if compact_aggregates
            else None
        )
        agg_out: dict[str, dict[str, Any]] = {}
        for agg in self.aggregates:
            if allowed_aggregate_ids is not None and agg.source_id not in allowed_aggregate_ids:
                continue
            agg_out[agg.label] = {"value": agg.value, "_sid": agg.source_id}
        payload["precomputed_aggregates"] = agg_out

        # Governance / tension signals
        if self.metric_ledger:
            metric_ledger = self.metric_ledger
            if allowed_aggregate_ids is not None:
                metric_ledger = [
                    entry for entry in self.metric_ledger
                    if str(entry.get("_sid") or "") in allowed_aggregate_ids
                ]
            if compact_metric_ledger:
                payload["metric_ledger"] = [
                    {
                        "label": entry.get("label"),
                        "scope": entry.get("scope"),
                        "_sid": entry.get("_sid"),
                    }
                    for entry in metric_ledger
                ]
            else:
                payload["metric_ledger"] = metric_ledger
        if self.contradiction_rows:
            payload["contradiction_rows"] = self.contradiction_rows
        if self.minority_signals:
            payload["minority_signals"] = self.minority_signals
        if self.coverage_gaps:
            payload["coverage_gaps"] = self.coverage_gaps
        if self.retention_proof:
            payload["retention_proof"] = self.retention_proof
        if self.witness_pack:
            payload["witness_pack"] = self.witness_pack
        if self.section_packets:
            payload["section_packets"] = self.section_packets

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


def _normalize_account_intent_score(val: Any) -> float:
    score = _safe_float(val, 0.0)
    if score <= 0:
        return 0.0
    if score > 1.0:
        score = score / 10.0
    return max(0.0, min(score, 1.0))


def _is_active_eval_stage(stage: Any) -> bool:
    text = str(stage or "").strip().lower()
    if not text:
        return False
    if text in {"evaluation", "active_purchase", "consideration", "trial", "poc"}:
        return True
    return "evaluat" in text or "consider" in text


def _looks_like_displacement_tool_label(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    generic_modifiers = ("custom", "homegrown", "home-grown", "in-house", "internal")
    generic_artifacts = (
        "integration", "utility", "workflow", "tool", "stack",
        "system", "automation", "bot",
    )
    return any(token in text for token in generic_modifiers) and any(
        token in text for token in generic_artifacts
    )


def _displacement_flow_counts(flow: dict[str, Any]) -> tuple[float, int, int]:
    summary = flow.get("flow_summary") or {}
    edge_metrics = flow.get("edge_metrics") or {}
    mentions = _safe_float(
        summary.get("mention_count")
        or summary.get("total_flow_mentions")
        or edge_metrics.get("mention_count"),
        0.0,
    )
    switches = _safe_int(summary.get("explicit_switch_count"), 0)
    evals = _safe_int(summary.get("active_evaluation_count"), 0)
    return mentions, switches, evals


def _merge_displacement_flow_rows(flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for flow in flows or []:
        if not isinstance(flow, dict):
            continue
        raw_to_vendor = str(flow.get("to_vendor") or flow.get("competitor") or "").strip()
        canonical_to_vendor = _canonicalize_competitor(raw_to_vendor) or raw_to_vendor
        if not canonical_to_vendor or _looks_like_displacement_tool_label(canonical_to_vendor):
            continue
        from_vendor = str(flow.get("from_vendor") or "").strip()
        if from_vendor and canonical_to_vendor.lower() == from_vendor.lower():
            continue

        key = canonical_to_vendor.lower()
        current = merged.get(key)
        if current is None:
            current = dict(flow)
            current["to_vendor"] = canonical_to_vendor
            current["flow_summary"] = dict(flow.get("flow_summary") or {})
            current["edge_metrics"] = dict(flow.get("edge_metrics") or {})
            current["switch_reasons"] = list(flow.get("switch_reasons") or [])
            current["evidence_breakdown"] = list(flow.get("evidence_breakdown") or [])
            merged[key] = current
            continue

        existing_mentions, existing_switches, existing_evals = _displacement_flow_counts(current)
        new_mentions, new_switches, new_evals = _displacement_flow_counts(flow)

        current_summary = dict(current.get("flow_summary") or {})
        current_summary["total_flow_mentions"] = existing_mentions + new_mentions
        current_summary["mention_count"] = existing_mentions + new_mentions
        current_summary["explicit_switch_count"] = existing_switches + new_switches
        current_summary["active_evaluation_count"] = existing_evals + new_evals
        current["flow_summary"] = current_summary

        current_edge = dict(current.get("edge_metrics") or {})
        new_edge = dict(flow.get("edge_metrics") or {})
        if new_mentions > existing_mentions:
            if new_edge.get("key_quote"):
                current_edge["key_quote"] = new_edge.get("key_quote")
            if new_edge.get("primary_driver"):
                current_edge["primary_driver"] = new_edge.get("primary_driver")
            if new_edge.get("signal_strength"):
                current_edge["signal_strength"] = new_edge.get("signal_strength")
        current_edge["mention_count"] = existing_mentions + new_mentions
        current_edge["velocity_7d"] = _safe_float(current_edge.get("velocity_7d"), 0.0) + _safe_float(new_edge.get("velocity_7d"), 0.0)
        current_edge["velocity_30d"] = _safe_float(current_edge.get("velocity_30d"), 0.0) + _safe_float(new_edge.get("velocity_30d"), 0.0)
        current_edge["confidence_score"] = max(
            _safe_float(current_edge.get("confidence_score"), 0.0),
            _safe_float(new_edge.get("confidence_score"), 0.0),
        )
        current["edge_metrics"] = current_edge

        current["switch_reasons"] = list(current.get("switch_reasons") or []) + list(flow.get("switch_reasons") or [])
        current["evidence_breakdown"] = list(current.get("evidence_breakdown") or []) + list(flow.get("evidence_breakdown") or [])

    return list(merged.values())


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


def _vault_item_recency_score(item: dict[str, Any]) -> float:
    """Derive a recency score from recent-count and trend metadata."""
    total = _safe_float(
        item.get("mention_count_total", item.get("mention_count", 0)),
        0.0,
    )
    recent = _safe_float(item.get("mention_count_recent"), 0.0)
    recency = RECENCY_DEFAULT
    if total > 0 and recent > 0:
        ratio = min(max(recent / total, 0.0), 1.0)
        recency = max(recency, 0.7 + 0.3 * ratio)

    trend = item.get("trend") if isinstance(item.get("trend"), dict) else {}
    direction = str(trend.get("direction") or "").strip().lower()
    if direction in {"accelerating", "new"}:
        recency = min(1.0, recency + 0.15)
    elif direction == "declining":
        recency = max(RECENCY_LOW, recency - 0.15)
    return recency


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
        "total_reviews",
        "reviews_in_analysis_window",
        "reviews_in_recent_window",
        "churn_density",
        "avg_urgency",
        "recommend_yes",
        "recommend_no",
        "recommend_ratio",
        "price_complaint_rate",
        "dm_churn_rate",
        "positive_review_pct",
        "displacement_mention_count",
        "keyword_spike_count",
        "avg_rating",
        "negative_review_pct",
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

    company_signals = ev.get("company_signals") or []
    if company_signals:
        high_urgency_count = 0
        evaluation_count = 0
        active_purchase_count = 0
        decision_maker_count = 0
        for signal in company_signals:
            if not isinstance(signal, dict):
                continue
            urgency = _safe_float(signal.get("urgency_score"), 0.0)
            stage = str(signal.get("buying_stage") or "").strip().lower()
            if urgency >= 8.0:
                high_urgency_count += 1
            if "evaluat" in stage:
                evaluation_count += 1
            elif stage == "active_purchase":
                active_purchase_count += 1
            if signal.get("decision_maker") is True:
                decision_maker_count += 1

        for label, value, sid in (
            ("company_signal_count", len(company_signals), "vault:company_signals:count"),
            ("company_signal_high_urgency_count", high_urgency_count, "vault:company_signals:high_urgency_count"),
            ("company_signal_evaluation_count", evaluation_count, "vault:company_signals:evaluation_count"),
            ("company_signal_active_purchase_count", active_purchase_count, "vault:company_signals:active_purchase_count"),
            ("company_signal_decision_maker_count", decision_maker_count, "vault:company_signals:decision_maker_count"),
        ):
            aggregates.append(TrackedAggregate(
                label=label,
                value=value,
                source_id=sid,
            ))

        higher_ranked_companies: list[dict[str, Any]] = []
        for signal in company_signals:
            if not isinstance(signal, dict):
                continue
            name = str(signal.get("company_name") or signal.get("company") or "").strip()
            if not name:
                continue
            intent = _normalize_account_intent_score(
                signal.get("urgency_score") or signal.get("intent_score") or signal.get("confidence_score", 0),
            )
            is_dm = bool(signal.get("decision_maker") or signal.get("is_decision_maker"))
            is_active_eval = _is_active_eval_stage(signal.get("buying_stage"))
            signal_strength = intent
            if is_dm:
                signal_strength = min(signal_strength + DM_INTENT_BOOST, 1.0)
            if is_active_eval:
                signal_strength = min(signal_strength + ACTIVE_EVAL_ACCOUNT_BOOST, 1.0)
            uniq = _uniqueness_penalty(signal, higher_ranked_companies)
            score = W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * signal_strength + W_UNIQUENESS * uniq
            review_id = str(signal.get("review_id") or "").strip()
            review_ids = (review_id,) if review_id else ()
            items.append(ScoredItem(
                data=signal,
                score=score,
                source_ref=SourceRef(
                    pool="vault",
                    kind="company",
                    key=_slug(name),
                    review_ids=review_ids,
                ),
            ))
            higher_ranked_companies.append(signal)

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
        slug = _slug(cat)
        mc = _safe_float(w.get("mention_count_total", w.get("mention_count", 0)))
        mc_recent = w.get("mention_count_recent")
        signal = mc / max(w_max_mc, 1.0) if w_max_mc > 0 else 0.0
        uniq = _uniqueness_penalty(w, higher_ranked)
        recency = _vault_item_recency_score(w)
        score = W_RECENCY * recency + W_SIGNAL * signal + W_UNIQUENESS * uniq

        aggregates.append(TrackedAggregate(
            label=f"vault_weakness_{slug}_mention_count_total",
            value=mc,
            source_id=f"vault:weakness:{slug}:mention_count_total",
        ))
        if mc_recent is not None:
            aggregates.append(TrackedAggregate(
                label=f"vault_weakness_{slug}_mention_count_recent",
                value=mc_recent,
                source_id=f"vault:weakness:{slug}:mention_count_recent",
            ))

        rids = tuple(w.get("supporting_review_ids") or w.get("review_ids") or [])
        items.append(ScoredItem(
            data=w,
            score=score,
            source_ref=SourceRef(
                pool="vault", kind="weakness", key=slug,
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
        slug = _slug(cat)
        mc = _safe_float(s.get("mention_count_total", s.get("mention_count", 0)))
        mc_recent = s.get("mention_count_recent")
        signal = mc / max(s_max_mc, 1.0) if s_max_mc > 0 else 0.0
        uniq = _uniqueness_penalty(s, higher_ranked)
        recency = _vault_item_recency_score(s)
        score = W_RECENCY * recency + W_SIGNAL * signal + W_UNIQUENESS * uniq

        aggregates.append(TrackedAggregate(
            label=f"vault_strength_{slug}_mention_count_total",
            value=mc,
            source_id=f"vault:strength:{slug}:mention_count_total",
        ))
        if mc_recent is not None:
            aggregates.append(TrackedAggregate(
                label=f"vault_strength_{slug}_mention_count_recent",
                value=mc_recent,
                source_id=f"vault:strength:{slug}:mention_count_recent",
            ))

        rids = tuple(s.get("supporting_review_ids") or s.get("review_ids") or [])
        items.append(ScoredItem(
            data=s,
            score=score,
            source_ref=SourceRef(
                pool="vault", kind="strength", key=slug,
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

    def _append_reach_aggregate(kind: str, name: str, sample_size: int) -> None:
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            return
        slug = _slug(name or "unknown")
        aggregates.append(TrackedAggregate(
            label=f"segment_reach_{kind}_{slug}",
            value=sample_size,
            source_id=f"segment:reach:{kind}:{slug}",
        ))

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
        _append_reach_aggregate("department", name, sample_size)

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
        priority_score = _safe_float(r.get("priority_score"), default=-1.0)
        if priority_score < 0:
            priority_score = sample_size * _segment_role_multiplier(name)
        role_bonus = priority_score / 100.0
        score = (
            W_RECENCY * RECENCY_LOW
            + W_SIGNAL * churn
            + W_UNIQUENESS * 1.0
            + role_bonus
        )
        items.append(ScoredItem(
            data=r,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="role", key=_slug(name),
            ),
        ))
        _append_reach_aggregate("role", name, sample_size)

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
        _append_reach_aggregate("contract", name, sample_size)

    # Usage duration segments (list of dicts with duration, count, churn_rate)
    durations = seg.get("usage_duration_segments") or []
    for d in durations:
        if not isinstance(d, dict):
            continue
        sample_size = _safe_int(d.get("count", 0))
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            continue
        name = d.get("duration") or "unknown"
        churn = _safe_float(d.get("churn_rate", 0))
        score = W_RECENCY * RECENCY_LOW + W_SIGNAL * churn + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=d,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="duration", key=_slug(name),
            ),
        ))
        _append_reach_aggregate("duration", name, sample_size)

    # Use cases under pressure (list of dicts with use_case, mention_count)
    use_cases = seg.get("top_use_cases_under_pressure") or []
    max_mentions = max(
        (_safe_float(u.get("mention_count", 0)) for u in use_cases if isinstance(u, dict)),
        default=1.0,
    )
    higher_ranked_use_cases: list[dict[str, Any]] = []
    for u in use_cases:
        if not isinstance(u, dict):
            continue
        sample_size = _safe_int(u.get("mention_count", 0))
        if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
            continue
        name = u.get("use_case") or "unknown"
        confidence = _safe_float(u.get("confidence_score", 0))
        signal = max(sample_size / max(max_mentions, 1.0), confidence)
        uniq = _uniqueness_penalty(u, higher_ranked_use_cases)
        score = W_RECENCY * RECENCY_LOW + W_SIGNAL * signal + W_UNIQUENESS * uniq
        items.append(ScoredItem(
            data=u,
            score=score,
            source_ref=SourceRef(
                pool="segment", kind="use_case", key=_slug(name),
            ),
        ))
        _append_reach_aggregate("use_case", name, sample_size)
        higher_ranked_use_cases.append(u)

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
        pi_count = bp.get("price_increase_count")
        if pi_count is not None:
            aggregates.append(TrackedAggregate(
                label="price_increase_count",
                value=pi_count,
                source_id="segment:budget:price_increase_count",
            ))
        annual_spend_signals = bp.get("annual_spend_signals") or []
        if annual_spend_signals:
            aggregates.append(TrackedAggregate(
                label="annual_spend_signal_count",
                value=len(annual_spend_signals),
                source_id="segment:budget:annual_spend_signal_count",
            ))
        price_per_seat_signals = bp.get("price_per_seat_signals") or []
        if price_per_seat_signals:
            aggregates.append(TrackedAggregate(
                label="price_per_seat_signal_count",
                value=len(price_per_seat_signals),
                source_id="segment:budget:price_per_seat_signal_count",
            ))

    # Company size signals (dict with avg/median/max seat counts)
    company_sizes = seg.get("affected_company_sizes")
    if isinstance(company_sizes, dict):
        for field in ("avg_seat_count", "median_seat_count", "max_seat_count"):
            value = company_sizes.get(field)
            if value is None:
                continue
            aggregates.append(TrackedAggregate(
                label=f"segment_{field}",
                value=value,
                source_id=f"segment:size:{field}",
            ))
        for entry in company_sizes.get("size_distribution") or []:
            if not isinstance(entry, dict):
                continue
            sample_size = _safe_int(entry.get("review_count", 0))
            if sample_size < MIN_SEGMENT_SAMPLE_SIZE:
                continue
            name = entry.get("segment") or "unknown"
            churn = _safe_float(entry.get("churn_rate", 0))
            score = W_RECENCY * RECENCY_LOW + W_SIGNAL * churn + W_UNIQUENESS * 1.0
            items.append(ScoredItem(
                data=entry,
                score=score,
                source_ref=SourceRef(
                    pool="segment", kind="size", key=_slug(name),
                ),
            ))
            _append_reach_aggregate("size", name, sample_size)

    # Buying-stage distribution can carry real evaluation pressure even when
    # named account extraction is sparse or aggressively sanitized.
    def _is_active_eval_stage(stage: Any) -> bool:
        text = str(stage or "").strip().lower()
        if not text:
            return False
        if text in {
            "evaluation", "active_purchase", "consideration", "trial", "poc",
        }:
            return True
        return "evaluat" in text or "consider" in text

    active_eval_count = 0
    for row in seg.get("buying_stage_distribution") or []:
        if not isinstance(row, dict):
            continue
        if not _is_active_eval_stage(row.get("stage")):
            continue
        active_eval_count += _safe_int(row.get("count", 0))

    if active_eval_count:
        aggregates.append(TrackedAggregate(
            label="segment_active_eval_signal_count",
            value=active_eval_count,
            source_id="segment:aggregate:active_eval_signal_count",
        ))

    return _retain_kind_coverage(
        items,
        max_items=max_items,
        required_kinds=(
            "role",
            "department",
            "contract",
            "duration",
            "size",
            "use_case",
        ),
    ), aggregates


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

    def _spike_signal_strength(spike: dict[str, Any]) -> float:
        magnitude = _safe_float(
            spike.get("magnitude") or spike.get("spike_ratio"),
            0.0,
        )
        base = min(magnitude / SPIKE_MAGNITUDE_MAX, 1.0) if magnitude > 0 else 0.0
        change_pct = abs(_safe_float(spike.get("change_pct"), 0.0))
        if change_pct > 1.0:
            change_pct = change_pct / 100.0
        volume = _safe_float(spike.get("volume"), 0.0)
        volume_signal = min(volume / 10.0, 1.0) if volume > 0 else 0.0
        if spike.get("is_spike"):
            base = max(base, 0.5)
        return min(max(base, change_pct, volume_signal * 0.5), 1.0)

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
        signal_strength = _spike_signal_strength(sp)
        score = W_RECENCY * RECENCY_HIGH + W_SIGNAL * signal_strength + W_UNIQUENESS * 1.0
        items.append(ScoredItem(
            data=sp,
            score=score,
            source_ref=SourceRef(
                pool="temporal", kind="spike", key=_slug(kw),
            ),
        ))

    sent = temp.get("sentiment_trajectory") or {}
    if isinstance(sent, dict):
        for key in ("declining", "stable", "improving", "total"):
            val = sent.get(key)
            if val is not None:
                aggregates.append(TrackedAggregate(
                    label=f"sentiment_{key}",
                    value=val,
                    source_id=f"temporal:sentiment:{key}_count",
                ))
        for key in ("declining_pct", "improving_pct"):
            val = sent.get(key)
            if val is not None:
                aggregates.append(TrackedAggregate(
                    label=key,
                    value=val,
                    source_id=f"temporal:sentiment:{key}",
                ))
        for direction in ("declining", "improving", "stable"):
            count = _safe_int(sent.get(direction))
            total = max(_safe_int(sent.get("total")), 1)
            if count <= 0:
                continue
            pct = _safe_float(sent.get(f"{direction}_pct"), count / total)
            items.append(ScoredItem(
                data={
                    "direction": direction,
                    "count": count,
                    "pct": pct,
                },
                score=W_RECENCY * RECENCY_DEFAULT + W_SIGNAL * min(pct, 1.0) + W_UNIQUENESS * 1.0,
                source_ref=SourceRef(
                    pool="temporal", kind="sentiment", key=_slug(direction),
                ),
            ))

    trigger_entries = temp.get("immediate_triggers") or []
    deadline_entries = temp.get("evaluation_deadlines") or []
    seen_triggers: set[tuple[str, str]] = set()

    def _append_trigger_item(entry: dict[str, Any]) -> None:
        trigger_type = str(
            entry.get("trigger_type")
            or entry.get("type")
            or "signal"
        ).strip().lower()
        if not trigger_type:
            trigger_type = "signal"
        label = (
            entry.get("label")
            or entry.get("trigger")
            or entry.get("deadline")
            or entry.get("evaluation_deadline")
            or entry.get("contract_end")
            or entry.get("decision_timeline")
            or entry.get("date")
            or entry.get("company")
            or trigger_type
        )
        key = _slug(str(label))
        dedupe = (trigger_type, key)
        if dedupe in seen_triggers:
            return
        seen_triggers.add(dedupe)
        urgency = _safe_float(entry.get("urgency"), 0.0)
        recency = RECENCY_HIGH if trigger_type in {"deadline", "contract_end", "spike"} else RECENCY_DEFAULT
        signal_strength = min(max(urgency / 10.0, 0.4 if trigger_type in {"timeline_signal", "signal"} else 0.0), 1.0)
        item = dict(entry)
        item.setdefault("type", trigger_type)
        item.setdefault("trigger_type", trigger_type)
        items.append(ScoredItem(
            data=item,
            score=W_RECENCY * recency + W_SIGNAL * signal_strength + W_UNIQUENESS * 1.0,
            source_ref=SourceRef(
                pool="temporal", kind=trigger_type, key=key,
            ),
        ))

    for trigger in trigger_entries:
        if isinstance(trigger, dict):
            _append_trigger_item(trigger)
    for dl in deadline_entries:
        if isinstance(dl, dict):
            _append_trigger_item(dl)

    turning_points = temp.get("turning_points") or []
    max_turning_mentions = max(
        (_safe_float(item.get("mentions", 0)) for item in turning_points if isinstance(item, dict)),
        default=1.0,
    )
    for tp in turning_points:
        if not isinstance(tp, dict):
            continue
        trigger = tp.get("trigger") or "unknown"
        mentions = _safe_float(tp.get("mentions", 0))
        score = (
            W_RECENCY * RECENCY_DEFAULT
            + W_SIGNAL * (mentions / max(max_turning_mentions, 1.0))
            + W_UNIQUENESS * 1.0
        )
        items.append(ScoredItem(
            data=tp,
            score=score,
            source_ref=SourceRef(
                pool="temporal", kind="turning_point", key=_slug(str(trigger)),
            ),
        ))

    sentiment_tenure = temp.get("sentiment_tenure") or []
    max_tenure_count = max(
        (_safe_float(item.get("count", 0)) for item in sentiment_tenure if isinstance(item, dict)),
        default=1.0,
    )
    for tenure in sentiment_tenure:
        if not isinstance(tenure, dict):
            continue
        label = tenure.get("tenure") or "unknown"
        count = _safe_float(tenure.get("count", 0))
        score = (
            W_RECENCY * RECENCY_LOW
            + W_SIGNAL * (count / max(max_tenure_count, 1.0))
            + W_UNIQUENESS * 1.0
        )
        items.append(ScoredItem(
            data=tenure,
            score=score,
            source_ref=SourceRef(
                pool="temporal", kind="tenure", key=_slug(str(label)),
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

    disp = _merge_displacement_flow_rows(disp)
    total_switches = 0
    total_evals = 0
    total_mentions = 0
    d_max_mc = (
        max(
            (_displacement_flow_counts(x)[0] for x in disp),
            default=1.0,
        )
        if disp else 1.0
    )
    d_max_switches = (
        max((_displacement_flow_counts(x)[1] for x in disp), default=1)
        if disp else 1
    )
    d_max_evals = (
        max((_displacement_flow_counts(x)[2] for x in disp), default=1)
        if disp else 1
    )

    for d in disp:
        to_vendor = d.get("to_vendor") or d.get("competitor") or "unknown"
        mc, switches, evals = _displacement_flow_counts(d)
        fs = dict(d.get("flow_summary") or {})
        fs["mention_count"] = mc
        fs["total_flow_mentions"] = mc
        fs["explicit_switch_count"] = switches
        fs["active_evaluation_count"] = evals
        d["flow_summary"] = fs
        total_switches += switches
        total_evals += evals
        total_mentions += int(mc)

        mention_signal = mc / max(d_max_mc, 1.0) if d_max_mc > 0 else 0.0
        switch_signal = switches / max(d_max_switches, 1.0) if d_max_switches > 0 else 0.0
        eval_signal = evals / max(d_max_evals, 1.0) if d_max_evals > 0 else 0.0
        signal = 0.55 * switch_signal + 0.3 * eval_signal + 0.15 * mention_signal
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
    aggregates.append(TrackedAggregate(
        label="total_flow_mentions",
        value=total_mentions,
        source_id="displacement:aggregate:total_flow_mentions",
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
        for field in ("confidence", "avg_churn_velocity", "avg_price_pressure"):
            value = regime.get(field)
            if value is None:
                continue
            aggregates.append(TrackedAggregate(
                label=f"regime_{field}",
                value=value,
                source_id=f"category:regime:{field}",
            ))
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
        council_confidence = _safe_float(council.get("confidence", 0.0))
        items.append(ScoredItem(
            data=council,
            score=W_RECENCY * RECENCY_LOW + W_SIGNAL * max(council_confidence, 0.5) + W_UNIQUENESS * 1.0,
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

    summary = dict(accts.get("summary") or {})
    account_list = accts.get("accounts") or []

    if "total_accounts" not in summary:
        summary["total_accounts"] = len(account_list)

    if "high_intent_count" not in summary:
        from ...config import settings

        threshold = float(
            getattr(settings.b2b_churn, "high_churn_urgency_threshold", 7.0),
        )
        summary["high_intent_count"] = sum(
            1
            for a in account_list
            if _safe_float(a.get("urgency_score") or a.get("intent_score"), 0.0)
            >= threshold
        )

    if "active_eval_signal_count" not in summary:
        summary["active_eval_signal_count"] = sum(
            1
            for a in account_list
            if _is_active_eval_stage(a.get("buying_stage"))
        )

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

    for a in account_list:
        if not isinstance(a, dict):
            continue
        name = a.get("company_name") or a.get("company") or a.get("name") or "unknown"
        intent = _normalize_account_intent_score(
            a.get("urgency_score") or a.get("intent_score") or a.get("confidence_score", 0),
        )
        is_dm = bool(a.get("decision_maker") or a.get("is_decision_maker"))
        is_active_eval = _is_active_eval_stage(a.get("buying_stage"))
        signal = intent
        if is_dm:
            signal = min(signal + DM_INTENT_BOOST, 1.0)
        if is_active_eval:
            signal = min(signal + ACTIVE_EVAL_ACCOUNT_BOOST, 1.0)
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
# Governance / tension signal extractors (Phase 2 packet widening)
# ---------------------------------------------------------------------------

# Dimensions that can carry contradictions across segments.
_CONTRADICTION_DIMENSIONS = (
    "support", "pricing", "usability", "integrations", "performance",
    "reliability", "onboarding", "documentation", "security",
)

# Segment kinds used for contradiction detection.
_SEGMENT_KINDS_FOR_CONTRADICTIONS = (
    "role", "department", "size", "duration", "contract",
)


def _build_metric_ledger(
    aggregates: list[TrackedAggregate],
    *,
    analysis_window_days: int = 90,
) -> list[dict[str, Any]]:
    """Build a scoped metric ledger from pre-computed aggregates.

    Every numeric claim the LLM might cite must come from this ledger so
    downstream validators can reject unsupported numbers.
    """
    # Metrics that are safe for all surfaces
    _ALL_SURFACES = ["report", "battle_card", "blog", "campaign"]
    # Metrics only appropriate for internal analysis
    _INTERNAL = ["report", "battle_card"]

    # (label, scope_category, surfaces)
    _METRIC_REGISTRY: dict[str, tuple[str, list[str]]] = {
        # Volume / density
        "total_reviews": ("review_volume", _ALL_SURFACES),
        "reviews_in_analysis_window": ("review_volume", _ALL_SURFACES),
        "reviews_in_recent_window": ("review_volume", _ALL_SURFACES),
        "churn_density": ("churn_intensity", _ALL_SURFACES),
        "avg_urgency": ("churn_intensity", _ALL_SURFACES),
        # Sentiment
        "recommend_ratio": ("sentiment", _ALL_SURFACES),
        "recommend_yes": ("sentiment", _ALL_SURFACES),
        "recommend_no": ("sentiment", _ALL_SURFACES),
        "avg_rating": ("sentiment", _ALL_SURFACES),
        "negative_review_pct": ("sentiment", _ALL_SURFACES),
        "positive_review_pct": ("sentiment", _ALL_SURFACES),
        # Pricing pressure
        "price_complaint_rate": ("pricing_pressure", _ALL_SURFACES),
        # Decision-maker signals
        "dm_churn_rate": ("decision_maker_signals", _INTERNAL),
        "company_signal_decision_maker_count": ("decision_maker_signals", _INTERNAL),
        # Displacement
        "displacement_mention_count": ("displacement", _ALL_SURFACES),
        "total_explicit_switches": ("displacement", _ALL_SURFACES),
        "total_active_evaluations": ("displacement", _ALL_SURFACES),
        "total_flow_mentions": ("displacement", _INTERNAL),
        # Category
        "vendor_count": ("category_dynamics", _INTERNAL),
        "displacement_flow_count": ("category_dynamics", _INTERNAL),
        "regime_confidence": ("category_dynamics", _INTERNAL),
        "regime_avg_churn_velocity": ("category_dynamics", _INTERNAL),
        "regime_avg_price_pressure": ("category_dynamics", _INTERNAL),
        # Temporal signals
        "keyword_spike_count": ("temporal_spikes", _INTERNAL),
        "spike_count": ("temporal_spikes", _INTERNAL),
        "evaluation_deadline_signals": ("temporal_signals", _INTERNAL),
        "contract_end_signals": ("temporal_signals", _INTERNAL),
        "renewal_signals": ("temporal_signals", _INTERNAL),
        "budget_cycle_signals": ("temporal_signals", _INTERNAL),
        # Temporal sentiment
        "sentiment_declining": ("temporal_sentiment", _INTERNAL),
        "sentiment_stable": ("temporal_sentiment", _INTERNAL),
        "sentiment_improving": ("temporal_sentiment", _INTERNAL),
        "sentiment_total": ("temporal_sentiment", _INTERNAL),
        "declining_pct": ("temporal_sentiment", _INTERNAL),
        "improving_pct": ("temporal_sentiment", _INTERNAL),
        # Segment budget
        "price_increase_rate": ("segment_budget", _INTERNAL),
        "price_increase_count": ("segment_budget", _INTERNAL),
        "annual_spend_signal_count": ("segment_budget", _INTERNAL),
        "price_per_seat_signal_count": ("segment_budget", _INTERNAL),
        # Account signals
        "company_signal_count": ("account_signals", _ALL_SURFACES),
        "company_signal_high_urgency_count": ("account_signals", _INTERNAL),
        "company_signal_evaluation_count": ("account_signals", _INTERNAL),
        "company_signal_active_purchase_count": ("account_signals", _INTERNAL),
        "segment_active_eval_signal_count": ("account_signals", _INTERNAL),
        # Account summary
        "total_accounts": ("account_signals", _ALL_SURFACES),
        "decision_maker_count": ("account_signals", _INTERNAL),
        "high_intent_count": ("account_signals", _INTERNAL),
        "active_eval_signal_count": ("account_signals", _INTERNAL),
    }

    ledger: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for agg in aggregates:
        dedupe_key = (agg.label, agg.source_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        reg = _METRIC_REGISTRY.get(agg.label)
        if reg is not None:
            scope_category, surfaces = reg
        elif agg.label.startswith("vault_weakness_"):
            scope_category, surfaces = "weakness_mentions", _INTERNAL
        elif agg.label.startswith("vault_strength_"):
            scope_category, surfaces = "strength_mentions", _INTERNAL
        elif agg.label.startswith("segment_reach_"):
            scope_category, surfaces = "segment_reach", _INTERNAL
        elif agg.label.startswith("segment_"):
            scope_category, surfaces = "segment_metrics", _INTERNAL
        else:
            continue
        ledger.append({
            "label": agg.label,
            "value": agg.value,
            "scope": scope_category,
            "time_window_days": analysis_window_days,
            "allowed_surfaces": surfaces,
            "_sid": agg.source_id,
        })
    return ledger


def _build_contradiction_rows(
    pools: dict[str, list[ScoredItem]],
) -> list[dict[str, Any]]:
    """Detect segment-level contradictions across dimensions.

    A contradiction exists when two segments within the same dimension
    (e.g., support) have opposing sentiment signals -- one positive, one
    negative. This forces the LLM to hedge rather than generalise.
    """
    segment_items = pools.get("segment", [])

    # Group items by weakness/strength dimension mentioned in the data
    # Key: (dimension, segment_kind) -> list of (segment_label, sentiment)
    signals_by_dim: dict[str, list[tuple[str, str, str]]] = {}

    for si in segment_items:
        kind = si.source_ref.kind
        if kind not in _SEGMENT_KINDS_FOR_CONTRADICTIONS:
            continue
        seg_label = si.source_ref.key
        churn_rate = _safe_float(si.data.get("churn_rate", 0))
        # Infer sentiment from churn rate
        if churn_rate >= 0.4:
            sentiment = "negative"
        elif churn_rate <= 0.15:
            sentiment = "positive"
        else:
            continue  # ambiguous, skip
        signals_by_dim.setdefault(kind, []).append((seg_label, sentiment, kind))

    # Also look at weakness/strength evidence vault items for dimension signals
    vault_items = pools.get("evidence_vault", [])
    weakness_dims: set[str] = set()
    strength_dims: set[str] = set()
    for si in vault_items:
        dim = si.source_ref.key
        if si.source_ref.kind == "weakness":
            weakness_dims.add(dim)
        elif si.source_ref.kind == "strength":
            strength_dims.add(dim)

    rows: list[dict[str, Any]] = []

    # Contradiction type 1: same segment kind has both high-churn and low-churn segments
    for kind, entries in signals_by_dim.items():
        positives = [e for e in entries if e[1] == "positive"]
        negatives = [e for e in entries if e[1] == "negative"]
        if positives and negatives:
            rows.append({
                "dimension": kind,
                "segment_a": negatives[0][0],
                "segment_b": positives[0][0],
                "statement_a": "negative",
                "statement_b": "positive",
                "_sid": f"segment:contradiction:{kind}",
            })

    # Contradiction type 2: dimension appears in both weakness AND strength evidence
    overlap_dims = weakness_dims & strength_dims
    for dim in sorted(overlap_dims):
        rows.append({
            "dimension": dim,
            "segment_a": "weakness_evidence",
            "segment_b": "strength_evidence",
            "statement_a": "negative",
            "statement_b": "positive",
            "_sid": f"vault:contradiction:{dim}",
        })

    return rows


def _build_minority_signals(
    pools: dict[str, list[ScoredItem]],
    aggregates: list[TrackedAggregate],
) -> list[dict[str, Any]]:
    """Extract rare-but-severe signals that scored compression might drop.

    A minority signal is one with high urgency but low mention count --
    the kind of thing that gets silently dropped by top-N truncation
    but could be a critical blocker for specific segments.
    """
    signals: list[dict[str, Any]] = []

    vault_items = pools.get("evidence_vault", [])
    for si in vault_items:
        if si.source_ref.kind != "weakness":
            continue
        mc = _safe_float(
            si.data.get("mention_count_total", si.data.get("mention_count", 0)),
        )
        urgency = _safe_float(si.data.get("avg_urgency", si.data.get("urgency", 0)))
        # Minority: low count but high urgency
        if mc <= 5 and urgency >= 7.0:
            label = si.source_ref.key
            signals.append({
                "label": label,
                "urgency": round(urgency, 1),
                "count": int(mc),
                "reason": "rare_but_severe",
                "_sid": f"vault:minority:{label}",
            })

    # Also check account signals for isolated high-urgency DM signals
    account_items = pools.get("accounts", [])
    for si in account_items:
        urgency = _safe_float(
            si.data.get("urgency_score", si.data.get("intent_score", 0)),
        )
        is_dm = bool(si.data.get("decision_maker") or si.data.get("is_decision_maker"))
        if is_dm and urgency >= 9.0:
            name = si.source_ref.key
            signals.append({
                "label": f"dm_alert_{name}",
                "urgency": round(urgency, 1),
                "count": 1,
                "reason": "decision_maker_extreme_urgency",
                "_sid": f"accounts:minority:{name}",
            })

    return signals


def _build_coverage_gaps(
    pools: dict[str, list[ScoredItem]],
    aggregates: list[TrackedAggregate],
) -> list[dict[str, Any]]:
    """Detect areas where evidence is thin and conclusions should be hedged.

    Coverage gaps are NOT missing data -- they are areas where data exists
    but is too sparse to support confident claims.
    """
    gaps: list[dict[str, Any]] = []

    # Gap type 1: thin segment samples (segment items that were retained
    # despite being near the MIN_SEGMENT_SAMPLE_SIZE threshold)
    segment_items = pools.get("segment", [])
    for si in segment_items:
        sample_size = _safe_int(
            si.data.get("review_count", si.data.get("count", 0)),
        )
        if 0 < sample_size < MIN_SEGMENT_SAMPLE_SIZE * 2:
            area = f"{si.source_ref.kind}_{si.source_ref.key}"
            gaps.append({
                "type": "thin_segment_sample",
                "area": area,
                "sample_size": sample_size,
                "_sid": f"gap:thin_segment:{area}",
            })

    # Gap type 2: no displacement evidence
    disp_items = pools.get("displacement", [])
    if not disp_items:
        gaps.append({
            "type": "missing_pool",
            "area": "displacement",
            "sample_size": 0,
            "_sid": "gap:missing_pool:displacement",
        })

    # Gap type 3: very few account signals
    account_items = pools.get("accounts", [])
    if len(account_items) < 3:
        gaps.append({
            "type": "thin_account_signals",
            "area": "accounts",
            "sample_size": len(account_items),
            "_sid": "gap:thin_accounts:accounts",
        })

    # Gap type 4: shallow evidence window
    window_start = None
    window_end = None
    for agg in aggregates:
        if agg.label == "enrichment_window_start":
            window_start = agg.value
        elif agg.label == "enrichment_window_end":
            window_end = agg.value
    if window_start and window_end:
        try:
            from datetime import date as _date
            start = _date.fromisoformat(str(window_start)[:10])
            end = _date.fromisoformat(str(window_end)[:10])
            window_days = (end - start).days
            if window_days < MIN_EVIDENCE_WINDOW_DAYS:
                gaps.append({
                    "type": "shallow_evidence_window",
                    "area": "temporal_depth",
                    "sample_size": window_days,
                    "_sid": "gap:shallow_window:temporal_depth",
                })
        except (ValueError, TypeError):
            pass

    return gaps


def _build_retention_proof(
    pools: dict[str, list[ScoredItem]],
) -> list[dict[str, Any]]:
    """Extract strength evidence that explains why customers stay despite frustration.

    Retention proof is the counter-signal to churn pressure. Without it,
    the LLM over-indexes on churn signals and produces unrealistically
    aggressive positioning.
    """
    proofs: list[dict[str, Any]] = []

    vault_items = pools.get("evidence_vault", [])
    for si in vault_items:
        if si.source_ref.kind != "strength":
            continue
        area = si.source_ref.key
        mc = _safe_float(
            si.data.get("mention_count_total", si.data.get("mention_count", 0)),
        )
        # Only include strengths with meaningful evidence
        if mc < 3:
            continue
        # Build a strength summary from available data
        summary_parts: list[str] = []
        for field in ("summary", "description", "key", "label", "category", "theme"):
            val = si.data.get(field)
            if val and isinstance(val, str) and len(val) > 5:
                summary_parts.append(val)
                break
        strength_text = summary_parts[0] if summary_parts else area.replace("_", " ")
        proofs.append({
            "area": area,
            "strength": strength_text,
            "mention_count": int(mc),
            "_sid": f"vault:strength:{area}",
        })

    return proofs


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
    disp = layers.get("displacement")
    if layers:
        items, aggs = _score_displacement(disp or [], max_items_per_pool)
        if items:
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

    # Build governance / tension signals
    from ...config import settings

    cfg = settings.b2b_churn
    metric_ledger = _build_metric_ledger(all_aggregates)
    contradiction_rows = _build_contradiction_rows(all_pools)
    minority_signals = _build_minority_signals(all_pools, all_aggregates)
    coverage_gaps = _build_coverage_gaps(all_pools, all_aggregates)
    retention_proof = _build_retention_proof(all_pools)
    witness_pack, section_packets = build_vendor_witness_artifacts(
        vendor_name,
        layers.get("reviews") or [],
        max_witnesses=int(getattr(cfg, "reasoning_witness_max_witnesses", 12)),
        min_specificity_score=float(
            getattr(cfg, "witness_specificity_min_score", 2.0),
        ),
        fallback_min_witnesses=int(
            getattr(cfg, "witness_specificity_fallback_min_witnesses", 4),
        ),
        generic_patterns=list(
            getattr(cfg, "witness_specificity_generic_patterns", []) or [],
        ),
        concrete_patterns=list(
            getattr(cfg, "witness_specificity_concrete_patterns", []) or [],
        ),
        short_excerpt_chars=int(
            getattr(cfg, "witness_specificity_short_excerpt_chars", 55),
        ),
        long_excerpt_chars=int(
            getattr(cfg, "witness_specificity_long_excerpt_chars", 80),
        ),
        specificity_weights=dict(
            getattr(cfg, "witness_specificity_weights", {}) or {},
        ),
    )

    return CompressedPacket(
        vendor_name=vendor_name,
        pools=all_pools,
        aggregates=all_aggregates,
        source_registry=source_reg,
        metric_ledger=metric_ledger,
        contradiction_rows=contradiction_rows,
        minority_signals=minority_signals,
        coverage_gaps=coverage_gaps,
        retention_proof=retention_proof,
        witness_pack=witness_pack,
        section_packets=section_packets,
    )
