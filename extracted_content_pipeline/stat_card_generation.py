"""Deterministic stat-card drafts from Content Ops source material."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

from .campaign_customer_data import CampaignOpportunityWarning
from .campaign_ports import TenantScope
from .campaign_source_adapters import (
    source_material_to_source_rows,
    source_row_to_campaign_opportunity,
)
from .stat_card_ports import StatCardDraft, StatCardRepository


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
_FIELD_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class StatCardGenerationConfig:
    """Config for deterministic source-material stat cards."""

    limit: int = 3
    max_text_chars: int = 600
    max_claim_chars: int = 90
    max_headline_chars: int = 90
    max_supporting_text_chars: int = 160
    max_evidence_chars: int = 220


@dataclass(frozen=True)
class StatCardGenerationResult:
    """Generated stat cards plus non-fatal source-material warnings."""

    stats: tuple[dict[str, Any], ...]
    warnings: tuple[CampaignOpportunityWarning, ...] = ()
    target_mode: str = "vendor_retention"
    saved_ids: tuple[str, ...] = ()

    @property
    def generated(self) -> int:
        return len(self.stats)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "target_mode": self.target_mode,
            "stats": [dict(stat) for stat in self.stats],
            "warnings": [warning.as_dict() for warning in self.warnings],
            "saved_ids": list(self.saved_ids),
        }


@dataclass(frozen=True)
class _MetricDefinition:
    field_names: tuple[str, ...]
    label: str
    support_focus: str


_METRICS: tuple[_MetricDefinition, ...] = (
    _MetricDefinition(("nps_score", "nps"), "NPS score", "customer sentiment"),
    _MetricDefinition(("csat_score", "csat"), "CSAT score", "customer satisfaction"),
    _MetricDefinition(("opportunity_score",), "Opportunity score", "opportunity strength"),
    _MetricDefinition(("urgency_score",), "Urgency score", "buyer urgency"),
    _MetricDefinition(("rating", "review_rating", "star_rating"), "Rating", "review rating"),
    _MetricDefinition(("ticket_count", "support_ticket_count", "case_count"), "Ticket count", "support volume"),
    _MetricDefinition(("response_count", "survey_response_count"), "Response count", "response volume"),
    _MetricDefinition(("win_rate", "conversion_rate", "retention_rate"), "Rate", "conversion signal"),
)


class StatCardGenerationService:
    """Build short, evidence-backed stat-card drafts from source material."""

    def __init__(
        self,
        config: StatCardGenerationConfig | None = None,
        *,
        stat_cards: StatCardRepository | None = None,
    ) -> None:
        self.config = config or StatCardGenerationConfig()
        self._stat_cards = stat_cards

    async def generate(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        source_material: Any,
        limit: int | None = None,
        max_text_chars: int | None = None,
        **kwargs: Any,
    ) -> StatCardGenerationResult:
        del kwargs
        resolved_limit = int(limit) if limit is not None else self.config.limit
        if resolved_limit < 1:
            raise ValueError("limit must be at least 1")
        resolved_max_text_chars = (
            int(max_text_chars)
            if max_text_chars is not None
            else self.config.max_text_chars
        )
        if resolved_max_text_chars < 1:
            raise ValueError("max_text_chars must be at least 1")
        result = _generate_stat_cards(
            source_material_to_source_rows(source_material),
            target_mode=target_mode,
            limit=resolved_limit,
            max_text_chars=resolved_max_text_chars,
            max_claim_chars=self.config.max_claim_chars,
            max_headline_chars=self.config.max_headline_chars,
            max_supporting_text_chars=self.config.max_supporting_text_chars,
            max_evidence_chars=self.config.max_evidence_chars,
        )
        if self._stat_cards is None or not result.stats:
            return result
        saved_ids = tuple(
            str(item)
            for item in await self._stat_cards.save_drafts(
                _drafts_from_stats(result.stats, target_mode=target_mode),
                scope=scope,
            )
        )
        return replace(result, saved_ids=saved_ids)


def _generate_stat_cards(
    rows: Sequence[Any],
    *,
    target_mode: str,
    limit: int,
    max_text_chars: int,
    max_claim_chars: int,
    max_headline_chars: int,
    max_supporting_text_chars: int,
    max_evidence_chars: int,
) -> StatCardGenerationResult:
    stats: list[dict[str, Any]] = []
    warnings: list[CampaignOpportunityWarning] = []
    for index, row in enumerate(rows, start=1):
        if len(stats) >= limit:
            break
        if not isinstance(row, Mapping):
            warnings.append(CampaignOpportunityWarning(
                code="row_not_object",
                row_index=index,
                message="Skipped source row because it is not an object.",
            ))
            continue
        opportunity, row_warnings = source_row_to_campaign_opportunity(
            row,
            row_index=index,
            max_text_chars=max_text_chars,
        )
        warnings.extend(row_warnings)
        card, card_warnings = _card_from_opportunity(
            opportunity,
            index=len(stats) + 1,
            row_index=index,
            max_claim_chars=max_claim_chars,
            max_headline_chars=max_headline_chars,
            max_supporting_text_chars=max_supporting_text_chars,
            max_evidence_chars=max_evidence_chars,
        )
        warnings.extend(card_warnings)
        if card is not None:
            stats.append(card)
    return StatCardGenerationResult(
        stats=tuple(stats),
        warnings=tuple(warnings),
        target_mode=target_mode,
    )


def _card_from_opportunity(
    opportunity: Mapping[str, Any],
    *,
    index: int,
    row_index: int,
    max_claim_chars: int,
    max_headline_chars: int,
    max_supporting_text_chars: int,
    max_evidence_chars: int,
) -> tuple[dict[str, Any] | None, tuple[CampaignOpportunityWarning, ...]]:
    if not opportunity:
        return None, (
            CampaignOpportunityWarning(
                code="missing_stat_card_metric",
                row_index=row_index,
                message="Skipped source row because it did not contain a supported numeric metric.",
            ),
        )
    evidence = _first_evidence(opportunity)
    if not evidence:
        return None, (
            CampaignOpportunityWarning(
                code="missing_stat_card_evidence",
                row_index=row_index,
                field="evidence",
                message="Skipped source row because it did not contain source evidence.",
            ),
        )

    warnings: list[CampaignOpportunityWarning] = []
    saw_metric = False
    for metric in _METRICS:
        field_name, raw_value = _metric_value(opportunity, metric.field_names)
        if field_name is None:
            continue
        saw_metric = True
        numeric_value = _clean_number(raw_value)
        if numeric_value is None:
            warnings.append(CampaignOpportunityWarning(
                code="invalid_stat_card_metric",
                row_index=row_index,
                field=field_name,
                message="Skipped metric because its value is not numeric.",
            ))
            continue
        evidence_snippet = _evidence_snippet_for_value(
            evidence,
            numeric_value,
            max_evidence_chars,
        )
        if evidence_snippet is None:
            warnings.append(CampaignOpportunityWarning(
                code="unsupported_numeric_claim",
                row_index=row_index,
                field=field_name,
                message="Skipped metric because the numeric value is not present in source evidence.",
            ))
            continue
        return _supported_card(
            opportunity,
            metric=metric,
            field_name=field_name,
            numeric_value=numeric_value,
            evidence=evidence_snippet,
            index=index,
            max_claim_chars=max_claim_chars,
            max_headline_chars=max_headline_chars,
            max_supporting_text_chars=max_supporting_text_chars,
        ), tuple(warnings)

    if not saw_metric:
        warnings.append(CampaignOpportunityWarning(
            code="missing_stat_card_metric",
            row_index=row_index,
            message="Skipped source row because it did not contain a supported numeric metric.",
        ))
    return None, tuple(warnings)


def _supported_card(
    opportunity: Mapping[str, Any],
    *,
    metric: _MetricDefinition,
    field_name: str,
    numeric_value: int | float,
    evidence: str,
    index: int,
    max_claim_chars: int,
    max_headline_chars: int,
    max_supporting_text_chars: int,
) -> dict[str, Any]:
    vendor = _clean(opportunity.get("vendor_name") or opportunity.get("vendor"))
    company = _clean(opportunity.get("company_name"))
    pain = _first_text(opportunity.get("pain_points"))
    source_id = _clean(opportunity.get("source_id") or opportunity.get("target_id"))
    metric_display = _format_number(numeric_value)
    label = _metric_label(metric, field_name)
    claim = f"{label}: {metric_display}"
    headline = f"Customer metric for {vendor}" if vendor else "Customer metric"
    supporting_parts = ["Use this stat"]
    if pain:
        supporting_parts.append(f"to frame {pain}")
    else:
        supporting_parts.append(f"to frame {metric.support_focus}")
    return {
        "id": source_id or f"stat-card-{index}",
        "theme": "customer_metric",
        "metric_label": label,
        "metric_value": numeric_value,
        "metric_display": metric_display,
        "claim": _truncate(claim, max_claim_chars),
        "headline": _truncate(headline, max_headline_chars),
        "supporting_text": _truncate(
            " ".join(supporting_parts) + ".",
            max_supporting_text_chars,
        ),
        "evidence": evidence,
        "source_id": source_id,
        "source_type": _clean(opportunity.get("source_type")),
        "target_id": _clean(opportunity.get("target_id")),
        "company_name": company,
        "vendor_name": vendor,
        "pain_points": list(opportunity.get("pain_points") or ()),
    }


def _metric_value(
    opportunity: Mapping[str, Any],
    field_names: Sequence[str],
) -> tuple[str | None, Any]:
    for field_name in field_names:
        if field_name in opportunity and opportunity.get(field_name) not in (None, ""):
            return field_name, opportunity.get(field_name)
    compact = {
        _compact_field_name(str(key)): str(key)
        for key, value in opportunity.items()
        if value not in (None, "")
    }
    for field_name in field_names:
        key = compact.get(_compact_field_name(field_name))
        if key is not None:
            return key, opportunity.get(key)
    return None, None


def _metric_label(metric: _MetricDefinition, field_name: str) -> str:
    if metric.label != "Rate":
        return metric.label
    normalized = field_name.replace("_", " ").strip()
    return normalized.title() if normalized else metric.label


def _first_evidence(opportunity: Mapping[str, Any]) -> str:
    raw = opportunity.get("evidence")
    if isinstance(raw, Mapping):
        return _clean(raw.get("text"))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                text = _clean(item.get("text"))
                if text:
                    return text
            else:
                text = _clean(item)
                if text:
                    return text
    return _clean(raw)


def _drafts_from_stats(
    stats: Sequence[Mapping[str, Any]],
    *,
    target_mode: str,
) -> tuple[StatCardDraft, ...]:
    drafts: list[StatCardDraft] = []
    for stat in stats:
        source_id = _clean(stat.get("source_id") or stat.get("id"))
        target_id = _clean(stat.get("target_id")) or source_id
        drafts.append(
            StatCardDraft(
                target_id=target_id,
                target_mode=target_mode,
                theme=_clean(stat.get("theme")) or "customer_metric",
                metric_label=_clean(stat.get("metric_label")),
                metric_value=stat.get("metric_value"),
                metric_display=_clean(stat.get("metric_display")),
                claim=_clean(stat.get("claim")),
                headline=_clean(stat.get("headline")),
                supporting_text=_clean(stat.get("supporting_text")),
                evidence=_clean(stat.get("evidence")),
                source_id=source_id,
                source_type=_clean(stat.get("source_type")),
                company_name=_clean(stat.get("company_name")),
                vendor_name=_clean(stat.get("vendor_name")),
                pain_points=_pain_points_from_stat(stat.get("pain_points")),
                metadata={"source_card": dict(stat)},
            )
        )
    return tuple(drafts)


def _pain_points_from_stat(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(_clean(item) for item in value if _clean(item))


def _evidence_snippet_for_value(
    evidence: str,
    value: int | float,
    max_chars: int,
) -> str | None:
    text = _clean(evidence)
    for match in _NUMBER_RE.finditer(text):
        matched = _clean_number(match.group(0))
        if matched is not None and _numbers_equal(matched, value):
            return _bounded_evidence_snippet(
                text,
                match_start=match.start(),
                match_end=match.end(),
                max_chars=max_chars,
            )
    return None


def _bounded_evidence_snippet(
    evidence: str,
    *,
    match_start: int,
    match_end: int,
    max_chars: int,
) -> str:
    if len(evidence) <= max_chars:
        return evidence
    if max_chars <= 3:
        return evidence[match_start:match_end][:max_chars]
    body_chars = max_chars - 3
    if match_end <= body_chars:
        return _truncate(evidence, max_chars)
    start = max(0, match_end - body_chars)
    if start > match_start:
        start = match_start
    end = min(len(evidence), start + body_chars)
    if match_end > end:
        end = match_end
        start = max(0, end - body_chars)
    snippet = evidence[start:end].strip()
    if start <= 0:
        return _truncate(evidence, max_chars)
    return "..." + snippet


def _numbers_equal(left: int | float, right: int | float) -> bool:
    return abs(float(left) - float(right)) < 0.000001


def _clean_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    text = _clean(value)
    if not text:
        return None
    text = text.rstrip("%").replace(",", "")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            text = _clean(item)
            if text:
                return text
    return _clean(value)


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _compact_field_name(value: str) -> str:
    return _FIELD_SEPARATOR_RE.sub("", value.lower())


def _truncate(value: str, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "StatCardGenerationConfig",
    "StatCardGenerationResult",
    "StatCardGenerationService",
]
