"""Shared policy helpers for enrichment repair gating and analytics."""

from __future__ import annotations

from typing import Any

STRICT_DISCUSSION_SKIP_MARKER = "repair_skipped_low_signal_discussion"

REPAIR_STRUCTURED_CHURN_SQL = """
(
  COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
  OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
  OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
  OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
)
"""
REPAIR_IDENTITY_SIGNAL_SQL = """
(
  COALESCE(
    CASE
      WHEN LOWER(COALESCE(reviewer_title, '')) LIKE 'repeat churn signal%' THEN NULL
      ELSE NULLIF(reviewer_title, '')
    END,
    NULLIF(reviewer_company, ''),
    NULLIF(enrichment->'reviewer_context'->>'company_name', '')
  ) IS NOT NULL
)
"""
STRONG_VALID_COMPETITOR_OBJECT_SQL = """
EXISTS (
  SELECT 1
  FROM jsonb_array_elements(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)) comp
  WHERE NULLIF(BTRIM(comp->>'name'), '') IS NOT NULL
    AND LOWER(BTRIM(comp->>'name')) <> LOWER(COALESCE(vendor_name, ''))
    AND LOWER(BTRIM(comp->>'name')) <> LOWER(COALESCE(product_name, ''))
    AND LOWER(BTRIM(comp->>'name')) !~ '(integration|app builder|template|cert(?:ification)?|course|academy|private app|custom-built|custom built|our own |api )'
    AND (
      COALESCE(comp->>'evidence_type', '') IN ('explicit_switch', 'active_evaluation')
      OR COALESCE(comp->>'displacement_confidence', '') IN ('high', 'medium')
      OR NULLIF(comp->>'reason', '') IS NOT NULL
      OR NULLIF(comp->>'reason_category', '') IS NOT NULL
      OR NULLIF(comp->>'reason_detail', '') IS NOT NULL
    )
)
"""
REPAIR_VENDOR_LITERAL_SQL = """
(
  review_text ILIKE ('%' || vendor_name || '%')
  OR (
    COALESCE(product_name, '') <> ''
    AND lower(COALESCE(product_name, '')) <> lower(COALESCE(vendor_name, ''))
    AND review_text ILIKE ('%' || product_name || '%')
  )
)
"""
REPAIR_CONTENT_EVIDENCE_SQL = """
(
  COALESCE(jsonb_array_length(enrichment->'pricing_phrases'), 0) > 0
  OR COALESCE(jsonb_array_length(enrichment->'specific_complaints'), 0) > 0
  OR COALESCE(jsonb_array_length(enrichment->'feature_gaps'), 0) > 0
)
"""
ACTIVE_REPAIR_POOL_SQL_TEMPLATE = """
(
  enrichment_status IN ('enriched', 'no_signal')
  AND COALESCE(low_fidelity, false) = false
  AND enrichment IS NOT NULL
  AND enrichment_repair_attempts < {max_attempts}
  AND (
    enrichment_repair_status IS NULL
    OR enrichment_repair_status = 'failed'
    OR enrichment_repair_status = 'promoted'
    OR (
      enrichment_repair_status = 'shadowed'
      AND enrichment_status = 'quarantined'
      AND jsonb_path_exists(
        COALESCE(enrichment_repair_applied_fields, '[]'::jsonb),
        '$[*] ? (@ like_regex "^adjudication:")'
      )
    )
  )
)
"""


def strict_discussion_keep_sql() -> str:
    return f"""
    (
      {REPAIR_STRUCTURED_CHURN_SQL}
      OR {REPAIR_IDENTITY_SIGNAL_SQL}
      OR ({STRONG_VALID_COMPETITOR_OBJECT_SQL})
      OR (
        {REPAIR_VENDOR_LITERAL_SQL}
        AND {REPAIR_CONTENT_EVIDENCE_SQL}
      )
    )
    """


def strict_discussion_gate_sql(source_param: int, content_type_param: int) -> str:
    return f"""
    (
      cardinality(${source_param}::text[]) = 0
      OR lower(source) <> ALL(${source_param}::text[])
      OR cardinality(${content_type_param}::text[]) = 0
      OR lower(COALESCE(content_type, '')) <> ALL(${content_type_param}::text[])
      OR {strict_discussion_keep_sql()}
    )
    """


def _parse_source_allowlist(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",")]
        return {value for value in values if value}
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {
            str(value or "").strip().lower()
            for value in raw
            if str(value or "").strip()
        }
    value = str(raw).strip().lower()
    return {value} if value else set()


def strict_discussion_lists(cfg: Any) -> tuple[list[str], list[str]]:
    raw_sources = getattr(cfg, "enrichment_repair_strict_discussion_sources", None)
    if not isinstance(raw_sources, (str, list, tuple, set, frozenset)):
        raw_sources = "reddit"
    sources = sorted(_parse_source_allowlist(raw_sources))
    raw_content_types = getattr(cfg, "enrichment_repair_strict_discussion_content_types", None)
    if not isinstance(raw_content_types, (list, tuple, set, frozenset)):
        raw_content_types = ["community_discussion", "insider_account", "comment"]
    content_types = sorted(
        {
            str(value or "").strip().lower()
            for value in (raw_content_types or [])
            if str(value or "").strip()
        }
    )
    return sources, content_types
