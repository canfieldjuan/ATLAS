"""Read-model helpers for B2B enrichment extraction health."""

from __future__ import annotations

import ast
import json
from typing import Any

_ENRICHED_SCOPE = "enrichment_status = 'enriched'"
_ARRAY_LEN = """
CASE
  WHEN jsonb_typeof(enrichment->{field}) = 'array'
  THEN jsonb_array_length(enrichment->{field})
  ELSE 0
END
"""
_PHRASE_ARRAYS_PRESENT = """
(
  {specific_complaints} > 0
  OR {pricing_phrases} > 0
  OR {recommendation_language} > 0
  OR {feature_gaps} > 0
  OR {event_mentions} > 0
  OR {competitors_mentioned} > 0
)
""".format(
    specific_complaints=_ARRAY_LEN.format(field="'specific_complaints'"),
    pricing_phrases=_ARRAY_LEN.format(field="'pricing_phrases'"),
    recommendation_language=_ARRAY_LEN.format(field="'recommendation_language'"),
    feature_gaps=_ARRAY_LEN.format(field="'feature_gaps'"),
    event_mentions=_ARRAY_LEN.format(field="'event_mentions'"),
    competitors_mentioned=_ARRAY_LEN.format(field="'competitors_mentioned'"),
)
_MISSING_OR_EMPTY_SPANS = """
(
  enrichment->'evidence_spans' IS NULL
  OR jsonb_typeof(enrichment->'evidence_spans') != 'array'
  OR {evidence_spans_len} = 0
)
""".format(evidence_spans_len=_ARRAY_LEN.format(field="'evidence_spans'"))
_EMPTY_SALIENCE_FLAGS = """
(
  enrichment->'salience_flags' IS NULL
  OR jsonb_typeof(enrichment->'salience_flags') != 'array'
  OR {salience_flags_len} = 0
)
""".format(salience_flags_len=_ARRAY_LEN.format(field="'salience_flags'"))
_BLANK_REPLACEMENT_MODE = "COALESCE(enrichment->>'replacement_mode', '') = ''"
_BLANK_OPERATING_MODEL_SHIFT = "COALESCE(enrichment->>'operating_model_shift', '') = ''"
_BLANK_PRODUCTIVITY_DELTA = "COALESCE(enrichment->>'productivity_delta_claim', '') = ''"
_BLANK_ORG_PRESSURE = "COALESCE(enrichment->>'org_pressure_type', '') = ''"
_BLANK_EVIDENCE_MAP_HASH = "COALESCE(enrichment->>'evidence_map_hash', '') = ''"
_PHRASE_ARRAYS_NO_SPANS = f"({_PHRASE_ARRAYS_PRESENT} AND {_MISSING_OR_EMPTY_SPANS})"
_HARD_GAP = f"""(
  {_PHRASE_ARRAYS_NO_SPANS}
  OR {_BLANK_REPLACEMENT_MODE}
  OR {_BLANK_OPERATING_MODEL_SHIFT}
  OR {_BLANK_PRODUCTIVITY_DELTA}
  OR {_BLANK_ORG_PRESSURE}
  OR {_MISSING_OR_EMPTY_SPANS}
  OR {_BLANK_EVIDENCE_MAP_HASH}
)"""
_COMMERCIAL_CONTEXT = r"(alternative|alternatives|budget|contract|cost|expensive|migrate|migration|pricing|renewal|replace|replaced|seat|seats|support|switch|switching)"
_AMBIGUOUS_VENDOR = "(LOWER(vendor_name) IN ('copper', 'close'))"
_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT = r"(crm|sales|pipeline|lead|leads|deal|deals|account|contact|contacts|prospect|prospects|software|saas)"
_VENDOR_REFERENCE = """
(
  (
    review_text ILIKE ('%%' || vendor_name || '%%')
    AND (
      NOT {ambiguous_vendor}
      OR (
        review_text ~* '{commercial_context}'
        AND review_text ~* '{ambiguous_vendor_product_context}'
      )
    )
  )
  OR (
    COALESCE(product_name, '') <> ''
    AND LOWER(COALESCE(product_name, '')) <> LOWER(COALESCE(vendor_name, ''))
    AND review_text ILIKE ('%%' || product_name || '%%')
  )
)
"""
_EMPLOYMENT_CONTEXT = r"(work(?:ing)? at|employee|career|manager|my manager|our team|interview|hiring|certification|freelance|rep at|joined|left my role|leaving my full time|promotion|salary)"
_VALID_COMPETITOR_OBJECT = """
EXISTS (
  SELECT 1
  FROM jsonb_array_elements(COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb)) comp
  WHERE NULLIF(BTRIM(comp->>'name'), '') IS NOT NULL
    AND LOWER(BTRIM(comp->>'name')) <> LOWER(COALESCE(vendor_name, ''))
    AND LOWER(BTRIM(comp->>'name')) <> LOWER(COALESCE(product_name, ''))
    AND LOWER(BTRIM(comp->>'name')) !~ '(integration|app builder|template|cert(?:ification)?|course|academy|private app|custom-built|custom built|our own |api )'
)
"""
_STRONG_VALID_COMPETITOR_OBJECT = """
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
_MONEY_WITHOUT_PRICING_SPAN = """
(
  ({money_signal})
  AND (
    COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    OR review_text ~* '(pricing|price|priced|cost|costly|expensive|cheaper|budget|billing|invoice|refund|overcharg|renewal|per seat|per user|subscription|license|licensed|plan|plan tier|seat|user)'
  )
  AND NOT (
    source = 'reddit'
    AND NOT {vendor_reference}
  )
  AND NOT (
    content_type IN ('community_discussion', 'insider_account')
    AND NOT (
      COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    )
    AND COALESCE(
      CASE
        WHEN LOWER(COALESCE(reviewer_title, '')) LIKE 'repeat churn signal%' THEN NULL
        ELSE NULLIF(reviewer_title, '')
      END,
      NULLIF(reviewer_company, ''),
      NULLIF(enrichment->'reviewer_context'->>'company_name', '')
    ) IS NULL
    AND NOT jsonb_path_exists(
      COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb),
      '$[*] ? (@.evidence_type == "explicit_switch" || @.evidence_type == "active_evaluation" || @.displacement_confidence == "high" || @.displacement_confidence == "medium" || @.reason != null || @.reason_category != null || @.reason_detail != null)'
    )
  )
  AND NOT jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == "pricing_backlash")')
)
""".format(
    money_signal=r"(review_text ~* '\$\s?\d' OR COALESCE(enrichment->'salience_flags', '[]'::jsonb) ? 'explicit_dollar')",
    vendor_reference=_VENDOR_REFERENCE.format(
        ambiguous_vendor=_AMBIGUOUS_VENDOR,
        commercial_context=_COMMERCIAL_CONTEXT,
        ambiguous_vendor_product_context=_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT,
    ),
)
_COMPETITOR_WITHOUT_DISPLACEMENT = """
(
  (
    review_text ~* '(switched to|moved to|replaced with|migrating to|migration to)'
    OR (
      review_text ~* '(evaluating|looking at|considering|shortlisting|shortlisted|poc with|proof of concept with)'
      AND {valid_competitor_object}
    )
    OR jsonb_path_exists(
      COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb),
      '$[*] ? (@.evidence_type == "explicit_switch" || @.evidence_type == "active_evaluation" || @.displacement_confidence == "high" || @.displacement_confidence == "medium" || @.reason != null || @.reason_category != null || @.reason_detail != null)'
    )
  )
  AND (
    COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
    OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    OR {specific_complaints_len} > 0
    OR {pricing_phrases_len} > 0
    OR {feature_gaps_len} > 0
    OR review_text ~* '(cancel|cancellation|refund|billing dispute|renewal|price increase|overcharg|not worth|switch|switched to|moved to|replaced with|evaluating|considering|alternative|frustrated|pain|issue|problem)'
  )
  AND NOT (
    source = 'reddit'
    AND NOT {vendor_reference}
    AND NOT review_text ~* '{commercial_context}'
  )
  AND NOT (
    content_type IN ('community_discussion', 'insider_account')
    AND {vendor_reference}
    AND NOT (
      COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    )
    AND review_text ~* '{employment_context}'
    AND NOT review_text ~* '{commercial_context}'
  )
  AND NOT (
    content_type IN ('community_discussion', 'insider_account')
    AND NOT (
      COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    )
    AND COALESCE(
      CASE
        WHEN LOWER(COALESCE(reviewer_title, '')) LIKE 'repeat churn signal%' THEN NULL
        ELSE NULLIF(reviewer_title, '')
      END,
      NULLIF(reviewer_company, ''),
      NULLIF(enrichment->'reviewer_context'->>'company_name', '')
    ) IS NULL
    AND NOT jsonb_path_exists(
      COALESCE(enrichment->'competitors_mentioned', '[]'::jsonb),
      '$[*] ? (@.evidence_type == "explicit_switch" || @.evidence_type == "active_evaluation" || @.displacement_confidence == "high" || @.displacement_confidence == "medium" || @.reason != null || @.reason_category != null || @.reason_detail != null)'
    )
  )
  AND NOT (
    {strong_valid_competitor_object}
    OR jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.signal_type == "competitor_pressure")')
    OR lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'competitor_switch'
  )
)
""".format(
    valid_competitor_object=_VALID_COMPETITOR_OBJECT,
    strong_valid_competitor_object=_STRONG_VALID_COMPETITOR_OBJECT,
    specific_complaints_len=_ARRAY_LEN.format(field="'specific_complaints'"),
    pricing_phrases_len=_ARRAY_LEN.format(field="'pricing_phrases'"),
    feature_gaps_len=_ARRAY_LEN.format(field="'feature_gaps'"),
    vendor_reference=_VENDOR_REFERENCE.format(
        ambiguous_vendor=_AMBIGUOUS_VENDOR,
        commercial_context=_COMMERCIAL_CONTEXT,
        ambiguous_vendor_product_context=_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT,
    ),
    commercial_context=_COMMERCIAL_CONTEXT,
    employment_context=_EMPLOYMENT_CONTEXT,
)
_NAMED_COMPANY_WITHOUT_ACCOUNT_EVIDENCE = """
(
  NULLIF(enrichment->'reviewer_context'->>'company_name', '') IS NOT NULL
  AND NOT (
    COALESCE(enrichment->'salience_flags', '[]'::jsonb) ? 'named_account'
    OR jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? (@.company_name != null && @.company_name != "")')
  )
)
"""
_TIMELINE_WITHOUT_ANCHOR = """
(
(
  NULLIF(enrichment->'churn_signals'->>'renewal_timing', '') IS NOT NULL
  OR review_text ~* '(renewal|contract end|contract expires|deadline)'
  OR (
    review_text ~* '(next quarter|q1|q2|q3|q4|30 days|60 days|90 days)'
    AND review_text ~* '(renewal|contract|evaluating|evaluation|considering|switch|switched|migrating|migration|replatform|cancel|deadline|go live|go-live|cutover)'
    )
  )
  AND NOT (
    source = 'reddit'
    AND NOT {vendor_reference}
  )
  AND NOT (
    content_type IN ('community_discussion', 'insider_account')
    AND NOT (
      COALESCE((enrichment->'churn_signals'->>'intent_to_leave')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'actively_evaluating')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'migration_in_progress')::boolean, false)
      OR COALESCE((enrichment->'churn_signals'->>'contract_renewal_mentioned')::boolean, false)
    )
    AND COALESCE(
      CASE
        WHEN LOWER(COALESCE(reviewer_title, '')) LIKE 'repeat churn signal%' THEN NULL
        ELSE NULLIF(reviewer_title, '')
      END,
      NULLIF(reviewer_company, ''),
      NULLIF(enrichment->'reviewer_context'->>'company_name', '')
    ) IS NULL
  )
  AND NOT jsonb_path_exists(COALESCE(enrichment->'evidence_spans', '[]'::jsonb), '$[*] ? ((@.time_anchor != null && @.time_anchor != "") || @.flags[*] == "deadline")')
)
""".format(
    vendor_reference=_VENDOR_REFERENCE.format(
        ambiguous_vendor=_AMBIGUOUS_VENDOR,
        commercial_context=_COMMERCIAL_CONTEXT,
        ambiguous_vendor_product_context=_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT,
    )
)
_WORKFLOW_WITHOUT_REPLACEMENT = """
(
  review_text ~* '(async|docs|documentation|notion|confluence|bundle|workspace|microsoft 365|google workspace|internal tool|homegrown|home-grown|custom tool)'
  AND lower(COALESCE(enrichment->>'replacement_mode', 'none')) = 'none'
)
"""
_STRATEGIC_CANDIDATE = f"""(
  {_MONEY_WITHOUT_PRICING_SPAN}
  OR {_COMPETITOR_WITHOUT_DISPLACEMENT}
  OR {_NAMED_COMPANY_WITHOUT_ACCOUNT_EVIDENCE}
  OR {_TIMELINE_WITHOUT_ANCHOR}
  OR {_WORKFLOW_WITHOUT_REPLACEMENT}
)"""
_TREND_DAY = "DATE(COALESCE(enriched_at, imported_at))"


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_result_payload(result_text: Any) -> dict[str, Any]:
    text = str(result_text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


async def summarize_extraction_health(
    pool,
    *,
    days: int = 30,
    top_n: int = 10,
) -> dict[str, Any]:
    current_row = await pool.fetchrow(
        f"""
        SELECT
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE}) AS enriched_rows,
          COUNT(*) FILTER (
            WHERE {_ENRICHED_SCOPE}
              AND enrichment->'evidence_spans' IS NOT NULL
              AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
              AND jsonb_array_length(enrichment->'evidence_spans') > 0
          ) AS rows_with_spans,
          COALESCE(
            SUM(
              CASE
                WHEN {_ENRICHED_SCOPE}
                  AND enrichment->'evidence_spans' IS NOT NULL
                  AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
                THEN jsonb_array_length(enrichment->'evidence_spans')
                ELSE 0
              END
            ),
            0
          ) AS span_count,
          COUNT(*) FILTER (WHERE enrichment_repair_attempts > 0) AS repair_triggered_rows,
          COUNT(*) FILTER (WHERE enrichment_repair_status = 'promoted') AS repair_promoted_rows,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_HARD_GAP}) AS hard_gap_rows,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_PHRASE_ARRAYS_NO_SPANS}) AS phrase_arrays_without_spans,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_BLANK_REPLACEMENT_MODE}) AS blank_replacement_mode,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_BLANK_OPERATING_MODEL_SHIFT}) AS blank_operating_model_shift,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_BLANK_PRODUCTIVITY_DELTA}) AS blank_productivity_delta_claim,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_BLANK_ORG_PRESSURE}) AS blank_org_pressure_type,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_MISSING_OR_EMPTY_SPANS}) AS missing_or_empty_evidence_spans,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_BLANK_EVIDENCE_MAP_HASH}) AS blank_evidence_map_hash,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_EMPTY_SALIENCE_FLAGS}) AS empty_salience_flags,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_STRATEGIC_CANDIDATE}) AS strategic_candidate_rows,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_MONEY_WITHOUT_PRICING_SPAN}) AS money_without_pricing_span,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_COMPETITOR_WITHOUT_DISPLACEMENT}) AS competitor_without_displacement_framing,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_NAMED_COMPANY_WITHOUT_ACCOUNT_EVIDENCE}) AS named_company_without_named_account_evidence,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_TIMELINE_WITHOUT_ANCHOR}) AS timeline_language_without_timing_anchor,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE} AND {_WORKFLOW_WITHOUT_REPLACEMENT}) AS workflow_language_without_replacement_mode
        FROM b2b_reviews
        """
    )

    daily_rows = await pool.fetch(
        f"""
        SELECT
          {_TREND_DAY} AS day,
          COUNT(*) AS enriched_rows,
          COUNT(*) FILTER (
            WHERE enrichment->'evidence_spans' IS NOT NULL
              AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
              AND jsonb_array_length(enrichment->'evidence_spans') > 0
          ) AS rows_with_spans,
          COALESCE(
            SUM(
              CASE
                WHEN enrichment->'evidence_spans' IS NOT NULL
                  AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
                THEN jsonb_array_length(enrichment->'evidence_spans')
                ELSE 0
              END
            ),
            0
          ) AS span_count,
          COUNT(*) FILTER (WHERE enrichment_repair_attempts > 0) AS repair_triggered_rows,
          COUNT(*) FILTER (WHERE {_HARD_GAP}) AS hard_gap_rows,
          COUNT(*) FILTER (WHERE {_PHRASE_ARRAYS_NO_SPANS}) AS phrase_arrays_without_spans,
          COUNT(*) FILTER (WHERE {_BLANK_REPLACEMENT_MODE}) AS blank_replacement_mode,
          COUNT(*) FILTER (WHERE {_BLANK_OPERATING_MODEL_SHIFT}) AS blank_operating_model_shift,
          COUNT(*) FILTER (WHERE {_MISSING_OR_EMPTY_SPANS}) AS missing_or_empty_evidence_spans,
          COUNT(*) FILTER (WHERE {_STRATEGIC_CANDIDATE}) AS strategic_candidate_rows
        FROM b2b_reviews
        WHERE {_ENRICHED_SCOPE}
          AND COALESCE(enriched_at, imported_at) >= CURRENT_DATE - ($1::int - 1)
        GROUP BY 1
        ORDER BY day DESC
        """,
        days,
    )

    top_source_rows = await pool.fetch(
        f"""
        SELECT
          source,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE}) AS enriched_rows,
          COUNT(*) FILTER (WHERE enrichment_repair_attempts > 0) AS repair_triggered_rows,
          COUNT(*) FILTER (WHERE enrichment_repair_status = 'promoted') AS repair_promoted_rows,
          COUNT(*) FILTER (
            WHERE {_ENRICHED_SCOPE}
              AND enrichment->'evidence_spans' IS NOT NULL
              AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
              AND jsonb_array_length(enrichment->'evidence_spans') > 0
          ) AS rows_with_spans,
          COALESCE(
            SUM(
              CASE
                WHEN {_ENRICHED_SCOPE}
                  AND enrichment->'evidence_spans' IS NOT NULL
                  AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
                THEN jsonb_array_length(enrichment->'evidence_spans')
                ELSE 0
              END
            ),
            0
          ) AS span_count
        FROM b2b_reviews
        WHERE {_ENRICHED_SCOPE}
          AND COALESCE(enriched_at, imported_at) >= CURRENT_DATE - ($1::int - 1)
        GROUP BY source
        HAVING COUNT(*) > 0
        ORDER BY
          COUNT(*) DESC,
          COALESCE(
            SUM(
              CASE
                WHEN {_ENRICHED_SCOPE}
                  AND enrichment->'evidence_spans' IS NOT NULL
                  AND jsonb_typeof(enrichment->'evidence_spans') = 'array'
                THEN jsonb_array_length(enrichment->'evidence_spans')
                ELSE 0
              END
            ),
            0
          ) DESC,
          source ASC
        LIMIT $2
        """,
        days,
        top_n,
    )

    top_vendor_rows = await pool.fetch(
        f"""
        SELECT
          vendor_name,
          COUNT(*) FILTER (WHERE {_HARD_GAP}) AS hard_gap_rows,
          COUNT(*) FILTER (WHERE {_PHRASE_ARRAYS_NO_SPANS}) AS phrase_arrays_without_spans,
          COUNT(*) FILTER (WHERE {_BLANK_REPLACEMENT_MODE}) AS blank_replacement_mode,
          COUNT(*) FILTER (WHERE {_BLANK_OPERATING_MODEL_SHIFT}) AS blank_operating_model_shift,
          COUNT(*) FILTER (WHERE {_MISSING_OR_EMPTY_SPANS}) AS missing_or_empty_evidence_spans,
          COUNT(*) FILTER (WHERE {_EMPTY_SALIENCE_FLAGS}) AS empty_salience_flags,
          COUNT(*) FILTER (WHERE {_STRATEGIC_CANDIDATE}) AS strategic_candidate_rows,
          COUNT(*) FILTER (WHERE {_ENRICHED_SCOPE}) AS enriched_rows
        FROM b2b_reviews
        WHERE {_ENRICHED_SCOPE}
        GROUP BY vendor_name
        HAVING COUNT(*) FILTER (WHERE {_HARD_GAP} OR {_EMPTY_SALIENCE_FLAGS} OR {_STRATEGIC_CANDIDATE}) > 0
        ORDER BY
          COUNT(*) FILTER (WHERE {_HARD_GAP}) DESC,
          COUNT(*) FILTER (WHERE {_STRATEGIC_CANDIDATE}) DESC,
          COUNT(*) FILTER (WHERE {_EMPTY_SALIENCE_FLAGS}) DESC,
          vendor_name ASC
        LIMIT $1
        """,
        top_n,
    )

    run_rows = await pool.fetch(
        """
        SELECT
          e.id AS run_id,
          t.name AS task_name,
          e.started_at,
          e.result_text
        FROM task_executions e
        JOIN scheduled_tasks t ON t.id = e.task_id
        WHERE e.started_at >= NOW() - make_interval(days => $1)
          AND e.status = 'completed'
          AND t.name = ANY($2::text[])
        ORDER BY e.started_at DESC
        LIMIT $3
        """,
        days,
        ["b2b_enrichment", "b2b_enrichment_repair"],
        top_n,
    )

    current = dict(current_row or {})
    secondary_write_hits_window = 0
    recent_runs = []
    for row in run_rows:
        payload = _parse_result_payload(row.get("result_text"))
        reviews_processed = _safe_int(payload.get("reviews_processed"))
        if reviews_processed == 0:
            if str(row.get("task_name") or "") == "b2b_enrichment":
                reviews_processed = sum(
                    _safe_int(payload.get(key))
                    for key in ("enriched", "quarantined", "failed", "no_signal")
                )
            else:
                reviews_processed = sum(
                    _safe_int(payload.get(key))
                    for key in ("promoted", "shadowed", "failed")
                )
        witness_count = _safe_int(payload.get("witness_count"))
        witness_rows = _safe_int(payload.get("witness_rows"))
        secondary_write_hits = _safe_int(payload.get("secondary_write_hits"))
        secondary_write_hits_window += secondary_write_hits
        witness_yield_rate = (
            round(witness_count / reviews_processed, 4)
            if reviews_processed > 0
            else 0.0
        )
        recent_runs.append({
            "run_id": str(row.get("run_id") or ""),
            "task_name": str(row.get("task_name") or ""),
            "started_at": row.get("started_at").isoformat() if row.get("started_at") else None,
            "reviews_processed": reviews_processed,
            "witness_rows": witness_rows,
            "witness_count": witness_count,
            "witness_yield_rate": witness_yield_rate,
            "secondary_write_hits": secondary_write_hits,
            "exact_cache_hits": _safe_int(payload.get("exact_cache_hits")),
            "generated": _safe_int(payload.get("generated")),
        })

    enriched_rows = int(current.get("enriched_rows", 0) or 0)
    span_count = int(current.get("span_count", 0) or 0)
    repair_triggered_rows = int(current.get("repair_triggered_rows", 0) or 0)
    repair_promoted_rows = int(current.get("repair_promoted_rows", 0) or 0)
    return {
        "days": days,
        "top_n": top_n,
        "current_snapshot": {
            "enriched_rows": enriched_rows,
            "rows_with_spans": int(current.get("rows_with_spans", 0) or 0),
            "span_count": span_count,
            "witness_yield_rate": round(span_count / enriched_rows, 4) if enriched_rows > 0 else 0.0,
            "repair_triggered_rows": repair_triggered_rows,
            "repair_promoted_rows": repair_promoted_rows,
            "repair_trigger_rate": round(repair_triggered_rows / enriched_rows, 4) if enriched_rows > 0 else 0.0,
            "repair_promoted_rate": round(repair_promoted_rows / enriched_rows, 4) if enriched_rows > 0 else 0.0,
            "secondary_write_hits_window": secondary_write_hits_window,
            "hard_gap_rows": int(current.get("hard_gap_rows", 0) or 0),
            "phrase_arrays_without_spans": int(current.get("phrase_arrays_without_spans", 0) or 0),
            "blank_replacement_mode": int(current.get("blank_replacement_mode", 0) or 0),
            "blank_operating_model_shift": int(current.get("blank_operating_model_shift", 0) or 0),
            "blank_productivity_delta_claim": int(current.get("blank_productivity_delta_claim", 0) or 0),
            "blank_org_pressure_type": int(current.get("blank_org_pressure_type", 0) or 0),
            "missing_or_empty_evidence_spans": int(current.get("missing_or_empty_evidence_spans", 0) or 0),
            "blank_evidence_map_hash": int(current.get("blank_evidence_map_hash", 0) or 0),
            "empty_salience_flags": int(current.get("empty_salience_flags", 0) or 0),
            "strategic_candidate_rows": int(current.get("strategic_candidate_rows", 0) or 0),
            "money_without_pricing_span": int(current.get("money_without_pricing_span", 0) or 0),
            "competitor_without_displacement_framing": int(current.get("competitor_without_displacement_framing", 0) or 0),
            "named_company_without_named_account_evidence": int(current.get("named_company_without_named_account_evidence", 0) or 0),
            "timeline_language_without_timing_anchor": int(current.get("timeline_language_without_timing_anchor", 0) or 0),
            "workflow_language_without_replacement_mode": int(current.get("workflow_language_without_replacement_mode", 0) or 0),
        },
        "daily_trend": [
            {
                **dict(row),
                "witness_yield_rate": (
                    round(_safe_int(row.get("span_count")) / _safe_int(row.get("enriched_rows")), 4)
                    if _safe_int(row.get("enriched_rows")) > 0
                    else 0.0
                ),
                "repair_trigger_rate": (
                    round(_safe_int(row.get("repair_triggered_rows")) / _safe_int(row.get("enriched_rows")), 4)
                    if _safe_int(row.get("enriched_rows")) > 0
                    else 0.0
                ),
            }
            for row in daily_rows
        ],
        "top_vendors": [dict(row) for row in top_vendor_rows],
        "top_sources": [
            {
                **dict(row),
                "witness_yield_rate": (
                    round(_safe_int(row.get("span_count")) / _safe_int(row.get("enriched_rows")), 4)
                    if _safe_int(row.get("enriched_rows")) > 0
                    else 0.0
                ),
                "repair_trigger_rate": (
                    round(_safe_int(row.get("repair_triggered_rows")) / _safe_int(row.get("enriched_rows")), 4)
                    if _safe_int(row.get("enriched_rows")) > 0
                    else 0.0
                ),
                "repair_promoted_rate": (
                    round(_safe_int(row.get("repair_promoted_rows")) / _safe_int(row.get("enriched_rows")), 4)
                    if _safe_int(row.get("enriched_rows")) > 0
                    else 0.0
                ),
            }
            for row in top_source_rows
        ],
        "recent_runs": recent_runs,
    }
