# Churn Signals Product Context Samples

## Purpose

This file provides a small, reusable sample payload set to attach alongside:

- `docs/product_context_pack.md`
- `docs/prompt_end_products.md`

These examples are intentionally compact. They are meant to help a model reason about Churn Signals product opportunities from concrete objects instead of abstract schema.

Each sample is labeled as either:

- `live-derived`
- `test-grounded`

All samples are redacted or synthetic enough to avoid exposing sensitive data directly.

---

## 1. Enriched Review With Witness Primitives

Source: `test-grounded`

Based on:

- [test_b2b_enrichment.py](/home/juan-canfield/Desktop/Atlas/tests/test_b2b_enrichment.py)

```json
{
  "review_id": "review-1",
  "vendor_name": "Slack",
  "source": "g2",
  "content_type": "review",
  "summary": "Renewal pushed us away from Slack",
  "review_text": "Slack wanted $200k/year at renewal. We became more productive using docs and async updates instead.",
  "reviewer_title": "VP Operations",
  "reviewer_company": "Hack Club",
  "rating": 2,
  "enrichment": {
    "churn_signals": {
      "intent_to_leave": true,
      "actively_evaluating": true,
      "migration_in_progress": false,
      "contract_renewal_mentioned": true
    },
    "reviewer_context": {
      "role_level": "executive",
      "decision_maker": true,
      "company_name": "Hack Club"
    },
    "budget_signals": {
      "annual_spend_estimate": 200000,
      "price_increase_mentioned": true
    },
    "specific_complaints": [
      "Slack wanted $200k/year at renewal"
    ],
    "quotable_phrases": [
      "We became more productive using docs and async updates instead"
    ],
    "pricing_phrases": [
      "$200k/year at renewal"
    ],
    "event_mentions": [
      {
        "event": "renewal",
        "timeframe": "next quarter"
      }
    ],
    "timeline": {
      "contract_end": "next quarter",
      "decision_timeline": "within_quarter"
    },
    "replacement_mode": "workflow_substitution",
    "operating_model_shift": "sync_to_async",
    "productivity_delta_claim": "more_productive",
    "org_pressure_type": "none",
    "salience_flags": [
      "explicit_dollar",
      "named_account",
      "decision_maker"
    ],
    "evidence_spans": [
      {
        "signal_type": "pricing_backlash",
        "excerpt_text": "Slack wanted $200k/year at renewal",
        "time_anchor": "next quarter",
        "reviewer_company": "Hack Club",
        "productivity_delta_claim": "more_productive"
      }
    ],
    "evidence_map_hash": "sha256:..."
  }
}
```

What this tells the model:

- Churn Signals can turn a single review into a commercially meaningful structured record.
- The strongest downstream product surfaces should use witness-ready fields, not just phrase arrays.

---

## 2. Vendor Reasoning Synthesis

Source: `test-grounded`

Based on:

- [test_b2b_synthesis_reader.py](/home/juan-canfield/Desktop/Atlas/tests/test_b2b_synthesis_reader.py)

```json
{
  "vendor_name": "Acme",
  "schema_version": "v2",
  "synthesis": {
    "reasoning_contracts": {
      "schema_version": "2.2",
      "vendor_core_reasoning": {
        "causal_narrative": {
          "primary_wedge": "price_squeeze",
          "summary": "Pricing pressure is acute.",
          "confidence": "high",
          "key_signals": [
            "Price complaints up 40%"
          ],
          "what_would_weaken_thesis": [
            {
              "condition": "Price cut announced",
              "monitorable": true
            }
          ]
        },
        "segment_playbook": {
          "segments": [],
          "confidence": "medium"
        },
        "timing_intelligence": {
          "confidence": "medium"
        }
      },
      "displacement_reasoning": {
        "competitive_reframes": {
          "confidence": "medium"
        },
        "migration_proof": {
          "confidence": "medium"
        }
      },
      "category_reasoning": {
        "market_regime": "consolidating"
      },
      "account_reasoning": {
        "total_accounts": 12
      }
    }
  }
}
```

What this tells the model:

- Churn Signals does not only store raw complaints. It stores reusable reasoning contracts.
- Product ideas can read from reasoning artifacts instead of regenerating analysis every time.

---

## 3. Witness-Backed Reasoning Context

Source: `test-grounded`

Based on:

- [test_b2b_synthesis_reader.py](/home/juan-canfield/Desktop/Atlas/tests/test_b2b_synthesis_reader.py)

```json
{
  "vendor_name": "Acme",
  "consumer_context": {
    "reference_ids": {
      "metric_ids": [
        "vault:metric:total_reviews"
      ],
      "witness_ids": [
        "witness:r1:0"
      ]
    },
    "packet_artifacts": {
      "witness_pack": [
        {
          "witness_id": "witness:r1:0",
          "_sid": "witness:r1:0",
          "excerpt_text": "Hack Club said the renewal jumped to $200k/year.",
          "reviewer_company": "Hack Club",
          "time_anchor": "Q2 renewal",
          "salience_score": 9.4
        }
      ],
      "section_packets": {
        "anchor_examples": {
          "outlier_or_named_account": [
            "witness:r1:0"
          ]
        }
      }
    },
    "witness_highlights": [
      {
        "witness_id": "witness:r1:0",
        "excerpt_text": "Hack Club said the renewal jumped to $200k/year.",
        "reviewer_company": "Hack Club"
      }
    ]
  }
}
```

What this tells the model:

- Churn Signals can support citation-aware and witness-backed UI patterns.
- Product UIs can show evidence highlights, named-account anchors, and reference IDs instead of only prose summaries.

---

## 4. Campaign Opportunity Row

Source: `test-grounded`

Based on:

- [test_read_campaign_opportunities_adapter.py](/home/juan-canfield/Desktop/Atlas/tests/test_read_campaign_opportunities_adapter.py)

```json
{
  "review_id": "uuid",
  "vendor_name": "Zendesk",
  "reviewer_company": "Acme Corp",
  "reviewer_name": "Jane Doe",
  "product_category": "Customer Support",
  "source": "g2",
  "urgency": 8.0,
  "is_dm": true,
  "role_type": "economic_buyer",
  "buying_stage": "evaluation",
  "seat_count": 150,
  "contract_end": "2026-06-30",
  "decision_timeline": "within_quarter",
  "competitors": [
    {
      "name": "Freshdesk",
      "context": "comparing pricing"
    }
  ],
  "pain_json": [
    {
      "category": "pricing",
      "severity": "primary"
    }
  ],
  "quotable_phrases": [
    "Way too expensive"
  ],
  "feature_gaps": [
    "Better automation"
  ],
  "primary_workflow": "ticket management",
  "integration_stack": [
    "Slack",
    "Jira"
  ],
  "sentiment_direction": "declining",
  "industry": "SaaS",
  "reviewer_title": "VP Operations",
  "company_size_raw": "201-500"
}
```

What this tells the model:

- Churn Signals can already generate account-targeting and campaign-facing commercial objects.
- Product ideas around lead queues, opportunity scoring, and outreach workbenches are grounded in current data.

---

## 5. Persisted Report / Delivery Artifact

Source: `test-grounded`

Based on:

- [test_b2b_dashboard_accounts_in_motion.py](/home/juan-canfield/Desktop/Atlas/tests/test_b2b_dashboard_accounts_in_motion.py)

```json
{
  "id": "2ea3fd03-7fd9-4b72-8f24-117667f723e9",
  "report_date": "2026-03-22",
  "report_type": "battle_card",
  "vendor_filter": "Zendesk",
  "category_filter": "Help Desk",
  "executive_summary": "summary",
  "status": "completed",
  "llm_model": "pipeline_deterministic",
  "data_density": {
    "status": "ok"
  },
  "intelligence_data": {
    "quality_status": null,
    "battle_card_quality": null
  },
  "created_at": "2026-03-22T18:00:00"
}
```

What this tells the model:

- Churn Signals has persisted, deliverable artifacts, not just transient prompts and responses.
- Product ideas can include subscription/report-delivery and report-library patterns.

---

## 6. Burn Dashboard Row

Source: `live-derived`

Based on:

- `GET /api/v1/admin/costs/burn-dashboard?days=1&top_n=5`

```json
{
  "summary": {
    "tracked_cost_usd": 2.513777,
    "model_call_count": 1321,
    "recent_runs": 1386,
    "rows_processed": 114,
    "rows_reprocessed": 13,
    "reprocess_pct": 0.114
  },
  "reasoning_budget_pressure": {
    "cross_vendor_rejections": 26,
    "last_rejection_at": "2026-04-03T21:52:42.475046+00:00",
    "max_cross_vendor_estimated_input_tokens": 24662,
    "max_cross_vendor_cap": 12000
  },
  "representative_row": {
    "task_name": "news_intake",
    "recent_runs": 52,
    "last_run_at": "2026-04-04T05:10:40.970190+00:00",
    "last_status": "completed",
    "model_call_count": 0,
    "total_cost_usd": 0.0,
    "top_trigger_reason": "News intake completed -- articles stored"
  }
}
```

What this tells the model:

- Churn Signals already exposes operator-grade unit economics and burn visibility.
- Internal products can be built around spend, retries, reprocessing, trigger discipline, and budget pressure.

---

## How To Use These Samples

When prompting a model for product ideas, do not dump every schema in the repo.

Attach:

1. `docs/product_context_pack.md`
2. `docs/product_context_samples.md`
3. Optionally one or two newer live payloads if the product question is very specific

That is usually enough context for the model to:

- understand the Churn Signals pipeline
- understand what artifacts already exist
- understand where UI patterns can read from existing objects
- avoid inventing unsupported products

---

## Product-Ideation Reminder

The best Churn Signals product ideas usually come from recombining these existing objects into:

- watchlists
- feeds
- review queues
- report libraries
- evidence-backed dashboards
- campaign workbenches
- API subscriptions

They usually do not come from inventing entirely new always-on model generation paths.
