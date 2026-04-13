import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PipelineReview from './PipelineReview'

const api = vi.hoisted(() => ({
  fetchVisibilitySummary: vi.fn(),
  fetchVisibilityQueue: vi.fn(),
  fetchVisibilityEvents: vi.fn(),
  fetchArtifactAttempts: vi.fn(),
  fetchEnrichmentQuarantines: vi.fn(),
  fetchExtractionHealth: vi.fn(),
  fetchSynthesisValidationResults: vi.fn(),
  resolveVisibilityReview: vi.fn(),
  fetchAdminCostSummary: vi.fn(),
  fetchAdminCostByOperation: vi.fn(),
  fetchAdminCostByVendor: vi.fn(),
  fetchAdminCostB2bEfficiency: vi.fn(),
  fetchAdminCostBurnDashboard: vi.fn(),
  fetchAdminCostGenericReasoning: vi.fn(),
  fetchAdminCostReconciliation: vi.fn(),
  fetchAdminCostRecent: vi.fn(),
  fetchAdminCostCacheHealth: vi.fn(),
  fetchAdminCostReasoningActivity: vi.fn(),
  fetchAdminCostRun: vi.fn(),
  fetchAdminTaskHealth: vi.fn(),
  fetchCompanySignalReviewImpactSummary: vi.fn(),
  fetchCompanySignalCandidateGroupSummary: vi.fn(),
  fetchCompanySignalCandidateGroups: vi.fn(),
  approveCompanySignalCandidateGroup: vi.fn(),
  suppressCompanySignalCandidateGroup: vi.fn(),
  approveCompanySignalCandidateGroups: vi.fn(),
  suppressCompanySignalCandidateGroups: vi.fn(),
  fetchWatchlistDeliveryOps: vi.fn(),
  fetchWatchlistDeliveryViewDetail: vi.fn(),
  runWatchlistDeliveryForView: vi.fn(),
  disableWatchlistDeliveryForView: vi.fn(),
  runAutonomousTask: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('PipelineReview watchlist delivery ops', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchVisibilitySummary.mockResolvedValue({
      open_actionable: 2,
      failures_period: 1,
      quarantines_period: 0,
      rejections_period: 0,
      recovered_validation_retries_period: 0,
    })
    api.fetchVisibilityQueue.mockResolvedValue({ items: [], limit: 100, offset: 0 })
    api.fetchVisibilityEvents.mockResolvedValue({ events: [], limit: 100, offset: 0 })
    api.fetchArtifactAttempts.mockResolvedValue({ attempts: [], limit: 100, offset: 0 })
    api.fetchEnrichmentQuarantines.mockResolvedValue({ quarantines: [], unreleased_only: true, limit: 100 })
    api.fetchExtractionHealth.mockResolvedValue({
      period_days: 30,
      current_snapshot: null,
      daily_trend: [],
      top_vendors: [],
      top_sources: [],
      recent_runs: [],
    })
    api.fetchSynthesisValidationResults.mockResolvedValue({ results: [], total: 0, limit: 100, offset: 0 })
    api.resolveVisibilityReview.mockResolvedValue({ status: 'ok' })
    api.fetchAdminCostSummary.mockResolvedValue({
      period_days: 30,
      total_cost_usd: 1.23,
      total_calls: 4,
      total_billable_input_tokens: 2200,
      total_cached_tokens: 300,
      total_cache_write_tokens: 100,
      total_output_tokens: 450,
      total_tokens: 2650,
      total_input_tokens: 2500,
      cache_hit_calls: 1,
      cache_write_calls: 1,
      avg_duration_ms: 420,
      avg_tokens_per_second: 25,
      today_cost_usd: 0.12,
      today_calls: 1,
    })
    api.fetchAdminCostByOperation.mockResolvedValue({ period_days: 30, operations: [] })
    api.fetchAdminCostByVendor.mockResolvedValue({ period_days: 30, vendors: [] })
    api.fetchCompanySignalReviewImpactSummary.mockResolvedValue({
      totals: {
        total_actions: 0,
        total_batches: 0,
        distinct_vendors: 0,
        approvals: 0,
        suppressions: 0,
        company_signal_creations: 0,
        company_signal_updates: 0,
        company_signal_deletions: 0,
        company_signal_noops: 0,
        rebuild_requests: 0,
        rebuild_triggered: 0,
        rebuild_blocked: 0,
        rebuild_persisted_runs: 0,
        rebuild_persisted_reports: 0,
        rebuild_total_accounts: 0,
        company_signal_effect_rate: 0,
        company_signal_creation_rate: 0,
        rebuild_trigger_rate: 0,
        avg_rebuild_reports_per_triggered: 0,
        avg_rebuild_accounts_per_triggered: 0,
      },
      scopes: [],
      unlock_paths: [],
      priority_bands: [],
      priority_reasons: [],
      top_vendors: [],
      top_vendor_reasons: [],
      rebuild_reasons: [],
      daily_trends: [],
      trend_comparison: {
        comparison_window_days: 7,
        anchor_day: null,
        recent_start_day: null,
        recent_end_day: null,
        recent_days_present: 0,
        prior_start_day: null,
        prior_end_day: null,
        prior_days_present: 0,
        recent: {
          action_count: 0,
          approvals: 0,
          suppressions: 0,
          company_signal_creations: 0,
          company_signal_updates: 0,
          company_signal_deletions: 0,
          company_signal_noops: 0,
          rebuild_requests: 0,
          rebuild_triggered: 0,
          rebuild_blocked: 0,
          rebuild_persisted_runs: 0,
          rebuild_persisted_reports: 0,
          rebuild_total_accounts: 0,
          company_signal_effect_rate: 0,
          company_signal_creation_rate: 0,
          rebuild_trigger_rate: 0,
          rebuild_block_rate: 0,
          avg_rebuild_reports_per_triggered: 0,
          avg_rebuild_accounts_per_triggered: 0,
        },
        prior: {
          action_count: 0,
          approvals: 0,
          suppressions: 0,
          company_signal_creations: 0,
          company_signal_updates: 0,
          company_signal_deletions: 0,
          company_signal_noops: 0,
          rebuild_requests: 0,
          rebuild_triggered: 0,
          rebuild_blocked: 0,
          rebuild_persisted_runs: 0,
          rebuild_persisted_reports: 0,
          rebuild_total_accounts: 0,
          company_signal_effect_rate: 0,
          company_signal_creation_rate: 0,
          rebuild_trigger_rate: 0,
          rebuild_block_rate: 0,
          avg_rebuild_reports_per_triggered: 0,
          avg_rebuild_accounts_per_triggered: 0,
        },
        deltas: {
          action_count: 0,
          approvals: 0,
          suppressions: 0,
          company_signal_creations: 0,
          company_signal_updates: 0,
          company_signal_deletions: 0,
          company_signal_noops: 0,
          rebuild_requests: 0,
          rebuild_triggered: 0,
          rebuild_blocked: 0,
          rebuild_persisted_runs: 0,
          rebuild_persisted_reports: 0,
          rebuild_total_accounts: 0,
          company_signal_effect_rate: 0,
          company_signal_creation_rate: 0,
          rebuild_trigger_rate: 0,
          rebuild_block_rate: 0,
          avg_rebuild_reports_per_triggered: 0,
          avg_rebuild_accounts_per_triggered: 0,
        },
      },
      trend_focus: {},
      trend_alerts: [
        {
          status: 'watch',
          focus: 'rebuild_blocks_up',
          metric: 'rebuild_blocked',
          direction: 'up',
          delta: 1,
          rationale: 'More requested rebuilds are blocking in the recent window than in the prior window.',
          queue_filters: {
            candidate_bucket: 'analyst_review',
            review_status: 'pending',
          },
          queue_snapshot: {
            pending_groups: 7,
            actionable_pending_groups: 3,
            blocked_pending_groups: 4,
            overdue_pending_groups: 2,
            oldest_pending_age_days: 6,
          },
        },
      ],
      trend_recommendation: { status: 'no_data' },
      trend_recommendation_filters: {},
      trend_recommendation_queue_filters: {},
      trend_recommendation_queue_snapshot: null,
      trend_queue_rankings: [
        {
          primary_driver: {
            label: 'rebuild_blocks_up',
            rationale: 'Blocked rebuilds are the top queue driver right now.',
          },
          queue_filters: {
            candidate_bucket: 'analyst_review',
            review_status: 'pending',
          },
          queue_snapshot: {
            pending_groups: 7,
            actionable_pending_groups: 3,
            blocked_pending_groups: 4,
            overdue_pending_groups: 2,
            oldest_pending_age_days: 6,
          },
          actionable_pending_groups: 3,
          blocked_pending_groups: 4,
          overdue_pending_groups: 2,
          pending_groups: 7,
          oldest_pending_age_days: 6,
        },
      ],
      trend_queue_focus: {
        primary_driver: {
          label: 'rebuild_blocks_up',
        },
      },
      trend_queue_recommendation: { status: 'no_data' },
      operator_focus: { status: 'no_data' },
      review_scope: null,
      canonical_gap_reason: null,
      rebuild_outcome: null,
      rebuild_reason: null,
      review_action: null,
      company_signal_action: null,
      review_priority_band: null,
      review_priority_reason: null,
      review_unlock_path: null,
      review_unlock_reason: null,
      candidate_source: null,
    })
    api.fetchAdminCostB2bEfficiency.mockResolvedValue({
      period_days: 30,
      top_n: 5,
      run_limit: 5,
      summary: {
        measured_runs: 3,
        tracked_cost_usd: 0.37,
        tracked_witness_count: 35,
        cost_per_witness_usd: 0.010571,
      },
      token_summary: {
        total_billable_input_tokens: 4750,
        total_output_tokens: 1050,
        by_pass: [
          { key: 'extraction', label: 'Extraction', calls: 4, cost_usd: 0.21, billable_input_tokens: 2100, output_tokens: 390 },
          { key: 'repair', label: 'Repair', calls: 1, cost_usd: 0.05, billable_input_tokens: 500, output_tokens: 100 },
          { key: 'reasoning', label: 'Reasoning', calls: 1, cost_usd: 0.2, billable_input_tokens: 1200, output_tokens: 300 },
          { key: 'battle_card_overlay', label: 'Battle Cards', calls: 1, cost_usd: 0.07, billable_input_tokens: 950, output_tokens: 260 },
        ],
        enrichment_tiers: [
          { key: 'tier1', label: 'Tier 1', calls: 2, cost_usd: 0.18, billable_input_tokens: 1050, output_tokens: 230 },
          { key: 'tier2', label: 'Tier 2', calls: 2, cost_usd: 0.03, billable_input_tokens: 1050, output_tokens: 160 },
        ],
      },
      vendor_passes: [],
      source_efficiency: [],
      recent_runs: [
        {
          run_id: 'run-enrich-1',
          task_name: 'b2b_enrichment',
          started_at: '2026-04-07T18:00:00Z',
          total_cost_usd: 0.12,
          calls: 2,
          total_billable_input_tokens: 1300,
          total_output_tokens: 210,
          reviews_processed: 10,
          witness_rows: 8,
          witness_count: 15,
          witness_yield_rate: 1.5,
          cost_per_witness_usd: 0.008,
          secondary_write_hits: 0,
          strict_discussion_candidates_kept: 4,
          strict_discussion_candidates_dropped: 0,
          low_signal_discussion_skipped: 0,
          exact_cache_hits: 0,
          generated: 8,
          extraction_cost_usd: 0.12,
          extraction_billable_input_tokens: 1300,
          extraction_output_tokens: 210,
          repair_cost_usd: 0,
          repair_billable_input_tokens: 0,
          repair_output_tokens: 0,
          reasoning_cost_usd: 0,
          reasoning_billable_input_tokens: 0,
          reasoning_output_tokens: 0,
          battle_card_overlay_cost_usd: 0,
          battle_card_overlay_calls: 0,
          battle_card_overlay_billable_input_tokens: 0,
          battle_card_overlay_output_tokens: 0,
          enrichment_tier1_cost_usd: 0.1,
          enrichment_tier1_calls: 1,
          enrichment_tier1_billable_input_tokens: 700,
          enrichment_tier1_output_tokens: 120,
          enrichment_tier2_cost_usd: 0.02,
          enrichment_tier2_calls: 1,
          enrichment_tier2_billable_input_tokens: 600,
          enrichment_tier2_output_tokens: 90,
          battle_card_cache_hits: 0,
          battle_card_llm_updated: 0,
          battle_card_llm_failures: 0,
        },
        {
          run_id: 'run-noop-1',
          task_name: 'b2b_enrichment_repair',
          started_at: '2026-04-07T17:30:00Z',
          total_cost_usd: 0,
          calls: 0,
          total_billable_input_tokens: 0,
          total_output_tokens: 0,
          reviews_processed: 0,
          witness_rows: 0,
          witness_count: 0,
          witness_yield_rate: null,
          cost_per_witness_usd: null,
          secondary_write_hits: 0,
          strict_discussion_candidates_kept: 0,
          strict_discussion_candidates_dropped: 0,
          low_signal_discussion_skipped: 0,
          exact_cache_hits: 0,
          generated: 0,
          extraction_cost_usd: 0,
          extraction_billable_input_tokens: 0,
          extraction_output_tokens: 0,
          repair_cost_usd: 0,
          repair_billable_input_tokens: 0,
          repair_output_tokens: 0,
          reasoning_cost_usd: 0,
          reasoning_billable_input_tokens: 0,
          reasoning_output_tokens: 0,
          battle_card_overlay_cost_usd: 0,
          battle_card_overlay_calls: 0,
          battle_card_overlay_billable_input_tokens: 0,
          battle_card_overlay_output_tokens: 0,
          enrichment_tier1_cost_usd: 0,
          enrichment_tier1_calls: 0,
          enrichment_tier1_billable_input_tokens: 0,
          enrichment_tier1_output_tokens: 0,
          enrichment_tier2_cost_usd: 0,
          enrichment_tier2_calls: 0,
          enrichment_tier2_billable_input_tokens: 0,
          enrichment_tier2_output_tokens: 0,
          battle_card_cache_hits: 0,
          battle_card_llm_updated: 0,
          battle_card_llm_failures: 0,
        },
      ],
    })
    api.fetchAdminCostBurnDashboard.mockResolvedValue({
      period_days: 30,
      top_n: 5,
      summary: { tracked_cost_usd: 0, model_call_count: 0, recent_runs: 0, rows_processed: null, rows_reprocessed: null, reprocess_pct: null },
      reasoning_budget_pressure: {
        vendor_rejections: 0,
        cross_vendor_rejections: 0,
        last_rejection_at: null,
        max_vendor_estimated_input_tokens: null,
        max_vendor_cap: null,
        max_cross_vendor_estimated_input_tokens: null,
        max_cross_vendor_cap: null,
        rows: [],
      },
      rows: [],
    })
    api.fetchAdminCostGenericReasoning.mockResolvedValue({
      period_days: 30,
      top_n: 5,
      summary: { total_cost_usd: 0, total_calls: 0, total_billable_input_tokens: 0, total_output_tokens: 0, top_source_name: null, top_event_type: null },
      by_source: [],
      by_event_type: [],
      top_source_events: [],
      top_entities: [],
    })
    api.fetchAdminCostReconciliation.mockResolvedValue({
      period_days: 30,
      status: 'ok',
      message: null,
      summary: { tracked_cost_usd: 0, provider_cost_usd: 0, delta_cost_usd: 0, delta_pct: 0 },
      daily_rows: [],
    })
    api.fetchAdminCostRecent.mockResolvedValue({ period_days: 30, limit: 25, recent: [] })
    api.fetchAdminCostCacheHealth.mockResolvedValue({
      period_days: 30,
      exact_cache: { enabled: true, total_rows: 0, total_hits: 0, writes_in_window: 0, rows_hit_in_window: 0, stages: [] },
      provider_prompt_cache: { total_calls: 0, cache_hit_calls: 0, cache_write_calls: 0, cached_tokens: 0, cache_write_tokens: 0, billable_input_tokens: 0, top_spans: [] },
      anthropic_batching: {
        enabled: false,
        total_jobs: 0,
        submitted_jobs: 0,
        total_items: 0,
        submitted_items: 0,
        cache_prefiltered_items: 0,
        fallback_single_call_items: 0,
        completed_items: 0,
        failed_items: 0,
        estimated_sequential_cost_usd: 0,
        estimated_batch_cost_usd: 0,
        estimated_savings_usd: 0,
        stages: [],
        stale_job_threshold_minutes: 30,
        stale_jobs_count: 0,
        stale_claims_count: 0,
        stale_jobs: [],
        stale_claims: [],
      },
      semantic_cache: { active_entries: 0, invalidated_entries: 0, recent_validations: 0, pattern_classes: [] },
      task_reuse: { tasks: [] },
      evidence_hash_reuse: {
        vendor_packet_rows: 0,
        vendor_packet_writes_in_window: 0,
        unique_vendors: 0,
        unique_hashes: 0,
        cross_vendor_rows: 0,
        cross_vendor_cached_rows: 0,
        cross_vendor_cached_rows_in_window: 0,
      },
    })
    api.fetchAdminCostReasoningActivity.mockResolvedValue({
      period_days: 30,
      phases: [],
      summary: { total_cost_usd: 0, total_tokens: 0, total_calls: 0 },
    })
    api.fetchAdminCostRun.mockResolvedValue(null)
    api.fetchAdminTaskHealth.mockResolvedValue({ tasks: [] })
    api.fetchCompanySignalCandidateGroupSummary.mockResolvedValue({
      totals: {
        total_groups: 12,
        total_reviews: 18,
        canonical_ready_reviews: 1,
        pending_groups: 12,
        actionable_pending_groups: 5,
        actionable_pending_reviews: 8,
        blocked_pending_groups: 7,
        blocked_pending_reviews: 10,
        near_threshold_blocked_groups: 1,
        near_threshold_blocked_reviews: 2,
        approved_groups: 0,
        suppressed_groups: 0,
        canonical_ready_groups: 1,
        analyst_review_groups: 11,
        pending_canonical_ready_groups: 1,
        pending_analyst_review_groups: 11,
        decision_maker_groups: 3,
        signal_evidence_groups: 4,
        avg_pending_age_days: 2.4,
        oldest_pending_age_days: 4.1,
        overdue_pending_groups: 1,
        overdue_pending_reviews: 2,
      },
      gap_reasons: [
        { gap_reason: 'low_confidence_low_trust_source', group_count: 7, review_count: 10 },
        { gap_reason: 'below_high_intent_threshold', group_count: 4, review_count: 6 },
      ],
      top_vendors: [
        { vendor_name: 'Shopify', group_count: 3, review_count: 6, pending_groups: 3, canonical_ready_groups: 0 },
      ],
      actionable_top_vendors: [
        {
          vendor_name: 'Shopify',
          actionable_group_count: 2,
          actionable_review_count: 4,
          promote_now_group_count: 0,
          high_group_count: 1,
          medium_group_count: 1,
          actionable_signal_evidence_groups: 1,
          actionable_decision_maker_groups: 1,
        },
      ],
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      review_priority_band: null,
      review_priority_reason: null,
      source_name: null,
    })
    api.fetchCompanySignalCandidateGroups.mockResolvedValue({
      groups: [
        {
          group_id: 'group-1',
          company: 'sunny side studio',
          display_company: 'Sunny Side Studio',
          vendor: 'Shopify',
          category: 'E-commerce',
          review_count: 1,
          distinct_source_count: 1,
          decision_maker_count: 1,
          signal_evidence_count: 1,
          canonical_ready_review_count: 0,
          avg_urgency: 8.5,
          max_urgency: 8.5,
          avg_confidence_score: 0.21,
          max_confidence_score: 0.21,
          corroborated_confidence_score: 0.31,
          confidence_tier: 'medium',
          source_distribution: { reddit: 1 },
          gap_reason_distribution: { low_confidence_low_trust_source: 1 },
          sample_review_ids: ['review-1'],
          representative_review_id: 'review-1',
          representative_source: 'reddit',
          representative_pain_category: 'pricing',
          representative_buyer_role: 'unknown',
          representative_decision_maker: true,
          representative_seat_count: null,
          representative_contract_end: null,
          representative_buying_stage: 'evaluation',
          representative_confidence_score: 0.21,
          representative_urgency_score: 8.5,
          canonical_gap_reason: 'low_confidence_low_trust_source',
          candidate_bucket: 'analyst_review',
          review_priority_band: 'high',
          review_priority_reason: 'has_signal_evidence_and_decision_maker',
          review_status: 'pending',
          review_status_updated_at: null,
          reviewed_by: null,
          review_notes: null,
          materialization_run_id: 'run-1',
          first_seen_at: '2026-04-11T06:51:24Z',
          last_seen_at: '2026-04-13T16:56:10Z',
          supporting_reviews: [
            {
              review_id: 'review-1',
              source: 'reddit',
              summary: 'Shopify or Squarespace for a surf art project?',
              review_excerpt: 'A surf art project deciding between Shopify and Squarespace.',
              source_url: 'https://example.com/review-1',
              reviewed_at: '2026-04-08T06:51:12Z',
              quote_excerpt: 'I know Shopify is better for selling and Squarespace for portfolios',
            },
          ],
        },
      ],
      count: 1,
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      review_priority_band: null,
      review_priority_reason: null,
      source_name: null,
    })
    api.approveCompanySignalCandidateGroup.mockResolvedValue({ review_status: 'approved' })
    api.suppressCompanySignalCandidateGroup.mockResolvedValue({ review_status: 'suppressed' })
    api.approveCompanySignalCandidateGroups.mockResolvedValue({ count: 1, groups: [] })
    api.suppressCompanySignalCandidateGroups.mockResolvedValue({ count: 1, groups: [] })
    api.fetchWatchlistDeliveryOps.mockResolvedValue({
      period_days: 30,
      summary: {
        enabled_views: 3,
        due_views: 1,
        open_event_count: 17,
        recent_sent: 2,
        recent_partial: 0,
        recent_failed: 1,
        recent_no_events: 4,
        recent_skipped: 2,
      },
      task: {
        id: 'task-1',
        name: 'b2b_watchlist_alert_delivery',
        task_type: 'builtin',
        schedule_type: 'interval',
        cron_expression: null,
        interval_seconds: 3600,
        enabled: true,
        last_run_at: '2026-04-07T18:00:00Z',
        next_run_at: '2026-04-07T19:00:00Z',
        last_status: 'completed',
        last_duration_ms: 812,
        last_error: null,
        recent_failure_rate: 0.2,
        recent_runs: 5,
      },
      views: [
        {
          view_id: 'view-1',
          view_name: 'Daily CRM Watch',
          account_id: 'acct-1',
          account_name: 'Effingham Office Maids',
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-07T19:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:00:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          last_alert_delivery_suppressed_preview_summary: {
            count: 1,
            reasons: { preview_low_confidence: 1 },
            reason_details: {
              preview_low_confidence: { short_summary: 'confidence >= 0.65 required' },
            },
          },
          preview_account_alert_policy: {
            applies_to_preview_only: true,
            enabled: true,
            enabled_source: 'view',
            min_confidence: 0.65,
            min_confidence_source: 'view',
            require_budget_authority: false,
            require_budget_authority_source: 'view',
            override_min_confidence: 0.65,
            override_require_budget_authority: false,
          },
          open_event_count: 17,
          due_now: true,
        },
      ],
      deliveries: [
        {
          id: 'log-1',
          watchlist_view_id: 'view-1',
          view_name: 'Daily CRM Watch',
          account_id: 'acct-1',
          account_name: 'Effingham Office Maids',
          status: 'sent',
          summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          error: null,
          event_count: 17,
          recipient_count: 1,
          delivered_at: '2026-04-07T18:00:00Z',
          created_at: '2026-04-07T18:00:00Z',
          scheduled_for: '2026-04-07T18:00:00Z',
          delivery_mode: 'scheduled',
          suppressed_preview_summary: {
            count: 1,
            reasons: { preview_low_confidence: 1 },
            reason_details: {
              preview_low_confidence: { short_summary: 'confidence >= 0.65 required' },
            },
          },
        },
      ],
    })
    api.fetchWatchlistDeliveryViewDetail.mockResolvedValue({
      view: {
        view_id: 'view-1',
        view_name: 'Daily CRM Watch',
        account_id: 'acct-1',
        account_name: 'Effingham Office Maids',
        alert_delivery_frequency: 'daily',
        next_alert_delivery_at: '2026-04-07T19:00:00Z',
        last_alert_delivery_at: '2026-04-07T18:00:00Z',
        last_alert_delivery_status: 'sent',
        last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
        last_alert_delivery_suppressed_preview_summary: {
          count: 1,
          reasons: { preview_low_confidence: 1 },
          reason_details: {
            preview_low_confidence: { short_summary: 'confidence >= 0.65 required' },
          },
        },
        preview_account_alert_policy: {
          applies_to_preview_only: true,
          enabled: true,
          enabled_source: 'view',
          min_confidence: 0.65,
          min_confidence_source: 'view',
          require_budget_authority: false,
          require_budget_authority_source: 'view',
          override_min_confidence: 0.65,
          override_require_budget_authority: false,
        },
        open_event_count: 17,
        due_now: true,
        vendor_name: 'Salesforce',
        category: 'CRM',
        source: 'g2',
        min_urgency: 7,
        include_stale: false,
        named_accounts_only: true,
        changed_wedges_only: false,
        vendor_alert_threshold: 8,
        account_alert_threshold: 7,
        stale_days_threshold: 3,
        alert_email_enabled: true,
        created_at: '2026-04-07T17:00:00Z',
        updated_at: '2026-04-07T18:00:00Z',
      },
      event_status: 'all',
      events: [
        {
          id: 'event-1',
          watchlist_view_id: 'view-1',
          event_type: 'vendor_alert',
          threshold_field: 'vendor_alert_threshold',
          entity_type: 'vendor',
          entity_key: 'Salesforce',
          vendor_name: 'Salesforce',
          company_name: null,
          category: 'CRM',
          source: 'g2',
          threshold_value: 8,
          summary: 'Salesforce urgency breached the saved view threshold.',
          payload: {},
          status: 'open',
          first_seen_at: '2026-04-07T18:00:00Z',
          last_seen_at: '2026-04-07T18:00:00Z',
          resolved_at: null,
          created_at: '2026-04-07T18:00:00Z',
          updated_at: '2026-04-07T18:00:00Z',
        },
      ],
      event_count: 1,
        deliveries: [
          {
            id: 'log-1',
            watchlist_view_id: 'view-1',
          view_name: 'Daily CRM Watch',
          account_id: 'acct-1',
          account_name: 'Effingham Office Maids',
          status: 'sent',
          summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          error: null,
          event_count: 17,
          recipient_count: 1,
            delivered_at: '2026-04-07T18:00:00Z',
            created_at: '2026-04-07T18:00:00Z',
            scheduled_for: '2026-04-07T18:00:00Z',
            delivery_mode: 'scheduled',
            suppressed_preview_summary: {
              count: 1,
              reasons: { preview_low_confidence: 1 },
              reason_details: {
                preview_low_confidence: { short_summary: 'confidence >= 0.65 required' },
              },
            },
          },
        ],
      delivery_count: 1,
    })
    api.runWatchlistDeliveryForView.mockResolvedValue({
      watchlist_view_id: 'view-1',
      watchlist_view_name: 'Daily CRM Watch',
      status: 'sent',
      recipient_emails: ['ops@example.com'],
      event_count: 1,
      message_ids: ['msg-1'],
      summary: 'Delivered watchlist alert email to 1 of 1 recipient',
      error: null,
    })
    api.disableWatchlistDeliveryForView.mockResolvedValue({
      disabled: true,
      view: {
        id: 'view-1',
        alert_email_enabled: false,
        next_alert_delivery_at: null,
        last_alert_delivery_status: 'sent',
        last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
      },
    })
    api.runAutonomousTask.mockResolvedValue({ status: 'started', message: 'Triggered delivery run' })
  })

  it('renders watchlist delivery ops and triggers the scheduler task', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Watchlist Alert Delivery')).toBeInTheDocument()
    expect(screen.getAllByText('Daily CRM Watch')).toHaveLength(2)
    expect(screen.getAllByText('Delivered watchlist alert email to 1 of 1 recipient')).toHaveLength(2)
    expect(screen.getByText('Preview on | conf >= 0.65 | budget optional')).toBeInTheDocument()
    expect(screen.getAllByText('1 blocked preview alert: confidence >= 0.65 required')).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: 'Run Delivery Now' }))

    await waitFor(() => {
      expect(api.runAutonomousTask).toHaveBeenCalledWith('b2b_watchlist_alert_delivery')
    })
    expect(await screen.findByText('Triggered delivery run')).toBeInTheDocument()
  })

  it('renders the company signal queue and approves a candidate group', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Company Signal Queue')).toBeInTheDocument()
    expect(screen.getByText('Operator Focus')).toBeInTheDocument()
    expect(screen.getByText('No review-action history yet. Once analysts approve or suppress groups, impact and rebuild outcomes will show here.')).toBeInTheDocument()
    expect(screen.getByText('Pending Candidate Groups')).toBeInTheDocument()
    expect(api.fetchCompanySignalReviewImpactSummary).toHaveBeenCalledWith({ window_days: 30, top_n: 8 })
    expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenCalledWith({
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      source_name: undefined,
      canonical_gap_reason: undefined,
      review_priority_band: undefined,
      decision_makers_only: undefined,
      top_n: 8,
    })
    expect(api.fetchCompanySignalCandidateGroups).toHaveBeenCalledWith({
      candidate_bucket: 'analyst_review',
      review_status: 'pending',
      source_name: undefined,
      canonical_gap_reason: undefined,
      review_priority_band: undefined,
      decision_makers_only: undefined,
      limit: 25,
    })
    expect(screen.getByText('Sunny Side Studio')).toBeInTheDocument()
    expect(screen.getAllByText('low confidence low trust source').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Shopify').length).toBeGreaterThan(0)

    await user.click(screen.getByText('Sunny Side Studio'))
    expect(await screen.findByText('Supporting Reviews')).toBeInTheDocument()
    expect(screen.getByText('All reviews currently backing this candidate group.')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: 'Source' }).length).toBeGreaterThan(0)
    expect(screen.getByText('Review State')).toBeInTheDocument()
    expect(screen.getByText('Notes: --')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Company signal review notes'), 'Promote based on clear decision-maker evidence')
    await user.click(screen.getByLabelText('Trigger rebuild after review'))

    await user.click(screen.getByRole('button', { name: 'Approve' }))

    await waitFor(() => {
      expect(api.approveCompanySignalCandidateGroup).toHaveBeenCalledWith('group-1', {
        trigger_rebuild: false,
        notes: 'Promote based on clear decision-maker evidence',
      })
    })
    expect(await screen.findByText('approved')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Source'), 'reddit')
    await user.selectOptions(screen.getByLabelText('Gap'), 'low_confidence_low_trust_source')
    await user.click(screen.getByLabelText('Decision makers only'))

    await waitFor(() => {
      expect(api.fetchCompanySignalReviewImpactSummary).toHaveBeenLastCalledWith({
        review_action: undefined,
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        candidate_source: 'reddit',
        window_days: 30,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        decision_makers_only: true,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        decision_makers_only: true,
        limit: 25,
      })
    })
    expect(screen.getByText('reddit source')).toBeInTheDocument()
    expect(screen.getAllByText('Decision makers only').length).toBeGreaterThan(0)

    await user.click(screen.getByRole('checkbox', { name: 'Select candidate group group-1' }))
    expect(screen.getByText('1 candidate group selected')).toBeInTheDocument()
    expect(
      screen.getAllByRole('link', { name: 'Evidence' }).some((node) =>
        node.getAttribute('href') === '/evidence?vendor=Shopify&tab=witnesses&back_to=%2Fpipeline-review',
      ),
    ).toBe(true)

    await user.click(screen.getByRole('button', { name: 'Approve Selected' }))

    await waitFor(() => {
      expect(api.approveCompanySignalCandidateGroups).toHaveBeenCalledWith({
        group_ids: ['group-1'],
        trigger_rebuild: false,
        notes: 'Promote based on clear decision-maker evidence',
      })
    })
    expect(await screen.findByText('Approved 1 candidate group')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Review Status'), 'approved')

    await waitFor(() => {
      expect(api.fetchCompanySignalReviewImpactSummary).toHaveBeenLastCalledWith({
        review_action: 'approved',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        candidate_source: 'reddit',
        window_days: 30,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'approved',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        decision_makers_only: true,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'approved',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: undefined,
        decision_makers_only: true,
        limit: 25,
      })
    })
    expect(screen.getByText('approved groups')).toBeInTheDocument()
    expect(screen.getByText('Review actions are disabled in audit mode.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Approve Selected' })).not.toBeInTheDocument()
  })

  it('renders company signal review activity in audit mode when impact history exists', async () => {
    const user = userEvent.setup()

    api.fetchCompanySignalReviewImpactSummary.mockResolvedValue({
      totals: {
        total_actions: 6,
        total_batches: 2,
        distinct_vendors: 2,
        approvals: 4,
        suppressions: 2,
        company_signal_creations: 3,
        company_signal_updates: 1,
        company_signal_deletions: 1,
        company_signal_noops: 1,
        rebuild_requests: 5,
        rebuild_triggered: 4,
        rebuild_blocked: 1,
        rebuild_persisted_runs: 3,
        rebuild_persisted_reports: 7,
        rebuild_total_accounts: 18,
        company_signal_effect_rate: 0.67,
        company_signal_creation_rate: 0.5,
        rebuild_trigger_rate: 0.8,
        avg_rebuild_reports_per_triggered: 1.75,
        avg_rebuild_accounts_per_triggered: 4.5,
      },
      scopes: [],
      unlock_paths: [],
      priority_bands: [],
      priority_reasons: [],
      top_vendors: [
        { vendor_name: 'Shopify', action_count: 4 },
        { vendor_name: 'Salesforce', action_count: 2 },
      ],
      top_vendor_reasons: [],
      rebuild_reasons: [
        { rebuild_reason: 'accounts_in_motion_refresh', count: 4 },
        { rebuild_reason: 'rebuild_disabled', count: 1 },
      ],
      daily_trends: [],
      trend_comparison: {
        comparison_window_days: 7,
        anchor_day: '2026-04-13',
        recent_start_day: '2026-04-07',
        recent_end_day: '2026-04-13',
        recent_days_present: 7,
        prior_start_day: '2026-03-31',
        prior_end_day: '2026-04-06',
        prior_days_present: 7,
        recent: {
          action_count: 6,
          approvals: 4,
          suppressions: 2,
          company_signal_creations: 3,
          company_signal_updates: 1,
          company_signal_deletions: 1,
          company_signal_noops: 1,
          rebuild_requests: 5,
          rebuild_triggered: 4,
          rebuild_blocked: 1,
          rebuild_persisted_runs: 3,
          rebuild_persisted_reports: 7,
          rebuild_total_accounts: 18,
          company_signal_effect_rate: 0.67,
          company_signal_creation_rate: 0.5,
          rebuild_trigger_rate: 0.8,
          rebuild_block_rate: 0.2,
          avg_rebuild_reports_per_triggered: 1.75,
          avg_rebuild_accounts_per_triggered: 4.5,
        },
        prior: {
          action_count: 3,
          approvals: 2,
          suppressions: 1,
          company_signal_creations: 1,
          company_signal_updates: 1,
          company_signal_deletions: 0,
          company_signal_noops: 1,
          rebuild_requests: 2,
          rebuild_triggered: 1,
          rebuild_blocked: 1,
          rebuild_persisted_runs: 1,
          rebuild_persisted_reports: 2,
          rebuild_total_accounts: 6,
          company_signal_effect_rate: 0.33,
          company_signal_creation_rate: 0.33,
          rebuild_trigger_rate: 0.5,
          rebuild_block_rate: 0.5,
          avg_rebuild_reports_per_triggered: 2,
          avg_rebuild_accounts_per_triggered: 6,
        },
        deltas: {
          action_count: 3,
          approvals: 2,
          suppressions: 1,
          company_signal_creations: 2,
          company_signal_updates: 0,
          company_signal_deletions: 1,
          company_signal_noops: 0,
          rebuild_requests: 3,
          rebuild_triggered: 3,
          rebuild_blocked: 0,
          rebuild_persisted_runs: 2,
          rebuild_persisted_reports: 5,
          rebuild_total_accounts: 12,
          company_signal_effect_rate: 0.34,
          company_signal_creation_rate: 0.17,
          rebuild_trigger_rate: 0.3,
          rebuild_block_rate: -0.3,
          avg_rebuild_reports_per_triggered: -0.25,
          avg_rebuild_accounts_per_triggered: -1.5,
        },
      },
      trend_focus: {},
      trend_alerts: [
        {
          status: 'watch',
          focus: 'pending_backlog',
          metric: 'pending_groups',
          direction: 'up',
          delta: 2,
          rationale: 'Pending analyst-review work is building in the live queue.',
          queue_filters: {
            candidate_bucket: 'analyst_review',
            review_status: 'pending',
          },
          queue_snapshot: {
            pending_groups: 9,
            actionable_pending_groups: 4,
            blocked_pending_groups: 5,
            overdue_pending_groups: 2,
            oldest_pending_age_days: 5,
          },
        },
      ],
      trend_recommendation: { status: 'ok' },
      trend_recommendation_filters: {},
      trend_recommendation_queue_filters: {
        candidate_bucket: 'canonical_ready',
        review_status: 'pending',
        source_name: 'reddit',
        review_priority_band: 'medium',
      },
      trend_recommendation_queue_snapshot: null,
      trend_queue_rankings: [
        {
          primary_driver: {
            label: 'pending_backlog',
            rationale: 'Pending analyst-review work is the highest-leverage slice right now.',
          },
          queue_filters: {
            candidate_bucket: 'analyst_review',
            review_status: 'pending',
          },
          queue_snapshot: {
            pending_groups: 9,
            actionable_pending_groups: 4,
            blocked_pending_groups: 5,
            overdue_pending_groups: 2,
            oldest_pending_age_days: 5,
          },
        },
      ],
      trend_queue_focus: null,
      trend_queue_recommendation: {
        status: 'focus',
        action: 'Work the actionable pending slice first',
        queue_filters: {
          candidate_bucket: 'canonical_ready',
          review_status: 'pending',
          source_name: 'reddit',
          review_priority_band: 'medium',
        },
      },
      operator_focus: {
        status: 'focus',
        action: 'Audit recently approved groups with rebuild activity',
        rationale: 'Recent approvals are creating signals and triggering rebuilds.',
        queue_filters: {
          candidate_bucket: 'analyst_review',
          review_status: 'pending',
          source_name: 'reddit',
          canonical_gap_reason: 'low_confidence_low_trust_source',
          review_priority_band: 'high',
          decision_makers_only: true,
        },
      },
      review_scope: null,
      canonical_gap_reason: null,
      rebuild_outcome: null,
      rebuild_reason: null,
      review_action: null,
      company_signal_action: null,
      review_priority_band: null,
      review_priority_reason: null,
      review_unlock_path: null,
      review_unlock_reason: null,
      candidate_source: null,
    })

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Company Signal Queue')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Review Status'), 'approved')

    expect(await screen.findByText('Review Activity')).toBeInTheDocument()
    expect(screen.getByText('Trend Alerts')).toBeInTheDocument()
    expect(screen.getByText('Ranked Queue Slices')).toBeInTheDocument()
    expect(screen.getByText('Recent 7d')).toBeInTheDocument()
    expect(screen.getByText('Vs Prior 7d')).toBeInTheDocument()
    expect(screen.getByText('Top Vendors')).toBeInTheDocument()
    expect(screen.getByText('Rebuild Reasons')).toBeInTheDocument()
    expect(screen.getAllByText('Shopify').length).toBeGreaterThan(0)
    expect(screen.getByText('accounts in motion refresh')).toBeInTheDocument()
    expect(screen.getByText('Audit recently approved groups with rebuild activity')).toBeInTheDocument()
    expect(screen.getAllByText('Changes current queue: sets review status pending').length).toBeGreaterThan(0)
    expect(screen.getAllByText((content) => content.includes('Queue impact:')).length).toBeGreaterThan(0)
    expect(screen.getAllByText((content) => content.includes('3 fewer pending')).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('button', { name: 'Preview Slice' }).length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Apply Operator Focus' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply Queue Recommendation' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reset Queue' })).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: 'Preview Slice' })[0])

    expect(await screen.findByText('Preview Queue Slice')).toBeInTheDocument()
    expect(screen.getByText('Trend alert: pending backlog')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply Preview' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear Preview' })).toBeInTheDocument()
    expect(screen.getByText('1 of 1 visible groups match preview')).toBeInTheDocument()
    expect(screen.getByText('Preview match')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Apply Preview' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: undefined,
        canonical_gap_reason: undefined,
        review_priority_band: undefined,
        decision_makers_only: undefined,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: undefined,
        canonical_gap_reason: undefined,
        review_priority_band: undefined,
        decision_makers_only: undefined,
        limit: 25,
      })
    })
    await waitFor(() => {
      expect(screen.queryByText('Preview Queue Slice')).not.toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: 'Apply Operator Focus' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: 'high',
        decision_makers_only: true,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: 'low_confidence_low_trust_source',
        review_priority_band: 'high',
        decision_makers_only: true,
        limit: 25,
      })
    })
    expect(screen.getByText('pending groups')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText(/Active preset: Apply Operator Focus/)).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: 'Apply Queue Recommendation' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'canonical_ready',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: undefined,
        review_priority_band: 'medium',
        decision_makers_only: undefined,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'canonical_ready',
        review_status: 'pending',
        source_name: 'reddit',
        canonical_gap_reason: undefined,
        review_priority_band: 'medium',
        decision_makers_only: undefined,
        limit: 25,
      })
    })
    await user.click(screen.getByRole('button', { name: 'Reset Queue' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: undefined,
        canonical_gap_reason: undefined,
        review_priority_band: undefined,
        decision_makers_only: undefined,
        top_n: 8,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        source_name: undefined,
        canonical_gap_reason: undefined,
        review_priority_band: undefined,
        decision_makers_only: undefined,
        limit: 25,
      })
    })
  })

  it('renders B2B token cards and run-level tier splits', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    const costTabs = await screen.findAllByRole('button', { name: 'Costs' })
    await user.click(costTabs[0])

    expect(await screen.findByText('B2B Tokens By Pass')).toBeInTheDocument()
    expect(screen.getByText('Enrichment Tier Tokens')).toBeInTheDocument()
    expect(screen.getByText('Tier 1 Billable In')).toBeInTheDocument()
    expect(screen.getByText('Tier 2 Billable In')).toBeInTheDocument()
    expect(screen.getByText('Show No-Op Runs (1)')).toBeInTheDocument()
    expect(screen.getByText('All 1.3K in / 210 out')).toBeInTheDocument()
    expect(screen.getByText('T1 700 / 120')).toBeInTheDocument()
    expect(screen.getByText('T2 600 / 90')).toBeInTheDocument()
    expect(screen.queryByText('b2b_enrichment_repair')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Show No-Op Runs (1)' }))

    expect(await screen.findByText('b2b_enrichment_repair')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Hide No-Op Runs' })).toBeInTheDocument()
  })

  it('hydrates the active tab from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/pipeline-review?tab=costs']}>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('B2B Tokens By Pass')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Costs' })).toHaveClass('text-cyan-400')
  })

  it('opens the saved-view drawer and runs per-view delivery actions', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findAllByText('Watchlist Alert Delivery')

    const queueTabs = screen.getAllByRole('button', { name: 'Queue' })
    await user.click(queueTabs[0])

    const inspectButtons = screen.getAllByRole('button', { name: 'Inspect saved view Daily CRM Watch' })
    await user.click(inspectButtons[0])

    await waitFor(() => {
      expect(api.fetchWatchlistDeliveryViewDetail).toHaveBeenCalledWith('view-1', {
        event_status: 'all',
        event_limit: 20,
        log_limit: 8,
      })
    })

    expect(await screen.findByText('Current Alert Events')).toBeInTheDocument()
    expect(screen.getByText('Salesforce urgency breached the saved view threshold.')).toBeInTheDocument()
    expect(screen.getByText('Preview Policy')).toBeInTheDocument()
    expect(screen.getByText('Min confidence: 0.65 (view)')).toBeInTheDocument()
    expect(screen.getByText('Budget authority: Optional (view)')).toBeInTheDocument()
    expect(screen.getByText('Last Blocked Preview Summary')).toBeInTheDocument()
    expect(screen.getAllByText('1 blocked preview alert: confidence >= 0.65 required').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Salesforce?back_to=%2Fpipeline-review',
    )
    expect(
      screen.getAllByRole('link', { name: 'Evidence' }).some((node) =>
        node.getAttribute('href') === '/evidence?vendor=Salesforce&tab=witnesses&back_to=%2Fpipeline-review',
      ),
    ).toBe(true)
    expect(
      screen.getAllByRole('link', { name: 'Reports' }).some((node) =>
        node.getAttribute('href') === '/reports?vendor_filter=Salesforce&back_to=%2Fpipeline-review',
      ),
    ).toBe(true)
    expect(
      screen.getAllByRole('link', { name: 'Opportunities' }).some((node) =>
        node.getAttribute('href') === '/opportunities?vendor=Salesforce&back_to=%2Fpipeline-review',
      ),
    ).toBe(true)

    await user.click(screen.getByRole('button', { name: 'Deliver Now' }))
    await waitFor(() => {
      expect(api.runWatchlistDeliveryForView).toHaveBeenCalledWith('view-1')
    })

    await user.click(screen.getByRole('button', { name: 'Disable Email' }))
    await waitFor(() => {
      expect(api.disableWatchlistDeliveryForView).toHaveBeenCalledWith('view-1')
    })
  })
})
