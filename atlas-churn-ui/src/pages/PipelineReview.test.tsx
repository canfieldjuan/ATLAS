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
  fetchCompanySignalCandidateGroups: vi.fn(),
  fetchCompanySignalCandidateGroupSummary: vi.fn(),
  fetchCompanySignalReviewImpactSummary: vi.fn(),
  approveCompanySignalCandidateGroup: vi.fn(),
  approveCompanySignalCandidateGroups: vi.fn(),
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
  fetchWatchlistDeliveryOps: vi.fn(),
  fetchWatchlistDeliveryViewDetail: vi.fn(),
  runWatchlistDeliveryForView: vi.fn(),
  disableWatchlistDeliveryForView: vi.fn(),
  runAutonomousTask: vi.fn(),
  suppressCompanySignalCandidateGroup: vi.fn(),
  suppressCompanySignalCandidateGroups: vi.fn(),
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
    api.fetchCompanySignalCandidateGroupSummary.mockResolvedValue({
      totals: {
        pending_groups: 7,
        actionable_pending_groups: 4,
        blocked_pending_groups: 3,
        overdue_pending_groups: 2,
        canonical_ready_groups: 5,
        analyst_review_groups: 2,
      },
      top_vendors: [
        {
          vendor_name: 'Salesforce',
          group_count: 3,
          review_count: 7,
          pending_groups: 2,
          canonical_ready_groups: 1,
        },
      ],
      pending_priority_reasons: [
        {
          review_priority_band: 'high',
          review_priority_reason: 'cross_source_corroboration',
          group_count: 2,
          review_count: 5,
        },
      ],
      candidate_bucket: null,
      review_status: null,
      review_priority_band: null,
      review_priority_reason: null,
      source_name: null,
    })
    api.fetchCompanySignalCandidateGroups.mockResolvedValue({
      groups: [
        {
          group_id: 'group-1',
          company: 'Acme Corp',
          display_company: 'Acme Corp',
          vendor: 'Salesforce',
          category: 'crm',
          review_count: 7,
          distinct_source_count: 3,
          decision_maker_count: 2,
          signal_evidence_count: 4,
          canonical_ready_review_count: 1,
          avg_urgency: 7.4,
          max_urgency: 8.8,
          corroborated_confidence_score: 0.62,
          confidence_tier: 'medium',
          representative_review_id: 'rev-1',
          representative_source: 'reddit',
          canonical_gap_reason: 'low_confidence_low_trust_source',
          candidate_bucket: 'analyst_review',
          review_priority_band: 'high',
          review_priority_reason: 'cross_source_corroboration',
          review_status: 'pending',
          review_status_updated_at: null,
          first_seen_at: '2026-04-07T18:00:00Z',
          last_seen_at: '2026-04-08T18:00:00Z',
          supporting_reviews: [
            {
              review_id: 'rev-2',
              source: 'g2',
              summary: 'Admins needed evidence before cutting over.',
              review_excerpt: 'Admins needed evidence before cutting over.',
              source_url: 'https://example.com/rev-2',
              reviewed_at: '2026-04-06T18:00:00Z',
              quote_excerpt: 'Needed evidence before cutting over.',
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
    api.fetchCompanySignalReviewImpactSummary.mockResolvedValue({
      totals: {},
      review_scope: null,
      canonical_gap_reason: null,
      rebuild_outcome: null,
      rebuild_reason: null,
      scopes: [],
      unlock_paths: [],
      priority_bands: [],
      priority_reasons: [],
      top_vendors: [],
      top_vendor_reasons: [],
      rebuild_reasons: [],
      daily_trends: [],
      trend_comparison: {
        anchor_day: '2026-04-07',
        recent_days: 15,
        prior_days: 15,
        recent_effect_rate: 0.5,
        prior_effect_rate: 0.75,
      },
      trend_focus: {
        status: 'alert',
        focus: 'effect_rate_down',
        metric: 'effect_rate',
        direction: 'down',
        delta: -0.25,
        recent_value: 0.5,
        prior_value: 0.75,
        rationale: 'Recent review actions are producing fewer downstream effects per action.',
      },
      trend_alerts: [
        {
          status: 'alert',
          focus: 'effect_rate_down',
          metric: 'effect_rate',
          direction: 'down',
          delta: -0.25,
          recent_value: 0.5,
          prior_value: 0.75,
          rationale: 'Recent review actions are producing fewer downstream effects per action.',
        },
      ],
      trend_recommendation: {
        status: 'act',
        action: 'review_effect_quality',
        priority: 'high',
        owner: 'review_ops',
        rationale: 'Recent review actions are producing fewer downstream company-signal effects per action.',
        supporting_focuses: ['effect_rate_down', 'approval_volume_up'],
      },
      trend_recommendation_filters: {
        company_signal_action: 'none',
        review_action: 'approved',
      },
    })
    api.approveCompanySignalCandidateGroup.mockResolvedValue({
      review_batch_id: 'batch-1',
      group_id: 'group-1',
      review_status: 'approved',
      company_name: 'Acme Corp',
      vendor_name: 'Salesforce',
      review_count: 7,
      company_signal_action: 'created',
      rebuild: { triggered: true },
    })
    api.approveCompanySignalCandidateGroups.mockResolvedValue({
      review_batch_id: 'batch-3',
      count: 1,
      groups: [
        {
          group_id: 'group-1',
          review_status: 'approved',
          company_name: 'Acme Corp',
          vendor_name: 'Salesforce',
          review_count: 7,
          company_signal_action: 'created',
        },
      ],
      rebuilds: [{ vendor_name: 'Salesforce', triggered: true }],
    })
    api.suppressCompanySignalCandidateGroup.mockResolvedValue({
      review_batch_id: 'batch-2',
      group_id: 'group-1',
      review_status: 'suppressed',
      company_name: 'Acme Corp',
      vendor_name: 'Salesforce',
      review_count: 7,
      company_signal_action: 'deleted',
      rebuild: { triggered: true },
    })
    api.suppressCompanySignalCandidateGroups.mockResolvedValue({
      review_batch_id: 'batch-4',
      count: 1,
      groups: [
        {
          group_id: 'group-1',
          review_status: 'suppressed',
          company_name: 'Acme Corp',
          vendor_name: 'Salesforce',
          review_count: 7,
          company_signal_action: 'deleted',
        },
      ],
      rebuilds: [{ vendor_name: 'Salesforce', triggered: true }],
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

    await user.click(screen.getByRole('button', { name: 'Run Delivery Now' }))

    await waitFor(() => {
      expect(api.runAutonomousTask).toHaveBeenCalledWith('b2b_watchlist_alert_delivery')
    })
    expect(await screen.findByText('Triggered delivery run')).toBeInTheDocument()
  })

  it('renders company signal queue summary with vendor shortcuts', async () => {
    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Company Signal Review Queue')).toBeInTheDocument()
    expect(screen.getAllByText('cross source corroboration').length).toBeGreaterThan(0)
    expect(screen.getByRole('link', { name: 'Salesforce' })).toHaveAttribute(
      'href',
      '/vendors/Salesforce?back_to=%2Fpipeline-review',
    )
    expect(screen.getByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open representative review Acme Corp' })).toHaveAttribute(
      'href',
      '/reviews/rev-1?back_to=%2Fpipeline-review',
    )
    expect(
      screen
        .getAllByRole('link', { name: 'Review' })
        .some((link) => link.getAttribute('href') === '/reviews/rev-2?back_to=%2Fpipeline-review'),
    ).toBe(true)
    expect(screen.getByText('Needed evidence before cutting over.')).toBeInTheDocument()
    expect(
      screen
        .getAllByRole('link', { name: 'Evidence' })
        .some((link) => link.getAttribute('href') === '/evidence?vendor=Salesforce&tab=witnesses&back_to=%2Fpipeline-review'),
    ).toBe(true)
    expect(
      screen
        .getAllByRole('link', { name: 'Reports' })
        .some((link) => link.getAttribute('href') === '/reports?vendor_filter=Salesforce&back_to=%2Fpipeline-review'),
    ).toBe(true)
    expect(
      screen
        .getAllByRole('link', { name: 'Opportunities' })
        .some((link) => link.getAttribute('href') === '/opportunities?vendor=Salesforce&back_to=%2Fpipeline-review'),
    ).toBe(true)

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        window_days: 90,
        limit: 10,
      })
    })
  })

  it('hydrates company signal queue filters from the URL and preserves drilldown context', async () => {
    render(
      <MemoryRouter
        initialEntries={[
          '/pipeline-review?queue_vendor=Salesforce&queue_priority_band=high&queue_priority_reason=cross_source_corroboration',
        ]}
      >
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Salesforce')).toBeInTheDocument()
    expect(screen.getByLabelText('Priority')).toHaveValue('high')

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: 'Salesforce',
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: 'cross_source_corroboration',
        window_days: 90,
        limit: 10,
      })
    })

    expect(screen.getByRole('link', { name: 'Open representative review Acme Corp' })).toHaveAttribute(
      'href',
      '/reviews/rev-1?back_to=%2Fpipeline-review%3Fqueue_vendor%3DSalesforce%26queue_priority_band%3Dhigh%26queue_priority_reason%3Dcross_source_corroboration',
    )
  })

  it('focuses the company signal queue from summary rows', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Company Signal Review Queue')
    await user.click(screen.getByRole('button', { name: 'Focus queue for Salesforce' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: 'Salesforce',
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: 'Salesforce',
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        limit: 10,
      })
    })

    await user.click(screen.getByRole('button', { name: 'Focus queue for cross_source_corroboration' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: 'Salesforce',
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: 'cross_source_corroboration',
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: 'Salesforce',
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: 'cross_source_corroboration',
        window_days: 90,
        limit: 10,
      })
    })

    expect(screen.getByRole('link', { name: 'Open representative review Acme Corp' })).toHaveAttribute(
      'href',
      '/reviews/rev-1?back_to=%2Fpipeline-review%3Fqueue_vendor%3DSalesforce%26queue_priority_band%3Dhigh%26queue_priority_reason%3Dcross_source_corroboration',
    )
  })

  it('lets operators edit priority reason filters directly', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.type(screen.getByLabelText('Priority Reason'), 'missing_signal_evidence')
    expect(screen.getByLabelText('Priority Reason')).toHaveValue('missing_signal_evidence')

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: 'missing_signal_evidence',
        window_days: 90,
        top_n: 6,
      })
    })
    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: 'missing_signal_evidence',
        window_days: 90,
        limit: 10,
      })
    })
  })

  it('focuses the queue source from supporting review badges', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.click(screen.getByRole('button', { name: 'Focus queue for source g2' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: 'g2',
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: 'g2',
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        limit: 10,
      })
    })

    expect(screen.getByRole('link', { name: 'Open representative review Acme Corp' })).toHaveAttribute(
      'href',
      '/reviews/rev-1?back_to=%2Fpipeline-review%3Fqueue_source%3Dg2',
    )
  })

  it('clears individual queue filter chips without resetting the rest of the queue', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.click(screen.getByRole('button', { name: 'Focus queue for source g2' }))
    await screen.findByRole('button', { name: 'Remove queue filter Source: g2' })

    await user.click(screen.getByRole('button', { name: 'Remove queue filter Source: g2' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: undefined,
        review_priority_reason: undefined,
        window_days: 90,
        limit: 10,
      })
    })
  })

  it('focuses queue priority filters directly from candidate group rows', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.click(screen.getByRole('button', { name: 'Focus queue for priority high' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: undefined,
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: undefined,
        window_days: 90,
        limit: 10,
      })
    })

    await user.click(screen.getByRole('button', { name: 'Focus queue for priority reason cross_source_corroboration' }))

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroupSummary).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: 'cross_source_corroboration',
        window_days: 90,
        top_n: 6,
      })
    })

    await waitFor(() => {
      expect(api.fetchCompanySignalCandidateGroups).toHaveBeenLastCalledWith({
        candidate_bucket: 'analyst_review',
        review_status: 'pending',
        vendor_name: undefined,
        source_name: undefined,
        review_priority_band: 'high',
        review_priority_reason: 'cross_source_corroboration',
        window_days: 90,
        limit: 10,
      })
    })
  })

  it('approves candidate groups from the queue with action defaults', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.type(screen.getByLabelText('Review notes'), 'Needs manual rebuild review')
    await user.click(screen.getByLabelText('Trigger rebuild after review'))
    await user.click(screen.getByRole('button', { name: 'Approve group Acme Corp' }))

    await waitFor(() => {
      expect(api.approveCompanySignalCandidateGroup).toHaveBeenCalledWith('group-1', {
        notes: 'Needs manual rebuild review',
        trigger_rebuild: false,
      })
    })
    expect(await screen.findByText('Approved Acme Corp for Salesforce')).toBeInTheDocument()
  })

  it('bulk approves selected candidate groups from the queue', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <PipelineReview />
      </MemoryRouter>,
    )

    await screen.findByText('Candidate Groups')
    await user.click(screen.getByRole('button', { name: 'Select visible candidate groups' }))
    await user.click(screen.getByRole('button', { name: 'Approve selected candidate groups' }))

    await waitFor(() => {
      expect(api.approveCompanySignalCandidateGroups).toHaveBeenCalledWith({
        group_ids: ['group-1'],
        trigger_rebuild: true,
      })
    })
    expect(await screen.findByText('Approved 1 candidate group')).toBeInTheDocument()
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

  it('renders review impact recommendations in the quality tab', async () => {
    render(
      <MemoryRouter initialEntries={['/pipeline-review?tab=quality']}>
        <PipelineReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Review Impact Recommendation')).toBeInTheDocument()
    expect(screen.getByText('review effect quality')).toBeInTheDocument()
    expect(screen.getByText('Recent review actions are producing fewer downstream company-signal effects per action.')).toBeInTheDocument()
    expect(screen.getByText('Primary focus: effect rate down')).toBeInTheDocument()
    expect(screen.getByText('company signal action: none')).toBeInTheDocument()
    expect(screen.getByText('review action: approved')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchCompanySignalReviewImpactSummary).toHaveBeenCalledWith({
        window_days: 30,
        top_n: 10,
      })
    })
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
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Salesforce?back_to=%2Fpipeline-review',
    )
    expect(
      screen
        .getAllByRole('link', { name: 'Evidence' })
        .some((link) => link.getAttribute('href') === '/evidence?vendor=Salesforce&tab=witnesses&back_to=%2Fpipeline-review'),
    ).toBe(true)
    expect(
      screen
        .getAllByRole('link', { name: 'Reports' })
        .some((link) => link.getAttribute('href') === '/reports?vendor_filter=Salesforce&back_to=%2Fpipeline-review'),
    ).toBe(true)
    expect(
      screen
        .getAllByRole('link', { name: 'Opportunities' })
        .some((link) => link.getAttribute('href') === '/opportunities?vendor=Salesforce&back_to=%2Fpipeline-review'),
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
