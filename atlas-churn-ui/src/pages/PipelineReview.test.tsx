import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
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
  fetchWatchlistDeliveryOps: vi.fn(),
  runAutonomousTask: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('PipelineReview watchlist delivery ops', () => {
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
})
