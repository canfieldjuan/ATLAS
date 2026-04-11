import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import IncidentAlerts from './IncidentAlerts'

const api = vi.hoisted(() => ({
  createWebhook: vi.fn(),
  deleteWebhookSubscription: vi.fn(),
  fetchWebhookDeliverySummary: vi.fn(),
  listWebhooks: vi.fn(),
  testWebhookSubscription: vi.fn(),
  updateWebhookSubscription: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('IncidentAlerts', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchWebhookDeliverySummary.mockResolvedValue({
      window_days: 7,
      active_subscriptions: 2,
      total_deliveries: 18,
      successful: 16,
      failed: 2,
      success_rate: 0.889,
      avg_success_duration_ms: 212.4,
      p95_success_duration_ms: 480.1,
      last_delivery_at: '2026-04-10T03:00:00Z',
    })
    api.listWebhooks.mockResolvedValue({
      webhooks: [
        {
          id: 'wh-1',
          url: 'https://hooks.example.com/churn',
          event_types: ['churn_alert', 'signal_update'],
          channel: 'generic',
          enabled: true,
          description: 'PagerDuty bridge',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 12,
          recent_success_rate_7d: 0.917,
        },
      ],
      count: 1,
    })
    api.createWebhook.mockResolvedValue({
      id: 'wh-2',
      url: 'https://hooks.example.com/new',
      event_types: ['churn_alert'],
      channel: 'generic',
      enabled: true,
      description: 'New webhook',
      created_at: '2026-04-10T04:00:00Z',
      updated_at: '2026-04-10T04:00:00Z',
      recent_deliveries_7d: 0,
      recent_success_rate_7d: null,
    })
    api.updateWebhookSubscription.mockResolvedValue({})
    api.testWebhookSubscription.mockResolvedValue({
      success: true,
      subscription_id: 'wh-1',
      channel: 'generic',
    })
    api.deleteWebhookSubscription.mockResolvedValue({ deleted: true, id: 'wh-1' })
  })

  it('renders delivery health and existing webhooks', async () => {
    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(screen.getByText('PagerDuty bridge')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
    expect(screen.getByText('89%')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Send Test' })).toBeInTheDocument()
  })

  it('creates a webhook from the form', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })

    await user.type(screen.getByLabelText('Endpoint URL'), 'https://hooks.example.com/new')
    await user.type(screen.getByLabelText('Description'), 'New webhook')
    await user.click(screen.getByRole('button', { name: 'Add Webhook' }))

    await waitFor(() => {
      expect(api.createWebhook).toHaveBeenCalledWith(expect.objectContaining({
        url: 'https://hooks.example.com/new',
        description: 'New webhook',
        event_types: ['churn_alert'],
        channel: 'generic',
      }))
    })
    expect(await screen.findByText('Added generic webhook')).toBeInTheDocument()
  })
})
