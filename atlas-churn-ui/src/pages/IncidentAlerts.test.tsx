import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import IncidentAlerts from './IncidentAlerts'

const api = vi.hoisted(() => ({
  createWebhook: vi.fn(),
  deleteWebhookSubscription: vi.fn(),
  fetchWebhookDeliverySummary: vi.fn(),
  listWebhookCrmPushLog: vi.fn(),
  listWebhookDeliveries: vi.fn(),
  listWebhooks: vi.fn(),
  testWebhookSubscription: vi.fn(),
  updateWebhookSubscription: vi.fn(),
}))

const clipboard = vi.hoisted(() => ({
  writeText: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('IncidentAlerts', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    Object.defineProperty(window.navigator, 'clipboard', {
      configurable: true,
      value: clipboard,
    })
    clipboard.writeText.mockResolvedValue(undefined)
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
          latest_failure_event_type: 'signal_update',
          latest_failure_status_code: 500,
          latest_failure_error: 'downstream timeout',
          latest_failure_at: '2026-04-10T02:55:00Z',
          latest_test_success: false,
          latest_test_status_code: 504,
          latest_test_error: 'test timeout',
          latest_test_at: '2026-04-10T02:40:00Z',
        },
      ],
      count: 1,
    })
    api.listWebhookDeliveries.mockResolvedValue({
      deliveries: [
        {
          id: 'delivery-1',
          event_type: 'churn_alert',
          status_code: 202,
          duration_ms: 180,
          attempt: 1,
          success: true,
          error: null,
          delivered_at: '2026-04-10T03:05:00Z',
        },
        {
          id: 'delivery-2',
          event_type: 'signal_update',
          status_code: 500,
          duration_ms: 210,
          attempt: 2,
          success: false,
          error: 'downstream timeout',
          delivered_at: '2026-04-10T02:55:00Z',
        },
      ],
      count: 2,
    })
    api.listWebhookCrmPushLog.mockResolvedValue({
      pushes: [],
      count: 0,
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
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(screen.getByText('PagerDuty bridge')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
    expect(screen.getByText('89%')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Re-test Endpoint' })).toBeInTheDocument()
    expect(screen.getByText('Latest failure · signal_update · 500')).toBeInTheDocument()
    expect(screen.getByText(/downstream timeout/)).toBeInTheDocument()
  })


  it('renders the latest persisted manual test result from the webhook list', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Latest manual test failed')).toBeInTheDocument()
    expect(screen.getByText(/test timeout/)).toBeInTheDocument()
  })

  it('shows the latest manual test result on the webhook card', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'Re-test Endpoint' }))

    expect(await screen.findByText('Latest manual test passed')).toBeInTheDocument()
    expect(await screen.findByText('Test webhook delivered')).toBeInTheDocument()
  })

  it('shows delivery activity drillthrough for a webhook', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'View Activity' }))

    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(screen.getByText(/attempt 1/i)).toBeInTheDocument()
    expect(screen.getByText('success')).toBeInTheDocument()
    expect(screen.getByText('CRM push history is only available for CRM webhook channels.')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.listWebhookDeliveries).toHaveBeenCalledWith('wh-1', { limit: 10 })
    })
  })

  it('hydrates activity drillthrough and filters from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-1&delivery_status=failed&delivery_event=signal_update']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(screen.getByLabelText('Delivery status filter')).toHaveValue('failed')
    expect(screen.getByLabelText('Delivery event filter')).toHaveValue('signal_update')
    expect(screen.getByText('downstream timeout')).toBeInTheDocument()
    expect(screen.queryByText(/attempt 1/i)).not.toBeInTheDocument()
  })

  it('links CRM push activity back into vendor workflows', async () => {
    api.listWebhooks.mockResolvedValue({
      webhooks: [
        {
          id: 'wh-crm',
          url: 'https://hooks.example.com/crm',
          event_types: ['churn_alert', 'report_generated'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM escalation',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 4,
          recent_success_rate_7d: 1,
        },
      ],
      count: 1,
    })
    api.listWebhookCrmPushLog.mockResolvedValue({
      pushes: [
        {
          id: 'push-1',
          signal_type: 'competitive_displacement',
          signal_id: 'sig-1',
          vendor_name: 'Acme Rival',
          company_name: 'Acme Bank',
          crm_record_id: 'deal-1',
          crm_record_type: 'deal',
          status: 'success',
          error: null,
          pushed_at: '2026-04-10T03:05:00Z',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-crm&crm_status=success']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess',
    )
  })

  it('filters delivery activity by result and event type', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'View Activity' }))

    expect(await screen.findAllByText('signal_update')).not.toHaveLength(0)
    await user.selectOptions(screen.getByLabelText('Delivery status filter'), 'failed')
    expect(screen.queryByText('attempt 1')).not.toBeInTheDocument()
    expect(screen.getByText('downstream timeout')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Delivery event filter'), 'churn_alert')
    expect(screen.getByText('No deliveries match the current filters.')).toBeInTheDocument()
  })

  it('applies presets and shows channel guidance', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: /CRM Escalation/i }))

    expect(screen.getByLabelText('Channel')).toHaveValue('crm_hubspot')
    expect(screen.getByLabelText('Auth Header')).toBeInTheDocument()
    expect(screen.getByText(/Requires an auth header/i)).toBeInTheDocument()
    expect(screen.getByText('2 event types selected')).toBeInTheDocument()
  })

  it('shows live setup checks before save', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })

    expect(screen.getByText('Webhook URL is required')).toBeInTheDocument()
    await user.type(screen.getByLabelText('Endpoint URL'), 'hooks.example.com/atlas')
    expect(screen.getByText('Webhook URL must start with http:// or https://')).toBeInTheDocument()

    await user.clear(screen.getByLabelText('Signing Secret'))
    await user.type(screen.getByLabelText('Signing Secret'), 'short-secret')
    expect(screen.getByText('Webhook secret must be at least 16 characters')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /CRM Escalation/i }))
    expect(screen.getByText('CRM channels require an auth header')).toBeInTheDocument()
  })

  it('updates the sample payload preview for channel-specific formats', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })

    expect(screen.getByText('Payload Preview')).toBeInTheDocument()
    expect(screen.getByText(/"event": "churn_alert"/)).toBeInTheDocument()
    expect(screen.getByText(/"vendor": "Acme Rival"/)).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Channel'), 'slack')
    expect(screen.getByText(/"text": "Atlas: churn_alert for Acme Rival"/)).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Channel'), 'teams')
    expect(screen.getByText(/"text": "Atlas Intelligence: churn_alert"/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /CRM Escalation/i }))
    await user.type(screen.getByLabelText('Auth Header'), 'Bearer atlas-token')
    expect(screen.getByText(/"dealname": "Churn Signal: Acme Rival"/)).toBeInTheDocument()
    expect(screen.getByText(/"Authorization": "Bearer atlas-token"/)).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Preview Event'), 'report_generated')
    expect(screen.getByText(/"hs_note_body": "Atlas Intelligence: report_generated for Acme Rival/)).toBeInTheDocument()
  })

  it('copies sample JSON and cURL from the preview panel', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })

    await user.type(screen.getByLabelText('Endpoint URL'), 'https://hooks.example.com/atlas')
    await user.click(screen.getByRole('button', { name: 'Copy Sample JSON' }))

    expect(await screen.findByText('Copied sample JSON')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy cURL' }))
    expect(await screen.findByText('Copied sample cURL')).toBeInTheDocument()
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
