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
          vendor_name: 'Acme Rival',
          company_name: 'Acme Bank',
          signal_id: '22222222-2222-2222-2222-222222222222',
          signal_type: 'competitive_displacement',
          review_id: '33333333-3333-4333-8333-333333333334',
          account_review_focus: {
            vendor: 'Acme Rival',
            company: 'Acme Bank',
            report_date: '2026-04-10',
            watch_vendor: 'Acme Rival',
            category: 'Switch Risk',
            track_mode: 'competitor',
          },
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
    expect(screen.getByRole('button', { name: 'Retry Failed Test' })).toBeInTheDocument()
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

  it('surfaces canonical refs on the webhook summary cards', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-refs',
          url: 'https://hooks.example.com/refs',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Ref-rich endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 3,
          recent_success_rate_7d: 0.667,
          latest_failure_event_type: 'signal_update',
          latest_failure_status_code: 500,
          latest_failure_error: 'downstream timeout',
          latest_failure_at: '2026-04-10T02:55:00Z',
          latest_failure_signal_id: 'sig-fail',
          latest_failure_review_id: 'review-fail',
          latest_failure_report_id: null,
          latest_test_success: false,
          latest_test_status_code: 504,
          latest_test_error: 'test timeout',
          latest_test_at: '2026-04-10T02:40:00Z',
          latest_test_signal_id: null,
          latest_test_review_id: null,
          latest_test_report_id: 'report-test',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Ref-rich endpoint')).toBeInTheDocument()
    expect(screen.getAllByText('Reference IDs')).toHaveLength(2)
    expect(screen.getByText('sig-fail')).toBeInTheDocument()
    expect(screen.getByText('review-fail')).toBeInTheDocument()
    expect(screen.getByText('report-test')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open Review' })).toHaveAttribute(
      'href',
      '/reviews/review-fail?back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Open Report' })).toHaveAttribute(
      'href',
      '/reports/report-test?back_to=%2Falerts',
    )

    await user.click(screen.getByRole('button', { name: 'Copy Report ID' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith('report-test')
    })
    expect(await screen.findByText('Copied report id')).toBeInTheDocument()
  })


  it('links latest failure cards into vendor workflows when only vendor context is available', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-vendor',
          url: 'https://hooks.example.com/vendor',
          event_types: ['signal_update'],
          channel: 'generic',
          enabled: true,
          description: 'Vendor-only endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 2,
          recent_success_rate_7d: 0.5,
          latest_failure_event_type: 'signal_update',
          latest_failure_status_code: 500,
          latest_failure_error: 'downstream timeout',
          latest_failure_at: '2026-04-10T02:55:00Z',
          latest_failure_signal_id: 'sig-vendor',
          latest_failure_review_id: null,
          latest_failure_report_id: null,
          latest_failure_vendor_name: 'Acme Rival',
          latest_failure_company_name: 'Acme Bank',
          latest_test_success: null,
          latest_test_status_code: null,
          latest_test_error: null,
          latest_test_at: null,
          latest_test_signal_id: null,
          latest_test_review_id: null,
          latest_test_report_id: null,
          latest_test_vendor_name: null,
          latest_test_company_name: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Vendor-only endpoint')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Open Review' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Open Report' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Falerts',
    )
  })


  it('shows the latest CRM push summary on CRM webhook cards', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-crm-summary',
          url: 'https://hooks.example.com/crm-summary',
          event_types: ['high_intent_push'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM summary endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 2,
          recent_success_rate_7d: 1,
          latest_crm_push: {
            id: 'push-latest',
            signal_type: 'company_signal',
            signal_id: 'sig-crm',
            vendor_name: 'Acme Rival',
            company_name: 'Acme Bank',
            review_id: null,
            report_id: null,
            report_type: null,
            report_title: null,
            account_review_focus: {
              vendor: 'Acme Rival',
              company: 'Acme Bank',
              report_date: '2026-04-10',
              watch_vendor: 'Acme Rival',
              category: 'Switch Risk',
              track_mode: 'competitor',
            },
            crm_record_id: 'deal-42',
            crm_record_type: 'deal',
            status: 'failed',
            error: 'crm timeout',
            pushed_at: '2026-04-10T03:05:00Z',
          },
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('CRM summary endpoint')).toBeInTheDocument()
    expect(screen.getByText('Latest CRM push failed')).toBeInTheDocument()
    expect(screen.getByText((text) => text.includes("company_signal") && text.includes("deal") && text.includes("deal-42"))).toBeInTheDocument()
    expect(screen.getByText('sig-crm')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Open Review' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Open Report' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Falerts',
    )
    expect(screen.getByText('crm timeout')).toBeInTheDocument()
  })

  it('shows the latest manual test result on the webhook card', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'Retry Failed Test' }))

    expect(await screen.findByText('Latest manual test passed')).toBeInTheDocument()
    expect(await screen.findByText('Test webhook delivered')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Re-test Endpoint' })).toBeInTheDocument()
  })


  it('does not duplicate a failed manual test as the latest failure banner', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-test',
          url: 'https://hooks.example.com/test-only',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Manual test endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 0,
          latest_failure_event_type: 'test',
          latest_failure_status_code: 504,
          latest_failure_error: 'test timeout',
          latest_failure_at: '2026-04-10T02:40:00Z',
          latest_test_success: false,
          latest_test_status_code: 504,
          latest_test_error: 'test timeout',
          latest_test_at: '2026-04-10T02:40:00Z',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Latest manual test failed')).toBeInTheDocument()
    expect(screen.queryByText(/Latest failure/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Retry Failed Test' })).toBeInTheDocument()
  })

  it('shows Send Test when the endpoint has no persisted test or failure history', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-fresh',
          url: 'https://hooks.example.com/new-endpoint',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Fresh endpoint',
          created_at: '2026-04-10T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 0,
          recent_success_rate_7d: null,
          latest_failure_at: null,
          latest_failure_event_type: null,
          latest_failure_status_code: null,
          latest_failure_error: null,
          latest_test_at: null,
          latest_test_success: null,
          latest_test_status_code: null,
          latest_test_error: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('button', { name: 'Send Test' })).toBeInTheDocument()
  })


  it('hydrates the summary window from the URL and preserves it in copied activity links', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/alerts?days=30&webhook=wh-1&delivery_status=failed']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('30-day window')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Activity Link' }))

    expect(await screen.findByText('Copied activity link')).toBeInTheDocument()
    expect(screen.getByLabelText('Window')).toHaveValue('30')
  })

  it('copies a stable webhook link without transient activity filters', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts?days=30&webhook=wh-1&delivery_status=failed&delivery_event=signal_update']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Webhook Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?days=30&webhook=wh-1`,
      )
    })
    expect(await screen.findByText('Copied webhook link')).toBeInTheDocument()
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

  it('copies canonical signal ids from delivery activity rows', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(screen.getByText('Reference IDs')).toBeInTheDocument()
    expect(screen.getByText('Signal ID')).toBeInTheDocument()
    expect(screen.getByText('22222222-2222-2222-2222-222222222222')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Signal ID' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith('22222222-2222-2222-2222-222222222222')
    })
    expect(await screen.findByText('Copied signal id')).toBeInTheDocument()
  })

  it('copies canonical signal ids from CRM push activity rows', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWebhooks.mockResolvedValue({
      webhooks: [
        {
          id: 'wh-crm',
          url: 'https://hooks.example.com/crm',
          event_types: ['churn_alert'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM escalation',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 1,
        },
      ],
      count: 1,
    })
    api.listWebhookDeliveries.mockResolvedValue({ deliveries: [], count: 0 })
    api.listWebhookCrmPushLog.mockResolvedValue({
      pushes: [
        {
          id: 'push-signal',
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
    expect(await screen.findByText('Acme Bank')).toBeInTheDocument()
    expect(screen.getByText('Reference IDs')).toBeInTheDocument()
    expect(screen.getByText('Signal ID')).toBeInTheDocument()
    expect(screen.getByText('sig-1')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Signal ID' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith('sig-1')
    })
    expect(await screen.findByText('Copied signal id')).toBeInTheDocument()
  })

  it('links delivery activity back into watchlists and vendor workflows', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'Review' })).toHaveAttribute(
      'href',
      '/reviews/33333333-3333-4333-8333-333333333334?back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
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
          review_id: '33333333-3333-4333-8333-333333333334',
          crm_record_id: 'deal-1',
          crm_record_type: 'deal',
          status: 'success',
          error: null,
          pushed_at: '2026-04-10T03:05:00Z',
          account_review_focus: {
            vendor: 'Acme Rival',
            company: 'Acme Bank',
            report_date: '2026-04-10',
            watch_vendor: 'Acme Rival',
            category: 'Switch Risk',
            track_mode: 'competitor',
          },
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
    expect(await screen.findAllByRole('link', { name: 'Review' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/reviews/33333333-3333-4333-8333-333333333334?back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Account Review' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Watchlists' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/watchlists?vendor_name=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Vendor workspace' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/vendors/Acme%20Rival?back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Evidence' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Reports' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/reports?vendor_filter=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
    expect(screen.getAllByRole('link', { name: 'Opportunities' })).toSatisfy((links) => (
      links.some((link: HTMLAnchorElement) => link.getAttribute('href') === '/opportunities?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess')
    ))
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
    expect(screen.getByText('3 event types selected')).toBeInTheDocument()
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

    await user.selectOptions(screen.getByLabelText('Preview Event'), 'high_intent_push')
    expect(screen.getByText(/"atlas_review_id": "33333333-3333-4333-8333-333333333334"/)).toBeInTheDocument()

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

  it('creates a CRM webhook with high intent pushes from the preset', async () => {
    const user = userEvent.setup()
    api.createWebhook.mockResolvedValueOnce({
      id: 'wh-crm-new',
      url: 'https://hooks.example.com/crm',
      event_types: ['churn_alert', 'report_generated', 'high_intent_push'],
      channel: 'crm_hubspot',
      enabled: true,
      description: 'CRM escalation',
      created_at: '2026-04-10T04:00:00Z',
      updated_at: '2026-04-10T04:00:00Z',
      recent_deliveries_7d: 0,
      recent_success_rate_7d: null,
    })

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: /CRM Escalation/i }))
    await user.type(screen.getByLabelText('Endpoint URL'), 'https://hooks.example.com/crm')
    await user.type(screen.getByLabelText('Auth Header'), 'Bearer atlas-token')
    await user.click(screen.getByRole('button', { name: 'Add Webhook' }))

    await waitFor(() => {
      expect(api.createWebhook).toHaveBeenCalledWith(expect.objectContaining({
        url: 'https://hooks.example.com/crm',
        event_types: ['churn_alert', 'report_generated', 'high_intent_push'],
        channel: 'crm_hubspot',
        auth_header: 'Bearer atlas-token',
      }))
    })
    expect(await screen.findByText('Added crm_hubspot webhook')).toBeInTheDocument()
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


  it('links report-generated deliveries to exact report detail', async () => {
    api.listWebhookDeliveries.mockResolvedValueOnce({
      deliveries: [
        {
          id: 'delivery-report',
          event_type: 'report_generated',
          status_code: 202,
          duration_ms: 95,
          attempt: 1,
          success: true,
          error: null,
          delivered_at: '2026-04-10T03:15:00Z',
          vendor_name: null,
          company_name: null,
          signal_type: 'report_generated',
          report_id: '33333333-3333-4333-8333-333333333333',
          report_type: 'battle_card',
          report_title: 'Acme Rival Battle Card',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Rival Battle Card')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Report' })).toHaveAttribute(
      'href',
      '/reports/33333333-3333-4333-8333-333333333333?back_to=%2Falerts%3Fwebhook%3Dwh-1',
    )
  })

  it('links report-generated CRM pushes to exact report detail', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-crm',
          url: 'https://hooks.example.com/crm',
          event_types: ['report_generated'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM escalation',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 1,
        },
      ],
      count: 1,
    })
    api.listWebhookDeliveries.mockResolvedValueOnce({
      deliveries: [],
      count: 0,
    })
    api.listWebhookCrmPushLog.mockResolvedValueOnce({
      pushes: [
        {
          id: 'push-report',
          signal_type: 'report_generated',
          signal_id: '33333333-3333-4333-8333-333333333333',
          vendor_name: null,
          company_name: null,
          report_id: '33333333-3333-4333-8333-333333333333',
          report_type: 'battle_card',
          report_title: 'Acme Rival Battle Card',
          crm_record_id: 'note-42',
          crm_record_type: 'note',
          status: 'success',
          error: null,
          pushed_at: '2026-04-10T03:20:00Z',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-crm&crm_status=success']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Rival Battle Card')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Report' })).toHaveAttribute(
      'href',
      '/reports/33333333-3333-4333-8333-333333333333?back_to=%2Falerts%3Fwebhook%3Dwh-crm%26crm_status%3Dsuccess',
    )
  })

})
