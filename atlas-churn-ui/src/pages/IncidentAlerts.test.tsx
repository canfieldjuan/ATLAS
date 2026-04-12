import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
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

const DEFAULT_DELIVERIES = [
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
]

function filterDeliveries(params?: { success?: boolean; event_type?: string; limit?: number; vendor_name?: string }) {
  let deliveries = [...DEFAULT_DELIVERIES]
  if (typeof params?.success === 'boolean') {
    deliveries = deliveries.filter((delivery) => delivery.success === params.success)
  }
  if (params?.event_type) {
    deliveries = deliveries.filter((delivery) => delivery.event_type === params.event_type)
  }
  if (params?.vendor_name) {
    deliveries = deliveries.filter((delivery) => delivery.vendor_name === params.vendor_name)
  }
  const limit = params?.limit ?? deliveries.length
  return { deliveries: deliveries.slice(0, limit), count: deliveries.length }
}

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
    api.listWebhookDeliveries.mockImplementation(async (_webhookId: string, params?: {
      success?: boolean
      event_type?: string
      limit?: number
      vendor_name?: string
    }) => filterDeliveries(params))
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

  it('hydrates vendor-scoped alert views through the API and URL state', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts?vendor=Acme%20Rival&webhook=wh-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Scoped to vendor: Acme Rival')).toBeInTheDocument()
    expect(screen.getByText('Webhooks with Activity')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Clear vendor scope' })).toHaveAttribute('href', '/alerts?webhook=wh-1')
    expect(api.fetchWebhookDeliverySummary).toHaveBeenCalledWith(7, { vendor_name: 'Acme Rival' })
    expect(api.listWebhooks).toHaveBeenCalledWith({ vendor_name: 'Acme Rival' })
    expect(api.listWebhookDeliveries).toHaveBeenCalledWith('wh-1', expect.objectContaining({
      limit: 10,
      vendor_name: 'Acme Rival',
    }))
  })

  it('shows vendor-scope workflow shortcuts in the header when a vendor filter is active', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-scope',
          url: 'https://hooks.example.com/vendor',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Vendor scoped endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 3,
          recent_success_rate_7d: 1,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?vendor=Acme%20Rival']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Acme+Rival&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Acme+Rival&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )

    await user.click(screen.getByRole('button', { name: 'Copy evidence link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival`,
      )
    })
    expect(await screen.findByText('Copied evidence link')).toBeInTheDocument()
  })

  it('reuses exact upstream workflow paths for vendor-scoped header shortcuts', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-upstream',
          url: 'https://hooks.example.com/vendor',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Vendor scoped endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 3,
          recent_success_rate_7d: 1,
        },
      ],
      count: 1,
    })
    const upstreamAccountReviewPath = '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor'
    const upstreamEvidencePath = `/evidence?vendor=Acme+Rival&tab=witnesses&witness_id=witness%3Aacme%3A1&back_to=${encodeURIComponent(upstreamAccountReviewPath)}`

    render(
      <MemoryRouter initialEntries={[`/alerts?vendor=Acme%20Rival&back_to=${encodeURIComponent(upstreamEvidencePath)}`]}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute('href', upstreamAccountReviewPath)
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute('href', upstreamEvidencePath)

    await user.click(screen.getByRole('button', { name: 'Copy account review link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${upstreamAccountReviewPath}`)
    })
    expect(await screen.findByText('Copied account review link')).toBeInTheDocument()
  })

  it('surfaces the newest exact target in the vendor-scoped header', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-report',
          url: 'https://hooks.example.com/reports',
          event_types: ['report_generated'],
          channel: 'generic',
          enabled: true,
          description: 'Report delivery',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 2,
          recent_success_rate_7d: 1,
          latest_failure_event_type: 'report_generated',
          latest_failure_status_code: 500,
          latest_failure_error: 'report publish failed',
          latest_failure_at: '2026-04-10T02:20:00Z',
          latest_failure_report_id: 'report-older',
          latest_failure_report_type: 'battle_card',
          latest_failure_report_title: 'Older Battle Card',
          latest_failure_vendor_name: 'Acme Rival',
        },
        {
          id: 'wh-crm',
          url: 'https://hooks.example.com/crm',
          event_types: ['high_intent_push'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM escalation',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 4,
          recent_success_rate_7d: 1,
          latest_crm_push: {
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
        },
      ],
      count: 2,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?vendor=Acme%20Rival']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    const region = await screen.findByRole('region', { name: 'Latest exact target' })
    expect(within(region).getByText('Account review target: Acme Bank (Switch Risk)')).toBeInTheDocument()
    expect(within(region).getByText(/Latest CRM push succeeded/)).toBeInTheDocument()
    expect(within(region).getByText(/CRM escalation/)).toBeInTheDocument()
    expect(within(region).getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival',
    )

    await user.click(within(region).getByRole('button', { name: 'Copy account review link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts%3Fvendor%3DAcme%2520Rival`,
      )
    })
    expect(await screen.findByText('Copied account review link')).toBeInTheDocument()
  })

  it('does not show a latest exact target header region for vendor-only activity', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-vendor-only',
          url: 'https://hooks.example.com/vendor-only',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Vendor-only endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 2,
          recent_success_rate_7d: 0.5,
          latest_failure_event_type: 'churn_alert',
          latest_failure_status_code: 500,
          latest_failure_error: 'vendor-only failure',
          latest_failure_at: '2026-04-10T03:05:00Z',
          latest_failure_vendor_name: 'Acme Rival',
          latest_failure_company_name: 'Acme Bank',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts?vendor=Acme%20Rival']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Latest exact target' })).not.toBeInTheDocument()
  })

  it('uses an in-app confirmation modal before deleting a webhook', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true)

    try {
      render(
        <MemoryRouter initialEntries={['/alerts']}>
          <IncidentAlerts />
        </MemoryRouter>,
      )

      const deleteButton = await screen.findByRole('button', { name: 'Delete' })
      await user.click(deleteButton)

      expect(confirmSpy).not.toHaveBeenCalled()
      expect(api.deleteWebhookSubscription).not.toHaveBeenCalled()

      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent('Delete webhook PagerDuty bridge?')
      expect(dialog).toHaveTextContent('https://hooks.example.com/churn')

      await user.click(within(dialog).getByRole('button', { name: 'Delete webhook' }))

      await waitFor(() => {
        expect(api.deleteWebhookSubscription).toHaveBeenCalledWith('wh-1')
      })
      expect(await screen.findByText('Webhook deleted')).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it('does not delete a webhook when the in-app confirmation modal is cancelled', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    const deleteButton = await screen.findByRole('button', { name: 'Delete' })
    await user.click(deleteButton)

    const dialog = await screen.findByRole('alertdialog')
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))

    await waitFor(() => {
      expect(api.deleteWebhookSubscription).not.toHaveBeenCalled()
    })
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
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

  it('does not render vendor shortcuts for synthetic persisted manual tests', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-test',
          url: 'https://hooks.example.com/test',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Synthetic test endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 1,
          latest_test_success: false,
          latest_test_status_code: 500,
          latest_test_error: 'HTTP 500: manual test failed',
          latest_test_at: '2026-04-10T02:40:00Z',
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

    const summaryCard = await screen.findByText('Latest manual test failed')
    expect(summaryCard).toBeInTheDocument()
    expect(summaryCard.parentElement).toHaveTextContent('500')
    expect(summaryCard.parentElement).toHaveTextContent('manual test failed')
    expect(screen.queryByRole('link', { name: 'Vendor workspace' })).not.toBeInTheDocument()
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
          latest_failure_vendor_name: 'Acme Rival',
          latest_failure_company_name: 'Acme Bank',
          latest_test_success: false,
          latest_test_status_code: 504,
          latest_test_error: 'test timeout',
          latest_test_at: '2026-04-10T02:40:00Z',
          latest_test_signal_id: null,
          latest_test_review_id: null,
          latest_test_report_id: 'report-test',
          latest_test_report_type: 'battle_card',
          latest_test_report_title: 'Battle Card · Acme Rival',
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
    expect(screen.getByText('Review target: Acme Bank')).toBeInTheDocument()
    expect(screen.getByText('Report target: Battle Card · Acme Rival')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open Review' })).toHaveAttribute(
      'href',
      '/reviews/review-fail?back_to=%2Falerts',
    )
    expect(screen.getByRole('link', { name: 'Open Report' })).toHaveAttribute(
      'href',
      '/reports/report-test?back_to=%2Falerts',
    )

    await user.click(screen.getByRole('button', { name: 'Copy review link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}/reviews/review-fail?back_to=%2Falerts`)
    })
    expect(await screen.findByText('Copied review link')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy report link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}/reports/report-test?back_to=%2Falerts`)
    })
    expect(await screen.findByText('Copied report link')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy Report ID' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith('report-test')
    })
    expect(await screen.findByText('Copied report id')).toBeInTheDocument()
  })


  it('links latest failure cards into vendor workflows when only vendor context is available', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

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
    expect(screen.getByText('Vendor target: Acme Rival')).toBeInTheDocument()
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

    await user.click(screen.getByRole('button', { name: 'Copy vendor workspace link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}/vendors/Acme%20Rival?back_to=%2Falerts`)
    })
    expect(await screen.findByText('Copied vendor workspace link')).toBeInTheDocument()
  })



  it('links latest failure cards into exact account review when focus is available', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-failure-focus',
          url: 'https://hooks.example.com/failure-focus',
          event_types: ['signal_update'],
          channel: 'generic',
          enabled: true,
          description: 'Failure focus endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 0,
          latest_failure_event_type: 'signal_update',
          latest_failure_status_code: 500,
          latest_failure_error: 'downstream timeout',
          latest_failure_at: '2026-04-10T02:55:00Z',
          latest_failure_signal_id: 'sig-failure-focus',
          latest_failure_review_id: null,
          latest_failure_report_id: null,
          latest_failure_vendor_name: 'Acme Rival',
          latest_failure_company_name: 'Acme Bank',
          latest_failure_account_review_focus: {
            vendor: 'Acme Rival',
            company: 'Acme Bank',
            report_date: '2026-04-10',
            watch_vendor: 'Acme Rival',
            category: 'Switch Risk',
            track_mode: 'competitor',
          },
          latest_test_success: null,
          latest_test_status_code: null,
          latest_test_error: null,
          latest_test_at: null,
          latest_test_signal_id: null,
          latest_test_review_id: null,
          latest_test_report_id: null,
          latest_test_vendor_name: null,
          latest_test_company_name: null,
          latest_crm_push: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Failure focus endpoint')).toBeInTheDocument()
    expect(screen.getByText('Account review target: Acme Bank (Switch Risk)')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts',
    )
  })

  it('links latest manual test cards into exact account review when focus is available', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-test-focus',
          url: 'https://hooks.example.com/test-focus',
          event_types: ['churn_alert'],
          channel: 'generic',
          enabled: true,
          description: 'Test focus endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 0,
          latest_failure_event_type: null,
          latest_failure_status_code: null,
          latest_failure_error: null,
          latest_failure_at: null,
          latest_failure_signal_id: null,
          latest_failure_review_id: null,
          latest_failure_report_id: null,
          latest_failure_vendor_name: null,
          latest_failure_company_name: null,
          latest_failure_account_review_focus: null,
          latest_test_success: false,
          latest_test_status_code: 504,
          latest_test_error: 'test timeout',
          latest_test_at: '2026-04-10T02:40:00Z',
          latest_test_signal_id: null,
          latest_test_review_id: null,
          latest_test_report_id: null,
          latest_test_vendor_name: 'Acme Rival',
          latest_test_company_name: 'Acme Bank',
          latest_test_account_review_focus: {
            vendor: 'Acme Rival',
            company: 'Acme Bank',
            report_date: '2026-04-10',
            watch_vendor: 'Acme Rival',
            category: 'Budget Risk',
            track_mode: 'competitor',
          },
          latest_crm_push: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Test focus endpoint')).toBeInTheDocument()
    expect(screen.getByText('Latest manual test failed')).toBeInTheDocument()
    expect(screen.getByText('Account review target: Acme Bank (Budget Risk)')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Budget+Risk&account_track_mode=competitor&back_to=%2Falerts',
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
    expect(screen.getByText('Account review target: Acme Bank (Switch Risk)')).toBeInTheDocument()
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

  it('shows report target context on report-generated CRM summary cards', async () => {
    api.listWebhooks.mockResolvedValueOnce({
      webhooks: [
        {
          id: 'wh-crm-report',
          url: 'https://hooks.example.com/crm-report',
          event_types: ['report_generated'],
          channel: 'crm_hubspot',
          enabled: true,
          description: 'CRM report endpoint',
          created_at: '2026-04-09T03:00:00Z',
          updated_at: '2026-04-10T03:00:00Z',
          recent_deliveries_7d: 1,
          recent_success_rate_7d: 1,
          latest_crm_push: {
            id: 'push-report',
            signal_type: 'report_generated',
            signal_id: 'report-crm',
            vendor_name: 'Acme Rival',
            company_name: null,
            review_id: null,
            report_id: 'report-crm',
            report_type: 'battle_card',
            report_title: 'Battle Card · Acme Rival',
            account_review_focus: null,
            crm_record_id: 'deal-99',
            crm_record_type: 'deal',
            status: 'success',
            error: null,
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

    expect(await screen.findByText('CRM report endpoint')).toBeInTheDocument()
    expect(screen.getByText('Latest CRM push succeeded')).toBeInTheDocument()
    expect(screen.getByText('Report target: Battle Card · Acme Rival')).toBeInTheDocument()
    expect(screen.getByText('report-crm')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open Report' })).toHaveAttribute(
      'href',
      '/reports/report-crm?back_to=%2Falerts',
    )
    expect(screen.queryByRole('link', { name: 'Account Review' })).not.toBeInTheDocument()
  })

  it('shows the refreshed persisted manual test context after a test webhook run', async () => {
    const user = userEvent.setup()
    api.listWebhooks
      .mockResolvedValueOnce({
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
      .mockResolvedValueOnce({
        webhooks: [
          {
            id: 'wh-1',
            url: 'https://hooks.example.com/churn',
            event_types: ['churn_alert', 'signal_update'],
            channel: 'generic',
            enabled: true,
            description: 'PagerDuty bridge',
            created_at: '2026-04-09T03:00:00Z',
            updated_at: '2026-04-10T03:05:00Z',
            recent_deliveries_7d: 13,
            recent_success_rate_7d: 0.923,
            latest_failure_event_type: 'signal_update',
            latest_failure_status_code: 500,
            latest_failure_error: 'downstream timeout',
            latest_failure_at: '2026-04-10T02:55:00Z',
            latest_test_success: true,
            latest_test_status_code: 202,
            latest_test_error: null,
            latest_test_at: '2026-04-10T03:10:00Z',
            latest_test_signal_id: null,
            latest_test_review_id: null,
            latest_test_report_id: 'report-refresh',
            latest_test_report_type: 'battle_card',
            latest_test_report_title: 'Battle Card · Acme Rival',
            latest_test_vendor_name: 'Acme Rival',
            latest_test_company_name: 'Acme Bank',
          },
        ],
        count: 1,
      })

    render(
      <MemoryRouter>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'Retry Failed Test' }))

    expect(await screen.findByText('Latest manual test passed')).toBeInTheDocument()
    expect(await screen.findByText('Test webhook delivered')).toBeInTheDocument()
    expect(screen.getByText('report-refresh')).toBeInTheDocument()
    expect(screen.getByText('Report target: Battle Card · Acme Rival')).toBeInTheDocument()
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

  it('renders a watchlists return link when alerts is opened from a saved view', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Back to Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1',
    )
  })

  it('copies the current alerts view from the page header', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts?days=30&webhook=wh-1&delivery_status=failed&delivery_event=signal_update&back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy View Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?days=30&webhook=wh-1&delivery_status=failed&delivery_event=signal_update&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(await screen.findByText('Copied current alerts view')).toBeInTheDocument()
  })


  it('copies a stable webhook link without transient activity filters', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts?days=30&webhook=wh-1&delivery_status=failed&delivery_event=signal_update&back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Webhook Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?days=30&webhook=wh-1&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(await screen.findByText('Copied webhook link')).toBeInTheDocument()
  })

  it('preserves vendor scope in copied webhook links', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts?vendor=Acme%20Rival&webhook=wh-1&delivery_status=failed']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Incident Alerts API' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Webhook Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?vendor=Acme+Rival&webhook=wh-1`,
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

  it('copies exact delivery drillthrough links from activity rows', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/alerts']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Incident Alerts API' })
    await user.click(screen.getByRole('button', { name: 'View Activity' }))

    await user.click(await screen.findByRole('button', { name: 'Copy account review link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?account_vendor=Acme+Rival&account_company=Acme+Bank&account_report_date=2026-04-10&account_watch_vendor=Acme+Rival&account_category=Switch+Risk&account_track_mode=competitor&back_to=%2Falerts%3Fwebhook%3Dwh-1`,
      )
    })
    expect(await screen.findByText('Copied account review link')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy review link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reviews/33333333-3333-4333-8333-333333333334?back_to=%2Falerts%3Fwebhook%3Dwh-1`,
      )
    })
    expect(await screen.findByText('Copied review link')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy vendor workspace link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/vendors/Acme%20Rival?back_to=%2Falerts%3Fwebhook%3Dwh-1`,
      )
    })
    expect(await screen.findByText('Copied vendor workspace link')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy evidence link' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/evidence?vendor=Acme+Rival&back_to=%2Falerts%3Fwebhook%3Dwh-1`,
      )
    })
    expect(await screen.findByText('Copied evidence link')).toBeInTheDocument()
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

    await waitFor(() => {
      expect(api.listWebhookCrmPushLog).toHaveBeenCalledWith('wh-crm', { limit: 10, status: 'success' })
    })

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

  it('preserves upstream watchlists context in delivery drillthrough links', async () => {
    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-1&back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Recent Activity' })).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'Review' })).toHaveAttribute(
      'href',
      '/reviews/33333333-3333-4333-8333-333333333334?back_to=%2Falerts%3Fwebhook%3Dwh-1%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
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

  it('keeps CRM filters visible when the backend returns no matching pushes', async () => {
    const user = userEvent.setup()

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
    api.listWebhookCrmPushLog.mockImplementation(async (_webhookId: string, params?: {
      limit?: number
      status?: 'success' | 'failed'
    }) => (
      params?.status === 'success'
        ? { pushes: [], count: 0 }
        : {
          pushes: [
            {
              id: 'push-failed',
              signal_type: 'competitive_displacement',
              signal_id: 'sig-1',
              vendor_name: 'Acme Rival',
              company_name: 'Acme Bank',
              crm_record_id: 'deal-1',
              crm_record_type: 'deal',
              status: 'failed',
              error: 'delivery failed',
              pushed_at: '2026-04-10T03:05:00Z',
            },
          ],
          count: 1,
        }
    ))

    render(
      <MemoryRouter initialEntries={['/alerts?webhook=wh-crm&crm_status=success']}>
        <IncidentAlerts />
      </MemoryRouter>,
    )

    expect(await screen.findByText('No CRM pushes match the current filters.')).toBeInTheDocument()
    expect(screen.getByLabelText('CRM push status filter')).toHaveValue('success')

    await user.selectOptions(screen.getByLabelText('CRM push status filter'), 'failed')

    await waitFor(() => {
      expect(api.listWebhookCrmPushLog).toHaveBeenLastCalledWith('wh-crm', { limit: 10, status: 'failed' })
    })
    expect(await screen.findByText('delivery failed')).toBeInTheDocument()
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
    await waitFor(() => {
      expect(api.listWebhookDeliveries).toHaveBeenLastCalledWith('wh-1', {
        success: false,
        limit: 10,
      })
    })
    expect(screen.queryByText('attempt 1')).not.toBeInTheDocument()
    expect(screen.getByText('downstream timeout')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Delivery event filter'), 'churn_alert')
    await waitFor(() => {
      expect(api.listWebhookDeliveries).toHaveBeenLastCalledWith('wh-1', {
        success: false,
        event_type: 'churn_alert',
        limit: 10,
      })
    })
    expect(screen.getByText('No deliveries match the current filters.')).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'signal_update' })).toBeInTheDocument()
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
