import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { RouterProvider, createMemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Reports from './Reports'

const planGate = vi.hoisted(() => ({
  canAccessReports: true,
}))

const modalState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const api = vi.hoisted(() => ({
  buildReportLibraryViewScopeKey: vi.fn((filters?: Record<string, string>) => {
    return [
      'library-view',
      filters?.report_type || 'all',
      filters?.vendor_filter || 'all',
      filters?.quality_status || 'all',
    ].join('::')
  }),
  fetchReports: vi.fn(),
  listReportSubscriptions: vi.fn(),
  generateAccountComparisonReport: vi.fn(),
  generateAccountDeepDiveReport: vi.fn(),
  generateVendorComparisonReport: vi.fn(),
  requestBattleCardReport: vi.fn(),
  downloadReportPdf: vi.fn(),
  fetchReport: vi.fn(),
}))

vi.mock('../api/client', () => api)
vi.mock('../hooks/usePlanGate', () => ({
  usePlanGate: () => planGate,
}))
vi.mock('../components/SubscriptionModal', () => ({
  default: (props: any) => {
    if (props.open) {
      modalState.lastProps = props
    }
    if (!props.open) return null
    return (
      <div data-testid="subscription-modal">
        {props.scopeType}:{props.scopeKey}:{props.scopeLabel}
        <button
          onClick={() => props.onSaved?.({
            id: `saved-${props.scopeKey}`,
            scope_type: props.scopeType,
            scope_key: props.scopeKey,
            scope_label: props.scopeLabel,
            enabled: true,
          })}
        >
          Save subscription
        </button>
      </div>
    )
  },
}))

describe('Reports', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    modalState.lastProps = null
    planGate.canAccessReports = true
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    })
    api.listReportSubscriptions.mockResolvedValue({ subscriptions: [] })
    api.fetchReports.mockResolvedValue({
      reports: [
        {
          id: 'report-1',
          report_type: 'battle_card',
          vendor_filter: 'Zendesk',
          category_filter: null,
          executive_summary: 'Summary',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'completed',
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: 'sales_ready',
          report_subscription: null,
          has_pdf_export: true,
          latest_failure_step: null,
          latest_error_summary: null,
        },
      ],
      count: 1,
    })
    api.generateVendorComparisonReport.mockResolvedValue({ execution_id: 'exec-1' })
    api.generateAccountComparisonReport.mockResolvedValue({ execution_id: 'exec-2' })
    api.generateAccountDeepDiveReport.mockResolvedValue({ execution_id: 'exec-3' })
    api.requestBattleCardReport.mockResolvedValue({ execution_id: 'exec-4' })
    api.fetchReport.mockResolvedValue({
      id: 'report-1',
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      category_filter: null,
      executive_summary: 'Summary',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      status: 'completed',
      report_subscription: null,
      has_pdf_export: true,
      blocker_count: 0,
      warning_count: 0,
      unresolved_issue_count: 0,
      quality_status: 'sales_ready',
      latest_failure_step: null,
      latest_error_summary: null,
      intelligence_data: {},
    })
  })

  it('opens a filtered library subscription as a library_view scope', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')

    fireEvent.change(screen.getByPlaceholderText('Filter by vendor...'), {
      target: { value: 'Zendesk' },
    })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'battle_card' } })
    fireEvent.change(selects[1], { target: { value: 'sales_ready' } })
    fireEvent.change(selects[2], { target: { value: 'stale' } })
    fireEvent.change(selects[3], { target: { value: 'blocked' } })

    await waitFor(() => {
      expect(api.fetchReports).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
        freshness_state: 'stale',
        review_state: 'blocked',
        include_stale: true,
        limit: 100,
      })
    })

    expect(screen.getByRole('button', { name: 'Subscribe to View' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Subscribe to View' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    await waitFor(() => {
      expect(api.buildReportLibraryViewScopeKey).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
        freshness_state: 'stale',
        review_state: 'blocked',
      })
    })
    expect(modalState.lastProps.scopeType).toBe('library_view')
    expect(modalState.lastProps.scopeKey).toBe('library-view::battle_card::Zendesk::sales_ready')
    expect(modalState.lastProps.scopeLabel).toBe('Battle Card • Zendesk • Sales Ready • Stale • Blocked Library')
    expect(modalState.lastProps.filterPayload).toEqual({
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      quality_status: 'sales_ready',
      freshness_state: 'stale',
      review_state: 'blocked',
    })
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'library_view:library-view::battle_card::Zendesk::sales_ready:Battle Card • Zendesk • Sales Ready • Stale • Blocked Library',
    )
  })

  it('renders API trust states from the report payload', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-trust-1',
          report_type: 'vendor_scorecard',
          vendor_filter: 'Intercom',
          category_filter: null,
          executive_summary: 'Trust summary',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'completed',
          artifact_state: 'failed',
          artifact_label: 'Attention needed',
          report_subscription: null,
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          freshness_state: 'stale',
          freshness_label: 'Stale',
          review_state: 'blocked',
          review_label: 'Blocked',
          quality_status: null,
          latest_failure_step: 'operator_review',
          latest_error_summary: 'Evidence coverage is below threshold',
          trust: {
            artifact_state: 'failed',
            artifact_label: 'Attention needed',
            freshness_state: 'stale',
            freshness_label: 'Stale',
            review_state: 'blocked',
            review_label: 'Blocked',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Trust summary')).toBeInTheDocument()
    const card = screen.getByTestId('report-card-report-trust-1')
    expect(within(card).getByText('Attention needed')).toBeInTheDocument()
    expect(within(card).getByText('Blocked')).toBeInTheDocument()
    expect(within(card).getByText('Stale')).toBeInTheDocument()
    expect(within(card).getByText('PDF failed')).toBeInTheDocument()
    expect(within(card).getByRole('button', { name: 'Subscribe' })).toBeInTheDocument()
    expect(within(card).getByText('step: operator review')).toBeInTheDocument()
    expect(within(card).getByText('Evidence coverage is below threshold')).toBeInTheDocument()
  })

  it('shows export-ready state on report cards when a PDF is available', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-1')
    expect(within(card).getByText('PDF ready')).toBeInTheDocument()
    fireEvent.click(within(card).getByRole('button', { name: 'Export PDF' }))
    expect(api.downloadReportPdf).toHaveBeenCalledWith('report-1')
  })

  it('copies a report-scoped library link from the report card', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports?report_type=battle_card&vendor_filter=Zendesk&quality_status=sales_ready'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: 'Copy Link' }))

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      `${window.location.origin}/reports?report_type=battle_card&vendor_filter=Zendesk&quality_status=sales_ready&report_subscription=report-1&report_focus_type=battle_card&report_focus_vendor=Zendesk&report_focus_label=battle+card+-+Zendesk`,
    )
    await waitFor(() => {
      expect(within(card).getByRole('button', { name: 'Copied' })).toBeInTheDocument()
    })
  })

  it('opens a report-scoped subscription modal from a non-exportable card', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-sub-1',
          report_type: 'battle_card',
          vendor_filter: 'Intercom',
          category_filter: null,
          executive_summary: 'Needs follow-up',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'processing',
          artifact_state: 'processing',
          artifact_label: 'Processing',
          report_subscription: null,
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: null,
          latest_failure_step: null,
          latest_error_summary: null,
          trust: {
            artifact_state: 'processing',
            artifact_label: 'Processing',
            freshness_state: 'fresh',
            freshness_label: 'Fresh',
            review_state: 'clean',
            review_label: 'Clean',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-sub-1')
    fireEvent.click(within(card).getByRole('button', { name: 'Subscribe' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-sub-1')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Intercom')
  })

  it('updates a card to manage subscription immediately after a modal save', async () => {
    api.fetchReports.mockResolvedValue({
      reports: [
        {
          id: 'report-sub-optimistic',
          report_type: 'battle_card',
          vendor_filter: 'Linear',
          category_filter: null,
          executive_summary: 'Needs follow-up',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'processing',
          artifact_state: 'processing',
          artifact_label: 'Processing',
          report_subscription: null,
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: null,
          latest_failure_step: null,
          latest_error_summary: null,
          trust: {
            artifact_state: 'processing',
            artifact_label: 'Processing',
            freshness_state: 'fresh',
            freshness_label: 'Fresh',
            review_state: 'clean',
            review_label: 'Clean',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-sub-optimistic')
    fireEvent.click(within(card).getByRole('button', { name: 'Subscribe' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Save subscription' }))

    await waitFor(() => {
      const updatedCard = screen.getByTestId('report-card-report-sub-optimistic')
      expect(within(updatedCard).getByRole('button', { name: 'Manage Subscription' })).toBeInTheDocument()
    })
  })

  it('uses backend report subscription state to render manage action on cards', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-sub-2',
          report_type: 'battle_card',
          vendor_filter: 'Zendesk',
          category_filter: null,
          executive_summary: 'Already subscribed',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'processing',
          artifact_state: 'processing',
          artifact_label: 'Processing',
          report_subscription: {
            id: 'sub-report-2',
            scope_type: 'report',
            scope_key: 'report-sub-2',
            scope_label: 'battle card - Zendesk',
            enabled: true,
          },
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: null,
          latest_failure_step: null,
          latest_error_summary: null,
          trust: {
            artifact_state: 'processing',
            artifact_label: 'Processing',
            freshness_state: 'fresh',
            freshness_label: 'Fresh',
            review_state: 'clean',
            review_label: 'Clean',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-sub-2')
    fireEvent.click(within(card).getByRole('button', { name: 'Manage Subscription' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-sub-2')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Zendesk')
  })

  it('renders a resume action for paused report subscriptions', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-sub-paused',
          report_type: 'battle_card',
          vendor_filter: 'Notion',
          category_filter: null,
          executive_summary: 'Paused subscription',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'processing',
          artifact_state: 'processing',
          artifact_label: 'Processing',
          report_subscription: {
            id: 'sub-report-paused',
            scope_type: 'report',
            scope_key: 'report-sub-paused',
            scope_label: 'battle card - Notion',
            enabled: false,
          },
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: null,
          latest_failure_step: null,
          latest_error_summary: null,
          trust: {
            artifact_state: 'processing',
            artifact_label: 'Processing',
            freshness_state: 'fresh',
            freshness_label: 'Fresh',
            review_state: 'clean',
            review_label: 'Clean',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports'] },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-sub-paused')
    fireEvent.click(within(card).getByRole('button', { name: 'Resume Subscription' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-sub-paused')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Notion')
  })

  it('hydrates a report-scoped subscription modal from the URL', async () => {
    api.fetchReports.mockResolvedValueOnce({
      reports: [
        {
          id: 'report-sub-url',
          report_type: 'battle_card',
          vendor_filter: 'Asana',
          category_filter: null,
          executive_summary: 'URL-opened subscription',
          created_at: '2026-04-10T00:00:00Z',
          report_date: '2026-04-10',
          status: 'processing',
          artifact_state: 'processing',
          artifact_label: 'Processing',
          report_subscription: {
            id: 'sub-report-url',
            scope_type: 'report',
            scope_key: 'report-sub-url',
            scope_label: 'battle card - Asana',
            enabled: true,
          },
          has_pdf_export: false,
          blocker_count: 0,
          warning_count: 0,
          unresolved_issue_count: 0,
          quality_status: null,
          latest_failure_step: null,
          latest_error_summary: null,
          trust: {
            artifact_state: 'processing',
            artifact_label: 'Processing',
            freshness_state: 'fresh',
            freshness_label: 'Fresh',
            review_state: 'clean',
            review_label: 'Clean',
          },
        },
      ],
      count: 1,
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      { initialEntries: ['/reports?report_subscription=report-sub-url&report_focus_label=battle%20card%20-%20Asana'] },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-sub-url')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Asana')
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'report:report-sub-url:battle card - Asana',
    )
  })

  it('uses report focus query params to keep a deep-linked report subscription target visible', async () => {
    api.fetchReports.mockImplementation(async (params?: Record<string, string | boolean | number | undefined>) => {
      if (params?.report_type === 'battle_card' && params?.vendor_filter === 'Asana') {
        return {
          reports: [
            {
              id: 'report-sub-focus',
              report_type: 'battle_card',
              vendor_filter: 'Asana',
              category_filter: null,
              executive_summary: 'Focused report',
              created_at: '2026-04-10T00:00:00Z',
              report_date: '2026-04-10',
              status: 'processing',
              artifact_state: 'processing',
              artifact_label: 'Processing',
              report_subscription: {
                id: 'sub-report-focus',
                scope_type: 'report',
                scope_key: 'report-sub-focus',
                scope_label: 'battle card - Asana',
                enabled: true,
              },
              has_pdf_export: false,
              blocker_count: 0,
              warning_count: 0,
              unresolved_issue_count: 0,
              quality_status: null,
              latest_failure_step: null,
              latest_error_summary: null,
              trust: {
                artifact_state: 'processing',
                artifact_label: 'Processing',
                freshness_state: 'fresh',
                freshness_label: 'Fresh',
                review_state: 'clean',
                review_label: 'Clean',
              },
            },
          ],
          count: 1,
        }
      }
      return { reports: [], count: 0 }
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?report_subscription=report-sub-focus&report_focus_type=battle_card&report_focus_vendor=Asana&report_focus_label=battle%20card%20-%20Asana',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(api.fetchReports).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Asana',
        quality_status: undefined,
        freshness_state: undefined,
        review_state: undefined,
        include_stale: false,
        limit: 100,
      })
    })

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeKey).toBe('report-sub-focus')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Asana')
  })

  it('opens a deep-linked report subscription modal immediately from report_focus_label before fallback hydration completes', async () => {
    api.fetchReports.mockResolvedValueOnce({ reports: [], count: 0 })
    api.fetchReport.mockImplementationOnce(
      () => new Promise(() => {}),
    )

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?report_subscription=report-sub-label&report_focus_label=battle%20card%20-%20Asana',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeKey).toBe('report-sub-label')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Asana')
  })

  it('falls back to fetching the target report when a deep-linked report subscription is outside the current library slice', async () => {
    api.fetchReports.mockResolvedValueOnce({ reports: [], count: 0 })
    api.fetchReport.mockResolvedValueOnce({
      id: 'report-sub-fallback',
      report_type: 'battle_card',
      vendor_filter: 'Asana',
      category_filter: null,
      executive_summary: 'Fetched report',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      status: 'processing',
      artifact_state: 'processing',
      artifact_label: 'Processing',
      report_subscription: {
        id: 'sub-report-fallback',
        scope_type: 'report',
        scope_key: 'report-sub-fallback',
        scope_label: 'battle card - Asana',
        enabled: true,
      },
      has_pdf_export: false,
      blocker_count: 0,
      warning_count: 0,
      unresolved_issue_count: 0,
      quality_status: null,
      latest_failure_step: null,
      latest_error_summary: null,
      trust: {
        artifact_state: 'processing',
        artifact_label: 'Processing',
        freshness_state: 'fresh',
        freshness_label: 'Fresh',
        review_state: 'clean',
        review_label: 'Clean',
      },
      intelligence_data: {},
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?report_subscription=report-sub-fallback',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(api.fetchReport).toHaveBeenCalledWith('report-sub-fallback')
    })

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeKey).toBe('report-sub-fallback')
    expect(modalState.lastProps.scopeLabel).toBe('battle card - Asana')
    expect(router.state.location.search).toContain('report_focus_type=battle_card')
    expect(router.state.location.search).toContain('report_focus_vendor=Asana')
    expect(router.state.location.search).toContain('report_focus_label=battle+card+-+Asana')
  })

  it('preserves the current library URL when navigating to a generated battle card report', async () => {
    api.requestBattleCardReport.mockResolvedValueOnce({ report_id: 'report-generated-1' })

    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/vendors/:name', element: <div data-testid="vendor-route">Vendor route</div> },
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?report_type=battle_card&vendor_filter=Zendesk&quality_status=sales_ready',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')

    fireEvent.change(screen.getByPlaceholderText('Example: Zendesk'), {
      target: { value: 'Asana' },
    })
    fireEvent.click(screen.getByRole('button', { name: 'Create Battle Card' }))

    await waitFor(() => {
      expect(api.requestBattleCardReport).toHaveBeenCalledWith({ vendor_name: 'Asana' })
    })

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-generated-1')
    expect(router.state.location.search).toBe(
      '?back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk%26quality_status%3Dsales_ready%26composer%3Dbattle_card%26battle_card_vendor%3DAsana',
    )
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/reports?report_type=battle_card&vendor_filter=Zendesk&quality_status=sales_ready&composer=battle_card&battle_card_vendor=Asana"',
    )
  })

  it('normalizes back_to so detail links do not carry transient report subscription params', async () => {
    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?report_type=battle_card&vendor_filter=Zendesk&report_subscription=report-1&report_focus_type=battle_card&report_focus_vendor=Zendesk',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: /summary/i }))

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-1')
    expect(router.state.location.search).toBe(
      '?back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk',
    )
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/reports?report_type=battle_card&vendor_filter=Zendesk"',
    )
  })

  it('preserves vendor back_to when opening report detail from a vendor-scoped library view', async () => {
    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Fvendors%2FZendesk',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    expect(screen.getByRole('button', { name: 'Back to Vendor' })).toBeInTheDocument()

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: /summary/i }))

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-1')
    expect(router.state.location.search).toBe('?back_to=%2Fvendors%2FZendesk')
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/vendors/Zendesk"',
    )
  })


  it('preserves review back_to when opening report detail from a review-scoped library view', async () => {
    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    expect(screen.getByRole('button', { name: 'Back to Review' })).toBeInTheDocument()

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: /summary/i }))

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-1')
    expect(router.state.location.search).toBe('?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1')
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1"',
    )
  })

  it('preserves evidence back_to when opening report detail from an evidence-scoped library view', async () => {
    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    expect(screen.getByRole('button', { name: 'Back to Evidence' })).toBeInTheDocument()

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: /summary/i }))

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-1')
    expect(router.state.location.search).toBe('?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit')
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/evidence?vendor=Zendesk&tab=witnesses&source=reddit"',
    )
  })

  it('preserves opportunities back_to when opening report detail from an opportunity-scoped library view', async () => {
    function ReportDetailRouteProbe() {
      const location = useLocation()
      return (
        <div data-testid="report-detail-route">
          {JSON.stringify(location.state)}
        </div>
      )
    }

    const router = createMemoryRouter(
      [
        { path: '/reports', element: <Reports /> },
        { path: '/reports/:id', element: <ReportDetailRouteProbe /> },
      ],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    expect(screen.getByRole('button', { name: 'Back to Opportunities' })).toBeInTheDocument()

    const card = await screen.findByTestId('report-card-report-1')
    fireEvent.click(within(card).getByRole('button', { name: /summary/i }))

    await waitFor(() => {
      expect(screen.getByTestId('report-detail-route')).toBeInTheDocument()
    })

    expect(router.state.location.pathname).toBe('/reports/report-1')
    expect(router.state.location.search).toBe('?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1')
    expect(screen.getByTestId('report-detail-route')).toHaveTextContent(
      '"backTo":"/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1"',
    )
  })

  it('shows an account-review back label for focused watchlist context', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    expect(screen.getByRole('button', { name: 'Back to Account Review' })).toBeInTheDocument()
  })

  it('shows vendor workspace, evidence, and opportunity shortcuts for an active vendor filter', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
        'href',
        '/vendors/Zendesk?back_to=%2Freports%3Fvendor_filter%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
      )
    })
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freports%3Fvendor_filter%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Freports%3Fvendor_filter%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
  })

  it('hydrates library filters from the URL query string', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?report_type=battle_card&vendor_filter=Zendesk&quality_status=sales_ready&freshness_state=stale&review_state=blocked',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')

    await waitFor(() => {
      expect(api.fetchReports).toHaveBeenLastCalledWith({
        report_type: 'battle_card',
        vendor_filter: 'Zendesk',
        quality_status: 'sales_ready',
        freshness_state: 'stale',
        review_state: 'blocked',
        include_stale: true,
        limit: 100,
      })
    })

    expect(screen.getByPlaceholderText('Filter by vendor...')).toHaveValue('Zendesk')
    const selects = screen.getAllByRole('combobox')
    expect(selects[0]).toHaveValue('battle_card')
    expect(selects[1]).toHaveValue('sales_ready')
    expect(selects[2]).toHaveValue('stale')
    expect(selects[3]).toHaveValue('blocked')

    fireEvent.click(screen.getByRole('button', { name: 'Subscribe to View' }))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('library_view')
    expect(modalState.lastProps.filterPayload).toEqual({
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      quality_status: 'sales_ready',
      freshness_state: 'stale',
      review_state: 'blocked',
    })
  })

  it('hydrates the subscriptions tab and opens the requested subscription editor from the URL', async () => {
    api.listReportSubscriptions.mockResolvedValueOnce({
      subscriptions: [
        {
          id: 'sub-1',
          scope_type: 'library_view',
          scope_key: 'library-view::battle_card::Zendesk::sales_ready',
          scope_label: 'Battle Card • Zendesk • Sales Ready Library',
          filter_payload: {
            report_type: 'battle_card',
            vendor_filter: 'Zendesk',
            quality_status: 'sales_ready',
          },
          delivery_frequency: 'weekly',
          deliverable_focus: 'all',
          freshness_policy: 'any',
          recipient_emails: ['ops@example.com'],
          delivery_note: null,
          enabled: true,
          next_delivery_at: '2026-04-17T00:00:00Z',
          last_delivery_status: 'sent',
          last_delivery_report_count: 2,
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?tab=subscriptions&subscription_id=sub-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Battle Card • Zendesk • Sales Ready Library')).toBeInTheDocument()
    expect(screen.getByText('library_view')).toBeInTheDocument()
    expect(screen.getByText('Type: Battle Card')).toBeInTheDocument()
    expect(screen.getByText('Vendor: Zendesk')).toBeInTheDocument()
    expect(screen.getByText('Quality: Sales Ready')).toBeInTheDocument()

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('library_view')
    expect(modalState.lastProps.scopeKey).toBe('library-view::battle_card::Zendesk::sales_ready')
    expect(modalState.lastProps.scopeLabel).toBe('Battle Card • Zendesk • Sales Ready Library')
    expect(modalState.lastProps.filterPayload).toEqual({
      report_type: 'battle_card',
      vendor_filter: 'Zendesk',
      quality_status: 'sales_ready',
    })
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'library_view:library-view::battle_card::Zendesk::sales_ready:Battle Card • Zendesk • Sales Ready Library',
    )
  })

  it('copies a manage link for a saved subscription from the subscriptions tab', async () => {
    api.listReportSubscriptions.mockResolvedValueOnce({
      subscriptions: [
        {
          id: 'sub-1',
          scope_type: 'library_view',
          scope_key: 'library-view::battle_card::Zendesk::sales_ready',
          scope_label: 'Battle Card • Zendesk • Sales Ready Library',
          filter_payload: {
            report_type: 'battle_card',
            vendor_filter: 'Zendesk',
            quality_status: 'sales_ready',
          },
          delivery_frequency: 'weekly',
          deliverable_focus: 'all',
          freshness_policy: 'any',
          recipient_emails: ['ops@example.com'],
          delivery_note: null,
          enabled: true,
          next_delivery_at: '2026-04-17T00:00:00Z',
          last_delivery_status: 'sent',
          last_delivery_report_count: 2,
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: ['/reports?tab=subscriptions'],
      },
    )

    render(<RouterProvider router={router} />)

    const subscriptionRow = await screen.findByText('Battle Card • Zendesk • Sales Ready Library')
    fireEvent.click(within(subscriptionRow.closest('div[class*="rounded-lg"]') as HTMLElement).getByRole('button', { name: 'Copy Link' }))

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      `${window.location.origin}/reports?tab=subscriptions&subscription_id=sub-1`,
    )
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
    })
  })

  it('hydrates battle-card composer drafts from the URL and keeps them shareable', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports', element: <Reports /> }],
      {
        initialEntries: [
          '/reports?composer=battle_card&battle_card_vendor=Zendesk',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Intelligence Library')

    const input = screen.getByDisplayValue('Zendesk')
    expect(input).toBeInTheDocument()

    fireEvent.change(input, { target: { value: 'Intercom' } })

    await waitFor(() => {
      const params = new URLSearchParams(router.state.location.search)
      expect(params.get('composer')).toBe('battle_card')
      expect(params.get('battle_card_vendor')).toBe('Intercom')
    })
  })
})
