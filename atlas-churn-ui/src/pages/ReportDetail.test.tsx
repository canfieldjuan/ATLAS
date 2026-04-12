import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { RouterProvider, createMemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ReportDetail from './ReportDetail'

const modalState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const drawerState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const actionBarState = vi.hoisted(() => ({
  lastProps: null as any,
}))

const api = vi.hoisted(() => ({
  fetchReport: vi.fn(),
}))

vi.mock('../api/client', () => ({
  fetchReport: api.fetchReport,
}))
vi.mock('../components/ReportActionBar', () => ({
  default: (props: any) => {
    actionBarState.lastProps = props
    return <button onClick={props.onSubscribe}>Open subscription</button>
  },
}))
vi.mock('../components/SubscriptionModal', () => ({
  default: (props: any) => {
    modalState.lastProps = props
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
            enabled: false,
          })}
        >
          Save paused subscription
        </button>
      </div>
    )
  },
}))
vi.mock('../components/EvidenceDrawer', () => ({
  default: (props: any) => {
    drawerState.lastProps = props
    return props.open ? <div data-testid="evidence-drawer">{props.vendorName}:{props.witnessId}</div> : null
  },
}))

describe('ReportDetail', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    modalState.lastProps = null
    drawerState.lastProps = null
    actionBarState.lastProps = null
    api.fetchReport.mockResolvedValue({
      id: 'report-1',
      report_type: 'vendor_scorecard',
      vendor_filter: 'Zendesk',
      category_filter: null,
      executive_summary: 'Executive summary',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      llm_model: 'gpt-test',
      status: 'completed',
      blocker_count: 0,
      warning_count: 1,
      unresolved_issue_count: 1,
      quality_status: 'needs_review',
      latest_failure_step: null,
      latest_error_summary: null,
      data_density: {},
      report_subscription: null,
      intelligence_data: {
        reasoning_reference_ids: { witness_ids: ['w1'] },
        reasoning_witness_highlights: [
          {
            witness_id: 'w1',
            reviewer_company: 'Acme',
            excerpt_text: 'Pricing changed overnight',
          },
        ],
        key_insights: [
          { label: 'Pricing friction', summary: 'Pricing created churn risk' },
        ],
        key_insights_reference_ids: { witness_ids: ['w1'] },
        key_insights_witness_highlights: [
          {
            witness_id: 'w1',
            reviewer_company: 'Acme',
            excerpt_text: 'Pricing created churn risk',
          },
        ],
      },
    })
  })

  it('initializes the action bar from backend report subscription state', async () => {
    api.fetchReport.mockResolvedValueOnce({
      id: 'report-subscribed',
      report_type: 'vendor_scorecard',
      vendor_filter: 'Zendesk',
      category_filter: null,
      executive_summary: 'Executive summary',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      llm_model: 'gpt-test',
      status: 'completed',
      blocker_count: 0,
      warning_count: 0,
      unresolved_issue_count: 0,
      quality_status: 'needs_review',
      latest_failure_step: null,
      latest_error_summary: null,
      data_density: {},
      report_subscription: {
        id: 'sub-1',
        scope_type: 'report',
        scope_key: 'report-subscribed',
        scope_label: 'custom report label',
        enabled: true,
      },
      intelligence_data: {},
    })

    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-subscribed'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(actionBarState.lastProps.hasSubscription).toBe(true)
    expect(actionBarState.lastProps.subscriptionState).toBe('active')
  })

  it('initializes the action bar as paused when the backend report subscription is disabled', async () => {
    api.fetchReport.mockResolvedValueOnce({
      id: 'report-paused',
      report_type: 'vendor_scorecard',
      vendor_filter: 'Notion',
      category_filter: null,
      executive_summary: 'Executive summary',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      llm_model: 'gpt-test',
      status: 'completed',
      blocker_count: 0,
      warning_count: 0,
      unresolved_issue_count: 0,
      quality_status: 'needs_review',
      latest_failure_step: null,
      latest_error_summary: null,
      data_density: {},
      report_subscription: {
        id: 'sub-paused',
        scope_type: 'report',
        scope_key: 'report-paused',
        scope_label: 'custom paused label',
        enabled: false,
      },
      intelligence_data: {},
    })

    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-paused'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Notion' })
    expect(actionBarState.lastProps.subscriptionState).toBe('paused')
  })

  it('opens the report subscription modal with the report scope metadata', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByText('Open subscription'))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-1')
    expect(modalState.lastProps.scopeLabel).toBe('vendor scorecard - Zendesk')
    expect(actionBarState.lastProps.hasSubscription).toBe(false)
    expect(actionBarState.lastProps.subscriptionState).toBe('none')
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'report:report-1:vendor scorecard - Zendesk',
    )
  })

  it('hydrates the report subscription modal from the URL', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1?subscription=report'] },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(modalState.lastProps.scopeType).toBe('report')
    expect(modalState.lastProps.scopeKey).toBe('report-1')
    expect(modalState.lastProps.scopeLabel).toBe('vendor scorecard - Zendesk')
    expect(screen.getByTestId('subscription-modal')).toHaveTextContent(
      'report:report-1:vendor scorecard - Zendesk',
    )
  })

  it('keeps the detail action bar paused after saving a paused subscription', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByText('Open subscription'))

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Save paused subscription' }))

    await waitFor(() => {
      expect(actionBarState.lastProps.subscriptionState).toBe('paused')
    })
  })

  it('returns to the preserved library URL from the back button', async () => {
    const router = createMemoryRouter(
      [
        { path: '/reports', element: <div data-testid="reports-route">Reports route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          {
            pathname: '/reports/report-1',
            state: {
              backTo: '/reports?report_type=battle_card&vendor_filter=Zendesk&freshness_state=stale',
            },
          },
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Reports' }))

    await waitFor(() => {
      expect(screen.getByTestId('reports-route')).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe('/reports')
    expect(router.state.location.search).toBe('?report_type=battle_card&vendor_filter=Zendesk&freshness_state=stale')
  })

  it('falls back to the back_to query when detail opens without router state', async () => {
    const router = createMemoryRouter(
      [
        { path: '/reports', element: <div data-testid="reports-route">Reports route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk%26freshness_state%3Dstale',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Reports' }))

    expect(await screen.findByTestId('reports-route')).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/reports')
    expect(router.state.location.search).toBe('?report_type=battle_card&vendor_filter=Zendesk&freshness_state=stale')
  })

  it('returns to the vendor workspace when back_to targets a vendor detail page', async () => {
    const router = createMemoryRouter(
      [
        { path: '/vendors/:name', element: <div data-testid="vendor-route">Vendor route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Fvendors%2FZendesk',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Vendor' }))

    await waitFor(() => {
      expect(screen.getByTestId('vendor-route')).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe('/vendors/Zendesk')
  })


  it('returns to review detail when back_to targets a review path', async () => {
    const router = createMemoryRouter(
      [
        { path: '/reviews/:id', element: <div data-testid="review-route">Review route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Review' }))

    await waitFor(() => {
      expect(screen.getByTestId('review-route')).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe('/reviews/review-1')
    expect(router.state.location.search).toBe('?back_to=%2Fwatchlists%3Fview%3Dview-1')
  })

  it('returns to opportunities when back_to targets the opportunity workbench', async () => {
    const router = createMemoryRouter(
      [
        { path: '/opportunities', element: <div data-testid="opportunities-route">Opportunities route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Opportunities' }))

    await waitFor(() => {
      expect(screen.getByTestId('opportunities-route')).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe('/opportunities')
    expect(router.state.location.search).toBe('?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1')
  })

  it('shows vendor workspace, evidence, and opportunity shortcuts for the report vendor', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
  })

  it('passes a normalized share URL to the detail action bar', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk%26freshness_state%3Dstale',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(actionBarState.lastProps.linkUrl).toBe(
      '/reports/report-1?back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk%26freshness_state%3Dstale',
    )
  })

  it('preserves the open report subscription modal in the copied detail link', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?subscription=report&back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await waitFor(() => {
      expect(modalState.lastProps?.open).toBe(true)
    })

    expect(actionBarState.lastProps.linkUrl).toBe(
      '/reports/report-1?subscription=report&report_focus_label=vendor+scorecard+-+Zendesk&back_to=%2Freports%3Freport_type%3Dbattle_card%26vendor_filter%3DZendesk',
    )
  })

  it('opens the evidence drawer from executive summary citations', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Executive summary')
    fireEvent.click(screen.getAllByRole('button', { name: '[1]' })[0])

    await waitFor(() => {
      expect(drawerState.lastProps?.open).toBe(true)
    })

    expect(drawerState.lastProps.vendorName).toBe('Zendesk')
    expect(drawerState.lastProps.witnessId).toBe('w1')
    expect(screen.getByTestId('evidence-drawer')).toHaveTextContent('Zendesk:w1')
  })

  it('renders partial and thin evidence states for generic report sections', async () => {
    api.fetchReport.mockResolvedValueOnce({
      id: 'report-2',
      report_type: 'exploratory_overview',
      vendor_filter: 'Intercom',
      category_filter: null,
      executive_summary: 'Evidence overview',
      created_at: '2026-04-10T00:00:00Z',
      report_date: '2026-04-10',
      llm_model: 'gpt-test',
      status: 'completed',
      blocker_count: 0,
      warning_count: 0,
      unresolved_issue_count: 0,
      quality_status: 'needs_review',
      latest_failure_step: null,
      latest_error_summary: null,
      data_density: {},
      section_evidence: {
        objection_handlers: {
          state: 'partial',
          label: 'Partial evidence',
          detail: 'Section has evidence metadata, but no linked witness citations yet.',
        },
        recommended_plays: {
          state: 'partial',
          label: 'Partial evidence',
          detail: 'Backend flagged this section for operator review.',
        },
      },
      intelligence_data: {
        objection_handlers: {
          summary: 'Pricing objections are rising',
          reference_ids: {
            metric_ids: ['m1'],
          },
        },
        recommended_plays: {
          summary: 'Lead with migration support',
        },
      },
    })

    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      { initialEntries: ['/reports/report-2'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Intercom' })
    expect(screen.getAllByText('Partial evidence')).toHaveLength(2)
    expect(screen.getByText('Section has evidence metadata, but no linked witness citations yet.')).toBeInTheDocument()
    expect(screen.getByText('Backend flagged this section for operator review.')).toBeInTheDocument()
  })

  it('returns to the evidence workspace when back_to targets evidence explorer', async () => {
    const router = createMemoryRouter(
      [
        { path: '/evidence', element: <div data-testid="evidence-route">Evidence route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Evidence' }))

    expect(await screen.findByTestId('evidence-route')).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/evidence')
    expect(router.state.location.search).toBe('?vendor=Zendesk&tab=witnesses&source=reddit')
  })

  it('returns to the focused account review when back_to targets a watchlist account path', async () => {
    const router = createMemoryRouter(
      [
        { path: '/watchlists', element: <div data-testid="watchlists-route">Watchlists route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Account Review' }))

    expect(await screen.findByTestId('watchlists-route')).toBeInTheDocument()
    expect(router.state.location.pathname).toBe('/watchlists')
    expect(router.state.location.search).toBe('?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor')
  })
})
