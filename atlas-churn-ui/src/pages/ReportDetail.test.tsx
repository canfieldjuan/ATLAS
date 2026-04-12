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
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    })
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

  it('returns to alerts when back_to targets the alerts workspace', async () => {
    const router = createMemoryRouter(
      [
        { path: '/alerts', element: <div data-testid="alerts-route">Alerts route</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Falerts%3Fwebhook%3Dwh-crm',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Back to Alerts' }))

    await waitFor(() => {
      expect(screen.getByTestId('alerts-route')).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe('/alerts')
    expect(router.state.location.search).toBe('?webhook=wh-crm')
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

  it('shows a direct account review shortcut for nested watchlist report detail context', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Faccount_vendor%253DZendesk%2526account_company%253DAcme%252BCorp%2526account_report_date%253D2026-04-05%2526account_watch_vendor%253DZendesk%2526account_category%253DHelpdesk%2526account_track_mode%253Dcompetitor',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor',
    )
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


  it.each([
    {
      name: 'returns to dashboard when back_to targets the dashboard',
      initialEntry: '/reports/report-1?back_to=%2Fdashboard',
      routePath: '/dashboard',
      targetId: 'dashboard-route',
      targetText: 'Dashboard route',
      buttonName: 'Back to Dashboard',
      expectedPath: '/dashboard',
      expectedSearch: '',
    },
    {
      name: 'returns to vendors when back_to targets the vendors list',
      initialEntry: '/reports/report-1?back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6',
      routePath: '/vendors',
      targetId: 'vendors-route',
      targetText: 'Vendors route',
      buttonName: 'Back to Vendors',
      expectedPath: '/vendors',
      expectedSearch: '?search=Zendesk&min_urgency=6',
    },
    {
      name: 'returns to affiliates when back_to targets affiliates',
      initialEntry: '/reports/report-1?back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
      routePath: '/affiliates',
      targetId: 'affiliates-route',
      targetText: 'Affiliates route',
      buttonName: 'Back to Affiliates',
      expectedPath: '/affiliates',
      expectedSearch: '?vendor=Zendesk&min_urgency=7&min_score=80&dm_only=true',
    },
    {
      name: 'returns to vendor targets when back_to targets vendor targets',
      initialEntry: '/reports/report-1?back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
      routePath: '/vendor-targets',
      targetId: 'vendor-targets-route',
      targetText: 'Vendor targets route',
      buttonName: 'Back to Vendor Targets',
      expectedPath: '/vendor-targets',
      expectedSearch: '?search=Zendesk&mode=challenger_intel',
    },
    {
      name: 'returns to briefing review when back_to targets briefing review',
      initialEntry: '/reports/report-1?back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DZendesk',
      routePath: '/briefing-review',
      targetId: 'briefing-review-route',
      targetText: 'Briefing review route',
      buttonName: 'Back to Briefing Review',
      expectedPath: '/briefing-review',
      expectedSearch: '?status=sent&vendor=Zendesk',
    },
    {
      name: 'returns to blog review when back_to targets blog review',
      initialEntry: '/reports/report-1?back_to=%2Fblog-review%3Fstatus%3Dpublished%26draft%3Ddraft-1',
      routePath: '/blog-review',
      targetId: 'blog-review-route',
      targetText: 'Blog review route',
      buttonName: 'Back to Blog Review',
      expectedPath: '/blog-review',
      expectedSearch: '?status=published&draft=draft-1',
    },
    {
      name: 'returns to campaign review when back_to targets campaign review',
      initialEntry: '/reports/report-1?back_to=%2Fcampaign-review%3Fstatus%3Dsent%26company%3DAcme%2BCorp',
      routePath: '/campaign-review',
      targetId: 'campaign-review-route',
      targetText: 'Campaign review route',
      buttonName: 'Back to Campaign Review',
      expectedPath: '/campaign-review',
      expectedSearch: '?status=sent&company=Acme+Corp',
    },
    {
      name: 'returns to challengers when back_to targets challengers',
      initialEntry: '/reports/report-1?back_to=%2Fchallengers%3Fsearch%3DZendesk',
      routePath: '/challengers',
      targetId: 'challengers-route',
      targetText: 'Challengers route',
      buttonName: 'Back to Challengers',
      expectedPath: '/challengers',
      expectedSearch: '?search=Zendesk',
    },
    {
      name: 'returns to prospects when back_to targets prospects',
      initialEntry: '/reports/report-1?back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
      routePath: '/prospects',
      targetId: 'prospects-route',
      targetText: 'Prospects route',
      buttonName: 'Back to Prospects',
      expectedPath: '/prospects',
      expectedSearch: '?company=Acme&status=active&seniority=vp',
    },
    {
      name: 'returns to pipeline review when back_to targets pipeline review',
      initialEntry: '/reports/report-1?back_to=%2Fpipeline-review%3Fqueue_vendor%3DZendesk',
      routePath: '/pipeline-review',
      targetId: 'pipeline-review-route',
      targetText: 'Pipeline review route',
      buttonName: 'Back to Pipeline Review',
      expectedPath: '/pipeline-review',
      expectedSearch: '?queue_vendor=Zendesk',
    },
    {
      name: 'returns to predictor when back_to targets predictor',
      initialEntry: '/reports/report-1?back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
      routePath: '/predictor',
      targetId: 'predictor-route',
      targetText: 'Predictor route',
      buttonName: 'Back to Predictor',
      expectedPath: '/predictor',
      expectedSearch: '?vendor=Zendesk&company_size=smb&industry=fintech',
    },
    {
      name: 'returns to the public report when back_to targets the public report surface',
      initialEntry: '/reports/report-1?back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview',
      routePath: '/report',
      targetId: 'public-report-route',
      targetText: 'Public report route',
      buttonName: 'Back to Report',
      expectedPath: '/report',
      expectedSearch: '?vendor=Zendesk&ref=test-token&mode=view',
    },
  ])('$name', async ({ initialEntry, routePath, targetId, targetText, buttonName, expectedPath, expectedSearch }) => {
    const router = createMemoryRouter(
      [
        { path: routePath, element: <div data-testid={targetId}>{targetText}</div> },
        { path: '/reports/:id', element: <ReportDetail /> },
      ],
      {
        initialEntries: [initialEntry],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: buttonName }))

    await waitFor(() => {
      expect(screen.getByTestId(targetId)).toBeInTheDocument()
    })
    expect(router.state.location.pathname).toBe(expectedPath)
    expect(router.state.location.search).toBe(expectedSearch)
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

  it('copies the direct account review shortcut for nested watchlist report detail context', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Faccount_vendor%253DZendesk%2526account_company%253DAcme%252BCorp%2526account_report_date%253D2026-04-05%2526account_watch_vendor%253DZendesk%2526account_category%253DHelpdesk%2526account_track_mode%253Dcompetitor',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    fireEvent.click(screen.getByRole('button', { name: 'Copy Account Review Link' }))

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      `${window.location.origin}/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor`,
    )
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
    })
  })

  it('prefers the exact upstream evidence shortcut for nested report detail context', async () => {
    const router = createMemoryRouter(
      [{ path: '/reports/:id', element: <ReportDetail /> }],
      {
        initialEntries: [
          '/reports/report-1?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fevidence%253Fvendor%253DZendesk%2526tab%253Dwitnesses%2526witness_id%253Dwit-1%2526source%253Dreddit%2526offset%253D20%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findByRole('heading', { name: 'Zendesk' })
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=wit-1&source=reddit&offset=20&back_to=%2Fwatchlists%3Fview%3Dview-1',
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
      { initialEntries: ['/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findByText('Executive summary')
    fireEvent.click(screen.getAllByRole('button', { name: '[1]' })[0])

    await waitFor(() => {
      expect(drawerState.lastProps?.open).toBe(true)
    })

    expect(drawerState.lastProps.vendorName).toBe('Zendesk')
    expect(drawerState.lastProps.witnessId).toBe('w1')
    expect(drawerState.lastProps.backToPath).toBe('/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1')
    expect(drawerState.lastProps.explorerUrl).toBe(
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=w1&back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
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
