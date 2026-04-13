import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { RouterProvider, createMemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Opportunities from './Opportunities'

const planGate = vi.hoisted(() => ({
  canAccessCampaigns: true,
}))

const api = vi.hoisted(() => ({
  approveCampaign: vi.fn(),
  bulkSetDisposition: vi.fn(),
  downloadCsv: vi.fn(),
  fetchCampaigns: vi.fn(),
  fetchCampaignQualityTrends: vi.fn(),
  fetchCampaignStats: vi.fn(),
  fetchDispositions: vi.fn(),
  fetchHighIntent: vi.fn(),
  generateCampaigns: vi.fn(),
  pushToCrm: vi.fn(),
  removeDispositions: vi.fn(),
  setDisposition: vi.fn(),
  updateCampaign: vi.fn(),
}))

const clipboard = vi.hoisted(() => ({
  writeText: vi.fn(),
}))

vi.mock('../api/client', () => api)
vi.mock('../hooks/usePlanGate', () => ({
  usePlanGate: () => planGate,
}))
vi.mock('../components/CompanyTimeline', () => ({
  default: () => <div>Company Timeline Mock</div>,
}))
vi.mock('../components/SignalEffectivenessPanel', () => ({
  default: () => <div>Signal Effectiveness Mock</div>,
}))

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

describe('Opportunities', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    planGate.canAccessCampaigns = true
    Object.defineProperty(window.navigator, 'clipboard', {
      configurable: true,
      value: clipboard,
    })
    clipboard.writeText.mockResolvedValue(undefined)
    api.fetchHighIntent.mockResolvedValue({ companies: [] })
    api.fetchCampaigns.mockResolvedValue({ campaigns: [] })
    api.fetchCampaignStats.mockResolvedValue({
      total: 4,
      by_status: { draft: 1, approved: 1, queued: 1, sent: 1 },
      by_channel: { email: 4 },
      top_vendors: [],
      quality: {
        pass: 2,
        fail: 1,
        missing: 1,
        blocker_total: 3,
        warning_total: 1,
        by_boundary: {},
        top_blockers: [{ reason: 'missing_roi', count: 2 }],
      },
    })
    api.fetchCampaignQualityTrends.mockResolvedValue({
      days: 14,
      top_n: 5,
      top_blockers: [{ reason: 'missing_roi', count: 2 }],
      series: [
        { day: '2026-04-06', reason: 'missing_roi', count: 1 },
        { day: '2026-04-07', reason: 'missing_roi', count: 2 },
      ],
      totals_by_day: [
        { day: '2026-04-06', blocker_total: 1 },
        { day: '2026-04-07', blocker_total: 2 },
      ],
    })
    api.fetchDispositions.mockResolvedValue({ dispositions: [] })
  })

  it('syncs the vendor filter when the query string changes', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <><Opportunities /><LocationProbe /></> }],
      { initialEntries: ['/opportunities?vendor=Zendesk'] },
    )

    render(<RouterProvider router={router} />)

    const input = await screen.findByPlaceholderText('Filter vendor...')
    expect(input).toHaveValue('Zendesk')

    await router.navigate('/opportunities?vendor=HubSpot')

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('HubSpot')
    })
  })

  it('clears same-route workbench filters without restoring stale query params', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <><Opportunities /><LocationProbe /></> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&min_urgency=7&window_days=30&stage=evaluation&intent=cancel&disposition=saved',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')

    await router.navigate('/opportunities')

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('')
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/opportunities')
    })

    await waitFor(() => {
      expect(api.fetchHighIntent).toHaveBeenLastCalledWith({
        min_urgency: 5,
        vendor_name: undefined,
        window_days: 90,
        limit: 100,
      })
    })
  })

  it('canonicalizes invalid route filters on load', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <><Opportunities /><LocationProbe /></> }],
      {
        initialEntries: [
          '/opportunities?vendor=%20Zendesk%20&min_urgency=99&window_days=999&stage=bogus&intent=cancel&intent=bogus&disposition=wat',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')

    await waitFor(() => {
      expect(api.fetchHighIntent).toHaveBeenLastCalledWith({
        min_urgency: 10,
        vendor_name: 'Zendesk',
        window_days: 90,
        limit: 100,
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/opportunities?vendor=Zendesk&min_urgency=10&intent=cancel')
    })
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('window_days=999')
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('stage=bogus')
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('disposition=wat')
  })

  it('hydrates the workbench filters from the URL', async () => {
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.6,
          buying_stage: 'evaluation',
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          company_size: '201-1000',
          reviewer_title: 'VP Support',
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&min_urgency=8&window_days=30&stage=evaluation&intent=cancel&intent=migration',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    const selects = screen.getAllByRole('combobox')
    expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')
    expect(screen.getByRole('slider')).toHaveValue('8')
    expect(selects[0]).toHaveValue('30')
    expect(selects[1]).toHaveValue('evaluation')
    expect(screen.getByLabelText('cancel')).toBeChecked()
    expect(screen.getByLabelText('migration')).toBeChecked()
    expect(api.fetchHighIntent).toHaveBeenCalledWith({
      limit: 100,
      min_urgency: 8,
      vendor_name: 'Zendesk',
      window_days: 30,
    })
  })

  it('copies the current workbench view link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&min_urgency=8&window_days=30&stage=evaluation&intent=cancel&back_to=%2Fwatchlists%3Fview%3Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findAllByPlaceholderText('Filter vendor...')
    await user.click(screen.getByRole('button', { name: 'Copy View Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/opportunities?vendor=Zendesk&min_urgency=8&window_days=30&stage=evaluation&intent=cancel&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
  })

  it('supports vendor workspace back_to navigation', async () => {
    const user = userEvent.setup()
    const router = createMemoryRouter(
      [
        { path: '/opportunities', element: <Opportunities /> },
        { path: '/vendors/:name', element: <div>Vendor workspace</div> },
      ],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Back to Vendor' })).toHaveAttribute('href', '/vendors/Zendesk')
    expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')

    await user.click(screen.getByRole('link', { name: 'Back to Vendor' }))

    await waitFor(() => {
      expect(screen.getByText('Vendor workspace')).toBeInTheDocument()
    })
  })

  it('supports alerts back_to navigation', async () => {
    const user = userEvent.setup()
    const router = createMemoryRouter(
      [
        { path: '/opportunities', element: <Opportunities /> },
        { path: '/alerts', element: <div>Alerts workspace</div> },
      ],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Falerts%3Fwebhook%3Dwh-1'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Back to Alerts' })).toHaveAttribute('href', '/alerts?webhook=wh-1')
    expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')

    await user.click(screen.getByRole('link', { name: 'Back to Alerts' }))

    await waitFor(() => {
      expect(screen.getByText('Alerts workspace')).toBeInTheDocument()
    })
  })

  it.each([
    {
      name: 'supports evidence back_to navigation',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit',
      routePath: '/evidence',
      destinationText: 'Evidence workspace',
      linkName: 'Back to Evidence',
      expectedHref: '/evidence?vendor=Zendesk&tab=witnesses&source=reddit',
    },
    {
      name: 'supports report detail back_to navigation',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
      routePath: '/reports/:id',
      destinationText: 'Report detail',
      linkName: 'Back to Report',
      expectedHref: '/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    },
    {
      name: 'supports reports back_to navigation',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freports%3Fvendor_filter%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
      routePath: '/reports',
      destinationText: 'Reports workspace',
      linkName: 'Back to Reports',
      expectedHref: '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1',
    },
    {
      name: 'supports review detail back_to navigation',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
      routePath: '/reviews/:id',
      destinationText: 'Review detail',
      linkName: 'Back to Review',
      expectedHref: '/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    },
    {
      name: 'supports vendor list back_to navigation',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6',
      routePath: '/vendors',
      destinationText: 'Vendor list',
      linkName: 'Back to Vendors',
      expectedHref: '/vendors?search=Zendesk&min_urgency=6',
    },
  ])('$name', async ({ initialEntry, routePath, destinationText, linkName, expectedHref }) => {
    const user = userEvent.setup()
    const router = createMemoryRouter(
      [
        { path: '/opportunities', element: <Opportunities /> },
        { path: routePath, element: <div>{destinationText}</div> },
      ],
      { initialEntries: [initialEntry] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: linkName })).toHaveAttribute('href', expectedHref)
    expect(screen.getByPlaceholderText('Filter vendor...')).toHaveValue('Zendesk')

    await user.click(screen.getByRole('link', { name: linkName }))

    await waitFor(() => {
      expect(screen.getByText(destinationText)).toBeInTheDocument()
    })
  })

  it('prefers the exact upstream alerts shortcut for nested opportunity context', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Falerts%253Fwebhook%253Dwh-1%2526days%253D30',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?webhook=wh-1&days=30',
    )
  })

  it('prefers the exact upstream vendor workspace shortcut for nested opportunity context', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fvendors%252FZendesk%253Fback_to%253D%25252Fwatchlists%25253Fview%25253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('prefers the exact upstream evidence shortcut for nested opportunity context', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fevidence%253Fvendor%253DZendesk%2526tab%253Dwitnesses%2526witness_id%253Dwit-1%2526source%253Dreddit%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=wit-1&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('preserves watchlist snapshot dates on generated evidence shortcuts', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-05&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Faccount_vendor%253DZendesk%2526account_company%253DAcme%252BCorp%2526account_report_date%253D2026-04-05%2526account_watch_vendor%253DZendesk%2526account_category%253DHelpdesk%2526account_track_mode%253Dcompetitor',
    )
  })

  it('prefers the exact upstream reports shortcut for nested opportunity context', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Freports%252Freport-1%253Fback_to%253D%25252Fwatchlists%25253Fview%25253Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('shows watchlists, vendor workspace, evidence, report, and alerts shortcuts for the active vendor filter', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&min_urgency=8&window_days=30&stage=evaluation&intent=cancel&back_to=%2Fwatchlists%3Fview%3Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    const vendorInputs = await screen.findAllByPlaceholderText('Filter vendor...')
    expect(vendorInputs[0]).toHaveValue('Zendesk')

    expect(screen.getByText('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1',
    )

    const vendorLink = new URL(screen.getByRole('link', { name: 'Vendor workspace' }).getAttribute('href') || '', 'https://atlas.test')
    const evidenceLink = new URL(screen.getByRole('link', { name: 'Evidence' }).getAttribute('href') || '', 'https://atlas.test')
    const reportsLink = new URL(screen.getByRole('link', { name: 'Reports' }).getAttribute('href') || '', 'https://atlas.test')
    const alertsLink = new URL(screen.getByRole('link', { name: 'Alerts API' }).getAttribute('href') || '', 'https://atlas.test')

    expect(vendorLink.pathname).toBe('/vendors/Zendesk')
    expect(evidenceLink.pathname).toBe('/evidence')
    expect(evidenceLink.searchParams.get('vendor')).toBe('Zendesk')
    expect(reportsLink.pathname).toBe('/reports')
    expect(reportsLink.searchParams.get('vendor_filter')).toBe('Zendesk')
    expect(alertsLink.pathname).toBe('/alerts')
    expect(alertsLink.searchParams.get('vendor')).toBe('Zendesk')

    for (const link of [vendorLink, evidenceLink, reportsLink, alertsLink]) {
      const backTo = new URL(link.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(backTo.pathname).toBe('/opportunities')
      expect(backTo.searchParams.get('vendor')).toBe('Zendesk')
      expect(backTo.searchParams.get('min_urgency')).toBe('8')
      expect(backTo.searchParams.get('window_days')).toBe('30')
      expect(backTo.searchParams.get('stage')).toBe('evaluation')
      expect(backTo.searchParams.getAll('intent')).toEqual(['cancel'])
      expect(backTo.searchParams.get('back_to')).toBe('/watchlists?view=view-1')
    }
  })

  it('copies the resolved header workflow shortcuts for the active vendor filter', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&min_urgency=8&window_days=30&stage=evaluation&intent=cancel&back_to=%2Fwatchlists%3Fview%3Dview-1',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    await screen.findAllByPlaceholderText('Filter vendor...')
    await user.click(screen.getByRole('button', { name: 'Copy Alerts API' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?vendor=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26min_urgency%3D8%26window_days%3D30%26stage%3Devaluation%26intent%3Dcancel%26back_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copied Alerts API' })).toBeInTheDocument()
  })

  it('keeps expanded opportunity links scoped back to the current workbench context', async () => {
    const user = userEvent.setup()
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.7,
          pain: 'support',
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          company_size: '201-1000',
          reviewer_title: 'VP Support',
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()

    await user.click(screen.getByText(/Acme Corp/))

    expect(await screen.findByRole('link', { name: 'View watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(await screen.findByRole('link', { name: 'View vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk',
    )
    expect(screen.getByRole('link', { name: 'Validate evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'View reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'View alerts' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'View full review' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
  })

  it('uses exact account review links for expanded opportunity rows when focus is provided', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.7,
          pain: 'support',
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          account_review_focus: {
            vendor: 'Zendesk',
            company: 'Acme Corp',
            report_date: '2026-04-10',
            watch_vendor: 'Zendesk',
            category: 'Helpdesk',
            track_mode: 'competitor',
          },
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()
    await user.click(screen.getByText(/Acme Corp/))

    const expectedPath = '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-10&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk'
    expect(await screen.findByRole('link', { name: 'View account review' })).toHaveAttribute('href', expectedPath)

    await user.click(screen.getByRole('button', { name: 'Copy account review' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${expectedPath}`)
    })
    expect(screen.getByRole('button', { name: 'Copied account review' })).toBeInTheDocument()
  })

  it('copies expanded opportunity links scoped back to the current workbench context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.7,
          pain: 'support',
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          company_size: '201-1000',
          reviewer_title: 'VP Support',
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()
    await user.click(screen.getByText(/Acme Corp/))
    await user.click(screen.getByRole('button', { name: 'Copy alerts' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copied alerts' })).toBeInTheDocument()
  })

  it('reuses the exact upstream account review shortcut on expanded rows', async () => {
    const user = userEvent.setup()
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.6,
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          account_review_focus: {
            vendor: 'Zendesk',
            company: 'Acme Corp',
            report_date: '2026-04-10',
            watch_vendor: 'Zendesk',
            category: 'Helpdesk',
            track_mode: 'competitor',
          },
        },
      ],
    })

    const exactAccountReviewPath = '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-10&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor'
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          `/opportunities?vendor=Zendesk&back_to=${encodeURIComponent(`/reviews/review-1?back_to=${encodeURIComponent(exactAccountReviewPath)}`)}`,
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()
    await user.click(screen.getByText(/Acme Corp/))

    expect(await screen.findByRole('link', { name: 'View account review' })).toHaveAttribute('href', exactAccountReviewPath)
  })

  it.each([
    {
      name: 'reuses the exact upstream evidence shortcut on expanded rows',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fevidence%253Fvendor%253DZendesk%2526tab%253Dwitnesses%2526source%253Dreddit',
      linkName: 'Validate evidence',
      expectedHref: '/evidence?vendor=Zendesk&tab=witnesses&source=reddit',
    },
    {
      name: 'reuses the exact upstream reports shortcut on expanded rows',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Freports%253Fvendor_filter%253DZendesk%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1',
      linkName: 'View reports',
      expectedHref: '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1',
    },
    {
      name: 'reuses the exact upstream alerts shortcut on expanded rows',
      initialEntry: '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Falerts%253Fvendor%253DZendesk%2526company%253DAcme%252BCorp%2526days%253D30',
      linkName: 'View alerts',
      expectedHref: '/alerts?vendor=Zendesk&company=Acme+Corp&days=30',
    },
  ])('$name', async ({ initialEntry, linkName, expectedHref }) => {
    const user = userEvent.setup()
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.6,
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          company_size: '201-1000',
          reviewer_title: 'VP Support',
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: [initialEntry] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()

    await user.click(screen.getByText(/Acme Corp/))

    expect(await screen.findByRole('link', { name: linkName })).toHaveAttribute('href', expectedHref)
  })

  it('does not reuse broad upstream alerts shortcuts for company rows', async () => {
    const user = userEvent.setup()
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          urgency: 8.6,
          category: 'helpdesk',
          review_id: 'review-1',
          source: 'g2',
          quotes: ['Support has regressed since renewal.'],
          intent_signals: { cancel: true, migration: true, evaluation: false, completed_switch: false },
          alternatives: [{ name: 'Freshdesk' }],
          company_size: '201-1000',
          reviewer_title: 'VP Support',
        },
      ],
    })

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      {
        initialEntries: [
          '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Falerts%253Fwebhook%253Dwh-1%2526days%253D30',
        ],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText(/Acme Corp/)).toBeInTheDocument()
    await user.click(screen.getByText(/Acme Corp/))

    expect(await screen.findByRole('link', { name: 'View alerts' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Freviews%252Freview-1%253Fback_to%253D%25252Falerts%25253Fwebhook%25253Dwh-1%252526days%25253D30',
    )
  })

  it('hides campaign-only analytics when the plan gate is off', async () => {
    planGate.canAccessCampaigns = false

    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findAllByPlaceholderText('Filter vendor...')

    expect(screen.queryByText('Signal Effectiveness Mock')).not.toBeInTheDocument()
    expect(screen.queryByText('Company Timeline Mock')).not.toBeInTheDocument()
    expect(screen.queryByText('Campaign Quality Trends')).not.toBeInTheDocument()
    expect(screen.queryByText('Total Campaigns')).not.toBeInTheDocument()
  })

  it('renders campaign analytics when the plan gate is on', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities'] },
    )

    render(<RouterProvider router={router} />)

    await screen.findAllByPlaceholderText('Filter vendor...')

    expect(screen.getByText('Total Campaigns')).toBeInTheDocument()
    expect(screen.getByText('Drafts Pending')).toBeInTheDocument()
    expect(screen.getByText('Campaign Quality Trends')).toBeInTheDocument()
    expect(api.fetchCampaignStats).toHaveBeenCalledTimes(1)
    expect(api.fetchCampaignQualityTrends).toHaveBeenCalledTimes(1)
  })
})
