import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { RouterProvider, createMemoryRouter } from 'react-router-dom'
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

describe('Opportunities', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    planGate.canAccessCampaigns = true
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
      [{ path: '/opportunities', element: <Opportunities /> }],
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
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Filtered to')).toBeInTheDocument()
    expect(screen.getByText('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
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

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()

    await user.click(screen.getByText('Acme Corp'))

    expect(await screen.findByRole('link', { name: 'View vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'Validate evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'View reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
    )
    expect(screen.getByRole('link', { name: 'View full review' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fvendors%252FZendesk',
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
