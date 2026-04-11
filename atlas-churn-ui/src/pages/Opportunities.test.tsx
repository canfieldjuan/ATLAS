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

  it('shows vendor workspace, evidence, and report shortcuts for the active vendor filter', async () => {
    const router = createMemoryRouter(
      [{ path: '/opportunities', element: <Opportunities /> }],
      { initialEntries: ['/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'] },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByText('Filtered to')).toBeInTheDocument()
    expect(screen.getByText('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1',
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

    expect(await screen.findByRole('link', { name: 'View vendor' })).toHaveAttribute('href', '/vendors/Zendesk')
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
