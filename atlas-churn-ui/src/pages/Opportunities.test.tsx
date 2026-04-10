import { cleanup, render, screen, waitFor } from '@testing-library/react'
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
