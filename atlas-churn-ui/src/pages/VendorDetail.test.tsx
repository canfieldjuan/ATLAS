import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import VendorDetail from './VendorDetail'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  compareVendorPeriods: vi.fn(),
  fetchReports: vi.fn(),
  fetchReviews: vi.fn(),
  fetchVendorHistory: vi.fn(),
  fetchVendorProfile: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('VendorDetail', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchVendorProfile.mockResolvedValue({
      vendor_name: 'Zendesk',
      churn_signal: null,
      review_counts: {
        total: 12,
        pending_enrichment: 1,
        enriched: 11,
      },
      high_intent_companies: [
        {
          company: 'Acme Corp',
          urgency: 8.4,
          pain: 'pricing',
        },
      ],
      pain_distribution: [],
    })
    api.fetchReviews.mockResolvedValue({
      reviews: [],
      total: 0,
      limit: 50,
      offset: 0,
    })
    api.fetchReports.mockResolvedValue({
      reports: [],
      count: 0,
    })
    api.fetchVendorHistory.mockResolvedValue({
      vendor_name: 'Zendesk',
      snapshots: [],
      count: 0,
    })
    api.compareVendorPeriods.mockResolvedValue({
      vendor_name: 'Zendesk',
      period_a: null,
      period_b: null,
      deltas: {},
    })
  })

  it('keeps the page usable when enriched reviews fail', async () => {
    const user = userEvent.setup()
    api.fetchReviews.mockRejectedValue(new Error('API 500: reviews unavailable'))

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByText('Some vendor data is temporarily unavailable.')).toBeInTheDocument()
    expect(
      screen.getByText('Enriched reviews is temporarily unavailable. API 500: reviews unavailable'),
    ).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'reviews' }))

    expect(await screen.findByText('No enriched reviews for this vendor')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'companies' })).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchVendorProfile).toHaveBeenCalledWith('Zendesk')
      expect(api.fetchReviews).toHaveBeenCalledWith({
        vendor_name: 'Zendesk',
        limit: 50,
        window_days: 365,
      })
      expect(api.fetchReports).toHaveBeenCalledWith({
        vendor_filter: 'Zendesk',
        limit: 3,
        include_stale: true,
      })
    })
  })

  it('routes opportunities, evidence, reports, and review drilldown back through the vendor workspace', async () => {
    const user = userEvent.setup()
    api.fetchReviews.mockResolvedValue({
      reviews: [
        {
          id: 'review-1',
          reviewer_company: 'Acme Corp',
          reviewer_title: 'VP Support',
          source: 'g2',
          urgency_score: 8.6,
          pain_category: 'support',
          sentiment_direction: 'declining',
          rating: 2.5,
          intent_to_leave: true,
          decision_maker: true,
          role_level: 'vp',
          buying_stage: 'evaluation',
          competitors_mentioned: [{ name: 'Freshdesk' }],
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    })

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: 'View Opportunities' })[0])
    expect(mockNavigate).toHaveBeenCalledWith('/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk')

    await user.click(screen.getAllByRole('button', { name: 'Validate Evidence' })[0])
    expect(mockNavigate).toHaveBeenCalledWith(
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendors%2FZendesk',
    )

    await user.click(screen.getAllByRole('button', { name: 'View Reports' })[0])
    expect(mockNavigate).toHaveBeenCalledWith('/reports?vendor_filter=Zendesk&back_to=%2Fvendors%2FZendesk')

    await user.click(screen.getByRole('button', { name: 'reviews' }))
    await user.click(screen.getByText('Acme Corp'))
    expect(mockNavigate).toHaveBeenCalledWith('/reviews/review-1?back_to=%2Fvendors%2FZendesk')
  })

  it('surfaces recent reports in the vendor workspace and keeps vendor back_to on drilldown', async () => {
    const user = userEvent.setup()
    api.fetchReports.mockResolvedValue({
      reports: [
        {
          id: 'report-1',
          report_type: 'battle_card',
          executive_summary: 'Zendesk churn is concentrated in enterprise support teams.',
          vendor_filter: 'Zendesk',
          quality_status: 'sales_ready',
          report_date: '2026-04-10T12:00:00Z',
          created_at: '2026-04-10T12:00:00Z',
          status: 'ready',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Recent Reports' })).toBeInTheDocument()
    expect(screen.getByText('battle card')).toBeInTheDocument()
    expect(screen.getByText('sales_ready')).toBeInTheDocument()
    expect(
      screen.getByText('Zendesk churn is concentrated in enterprise support teams.'),
    ).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /battle card/i }))
    expect(mockNavigate).toHaveBeenCalledWith('/reports/report-1?back_to=%2Fvendors%2FZendesk')
  })

  it('returns to the originating watchlist workspace when back_to is present', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Watchlists' }))

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?view=view-1&account_vendor=Zendesk')
  })

  it('surfaces the upstream watchlist path when entered from evidence explorer', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Open Account Review' })[0])

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp')
  })

  it('preserves the upstream evidence path when re-opening validation from the vendor workspace', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Validate Evidence' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk%26account_company%3DAcme%2BCorp',
    )
  })

  it('preserves watchlist account review evidence context from the vendor workspace', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor%26witness_id%3Dwitness%253Azendesk%253A1%26witness_vendor%3DZendesk%26source%3Dreddit']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Validate Evidence' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor%26witness_id%3Dwitness%253Azendesk%253A1%26witness_vendor%3DZendesk%26source%3Dreddit',
    )
  })
})
