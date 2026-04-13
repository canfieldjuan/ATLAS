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
          vendor: 'Zendesk',
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

  it('preserves nested watchlist context in vendor shortcut fallbacks and drilldowns', async () => {
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
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()

    await user.click(screen.getAllByRole('button', { name: 'View Reports' })[0])
    expect(mockNavigate).toHaveBeenCalledWith(
      '/reports?vendor_filter=Zendesk&back_to=%2Fvendors%2FZendesk%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )

    mockNavigate.mockClear()
    await user.click(screen.getByRole('button', { name: /battle card/i }))
    expect(mockNavigate).toHaveBeenCalledWith(
      '/reports/report-1?back_to=%2Fvendors%2FZendesk%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )

    mockNavigate.mockClear()
    await user.click(screen.getByRole('button', { name: 'reviews' }))
    await user.click(screen.getByText('Acme Corp'))
    expect(mockNavigate).toHaveBeenCalledWith(
      '/reviews/review-1?back_to=%2Fvendors%2FZendesk%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
  })

  it('opens an exact account review from vendor high-intent companies when focus is available', async () => {
    const user = userEvent.setup()
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
          vendor: 'Zendesk',
          urgency: 8.4,
          pain: 'pricing',
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
      pain_distribution: [],
    })

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'companies' }))
    await user.click(screen.getByRole('button', { name: 'Account Review' }))

    expect(mockNavigate).toHaveBeenCalledWith(
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-10&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Fvendors%2FZendesk',
    )
  })

  it('reuses the exact upstream account review path from vendor high-intent companies', async () => {
    const user = userEvent.setup()
    const exactAccountReviewPath = '/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-10&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor'
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
          vendor: 'Zendesk',
          urgency: 8.4,
          pain: 'pricing',
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
      pain_distribution: [],
    })

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(exactAccountReviewPath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'companies' }))
    await user.click(screen.getByRole('button', { name: 'Account Review' }))

    expect(mockNavigate).toHaveBeenCalledWith(exactAccountReviewPath)
  })

  it('falls back to vendor-scoped watchlists from vendor high-intent companies without exact focus', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'companies' }))
    await user.click(screen.getByRole('button', { name: 'Watchlists' }))

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?vendor_name=Zendesk&back_to=%2Fvendors%2FZendesk')
  })

  it('returns to alerts when back_to points at an alerts page', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Falerts%3Fwebhook%3Dwh-crm']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Alerts' }))

    expect(mockNavigate).toHaveBeenCalledWith('/alerts?webhook=wh-crm')
  })

  it.each([
    {
      name: 'returns to dashboard when back_to points at dashboard',
      initialEntry: '/vendors/Zendesk?back_to=%2Fdashboard',
      buttonName: 'Back to Dashboard',
      expectedNavigate: '/dashboard',
    },
    {
      name: 'returns to vendors when back_to points at the vendors list',
      initialEntry: '/vendors/Zendesk?back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6',
      buttonName: 'Back to Vendors',
      expectedNavigate: '/vendors?search=Zendesk&min_urgency=6',
    },
    {
      name: 'returns to affiliates when back_to points at affiliates',
      initialEntry: '/vendors/Zendesk?back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
      buttonName: 'Back to Affiliates',
      expectedNavigate: '/affiliates?vendor=Zendesk&min_urgency=7&min_score=80&dm_only=true',
    },
    {
      name: 'returns to vendor targets when back_to points at vendor targets',
      initialEntry: '/vendors/Zendesk?back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
      buttonName: 'Back to Vendor Targets',
      expectedNavigate: '/vendor-targets?search=Zendesk&mode=challenger_intel',
    },
    {
      name: 'returns to briefing review when back_to points at briefing review',
      initialEntry: '/vendors/Zendesk?back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DZendesk',
      buttonName: 'Back to Briefing Review',
      expectedNavigate: '/briefing-review?status=sent&vendor=Zendesk',
    },
    {
      name: 'returns to blog review when back_to points at blog review',
      initialEntry: '/vendors/Zendesk?back_to=%2Fblog-review%3Fstatus%3Dpublished%26draft%3Ddraft-1',
      buttonName: 'Back to Blog Review',
      expectedNavigate: '/blog-review?status=published&draft=draft-1',
    },
    {
      name: 'returns to campaign review when back_to points at campaign review',
      initialEntry: '/vendors/Zendesk?back_to=%2Fcampaign-review%3Fstatus%3Dsent%26company%3DAcme%2BCorp',
      buttonName: 'Back to Campaign Review',
      expectedNavigate: '/campaign-review?status=sent&company=Acme+Corp',
    },
    {
      name: 'returns to challengers when back_to points at challengers',
      initialEntry: '/vendors/Zendesk?back_to=%2Fchallengers%3Fsearch%3DZendesk',
      buttonName: 'Back to Challengers',
      expectedNavigate: '/challengers?search=Zendesk',
    },
    {
      name: 'returns to prospects when back_to points at prospects',
      initialEntry: '/vendors/Zendesk?back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
      buttonName: 'Back to Prospects',
      expectedNavigate: '/prospects?company=Acme&status=active&seniority=vp',
    },
    {
      name: 'returns to pipeline review when back_to points at pipeline review',
      initialEntry: '/vendors/Zendesk?back_to=%2Fpipeline-review%3Fqueue_vendor%3DZendesk',
      buttonName: 'Back to Pipeline Review',
      expectedNavigate: '/pipeline-review?queue_vendor=Zendesk',
    },
    {
      name: 'returns to predictor when back_to points at predictor',
      initialEntry: '/vendors/Zendesk?back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
      buttonName: 'Back to Predictor',
      expectedNavigate: '/predictor?vendor=Zendesk&company_size=smb&industry=fintech',
    },
    {
      name: 'returns to onboarding when back_to points at onboarding',
      initialEntry: '/vendors/Zendesk?back_to=%2Fonboarding%3Fq%3DZendesk',
      buttonName: 'Back to Onboarding',
      expectedNavigate: '/onboarding?q=Zendesk',
    },
  ])('$name', async ({ initialEntry, buttonName, expectedNavigate }) => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: buttonName }))

    expect(mockNavigate).toHaveBeenCalledWith(expectedNavigate)
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

  it('returns to the originating review detail when back_to points at a review page', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Review' }))

    expect(mockNavigate).toHaveBeenCalledWith('/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1')
  })

  it('returns to the originating watchlist workspace when back_to is present', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk%26account_company%3DAcme%2BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Account Review' }))

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp')
  })

  it('copies a vendor detail link with preserved evidence back_to context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26back_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26back_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
  })

  it('copies the evidence shortcut link with preserved watchlist back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy evidence link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
  })

  it('preserves the exact upstream evidence path through nested review detail context', async () => {
    const user = userEvent.setup()
    const directEvidencePath = '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedReviewPath = `/reviews/review-1?back_to=${encodeURIComponent(directEvidencePath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedReviewPath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Validate Evidence' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(directEvidencePath)
  })

  it('copies the exact upstream evidence path through nested review detail context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directEvidencePath = '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedReviewPath = `/reviews/review-1?back_to=${encodeURIComponent(directEvidencePath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedReviewPath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy evidence link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directEvidencePath}`)
    })
  })

  it('copies the upstream watchlists shortcut link through nested review detail context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy watchlists link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp`,
      )
    })
  })

  it('copies the upstream watchlists shortcut link with preserved account-review context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy watchlists link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp`,
      )
    })
  })

  it('copies the reports shortcut link with preserved vendor back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy reports link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reports?vendor_filter=Zendesk&back_to=%2Fvendors%2FZendesk`,
      )
    })
  })

  it('copies the opportunities shortcut link with preserved vendor back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy opportunities link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/opportunities?vendor=Zendesk&back_to=%2Fvendors%2FZendesk`,
      )
    })
  })

  it('copies the alerts shortcut link with preserved vendor back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy alerts link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?back_to=%2Fvendors%2FZendesk`,
      )
    })
  })

  it('preserves the exact upstream alerts path through nested evidence context', async () => {
    const user = userEvent.setup()
    const directAlertsPath = '/alerts?webhook=wh-crm&window=30d'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directAlertsPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Alerts API' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(directAlertsPath)
  })

  it('copies the exact upstream alerts path through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directAlertsPath = '/alerts?webhook=wh-crm&window=30d'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directAlertsPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy alerts link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directAlertsPath}`)
    })
  })

  it('preserves the exact upstream reports path through nested evidence context', async () => {
    const user = userEvent.setup()
    const directReportsPath = '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directReportsPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'View Reports' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(directReportsPath)
  })

  it('copies the exact upstream reports path through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directReportsPath = '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directReportsPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy reports link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directReportsPath}`)
    })
  })

  it('preserves the exact upstream opportunities path through nested evidence context', async () => {
    const user = userEvent.setup()
    const directOpportunitiesPath = '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directOpportunitiesPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'View Opportunities' })[0])

    expect(mockNavigate).toHaveBeenCalledWith(directOpportunitiesPath)
  })

  it('copies the exact upstream opportunities path through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directOpportunitiesPath = '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directOpportunitiesPath)}`

    render(
      <MemoryRouter initialEntries={[`/vendors/Zendesk?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Copy opportunities link' })[0])

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directOpportunitiesPath}`)
    })
  })


  it('surfaces the upstream watchlist path when entered from review detail', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Open Account Review' })[0])

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp')
  })

  it('shows alerts, opportunities, evidence, and reports shortcuts on the vendor surface', async () => {
    render(
      <MemoryRouter initialEntries={['/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/vendors/:name" element={<VendorDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Alerts API' })[0]).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'View Opportunities' })[0]).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Validate Evidence' })[0]).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'View Reports' })[0]).toBeInTheDocument()
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
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&as_of_date=2026-04-05&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor%26witness_id%3Dwitness%253Azendesk%253A1%26witness_vendor%3DZendesk%26source%3Dreddit',
    )
  })
})
