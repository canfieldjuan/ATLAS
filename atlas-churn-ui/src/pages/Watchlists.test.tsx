import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Watchlists from './Watchlists'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  addTrackedVendor: vi.fn(),
  createCompetitiveSet: vi.fn(),
  deleteCompetitiveSet: vi.fn(),
  fetchCompetitiveSetPlan: vi.fn(),
  fetchAccountsInMotionFeed: vi.fn(),
  fetchSlowBurnWatchlist: vi.fn(),
  listTrackedVendors: vi.fn(),
  listCompetitiveSets: vi.fn(),
  removeTrackedVendor: vi.fn(),
  runCompetitiveSetNow: vi.fn(),
  searchAvailableVendors: vi.fn(),
  updateCompetitiveSet: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Watchlists', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.listTrackedVendors.mockResolvedValue({
      vendors: [
        {
          id: 'vendor-1',
          vendor_name: 'Intercom',
          track_mode: 'competitor',
          label: 'Messaging',
          added_at: '2026-04-06T12:00:00Z',
          avg_urgency: 7.4,
          churn_intent_count: 14,
          total_reviews: 220,
          nps_proxy: 21.5,
        },
      ],
      count: 1,
    })
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [],
      count: 0,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })
    api.fetchSlowBurnWatchlist.mockResolvedValue({
      signals: [
        {
          vendor_name: 'Zendesk',
          product_category: 'Helpdesk',
          total_reviews: 300,
          churn_intent_count: 22,
          avg_urgency_score: 8.2,
          avg_rating_normalized: 4.1,
          nps_proxy: 18.2,
          price_complaint_rate: 0.3,
          decision_maker_churn_rate: 0.2,
          support_sentiment: 2.4,
          legacy_support_score: 1.8,
          new_feature_velocity: 0.4,
          employee_growth_rate: 3.2,
          archetype: 'reliability',
          archetype_confidence: 0.81,
          reasoning_mode: 'persisted',
          last_computed_at: '2026-04-07T16:00:00Z',
          synthesis_wedge_label: 'Reliability pressure',
          reasoning_delta: {
            wedge_changed: true,
            confidence_changed: false,
            top_destination_changed: false,
            new_timing_windows: [],
            new_account_signals: ['Acme Corp'],
          },
        },
      ],
      count: 1,
    })
    api.fetchAccountsInMotionFeed.mockResolvedValue({
      accounts: [
        {
          company: 'Acme Corp',
          vendor: 'Zendesk',
          watch_vendor: 'Zendesk',
          track_mode: 'competitor',
          watchlist_label: 'Support',
          category: 'Helpdesk',
          urgency: 8.8,
          role_type: 'executive',
          buying_stage: 'evaluation',
          budget_authority: true,
          pain_categories: [{ category: 'pricing', severity: 'high' }],
          evidence: ['We need to move fast before renewal.'],
          alternatives_considering: [{ name: 'Freshdesk', reason: 'pricing' }],
          contract_signal: 'Q3 2026',
          reviewer_title: 'VP Support',
          company_size_raw: '500',
          quality_flags: [],
          opportunity_score: 84,
          quote_match_type: 'company_match',
          confidence: 7.5,
          reasoning_reference_ids: { witness_ids: ['witness:zendesk:1'] },
          source_distribution: { reddit: 2 },
          source_review_ids: ['review-1'],
          source_reviews: [
            {
              id: 'review-1',
              source: 'reddit',
              source_url: 'https://reddit.example/review-1',
              vendor_name: 'Zendesk',
              rating: 2,
              summary: 'Support is slipping',
              review_excerpt: 'We need to move fast before renewal.',
              reviewer_name: 'Taylor',
              reviewer_title: 'VP Support',
              reviewer_company: 'Acme Corp',
              reviewed_at: '2026-04-03T00:00:00Z',
            },
          ],
          evidence_count: 2,
          enriched_at: '2026-04-06T10:00:00Z',
          employee_count: 500,
          industry: 'SaaS',
          annual_revenue: '$10M-$50M',
          domain: 'acme.com',
          contacts: [],
          contact_count: 0,
          report_date: '2026-04-05',
          stale_days: 2,
          is_stale: true,
          data_source: 'persisted_report',
        },
        {
          company: null,
          vendor: 'Freshdesk',
          watch_vendor: 'Freshdesk',
          track_mode: 'competitor',
          watchlist_label: 'Support',
          category: 'Helpdesk',
          urgency: 6.2,
          role_type: 'manager',
          buying_stage: 'research',
          budget_authority: null,
          pain_categories: [{ category: 'workflow', severity: 'medium' }],
          evidence: ['The team is testing alternatives before committing.'],
          alternatives_considering: [{ name: 'Zendesk' }],
          contract_signal: null,
          reviewer_title: 'Support Manager',
          company_size_raw: '201-500',
          quality_flags: ['anonymous_account_resolution'],
          opportunity_score: 42,
          quote_match_type: 'cluster_match',
          confidence: 2.8,
          reasoning_reference_ids: { witness_ids: ['witness:freshdesk:2'] },
          source_distribution: { reddit: 1 },
          source_review_ids: ['review-2'],
          source_reviews: [
            {
              id: 'review-2',
              source: 'reddit',
              source_url: 'https://reddit.example/review-2',
              vendor_name: 'Freshdesk',
              rating: null,
              summary: 'Testing alternatives',
              review_excerpt: 'The team is testing alternatives before committing.',
              reviewer_name: 'Morgan',
              reviewer_title: 'Support Manager',
              reviewer_company: null,
              reviewed_at: '2026-04-04T00:00:00Z',
            },
          ],
          evidence_count: 1,
          enriched_at: '2026-04-06T11:00:00Z',
          employee_count: null,
          industry: 'SaaS',
          annual_revenue: null,
          domain: null,
          contacts: [],
          contact_count: 0,
          report_date: '2026-04-05',
          stale_days: 2,
          is_stale: true,
          data_source: 'persisted_report',
        },
      ],
      count: 2,
      tracked_vendor_count: 1,
      vendors_with_accounts: 1,
      min_urgency: 7,
      per_vendor_limit: 10,
      freshest_report_date: '2026-04-05',
    })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [], count: 0 })
  })

  it('loads tracked vendors and watchlist feeds, renders freshness, and navigates from rows', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Watchlists' })).toBeInTheDocument()
    await screen.findByText('Acme Corp')

    expect(api.listTrackedVendors).toHaveBeenCalledTimes(1)
    expect(api.listCompetitiveSets).toHaveBeenCalledWith(true)
    expect(api.fetchSlowBurnWatchlist).toHaveBeenCalledTimes(1)
    expect(api.fetchAccountsInMotionFeed).toHaveBeenCalledTimes(1)

    expect(screen.getByRole('heading', { name: 'Vendor Movement Feed' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Tracked Vendors' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Accounts In Motion' })).toBeInTheDocument()
    expect(screen.getByText('1 review-needed cluster')).toBeInTheDocument()
    expect(screen.getByText(/Freshest report/)).toBeInTheDocument()
    expect(screen.getByText('stale report')).toBeInTheDocument()
    expect(screen.getByText('We need to move fast before renewal.')).toBeInTheDocument()
    expect(screen.getByText('higher confidence')).toBeInTheDocument()
    expect(screen.queryByText('Anonymous signal cluster')).not.toBeInTheDocument()

    await user.click(screen.getByText('Messaging'))
    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Intercom')

    mockNavigate.mockClear()
    await user.click(screen.getByText('Acme Corp'))
    expect(mockNavigate).not.toHaveBeenCalled()
    expect(await screen.findByLabelText('Account movement evidence')).toBeInTheDocument()
    expect(screen.getByText('Source reviews')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /source/i })).toHaveAttribute(
      'href',
      'https://reddit.example/review-1',
    )

    await user.click(screen.getAllByRole('button', { name: 'View vendor' })[0])
    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Zendesk')

    mockNavigate.mockClear()
    await user.click(screen.getByRole('button', { name: 'Show 1 cluster' }))
    expect(await screen.findByText('Anonymous signal cluster')).toBeInTheDocument()
    expect(screen.getByText('low confidence')).toBeInTheDocument()
  })

  it('renders empty states when no tracked vendors or persisted accounts exist', async () => {
    api.listTrackedVendors.mockResolvedValue({ vendors: [], count: 0 })
    api.listCompetitiveSets.mockResolvedValue({ competitive_sets: [], count: 0, defaults: null })
    api.fetchSlowBurnWatchlist.mockResolvedValue({ signals: [], count: 0 })
    api.fetchAccountsInMotionFeed.mockResolvedValue({
      accounts: [],
      count: 0,
      tracked_vendor_count: 0,
      vendors_with_accounts: 0,
      min_urgency: 7,
      per_vendor_limit: 10,
      freshest_report_date: null,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await waitFor(() => {
      expect(screen.getByText('No watchlist movement yet. Add tracked vendors to start monitoring.')).toBeInTheDocument()
    })

    expect(screen.getByText('No tracked vendors yet.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add your first vendor' })).toBeInTheDocument()
    expect(screen.getByText('No persisted accounts-in-motion rows yet for your tracked vendors.')).toBeInTheDocument()
    expect(screen.getByText('No persisted account movement yet')).toBeInTheDocument()
  })

  it('applies watchlist filters through the real feed clients', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenCalledTimes(1)
      expect(api.fetchAccountsInMotionFeed).toHaveBeenCalledTimes(1)
    })

    const feedControls = screen.getByRole('group', { name: 'Feed controls' })

    await user.selectOptions(within(feedControls).getByLabelText('Vendor'), 'Intercom')
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({ vendor_name: 'Intercom', category: undefined })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: 'Intercom',
        category: undefined,
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Category'), 'Helpdesk')
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({ vendor_name: 'Intercom', category: 'Helpdesk' })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: 'Intercom',
        category: 'Helpdesk',
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Source'), 'reddit')
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: 'Intercom',
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: undefined,
        include_stale: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Min Urgency'), '8')
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: 'Intercom',
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: undefined,
      })
    })

    await user.click(within(feedControls).getByLabelText('Fresh only'))
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: 'Intercom',
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
      })
    })
  })
})
