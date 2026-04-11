import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Watchlists from './Watchlists'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  addTrackedVendor: vi.fn(),
  createCompetitiveSet: vi.fn(),
  createWatchlistView: vi.fn(),
  deleteCompetitiveSet: vi.fn(),
  deleteWatchlistView: vi.fn(),
  fetchCompetitiveSetPlan: vi.fn(),
  fetchAccountsInMotionFeed: vi.fn(),
  fetchSlowBurnWatchlist: vi.fn(),
  listCompetitiveSets: vi.fn(),
  listTrackedVendors: vi.fn(),
  listWatchlistViews: vi.fn(),
  removeTrackedVendor: vi.fn(),
  runCompetitiveSetNow: vi.fn(),
  searchAvailableVendors: vi.fn(),
  updateCompetitiveSet: vi.fn(),
  updateWatchlistView: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Watchlists account drawer', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.listTrackedVendors.mockResolvedValue({
      vendors: [{
        id: 'vendor-1',
        vendor_name: 'Intercom',
        track_mode: 'competitor',
        label: 'Messaging',
        added_at: '2026-04-06T12:00:00Z',
        avg_urgency: 7.4,
        churn_intent_count: 14,
        total_reviews: 220,
        nps_proxy: 21.5,
      }],
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
    api.listWatchlistViews.mockResolvedValue({
      views: [],
      count: 0,
    })
    api.fetchSlowBurnWatchlist.mockResolvedValue({
      signals: [],
      count: 0,
    })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [], count: 0 })
  })

  it('keeps the drawer aligned to the refreshed account row payload', async () => {
    const user = userEvent.setup()

    api.fetchAccountsInMotionFeed
      .mockResolvedValueOnce({
        accounts: [{
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
          evidence: ['Old renewal warning.'],
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
          source_reviews: [{
            id: 'review-1',
            source: 'reddit',
            source_url: 'https://reddit.example/review-1',
            vendor_name: 'Zendesk',
            rating: 2,
            summary: 'Support is slipping',
            review_excerpt: 'Old renewal warning.',
            reviewer_name: 'Taylor',
            reviewer_title: 'VP Support',
            reviewer_company: 'Acme Corp',
            reviewed_at: '2026-04-03T00:00:00Z',
          }],
          evidence_count: 1,
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
        }],
        count: 1,
        tracked_vendor_count: 1,
        vendors_with_accounts: 1,
        min_urgency: 7,
        per_vendor_limit: 10,
        freshest_report_date: '2026-04-05',
      })
      .mockResolvedValueOnce({
        accounts: [{
          company: 'Acme Corp',
          vendor: 'Zendesk',
          watch_vendor: 'Zendesk',
          track_mode: 'competitor',
          watchlist_label: 'Support',
          category: 'Helpdesk',
          urgency: 8.9,
          role_type: 'executive',
          buying_stage: 'evaluation',
          budget_authority: true,
          pain_categories: [{ category: 'pricing', severity: 'high' }],
          evidence: ['Updated renewal warning.'],
          alternatives_considering: [{ name: 'Freshdesk', reason: 'pricing' }],
          contract_signal: 'Q3 2026',
          reviewer_title: 'VP Support',
          company_size_raw: '500',
          quality_flags: [],
          opportunity_score: 85,
          quote_match_type: 'company_match',
          confidence: 7.6,
          reasoning_reference_ids: { witness_ids: ['witness:zendesk:1'] },
          source_distribution: { reddit: 2 },
          source_review_ids: ['review-1'],
          source_reviews: [{
            id: 'review-1',
            source: 'reddit',
            source_url: 'https://reddit.example/review-1',
            vendor_name: 'Zendesk',
            rating: 2,
            summary: 'Support is slipping',
            review_excerpt: 'Updated renewal warning.',
            reviewer_name: 'Taylor',
            reviewer_title: 'VP Support',
            reviewer_company: 'Acme Corp',
            reviewed_at: '2026-04-04T00:00:00Z',
          }],
          evidence_count: 1,
          enriched_at: '2026-04-06T11:00:00Z',
          employee_count: 500,
          industry: 'SaaS',
          annual_revenue: '$10M-$50M',
          domain: 'acme.com',
          contacts: [],
          contact_count: 0,
          report_date: '2026-04-05',
          stale_days: 0,
          is_stale: false,
          data_source: 'persisted_report',
        }],
        count: 1,
        tracked_vendor_count: 1,
        vendors_with_accounts: 1,
        min_urgency: 7,
        per_vendor_limit: 10,
        freshest_report_date: '2026-04-06',
      })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    expect(within(drawer).getByText('Old renewal warning.')).toBeInTheDocument()

    const feedControls = screen.getByRole('group', { name: 'Feed controls' })
    await user.click(within(feedControls).getByLabelText('Fresh only'))

    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_name: undefined,
        category: undefined,
        source: undefined,
        min_urgency: undefined,
        include_stale: false,
      })
    })

    expect(await within(drawer).findByText('Updated renewal warning.')).toBeInTheDocument()
    expect(within(drawer).queryByText('Old renewal warning.')).not.toBeInTheDocument()
  })

  it('hydrates the account drawer from URL focus params', async () => {
    api.fetchAccountsInMotionFeed.mockResolvedValue({
      accounts: [{
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
        evidence: ['Renewal warning from URL hydration.'],
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
        source_reviews: [{
          id: 'review-1',
          source: 'reddit',
          source_url: 'https://reddit.example/review-1',
          vendor_name: 'Zendesk',
          rating: 2,
          summary: 'Support is slipping',
          review_excerpt: 'Renewal warning from URL hydration.',
          reviewer_name: 'Taylor',
          reviewer_title: 'VP Support',
          reviewer_company: 'Acme Corp',
          reviewed_at: '2026-04-03T00:00:00Z',
        }],
        evidence_count: 1,
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
      }],
      count: 1,
      tracked_vendor_count: 1,
      vendors_with_accounts: 1,
      min_urgency: 7,
      per_vendor_limit: 10,
      freshest_report_date: '2026-04-05',
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    expect(within(drawer).getByText('Renewal warning from URL hydration.')).toBeInTheDocument()
  })
})
