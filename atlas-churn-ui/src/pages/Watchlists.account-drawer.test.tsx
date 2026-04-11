import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Watchlists from './Watchlists'

const clipboard = vi.hoisted(() => ({
  writeText: vi.fn(),
}))

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  addTrackedVendor: vi.fn(),
  createCompetitiveSet: vi.fn(),
  createWatchlistView: vi.fn(),
  deleteCompetitiveSet: vi.fn(),
  deleteWatchlistView: vi.fn(),
  fetchAnnotations: vi.fn(),
  fetchWitness: vi.fn(),
  fetchCompetitiveSetPlan: vi.fn(),
  fetchAccountsInMotionFeed: vi.fn(),
  fetchSlowBurnWatchlist: vi.fn(),
  listCompetitiveSets: vi.fn(),
  listTrackedVendors: vi.fn(),
  listWatchlistViews: vi.fn(),
  removeTrackedVendor: vi.fn(),
  removeAnnotations: vi.fn(),
  runCompetitiveSetNow: vi.fn(),
  searchAvailableVendors: vi.fn(),
  setAnnotation: vi.fn(),
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
    Object.defineProperty(window.navigator, 'clipboard', {
      configurable: true,
      value: clipboard,
    })
    clipboard.writeText.mockResolvedValue(undefined)
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
    api.fetchWitness.mockResolvedValue({
      witness: {
        witness_id: 'witness:zendesk:1',
        excerpt_text: 'Renewal warning from URL hydration.',
        source: 'reddit',
        reviewed_at: '2026-04-03T00:00:00Z',
        reviewer_company: 'Acme Corp',
        reviewer_title: 'VP Support',
        pain_category: 'pricing',
        competitor: 'Freshdesk',
        salience_score: 0.92,
        specificity_score: 0.76,
        selection_reason: 'named_account',
        signal_tags: ['pricing_backlash'],
        review_text: 'Renewal warning from URL hydration.',
        evidence_spans: [],
        all_evidence_span_count: 0,
      },
    })
    api.fetchAnnotations.mockResolvedValue({ annotations: [] })
    api.setAnnotation.mockResolvedValue({
      id: 'ann-1',
      witness_id: 'witness:zendesk:1',
      vendor_name: 'Zendesk',
      annotation_type: 'pin',
      note_text: null,
      created_at: '2026-04-08T12:00:00Z',
      updated_at: '2026-04-08T12:00:00Z',
    })
    api.removeAnnotations.mockResolvedValue({ removed: 1 })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [], count: 0 })
  })

  it('copies a review detail link from the account drawer', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

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
        evidence: ['Renewal warning.'],
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
          review_excerpt: 'Renewal warning.',
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy review link for review-1' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor`,
      )
    })
    expect(await screen.findByText('Copied review link for Acme Corp')).toBeInTheDocument()
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
    expect(within(drawer).getByRole('link', { name: 'Evidence Explorer' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })

  it('opens preserved source reviews with account review back context', async () => {
    const user = userEvent.setup()

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
        evidence: ['Renewal warning.'],
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
          review_excerpt: 'Renewal warning.',
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
      <MemoryRouter initialEntries={['/watchlists']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Open review detail' }))

    expect(mockNavigate).toHaveBeenCalledWith(
      '/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })

  it('opens reports from the account movement drawer with focused account context', async () => {
    const user = userEvent.setup()
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
        evidence: ['High urgency account movement.'],
        alternatives_considering: [{ name: 'Freshdesk', reason: 'pricing' }],
        contract_signal: 'Q3 2026',
        reviewer_title: 'VP Support',
        company_size_raw: '500',
        reasoning_summary: 'Support leaders are actively reviewing Freshdesk after a renewal shock.',
        reasoning_delta: { company_changed: true },
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
          review_excerpt: 'Renewal warning.',
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
      <MemoryRouter initialEntries={['/watchlists']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'View reports' }))

    expect(mockNavigate).toHaveBeenCalledWith(
      '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })

  it('hydrates the witness drawer from URL focus params', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&witness_id=witness%3Azendesk%3A1&witness_vendor=Zendesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByLabelText('Account movement evidence')).toBeInTheDocument()
    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchWitness).toHaveBeenCalledWith('witness:zendesk:1', 'Zendesk', {
        as_of_date: undefined,
        window_days: undefined,
      })
    })
    expect(screen.getByRole('link', { name: 'Open in Evidence Explorer' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor%26witness_id%3Dwitness%253Azendesk%253A1%26witness_vendor%3DZendesk',
    )
  })

  it('copies a share link for the selected account review', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

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
        evidence: ['Renewal warning.'],
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
          review_excerpt: 'Renewal warning.',
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
      <MemoryRouter initialEntries={['/watchlists']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor`,
      )
    })
  })
})
