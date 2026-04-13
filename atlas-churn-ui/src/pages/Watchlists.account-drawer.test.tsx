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
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedReviewUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedReviewUrl.pathname).toBe('/reviews/review-1')
    const copiedReviewBackTo = new URL(copiedReviewUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedReviewBackTo.pathname).toBe('/watchlists')
    expect(copiedReviewBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedReviewBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedReviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedReviewBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedReviewBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedReviewBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedReviewBackTo.searchParams.get('account_track_mode')).toBe('competitor')
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
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-05&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })





  it('opens the primary witness from the account drawer header with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Open primary witness detail' }))

    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchWitness).toHaveBeenCalledWith('witness:zendesk:1', 'Zendesk', {
        as_of_date: '2026-04-05',
        window_days: undefined,
      })
    })
  })

  it('copies the primary witness link from the account drawer header with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy primary witness link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedWitnessUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedWitnessUrl.pathname).toBe('/evidence')
    expect(copiedWitnessUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedWitnessUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedWitnessUrl.searchParams.get('witness_id')).toBe('witness:zendesk:1')
    expect(copiedWitnessUrl.searchParams.get('source')).toBe('reddit')
    const copiedWitnessBackTo = new URL(copiedWitnessUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedWitnessBackTo.pathname).toBe('/watchlists')
    expect(copiedWitnessBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedWitnessBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedWitnessBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedWitnessBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedWitnessBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedWitnessBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedWitnessBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('opens the primary review from the account drawer header with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Open primary review detail' }))

    const reviewPath = mockNavigate.mock.calls[mockNavigate.mock.calls.length - 1]?.[0] as string
    const reviewUrl = new URL(reviewPath, window.location.origin)
    expect(reviewUrl.pathname).toBe('/reviews/review-1')
    const reviewBackTo = new URL(reviewUrl.searchParams.get('back_to')!, window.location.origin)
    expect(reviewBackTo.pathname).toBe('/watchlists')
    expect(reviewBackTo.searchParams.get('view')).toBe('view-1')
    expect(reviewBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(reviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(reviewBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(reviewBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(reviewBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(reviewBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('copies the primary review link from the account drawer header with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy primary review link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedReviewUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedReviewUrl.pathname).toBe('/reviews/review-1')
    const copiedReviewBackTo = new URL(copiedReviewUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedReviewBackTo.pathname).toBe('/watchlists')
    expect(copiedReviewBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedReviewBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedReviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedReviewBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedReviewBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedReviewBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedReviewBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('copies the opportunities link from the account drawer with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy opportunities link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedOpportunitiesUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedOpportunitiesUrl.pathname).toBe('/opportunities')
    expect(copiedOpportunitiesUrl.searchParams.get('vendor')).toBe('Zendesk')
    const copiedOpportunitiesBackTo = new URL(copiedOpportunitiesUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedOpportunitiesBackTo.pathname).toBe('/watchlists')
    expect(copiedOpportunitiesBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedOpportunitiesBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('copies the vendor workspace link from the account drawer with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy vendor' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedVendorUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedVendorUrl.pathname).toBe('/vendors/Zendesk')
    const copiedVendorBackTo = new URL(copiedVendorUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedVendorBackTo.pathname).toBe('/watchlists')
    expect(copiedVendorBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedVendorBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedVendorBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedVendorBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedVendorBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedVendorBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedVendorBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('copies the evidence explorer link from the account drawer with saved view context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme%20Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy evidence' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedEvidenceUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedEvidenceUrl.pathname).toBe('/evidence')
    expect(copiedEvidenceUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedEvidenceUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedEvidenceUrl.searchParams.get('as_of_date')).toBe('2026-04-05')
    const copiedEvidenceBackTo = new URL(copiedEvidenceUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedEvidenceBackTo.pathname).toBe('/watchlists')
    expect(copiedEvidenceBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedEvidenceBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedEvidenceBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedEvidenceBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedEvidenceBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedEvidenceBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedEvidenceBackTo.searchParams.get('account_track_mode')).toBe('competitor')
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Open review detail' }))

    const reviewPath = mockNavigate.mock.calls[mockNavigate.mock.calls.length - 1]?.[0] as string
    const reviewUrl = new URL(reviewPath, window.location.origin)
    expect(reviewUrl.pathname).toBe('/reviews/review-1')
    const reviewBackTo = new URL(reviewUrl.searchParams.get('back_to')!, window.location.origin)
    expect(reviewBackTo.pathname).toBe('/watchlists')
    expect(reviewBackTo.searchParams.get('view')).toBe('view-1')
    expect(reviewBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(reviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(reviewBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(reviewBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(reviewBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(reviewBackTo.searchParams.get('account_track_mode')).toBe('competitor')
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'View reports' }))

    const reportsPath = mockNavigate.mock.calls[mockNavigate.mock.calls.length - 1]?.[0] as string
    const reportsUrl = new URL(reportsPath, window.location.origin)
    expect(reportsUrl.pathname).toBe('/reports')
    expect(reportsUrl.searchParams.get('vendor_filter')).toBe('Zendesk')
    const reportsBackTo = new URL(reportsUrl.searchParams.get('back_to')!, window.location.origin)
    expect(reportsBackTo.pathname).toBe('/watchlists')
    expect(reportsBackTo.searchParams.get('view')).toBe('view-1')
    expect(reportsBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(reportsBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(reportsBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(reportsBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(reportsBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(reportsBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('copies a reports link from the account movement drawer with focused account context', async () => {
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
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'Copy reports' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedReportsUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedReportsUrl.pathname).toBe('/reports')
    expect(copiedReportsUrl.searchParams.get('vendor_filter')).toBe('Zendesk')
    const copiedReportsBackTo = new URL(copiedReportsUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedReportsBackTo.pathname).toBe('/watchlists')
    expect(copiedReportsBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedReportsBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedReportsBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedReportsBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedReportsBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedReportsBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedReportsBackTo.searchParams.get('account_track_mode')).toBe('competitor')
  })

  it('links and copies the alerts api path from the account drawer', async () => {
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
    const alertsLink = within(drawer).getByRole('link', { name: 'Alerts API' })
    const alertsUrl = new URL(alertsLink.getAttribute('href')!, window.location.origin)
    expect(alertsUrl.pathname).toBe('/alerts')
    expect(alertsUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(alertsUrl.searchParams.get('company')).toBe('Acme Corp')
    const alertsBackTo = new URL(alertsUrl.searchParams.get('back_to')!, window.location.origin)
    expect(alertsBackTo.pathname).toBe('/watchlists')
    expect(alertsBackTo.searchParams.get('view')).toBe('view-1')
    expect(alertsBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(alertsBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(alertsBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(alertsBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(alertsBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(alertsBackTo.searchParams.get('account_track_mode')).toBe('competitor')

    await user.click(within(drawer).getByRole('button', { name: 'Copy alerts' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedAlertsUrl = new URL(clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string)
    expect(copiedAlertsUrl.pathname).toBe('/alerts')
    expect(copiedAlertsUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedAlertsUrl.searchParams.get('company')).toBe('Acme Corp')
    const copiedAlertsBackTo = new URL(copiedAlertsUrl.searchParams.get('back_to')!, window.location.origin)
    expect(copiedAlertsBackTo.pathname).toBe('/watchlists')
    expect(copiedAlertsBackTo.searchParams.get('view')).toBe('view-1')
    expect(copiedAlertsBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedAlertsBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedAlertsBackTo.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedAlertsBackTo.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedAlertsBackTo.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedAlertsBackTo.searchParams.get('account_track_mode')).toBe('competitor')
    expect(await screen.findByText('Copied Alerts API link for Acme Corp')).toBeInTheDocument()
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
        as_of_date: '2026-04-05',
        window_days: undefined,
      })
    })
    expect(screen.getByRole('link', { name: 'Open in Evidence Explorer' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-05&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor%26witness_id%3Dwitness%253Azendesk%253A1%26witness_vendor%3DZendesk',
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
