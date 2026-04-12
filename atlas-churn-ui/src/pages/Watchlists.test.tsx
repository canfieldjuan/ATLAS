import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Watchlists from './Watchlists'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  addTrackedVendor: vi.fn(),
  createCompetitiveSet: vi.fn(),
  deliverWatchlistAlertEmail: vi.fn(),
  evaluateWatchlistAlertEvents: vi.fn(),
  createWatchlistView: vi.fn(),
  deleteCompetitiveSet: vi.fn(),
  deleteWatchlistView: vi.fn(),
  fetchAnnotations: vi.fn(),
  fetchWitness: vi.fn(),
  fetchCompetitiveSetPlan: vi.fn(),
  fetchAccountsInMotionFeed: vi.fn(),
  fetchSlowBurnWatchlist: vi.fn(),
  listWatchlistAlertEmailLog: vi.fn(),
  listWatchlistAlertEvents: vi.fn(),
  listTrackedVendors: vi.fn(),
  listCompetitiveSets: vi.fn(),
  listWatchlistViews: vi.fn(),
  removeTrackedVendor: vi.fn(),
  runCompetitiveSetNow: vi.fn(),
  searchAvailableVendors: vi.fn(),
  setAnnotation: vi.fn(),
  removeAnnotations: vi.fn(),
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

function LocationSearchProbe() {
  const location = useLocation()
  return <div data-testid="location-search">{location.search}</div>
}

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
          last_computed_at: '2026-04-07T15:00:00Z',
          latest_snapshot_date: '2026-04-07',
          latest_accounts_report_date: '2026-04-06',
          freshness_status: 'fresh',
          freshness_reason: null,
          freshness_timestamp: '2026-04-07T15:00:00Z',
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
    api.listWatchlistViews.mockResolvedValue({
      views: [],
      count: 0,
    })
    api.listWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: '',
      watchlist_view_name: '',
      status: 'open',
      events: [],
      count: 0,
    })
    api.listWatchlistAlertEmailLog.mockResolvedValue({
      watchlist_view_id: '',
      watchlist_view_name: '',
      deliveries: [],
      count: 0,
    })
    api.evaluateWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: '',
      watchlist_view_name: '',
      evaluated_at: '2026-04-07T18:00:00Z',
      events: [],
      count: 0,
      new_open_event_count: 0,
      resolved_event_count: 0,
    })
    api.deliverWatchlistAlertEmail.mockResolvedValue({
      watchlist_view_id: '',
      watchlist_view_name: '',
      status: 'sent',
      recipient_emails: ['owner@example.com'],
      event_count: 1,
      message_ids: ['msg-1'],
      summary: 'Delivered watchlist alert email to 1 of 1 recipient',
      error: null,
    })
    api.fetchWitness.mockResolvedValue({
      witness: {
        witness_id: 'witness:zendesk:1',
        excerpt_text: 'We need to move fast before renewal.',
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
        review_text: 'We need to move fast before renewal.',
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
          freshness_status: 'synthesis_pending',
          freshness_reason: 'Reasoning synthesis has not been materialized for this vendor yet',
          freshness_timestamp: '2026-04-07T16:00:00Z',
          synthesis_wedge_label: 'Reliability pressure',
          reasoning_reference_ids: { witness_ids: ['witness:vendor:zendesk:1'] },
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
          freshness_status: 'stale',
          freshness_reason: 'Persisted report is older than the current watchlist window',
          freshness_timestamp: '2026-04-05',
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
          freshness_status: 'stale',
          freshness_reason: 'Persisted report is older than the current watchlist window',
          freshness_timestamp: '2026-04-05',
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
    expect(screen.getAllByText('stale').length).toBeGreaterThan(0)
    expect(screen.getAllByText('fresh').length).toBeGreaterThan(0)
    expect(screen.getByText('synthesis pending')).toBeInTheDocument()
    expect(screen.getByText('We need to move fast before renewal.')).toBeInTheDocument()
    expect(screen.getByText('higher confidence')).toBeInTheDocument()
    expect(screen.queryByText('Anonymous signal cluster')).not.toBeInTheDocument()

    await user.click(screen.getByText('Messaging'))
    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Intercom?back_to=%2Fwatchlists')

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
    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )

    mockNavigate.mockClear()
    await user.click(screen.getByRole('button', { name: 'Show 1 cluster' }))
    expect(await screen.findByText('Anonymous signal cluster')).toBeInTheDocument()
    expect(screen.getByText('low confidence')).toBeInTheDocument()
  })

  it('preserves focused account context when opening review detail from an account row', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open review detail for Zendesk' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })

  it('copies a review detail link directly from an account row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy review link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/reviews/review-1')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(backToUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(backToUrl.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(backToUrl.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(backToUrl.searchParams.get('account_category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('account_track_mode')).toBe('competitor')
    expect(await screen.findByText('Copied review link for Acme Corp')).toBeInTheDocument()
  })

  it('preserves watchlist context when opening opportunities from an account row', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'View opportunities for Zendesk' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('preserves watchlist context when opening vendor workspace from an account row', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    mockNavigate.mockClear()
    await user.click(screen.getAllByRole('button', { name: 'View vendor' })[0])
    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies a vendor workspace link directly from an account row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account vendor link for Zendesk' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(await screen.findByText('Copied vendor link for Acme Corp')).toBeInTheDocument()
  })

  it('copies an opportunity link directly from an account row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account opportunity link for Zendesk' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(await screen.findByText('Copied opportunity link for Acme Corp')).toBeInTheDocument()
  })

  it('preserves watchlist context when opening reports from an account row', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'View reports for Zendesk' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies a reports link directly from an account row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account reports link for Zendesk' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(await screen.findByText('Copied reports link for Acme Corp')).toBeInTheDocument()
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

    await user.selectOptions(within(feedControls).getByLabelText('Vendors'), 'Intercom')
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({ vendor_names: ['Intercom'], category: undefined })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: undefined,
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Category'), 'Helpdesk')
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({ vendor_names: ['Intercom'], category: 'Helpdesk' })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Source'), 'reddit')
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: undefined,
        include_stale: undefined,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })

    await user.selectOptions(within(feedControls).getByLabelText('Min Urgency'), '8')
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: undefined,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })

    await user.click(within(feedControls).getByLabelText('Fresh only'))
    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })

    await user.click(within(feedControls).getByLabelText('Named accounts only'))
    expect(screen.queryByRole('button', { name: 'Show 1 cluster' })).not.toBeInTheDocument()

    await user.click(within(feedControls).getByLabelText('Changed wedges only'))
    expect(screen.queryByText('No vendor movement matches the current filters.')).not.toBeInTheDocument()

    await user.clear(screen.getByLabelText('Vendor alert threshold'))
    await user.type(screen.getByLabelText('Vendor alert threshold'), '7.5')
    await user.clear(screen.getByLabelText('Stale days threshold'))
    await user.type(screen.getByLabelText('Stale days threshold'), '3')
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        vendor_alert_threshold: 7.5,
        stale_days_threshold: 3,
      })
    })

    await user.type(screen.getByLabelText('Saved view name'), 'Intercom high urgency')
    await user.type(screen.getByLabelText('Account alert threshold'), '8.5')
    api.createWatchlistView.mockResolvedValue({
      id: 'view-1',
      name: 'Intercom high urgency',
      vendor_names: ['Intercom'],
      vendor_name: 'Intercom',
      category: 'Helpdesk',
      source: 'reddit',
      min_urgency: 8,
      include_stale: false,
      named_accounts_only: true,
      changed_wedges_only: true,
      vendor_alert_threshold: 7.5,
      account_alert_threshold: 8.5,
      stale_days_threshold: 3,
      alert_email_enabled: false,
      alert_delivery_frequency: 'daily',
      next_alert_delivery_at: null,
      last_alert_delivery_at: null,
      last_alert_delivery_status: null,
      last_alert_delivery_summary: null,
      created_at: null,
      updated_at: null,
    })
    await user.click(screen.getByRole('button', { name: 'Save current view' }))
    await waitFor(() => {
      expect(api.createWatchlistView).toHaveBeenCalledWith({
        name: 'Intercom high urgency',
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
        named_accounts_only: true,
        changed_wedges_only: true,
        vendor_alert_threshold: 7.5,
        account_alert_threshold: 8.5,
        stale_days_threshold: 3,
        alert_email_enabled: false,
        alert_delivery_frequency: 'daily',
      })
    })
  })

  it('applies a saved view and rehydrates the persisted thresholds', async () => {
    const user = userEvent.setup()
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T18:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:30:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.listWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: 'view-1',
      watchlist_view_name: 'Fresh named Intercom',
      status: 'open',
      events: [
        {
          id: 'event-1',
          watchlist_view_id: 'view-1',
          event_type: 'vendor_alert',
          threshold_field: 'vendor_alert_threshold',
          entity_type: 'vendor',
          entity_key: 'vendor_alert:vendor:zendesk',
          vendor_name: 'Zendesk',
          company_name: null,
          category: 'Helpdesk',
          source: null,
          threshold_value: 7.5,
          summary: 'Zendesk crossed the vendor alert threshold at 8.2',
          payload: {},
          reasoning_reference_ids: null,
          source_review_ids: [],
          account_review_focus: null,
          status: 'open',
          first_seen_at: null,
          last_seen_at: '2026-04-07T17:00:00Z',
          resolved_at: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.listWatchlistAlertEmailLog.mockResolvedValue({
      watchlist_view_id: 'view-1',
      watchlist_view_name: 'Fresh named Intercom',
      deliveries: [
        {
          id: 'delivery-1',
          recipient_emails: ['owner@example.com'],
          message_ids: ['msg-1'],
          event_count: 1,
          status: 'sent',
          summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          error: null,
          delivered_at: '2026-04-07T18:30:00Z',
          created_at: '2026-04-07T18:30:00Z',
          updated_at: '2026-04-07T18:30:00Z',
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await waitFor(() => {
      expect(api.listWatchlistViews).toHaveBeenCalledTimes(1)
    })

    await user.click(screen.getAllByRole('button', { name: /Fresh named Intercom/i })[0])
    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        vendor_alert_threshold: 7.5,
        stale_days_threshold: 1,
      })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
        account_alert_threshold: 8.5,
        stale_days_threshold: 1,
      })
    })

    expect(screen.queryByRole('button', { name: 'Show 1 cluster' })).not.toBeInTheDocument()
    expect(screen.getByDisplayValue('7.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('8.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('1')).toBeInTheDocument()
    expect(screen.getByText('Vendor alerts at 7.5+ urgency: 1 hit')).toBeInTheDocument()
    expect(screen.getByText('Account alerts at 8.5+ urgency: 1 hit')).toBeInTheDocument()
    expect(screen.getByText('Stale policy after 1 day: 2 hits')).toBeInTheDocument()
    expect(screen.getByText('Saved View Alert Events')).toBeInTheDocument()
    expect(screen.getByText('Zendesk crossed the vendor alert threshold at 8.2')).toBeInTheDocument()
    expect(screen.getByText('Email delivery log')).toBeInTheDocument()
    expect(screen.getByText('Delivered watchlist alert email to 1 of 1 recipient')).toBeInTheDocument()
    expect(screen.getByText('vendor alert hit')).toBeInTheDocument()
    expect(screen.getByText('account alert hit')).toBeInTheDocument()
    expect(screen.getAllByText('stale policy hit').length).toBeGreaterThan(0)
  })

  it('renders exact shortcuts for persisted account alert events', async () => {
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T18:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:30:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.listWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: 'view-1',
      watchlist_view_name: 'Fresh named Intercom',
      status: 'open',
      events: [
        {
          id: 'event-2',
          watchlist_view_id: 'view-1',
          event_type: 'account_alert',
          threshold_field: 'account_alert_threshold',
          entity_type: 'account',
          entity_key: 'account_alert:account:zendesk:acme corp:helpdesk:reddit:2026-04-07',
          vendor_name: 'Zendesk',
          company_name: 'Acme Corp',
          category: 'Helpdesk',
          source: 'reddit',
          threshold_value: 8.5,
          summary: 'Acme Corp crossed the account alert threshold at 8.9',
          payload: {
            urgency: 8.9,
          },
          reasoning_reference_ids: { witness_ids: ['witness:zendesk:1'] },
          source_review_ids: ['review-1'],
          account_review_focus: {
            vendor: 'Zendesk',
            company: 'Acme Corp',
            report_date: '2026-04-07',
            watch_vendor: 'Intercom',
            category: 'Helpdesk',
            track_mode: 'competitor',
          },
          status: 'open',
          first_seen_at: null,
          last_seen_at: '2026-04-07T17:00:00Z',
          resolved_at: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const accountReviewLink = await screen.findByLabelText('Open alert account review for Acme Corp')
    const reviewLink = screen.getByLabelText('Open alert review detail for Acme Corp')
    const witnessLink = screen.getByLabelText('Open alert witness for Acme Corp')
    const vendorLink = screen.getByLabelText('Open alert vendor workspace for Zendesk')
    const reportsLink = screen.getByLabelText('Open alert reports for Zendesk')
    const opportunitiesLink = screen.getByLabelText('Open alert opportunities for Zendesk')

    const accountReviewUrl = new URL(accountReviewLink.getAttribute('href') || '', 'https://atlas.test')
    expect(accountReviewUrl.pathname).toBe('/watchlists')
    expect(accountReviewUrl.searchParams.get('view')).toBe('view-1')
    expect(accountReviewUrl.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(accountReviewUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(accountReviewUrl.searchParams.get('account_report_date')).toBe('2026-04-07')
    expect(accountReviewUrl.searchParams.get('account_watch_vendor')).toBe('Intercom')
    expect(accountReviewUrl.searchParams.get('account_category')).toBe('Helpdesk')
    expect(accountReviewUrl.searchParams.get('account_track_mode')).toBe('competitor')

    const reviewUrl = new URL(reviewLink.getAttribute('href') || '', 'https://atlas.test')
    expect(reviewUrl.pathname).toBe('/reviews/review-1')
    const reviewBackTo = new URL(reviewUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(reviewBackTo.pathname).toBe('/watchlists')
    expect(reviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(reviewBackTo.searchParams.get('account_report_date')).toBe('2026-04-07')

    const witnessUrl = new URL(witnessLink.getAttribute('href') || '', 'https://atlas.test')
    expect(witnessUrl.pathname).toBe('/evidence')
    expect(witnessUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(witnessUrl.searchParams.get('witness_id')).toBe('witness:zendesk:1')
    expect(witnessUrl.searchParams.get('source')).toBe('reddit')
    const witnessBackTo = new URL(witnessUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(witnessBackTo.pathname).toBe('/watchlists')
    expect(witnessBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(witnessBackTo.searchParams.get('account_report_date')).toBe('2026-04-07')

    const vendorUrl = new URL(vendorLink.getAttribute('href') || '', 'https://atlas.test')
    expect(vendorUrl.pathname).toBe('/vendors/Zendesk')
    const vendorBackTo = new URL(vendorUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(vendorBackTo.pathname).toBe('/watchlists')
    expect(vendorBackTo.searchParams.get('view')).toBe('view-1')
    expect(vendorBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(vendorBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(vendorBackTo.searchParams.get('account_report_date')).toBe('2026-04-07')
    expect(vendorBackTo.searchParams.get('account_watch_vendor')).toBe('Intercom')

    const reportsUrl = new URL(reportsLink.getAttribute('href') || '', 'https://atlas.test')
    expect(reportsUrl.pathname).toBe('/reports')
    expect(reportsUrl.searchParams.get('vendor_filter')).toBe('Zendesk')
    const reportsBackTo = new URL(reportsUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(reportsBackTo.pathname).toBe('/watchlists')
    expect(reportsBackTo.searchParams.get('view')).toBe('view-1')
    expect(reportsBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(reportsBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(reportsBackTo.searchParams.get('account_report_date')).toBe('2026-04-07')
    expect(reportsBackTo.searchParams.get('account_watch_vendor')).toBe('Intercom')

    const opportunitiesUrl = new URL(opportunitiesLink.getAttribute('href') || '', 'https://atlas.test')
    expect(opportunitiesUrl.pathname).toBe('/opportunities')
    expect(opportunitiesUrl.searchParams.get('vendor')).toBe('Zendesk')
    const opportunitiesBackTo = new URL(opportunitiesUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(opportunitiesBackTo.pathname).toBe('/watchlists')
    expect(opportunitiesBackTo.searchParams.get('view')).toBe('view-1')
    expect(opportunitiesBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(opportunitiesBackTo.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(opportunitiesBackTo.searchParams.get('account_report_date')).toBe('2026-04-07')
    expect(opportunitiesBackTo.searchParams.get('account_watch_vendor')).toBe('Intercom')
  })

  it('copies exact saved alert event drillthrough links', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T18:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:30:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.listWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: 'view-1',
      watchlist_view_name: 'Fresh named Intercom',
      status: 'open',
      events: [
        {
          id: 'event-2',
          watchlist_view_id: 'view-1',
          event_type: 'account_alert',
          threshold_field: 'account_alert_threshold',
          entity_type: 'account',
          entity_key: 'account_alert:account:zendesk:acme corp:helpdesk:reddit:2026-04-07',
          vendor_name: 'Zendesk',
          company_name: 'Acme Corp',
          category: 'Helpdesk',
          source: 'reddit',
          threshold_value: 8.5,
          summary: 'Acme Corp crossed the account alert threshold at 8.9',
          payload: { urgency: 8.9 },
          reasoning_reference_ids: { witness_ids: ['witness:zendesk:1'] },
          source_review_ids: ['review-1'],
          account_review_focus: {
            vendor: 'Zendesk',
            company: 'Acme Corp',
            report_date: '2026-04-07',
            watch_vendor: 'Intercom',
            category: 'Helpdesk',
            track_mode: 'competitor',
          },
          status: 'open',
          first_seen_at: null,
          last_seen_at: '2026-04-07T17:00:00Z',
          resolved_at: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('button', { name: 'Copy alert account review link for Acme Corp' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    let copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    let copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/watchlists')
    expect(copiedUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedUrl.searchParams.get('account_report_date')).toBe('2026-04-07')
    expect(await screen.findByText('Copied account review link for Acme Corp')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert review link for Acme Corp' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(2)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/reviews/review-1')
    const reviewBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(reviewBackTo.pathname).toBe('/watchlists')
    expect(reviewBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(await screen.findByText('Copied review link for Acme Corp')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert witness link for Acme Corp' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(3)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('witness_id')).toBe('witness:zendesk:1')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const witnessBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(witnessBackTo.pathname).toBe('/watchlists')
    expect(witnessBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(await screen.findByText('Copied witness link for Acme Corp')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert evidence link for Zendesk' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(4)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const evidenceBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(evidenceBackTo.pathname).toBe('/watchlists')
    expect(evidenceBackTo.searchParams.get('account_company')).toBe('Acme Corp')
    expect(await screen.findByText('Copied evidence link for Zendesk')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert vendor workspace link for Zendesk' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(5)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/vendors/Zendesk')
    let vendorBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(vendorBackTo.pathname).toBe('/watchlists')
    expect(vendorBackTo.searchParams.get('view')).toBe('view-1')
    expect(vendorBackTo.searchParams.get('source')).toBe('reddit')
    expect(vendorBackTo.searchParams.get('category')).toBe('Helpdesk')
    expect(vendorBackTo.searchParams.get('min_urgency')).toBe('8')
    expect(vendorBackTo.searchParams.get('fresh_only')).toBe('true')
    expect(vendorBackTo.searchParams.get('named_accounts_only')).toBe('true')
    expect(vendorBackTo.searchParams.get('changed_wedges_only')).toBe('true')
    expect(vendorBackTo.searchParams.get('vendor_alert_threshold')).toBe('7.5')
    expect(vendorBackTo.searchParams.get('account_alert_threshold')).toBe('8.5')
    expect(vendorBackTo.searchParams.get('stale_days_threshold')).toBe('1')
    expect(await screen.findByText('Copied vendor workspace link for Zendesk')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert reports link for Zendesk' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(6)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/reports')
    expect(copiedUrl.searchParams.get('vendor_filter')).toBe('Zendesk')
    let reportsBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(reportsBackTo.pathname).toBe('/watchlists')
    expect(reportsBackTo.searchParams.get('view')).toBe('view-1')
    expect(reportsBackTo.searchParams.get('source')).toBe('reddit')
    expect(reportsBackTo.searchParams.get('category')).toBe('Helpdesk')
    expect(reportsBackTo.searchParams.get('min_urgency')).toBe('8')
    expect(reportsBackTo.searchParams.get('fresh_only')).toBe('true')
    expect(reportsBackTo.searchParams.get('named_accounts_only')).toBe('true')
    expect(reportsBackTo.searchParams.get('changed_wedges_only')).toBe('true')
    expect(reportsBackTo.searchParams.get('vendor_alert_threshold')).toBe('7.5')
    expect(reportsBackTo.searchParams.get('account_alert_threshold')).toBe('8.5')
    expect(reportsBackTo.searchParams.get('stale_days_threshold')).toBe('1')
    expect(await screen.findByText('Copied reports link for Zendesk')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy alert opportunities link for Zendesk' }))
    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledTimes(7)
    })
    copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/opportunities')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    let opportunitiesBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
    expect(opportunitiesBackTo.pathname).toBe('/watchlists')
    expect(opportunitiesBackTo.searchParams.get('view')).toBe('view-1')
    expect(opportunitiesBackTo.searchParams.get('source')).toBe('reddit')
    expect(opportunitiesBackTo.searchParams.get('category')).toBe('Helpdesk')
    expect(opportunitiesBackTo.searchParams.get('min_urgency')).toBe('8')
    expect(opportunitiesBackTo.searchParams.get('fresh_only')).toBe('true')
    expect(opportunitiesBackTo.searchParams.get('named_accounts_only')).toBe('true')
    expect(opportunitiesBackTo.searchParams.get('changed_wedges_only')).toBe('true')
    expect(opportunitiesBackTo.searchParams.get('vendor_alert_threshold')).toBe('7.5')
    expect(opportunitiesBackTo.searchParams.get('account_alert_threshold')).toBe('8.5')
    expect(opportunitiesBackTo.searchParams.get('stale_days_threshold')).toBe('1')
    expect(await screen.findByText('Copied opportunities link for Zendesk')).toBeInTheDocument()
  })

  it('hydrates a saved view from the URL query', async () => {
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T18:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:30:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        vendor_alert_threshold: 7.5,
        stale_days_threshold: 1,
      })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
        account_alert_threshold: 8.5,
        stale_days_threshold: 1,
      })
    })

    expect(screen.getByDisplayValue('Fresh named Intercom')).toBeInTheDocument()
    expect(screen.getByDisplayValue('7.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('8.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('1')).toBeInTheDocument()
  })

  it('hydrates category from the URL and preserves it in vendor evidence links', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26category%3DHelpdesk',
    )

    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: 'Helpdesk',
        vendor_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: 'Helpdesk',
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })
  })


  it('hydrates min urgency and fresh-only from the URL and preserves them in vendor evidence links', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&min_urgency=8&fresh_only=true']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByDisplayValue('8+')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Fresh only' })).toBeChecked()
    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26min_urgency%3D8%26fresh_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Open vendor witness for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Avendor%3Azendesk%3A1&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26min_urgency%3D8%26fresh_only%3Dtrue',
    )

    await waitFor(() => {
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: undefined,
        source: undefined,
        min_urgency: 8,
        include_stale: false,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })
  })


  it('hydrates named-account and changed-wedge toggles from the URL and preserves them in account drilldowns', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&named_accounts_only=true&changed_wedges_only=true']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: 'Named accounts only' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Changed wedges only' })).toBeChecked()
    expect(screen.queryByRole('link', { name: 'Open primary witness for Freshdesk' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26named_accounts_only%3Dtrue%26changed_wedges_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Open primary witness for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26named_accounts_only%3Dtrue%26changed_wedges_only%3Dtrue%26account_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })


  it('hydrates alert thresholds from the URL and preserves them in vendor evidence links', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&vendor_alert_threshold=7.5&account_alert_threshold=8.5&stale_days_threshold=1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByDisplayValue('7.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('8.5')).toBeInTheDocument()
    expect(screen.getByDisplayValue('1')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Fvendor_name%3DZendesk%26vendor_alert_threshold%3D7.5%26account_alert_threshold%3D8.5%26stale_days_threshold%3D1',
    )

    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: undefined,
        vendor_alert_threshold: 7.5,
        stale_days_threshold: 1,
      })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: undefined,
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
        account_alert_threshold: 8.5,
        stale_days_threshold: 1,
      })
    })
  })

  it('hydrates a vendor-focused watchlist URL and renders an evidence return link', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Back to Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&source=reddit',
    )

    await waitFor(() => {
      expect(api.fetchSlowBurnWatchlist).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: undefined,
        vendor_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
      expect(api.fetchAccountsInMotionFeed).toHaveBeenLastCalledWith({
        vendor_names: ['Zendesk'],
        category: undefined,
        source: undefined,
        min_urgency: undefined,
        include_stale: undefined,
        account_alert_threshold: undefined,
        stale_days_threshold: undefined,
      })
    })
  })

  it('preserves alerts return links on a vendor-focused watchlist URL', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?vendor_name=Zendesk&back_to=%2Falerts%3Fwebhook%3Dwh-1%26days%3D30']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Back to Alerts' })).toHaveAttribute(
      'href',
      '/alerts?webhook=wh-1&days=30',
    )
  })

  it('preserves alerts return links when account review state is opened from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Falerts%3Fwebhook%3Dwh-1%26days%3D30']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Back to Alerts' })).toHaveAttribute(
      'href',
      '/alerts?webhook=wh-1&days=30',
    )
    expect(await screen.findByLabelText('Account movement evidence')).toBeInTheDocument()
  })

  it('adds account drawer focus to the URL when an account row is opened', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <LocationSearchProbe />
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    await screen.findByLabelText('Account movement evidence')

    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_vendor=Zendesk')
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_company=Acme+Corp')
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_report_date=2026-04-05')
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_watch_vendor=Zendesk')
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_category=Helpdesk')
      expect(screen.getByTestId('location-search')).toHaveTextContent('account_track_mode=competitor')
    })
  })

  it('adds witness drawer focus to the URL when a witness is opened from account review', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <LocationSearchProbe />
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    await user.click(screen.getByText('Acme Corp'))
    const drawer = await screen.findByLabelText('Account movement evidence')
    await user.click(within(drawer).getByRole('button', { name: 'witness:zendesk:1' }))
    await screen.findByRole('heading', { name: 'Witness Detail' })

    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('witness_id=witness%3Azendesk%3A1')
      expect(screen.getByTestId('location-search')).toHaveTextContent('witness_vendor=Zendesk')
    })
  })

  it('preserves focused witness drilldown context from an account row', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open primary witness for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })


  it('copies a focused witness drilldown link directly from an account row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy account witness link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('witness_id')).toBe('witness:zendesk:1')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(backToUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(backToUrl.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(backToUrl.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(backToUrl.searchParams.get('account_category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('account_track_mode')).toBe('competitor')
    expect(await screen.findByText('Copied witness link for Acme Corp')).toBeInTheDocument()
  })
  it('links vendor movement rows into focused account review when a named account is available', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open account review for Zendesk' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor',
    )
  })

  it('copies a focused account review link directly from a vendor movement row', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor account review link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/watchlists')
    expect(copiedUrl.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedUrl.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedUrl.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedUrl.searchParams.get('account_track_mode')).toBe('competitor')
    expect(await screen.findByText('Copied account review link for Zendesk')).toBeInTheDocument()
  })

  it('preserves the selected source filter on vendor evidence shortcuts', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1%26source%3Dreddit',
    )
    expect(screen.getByRole('link', { name: 'Open vendor witness for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Avendor%3Azendesk%3A1&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1%26source%3Dreddit',
    )
  })

  it('copies a vendor-scoped evidence link directly from the movement feed', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor evidence link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied evidence link for Zendesk')).toBeInTheDocument()
  })

  it('copies a vendor-scoped opportunities link directly from the movement feed', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor opportunities link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/opportunities')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied opportunities link for Zendesk')).toBeInTheDocument()
  })

  it('copies a vendor-scoped reports link directly from the movement feed', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor reports link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/reports')
    expect(copiedUrl.searchParams.get('vendor_filter')).toBe('Zendesk')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied reports link for Zendesk')).toBeInTheDocument()
  })

  it('copies a focused vendor witness drilldown link directly from the movement feed', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor witness link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('witness_id')).toBe('witness:vendor:zendesk:1')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied witness link for Zendesk')).toBeInTheDocument()
  })

  it('preserves direct vendor witness drilldown from the movement feed', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open vendor witness for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Avendor%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('shows direct evidence explorer links for vendor and account rows', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Corp')).toBeInTheDocument()

    expect(screen.getByRole('link', { name: 'Open vendor evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Open vendor reports for Zendesk' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Open vendor opportunities for Zendesk' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Open account evidence for Zendesk' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor',
    )
  })

  it('shows direct report and opportunity links for tracked vendors', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Messaging')).toBeInTheDocument()

    expect(screen.getByRole('link', { name: 'Open reports for Intercom' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Intercom&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Open opportunities for Intercom' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Intercom&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies a vendor-focused link from the tracked vendor list', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor link for Intercom' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?source=reddit&category=Helpdesk&vendor_name=Intercom`,
      )
    })
    expect(await screen.findByText('Copied vendor link for Intercom')).toBeInTheDocument()
  })

  it('opens evidence explorer directly from a tracked vendor row', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const evidenceLink = await screen.findByRole('link', { name: 'Open evidence for Intercom' })
    const evidenceUrl = new URL(evidenceLink.getAttribute('href')!, window.location.origin)
    expect(evidenceUrl.pathname).toBe('/evidence')
    expect(evidenceUrl.searchParams.get('vendor')).toBe('Intercom')
    expect(evidenceUrl.searchParams.get('tab')).toBe('witnesses')
    expect(evidenceUrl.searchParams.get('source')).toBe('reddit')
    const backTo = evidenceUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
  })

  it('copies a vendor-scoped evidence link from the tracked vendor list', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy evidence link for Intercom' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Intercom')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied evidence link for Intercom')).toBeInTheDocument()
  })

  it('copies a vendor-scoped opportunities link from the tracked vendor list', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy opportunities link for Intercom' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/opportunities')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Intercom')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied opportunities link for Intercom')).toBeInTheDocument()
  })

  it('copies a vendor-scoped reports link from the tracked vendor list', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy reports link for Intercom' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/reports')
    expect(copiedUrl.searchParams.get('vendor_filter')).toBe('Intercom')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('category')).toBe('Helpdesk')
    expect(backToUrl.searchParams.get('source')).toBe('reddit')
    expect(await screen.findByText('Copied reports link for Intercom')).toBeInTheDocument()
  })

  it('shows a header evidence explorer shortcut for single-vendor saved views', async () => {
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Open Current View in Evidence Explorer' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Intercom&tab=witnesses&source=reddit&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('opens the current filtered vendor slice in vendor workspace from the header', async () => {
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Open Current View in Vendor Workspace' })).toHaveAttribute(
      'href',
      '/vendors/Intercom?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies the current vendor workspace shortcut from the header', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByRole('link', { name: 'Open Current View in Vendor Workspace' })
    await user.click(screen.getByRole('button', { name: 'Copy current vendor link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/vendors/Intercom')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('view')).toBe('view-1')
    expect(await screen.findByText('Copied vendor workspace link for Intercom')).toBeInTheDocument()
  })

  it('copies the current evidence explorer shortcut from the header', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByRole('link', { name: 'Open Current View in Evidence Explorer' })
    await user.click(screen.getByRole('button', { name: 'Copy current evidence link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/evidence')
    expect(copiedUrl.searchParams.get('vendor')).toBe('Intercom')
    expect(copiedUrl.searchParams.get('tab')).toBe('witnesses')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    const backTo = copiedUrl.searchParams.get('back_to')
    expect(backTo).toBeTruthy()
    const backToUrl = new URL(backTo!, window.location.origin)
    expect(backToUrl.pathname).toBe('/watchlists')
    expect(backToUrl.searchParams.get('view')).toBe('view-1')
    expect(await screen.findByText('Copied evidence link for Intercom')).toBeInTheDocument()
  })

  it('copies the current live watchlists view from the header', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?source=reddit&category=Helpdesk&min_urgency=8&fresh_only=true']}>
        <Watchlists />
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('button', { name: 'Copy current view link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/watchlists')
    expect(copiedUrl.searchParams.get('source')).toBe('reddit')
    expect(copiedUrl.searchParams.get('category')).toBe('Helpdesk')
    expect(copiedUrl.searchParams.get('min_urgency')).toBe('8')
    expect(copiedUrl.searchParams.get('fresh_only')).toBe('true')
    expect(await screen.findByText('Copied current view link')).toBeInTheDocument()
  })

  it('copies a deep link for a saved view', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T18:00:00Z',
          last_alert_delivery_at: '2026-04-07T18:30:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_summary: 'Delivered watchlist alert email to 1 of 1 recipient',
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })

    render(
      <MemoryRouter initialEntries={['/watchlists']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy link for saved view Fresh named Intercom' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}/watchlists?view=view-1`)
    })
    expect(await screen.findByText('Copied link for Fresh named Intercom')).toBeInTheDocument()
  })


  it('copies a deep link for an account row directly from the feed', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy account link for Acme Corp' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalled()
    })
    const copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
    const copiedUrl = new URL(copiedText)
    expect(copiedUrl.pathname).toBe('/watchlists')
    expect(copiedUrl.searchParams.get('account_vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('account_company')).toBe('Acme Corp')
    expect(copiedUrl.searchParams.get('account_report_date')).toBe('2026-04-05')
    expect(copiedUrl.searchParams.get('account_watch_vendor')).toBe('Zendesk')
    expect(copiedUrl.searchParams.get('account_category')).toBe('Helpdesk')
    expect(copiedUrl.searchParams.get('account_track_mode')).toBe('competitor')
    expect(await screen.findByText('Copied account link for Acme Corp')).toBeInTheDocument()
  })

  it('copies a vendor-focused link from the vendor movement feed without transient drawer state', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1&source=reddit&category=Helpdesk&account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&witness_id=witness%3Azendesk%3A1&witness_vendor=Zendesk']}>
        <Watchlists />
      </MemoryRouter>,
    )

    const copyButton = await screen.findByRole('button', { name: 'Copy vendor link for Zendesk' })
    await user.click(copyButton)

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?source=reddit&category=Helpdesk&vendor_name=Zendesk`,
      )
    })
    expect(await screen.findByText('Copied vendor link for Zendesk')).toBeInTheDocument()
  })

  it('links saved alert events into the alerts api with preserved watchlist context', async () => {
    render(
      <MemoryRouter initialEntries={['/watchlists?view=view-1']}>
        <Watchlists />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('uses an in-app confirmation modal before deleting a saved view', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true)
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_names: ['Intercom'],
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: false,
          vendor_alert_threshold: 8,
          account_alert_threshold: 9,
          stale_days_threshold: 7,
          alert_email_enabled: true,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: '2026-04-08T09:00:00Z',
          last_alert_delivery_status: 'sent',
          last_alert_delivery_at: '2026-04-07T09:00:00Z',
          created_at: '2026-04-07T08:00:00Z',
          updated_at: '2026-04-07T08:30:00Z',
        },
      ],
      count: 1,
    })

    try {
      render(
        <MemoryRouter initialEntries={['/watchlists']}>
          <Watchlists />
        </MemoryRouter>,
      )

      const deleteButton = await screen.findByRole('button', { name: 'Delete saved view Fresh named Intercom' })
      await user.click(deleteButton)

      expect(confirmSpy).not.toHaveBeenCalled()
      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent('Delete saved view Fresh named Intercom?')

      await user.click(within(dialog).getByRole('button', { name: 'Delete saved view' }))

      await waitFor(() => {
        expect(api.deleteWatchlistView).toHaveBeenCalledWith('view-1')
      })
      expect(await screen.findByText('Deleted saved view Fresh named Intercom')).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it('uses an in-app confirmation modal before removing a tracked vendor', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true)

    try {
      render(
        <MemoryRouter>
          <Watchlists />
        </MemoryRouter>,
      )

      const removeButton = await screen.findByRole('button', { name: 'Remove' })
      await user.click(removeButton)

      expect(confirmSpy).not.toHaveBeenCalled()
      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent('Remove Intercom from your watchlists?')

      await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))

      await waitFor(() => {
        expect(api.removeTrackedVendor).not.toHaveBeenCalled()
      })
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it('uses an in-app confirmation modal before deleting a competitive set', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true)
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [
        {
          id: 'set-1',
          name: 'Helpdesk core',
          focal_vendor_name: 'Intercom',
          competitor_vendor_names: ['Zendesk', 'Freshdesk'],
          active: true,
          refresh_mode: 'manual',
          refresh_interval_hours: null,
          vendor_synthesis_enabled: true,
          pairwise_enabled: true,
          category_council_enabled: false,
          asymmetry_enabled: true,
          last_run_at: null,
          last_success_at: null,
          last_run_status: null,
          last_run_summary: {},
          created_at: '2026-04-07T12:00:00Z',
          updated_at: '2026-04-07T12:00:00Z',
        },
      ],
      count: 1,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })

    try {
      render(
        <MemoryRouter>
          <Watchlists />
        </MemoryRouter>,
      )

      const deleteButton = await screen.findByRole('button', { name: 'Delete' })
      await user.click(deleteButton)

      expect(confirmSpy).not.toHaveBeenCalled()
      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent('Delete competitive set Helpdesk core?')

      await user.click(within(dialog).getByRole('button', { name: 'Delete competitive set' }))

      await waitFor(() => {
        expect(api.deleteCompetitiveSet).toHaveBeenCalledWith('set-1')
      })
      expect(await screen.findByText('Deleted competitive set Helpdesk core')).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it('evaluates persisted alert events for the active saved view', async () => {
    const user = userEvent.setup()
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-2',
          name: 'CRM pressure',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: false,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: false,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.evaluateWatchlistAlertEvents.mockResolvedValue({
      watchlist_view_id: 'view-2',
      watchlist_view_name: 'CRM pressure',
      evaluated_at: '2026-04-07T18:00:00Z',
      events: [],
      count: 0,
      new_open_event_count: 2,
      resolved_event_count: 1,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('CRM pressure')
    await user.click(screen.getByText('CRM pressure'))
    await user.click(screen.getByRole('button', { name: 'Evaluate alerts' }))

    await waitFor(() => {
      expect(api.evaluateWatchlistAlertEvents).toHaveBeenCalledWith('view-2')
    })
    expect(await screen.findByText('Evaluated CRM pressure: 2 new open, 1 resolved')).toBeInTheDocument()
  })

  it('emails open alerts for the active saved view', async () => {
    const user = userEvent.setup()
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-3',
          name: 'Support pressure',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: false,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: false,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.deliverWatchlistAlertEmail.mockResolvedValue({
      watchlist_view_id: 'view-3',
      watchlist_view_name: 'Support pressure',
      status: 'sent',
      recipient_emails: ['owner@example.com'],
      event_count: 2,
      message_ids: ['msg-2'],
      summary: 'Delivered watchlist alert email to 1 of 1 recipient',
      error: null,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('Support pressure')
    await user.click(screen.getByText('Support pressure'))
    await user.click(screen.getByRole('button', { name: 'Email open alerts' }))

    await waitFor(() => {
      expect(api.deliverWatchlistAlertEmail).toHaveBeenCalledWith('view-3', {
        evaluate_before_send: true,
      })
    })
    expect(await screen.findByText('Delivered watchlist alert email to 1 of 1 recipient')).toBeInTheDocument()
  })

  it('updates the active saved view when renamed instead of creating a duplicate', async () => {
    const user = userEvent.setup()
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Fresh named Intercom',
          vendor_name: 'Intercom',
          category: 'Helpdesk',
          source: 'reddit',
          min_urgency: 8,
          include_stale: false,
          named_accounts_only: true,
          changed_wedges_only: true,
          vendor_alert_threshold: 7.5,
          account_alert_threshold: 8.5,
          stale_days_threshold: 1,
          alert_email_enabled: false,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 1,
    })
    api.updateWatchlistView.mockResolvedValue({
      id: 'view-1',
      name: 'Exec helpdesk watch',
      vendor_names: ['Intercom'],
      vendor_name: 'Intercom',
      category: 'Helpdesk',
      source: 'reddit',
      min_urgency: 8,
      include_stale: false,
      named_accounts_only: true,
      changed_wedges_only: true,
      vendor_alert_threshold: 7.5,
      account_alert_threshold: 8.5,
      stale_days_threshold: 1,
      alert_email_enabled: false,
      alert_delivery_frequency: 'daily',
      next_alert_delivery_at: null,
      last_alert_delivery_at: null,
      last_alert_delivery_status: null,
      last_alert_delivery_summary: null,
      created_at: null,
      updated_at: null,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await waitFor(() => {
      expect(api.listWatchlistViews).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /Fresh named Intercom/i }).length).toBeGreaterThan(0)
    })
    await user.click(screen.getAllByRole('button', { name: /Fresh named Intercom/i })[0])
    const nameInput = await screen.findByLabelText('Saved view name')
    await user.clear(nameInput)
    await user.type(nameInput, 'Exec helpdesk watch')
    await user.click(screen.getByRole('button', { name: 'Update active view' }))

    await waitFor(() => {
      expect(api.updateWatchlistView).toHaveBeenCalledWith('view-1', {
        name: 'Exec helpdesk watch',
        vendor_names: ['Intercom'],
        category: 'Helpdesk',
        source: 'reddit',
        min_urgency: 8,
        include_stale: false,
        named_accounts_only: true,
        changed_wedges_only: true,
        vendor_alert_threshold: 7.5,
        account_alert_threshold: 8.5,
        stale_days_threshold: 1,
        alert_email_enabled: false,
        alert_delivery_frequency: 'daily',
      })
    })
    expect(api.createWatchlistView).not.toHaveBeenCalled()
  })

  it('sends changed-only and force flags when starting a competitive-set run', async () => {
    const user = userEvent.setup()
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [
        {
          id: 'set-1',
          name: 'Helpdesk core',
          focal_vendor_name: 'Intercom',
          competitor_vendor_names: ['Zendesk', 'Freshdesk'],
          active: true,
          refresh_mode: 'manual',
          refresh_interval_hours: null,
          vendor_synthesis_enabled: true,
          pairwise_enabled: true,
          category_council_enabled: false,
          asymmetry_enabled: true,
          last_run_at: null,
          last_success_at: null,
          last_run_status: null,
          last_run_summary: {},
          created_at: '2026-04-07T12:00:00Z',
          updated_at: '2026-04-07T12:00:00Z',
        },
      ],
      count: 1,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })
    api.fetchCompetitiveSetPlan.mockResolvedValue({
      competitive_set: {
        id: 'set-1',
        name: 'Helpdesk core',
      },
      plan: {
        competitive_set_id: 'set-1',
        focal_vendor_name: 'Intercom',
        vendor_names: ['Intercom', 'Zendesk', 'Freshdesk'],
        pairwise_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        category_names: [],
        asymmetry_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        vendor_synthesis_enabled: true,
        pairwise_enabled: true,
        category_council_enabled: false,
        asymmetry_enabled: true,
        vendor_job_count: 3,
        pairwise_job_count: 2,
        category_job_count: 0,
        asymmetry_job_count: 2,
        estimated_total_jobs: 7,
        estimate: {
          lookback_days: 30,
          vendor_jobs_planned: 3,
          pairwise_jobs_planned: 2,
          category_jobs_planned: 0,
          asymmetry_jobs_planned: 2,
          estimated_vendor_tokens: 1200,
          estimated_cross_vendor_tokens: 2400,
          estimated_total_tokens: 3600,
          estimated_vendor_cost_usd: 0.12,
          estimated_cross_vendor_cost_usd: 0.24,
          estimated_total_cost_usd: 0.36,
          estimated_vendor_tokens_likely_to_reason: 800,
          estimated_vendor_cost_usd_likely_to_reason: 0.08,
          vendor_jobs_with_history: 2,
          vendor_jobs_using_fallback: 1,
          cross_vendor_jobs_with_history: 3,
          cross_vendor_jobs_using_fallback: 1,
          vendor_jobs_with_matching_pools: 2,
          vendor_jobs_missing_pools: 1,
          vendor_jobs_likely_to_reason: 2,
          vendor_jobs_likely_hash_reuse: 1,
          vendor_jobs_likely_stale_reuse: 0,
          vendor_jobs_likely_missing_prior: 0,
          vendor_jobs_likely_hash_changed: 1,
          vendor_jobs_likely_prior_quality_weak: 0,
          vendor_jobs_likely_missing_packet_artifacts: 0,
          vendor_jobs_likely_missing_reference_ids: 0,
          likely_rerun_vendors: ['Intercom:changed'],
          likely_reuse_vendors: ['Zendesk:reuse'],
          recent_vendor_sample_count: 3,
          recent_cross_vendor_sample_count: 2,
          note: 'Estimated from recent runs.',
        },
      },
      recent_runs: [],
    })
    api.runCompetitiveSetNow.mockResolvedValue({
      execution_id: 'exec-1',
      status: 'started',
      message: 'queued',
      competitive_set_id: 'set-1',
      plan: {
        competitive_set_id: 'set-1',
      },
    })
    api.fetchCompetitiveSetPlan.mockResolvedValueOnce({
      competitive_set: {
        id: 'set-1',
        name: 'Helpdesk core',
      },
      plan: {
        competitive_set_id: 'set-1',
        focal_vendor_name: 'Intercom',
        vendor_names: ['Intercom', 'Zendesk', 'Freshdesk'],
        pairwise_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        category_names: [],
        asymmetry_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        vendor_synthesis_enabled: true,
        pairwise_enabled: true,
        category_council_enabled: false,
        asymmetry_enabled: true,
        vendor_job_count: 3,
        pairwise_job_count: 2,
        category_job_count: 0,
        asymmetry_job_count: 2,
        estimated_total_jobs: 7,
        estimate: {
          lookback_days: 30,
          vendor_jobs_planned: 3,
          pairwise_jobs_planned: 2,
          category_jobs_planned: 0,
          asymmetry_jobs_planned: 2,
          estimated_vendor_tokens: 1200,
          estimated_cross_vendor_tokens: 2400,
          estimated_total_tokens: 3600,
          estimated_vendor_cost_usd: 0.12,
          estimated_cross_vendor_cost_usd: 0.24,
          estimated_total_cost_usd: 0.36,
          estimated_vendor_tokens_likely_to_reason: 800,
          estimated_vendor_cost_usd_likely_to_reason: 0.08,
          vendor_jobs_with_history: 2,
          vendor_jobs_using_fallback: 1,
          cross_vendor_jobs_with_history: 3,
          cross_vendor_jobs_using_fallback: 1,
          vendor_jobs_with_matching_pools: 2,
          vendor_jobs_missing_pools: 1,
          vendor_jobs_likely_to_reason: 2,
          vendor_jobs_likely_hash_reuse: 1,
          vendor_jobs_likely_stale_reuse: 0,
          vendor_jobs_likely_missing_prior: 0,
          vendor_jobs_likely_hash_changed: 1,
          vendor_jobs_likely_prior_quality_weak: 0,
          vendor_jobs_likely_missing_packet_artifacts: 0,
          vendor_jobs_likely_missing_reference_ids: 0,
          likely_rerun_vendors: ['Intercom:changed'],
          likely_reuse_vendors: ['Zendesk:reuse'],
          recent_vendor_sample_count: 3,
          recent_cross_vendor_sample_count: 2,
          note: 'Estimated from recent runs.',
        },
      },
      recent_runs: [
        {
          id: 'run-1',
          competitive_set_id: 'set-1',
          account_id: 'acct-1',
          run_id: 'scope-run-1',
          trigger: 'manual',
          status: 'succeeded',
          execution_id: 'exec-prev',
          summary: {
            vendors_reasoned: 2,
            cross_vendor_succeeded: 1,
            vendors_skipped_hash_reuse: 1,
            vendors_failed: 0,
            total_tokens: 1234,
            force: true,
            force_cross_vendor: true,
            changed_vendors_only: false,
          },
          started_at: '2026-04-07T12:00:00Z',
          completed_at: '2026-04-07T12:01:00Z',
          created_at: '2026-04-07T12:00:00Z',
        },
        {
          id: 'run-2',
          competitive_set_id: 'set-1',
          account_id: 'acct-1',
          run_id: 'scope-run-2',
          trigger: 'manual',
          status: 'failed',
          execution_id: 'exec-skip',
          summary: {
            vendors_reasoned: 0,
            cross_vendor_failed: 1,
            vendors_skipped_hash_reuse: 0,
            vendors_failed: 0,
            total_tokens: 0,
            force: true,
            force_cross_vendor: true,
            changed_vendors_only: true,
            _skip_synthesis: 'No pool data available',
          },
          started_at: '2026-04-07T12:05:00Z',
          completed_at: '2026-04-07T12:05:30Z',
          created_at: '2026-04-07T12:05:00Z',
        },
        {
          id: 'run-3',
          competitive_set_id: 'set-1',
          account_id: 'acct-1',
          run_id: 'scope-run-3',
          trigger: 'manual',
          status: 'running',
          execution_id: 'exec-running',
          summary: {
            vendors_reasoned: 0,
            vendors_skipped_hash_reuse: 0,
            vendors_failed: 0,
            total_tokens: 0,
            force: false,
            force_cross_vendor: true,
            changed_vendors_only: true,
          },
          started_at: '2026-04-07T12:06:00Z',
          completed_at: null,
          created_at: '2026-04-07T12:06:00Z',
        },
      ],
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('Helpdesk core')
    await user.click(screen.getByRole('button', { name: 'Preview cost' }))

    await waitFor(() => {
      expect(api.fetchCompetitiveSetPlan).toHaveBeenCalledWith('set-1')
    })
    expect(screen.getAllByText('vendor forced')).toHaveLength(2)
    expect(screen.getAllByText('cross-vendor forced')).toHaveLength(3)
    expect(screen.getAllByText('changed only')).toHaveLength(2)
    expect(screen.getByText('3 reasoned')).toBeInTheDocument()
    expect(screen.getByText('1 failed')).toBeInTheDocument()
    expect(screen.getByText('running')).toBeInTheDocument()
    expect(screen.getByText('No pool data available')).toBeInTheDocument()

    await user.click(screen.getByLabelText('Run changed vendors only'))
    await user.click(screen.getByLabelText('Force vendor rerun'))
    await user.click(screen.getByLabelText('Force cross-vendor synthesis'))
    await user.click(screen.getByRole('button', { name: 'Run now' }))

    await waitFor(() => {
      expect(api.runCompetitiveSetNow).toHaveBeenCalledWith('set-1', {
        changed_vendors_only: false,
        force: true,
        force_cross_vendor: true,
      })
    })

    expect(screen.getAllByText('running').length).toBeGreaterThan(0)
    expect(screen.getAllByText('vendor forced').length).toBeGreaterThanOrEqual(3)
    expect(screen.getAllByText('cross-vendor forced').length).toBeGreaterThanOrEqual(4)

    expect(await screen.findByText('Helpdesk core refresh started (exec-1)')).toBeInTheDocument()
  })

  it('shows last scoped-run markers on the competitive-set card without opening preview', async () => {
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [
        {
          id: 'set-1',
          name: 'Helpdesk core',
          focal_vendor_name: 'Intercom',
          competitor_vendor_names: ['Zendesk', 'Freshdesk'],
          active: true,
          refresh_mode: 'manual',
          refresh_interval_hours: null,
          vendor_synthesis_enabled: true,
          pairwise_enabled: true,
          category_council_enabled: false,
          asymmetry_enabled: true,
          last_run_at: '2026-04-07T12:05:00Z',
          last_success_at: null,
          last_run_status: 'failed',
          last_run_summary: {
            changed_vendors_only: true,
            force_cross_vendor: true,
            _skip_synthesis: 'No pool data available',
          },
          created_at: '2026-04-07T12:00:00Z',
          updated_at: '2026-04-07T12:05:30Z',
        },
      ],
      count: 1,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('Helpdesk core')
    expect(screen.getAllByText('changed only').length).toBeGreaterThan(0)
    expect(screen.getAllByText('cross-vendor forced').length).toBeGreaterThan(0)
    expect(screen.getAllByText('No pool data available').length).toBeGreaterThan(0)
  })

  it('shows an optimistic running state on the card immediately after starting a competitive-set run', async () => {
    const user = userEvent.setup()
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [
        {
          id: 'set-1',
          name: 'Helpdesk core',
          focal_vendor_name: 'Intercom',
          competitor_vendor_names: ['Zendesk', 'Freshdesk'],
          active: true,
          refresh_mode: 'manual',
          refresh_interval_hours: null,
          vendor_synthesis_enabled: true,
          pairwise_enabled: true,
          category_council_enabled: false,
          asymmetry_enabled: true,
          last_run_at: null,
          last_success_at: null,
          last_run_status: null,
          last_run_summary: {},
          created_at: '2026-04-07T12:00:00Z',
          updated_at: '2026-04-07T12:00:00Z',
        },
      ],
      count: 1,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })
    api.fetchCompetitiveSetPlan
      .mockResolvedValueOnce({
      competitive_set: { id: 'set-1', name: 'Helpdesk core' },
      plan: {
        competitive_set_id: 'set-1',
        focal_vendor_name: 'Intercom',
        vendor_names: ['Intercom', 'Zendesk', 'Freshdesk'],
        pairwise_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        category_names: [],
        asymmetry_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        vendor_synthesis_enabled: true,
        pairwise_enabled: true,
        category_council_enabled: false,
        asymmetry_enabled: true,
        vendor_job_count: 3,
        pairwise_job_count: 2,
        category_job_count: 0,
        asymmetry_job_count: 2,
        estimated_total_jobs: 7,
        estimate: {
          lookback_days: 30,
          vendor_jobs_planned: 3,
          pairwise_jobs_planned: 2,
          category_jobs_planned: 0,
          asymmetry_jobs_planned: 2,
          estimated_vendor_tokens: 1200,
          estimated_cross_vendor_tokens: 2400,
          estimated_total_tokens: 3600,
          estimated_vendor_cost_usd: 0.12,
          estimated_cross_vendor_cost_usd: 0.24,
          estimated_total_cost_usd: 0.36,
          estimated_vendor_tokens_likely_to_reason: 800,
          estimated_vendor_cost_usd_likely_to_reason: 0.08,
          vendor_jobs_with_history: 2,
          vendor_jobs_using_fallback: 1,
          cross_vendor_jobs_with_history: 3,
          cross_vendor_jobs_using_fallback: 1,
          vendor_jobs_with_matching_pools: 2,
          vendor_jobs_missing_pools: 1,
          vendor_jobs_likely_to_reason: 2,
          vendor_jobs_likely_hash_reuse: 1,
          vendor_jobs_likely_stale_reuse: 0,
          vendor_jobs_likely_missing_prior: 0,
          vendor_jobs_likely_hash_changed: 1,
          vendor_jobs_likely_prior_quality_weak: 0,
          vendor_jobs_likely_missing_packet_artifacts: 0,
          vendor_jobs_likely_missing_reference_ids: 0,
          likely_rerun_vendors: ['Intercom:changed'],
          likely_reuse_vendors: ['Zendesk:reuse'],
          recent_vendor_sample_count: 3,
          recent_cross_vendor_sample_count: 2,
          note: 'Estimated from recent runs.',
        },
      },
      recent_runs: [],
    })
    api.runCompetitiveSetNow.mockResolvedValue({
      execution_id: 'exec-new',
      status: 'started',
      message: 'queued',
      competitive_set_id: 'set-1',
      plan: {
        competitive_set_id: 'set-1',
      },
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('Helpdesk core')
    await user.click(screen.getByRole('button', { name: 'Preview cost' }))
    await screen.findByRole('button', { name: 'Run now' })
    await user.click(screen.getByRole('button', { name: 'Run now' }))

    expect(screen.getAllByText('running')).toHaveLength(2)
    expect(screen.getAllByText('changed only').length).toBeGreaterThanOrEqual(2)
    expect(await screen.findByText('Helpdesk core refresh started (exec-new)')).toBeInTheDocument()
  })

  it('does not inject a new optimistic run when the competitive set is already running', async () => {
    const user = userEvent.setup()
    api.listCompetitiveSets.mockResolvedValue({
      competitive_sets: [
        {
          id: 'set-1',
          name: 'Helpdesk core',
          focal_vendor_name: 'Intercom',
          competitor_vendor_names: ['Zendesk', 'Freshdesk'],
          active: true,
          refresh_mode: 'manual',
          refresh_interval_hours: null,
          vendor_synthesis_enabled: true,
          pairwise_enabled: true,
          category_council_enabled: false,
          asymmetry_enabled: true,
          last_run_at: '2026-04-07T12:06:00Z',
          last_success_at: null,
          last_run_status: 'running',
          last_run_summary: {
            changed_vendors_only: true,
            force_cross_vendor: true,
          },
          created_at: '2026-04-07T12:00:00Z',
          updated_at: '2026-04-07T12:06:00Z',
        },
      ],
      count: 1,
      defaults: {
        default_refresh_interval_hours: 24,
        max_competitors: 5,
        default_changed_vendors_only: true,
      },
    })
    api.fetchCompetitiveSetPlan.mockResolvedValue({
      competitive_set: { id: 'set-1', name: 'Helpdesk core' },
      plan: {
        competitive_set_id: 'set-1',
        focal_vendor_name: 'Intercom',
        vendor_names: ['Intercom', 'Zendesk', 'Freshdesk'],
        pairwise_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        category_names: [],
        asymmetry_pairs: [['Intercom', 'Zendesk'], ['Intercom', 'Freshdesk']],
        vendor_synthesis_enabled: true,
        pairwise_enabled: true,
        category_council_enabled: false,
        asymmetry_enabled: true,
        vendor_job_count: 3,
        pairwise_job_count: 2,
        category_job_count: 0,
        asymmetry_job_count: 2,
        estimated_total_jobs: 7,
        estimate: {
          lookback_days: 30,
          vendor_jobs_planned: 3,
          pairwise_jobs_planned: 2,
          category_jobs_planned: 0,
          asymmetry_jobs_planned: 2,
          estimated_vendor_tokens: 1200,
          estimated_cross_vendor_tokens: 2400,
          estimated_total_tokens: 3600,
          estimated_vendor_cost_usd: 0.12,
          estimated_cross_vendor_cost_usd: 0.24,
          estimated_total_cost_usd: 0.36,
          estimated_vendor_tokens_likely_to_reason: 800,
          estimated_vendor_cost_usd_likely_to_reason: 0.08,
          vendor_jobs_with_history: 2,
          vendor_jobs_using_fallback: 1,
          cross_vendor_jobs_with_history: 3,
          cross_vendor_jobs_using_fallback: 1,
          vendor_jobs_with_matching_pools: 2,
          vendor_jobs_missing_pools: 1,
          vendor_jobs_likely_to_reason: 2,
          vendor_jobs_likely_hash_reuse: 1,
          vendor_jobs_likely_stale_reuse: 0,
          vendor_jobs_likely_missing_prior: 0,
          vendor_jobs_likely_hash_changed: 1,
          vendor_jobs_likely_prior_quality_weak: 0,
          vendor_jobs_likely_missing_packet_artifacts: 0,
          vendor_jobs_likely_missing_reference_ids: 0,
          likely_rerun_vendors: ['Intercom:changed'],
          likely_reuse_vendors: ['Zendesk:reuse'],
          recent_vendor_sample_count: 3,
          recent_cross_vendor_sample_count: 2,
          note: 'Estimated from recent runs.',
        },
      },
      recent_runs: [
        {
          id: 'run-1',
          competitive_set_id: 'set-1',
          account_id: 'acct-1',
          run_id: 'scope-run-running',
          trigger: 'manual',
          status: 'running',
          execution_id: 'exec-existing',
          summary: {
            changed_vendors_only: true,
            force_cross_vendor: true,
          },
          started_at: '2026-04-07T12:06:00Z',
          completed_at: null,
          created_at: '2026-04-07T12:06:00Z',
        },
      ],
    })
    api.runCompetitiveSetNow.mockResolvedValue({
      execution_id: 'exec-existing',
      status: 'running',
      message: "Task 'b2b_reasoning_synthesis' is already running.",
      already_running: true,
      competitive_set_id: 'set-1',
      plan: {
        competitive_set_id: 'set-1',
      },
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByText('Helpdesk core')
    await user.click(screen.getByRole('button', { name: 'Preview cost' }))
    await screen.findByRole('button', { name: 'Run now' })
    await user.click(screen.getByLabelText('Run changed vendors only'))
    await user.click(screen.getByLabelText('Force vendor rerun'))
    await user.click(screen.getByRole('button', { name: 'Run now' }))

    await waitFor(() => expect(api.fetchCompetitiveSetPlan).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.getAllByText('running')).toHaveLength(2))
    await waitFor(() => expect(screen.getAllByText('cross-vendor forced')).toHaveLength(2))
    expect(screen.queryAllByText('vendor forced')).toHaveLength(0)
    expect(await screen.findByText("Task 'b2b_reasoning_synthesis' is already running.")).toBeInTheDocument()
  })

  it('matches the active saved view using alert delivery settings too', async () => {
    const user = userEvent.setup()
    api.listWatchlistViews.mockResolvedValue({
      views: [
        {
          id: 'view-1',
          name: 'Intercom no email',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: '',
          source: '',
          min_urgency: null,
          include_stale: true,
          named_accounts_only: false,
          changed_wedges_only: false,
          vendor_alert_threshold: null,
          account_alert_threshold: null,
          stale_days_threshold: null,
          alert_email_enabled: false,
          alert_delivery_frequency: 'daily',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
        {
          id: 'view-2',
          name: 'Intercom weekly email',
          vendor_names: ['Intercom'],
          vendor_name: 'Intercom',
          category: '',
          source: '',
          min_urgency: null,
          include_stale: true,
          named_accounts_only: false,
          changed_wedges_only: false,
          vendor_alert_threshold: null,
          account_alert_threshold: null,
          stale_days_threshold: null,
          alert_email_enabled: true,
          alert_delivery_frequency: 'weekly',
          next_alert_delivery_at: null,
          last_alert_delivery_at: null,
          last_alert_delivery_status: null,
          last_alert_delivery_summary: null,
          created_at: null,
          updated_at: null,
        },
      ],
      count: 2,
    })
    api.updateWatchlistView.mockResolvedValue({
      id: 'view-2',
      name: 'Intercom weekly email',
      vendor_names: ['Intercom'],
      vendor_name: 'Intercom',
      category: '',
      source: '',
      min_urgency: null,
      include_stale: true,
      named_accounts_only: false,
      changed_wedges_only: false,
      vendor_alert_threshold: null,
      account_alert_threshold: null,
      stale_days_threshold: null,
      alert_email_enabled: true,
      alert_delivery_frequency: 'weekly',
      next_alert_delivery_at: null,
      last_alert_delivery_at: null,
      last_alert_delivery_status: null,
      last_alert_delivery_summary: null,
      created_at: null,
      updated_at: null,
    })

    render(
      <MemoryRouter>
        <Watchlists />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Watchlists' })
    await user.click(screen.getAllByRole('button', { name: /Intercom weekly email/i })[0])
    await user.click(screen.getByRole('button', { name: 'Update active view' }))

    await waitFor(() => {
      expect(api.updateWatchlistView).toHaveBeenCalledWith('view-2', expect.objectContaining({
        alert_email_enabled: true,
        alert_delivery_frequency: 'weekly',
      }))
    })
  })

  it('keeps the account drawer aligned with the refreshed row payload', async () => {
    const user = userEvent.setup()

    api.fetchAccountsInMotionFeed
      .mockResolvedValueOnce({
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
            source_reviews: [
              {
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
              },
            ],
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
          },
        ],
        count: 1,
        tracked_vendor_count: 1,
        vendors_with_accounts: 1,
        min_urgency: 7,
        per_vendor_limit: 10,
        freshest_report_date: '2026-04-05',
      })
      .mockResolvedValueOnce({
        accounts: [
          {
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
            source_reviews: [
              {
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
              },
            ],
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
          },
        ],
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
})
