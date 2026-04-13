import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import EvidenceExplorer from './EvidenceExplorer'

const api = vi.hoisted(() => ({
  fetchAccountsInMotionFeed: vi.fn(),
  listTrackedVendors: vi.fn(),
  listWatchlistViews: vi.fn(),
  searchAvailableVendors: vi.fn(),
  fetchWitnesses: vi.fn(),
  fetchEvidenceVault: vi.fn(),
  fetchEvidenceTrace: vi.fn(),
  fetchWitness: vi.fn(),
  fetchAnnotations: vi.fn(),
  setAnnotation: vi.fn(),
  removeAnnotations: vi.fn(),
}))

vi.mock('../api/client', () => api)

function LocationSearchProbe() {
  const location = useLocation()
  return <div data-testid="location-search">{location.search}</div>
}

function activeVendorShortcuts(vendorName = 'Zendesk') {
  const headerScope = screen
    .getByText((_, node) => node?.textContent?.trim() === `Focused on ${vendorName}`)
    .closest('div')

  if (!(headerScope instanceof HTMLElement)) {
    throw new Error(`Expected active vendor shortcuts for ${vendorName}`)
  }

  return within(headerScope)
}

describe('EvidenceExplorer', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.listTrackedVendors.mockResolvedValue({
      vendors: [{ vendor_name: 'Zendesk' }],
      count: 1,
    })
    api.listWatchlistViews.mockResolvedValue({
      views: [{
        id: 'view-zendesk',
        vendor_name: 'Zendesk',
        vendor_names: ['Zendesk'],
      }],
      count: 1,
    })
    api.fetchAccountsInMotionFeed.mockResolvedValue({
      accounts: [{
        company: 'Acme Corp',
        vendor: 'Zendesk',
        watch_vendor: 'Zendesk',
        track_mode: 'competitor',
        category: 'Helpdesk',
        report_date: '2026-04-05',
      }],
      count: 1,
      tracked_vendor_count: 1,
      vendors_with_accounts: 1,
      min_urgency: 7,
      per_vendor_limit: 10,
      freshest_report_date: '2026-04-05',
    })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [], count: 0 })
    api.fetchWitnesses.mockResolvedValue({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 1,
      limit: 30,
      offset: 0,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:1',
        review_id: 'review-1',
        witness_type: 'pricing',
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
        as_of_date: '2026-04-09',
      }],
    })
    api.fetchEvidenceVault.mockResolvedValue({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      schema_version: 'v1',
      created_at: '2026-04-09T00:00:00Z',
      weakness_evidence: [],
      strength_evidence: [],
      company_signals: [],
      metric_snapshot: {},
      provenance: {},
      witness_count: 1,
    })
    api.fetchEvidenceTrace.mockResolvedValue({
      vendor_name: 'Zendesk',
      trace: {
        synthesis: null,
        reasoning_packet: null,
        witnesses: [],
        source_reviews: [],
        evidence_diff: null,
      },
      stats: {
        witness_count: 0,
        unique_reviews: 0,
        has_synthesis: false,
        has_packet: false,
        has_diff: false,
      },
    })
    api.fetchWitness.mockResolvedValue({
      witness: {
        witness_id: 'witness:zendesk:1',
        review_id: 'review-1',
        witness_type: 'pricing',
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
        as_of_date: '2026-04-09',
        review_text: 'We need to move fast before renewal.',
        summary: null,
        pros: null,
        cons: null,
        rating: 2,
        review_source: 'reddit',
        source_url: 'https://reddit.example/review-1',
        enrichment_status: 'completed',
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
  })

  it('hydrates vendor, filters, and witness focus from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&pain_category=pricing&source=reddit&witness_type=pricing&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1&account_vendor=Zendesk',
    )

    await waitFor(() => {
      expect(api.fetchWitnesses).toHaveBeenLastCalledWith({
        vendor_name: 'Zendesk',
        as_of_date: '2026-04-08',
        window_days: 45,
        pain_category: 'pricing',
        source: 'reddit',
        witness_type: 'pricing',
        limit: 30,
        offset: 0,
      })
    })

    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchWitness).toHaveBeenLastCalledWith('witness:zendesk:1', 'Zendesk', {
        as_of_date: '2026-04-09',
        window_days: 45,
      })
    })
    const accountReviewLinks = await screen.findAllByRole('link', { name: 'Open account review' })
    expect(accountReviewLinks.length).toBeGreaterThan(0)
    for (const link of accountReviewLinks) {
      expect(link).toHaveAttribute(
        'href',
        '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26as_of_date%3D2026-04-08%26window_days%3D45%26pain_category%3Dpricing%26source%3Dreddit%26witness_type%3Dpricing%26witness_id%3Dwitness%253Azendesk%253A1%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk',
      )
    }
  })

  it('wires exact explorer shortcuts into the witness drawer', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    try {
      render(
        <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&pain_category=pricing&source=reddit&witness_type=pricing&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk']}>
          <EvidenceExplorer />
        </MemoryRouter>,
      )

      const explorerLink = await screen.findByRole('link', { name: 'Open in Evidence Explorer' })
      const drawerShortcutsRow = explorerLink.closest('div')
      expect(drawerShortcutsRow).not.toBeNull()
      const drawerShortcuts = within(drawerShortcutsRow as HTMLElement)
      expect(explorerLink).toHaveAttribute(
        'href',
        '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&pain_category=pricing&source=reddit&witness_type=pricing&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk',
      )
      expect(drawerShortcuts.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
        'href',
        '/watchlists?view=view-1&account_vendor=Zendesk',
      )
      expect(drawerShortcuts.getByRole('link', { name: 'Alerts API' })).toHaveAttribute(
        'href',
        '/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26as_of_date%3D2026-04-08%26window_days%3D45%26pain_category%3Dpricing%26source%3Dreddit%26witness_type%3Dpricing%26witness_id%3Dwitness%253Azendesk%253A1%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk',
      )
      expect(drawerShortcuts.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
        'href',
        '/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26as_of_date%3D2026-04-08%26window_days%3D45%26pain_category%3Dpricing%26source%3Dreddit%26witness_type%3Dpricing%26witness_id%3Dwitness%253Azendesk%253A1%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk',
      )

      await user.click(screen.getByRole('button', { name: 'Copy evidence explorer link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledWith(
          `${window.location.origin}/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&pain_category=pricing&source=reddit&witness_type=pricing&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk`,
        )
      })
    } finally {
      clipboardSpy.mockRestore()
    }
  })



  it('preserves review detail back navigation from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Back to Review' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('hydrates pagination offset from the URL and preserves it in downstream report links', async () => {
    api.fetchWitnesses.mockResolvedValueOnce({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 30,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:31',
        review_id: 'review-31',
        witness_type: 'pricing',
        excerpt_text: 'The renewal window is now urgent.',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchWitnesses).toHaveBeenLastCalledWith({
        vendor_name: 'Zendesk',
        window_days: 30,
        pain_category: undefined,
        source: 'reddit',
        witness_type: undefined,
        limit: 30,
        offset: 30,
      })
    })

    expect(screen.getByText('Showing 31-60 of 61')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'View library for Zendesk' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26offset%3D30',
    )
  })

  it('accepts vendor back_to and renders a vendor return link', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendors%2FZendesk']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk',
    )
  })

  it('accepts alerts back_to and renders an alerts return link', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Falerts%3Fwebhook%3Dwh-crm']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to Alerts' })).toHaveAttribute(
      'href',
      '/alerts?webhook=wh-crm',
    )
  })

  it.each([
    {
      name: 'accepts dashboard back_to and renders a dashboard return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fdashboard',
      buttonName: 'Back to Dashboard',
      expectedHref: '/dashboard',
    },
    {
      name: 'accepts vendors-list back_to and renders a vendors return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendors%3Fsearch%3DZendesk%26min_urgency%3D6',
      buttonName: 'Back to Vendors',
      expectedHref: '/vendors?search=Zendesk&min_urgency=6',
    },
    {
      name: 'accepts affiliates back_to and renders an affiliates return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
      buttonName: 'Back to Affiliates',
      expectedHref: '/affiliates?vendor=Zendesk&min_urgency=7&min_score=80&dm_only=true',
    },
    {
      name: 'accepts vendor-targets back_to and renders a vendor-targets return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fvendor-targets%3Fsearch%3DZendesk%26mode%3Dchallenger_intel',
      buttonName: 'Back to Vendor Targets',
      expectedHref: '/vendor-targets?search=Zendesk&mode=challenger_intel',
    },
    {
      name: 'accepts briefing-review back_to and renders a briefing-review return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DZendesk',
      buttonName: 'Back to Briefing Review',
      expectedHref: '/briefing-review?status=sent&vendor=Zendesk',
    },
    {
      name: 'accepts blog-review back_to and renders a blog-review return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fblog-review%3Fstatus%3Dpublished%26draft%3Ddraft-1',
      buttonName: 'Back to Blog Review',
      expectedHref: '/blog-review?status=published&draft=draft-1',
    },
    {
      name: 'accepts campaign-review back_to and renders a campaign-review return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fcampaign-review%3Fstatus%3Dsent%26company%3DAcme%2BCorp',
      buttonName: 'Back to Campaign Review',
      expectedHref: '/campaign-review?status=sent&company=Acme+Corp',
    },
    {
      name: 'accepts challengers back_to and renders a challengers return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fchallengers%3Fsearch%3DZendesk',
      buttonName: 'Back to Challengers',
      expectedHref: '/challengers?search=Zendesk',
    },
    {
      name: 'accepts prospects back_to and renders a prospects return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fprospects%3Fcompany%3DAcme%26status%3Dactive%26seniority%3Dvp',
      buttonName: 'Back to Prospects',
      expectedHref: '/prospects?company=Acme&status=active&seniority=vp',
    },
    {
      name: 'accepts pipeline-review back_to and renders a pipeline-review return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fpipeline-review%3Fqueue_vendor%3DZendesk',
      buttonName: 'Back to Pipeline Review',
      expectedHref: '/pipeline-review?queue_vendor=Zendesk',
    },
    {
      name: 'accepts predictor back_to and renders a predictor return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
      buttonName: 'Back to Predictor',
      expectedHref: '/predictor?vendor=Zendesk&company_size=smb&industry=fintech',
    },
    {
      name: 'accepts report-detail back_to and renders a report return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
      buttonName: 'Back to Report',
      expectedHref: '/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    },
    {
      name: 'accepts public-report back_to and renders a public-report return link',
      initialEntry: '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview',
      buttonName: 'Back to Report',
      expectedHref: '/report?vendor=Zendesk&ref=test-token&mode=view',
    },
  ])('$name', async ({ initialEntry, buttonName, expectedHref }) => {
    render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: buttonName })).toHaveAttribute('href', expectedHref)
  })

  it('renders an account review return label for focused watchlist back_to paths', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor',
    )
  })

  it('writes witness focus back into the URL when a witness is opened', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk']}>
        <LocationSearchProbe />
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const witnessCard = await screen.findByRole('button', { name: /We need to move fast before renewal\./i })
    await user.click(witnessCard)
    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('vendor=Zendesk')
      expect(screen.getByTestId('location-search')).toHaveTextContent('tab=witnesses')
      expect(screen.getByTestId('location-search')).toHaveTextContent('witness_id=witness%3Azendesk%3A1')
    })
  })


  it('writes pagination offset back into the URL when paging witnesses', async () => {
    const user = userEvent.setup()
    api.fetchWitnesses.mockResolvedValue({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 0,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:1',
        review_id: 'review-1',
        witness_type: 'pricing',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk']}>
        <LocationSearchProbe />
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Showing 1-30 of 61')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => {
      expect(screen.getByTestId('location-search')).toHaveTextContent('vendor=Zendesk')
      expect(screen.getByTestId('location-search')).toHaveTextContent('offset=30')
    })
  })

  it('copies the current explorer URL', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&pain_category=pricing&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Link' }))

    expect(screen.getByText('Copied')).toBeInTheDocument()
  })


  it('copies an exact witness-focused explorer link from the card footer', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchWitnesses.mockResolvedValueOnce({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 30,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:31',
        review_id: 'review-31',
        witness_type: 'pricing',
        excerpt_text: 'The renewal window is now urgent.',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy link for witness witness:zendesk:31' }))

    expect(clipboardSpy).toHaveBeenCalledWith(
      `${window.location.origin}/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30&witness_id=witness%3Azendesk%3A31`,
    )
    expect(screen.getByText('Copied')).toBeInTheDocument()
  })

  it('links the vendor report library back to the current explorer state', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'View library for Zendesk' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
  })

  it('shows watchlists, alerts, vendor workspace, opportunities, and reports shortcuts for the active vendor', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    const headerShortcuts = activeVendorShortcuts()
    expect(headerShortcuts.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
    expect(headerShortcuts.getByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
    expect(headerShortcuts.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
    expect(headerShortcuts.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
  })

  it('prefers the exact upstream alerts shortcut when entered from an alerts path', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Falerts%253Fwebhook%253Dwh-crm%2526window%253D30d']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const alertsLink = activeVendorShortcuts().getByRole('link', { name: 'Alerts API' })
    expect(alertsLink).toHaveAttribute(
      'href',
      '/alerts?webhook=wh-crm&window=30d',
    )
  })

  it('copies the exact upstream alerts shortcut when entered from an alerts path', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Falerts%253Fwebhook%253Dwh-crm%2526window%253D30d']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const alertsLink = headerShortcuts.getByRole('link', { name: 'Alerts API' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy alerts link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${alertsLink.getAttribute('href')}`)
    })
  })

  it('copies the active alerts shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const alertsLink = headerShortcuts.getByRole('link', { name: 'Alerts API' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy alerts link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${alertsLink.getAttribute('href')}`)
    })
  })

  it('prefers the exact upstream vendor workspace shortcut when entered from a vendor path', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fvendors%252FZendesk%253Ftab%253Dreviews%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const vendorLink = activeVendorShortcuts().getByRole('link', { name: 'Vendor workspace' })
    expect(vendorLink).toHaveAttribute(
      'href',
      '/vendors/Zendesk?tab=reviews&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies the exact upstream vendor workspace shortcut when entered from a vendor path', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fvendors%252FZendesk%253Ftab%253Dreviews%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const vendorLink = headerShortcuts.getByRole('link', { name: 'Vendor workspace' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy vendor workspace link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${vendorLink.getAttribute('href')}`)
    })
  })

  it('copies the active vendor workspace shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const vendorLink = headerShortcuts.getByRole('link', { name: 'Vendor workspace' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy vendor workspace link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${vendorLink.getAttribute('href')}`)
    })
  })

  it('prefers the exact upstream opportunities shortcut when entered from an opportunity path', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fopportunities%253Fvendor%253DZendesk%2526stage%253Dexpansion%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const opportunitiesLink = await screen.findByRole('link', { name: 'Opportunities' })
    expect(opportunitiesLink).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&stage=expansion&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies the exact upstream opportunities shortcut when entered from an opportunity path', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fopportunities%253Fvendor%253DZendesk%2526stage%253Dexpansion%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const opportunitiesLink = headerShortcuts.getByRole('link', { name: 'Opportunities' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy opportunities link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${opportunitiesLink.getAttribute('href')}`)
    })
  })

  it('copies the active opportunities shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const opportunitiesLink = headerShortcuts.getByRole('link', { name: 'Opportunities' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy opportunities link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${opportunitiesLink.getAttribute('href')}`)
    })
  })

  it('prefers the exact upstream reports shortcut when entered from a report path', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Freports%253Fvendor_filter%253DZendesk%2526report_type%253Dbattle_card%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const reportsLink = await screen.findByRole('link', { name: 'Reports' })
    expect(reportsLink).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&report_type=battle_card&back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('copies the exact upstream reports shortcut when entered from a report path', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Freports%253Fvendor_filter%253DZendesk%2526report_type%253Dbattle_card%2526back_to%253D%25252Fwatchlists%25253Fview%25253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const reportsLink = await screen.findByRole('link', { name: 'Reports' })
    await user.click(screen.getByRole('button', { name: 'Copy reports link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${reportsLink.getAttribute('href')}`)
    })
  })

  it('copies the active reports shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const reportsLink = await screen.findByRole('link', { name: 'Reports' })
    await user.click(screen.getByRole('button', { name: 'Copy reports link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${reportsLink.getAttribute('href')}`)
    })
  })

  it('copies the active watchlists shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const headerShortcuts = activeVendorShortcuts()
    const watchlistsLink = await headerShortcuts.findByRole('link', { name: 'Watchlists' })
    await user.click(headerShortcuts.getByRole('button', { name: 'Copy watchlists link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${watchlistsLink.getAttribute('href')}`)
    })
  })

  it('prefers the upstream account review path for the watchlists shortcut', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}> 
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp',
    )
  })

  it('copies the upstream account review shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DZendesk%2526account_company%253DAcme%252BCorp']}> 
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account review link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?view=view-1&account_vendor=Zendesk&account_company=Acme+Corp`,
      )
    })
  })

  it('keeps the watchlists shortcut when a saved view exists even without tracked-vendor membership', async () => {
    api.listTrackedVendors.mockResolvedValueOnce({
      vendors: [],
      count: 0,
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(activeVendorShortcuts().getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses',
    )
  })

  it('shows a direct review detail shortcut on witness cards', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open review detail' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit',
    )
  })

  it('prefers the exact upstream review detail path on witness cards', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open review detail' })).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })


  it('copies a direct review detail link from a witness card', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchWitnesses.mockResolvedValueOnce({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 30,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:31',
        review_id: 'review-31',
        witness_type: 'pricing',
        excerpt_text: 'The renewal window is now urgent.',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy review link for witness witness:zendesk:31' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reviews/review-31?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26offset%3D30`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copy review link for witness witness:zendesk:31' })).toHaveTextContent('Copied')
  })

  it('copies the exact upstream review detail path from a witness card', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy review link for witness witness:zendesk:1' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copy review link for witness witness:zendesk:1' })).toHaveTextContent('Copied')
  })

  it('shows a direct account review shortcut on witness cards when the reviewer company matches a tracked account', async () => {
    api.fetchWitnesses.mockResolvedValueOnce({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 30,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:31',
        review_id: 'review-31',
        witness_type: 'pricing',
        excerpt_text: 'The renewal window is now urgent.',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open account review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26offset%3D30%26witness_id%3Dwitness%253Azendesk%253A31',
    )
  })

  it('copies a direct account review link from a witness card', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchWitnesses.mockResolvedValueOnce({
      vendor_name: 'Zendesk',
      as_of_date: '2026-04-09',
      analysis_window_days: 30,
      total: 61,
      limit: 30,
      offset: 30,
      facets: {
        pain_categories: ['pricing'],
        sources: ['reddit'],
        witness_types: ['pricing'],
      },
      witnesses: [{
        witness_id: 'witness:zendesk:31',
        review_id: 'review-31',
        witness_type: 'pricing',
        excerpt_text: 'The renewal window is now urgent.',
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
        as_of_date: '2026-04-09',
      }],
    })

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&offset=30']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account review link for witness witness:zendesk:31' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26offset%3D30%26witness_id%3Dwitness%253Azendesk%253A31`,
      )
    })
    expect(screen.getByRole('button', { name: 'Copy account review link for witness witness:zendesk:31' })).toHaveTextContent('Copied')
  })

  it('opens review detail from witness drilldown with evidence back_to preserved', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    const witnessCard = await screen.findByRole('button', { name: /We need to move fast before renewal\./i })
    await user.click(witnessCard)

    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    const reviewLinks = screen.getAllByRole('link', { name: 'Open review detail' })
    expect(reviewLinks.at(-1)).toHaveAttribute(
      'href',
      '/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
  })

  it('keeps the page usable when witness loading fails', async () => {
    api.fetchWitnesses.mockRejectedValue(new Error('API 500: witness search unavailable'))

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(await screen.findByText('No witnesses found')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Witnesses/i })).toBeInTheDocument()

    await waitFor(() => {
      expect(api.fetchWitnesses).toHaveBeenCalledWith({
        vendor_name: 'Zendesk',
        window_days: 30,
        pain_category: undefined,
        source: undefined,
        witness_type: undefined,
        limit: 30,
        offset: 0,
      })
    })
  })
})
