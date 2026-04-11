import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import EvidenceExplorer from './EvidenceExplorer'

const api = vi.hoisted(() => ({
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

describe('EvidenceExplorer', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
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
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&pain_category=pricing&source=reddit&witness_type=pricing&witness_id=witness%3Azendesk%3A1&back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DZendesk']}>
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
        window_days: 30,
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
        window_days: 30,
      })
    })
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

  it('copies the current explorer URL', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&pain_category=pricing&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Link' }))

    expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
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

  it('shows vendor workspace, opportunities, and reports shortcuts for the active vendor', async () => {
    render(
      <MemoryRouter initialEntries={['/evidence?vendor=Zendesk&tab=witnesses&source=reddit&witness_id=witness%3Azendesk%3A1']}>
        <EvidenceExplorer />
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit%26witness_id%3Dwitness%253Azendesk%253A1',
    )
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
    expect(screen.getByRole('link', { name: 'Open review detail' })).toHaveAttribute(
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
