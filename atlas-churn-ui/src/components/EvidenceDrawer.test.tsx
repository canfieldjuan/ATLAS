import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import EvidenceDrawer from './EvidenceDrawer'

const api = vi.hoisted(() => ({
  fetchWitness: vi.fn(),
  fetchAnnotations: vi.fn(),
  setAnnotation: vi.fn(),
  removeAnnotations: vi.fn(),
  fetchAccountsInMotionFeed: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('EvidenceDrawer', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchWitness.mockResolvedValue({
      witness: {
        witness_id: 'w1',
        excerpt_text: 'Pricing spiked after renewal.',
        source: 'g2',
        reviewed_at: '2026-04-07T18:00:00Z',
        reviewer_company: 'Acme Corp',
        reviewer_title: 'VP IT',
        pain_category: 'pricing',
        competitor: 'Zendesk',
        salience_score: 0.92,
        specificity_score: 0.76,
        selection_reason: 'named_account',
        signal_tags: ['pricing_backlash'],
        review_text: 'Pricing spiked after renewal and support dropped.',
        evidence_spans: [],
        all_evidence_span_count: 0,
      },
    })
    api.fetchAnnotations.mockResolvedValue({ annotations: [] })
    api.setAnnotation.mockResolvedValue({
      id: 'ann-1',
      witness_id: 'w1',
      vendor_name: 'Salesforce',
      annotation_type: 'pin',
      note_text: null,
      created_at: '2026-04-08T12:00:00Z',
      updated_at: '2026-04-08T12:00:00Z',
    })
    api.removeAnnotations.mockResolvedValue({ removed: 1 })
    api.fetchAccountsInMotionFeed.mockResolvedValue({ accounts: [] })
  })

  it('loads witness annotations and lets analysts pin a witness', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <EvidenceDrawer
          vendorName="Salesforce"
          witnessId="w1"
          open
          backToPath="/report?vendor=Salesforce&ref=test-token&mode=view"
          onClose={() => {}}
        />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Pin' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?vendor_name=Salesforce&back_to=%2Freport%3Fvendor%3DSalesforce%26ref%3Dtest-token%26mode%3Dview',
    )
    expect(screen.getByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Salesforce&company=Acme+Corp&back_to=%2Freport%3Fvendor%3DSalesforce%26ref%3Dtest-token%26mode%3Dview',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Salesforce&back_to=%2Freport%3Fvendor%3DSalesforce%26ref%3Dtest-token%26mode%3Dview',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Salesforce?back_to=%2Freport%3Fvendor%3DSalesforce%26ref%3Dtest-token%26mode%3Dview',
    )
    expect(screen.getByRole('link', { name: /View library/i })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Salesforce&back_to=%2Freport%3Fvendor%3DSalesforce%26ref%3Dtest-token%26mode%3Dview',
    )

    await user.click(screen.getByRole('button', { name: 'Pin' }))

    await waitFor(() => {
      expect(api.fetchAnnotations).toHaveBeenCalledWith({ vendor_name: 'Salesforce' })
      expect(api.setAnnotation).toHaveBeenCalledWith({
        witness_id: 'w1',
        vendor_name: 'Salesforce',
        annotation_type: 'pin',
      })
    })

    expect(await screen.findByText('Remove pin')).toBeInTheDocument()
  })

  it('copies exact drawer handoff links', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchWitness.mockResolvedValueOnce({
      witness: {
        witness_id: 'w1',
        excerpt_text: 'Pricing spiked after renewal.',
        source: 'g2',
        reviewed_at: '2026-04-07T18:00:00Z',
        reviewer_company: 'Acme Corp',
        reviewer_title: 'VP IT',
        pain_category: 'pricing',
        competitor: 'Zendesk',
        salience_score: 0.92,
        specificity_score: 0.76,
        selection_reason: 'named_account',
        signal_tags: ['pricing_backlash'],
        review_id: 'review-1',
        source_url: 'https://g2.example/review-1',
        review_text: 'Pricing spiked after renewal and support dropped.',
        evidence_spans: [],
        all_evidence_span_count: 0,
      },
    })
    api.fetchAccountsInMotionFeed.mockResolvedValueOnce({
      accounts: [{
        company: 'Acme Corp',
        vendor: 'Salesforce',
        watch_vendor: 'HubSpot',
        category: 'CRM',
        track_mode: 'competitor',
        report_date: '2026-04-08',
      }],
    })

    try {
      render(
        <MemoryRouter>
          <EvidenceDrawer
            vendorName="Salesforce"
            witnessId="w1"
            open
            explorerUrl="/evidence?vendor=Salesforce&witness_id=w1&back_to=%2Freports%3Fvendor_filter%3DSalesforce"
            backToPath="/reports?vendor_filter=Salesforce"
            onClose={() => {}}
          />
        </MemoryRouter>,
      )

      await user.click(await screen.findByRole('button', { name: 'Copy evidence explorer link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(1)
      })
      let copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      let copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/evidence')
      expect(copiedUrl.searchParams.get('vendor')).toBe('Salesforce')
      expect(copiedUrl.searchParams.get('witness_id')).toBe('w1')

      await user.click(await screen.findByRole('button', { name: 'Copy account review link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(2)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/watchlists')
      expect(copiedUrl.searchParams.get('account_company')).toBe('Acme Corp')
      expect(copiedUrl.searchParams.get('account_vendor')).toBe('Salesforce')
      expect(copiedUrl.searchParams.get('account_watch_vendor')).toBe('HubSpot')

      await user.click(screen.getByRole('button', { name: 'Copy watchlists link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(3)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/watchlists')
      expect(copiedUrl.searchParams.get('vendor_name')).toBe('Salesforce')
      const watchlistsBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(watchlistsBackTo.pathname).toBe('/reports')
      expect(watchlistsBackTo.searchParams.get('vendor_filter')).toBe('Salesforce')

      await user.click(screen.getByRole('button', { name: 'Copy report library link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(4)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/reports')
      expect(copiedUrl.searchParams.get('vendor_filter')).toBe('Salesforce')

      await user.click(screen.getByRole('button', { name: 'Copy alerts link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(5)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/alerts')
      expect(copiedUrl.searchParams.get('vendor')).toBe('Salesforce')
      expect(copiedUrl.searchParams.get('company')).toBe('Acme Corp')
      const alertsBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(alertsBackTo.pathname).toBe('/reports')
      expect(alertsBackTo.searchParams.get('vendor_filter')).toBe('Salesforce')

      await user.click(screen.getByRole('button', { name: 'Copy opportunities link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(6)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/opportunities')
      expect(copiedUrl.searchParams.get('vendor')).toBe('Salesforce')
      const opportunitiesBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(opportunitiesBackTo.pathname).toBe('/reports')
      expect(opportunitiesBackTo.searchParams.get('vendor_filter')).toBe('Salesforce')

      await user.click(screen.getByRole('button', { name: 'Copy vendor workspace link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(7)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/vendors/Salesforce')
      const vendorBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(vendorBackTo.pathname).toBe('/reports')
      expect(vendorBackTo.searchParams.get('vendor_filter')).toBe('Salesforce')

      await user.click(screen.getByRole('button', { name: 'Copy review detail link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledTimes(8)
      })
      copiedText = clipboardSpy.mock.calls[clipboardSpy.mock.calls.length - 1]?.[0] as string
      copiedUrl = new URL(copiedText)
      expect(copiedUrl.pathname).toBe('/reviews/review-1')
      const reviewBackTo = new URL(copiedUrl.searchParams.get('back_to') || '', 'https://atlas.test')
      expect(reviewBackTo.pathname).toBe('/reports')
      expect(reviewBackTo.searchParams.get('vendor_filter')).toBe('Salesforce')
    } finally {
      clipboardSpy.mockRestore()
    }
  })

  it('prefers exact upstream alerts and vendor workspace paths when present', async () => {
    render(
      <MemoryRouter>
        <EvidenceDrawer
          vendorName="Salesforce"
          witnessId="w1"
          open
          backToPath="/reports?vendor_filter=Salesforce&back_to=%2Falerts%3Fvendor%3DSalesforce%26back_to%3D%252Fvendors%252FSalesforce%253Fback_to%253D%25252Fwatchlists%25253Fview%25253Dview-1"
          onClose={() => {}}
        />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1',
    )
    expect(await screen.findByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Salesforce&back_to=%2Fvendors%2FSalesforce%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Salesforce&back_to=%2Freports%3Fvendor_filter%3DSalesforce%26back_to%3D%252Falerts%253Fvendor%253DSalesforce%2526back_to%253D%25252Fvendors%25252FSalesforce%25253Fback_to%25253D%2525252Fwatchlists%2525253Fview%2525253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Salesforce?back_to=%2Fwatchlists%3Fview%3Dview-1',
    )
  })

  it('prefers the exact upstream opportunities path when present', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    try {
      render(
        <MemoryRouter>
          <EvidenceDrawer
            vendorName="Salesforce"
            witnessId="w1"
            open
            backToPath="/vendors/Salesforce?back_to=%2Fopportunities%3Fvendor%3DSalesforce%26back_to%3D%252Fwatchlists%253Fview%253Dview-1"
            onClose={() => {}}
          />
        </MemoryRouter>,
      )

      expect(await screen.findByRole('link', { name: 'Opportunities' })).toHaveAttribute(
        'href',
        '/opportunities?vendor=Salesforce&back_to=%2Fwatchlists%3Fview%3Dview-1',
      )

      await user.click(screen.getByRole('button', { name: 'Copy opportunities link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledWith(
          `${window.location.origin}/opportunities?vendor=Salesforce&back_to=%2Fwatchlists%3Fview%3Dview-1`,
        )
      })
    } finally {
      clipboardSpy.mockRestore()
    }
  })

  it('prefers the exact upstream report detail path when present', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    try {
      render(
        <MemoryRouter>
          <EvidenceDrawer
            vendorName="Salesforce"
            witnessId="w1"
            open
            backToPath="/vendors/Salesforce?back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1"
            onClose={() => {}}
          />
        </MemoryRouter>,
      )

      expect(await screen.findByRole('link', { name: 'Open report detail' })).toHaveAttribute(
        'href',
        '/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1',
      )

      await user.click(screen.getByRole('button', { name: 'Copy report detail link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledWith(
          `${window.location.origin}/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1`,
        )
      })
    } finally {
      clipboardSpy.mockRestore()
    }
  })

  it('prefers exact upstream review and account review paths when present', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    api.fetchWitness.mockResolvedValueOnce({
      witness: {
        witness_id: 'w1',
        excerpt_text: 'Pricing spiked after renewal.',
        source: 'g2',
        reviewed_at: '2026-04-07T18:00:00Z',
        reviewer_company: 'Acme Corp',
        reviewer_title: 'VP IT',
        pain_category: 'pricing',
        competitor: 'Zendesk',
        salience_score: 0.92,
        specificity_score: 0.76,
        selection_reason: 'named_account',
        signal_tags: ['pricing_backlash'],
        review_id: 'review-1',
        review_text: 'Pricing spiked after renewal and support dropped.',
        evidence_spans: [],
        all_evidence_span_count: 0,
      },
    })

    try {
      render(
        <MemoryRouter>
          <EvidenceDrawer
            vendorName="Salesforce"
            witnessId="w1"
            open
            backToPath="/vendors/Salesforce?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1%2526account_vendor%253DSalesforce%2526account_company%253DAcme%252BCorp"
            onClose={() => {}}
          />
        </MemoryRouter>,
      )

      expect(await screen.findByRole('link', { name: 'Open account review' })).toHaveAttribute(
        'href',
        '/watchlists?view=view-1&account_vendor=Salesforce&account_company=Acme+Corp',
      )
      expect(screen.getByRole('link', { name: 'Open review detail' })).toHaveAttribute(
        'href',
        '/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DSalesforce%26account_company%3DAcme%2BCorp',
      )
      expect(api.fetchAccountsInMotionFeed).not.toHaveBeenCalled()

      await user.click(screen.getByRole('button', { name: 'Copy account review link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledWith(
          `${window.location.origin}/watchlists?view=view-1&account_vendor=Salesforce&account_company=Acme+Corp`,
        )
      })

      await user.click(screen.getByRole('button', { name: 'Copy review detail link' }))
      await waitFor(() => {
        expect(clipboardSpy).toHaveBeenCalledWith(
          `${window.location.origin}/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1%26account_vendor%3DSalesforce%26account_company%3DAcme%2BCorp`,
        )
      })
    } finally {
      clipboardSpy.mockRestore()
    }
  })

  it('surfaces annotation mutation failures inline', async () => {
    const user = userEvent.setup()
    api.setAnnotation.mockRejectedValueOnce(new Error('annotation service unavailable'))

    render(
      <MemoryRouter>
        <EvidenceDrawer
          vendorName="Salesforce"
          witnessId="w1"
          open
          onClose={() => {}}
        />
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('button', { name: 'Pin' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('annotation service unavailable')
    expect(screen.queryByText('Remove pin')).not.toBeInTheDocument()
  })

  it('uses an in-app confirmation modal before removing an annotation', async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true)
    api.fetchAnnotations.mockResolvedValueOnce({
      annotations: [{
        id: 'ann-1',
        witness_id: 'w1',
        vendor_name: 'Salesforce',
        annotation_type: 'pin',
        note_text: null,
        created_at: '2026-04-08T12:00:00Z',
        updated_at: '2026-04-08T12:00:00Z',
      }],
    })

    try {
      render(
        <MemoryRouter>
          <EvidenceDrawer
            vendorName="Salesforce"
            witnessId="w1"
            open
            onClose={() => {}}
          />
        </MemoryRouter>,
      )

      await user.click(await screen.findByRole('button', { name: 'Remove pin' }))

      expect(confirmSpy).not.toHaveBeenCalled()
      expect(api.removeAnnotations).not.toHaveBeenCalled()

      const dialog = await screen.findByRole('alertdialog')
      expect(dialog).toHaveTextContent('Remove pin')
      expect(dialog).toHaveTextContent('Remove pin annotation from this witness?')

      await user.click(within(dialog).getByRole('button', { name: 'Remove pin' }))

      await waitFor(() => {
        expect(api.removeAnnotations).toHaveBeenCalledWith({ witness_ids: ['w1'] })
      })
      expect(await screen.findByRole('button', { name: 'Pin' })).toBeInTheDocument()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it('does not remove an annotation when the confirmation modal is cancelled', async () => {
    const user = userEvent.setup()
    api.fetchAnnotations.mockResolvedValueOnce({
      annotations: [{
        id: 'ann-1',
        witness_id: 'w1',
        vendor_name: 'Salesforce',
        annotation_type: 'pin',
        note_text: null,
        created_at: '2026-04-08T12:00:00Z',
        updated_at: '2026-04-08T12:00:00Z',
      }],
    })

    render(
      <MemoryRouter>
        <EvidenceDrawer
          vendorName="Salesforce"
          witnessId="w1"
          open
          onClose={() => {}}
        />
      </MemoryRouter>,
    )

    await user.click(await screen.findByRole('button', { name: 'Remove pin' }))

    const dialog = await screen.findByRole('alertdialog')
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))

    await waitFor(() => {
      expect(api.removeAnnotations).not.toHaveBeenCalled()
    })
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Remove pin' }).length).toBeGreaterThan(0)
  })
})
