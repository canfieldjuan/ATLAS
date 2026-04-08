import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import EvidenceDrawer from './EvidenceDrawer'

const api = vi.hoisted(() => ({
  fetchWitness: vi.fn(),
  fetchAnnotations: vi.fn(),
  setAnnotation: vi.fn(),
  removeAnnotations: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('EvidenceDrawer', () => {
  beforeEach(() => {
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
  })

  it('loads witness annotations and lets analysts pin a witness', async () => {
    const user = userEvent.setup()

    render(
      <EvidenceDrawer
        vendorName="Salesforce"
        witnessId="w1"
        open
        onClose={() => {}}
      />,
    )

    expect(await screen.findByRole('heading', { name: 'Witness Detail' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: 'Pin' })).toBeInTheDocument()

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
})
