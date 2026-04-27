import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import WinLossPredictor from './WinLossPredictor'

const api = vi.hoisted(() => ({
  predictWinLoss: vi.fn(),
  searchAvailableVendors: vi.fn(),
  fetchRecentPredictions: vi.fn(),
  fetchPredictionById: vi.fn(),
  compareWinLoss: vi.fn(),
  downloadPredictionCsv: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('WinLossPredictor', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchRecentPredictions.mockResolvedValue({ predictions: [] })
    api.searchAvailableVendors.mockResolvedValue({ vendors: [] })
    api.fetchPredictionById.mockResolvedValue(null)
    api.compareWinLoss.mockResolvedValue(null)
    api.predictWinLoss.mockResolvedValue({
      vendor_name: 'Zendesk',
      win_probability: 0.72,
      confidence: 'high',
      verdict: 'Strong displacement likelihood based on current signals.',
      is_gated: false,
      data_gates: [],
      factors: [
        {
          name: 'Switching intent',
          score: 0.8,
          weight: 0.2,
          evidence: 'Recent displacement intent is elevated.',
          data_points: 12,
          gated: false,
        },
      ],
      switching_triggers: [],
      proof_quotes: [],
      objections: [],
      displacement_targets: [],
      segment_match: null,
      data_coverage: { reviews: 12, quotes: 4 },
      weights_source: 'default',
      calibration_version: null,
      recommended_approach: 'Lead with support pain and switching urgency.',
      lead_with: ['support pain'],
      talking_points: ['Focus on defector urgency.'],
      timing_advice: 'Act within the current renewal window.',
      risk_factors: ['Incumbent discounting may slow the cycle.'],
      prediction_id: 'pred-1',
    })
  })

  it('hydrates predictor controls from the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/predictor?vendor=Zendesk&compare=1&vendor_b=Freshdesk&company_size=smb&industry=fintech']}>
        <WinLossPredictor />
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Win/Loss Predictor' })).toBeInTheDocument()
    expect(screen.getByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Freshdesk')).toBeInTheDocument()
    expect(screen.getByRole('combobox')).toHaveValue('smb')
    expect(screen.getByDisplayValue('fintech')).toBeInTheDocument()
  })

  it('renders the gated banner and suppresses easier-target claim when comparison.is_gated', async () => {
    const user = userEvent.setup()

    api.compareWinLoss.mockResolvedValueOnce({
      vendor_a: {
        vendor_name: 'Zendesk',
        win_probability: 0.62,
        confidence: 'medium',
        verdict: 'Moderate.',
        is_gated: false,
        data_gates: [],
        factors: [],
        switching_triggers: [],
        proof_quotes: [],
        objections: [],
        displacement_targets: [],
        segment_match: null,
        data_coverage: { reviews: 12 },
        weights_source: 'default',
        calibration_version: null,
        recommended_approach: null,
        lead_with: [],
        talking_points: [],
        timing_advice: null,
        risk_factors: [],
        prediction_id: 'pred-a',
      },
      vendor_b: {
        vendor_name: 'Freshdesk',
        win_probability: null,
        confidence: 'insufficient',
        verdict: 'Insufficient data.',
        is_gated: true,
        data_gates: [],
        factors: [],
        switching_triggers: [],
        proof_quotes: [],
        objections: [],
        displacement_targets: [],
        segment_match: null,
        data_coverage: {},
        weights_source: 'default',
        calibration_version: null,
        recommended_approach: null,
        lead_with: [],
        talking_points: [],
        timing_advice: null,
        risk_factors: [],
        prediction_id: 'pred-b',
      },
      easier_target: 'insufficient_data',
      probability_delta: 0,
      factor_comparison: [],
      is_gated: true,
      gated_reason: 'Insufficient data for Freshdesk; comparison cannot be drawn.',
    })

    render(
      <MemoryRouter
        initialEntries={['/predictor?vendor=Zendesk&compare=1&vendor_b=Freshdesk']}
      >
        <WinLossPredictor />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Win/Loss Predictor' })
    await user.click(screen.getByRole('button', { name: 'Predict Win Probability' }))

    await waitFor(() => {
      expect(api.compareWinLoss).toHaveBeenCalled()
    })

    const banner = await screen.findByTestId('compare-gated-banner')
    expect(banner).toHaveTextContent('Insufficient data for reliable comparison')
    expect(banner).toHaveTextContent(
      'Insufficient data for Freshdesk; comparison cannot be drawn.',
    )
    expect(screen.queryByText('Easier target:')).not.toBeInTheDocument()
    expect(screen.queryByText('probability advantage')).not.toBeInTheDocument()
    // Per-vendor "Easier" badge in the side-by-side gauges should not appear
    // because no vendor_name matches the literal "insufficient_data" target.
    expect(screen.queryByText('Easier')).not.toBeInTheDocument()
  })

  it('links single-vendor results back into the primary workflows', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/predictor?vendor=Zendesk&company_size=smb&industry=fintech']}>
        <WinLossPredictor />
      </MemoryRouter>,
    )

    await screen.findByRole('heading', { name: 'Win/Loss Predictor' })
    await user.click(screen.getByRole('button', { name: 'Predict Win Probability' }))

    await waitFor(() => {
      expect(api.predictWinLoss).toHaveBeenCalledWith({
        vendor_name: 'Zendesk',
        company_size: 'smb',
        industry: 'fintech',
      })
    })

    expect(await screen.findByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fpredictor%3Fvendor%3DZendesk%26company_size%3Dsmb%26industry%3Dfintech',
    )
  })
})
