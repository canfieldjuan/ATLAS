import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, RouterProvider, createMemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Challengers from './Challengers'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchVendorTargets: vi.fn(),
  fetchHighIntent: vi.fn(),
  fetchChallengerClaims: vi.fn(),
  generateCampaigns: vi.fn(),
  downloadCsv: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

function challengerClaim(overrides: Record<string, unknown> = {}) {
  return {
    claim_id: 'claim-1',
    claim_key: 'incumbent:Freshdesk',
    claim_scope: 'competitor_pair',
    claim_type: 'direct_displacement',
    claim_text: 'Freshdesk shows direct displacement pressure toward Zendesk',
    target_entity: 'Zendesk',
    secondary_target: 'Freshdesk',
    supporting_count: 3,
    direct_evidence_count: 3,
    witness_count: 2,
    contradiction_count: 0,
    denominator: null,
    sample_size: 3,
    has_grounded_evidence: true,
    confidence: 'medium',
    evidence_posture: 'usable',
    render_allowed: true,
    report_allowed: false,
    suppression_reason: 'low_confidence',
    evidence_links: ['review-1'],
    contradicting_links: [],
    as_of_date: '2026-04-26',
    analysis_window_days: 30,
    schema_version: 'product_claim.v1',
    ...overrides,
  }
}

function challengerClaimsResponse(rows: Array<Record<string, unknown>> = []) {
  return {
    challenger: 'Zendesk',
    as_of_date: '2026-04-26',
    analysis_window_days: 30,
    rows: rows.map((row) => ({
      challenger: 'Zendesk',
      incumbent: String(row.incumbent ?? 'Freshdesk'),
      claim: challengerClaim(row.claim as Record<string, unknown> | undefined),
    })),
  }
}

describe('Challengers', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchVendorTargets.mockResolvedValue({
      targets: [
        {
          id: 'target-1',
          company_name: 'Zendesk',
          competitors_tracked: ['Freshdesk'],
        },
      ],
    })
    api.fetchHighIntent.mockResolvedValue({
      companies: [
        {
          id: 'company-1',
          company_name: 'Acme Corp',
          vendor: 'Freshdesk',
          alternatives: [{ name: 'Zendesk' }],
          buying_stage: 'active_purchase',
          pain: 'support',
          urgency: 8,
        },
      ],
    })
    api.fetchChallengerClaims.mockResolvedValue(challengerClaimsResponse())
    api.generateCampaigns.mockResolvedValue({ generated: 1 })
  })

  it('hydrates search from the URL and keeps direct handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/challengers?search=Zendesk']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(api.fetchVendorTargets).toHaveBeenCalledWith({ target_mode: 'challenger_intel', limit: 100 })
    expect(api.fetchHighIntent).toHaveBeenCalledWith({ min_urgency: 3, limit: 100 })
    expect(api.fetchChallengerClaims).toHaveBeenCalledWith('Zendesk')

    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
  })

  it('navigates row clicks to vendor detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/challengers?search=Zendesk']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Fchallengers%3Fsearch%3DZendesk',
    )
  })

  it('updates the URL-backed handoff links after the applied search filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Search challenger...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
        'href',
        '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Fchallengers%3Fsearch%3DZendesk',
      )
    })
  })

  it('clears same-route list filters without restoring stale query params', async () => {
    const router = createMemoryRouter(
      [{ path: '/challengers', element: <><Challengers /><LocationProbe /></> }],
      {
        initialEntries: ['/challengers?search=Zendesk'],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()

    await router.navigate('/challengers')

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search challenger...')).toHaveValue('')
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/challengers')
    })
  })

  it('canonicalizes invalid route filters on load', async () => {
    const router = createMemoryRouter(
      [{ path: '/challengers', element: <><Challengers /><LocationProbe /></> }],
      {
        initialEntries: ['/challengers?search=%20Zendesk%20'],
      },
    )

    render(<RouterProvider router={router} />)

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByTestId('location-probe')).toHaveTextContent('/challengers?search=Zendesk')
    })
  })

  it('fails closed when challenger claim validation is unavailable', async () => {
    api.fetchChallengerClaims.mockRejectedValueOnce(new Error('claims unavailable'))

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText(/Validation unavailable for 1 challenger row/)).toBeInTheDocument()
    expect(screen.getAllByText('Validation unavailable').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /Validation unavailable: claim service/ })).toBeDisabled()
    expect(api.generateCampaigns).not.toHaveBeenCalled()
  })

  it('suppresses winner-call fields when the direct displacement claim is not render-safe', async () => {
    api.fetchChallengerClaims.mockResolvedValueOnce(challengerClaimsResponse([
      {
        incumbent: 'Freshdesk',
        claim: {
          render_allowed: false,
          report_allowed: false,
          suppression_reason: 'unverified_evidence',
          evidence_posture: 'unverified',
          confidence: 'low',
        },
      },
    ]))

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findAllByText('Insufficient')).not.toHaveLength(0)
    expect(screen.queryByText('Freshdesk')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Unverified evidence/ })).toBeDisabled()
  })

  it('renders monitor-only challenger evidence but blocks campaign generation', async () => {
    api.fetchChallengerClaims.mockResolvedValueOnce(challengerClaimsResponse([
      {
        incumbent: 'Freshdesk',
        claim: {
          render_allowed: true,
          report_allowed: false,
          suppression_reason: 'low_confidence',
          confidence: 'low',
          witness_count: 1,
        },
      },
    ]))

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Monitor only')).toBeInTheDocument()
    expect(screen.getByText('Freshdesk')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Low confidence/ })).toBeDisabled()
  })

  it('allows campaign generation only when the challenger claim is report-safe', async () => {
    const user = userEvent.setup()
    api.fetchChallengerClaims.mockResolvedValueOnce(challengerClaimsResponse([
      {
        incumbent: 'Freshdesk',
        claim: {
          render_allowed: true,
          report_allowed: true,
          suppression_reason: null,
        },
      },
    ]))

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Report-safe')).toBeInTheDocument()
    await user.click(screen.getByTitle('Generate Campaign'))

    expect(api.generateCampaigns).toHaveBeenCalledWith({
      vendor_name: 'Zendesk',
      target_mode: 'challenger_intel',
      min_score: 50,
      limit: 5,
    })
  })

  it('does not inflate validated lead metrics from tracked-incumbent rows alone', async () => {
    api.fetchHighIntent.mockResolvedValueOnce({
      companies: [
        {
          id: 'company-safe',
          company_name: 'Acme Corp',
          vendor: 'Freshdesk',
          alternatives: [{ name: 'Zendesk' }],
          buying_stage: 'evaluation',
          pain: 'support',
          urgency: 8,
        },
        {
          id: 'company-unsafe',
          company_name: 'Globex',
          vendor: 'Freshdesk',
          alternatives: [{ name: 'Intercom' }],
          buying_stage: 'active_purchase',
          pain: 'pricing',
          urgency: 10,
        },
      ],
    })
    api.fetchChallengerClaims.mockResolvedValueOnce(challengerClaimsResponse([
      {
        incumbent: 'Freshdesk',
        claim: {
          render_allowed: true,
          report_allowed: true,
          suppression_reason: null,
        },
      },
    ]))

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Report-safe')).toBeInTheDocument()
    const totalIntentCard = screen.getByText('Total Intent Leads').parentElement?.parentElement
    expect(totalIntentCard).not.toBeNull()
    expect(within(totalIntentCard as HTMLElement).getByText('1')).toBeInTheDocument()
    expect(screen.queryByText('2')).not.toBeInTheDocument()
  })

  it('marks absent challenger claims as legacy but suppresses winner-call values and campaign actions', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/challengers']}>
        <Routes>
          <Route path="/challengers" element={<Challengers />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Legacy')).toBeInTheDocument()
    expect(screen.getAllByText('Insufficient').length).toBeGreaterThan(0)
    expect(screen.queryByText('Freshdesk')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Legacy row: validation claim unavailable/ }))

    expect(api.generateCampaigns).not.toHaveBeenCalled()
  })
})
