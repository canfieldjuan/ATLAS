import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Affiliates from './Affiliates'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchAffiliateOpportunities: vi.fn(),
  fetchAffiliatePartners: vi.fn(),
  fetchClickSummary: vi.fn(),
  createAffiliatePartner: vi.fn(),
  updateAffiliatePartner: vi.fn(),
  deleteAffiliatePartner: vi.fn(),
  recordAffiliateClick: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Affiliates', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchAffiliateOpportunities.mockResolvedValue({
      opportunities: [
        {
          id: 'opp-1',
          opportunity_score: 82,
          vendor_name: 'Zendesk',
          reviewer_company: 'Acme Corp',
          reviewer_company_display: null,
          competitor_name: 'Freshdesk',
          mention_context: 'Switch evaluation',
          mention_reason: 'support',
          urgency: 8,
          is_dm: true,
          buying_stage: 'evaluation',
          seat_count: 120,
          affiliate_url: 'https://partner.example/zendesk',
          partner_id: 'partner-1',
          review_id: 'review-1',
        },
      ],
    })
    api.fetchAffiliatePartners.mockResolvedValue({ partners: [] })
    api.fetchClickSummary.mockResolvedValue({ clicks: [] })
    api.createAffiliatePartner.mockResolvedValue({})
    api.updateAffiliatePartner.mockResolvedValue({})
    api.deleteAffiliatePartner.mockResolvedValue({})
    api.recordAffiliateClick.mockResolvedValue({})
  })

  it('hydrates filters from the URL and keeps vendor handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/affiliates?vendor=Zendesk&min_urgency=7&min_score=80&dm_only=true']}>
        <Routes>
          <Route path="/affiliates" element={<Affiliates />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchAffiliateOpportunities).toHaveBeenLastCalledWith({
        min_urgency: 7,
        min_score: 80,
        vendor_name: 'Zendesk',
        dm_only: true,
        limit: 100,
      })
    })

    expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
    )
  })

  it('navigates row clicks to vendor detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/affiliates?vendor=Zendesk&min_urgency=7&min_score=80&dm_only=true']}>
        <Routes>
          <Route path="/affiliates" element={<Affiliates />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/vendors/Zendesk?back_to=%2Faffiliates%3Fvendor%3DZendesk%26min_urgency%3D7%26min_score%3D80%26dm_only%3Dtrue',
    )
  })

  it('updates the URL-backed handoff links after the applied vendor filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/affiliates']}>
        <Routes>
          <Route path="/affiliates" element={<Affiliates />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Filter vendor...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
        'href',
        '/vendors/Zendesk?back_to=%2Faffiliates%3Fvendor%3DZendesk',
      )
    })
  })
})
