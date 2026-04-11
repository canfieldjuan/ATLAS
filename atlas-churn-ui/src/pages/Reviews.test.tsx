import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Reviews from './Reviews'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchReviews: vi.fn(),
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

describe('Reviews', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchReviews.mockResolvedValue({
      reviews: [
        {
          id: 'review-1',
          vendor_name: 'Zendesk',
          product_category: 'Helpdesk',
          reviewer_company: 'Acme Corp',
          rating: 2.5,
          urgency_score: 8.6,
          pain_category: 'support',
          intent_to_leave: true,
          decision_maker: true,
          source: 'g2',
          reviewed_at: '2026-04-10T00:00:00Z',
          role_level: 'vp',
          buying_stage: 'renewal_decision',
          sentiment_direction: 'declining',
          competitors_mentioned: ['Freshdesk'],
          quotable_phrases: [],
          positive_aspects: [],
          specific_complaints: [],
          enriched_at: '2026-04-10T00:00:00Z',
          reviewer_title: 'VP Support',
          company_size: null,
          industry: null,
        },
      ],
    })
  })

  it('hydrates filters from the URL and keeps vendor handoff links scoped back to the list', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews?vendor=Zendesk&company=Acme&min_urgency=6&churn_only=true']}>
        <Routes>
          <Route path="/reviews" element={<Reviews />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByDisplayValue('Zendesk')).toBeInTheDocument()
    expect(screen.getByDisplayValue('Acme')).toBeInTheDocument()
    await waitFor(() => {
      expect(api.fetchReviews).toHaveBeenLastCalledWith({
        vendor_name: 'Zendesk',
        company: 'Acme',
        min_urgency: 6,
        has_churn_intent: true,
        window_days: 365,
        limit: 100,
      })
    })

    expect(screen.getByRole('link', { name: 'Vendor' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Freviews%3Fvendor%3DZendesk%26company%3DAcme%26min_urgency%3D6%26churn_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%3Fvendor%3DZendesk%26company%3DAcme%26min_urgency%3D6%26churn_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Freviews%3Fvendor%3DZendesk%26company%3DAcme%26min_urgency%3D6%26churn_only%3Dtrue',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Freviews%3Fvendor%3DZendesk%26company%3DAcme%26min_urgency%3D6%26churn_only%3Dtrue',
    )
  })

  it('navigates row clicks to review detail with the current list state as back_to', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews?vendor=Zendesk&company=Acme&min_urgency=6&churn_only=true']}>
        <Routes>
          <Route path="/reviews" element={<Reviews />} />
        </Routes>
      </MemoryRouter>,
    )

    const vendorCell = await screen.findByText('Zendesk')
    await user.click(vendorCell)

    expect(mockNavigate).toHaveBeenCalledWith(
      '/reviews/review-1?back_to=%2Freviews%3Fvendor%3DZendesk%26company%3DAcme%26min_urgency%3D6%26churn_only%3Dtrue',
    )
  })

  it('updates the URL-backed handoff links after the applied vendor filter changes', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews']}>
        <Routes>
          <Route path="/reviews" element={<Reviews />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText('Zendesk')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Filter by vendor...'), {
      target: { value: 'Zendesk' },
    })

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
        'href',
        '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%3Fvendor%3DZendesk',
      )
    })
  })
})
