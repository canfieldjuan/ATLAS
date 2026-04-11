import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReviewDetail from './ReviewDetail'

const mockNavigate = vi.hoisted(() => vi.fn())

const api = vi.hoisted(() => ({
  fetchReview: vi.fn(),
}))

vi.mock('../api/client', () => api)

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('ReviewDetail', () => {
  afterEach(() => {
    cleanup()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchReview.mockResolvedValue({
      id: 'review-1',
      vendor_name: 'Zendesk',
      reviewer_company: 'Acme Corp',
      reviewer_title: 'VP Support',
      rating: 2.5,
      source: 'g2',
      source_url: 'https://example.com/review-1',
      review_text: 'Support quality dropped at renewal.',
      pros: null,
      cons: 'Support quality dropped at renewal.',
      enrichment: {
        urgency_score: 8.6,
        pain_category: 'support',
        churn_signals: { intent_to_leave: true },
        reviewer_context: { decision_maker: true, role_level: 'vp' },
      },
    })
  })

  it('returns to the vendor workspace when back_to points at a vendor detail page', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fvendors%2FZendesk']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Vendor' }))

    expect(mockNavigate).toHaveBeenCalledWith('/vendors/Zendesk')
  })

  it('returns to evidence explorer when back_to points at an evidence workspace', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26source%3Dreddit']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Evidence' }))

    expect(mockNavigate).toHaveBeenCalledWith('/evidence?vendor=Zendesk&tab=witnesses&source=reddit')
  })
})
