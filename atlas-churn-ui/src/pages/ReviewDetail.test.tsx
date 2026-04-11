import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReviewDetail from './ReviewDetail'

const mockNavigate = vi.hoisted(() => vi.fn())
const clipboard = vi.hoisted(() => ({
  writeText: vi.fn(),
}))

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
    Object.defineProperty(window.navigator, 'clipboard', {
      configurable: true,
      value: clipboard,
    })
    clipboard.writeText.mockResolvedValue(undefined)
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

  it('returns to the focused account review when back_to points at a watchlist account path', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Account Review' }))

    expect(mockNavigate).toHaveBeenCalledWith('/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor')
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

  it('returns to opportunities when back_to points at an opportunity workspace', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fopportunities%3Fvendor%3DZendesk%26back_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Opportunities' }))

    expect(mockNavigate).toHaveBeenCalledWith('/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1')
  })

  it('copies a shareable review detail link with preserved back context', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByTitle('Copy link'))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Copied' })).toBeInTheDocument()
    })
  })

  it('shows vendor workspace, evidence, opportunities, and reports shortcuts for the review vendor', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Zendesk?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
  })
})
