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

  it('returns to report detail when back_to points at a report detail page', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Freports%2Freport-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Report' }))

    expect(mockNavigate).toHaveBeenCalledWith('/reports/report-1?back_to=%2Fwatchlists%3Fview%3Dview-1')
  })

  it('returns to the public report when back_to points at the public report page', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Report' }))

    expect(mockNavigate).toHaveBeenCalledWith('/report?vendor=Zendesk&ref=test-token&mode=view')
  })

  it('returns to pipeline review when back_to points at the grouped review queue', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fpipeline-review%3Fqueue_vendor%3DZendesk']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Back to Pipeline Review' }))

    expect(mockNavigate).toHaveBeenCalledWith('/pipeline-review?queue_vendor=Zendesk')
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

  it('preserves witness drilldown state on the evidence shortcut when entered from evidence explorer', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26as_of_date%3D2026-04-08%26window_days%3D45%26pain_category%3Dpricing%26source%3Dreddit%26witness_type%3Dpricing%26offset%3D30%26witness_id%3Dwitness%253Azendesk%253A1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&pain_category=pricing&witness_type=pricing&offset=30&as_of_date=2026-04-08&window_days=45&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fevidence%253Fvendor%253DZendesk%2526tab%253Dwitnesses%2526as_of_date%253D2026-04-08%2526window_days%253D45%2526pain_category%253Dpricing%2526source%253Dreddit%2526witness_type%253Dpricing%2526offset%253D30%2526witness_id%253Dwitness%25253Azendesk%25253A1',
    )
  })

  it('preserves watchlist snapshot dates on the evidence shortcut from account review context', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-05&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Faccount_vendor%253DZendesk%2526account_company%253DAcme%252BCorp%2526account_report_date%253D2026-04-05%2526account_watch_vendor%253DZendesk%2526account_category%253DHelpdesk%2526account_track_mode%253Dcompetitor',
    )
  })

  it('shows a direct watchlists shortcut for generic watchlist context', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1%26vendor_name%3DZendesk']}> 
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1&vendor_name=Zendesk',
    )
  })

  it('surfaces the watchlists shortcut through nested evidence context', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26back_to%3D%252Fwatchlists%253Fview%253Dview-1%2526vendor_name%253DZendesk']}> 
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Watchlists' })).toHaveAttribute(
      'href',
      '/watchlists?view=view-1&vendor_name=Zendesk',
    )
  })

  it('shows a direct account review shortcut for focused watchlist context', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor',
    )
  })

  it('surfaces the account review shortcut through nested evidence context', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26witness_id%3Dwitness%253Azendesk%253A1%26source%3Dreddit%26back_to%3D%252Fwatchlists%253Faccount_vendor%253DZendesk%2526account_company%253DAcme%252BCorp%2526account_report_date%253D2026-04-05%2526account_watch_vendor%253DZendesk%2526account_category%253DHelpdesk%2526account_track_mode%253Dcompetitor']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Account Review' })).toHaveAttribute(
      'href',
      '/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor',
    )
  })

  it('copies the direct watchlists shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1%26vendor_name%3DZendesk']}> 
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy watchlists link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?view=view-1&vendor_name=Zendesk`,
      )
    })
  })

  it('copies the direct account review shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Faccount_vendor%3DZendesk%26account_company%3DAcme%2BCorp%26account_report_date%3D2026-04-05%26account_watch_vendor%3DZendesk%26account_category%3DHelpdesk%26account_track_mode%3Dcompetitor']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy account review link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/watchlists?account_vendor=Zendesk&account_company=Acme+Corp&account_report_date=2026-04-05&account_watch_vendor=Zendesk&account_category=Helpdesk&account_track_mode=competitor`,
      )
    })
  })

  it('copies the evidence shortcut link with preserved upstream witness context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fevidence%3Fvendor%3DZendesk%26tab%3Dwitnesses%26pain_category%3Dpricing%26source%3Dreddit%26witness_type%3Dpricing%26offset%3D30%26witness_id%3Dwitness%253Azendesk%253A1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy evidence link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/evidence?vendor=Zendesk&tab=witnesses&witness_id=witness%3Azendesk%3A1&source=reddit&pain_category=pricing&witness_type=pricing&offset=30&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fevidence%253Fvendor%253DZendesk%2526tab%253Dwitnesses%2526pain_category%253Dpricing%2526source%253Dreddit%2526witness_type%253Dpricing%2526offset%253D30%2526witness_id%253Dwitness%25253Azendesk%25253A1`,
      )
    })
  })

  it('prefers the exact upstream alerts shortcut through nested evidence context', async () => {
    const directAlertsPath = '/alerts?webhook=wh-crm&window=30d'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directAlertsPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Alerts API' })).toHaveAttribute('href', directAlertsPath)
  })

  it('copies the exact upstream alerts shortcut through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directAlertsPath = '/alerts?webhook=wh-crm&window=30d'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directAlertsPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy alerts link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directAlertsPath}`)
    })
  })

  it('prefers the exact upstream vendor workspace shortcut through nested evidence context', async () => {
    const directVendorPath = '/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directVendorPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Vendor workspace' })).toHaveAttribute('href', directVendorPath)
  })

  it('copies the exact upstream vendor workspace shortcut through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directVendorPath = '/vendors/Zendesk?back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directVendorPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy vendor workspace link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directVendorPath}`)
    })
  })

  it('prefers the exact upstream reports shortcut through nested evidence context', async () => {
    const directReportsPath = '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directReportsPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute('href', directReportsPath)
  })

  it('copies the exact upstream reports shortcut through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directReportsPath = '/reports?vendor_filter=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directReportsPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy reports link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directReportsPath}`)
    })
  })

  it('prefers the exact upstream opportunities shortcut through nested evidence context', async () => {
    const directOpportunitiesPath = '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directOpportunitiesPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute('href', directOpportunitiesPath)
  })

  it('copies the exact upstream opportunities shortcut through nested evidence context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)
    const directOpportunitiesPath = '/opportunities?vendor=Zendesk&back_to=%2Fwatchlists%3Fview%3Dview-1'
    const nestedEvidencePath = `/evidence?vendor=Zendesk&tab=witnesses&back_to=${encodeURIComponent(directOpportunitiesPath)}`

    render(
      <MemoryRouter initialEntries={[`/reviews/review-1?back_to=${encodeURIComponent(nestedEvidencePath)}`]}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy opportunities link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(`${window.location.origin}${directOpportunitiesPath}`)
    })
  })

  it('copies the vendor workspace shortcut link with preserved back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy vendor workspace link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/vendors/Zendesk?back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
  })

  it('copies the reports shortcut link with preserved back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy reports link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/reports?vendor_filter=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
  })

  it('copies the opportunities shortcut link with preserved back context', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy opportunities link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/opportunities?vendor=Zendesk&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
  })

  it('shows alerts, vendor workspace, evidence, opportunities, and reports shortcuts for the review vendor', async () => {
    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Alerts API' })).toHaveAttribute(
      'href',
      '/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1',
    )
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

  it('copies the generated alerts shortcut link', async () => {
    const user = userEvent.setup()
    const clipboardSpy = vi.spyOn(window.navigator.clipboard, 'writeText').mockResolvedValue(undefined)

    render(
      <MemoryRouter initialEntries={['/reviews/review-1?back_to=%2Fwatchlists%3Fview%3Dview-1']}>
        <Routes>
          <Route path="/reviews/:id" element={<ReviewDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByRole('heading', { name: 'Zendesk' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy alerts link' }))

    await waitFor(() => {
      expect(clipboardSpy).toHaveBeenCalledWith(
        `${window.location.origin}/alerts?vendor=Zendesk&company=Acme+Corp&back_to=%2Freviews%2Freview-1%3Fback_to%3D%252Fwatchlists%253Fview%253Dview-1`,
      )
    })
  })
})
