import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BriefingReview from './BriefingReview'

const api = vi.hoisted(() => ({
  fetchBriefingReviewQueue: vi.fn(),
  fetchBriefingReviewSummary: vi.fn(),
  bulkApproveBriefings: vi.fn(),
  bulkRejectBriefings: vi.fn(),
  downloadBriefingsCsv: vi.fn(),
}))

vi.mock('../api/client', () => api)

function LocationEcho() {
  const location = useLocation()
  return <div>{`${location.pathname}${location.search}`}</div>
}

describe('BriefingReview', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    api.fetchBriefingReviewSummary.mockResolvedValue({
      pending_approval: 2,
      sent: 1,
      rejected: 0,
      failed: 0,
      oldest_pending_hours: 3,
    })
    api.fetchBriefingReviewQueue.mockResolvedValue({
      briefings: [
        {
          id: 'briefing-1',
          vendor_name: 'Acme Rival',
          recipient_email: 'owner@example.com',
          subject: 'Acme Rival weekly brief',
          briefing_html: '<p>Signal summary</p>',
          status: 'sent',
          target_mode: 'challenger_intel',
          created_at: '2026-04-10T03:00:00Z',
          approved_at: null,
          rejected_at: null,
          reject_reason: null,
        },
      ],
    })
    api.bulkApproveBriefings.mockResolvedValue({})
    api.bulkRejectBriefings.mockResolvedValue({})
    api.downloadBriefingsCsv.mockResolvedValue(undefined)
  })

  it('hydrates status and vendor focus from the URL and renders workflow links with back_to', async () => {
    render(
      <MemoryRouter initialEntries={['/briefing-review?status=sent&vendor=Acme+Rival']}>
        <BriefingReview />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Acme Rival')).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'Vendor workspace' })).toHaveAttribute(
      'href',
      '/vendors/Acme%20Rival?back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DAcme%2BRival',
    )
    expect(screen.getByRole('link', { name: 'Evidence' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Acme+Rival&tab=witnesses&back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DAcme%2BRival',
    )
    expect(screen.getByRole('link', { name: 'Reports' })).toHaveAttribute(
      'href',
      '/reports?vendor_filter=Acme+Rival&back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DAcme%2BRival',
    )
    expect(screen.getByRole('link', { name: 'Opportunities' })).toHaveAttribute(
      'href',
      '/opportunities?vendor=Acme+Rival&back_to=%2Fbriefing-review%3Fstatus%3Dsent%26vendor%3DAcme%2BRival',
    )
  })

  it('persists status and vendor focus into the URL when switching tabs and groups', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/briefing-review']}>
        <Routes>
          <Route path="/briefing-review" element={<BriefingReview />} />
          <Route path="*" element={<LocationEcho />} />
        </Routes>
        <LocationEcho />
      </MemoryRouter>,
    )

    expect(await screen.findByText('/briefing-review')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Acme Rival/i }))
    expect(await screen.findByText('/briefing-review?vendor=Acme+Rival')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Sent' }))
    expect(await screen.findByText('/briefing-review?status=sent&vendor=Acme+Rival')).toBeInTheDocument()
  })
})
