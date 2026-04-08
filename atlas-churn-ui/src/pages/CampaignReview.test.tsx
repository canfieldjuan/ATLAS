import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CampaignReview from './CampaignReview'

const api = vi.hoisted(() => ({
  bulkApproveCampaigns: vi.fn(),
  bulkRejectCampaigns: vi.fn(),
  fetchCampaignAuditLog: vi.fn(),
  fetchCampaignQualityTrends: vi.fn(),
  fetchReviewQueue: vi.fn(),
  fetchReviewQueueSummary: vi.fn(),
  updateCampaign: vi.fn(),
}))

vi.mock('../api/client', () => api)

describe('CampaignReview', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchReviewQueueSummary.mockResolvedValue({
      pending_review: 1,
      ready_to_send: 0,
      pending_recipient: 0,
      suppressed: 0,
      quality_fail: 0,
      quality_pass: 1,
      quality_missing: 0,
      blocker_total: 0,
      top_blockers: [],
      by_boundary: [],
    })
    api.fetchCampaignQualityTrends.mockResolvedValue({
      days: 14,
      top_n: 5,
      top_blockers: [],
      series: [],
      totals_by_day: [],
    })
    api.fetchReviewQueue.mockResolvedValue({
      drafts: [
        {
          id: 'draft-1',
          company_name: 'Acme Corp',
          vendor_name: 'Zendesk',
          channel: 'email',
          subject: 'Initial subject',
          body: '<p>Initial body</p>',
          cta: 'Reply here',
          status: 'draft',
          step_number: 1,
          recipient_email: 'owner@acme.com',
          seq_recipient: null,
          partner_name: null,
          product_name: null,
          is_suppressed: 0,
          seq_status: null,
          current_step: null,
          max_steps: null,
          open_count: null,
          click_count: null,
          target_persona: 'economic_buyer',
          prospect_first_name: 'Taylor',
          prospect_last_name: 'Lee',
          prospect_title: 'VP Support',
          prospect_seniority: 'vp',
          prospect_email_status: null,
          created_at: '2026-04-07T10:00:00Z',
          quality_status: 'pass',
          blocker_count: 0,
          warning_count: 0,
          latest_error_summary: null,
          failure_explanation: null,
        },
      ],
    })
    api.fetchCampaignAuditLog.mockResolvedValue({ events: [] })
    api.updateCampaign.mockResolvedValue({ ok: true })
  })

  it('sends cleared fields as empty strings when editing a draft', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter initialEntries={['/campaign-review?company=Acme%20Corp']}>
        <CampaignReview />
      </MemoryRouter>,
    )

    await screen.findByText('Initial subject')
    await user.click(screen.getByTitle('Edit content'))

    const textboxes = await screen.findAllByRole('textbox')
    const [subjectInput, bodyInput, ctaInput] = textboxes

    await user.clear(subjectInput)
    await user.clear(bodyInput)
    await user.clear(ctaInput)
    await user.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => {
      expect(api.updateCampaign).toHaveBeenCalledWith('draft-1', {
        subject: '',
        body: '',
        cta: '',
      })
    })
  })
})
