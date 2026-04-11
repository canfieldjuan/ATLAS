import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AccountMovementDrawer from './AccountMovementDrawer'
import type { AccountsInMotionFeedItem } from '../api/client'

const baseItem: AccountsInMotionFeedItem = {
  source_reviews: [{
    id: 'review-1',
    source: 'g2',
    source_url: 'https://example.com/review-1',
    vendor_name: 'Zendesk',
    rating: 2,
    summary: 'Support fell apart',
    review_excerpt: 'Renewal warning from finance review.',
    reviewer_name: 'Taylor',
    reviewer_title: 'VP Support',
    reviewer_company: 'Acme Corp',
    reviewed_at: '2026-04-03T00:00:00Z',
  }],
  company: 'Acme Corp',
  vendor: 'Zendesk',
  watch_vendor: 'Zendesk',
  track_mode: 'competitor',
  watchlist_label: 'Support',
  category: 'Helpdesk',
  urgency: 8.8,
  role_type: 'executive',
  buying_stage: 'evaluation',
  budget_authority: true,
  pain_categories: [{ category: 'pricing', severity: 'high' }],
  evidence: ['Renewal warning from finance review.'],
  alternatives_considering: [{ name: 'Freshdesk', reason: 'pricing' }],
  contract_signal: 'Q3 2026',
  reviewer_title: 'VP Support',
  company_size_raw: '500',
  quality_flags: ['needs_company_resolution'],
  opportunity_score: 84,
  quote_match_type: 'company_match',
  confidence: 7.5,
  reasoning_reference_ids: {
    witness_ids: ['witness:zendesk:1'],
    metric_ids: ['metric:renewal-risk'],
  },
  source_distribution: { reddit: 2, g2: 1 },
  source_review_ids: ['review-1'],
  evidence_count: 1,
  enriched_at: '2026-04-06T10:00:00Z',
  employee_count: 500,
  industry: 'SaaS',
  annual_revenue: '$10M-$50M',
  domain: 'acme.com',
  contacts: [],
  contact_count: 0,
  report_date: '2026-04-05',
  stale_days: 2,
  is_stale: true,
  data_source: 'persisted_report',
  freshness_status: 'stale',
  freshness_reason: 'older_than_threshold',
  freshness_timestamp: '2026-04-06T10:00:00Z',
  account_alert_hit: true,
  stale_threshold_hit: true,
}

describe('AccountMovementDrawer', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders nothing when closed', () => {
    const { container } = render(
      <MemoryRouter>
        <AccountMovementDrawer
          item={baseItem}
          open={false}
          onClose={() => {}}
          onViewVendor={() => {}}
        />
      </MemoryRouter>,
    )

    expect(container).toBeEmptyDOMElement()
  })

  it('closes from the explicit close button, escape key, and backdrop click', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const { container } = render(
      <MemoryRouter>
        <AccountMovementDrawer
          item={baseItem}
          open
          onClose={onClose}
          onViewVendor={() => {}}
        />
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('button', { name: 'Close account movement evidence' }))
    await user.keyboard('{Escape}')
    const backdrop = container.querySelector('[aria-hidden="true"]')
    expect(backdrop).not.toBeNull()
    await user.click(backdrop as HTMLElement)

    expect(onClose).toHaveBeenCalledTimes(3)
  })

  it('fires the vendor, witness, review, campaign, opportunity, and copy actions', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onViewVendor = vi.fn()
    const onOpenWitness = vi.fn()
    const onGenerateCampaign = vi.fn()
    const onViewOpportunity = vi.fn()
    const onViewReview = vi.fn()
    const onCopyLink = vi.fn()

    render(
      <MemoryRouter>
        <AccountMovementDrawer
          item={baseItem}
          open
          onClose={onClose}
          onViewVendor={onViewVendor}
          onOpenWitness={onOpenWitness}
          onGenerateCampaign={onGenerateCampaign}
          onViewOpportunity={onViewOpportunity}
          onViewReview={onViewReview}
          onCopyLink={onCopyLink}
          evidenceExplorerUrl="/evidence?vendor=Zendesk"
        />
      </MemoryRouter>,
    )

    const drawer = screen.getByLabelText('Account movement evidence')
    expect(within(drawer).getByText('Renewal warning from finance review.')).toBeInTheDocument()
    expect(within(drawer).getByText('needs company resolution')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Copy link' }))
    await user.click(screen.getByRole('button', { name: 'View vendor' }))
    await user.click(screen.getByRole('button', { name: 'View opportunities' }))
    await user.click(screen.getByRole('button', { name: 'Generate campaigns' }))
    await user.click(screen.getByRole('button', { name: 'Open review detail' }))
    await user.click(screen.getByRole('button', { name: 'witness:zendesk:1' }))

    expect(screen.getByRole('link', { name: 'Evidence Explorer' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk',
    )

    expect(onCopyLink).toHaveBeenCalledTimes(1)
    expect(onViewVendor).toHaveBeenCalledWith('Zendesk')
    expect(onViewOpportunity).toHaveBeenCalledWith(baseItem)
    expect(onGenerateCampaign).toHaveBeenCalledWith(baseItem)
    expect(onViewReview).toHaveBeenCalledWith('review-1')
    expect(onOpenWitness).toHaveBeenCalledWith('witness:zendesk:1', 'Zendesk')
  })

  it('fires the report actions and shows the persisted report badge', async () => {
    const user = userEvent.setup()
    const onViewReport = vi.fn()
    const onCopyReportLink = vi.fn()

    render(
      <MemoryRouter>
        <AccountMovementDrawer
          item={{ ...baseItem, is_stale: false }}
          open
          onClose={() => {}}
          onViewVendor={() => {}}
          onViewReport={onViewReport}
          onCopyReportLink={onCopyReportLink}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('persisted report')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'View reports' }))
    await user.click(screen.getByRole('button', { name: 'Copy reports' }))

    expect(onViewReport).toHaveBeenCalledWith(expect.objectContaining({ vendor: 'Zendesk' }))
    expect(onCopyReportLink).toHaveBeenCalledWith(expect.objectContaining({ vendor: 'Zendesk' }))
  })


  it('shows the generating state on the campaign action', () => {
    render(
      <MemoryRouter>
        <AccountMovementDrawer
          item={baseItem}
          open
          onClose={() => {}}
          onViewVendor={() => {}}
          onGenerateCampaign={() => {}}
          generating
        />
      </MemoryRouter>,
    )

    expect(screen.getByRole('button', { name: 'Generating...' })).toBeDisabled()
  })
})
