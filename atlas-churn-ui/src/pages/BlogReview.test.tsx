import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BlogReview from './BlogReview'

const api = vi.hoisted(() => ({
  fetchBlogDrafts: vi.fn(),
  fetchBlogDraftSummary: vi.fn(),
  fetchBlogDraft: vi.fn(),
  fetchBlogEvidence: vi.fn(),
  publishBlogDraft: vi.fn(),
  updateBlogDraft: vi.fn(),
}))

vi.mock('../api/client', () => api)

const draftSummary = {
  id: 'draft-1',
  slug: 'switch-to-shopify-2026-03',
  title: 'Migration Guide: Why Teams Are Switching to Shopify',
  topic_type: 'migration_guide',
  status: 'draft',
  llm_model: 'openai/gpt-oss-120b',
  created_at: '2026-03-30T18:00:00Z',
  published_at: null,
  rejected_at: null,
  rejection_reason: null,
  quality_score: 94,
  quality_threshold: 70,
  blocker_count: 0,
  warning_count: 1,
  latest_failure_step: null,
  latest_error_code: null,
  latest_error_summary: null,
  unresolved_issue_count: 0,
}

const draftDetail = {
  ...draftSummary,
  description: 'Shopify migration guide',
  tags: ['shopify', 'migration'],
  content: '<p><a href="https://example.com/shopify">Shopify trial</a></p>',
  charts: [],
  data_context: {
    affiliate_url: 'https://example.com/shopify',
    affiliate_partner: { name: 'Shopify' },
  },
  reviewer_notes: null,
  source_report_date: '2026-03-30',
  seo_title: 'SEO title',
  seo_description: 'SEO desc',
  target_keyword: 'switch to shopify',
  secondary_keywords: ['shopify migration'],
  faq: [],
  related_slugs: [],
  cta: {
    headline: 'See the full migration brief',
    body: 'Get the full report before renewal.',
    button_text: 'Book briefing',
    report_type: 'migration_guide',
  },
}

const draftRollup = {
  by_status: { draft: 1 },
  quality: {
    clean: 1,
    warning_only: 0,
    failing: 0,
    unresolved: 0,
    blocker_total: 0,
    warning_total: 1,
    by_failure_step: [],
    top_blockers: [],
  },
}

describe('BlogReview preview mode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.fetchBlogDrafts.mockResolvedValue([draftSummary])
    api.fetchBlogDraftSummary.mockResolvedValue(draftRollup)
    api.fetchBlogDraft.mockResolvedValue(draftDetail)
    api.fetchBlogEvidence.mockResolvedValue({ reviews: [], count: 0 })
    api.publishBlogDraft.mockResolvedValue({ ok: true, id: 'draft-1', slug: draftSummary.slug, published_at: '2026-03-30T19:00:00Z' })
    api.updateBlogDraft.mockResolvedValue({ ok: true, id: 'draft-1' })
  })

  it('shows rendered affiliate placements and CTA details in preview mode', async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <BlogReview />
      </MemoryRouter>,
    )

    await user.click(await screen.findByText(draftSummary.title))

    await screen.findByText('Reviewer Notes')
    await user.click(screen.getByRole('button', { name: 'Preview' }))

    await waitFor(() =>
      expect(screen.getByText('Prepublish Map')).toBeInTheDocument(),
    )

    expect(screen.getByText('Affiliate Placements')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getAllByText('Rendered CTA').length).toBeGreaterThan(0)
    expect(screen.getByText('Affiliate CTA')).toBeInTheDocument()
    expect(screen.getByText('Body copy')).toBeInTheDocument()
    expect(screen.getAllByText('Stored CTA Payload').length).toBeGreaterThan(0)
    expect(screen.getAllByText('See the full migration brief').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Book briefing').length).toBeGreaterThan(0)
  })
})
