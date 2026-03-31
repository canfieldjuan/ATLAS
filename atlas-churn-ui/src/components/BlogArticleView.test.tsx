import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import BlogArticleView from './BlogArticleView'
import type { BlogPost } from '../content/blog'

const post: BlogPost = {
  slug: 'switch-to-shopify-2026-03',
  title: 'Migration Guide: Why Teams Are Switching to Shopify',
  description: 'Shopify migration guide',
  date: '2026-03-30',
  author: 'Churn Signals Team',
  tags: ['shopify', 'migration'],
  content:
    '<p><a href="https://example.com/shopify">Shopify trial</a></p><p>{{chart:sources-bar}}</p>',
  charts: [],
  topic_type: 'migration_guide',
  data_context: {
    affiliate_url: 'https://example.com/shopify',
    affiliate_partner: { name: 'Shopify' },
  },
  cta: {
    headline: 'See the full migration brief',
    body: 'Get the full report before renewal.',
    button_text: 'Book briefing',
    report_type: 'migration_guide',
  },
}

describe('BlogArticleView', () => {
  it('renders stored CTA copy and preview overlays', () => {
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView
          post={post}
          preview
          showBackLink={false}
          highlightAffiliateLinks
          highlightCtas
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('See the full migration brief')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Book briefing' })).toHaveAttribute(
      'href',
      'https://example.com/shopify',
    )
    expect(screen.getByText('Rendered CTA')).toBeInTheDocument()
    expect(screen.getByText('Affiliate Disclosure')).toBeInTheDocument()
    expect(container.querySelector('a[data-preview-affiliate="true"]')).not.toBeNull()
  })
})
