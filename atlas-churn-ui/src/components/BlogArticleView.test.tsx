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

  it('tags inline affiliate anchors with rel="sponsored" in production rendering', () => {
    // Render without preview / highlight flags -- this mirrors what a
    // real reader sees on /blog/<slug>. The rel="sponsored" tagging must
    // run regardless of the preview-decoration flag for FTC / Google
    // compliance.
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView post={post} showBackLink={false} />
      </MemoryRouter>,
    )

    const affiliateAnchor = container.querySelector(
      'a[href="https://example.com/shopify"]',
    )
    expect(affiliateAnchor).not.toBeNull()
    const rel = affiliateAnchor!.getAttribute('rel') || ''
    expect(rel.split(/\s+/)).toContain('sponsored')
    // No preview decoration without the highlight flag.
    expect(affiliateAnchor!.getAttribute('data-preview-affiliate')).toBeNull()
  })

  it('preserves existing rel tokens when adding sponsored to affiliate anchors', () => {
    // Prose authors may already set rel="noopener noreferrer" on
    // outbound links. The sponsored decoration must merge tokens, not
    // overwrite them.
    const postWithRel: BlogPost = {
      ...post,
      content:
        '<p><a href="https://example.com/shopify" rel="noopener noreferrer">Shopify trial</a></p>',
    }
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView post={postWithRel} showBackLink={false} />
      </MemoryRouter>,
    )

    const affiliateAnchor = container.querySelector(
      'a[href="https://example.com/shopify"]',
    )
    const relTokens = (affiliateAnchor!.getAttribute('rel') || '').split(/\s+/)
    expect(relTokens).toContain('sponsored')
    expect(relTokens).toContain('noopener')
    expect(relTokens).toContain('noreferrer')
  })

  it('does NOT tag non-affiliate anchors with sponsored', () => {
    // Only the anchor whose href matches data_context.affiliate_url
    // should be tagged. Internal blog links, methodology links, and
    // unrelated citation links must stay untouched.
    const postWithMixedLinks: BlogPost = {
      ...post,
      content:
        '<p><a href="https://example.com/shopify">Shopify</a> vs <a href="https://example.com/bigcommerce">BigCommerce</a></p>',
    }
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView post={postWithMixedLinks} showBackLink={false} />
      </MemoryRouter>,
    )

    const bigcommerceAnchor = container.querySelector(
      'a[href="https://example.com/bigcommerce"]',
    )
    expect(bigcommerceAnchor).not.toBeNull()
    const rel = bigcommerceAnchor!.getAttribute('rel') || ''
    expect(rel).not.toContain('sponsored')
  })

  it('tags the affiliate-mode CTA anchor with rel="sponsored"', () => {
    // Scope to `container` rather than `screen` so the assertion isn't
    // affected by leftover DOM from earlier tests in the file.
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView post={post} showBackLink={false} />
      </MemoryRouter>,
    )

    const ctaAnchor = Array.from(container.querySelectorAll('a')).find(
      (a) => a.textContent?.trim() === 'Book briefing',
    )
    expect(ctaAnchor).toBeDefined()
    const relTokens = (ctaAnchor!.getAttribute('rel') || '').split(/\s+/)
    expect(relTokens).toContain('sponsored')
    // Security tokens must remain.
    expect(relTokens).toContain('noopener')
    expect(relTokens).toContain('noreferrer')
  })

  it('does NOT tag the generic-mode CTA anchor with sponsored', () => {
    // Posts without an affiliate_url fall through to the generic CTA
    // path (cal.com booking, internal report). That destination is
    // not a commercial relationship and must NOT carry rel="sponsored".
    const genericPost: BlogPost = {
      ...post,
      data_context: {}, // strips affiliate_url + affiliate_partner
    }
    const { container } = render(
      <MemoryRouter>
        <BlogArticleView post={genericPost} showBackLink={false} />
      </MemoryRouter>,
    )

    // resolveBlogArticleCta() returns mode='generic' when affiliate_url
    // is missing; the stored CTA buttonText is preserved either way.
    const ctaAnchor = Array.from(container.querySelectorAll('a')).find(
      (a) => a.textContent?.trim() === 'Book briefing',
    )
    expect(ctaAnchor).toBeDefined()
    const rel = ctaAnchor!.getAttribute('rel') || ''
    expect(rel).not.toContain('sponsored')
  })
})
