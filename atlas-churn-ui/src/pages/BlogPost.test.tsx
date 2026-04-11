import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import BlogPost from './BlogPost'

vi.mock('../hooks/useApiData', () => ({
  default: vi.fn(() => ({
    data: {
      post: {
        slug: 'zendesk-vs-intercom',
        title: 'Zendesk vs Intercom',
        description: 'Comparison',
        seo_title: null,
        seo_description: null,
        date: '2026-04-01',
        author: 'Atlas',
        tags: ['Helpdesk'],
        content: '<p>Test article</p>',
        faq: [],
        related_slugs: [],
      },
      relatedPosts: [],
    },
  })),
}))

vi.mock('../components/BlogArticleView', () => ({
  default: function BlogArticleView() {
    return <div>Article Body</div>
  },
}))

describe('BlogPost', () => {
  it('routes article readers into the primary product workflows', () => {
    render(
      <MemoryRouter initialEntries={['/blog/zendesk-vs-intercom']}>
        <Routes>
          <Route path="/blog/:slug" element={<BlogPost />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(screen.getByText('Article Body')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Start Vendor Retention' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_retention&redirect_to=%2Fwatchlists',
    )
    expect(screen.getByRole('link', { name: 'Start Challenger Lead Gen' })).toHaveAttribute(
      'href',
      '/signup?product=b2b_challenger&redirect_to=%2Fchallengers',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Watchlists' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fwatchlists&product=b2b_retention',
    )
    expect(screen.getByRole('link', { name: 'Sign in to Challengers' })).toHaveAttribute(
      'href',
      '/login?redirect_to=%2Fchallengers&product=b2b_challenger',
    )
  })
})
