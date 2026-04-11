import { cleanup, render } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import SeoHead from './SeoHead'

describe('SeoHead', () => {
  beforeEach(() => {
    cleanup()
    document.head.innerHTML = ''
    document.title = ''
  })

  it('writes the expected title, meta tags, canonical link, and schema blocks', () => {
    render(
      <SeoHead
        title="Atlas Report"
        description="Decision-ready churn intelligence"
        canonical="https://atlas.example/reports/atlas-report"
        keywords={['atlas', 'churn']}
        jsonLd={{ '@type': 'Article', headline: 'Atlas Report' }}
        faq={[
          { question: 'What is Atlas?', answer: 'A churn intelligence platform.' },
        ]}
      />,
    )

    expect(document.title).toBe('Atlas Report')
    expect(document.head.querySelector('meta[name="description"]')).toHaveAttribute(
      'content',
      'Decision-ready churn intelligence',
    )
    expect(document.head.querySelector('meta[property="og:title"]')).toHaveAttribute(
      'content',
      'Atlas Report',
    )
    expect(document.head.querySelector('meta[name="keywords"]')).toHaveAttribute(
      'content',
      'atlas, churn',
    )
    expect(document.head.querySelector('link[rel="canonical"]')).toHaveAttribute(
      'href',
      'https://atlas.example/reports/atlas-report',
    )
    expect(document.getElementById('seo-jsonld')?.textContent).toContain('"headline":"Atlas Report"')
    expect(document.getElementById('seo-faq-jsonld')?.textContent).toContain('"FAQPage"')
    expect(document.getElementById('seo-faq-jsonld')?.textContent).toContain('What is Atlas?')
  })

  it('cleans up the managed head tags on unmount', () => {
    const { unmount } = render(
      <SeoHead
        title="Atlas Blog"
        description="Blog description"
        canonical="https://atlas.example/blog/atlas"
      />,
    )

    expect(document.head.querySelector('meta[property="og:title"]')).not.toBeNull()
    expect(document.head.querySelector('link[rel="canonical"]')).not.toBeNull()

    unmount()

    expect(document.head.querySelector('meta[property="og:title"]')).toBeNull()
    expect(document.head.querySelector('link[rel="canonical"]')).toBeNull()
    expect(document.getElementById('seo-jsonld')).toBeNull()
    expect(document.getElementById('seo-faq-jsonld')).toBeNull()
  })
})
