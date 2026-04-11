import { describe, expect, it } from 'vitest'
import { resolveBlogArticleCta } from './blogCta'

describe('blogCta', () => {
  it('resolves affiliate CTAs with stored copy overrides', () => {
    const cta = resolveBlogArticleCta({
      data_context: {
        affiliate_url: ' https://example.com/shopify ',
        affiliate_partner: {
          product_name: 'Shopify',
        },
      },
      cta: {
        headline: ' Try Shopify now ',
        body: ' Evaluate the migration path. ',
        button_text: ' Start Free ',
        report_type: 'migration_guide',
      },
    } as any)

    expect(cta).toEqual({
      mode: 'affiliate',
      url: 'https://example.com/shopify',
      name: 'Shopify',
      headline: 'Try Shopify now',
      body: 'Evaluate the migration path.',
      buttonText: 'Start Free',
    })
  })

  it('falls back to the generic booking CTA when affiliate data is missing', () => {
    const cta = resolveBlogArticleCta({
      data_context: {
        booking_url: 'https://atlas.example/briefing',
      },
      cta: null,
    } as any)

    expect(cta.mode).toBe('generic')
    expect(cta.url).toBe('https://atlas.example/briefing')
    expect(cta.headline).toBe('Want churn intelligence on vendors in this space?')
    expect(cta.buttonText).toBe('Book a 15-Min Intel Briefing')
  })
})
